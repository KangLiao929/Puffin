import math
import json
import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import random
from einops import rearrange
from glob import glob
from torch.utils.data import Dataset, DataLoader
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object

from scripts.camera.cam_dataset import Cam_Generator

class TestDataset(Dataset):
    # The diffusion pipeline requires height/width divisible by 16.
    SIZE_MULTIPLE = 16

    def __init__(self, prompt_path, num, camera_model='radial'):
        all_paths = glob(f"{prompt_path}/*.json")
        # Drop prompts whose embedded image size is not divisible by SIZE_MULTIPLE
        # (e.g. 426 / 360). Prompts without size info are kept (random fallback).
        self.prompt_path = [p for p in all_paths if self._prompt_size_ok(p)]
        n_dropped = len(all_paths) - len(self.prompt_path)
        if n_dropped:
            print(f"[TestDataset] Filtered out {n_dropped}/{len(all_paths)} prompts "
                  f"with size not divisible by {self.SIZE_MULTIPLE}.", flush=True)
        # num == -1 (or any non-positive) -> use all prompts; otherwise random subset.
        if num is not None and num >= 0:
            self.prompt_path = random.sample(self.prompt_path, min(num, len(self.prompt_path)))
        self.cam_generator = Cam_Generator(mode=camera_model)
        # All 640-scale resolutions used when building the dataset.
        self.resolutions = [
            (640, 640),
            (640, 480), (480, 640),
            (640, 426), (426, 640),
            (640, 360), (360, 640),
        ]
        # Only valid sizes are eligible for the random fallback (no embedded size).
        self.valid_resolutions = [
            (h, w) for (h, w) in self.resolutions if self._size_divisible(h, w)
        ]

    @classmethod
    def _size_divisible(cls, h, w):
        return int(h) % cls.SIZE_MULTIPLE == 0 and int(w) % cls.SIZE_MULTIPLE == 0

    def _prompt_size_ok(self, path):
        """Keep prompts with no size info, or whose size is divisible by SIZE_MULTIPLE."""
        try:
            with open(path, 'r') as f:
                d = json.load(f)
        except Exception:
            return True  # unreadable here -> defer to __getitem__
        if 'height' in d and 'width' in d:
            return self._size_divisible(d['height'], d['width'])
        return True

    def __len__(self):
        return len(self.prompt_path)
    
    def _read_camera(self, caption, h, w):
        return self.cam_generator.get_cam(caption, h, w)
    
    def _process_camera(self, camera):
        cam_values = camera / (math.pi / 2)
        return cam_values

    def __getitem__(self, idx):

        with open(self.prompt_path[idx], 'r') as f:
            data_sample = json.load(f)

        sample_id = os.path.basename(self.prompt_path[idx]).replace('.json', '')
        data_sample.update(sample_id=sample_id)
        # If the prompt carries an image size (written at caption time), use it;
        # otherwise fall back to a random resolution.
        if 'height' in data_sample and 'width' in data_sample:
            h, w = int(data_sample['height']), int(data_sample['width'])
        else:
            h, w = random.choice(self.valid_resolutions)
        caption = data_sample['caption'].strip()
        camera = self._read_camera(caption, h, w)
        data_sample['camera'] = self._process_camera(camera)
        data_sample['height'] = h
        data_sample['width'] = w

        return data_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num', default=100, type=int,
                        help='Number of prompts to sample; -1 = use all prompts.')
    parser.add_argument('--prompt_path', default='/mnt/afs_100t/NTU_slab/kliao/data/Puffin-Pro/Projection/Vecteezy/train_scene_cam/', type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument("--cfg_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    accelerator = Accelerator()
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)
    config = Config.fromfile(args.config)
    print(f'Device: {accelerator.device}', flush=True)
    
    dataset = TestDataset(prompt_path=args.prompt_path, num=args.num)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=lambda x: x
                            )

    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()

    dataloader = accelerator.prepare(dataloader)

    print(f'Number of samples: {len(dataloader)}', flush=True)

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    results = []
    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        device_idx = accelerator.process_index

        prompt = [data_sample['caption'].strip() for data_sample in data_samples]
        cam_values = [[cam] for cam in [data_sample.pop('camera') for data_sample in data_samples]]
        heights = [data_sample['height'] for data_sample in data_samples]
        widths = [data_sample['width'] for data_sample in data_samples]
        cfg_prompt = [args.cfg_prompt] * len(prompt)

        images = model.generate(prompt=prompt, cfg_prompt=cfg_prompt,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps, cam_values=cam_values,
                                progress_bar=False,
                                generator=generator, height=heights[0], width=widths[0])
        images = rearrange(images, 'b c h w -> b h w c')

        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        for image, data_sample in zip(images, data_samples):
            Image.fromarray(image).save(f"{args.output}/{data_sample['sample_id']}.png")