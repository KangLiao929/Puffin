import math
import json
import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from glob import glob
from torch.utils.data import Dataset, DataLoader
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object

class TestDataset(Dataset):
    def __init__(self, prompt_path, camera_path=None, image_size=512):
        self.prompt_path = prompt_path
        self.camera_path = camera_path
        self.image_size = image_size
        self.prompt_path = sorted(glob(f"{self.prompt_path}/*.json"))
        if self.camera_path is not None:
            self.camera_path = sorted(glob(f"{self.camera_path}/*.pt"))

    def __len__(self):
        return len(self.prompt_path)
    
    def _process_camera(self, camera):
        cam_values = camera / (math.pi / 2)
        return cam_values

    def __getitem__(self, idx):

        with open(self.prompt_path[idx], 'r') as f:
            data_sample = json.load(f)

        sample_id = os.path.basename(self.prompt_path[idx]).replace('.json', '')
        data_sample.update(sample_id=sample_id)
        
        if self.camera_path is not None:
            camera = torch.load(self.camera_path[idx])
            data_sample['camera'] = self._process_camera(camera)
        else:
            data_sample['camera'] = None

        return data_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--prompt_path', default='Puffin-Gen/caption/caption_src', type=str)
    parser.add_argument('--camera_path', default='Puffin-Gen/camera', type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument("--cfg_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="enable thinking mode")
    parser.add_argument('--output_thinking', default='output_thinking', type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    accelerator = Accelerator()
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)
    config = Config.fromfile(args.config)
    print(f'Device: {accelerator.device}', flush=True)
    
    prompt_thinking = (
            "Given a scene description and corresponding camera parameters, "
            "merge them into a coherent prompt and generate an accurate visualization "
            "that highlights visual cues for spatial reasoning."
    )

    dataset = TestDataset(prompt_path=args.prompt_path,
                          camera_path=args.camera_path
                          )
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
        cfg_prompt = [args.cfg_prompt] * len(prompt)

        images, output_reasoning = model.generate(prompt=prompt, cfg_prompt=cfg_prompt, pixel_values_init=None,
                                        cfg_scale=args.cfg_scale, num_steps=args.num_steps, cam_values=cam_values, 
                                        progress_bar=False, reasoning=args.thinking, prompt_reasoning=[prompt_thinking]*len(prompt),
                                        generator=generator, height=args.height, width=args.width)
        images = rearrange(images, 'b c h w -> b h w c')

        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        for image, data_sample in zip(images, data_samples):
            Image.fromarray(image).save(f"{args.output}/{data_sample['sample_id']}.png")
        for output_text, data_sample in zip(output_reasoning, data_samples):
            data_sample['output_text'] = output_text
            results.append(data_sample)
                
    if args.thinking:
        results = gather_object(results)
        if accelerator.is_main_process:
            accelerator.print(f"Collected {len(results)} result samples from all gpus")

            with open(args.output_thinking, 'w') as f:
                json.dump(results, f)