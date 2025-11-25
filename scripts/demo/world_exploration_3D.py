import os
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from einops import rearrange
from glob import glob
import numpy as np
import math

from scripts.camera.cam_dataset import Cam_Generator

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

class TestDataset(Dataset):
    def __init__(self, init_view_path=None, height=512, width=512):
        self.height = height
        self.width = width
        self.files = []
        if init_view_path is not None:
            for ext in IMG_EXTS:
                self.files += glob(os.path.join(init_view_path, f"*{ext}"))
            self.files = sorted(self.files)

    def __len__(self):
        return max(1, len(self.files))

    def _process_image(self, image: Image.Image):
        image = image.resize((self.width, self.height), Image.BICUBIC)
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255.0
        pixel_values = 2.0 * pixel_values - 1.0
        pixel_values = rearrange(pixel_values, 'h w c -> c h w').contiguous()
        return pixel_values

    def __getitem__(self, idx):
        if len(self.files) == 0:
            return dict(sample_id=f"sample_{idx:06d}", pixel_values=None)

        fp = self.files[idx]
        sample_id = os.path.splitext(os.path.basename(fp))[0]
        image = Image.open(fp).convert('RGB')
        pixel_values = self._process_image(image)
        return dict(sample_id=sample_id, pixel_values=pixel_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--init_view_path', default="imgs/", type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument("--cfg_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_cams", type=int, default=3)
    parser.add_argument("--range_rad", type=float, default=0.3927, help="roll/pitch/yaw range")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    accelerator = Accelerator()

    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)
    config = Config.fromfile(args.config)
    print(f'Device: {accelerator.device}', flush=True)

    dataset = TestDataset(
        init_view_path=args.init_view_path,
        height=args.height,
        width=args.width
    )
    dataloader = DataLoader(
        dataset=dataset,
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

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)
    accelerator.wait_for_everyone()

    print(f'Number of samples: {len(dataloader)}', flush=True)

    vfov_rad = 1.48
    rng = torch.Generator().manual_seed(args.seed + accelerator.process_index)
    lo, hi = -args.range_rad, args.range_rad
    gen = Cam_Generator()

    sampler_generator = torch.Generator(device=model.device).manual_seed(args.seed)

    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        bsz = len(data_samples)

        sampled = (torch.rand(bsz, args.num_cams, 3, generator=rng) * (hi - lo) + lo).tolist()

        for cam_idx in range(args.num_cams):
            prompts = []
            cfg_prompts = []
            cam_values = []
            pixel_values_init = []
            save_tags = []

            for i, ds in enumerate(data_samples):
                roll_rad = sampled[i][cam_idx][0]
                pitch_rad = sampled[i][cam_idx][1]
                yaw_rad = sampled[i][cam_idx][2] * 2
                if yaw_rad < 0:
                    yaw_rad = yaw_rad + 2 * math.pi

                prompt_template = (
                    "The camera parameters (roll, pitch, and field-of-view) are: "
                    f"{roll_rad:.4f}, {pitch_rad:.4f}, {vfov_rad:.4f}. "
                    f"The deviation of the camera yaw angle is {yaw_rad:.4f}."
                )
                prompt_template_cam = (
                    "The camera parameters (roll, pitch, and field-of-view) are: "
                    f"{roll_rad:.4f}, {pitch_rad:.4f}, {vfov_rad:.4f}."
                )

                cam_map = gen.get_cam(prompt_template_cam)
                cam_map = cam_map / (math.pi / 2)

                prompts.append(prompt_template)
                cfg_prompts.append(args.cfg_prompt)
                cam_values.append([cam_map])
                pixel_values_init.append([ds['pixel_values']])

                tag = f"roll{roll_rad:.2f}_pitch{pitch_rad:.2f}_yaw{yaw_rad:.2f}"
                save_tags.append(tag)

            out = model.generate(
                prompt=prompts,
                cfg_prompt=cfg_prompts,
                pixel_values_init=pixel_values_init,
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
                cam_values=cam_values,
                progress_bar=False,
                reasoning=False,
                generator=sampler_generator,
                height=args.height,
                width=args.width
            )
            images = out[0] if isinstance(out, (tuple, list)) else out

            images = rearrange(images, 'b c h w -> b h w c')
            images = torch.clamp(127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

            for img, ds, tag in zip(images, data_samples, save_tags):
                base = ds['sample_id'] if 'sample_id' in ds else f"sample_{batch_idx:06d}"
                save_path = os.path.join(args.output, f"{base}_{tag}.jpg")
                Image.fromarray(img).save(save_path)
