"""Inference test script for the VLM expert model (`src.models.puffin.vlm`).

Loads a Qwen2p5RadioVLM checkpoint and runs `model.understand(...)` over every
image in a directory. Saves the per-image responses as JSON. Mirrors the
structure of scripts/evaluation/understanding_src.py.

Example:
    python scripts/evaluation/vlm.py \\
        configs/pipelines/vlm_qwen2_5_7b_radiov3H.py \\
        --checkpoint work_dirs/vlm_qwen2_5_7b_radiov3H/iter_60000.pth \\
        --image_dir data/test_images \\
        --prompt "Describe the image in detail." \\
        --output output/vlm_results.json
"""
import argparse
import copy
import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import gather_object
from einops import rearrange
from mmengine.config import Config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER


class VLMImageDataset(Dataset):
    """Loads every image under `image_dir`. Preprocessing is IDENTICAL to
    scripts/evaluation/understanding.py: optional central crop to `ratio`,
    longest edge -> `image_size`, normalize to [-1, 1], pad to square."""

    def __init__(self, image_dir, image_size=640, ratio=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.ratio = ratio
        self.data = sorted(
            p for p in glob(f"{image_dir}/*")
            if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )

    def __len__(self):
        return len(self.data)

    def _pad_an_image_tensor(self, image, pad_value=0):
        """Pads the image tensor to make it square."""
        h, w = image.shape[-2:]

        if h == w:
            return image
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            p2d = (pad_left, pad_right, 0, 0)  # (left, right, top, bottom)
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            p2d = (0, 0, pad_top, pad_bottom)

        image = F.pad(image, p2d, "constant", pad_value)
        return image

    def _crop_to_ratio(self, image):
        """Central crop to self.ratio (e.g. '16_9'), keeping the shortest edge."""
        if not self.ratio:
            return image

        w, h = image.size
        try:
            tgt_w_ratio, tgt_h_ratio = map(int, self.ratio.split('_'))
            target_aspect = tgt_w_ratio / tgt_h_ratio
        except Exception:
            print(f"Warning: Invalid ratio format '{self.ratio}'. Skipping crop.")
            return image

        current_aspect = w / h
        if current_aspect > target_aspect:
            new_h = h
            new_w = int(h * target_aspect)
        else:
            new_w = w
            new_h = int(w / target_aspect)

        left = (w - new_w) // 2
        top = (h - new_h) // 2
        image = image.crop((left, top, left + new_w, top + new_h))
        return image

    def _process_image(self, image):
        # 1. If ratio is set, perform Central Crop first
        if self.ratio is not None:
            image = self._crop_to_ratio(image)

        # 2. Resize (keep longest edge as self.image_size)
        w, h = image.size
        if w >= h:
            new_w = self.image_size
            new_h = int(h * (self.image_size / w))
        else:
            new_h = self.image_size
            new_w = int(w * (self.image_size / h))
        image = image.resize(size=(new_w, new_h))

        # 3. Convert to Tensor, normalize to [0, 1], then scale to [-1, 1]
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255.0
        pixel_values = 2.0 * pixel_values - 1.0
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        # 4. Pad the tensor to a centered square
        pixel_values = self._pad_an_image_tensor(pixel_values, pad_value=0)

        return pixel_values

    def __getitem__(self, idx):
        image_path = copy.deepcopy(self.data[idx])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self._process_image(image)
        return dict(id=os.path.basename(image_path), pixel_values=pixel_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='Path to a VLM pipeline / model config.')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to the trained ckpt (xtuner-format).')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--image_dir', required=True, type=str,
                        help='Directory containing test images.')
    parser.add_argument('--output', default='output/vlm_results.json', type=str)
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--ratio', type=str, default=None,
                        help="Optional central-crop aspect ratio, e.g. '16_9' "
                             "(same as understanding.py).")
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--prompt', type=str,
                        default="Describe the image in detail. Then reason its "
                                "spatial distribution and estimate its camera "
                                "parameters (roll, pitch, and field-of-view).",
                        help='Single instruction reused for every image.')

    args = parser.parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    config = Config.fromfile(args.config)
    print(f'Device: {accelerator.device}', flush=True)

    dataset = VLMImageDataset(image_dir=args.image_dir, image_size=args.image_size,
                              ratio=args.ratio)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,   # keep as List[Dict]
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
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        print(f'Number of samples: {len(dataset)}', flush=True)

    results = []
    for data_samples in tqdm(dataloader, disable=not accelerator.is_main_process):
        pixel_values = [d.pop('pixel_values') for d in data_samples]
        bsz = len(pixel_values)
        output_texts = model.understand(
            prompt=[args.prompt] * bsz,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            progress_bar=False,
        )
        for output_text, data_sample in zip(output_texts, data_samples):
            data_sample['output_text'] = output_text
            results.append(data_sample)

    results = gather_object(results)
    if accelerator.is_main_process:
        accelerator.print(f"Collected {len(results)} results across all GPUs")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        accelerator.print(f"Saved -> {args.output}")
