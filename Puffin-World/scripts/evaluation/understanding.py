from PIL import Image
import json
from einops import rearrange
import numpy as np
from glob import glob
import os
import copy
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch.nn.functional as F

class TestDataset(Dataset):
    def __init__(self, image_dir, image_size=640, ratio=None):
        """
        Args:
            image_dir: Path to the image directory.
            image_size: Target size for the longest edge of the image.
            ratio: Target aspect ratio (str), e.g., '1_1', '16_9', '3_4', '4_3', etc.
                   Defaults to None (no aspect ratio change).
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.data = sorted(glob(f"{image_dir}/*"))
        self.ratio = ratio

    def __len__(self):
        return len(self.data)

    def _pad_an_image_tensor(self, image, pad_value=0):
        """
        Pads the image tensor to make it square.
        """
        h, w = image.shape[-2:]
        
        if h == w:
            return image
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            p2d = (pad_left, pad_right, 0, 0) # (left, right, top, bottom)
        else:
            pad_top = (w - h) // 2  
            pad_bottom = w - h - pad_top
            p2d = (0, 0, pad_top, pad_bottom)

        image = F.pad(image, p2d, "constant", pad_value)
        return image

    def _crop_to_ratio(self, image):
        """
        Performs a Central Crop on the image based on self.ratio.
        Keeps the shortest edge fixed to maximize the crop area.
        """
        if not self.ratio:
            return image
        
        w, h = image.size
        
        # Parse the ratio string, e.g., "16_9" -> 16, 9
        try:
            tgt_w_ratio, tgt_h_ratio = map(int, self.ratio.split('_'))
            target_aspect = tgt_w_ratio / tgt_h_ratio
        except:
            print(f"Warning: Invalid ratio format '{self.ratio}'. Skipping crop.")
            return image

        current_aspect = w / h

        if current_aspect > target_aspect:
            # Image is wider than target ratio -> Fix height, crop width
            # new_h remains the same, calculate new_w
            new_h = h
            new_w = int(h * target_aspect)
        else:
            # Image is taller/narrower than target ratio -> Fix width, crop height
            # new_w remains the same, calculate new_h
            new_w = w
            new_h = int(w / target_aspect)

        # Calculate central crop coordinates
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h

        image = image.crop((left, top, right, bottom))
        return image

    def _process_image(self, image):
        # 1. If ratio is set, perform Central Crop first
        if self.ratio is not None:
            image = self._crop_to_ratio(image)

        # 2. Original logic: Resize (keep longest edge as self.image_size)
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
        
        # 4. Padding (Original logic: pads the tensor to be a square)
        pixel_values = self._pad_an_image_tensor(pixel_values, pad_value=0)
        
        return pixel_values

    def __getitem__(self, idx):
        image_path = copy.deepcopy(self.data[idx])
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self._process_image(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            pixel_values = torch.zeros((3, self.image_size, self.image_size))

        return dict(id=os.path.basename(image_path), pixel_values=pixel_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--image_dir', default='data/puffin_test_dataset/stanford2d3d/images', type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--ratio", type=str, default=None)

    args = parser.parse_args()

    accelerator = Accelerator()
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)
    config = Config.fromfile(args.config)
    print(f'Device: {accelerator.device}', flush=True)
    
    prompt = (
        "Describe the image in detail. Then reason its spatial distribution "
            "and estimate its camera parameters (roll, pitch, field-of-view, and radial distortion)."
    )
    
    dataset = TestDataset(image_size=args.image_size, image_dir=args.image_dir, ratio=args.ratio)
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

    results = []

    for data_samples in tqdm(dataloader, disable=not accelerator.is_main_process):

        pixel_values = [data_sample.pop('pixel_values') for data_sample in data_samples]
        bsz = len(pixel_values)

        output_texts = model.understand(prompt=[prompt]*bsz, pixel_values=pixel_values, progress_bar=False)

        for output_text, data_sample in zip(output_texts, data_samples):
            data_sample['output_text'] = output_text
            results.append(data_sample)

    results = gather_object(results)

    if accelerator.is_main_process:
        accelerator.print(f"Collected {len(results)} result samples from all gpus")

        with open(args.output, 'w') as f:
            json.dump(results, f)
