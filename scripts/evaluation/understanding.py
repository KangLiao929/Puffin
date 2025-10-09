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

def center_crop(image):
    width, height = image.size
    new_width, new_height = min(width, height), min(width, height)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))

class TestDataset(Dataset):
    def __init__(self, image_dir, image_size=512):
        self.image_dir = image_dir
        self.image_size = image_size
        self.data = sorted(glob(f"{image_dir}/*"))

    def __len__(self):
        return len(self.data)

    def _process_image(self, image):
        image = center_crop(image)
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def __getitem__(self, idx):
        image_path = copy.deepcopy(self.data[idx])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self._process_image(image)
        return dict(id=os.path.basename(image_path), pixel_values=pixel_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument("--thinking", action="store_true", help="enable thinking mode")
    parser.add_argument('--image_dir', default='Puffin-Und/images', type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument("--image_size", type=int, default=512)

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
            "and estimate its camera parameters (roll, pitch, and field-of-view)."
    )
    if args.thinking:
        prompt = (
        "Reason the spatial distribution of this image in a thinking mode, "
            "and then estimate its camera parameters (roll, pitch, and field-of-view)."
        )
    
    dataset = TestDataset(image_size=args.image_size, image_dir=args.image_dir)
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
