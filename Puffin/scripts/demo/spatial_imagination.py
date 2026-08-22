import os
import argparse
import torch
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from PIL import Image
import numpy as np
from einops import rearrange

def process_image(image_path, image_size=512):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size=(image_size, image_size))
    pixel_values = torch.from_numpy(np.array(image)).float()
    pixel_values = pixel_values / 255.0
    pixel_values = 2 * pixel_values - 1
    pixel_values = rearrange(pixel_values, 'h w c -> c h w')
    return pixel_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--image', required=True, type=str, help='path to a single image')
    parser.add_argument('--prompt', type=str,
                        default=("Given the initial view and the camera parameters of the target view with "
                                 "the deviation yaw angle, describe the target image in detail."))
    parser.add_argument('--location', type=str, choices=['left', 'behind', 'right'], required=True,
                        help="relative target view position: left | behind | right")
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    yaw_map = {
        'left':   "1.5710",  # +90°
        'behind': "3.1416",  # +180°
        'right':  "4.7120",  # +270°
    }
    yaw_str = yaw_map[args.location]

    camera_str = (
        "The camera parameters (roll, pitch, and field-of-view) are: "
        "0.0000, 0.0000, 1.4800. "
        f"The deviation of the camera yaw angle is {yaw_str}."
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).to(device).bfloat16().eval()
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)

    pixel_values = process_image(args.image, image_size=args.image_size).to(device)
    prompt = (args.prompt + " " + camera_str).strip()

    with torch.no_grad():
        outputs = model.understand(prompt=[prompt], pixel_values=[pixel_values], progress_bar=False)

    print("Prompt:", prompt)
    print("Spatial imagination:", outputs[0])