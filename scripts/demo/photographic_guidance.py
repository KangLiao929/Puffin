import os
import argparse
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

def load_and_preprocess(image_path, image_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = 2.0 * arr - 1.0
    tensor = torch.from_numpy(arr)
    tensor = rearrange(tensor, "h w c -> c h w").contiguous()
    return tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", help="config file path.")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--image", required=True, type=str, help="path to one input image")
    parser.add_argument("--prompt", type=str, default=(
        "Estimate the camera parameters (roll, pitch, and field-of-view) of this image. "
        "And then predict the deviation camera yaw angle and pitch angle of the target view "
        "with high photographic aesthetics."
    ))
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).to(device).bfloat16().eval()
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)

    pixel_values = load_and_preprocess(args.image, args.image_size).to(device)

    with torch.no_grad():
        outputs = model.understand(prompt=[args.prompt], pixel_values=[pixel_values], progress_bar=False)

    print("Prompt:", args.prompt)
    print("Photographic guidance:", outputs[0])
