import os
import re
import torch
import argparse
import numpy as np
from PIL import Image
from einops import rearrange
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.camera.cam_dataset import Cam_Generator
from scripts.camera.visualization.visualize_batch import make_perspective_figures

NUM = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
CAM_PATTERN = re.compile(r"(?:camera parameters.*?:|roll.*?:)\s*("+NUM+r")\s*,\s*("+NUM+r")\s*,\s*("+NUM+r")", re.IGNORECASE|re.DOTALL)

def center_crop(image):
    w, h = image.size
    s = min(w, h)
    l = (w - s) // 2
    t = (h - s) // 2
    return image.crop((l, t, l + s, t + s))

def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert('RGB')
    image = center_crop(image)
    image = image.resize((image_size, image_size))
    x = torch.from_numpy(np.array(image)).float()
    x = x / 255.0
    x = 2 * x - 1
    x = rearrange(x, 'h w c -> c h w')
    return x

def load_square_rgb(image_path, image_size):
    image = Image.open(image_path).convert('RGB')
    image = center_crop(image)
    image = image.resize((image_size, image_size))
    arr = np.array(image)[:, :, ::-1]
    return arr.copy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--image_path', required=True, type=str, help='path to a single image')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prompt = ("Describe the image in detail. Then reason its spatial distribution and estimate its camera parameters (roll, pitch, and field-of-view).")
    if args.thinking:
        prompt = ("Reason the spatial distribution of this image in a thinking mode, and then estimate its camera parameters (roll, pitch, and field-of-view).")

    pixel_values = preprocess_image(args.image_path, args.image_size).to(device)

    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).to(device).bfloat16().eval()
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        outputs = model.understand(prompt=[prompt], pixel_values=[pixel_values], progress_bar=False)

    text = outputs[0]
    print(text)
    
    # covert the estimated camera parameters into the pixel-wise camera map
    save_dir = args.save_dir or os.path.dirname(args.image_path)
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image_path))[0]

    gen = Cam_Generator(mode="cot") if args.thinking else Cam_Generator(mode="base")
    cam = gen.get_cam(text)
    bgr = load_square_rgb(args.image_path, args.image_size).astype(np.float32) / 255.0
    rgb = bgr[:, :, ::-1].copy()
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    single_batch = {}
    single_batch["image"] = image_tensor
    single_batch["up_field"] = cam[:2].unsqueeze(0)
    single_batch["latitude_field"] = cam[2:].unsqueeze(0)

    figs = make_perspective_figures(single_batch, single_batch, n_pairs=1)
    for k, fig in figs.items():
        if "up_field" in k:
            suffix = "_up"
        elif "latitude_field" in k:
            suffix = "_lat"
        else:
            suffix = f"_{k}"
        out_path = os.path.join(save_dir, f"{stem}_camera_map_vis{suffix}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"The visualization results have been saved at: {args.save_dir}")
