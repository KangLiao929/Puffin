import os
import math
import argparse
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from scripts.camera.cam_dataset import Cam_Generator

def load_init_image(path, width, height):
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = 2.0 * arr - 1.0
    tensor = torch.from_numpy(arr)
    tensor = rearrange(tensor, "h w c -> c h w").contiguous()
    return tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", help="config file path")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--cfg_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--grid_size", type=int, default=2, help="final grid is grid_size x grid_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_image", type=str, default=None, help="path to the initial view")
    parser.add_argument("--output", default="output.jpg", type=str, help="path to save the generated target view")
    parser.add_argument("-r", "--roll", type=float, default=0.0)
    parser.add_argument("-p", "--pitch", type=float, default=0.0)
    parser.add_argument("-y", "--yaw", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).to(device).bfloat16().eval()
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    vfov_rad = 1.48
    roll_rad = args.roll
    pitch_rad = args.pitch
    yaw_rad = args.yaw
    if yaw_rad < 0:
        yaw_rad = yaw_rad + 2 * math.pi

    prompt = (
        "The camera parameters (roll, pitch, and field-of-view) are: "
        f"{roll_rad:.4f}, {pitch_rad:.4f}, {vfov_rad:.4f}. "
        f"The deviation of the camera yaw angle is {yaw_rad:.4f}."
    )
    prompt_camera = (
        "The camera parameters (roll, pitch, and field-of-view) are: "
        f"{roll_rad:.4f}, {pitch_rad:.4f}, {vfov_rad:.4f}."
    )

    gen = Cam_Generator()
    cam_map = gen.get_cam(prompt_camera).to(device)
    cam_map = cam_map / (math.pi / 2)

    init_tensor = load_init_image(args.init_image, args.width, args.height).to(device)
    bsz = args.grid_size ** 2
    prompts = [prompt] * bsz
    cfg_prompts = [args.cfg_prompt] * bsz
    cam_values = [[cam_map]] * bsz
    pixel_values_init = [[init_tensor]] * bsz

    result = model.generate(
        prompt=prompts,
        cfg_prompt=cfg_prompts,
        pixel_values_init=pixel_values_init,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        cam_values=cam_values,
        progress_bar=False,
        reasoning=False,
        generator=generator,
        height=args.height,
        width=args.width,
    )

    images = result[0] if isinstance(result, (tuple, list)) else result
    tiled = rearrange(images, "(m n) c h w -> (m h) (n w) c", m=args.grid_size, n=args.grid_size)
    img_uint8 = torch.clamp(127.5 * tiled + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(img_uint8).save(args.output)
