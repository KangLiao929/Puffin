import math
import torch
import argparse
from PIL import Image
from einops import rearrange
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from scripts.camera.cam_dataset import Cam_Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--prompt', type=str,
                        default="A streetlamp casts light on an outdoor mural with intricate floral designs and text, "
                        "set against a building wall; a yellow chair with a unique pattern sits in the foreground, "
                        "while a red structure and metal railing add to the scene\'s ambiance in a public park-like setting.")
    parser.add_argument('-r', '--roll', default="-0.3939", type=str, help="range=[-0.7854, 0.7854]")
    parser.add_argument('-p', '--pitch', default="0.0277", type=str, help="range=[-0.7854, 0.7854]")
    parser.add_argument('-f', '--fov', default="0.7595", type=str, help="range=[0.3491, 1.8326]")
    parser.add_argument('--output', default='image.jpg', type=str)
    parser.add_argument("--cfg_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).cuda().bfloat16().eval()
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    
    roll_rad, pitch_rad, vfov_rad = float(args.roll), float(args.pitch), float(args.fov)
    prompt_camera = (
        "The camera parameters (roll, pitch, and field-of-view) are: "
        f"{roll_rad:.4f}, {pitch_rad:.4f}, {vfov_rad:.4f}."
    )
    gen = Cam_Generator()
    cam_map = gen.get_cam(prompt_camera).to(device)
    cam_map = cam_map / (math.pi / 2)

    prompt_thinking = ("Given a scene description and corresponding camera parameters, "
                       "merge them into a coherent prompt and generate an accurate visualization "
                       "that highlights visual cues for spatial reasoning.")

    prompt = args.prompt + " " + prompt_camera
    bsz = args.grid_size ** 2
    images, output_reasoning = model.generate(
        prompt=[prompt]*bsz,
        cfg_prompt=[args.cfg_prompt]*bsz,
        pixel_values_init=None,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        cam_values=[[cam_map]]*bsz,
        progress_bar=False,
        reasoning=args.thinking,
        prompt_reasoning=[prompt_thinking]*bsz,
        generator=generator,
        height=args.height,
        width=args.width
    )

    images = rearrange(images, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    image = torch.clamp(127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    Image.fromarray(image).save(args.output)
    
    if args.thinking:
        print("Vanilla prompt: ", prompt)
        print("Reasoning prompt: ", output_reasoning[0])
