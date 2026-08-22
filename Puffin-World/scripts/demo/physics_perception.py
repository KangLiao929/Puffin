"""Single-image camera estimation demo (understanding branch + PF visualization).

Pick a model by NAME instead of passing a config -- the demo resolves it to the
matching pipeline config internally, builds only what inference needs, runs
`model.understand(...)` on ONE image, parses the estimated camera parameters
(roll, pitch, vfov, k1; radians), and renders the corresponding perspective
field (up + latitude) over the input image.

No accelerate / distributed setup: this is single-sample, single-GPU inference.

Example:
    python scripts/demo/camera_estimation.py demo.jpg \\
        --model Puffin-World-Base \\
        --checkpoint work_dirs/final_stage_2_base_qwen2_5_7b_radiov3H_sd3p5M/iter_120000.pth \\
        --output output/demo
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.camera.utils.text import parse_camera_params
from scripts.camera.utils.conversions import fov2focal
from scripts.camera.geometry.camera import SimpleRadial
from scripts.camera.geometry.gravity import Gravity
from scripts.camera.geometry.perspective_fields import get_perspective_field
from scripts.camera.visualization.visualize_batch import make_perspective_figures


# Model name -> pipeline config + default input size / instruction. All models
# share the same preprocessing (identical to scripts/evaluation/understanding.py
# and vlm.py): longest edge -> image_size, normalize to [-1, 1], pad to square.
MODEL_REGISTRY = {
    'Puffin-World-Base': dict(
        config='configs/pipelines/final_stage_2_base_qwen2_5_7b_radiov3H_sd3p5M.py',
        image_size=640,
        prompt=("Describe the image in detail. Then reason its spatial "
                "distribution and estimate its camera parameters (roll, "
                "pitch, field-of-view, and radial distortion)."),
    ),
    'Puffin-World-Pro': dict(
        config='configs/pipelines/final_stage_2_base_qwen2_5_1_5b_radiov4H_sd3p5L.py',
        image_size=640,
        prompt=("Describe the image in detail. Then reason its spatial "
                "distribution and estimate its camera parameters (roll, "
                "pitch, field-of-view, and radial distortion)."),
    ),
    'Puffin-World-Caption': dict(
        config='configs/pipelines/vlm_qwen3_5_0_8b_radiov3H_stage_2.py',
        image_size=640,
        prompt=("Describe the image in detail. Then reason its spatial "
                "distribution and estimate its camera parameters (roll, "
                "pitch, field-of-view, and radial distortion)."),
    ),
}


def _pad_square(image, pad_value=0):
    """Pad a [C, H, W] tensor to a centered square."""
    h, w = image.shape[-2:]
    if h == w:
        return image
    if h > w:
        pad_left = (h - w) // 2
        p2d = (pad_left, h - w - pad_left, 0, 0)
    else:
        pad_top = (w - h) // 2
        p2d = (0, 0, pad_top, w - h - pad_top)
    return torch.nn.functional.pad(image, p2d, "constant", pad_value)


def load_image(image_path, image_size):
    """Load ONE image -> (model_input [3,S,S] in [-1,1], viz_image [3,H,W] in [0,1]).

    Same preprocessing as scripts/evaluation/understanding.py / vlm.py:
    longest edge -> image_size, normalize, pad to a centered square.
    `viz_image` is the resized image BEFORE square padding, so the perspective
    field is rendered at the true aspect ratio of what the model looked at.
    """
    image = Image.open(image_path).convert('RGB')

    w, h = image.size
    if w >= h:
        new_w, new_h = image_size, int(h * image_size / w)
    else:
        new_h, new_w = image_size, int(w * image_size / h)
    image = image.resize((new_w, new_h))

    pixel_values = torch.from_numpy(np.array(image)).float() / 255.0
    pixel_values = rearrange(pixel_values, 'h w c -> c h w')
    viz_image = pixel_values.clone()                    # [0, 1], unpadded
    pixel_values = 2.0 * pixel_values - 1.0             # [-1, 1]
    pixel_values = _pad_square(pixel_values)
    return pixel_values, viz_image


def parse_params(text):
    """Parse (roll, pitch, vfov, k1) in radians; radial first, pinhole fallback."""
    try:
        return parse_camera_params(text, mode='radial')
    except ValueError:
        return parse_camera_params(text, mode='pinhole')


_PARAM_BLOCK = re.compile(
    r"[+-]?\d+(?:\.\d+)?(?:\s*,\s*[+-]?\d+(?:\.\d+)?){2,3}")


def split_caption(text):
    """Split the raw model output into the scene DESCRIPTION and the trailing
    camera-parameter sentence, and parse the numeric parameters.

    The caption format appends the parameters at the end ("... The camera
    parameters ... are: r, p, fov[, k1]."). We locate the LAST comma-separated
    number block (mirroring parse_camera_params) and cut at the start of the
    sentence containing it.

    Returns:
        (description, camera_text, (roll, pitch, vfov, k1))
    Raises:
        ValueError if no parameter block is found.
    """
    params = parse_params(text)
    m = list(_PARAM_BLOCK.finditer(text))[-1]  # parse succeeded -> non-empty
    sent_start = text.rfind('.', 0, m.start()) + 1  # 0 when no '.' before
    description = text[:sent_start].strip()
    camera_text = text[sent_start:].strip()
    return description, camera_text, params


def save_pf_visualization(viz_image, roll, pitch, vfov, k1, out_dir, stem):
    """Build the perspective field from the estimated parameters and save the
    up-field / latitude overlays (same construction as the model's PF path:
    SimpleRadial + Gravity.from_rp + get_perspective_field)."""
    _, H, W = viz_image.shape
    f = fov2focal(torch.tensor(vfov), H)
    params = torch.tensor([W, H, f, f, W / 2.0, H / 2.0, k1, 0.0]).float()
    camera = SimpleRadial(params).float().scale(torch.Tensor([1, 1]))
    gravity = Gravity.from_rp(torch.tensor(roll).float(), torch.tensor(pitch).float())

    up_field, lat_field = get_perspective_field(
        camera, gravity, use_up=True, use_latitude=True)

    single = {
        "image": viz_image.unsqueeze(0),        # [1, 3, H, W] in [0, 1]
        "up_field": up_field.float().cpu(),     # [1, 2, H, W], radians
        "latitude_field": lat_field.float().cpu(),  # [1, 1, H, W], radians
    }
    figs = make_perspective_figures(single, single, n_pairs=1)
    saved = []
    for k, fig in figs.items():
        suffix = "up" if "up" in k else "lat" if "lat" in k else k
        out_path = os.path.join(out_dir, f"{stem}_pf_{suffix}.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        saved.append(out_path)
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('image', help='Path to ONE input image.')
    parser.add_argument('--model', default='Puffin-World-Base',
                        choices=sorted(MODEL_REGISTRY.keys()),
                        help='Which pretrained pipeline to use (default: Puffin-World-Base).')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Trained weights (.pth file or DeepSpeed iter_* dir).')
    parser.add_argument('--output', default='output/demo_camera', type=str,
                        help='Directory for the caption/params JSON and PF figures.')
    parser.add_argument('--prompt', default=None, type=str,
                        help='Override the default instruction.')
    parser.add_argument('--image_size', default=None, type=int,
                        help='Override the model\'s default input size.')
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    entry = MODEL_REGISTRY[args.model]
    image_size = args.image_size or entry['image_size']
    prompt = args.prompt or entry['prompt']

    # ---- build the model from the registered pipeline config ----
    cfg = Config.fromfile(entry['config'])
    cfg.model.pretrained_pth = None                 # weights come from --checkpoint
    cfg.model.use_activation_checkpointing = False  # inference only
    model = BUILDER.build(cfg.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=args.device)
    model = model.to(model.dtype)
    model.eval()

    # ---- single-image inference ----
    pixel_values, viz_image = load_image(args.image, image_size)
    with torch.no_grad():
        output_text = model.understand(
            prompt=[prompt],
            pixel_values=[pixel_values],
            max_new_tokens=args.max_new_tokens,
        )[0]
    # ---- split description / camera params, then PF visualization ----
    os.makedirs(args.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    result = dict(model=args.model, image=args.image, output_text=output_text)
    try:
        description, camera_text, (roll, pitch, vfov, k1) = split_caption(output_text)
        result.update(description=description, camera_text=camera_text,
                      roll=roll, pitch=pitch, vfov=vfov, k1=k1)
        print(f"\n[{args.model}] scene description:\n{description}\n")
        print(f"camera sentence: {camera_text}")
        print(f"camera estimate (radians): roll={roll:.4f} pitch={pitch:.4f} "
              f"vfov={vfov:.4f} k1={k1:.4f}"
              f"  ->  degrees: roll={math.degrees(roll):.2f} "
              f"pitch={math.degrees(pitch):.2f} vfov={math.degrees(vfov):.2f}")
        saved = save_pf_visualization(
            viz_image, roll, pitch, vfov, k1, args.output, stem)
        print("perspective-field figures: " + ", ".join(saved))
    except ValueError as e:
        result.update(description=output_text.strip())
        print(f"\n[{args.model}] caption (no camera parameters found):\n"
              f"{output_text}\n")
        print(f"[warn] could not parse camera parameters ({e}); "
              f"PF visualization skipped.")

    json_path = os.path.join(args.output, f"{stem}_result.json")
    with open(json_path, 'w') as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print(f"Saved -> {json_path}")
