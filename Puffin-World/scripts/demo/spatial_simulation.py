"""Camera-controlled spatial simulation demo (cam2image branch, N samples).

Pick a model by NAME instead of passing a config -- the demo resolves it to the
matching pipeline config internally, builds the camera map (6-ch ray map +
3-ch perspective field) from the camera parameters, and generates N images
(default 4) with the SAME caption and camera. Each sample draws its resolution
randomly from the 640-scale combinations used at dataset-construction time
(scripts/evaluation/generation.py), filtered to the diffusion pipeline's
divisible-by-16 constraint. The N results are tiled side by side in ONE row.

With --vis_pf, two extra figures are saved that visualize, one-to-one with the
generated row, the conditioning up field and latitude field overlaid on each
result (same renderers as the GeoCalib-style eval visualizations).

The camera parameters (roll, pitch, vfov in radians; k1 optional, default 0)
can either be embedded in the caption itself (trailing "... are: r, p, v, k."
sentence, as in the training captions) or given as ONE --camera argument, in
which case the sentence is appended to the caption automatically. Both the
prompt and --camera have defaults, so the demo runs with just a checkpoint.

No accelerate / distributed setup: single-GPU inference.

Example:
    python scripts/demo/spatial_simulation.py \\
        "A cozy living room with wooden furniture and warm sunlight." \\
        --model Puffin-World-Base \\
        --checkpoint work_dirs/final_stage_2_base_qwen2_5_7b_radiov3H_sd3p5M/iter_120000.pth \\
        --camera "0.05, -0.12, 1.2" \\
        --num 4 --vis_pf \\
        --output output/demo_gen/spatial_simulation.png
"""
import argparse
import io
import math
import os
import random
import sys

sys.path.insert(0, os.getcwd())

import torch
from PIL import Image
from einops import rearrange
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.camera.cam_dataset import Cam_Generator
from scripts.camera.utils.text import parse_camera_params
from scripts.camera.visualization.viz2d import (
    plot_images,
    plot_latitudes,
    plot_vector_fields,
)


# Model name -> pipeline config. Only the world models have the generation
# branch (Puffin-World-Caption is an understanding-only VLM), so it is not
# listed here.
MODEL_REGISTRY = {
    'Puffin-World-Base': dict(
        config='configs/pipelines/final_stage_2_base_qwen2_5_7b_radiov3H_sd3p5M.py',
    ),
    'Puffin-World-Pro': dict(
        config='configs/pipelines/final_stage_2_base_qwen2_5_1_5b_radiov4H_sd3p5L.py',
    ),
}

# The diffusion pipeline requires height/width divisible by 16.
SIZE_MULTIPLE = 16

# All 640-scale (h, w) combinations used when building the dataset
# (scripts/evaluation/generation.py); only those divisible by SIZE_MULTIPLE
# are eligible at inference time.
RESOLUTIONS = [
    (640, 640),
    (640, 480), (480, 640),
    (640, 426), (426, 640),
    (640, 360), (360, 640),
]
VALID_RESOLUTIONS = [
    (h, w) for (h, w) in RESOLUTIONS
    if h % SIZE_MULTIPLE == 0 and w % SIZE_MULTIPLE == 0
]

DEFAULT_PROMPT = ("A cozy living room with wooden furniture, a soft sofa, "
                  "and warm sunlight streaming through a large window.")
DEFAULT_CAMERA = "0.05, -0.12, 1.2"   # roll, pitch, vfov (radians); k1 = 0


def build_caption(args):
    """Return the final caption WITH a trailing camera-parameter sentence.

    If the caption already embeds a parameter block ("... are: r, p, v, k."),
    it is used as-is; otherwise the --camera values are appended.
    """
    caption = args.prompt.strip()
    try:
        parse_camera_params(caption, mode='radial')
        return caption  # caption already carries the parameters
    except ValueError:
        pass

    values = [v for v in args.camera.replace(',', ' ').split() if v]
    assert len(values) in (3, 4), (
        f"--camera expects 'roll, pitch, vfov[, k1]' in radians "
        f"(3 or 4 comma-separated numbers), got {args.camera!r}.")
    if len(values) == 3:
        values.append('0.0')  # k1 defaults to 0 (no radial distortion)
    roll, pitch, vfov, k1 = (float(v) for v in values)
    return (f"{caption} The camera parameters (roll, pitch, "
            f"field-of-view, and radial distortion) are: "
            f"{roll}, {pitch}, {vfov}, {k1}.")


def hstack_adaptive(images, target_h=None):
    """Seamless 1-row tile: NO gaps, NO cropping. Every image is rescaled
    (aspect ratio preserved) to a common height; widths adapt per image."""
    target_h = target_h or max(im.height for im in images)
    scaled = [
        im.resize((max(1, round(im.width * target_h / im.height)), target_h),
                  Image.LANCZOS)
        for im in images
    ]
    row = Image.new('RGB', (sum(im.width for im in scaled), target_h))
    x = 0
    for im in scaled:
        row.paste(im, (x, 0))
        x += im.width
    return row


def fig_to_pil(fig):
    """Rasterize a matplotlib figure to a PIL image and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    image.load()
    return image


def render_pf_overlay(image_uint8_hwc, pf_field, kind):
    """Overlay one PF component on one generated image -> PIL image.

    pf_field: up [2, H, W] or latitude [H, W], radians (unnormalized).
    kind: 'up' | 'lat'.
    """
    img01 = torch.from_numpy(image_uint8_hwc).float() / 255.0  # [H, W, 3]
    fig = plot_images([img01.numpy()])
    if kind == 'up':
        plot_vector_fields([pf_field], axes=fig.axes)
    else:
        plot_latitudes([pf_field], is_radians=True, axes=fig.axes)
    return fig_to_pil(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('prompt', nargs='?', default=DEFAULT_PROMPT,
                        help='Scene caption shared by ALL generated samples '
                             f'(default: "{DEFAULT_PROMPT}").')
    parser.add_argument('--model', default='Puffin-World-Base',
                        choices=sorted(MODEL_REGISTRY.keys()),
                        help='Which pretrained pipeline to use (default: Puffin-World-Base).')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Trained weights (.pth file or DeepSpeed iter_* dir).')
    parser.add_argument('--output', default='output/demo_gen/spatial_simulation.png',
                        type=str, help='Output image FILE path (the 1-row tile).')
    parser.add_argument('--camera', default=DEFAULT_CAMERA, type=str,
                        help="Camera parameters 'roll, pitch, vfov[, k1]' in "
                             "radians, one comma-separated string; k1 is "
                             f"optional and defaults to 0 (default: "
                             f"'{DEFAULT_CAMERA}'). Ignored when the caption "
                             "already embeds a parameter block.")
    parser.add_argument('--num', default=4, type=int,
                        help='Number of samples to generate with the same '
                             'prompt and camera (default 4).')
    parser.add_argument('--vis_pf', action='store_true',
                        help='Also save two figures visualizing the up field '
                             'and latitude field overlaid on each result '
                             '(<output>_pf_up.png / <output>_pf_lat.png).')
    parser.add_argument('--cfg_prompt', default='', type=str)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--num_steps', default=50, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    caption = build_caption(args)
    roll, pitch, vfov, k1 = parse_camera_params(caption, mode='radial')
    print(f"caption: {caption}")
    print(f"camera (radians): roll={roll:.4f} pitch={pitch:.4f} "
          f"vfov={vfov:.4f} k1={k1:.4f}"
          f"  ->  degrees: roll={math.degrees(roll):.2f} "
          f"pitch={math.degrees(pitch):.2f} vfov={math.degrees(vfov):.2f}")

    # ---- per-sample resolutions (reproducible) + camera maps ----
    rng = random.Random(args.seed)
    resolutions = [rng.choice(VALID_RESOLUTIONS) for _ in range(args.num)]
    print(f"resolutions (h, w): {resolutions}")

    cam_generator = Cam_Generator(mode='radial')
    # 9-ch [ray 6 | PF 3] per sample; PF channels stay in radians here and are
    # normalized by pi/2 only for the model input (dataloader convention).
    cameras = [cam_generator.get_cam(caption, h, w) for (h, w) in resolutions]

    # ---- build the model from the registered pipeline config ----
    entry = MODEL_REGISTRY[args.model]
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

    # ---- N generations, same caption + camera, per-sample resolution ----
    results = []   # list of uint8 [H, W, 3]
    for i, ((h, w), camera) in enumerate(zip(resolutions, cameras)):
        generator = torch.Generator(device=model.device).manual_seed(args.seed + i)
        with torch.no_grad():
            images = model.generate(
                prompt=[caption],
                cfg_prompt=[args.cfg_prompt],
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
                cam_values=[[camera / (math.pi / 2)]],
                generator=generator,
                height=h,
                width=w,
            )
        images = rearrange(images, 'b c h w -> b h w c')
        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        results.append(images[0])
        print(f"sample {i}: {h}x{w} done")

    # ---- 1-row tile of the N results ----
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    row = hstack_adaptive([Image.fromarray(r) for r in results])
    row.save(args.output)
    print(f"Saved -> {args.output}")

    # ---- optional PF visualization: up / latitude rows, 1-1 with results ----
    if args.vis_pf:
        stem, _ = os.path.splitext(args.output)
        up_panels, lat_panels = [], []
        for image, camera in zip(results, cameras):
            up_panels.append(render_pf_overlay(image, camera[6:8].float(), 'up'))
            lat_panels.append(render_pf_overlay(image, camera[8].float(), 'lat'))
        up_path, lat_path = f"{stem}_pf_up.png", f"{stem}_pf_lat.png"
        hstack_adaptive(up_panels).save(up_path)
        hstack_adaptive(lat_panels).save(lat_path)
        print(f"Saved -> {up_path}")
        print(f"Saved -> {lat_path}")
