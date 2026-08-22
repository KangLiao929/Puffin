"""World-modeling demo: ONE dataset sample, model picked by NAME.

Fetches a single sample (initial view + camera trajectory) from an eval
dataset, generates the remaining views with the chosen world model, and
saves the compact artifact set: per-view PNGs, GT/generated GIF + mp4
(with the Genie-style keyboard-control overlay), the frustum
camera-trajectory render and -- when geometry is on -- depth maps plus the
gauge-aligned reconstruction.glb.

Stage IV models trained with ``rgb_blind_to_depth=True`` can also run
RGB-only via ``--geometry off``: the isolation makes RGB/text tokens blind
to depth tokens at inference, so dropping the depth block leaves the RGB
pathway mathematically unchanged (no train/test conflict); generation is
~2x faster and depth/.glb artifacts are skipped.

No accelerate / distributed setup: single-GPU inference.

Example:
    python scripts/demo/world_modeling.py \\
        --model Puffin-World \\
        --checkpoint work_dirs/final_stage_4_world_all_asym_attn_qwen2_5_1_5b_radiov4H_sd3p5L/model_itr23200.pth \\
        --dataset re10k --sample_index 0 \\
        --output output/demo_world
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from scripts.evaluation.modules import (
    annotate_motion_keys,
    build_eval_dataset,
    export_reconstruction_glb,
    normalize_gen_outputs_multi,
    save_frames_as_gif,
    save_frames_as_mp4,
    save_image_tensor,
    save_uint8_image,
    visualize_camera_trajectory,
)

# Model name -> world-model pipeline config (all have the generation branch).
MODEL_REGISTRY = {
    'Puffin-World': dict(
        config='configs/pipelines/final_stage_4_world_all_asym_attn_qwen2_5_1_5b_radiov4H_sd3p5L.py',
    ),
    'Puffin-World-7B': dict(
        config='configs/pipelines/final_stage_4_world_all_asym_attn_qwen2_5_7b_radiov3H_sd3p5M.py',
    ),
    'Puffin-World-RGB': dict(
        config='configs/pipelines/final_stage_3_world_dl3dv_re10k_omni_cam_inject_qwen2_5_1_5b_radiov4H_sd3p5L.py',
    ),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--model', default='Puffin-World',
                   choices=sorted(MODEL_REGISTRY.keys()),
                   help='Which world-model pipeline to use (default: Puffin-World).')
    p.add_argument('--checkpoint', required=True, type=str,
                   help='Trained weights (.pth file or DeepSpeed iter_* dir).')
    p.add_argument('--dataset', default='re10k', type=str,
                   help='Eval dataset name (re10k / dl3dv / puffin_omni / ...).')
    p.add_argument('--sample_index', default=0, type=int,
                   help='Dataset window index to test (default 0).')
    p.add_argument('--test_sceneids_path', default=None, type=str,
                   help='Optional held-out scene-list pickle: restrict the '
                        'dataset to exactly those scenes before indexing.')
    p.add_argument('--geometry', default='config',
                   choices=('config', 'on', 'off'),
                   help="Depth branch at inference. 'config' follows the "
                        "model config; 'off' = RGB-only (valid for "
                        "rgb_blind_to_depth checkpoints: RGB is blind to "
                        "depth tokens, so results are unchanged and "
                        "depth/.glb are skipped).")
    p.add_argument('--num_views', default=8, type=int)
    p.add_argument('--cfg_scale', default=2.0, type=float)
    p.add_argument('--num_steps', default=50, type=int)
    p.add_argument('--height', default=640, type=int)
    p.add_argument('--width', default=640, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--gif_fps', default=6, type=int)
    p.add_argument('--mp4_fps', default=None, type=int,
                   help='mp4 frame rate (default: same as --gif_fps).')
    p.add_argument('--output', default='output/demo_world', type=str,
                   help='Output DIRECTORY.')
    p.add_argument('--device', default='cuda', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- one dataset sample ----
    ds_args = argparse.Namespace(
        data_root=None, cache_path=None, num_views=args.num_views,
        height=args.height, width=args.width, control=None,
        min_interval=None, max_interval=None, chunk=1,
        test_sceneids_path=args.test_sceneids_path)
    dataset = build_eval_dataset(args.dataset, ds_args)
    sample = dataset[args.sample_index]
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    gt_views = sample['pixel_values_init']            # list[T] of [3, H, W]
    pose_strs = [str(p) for p in sample['cam_pose']]
    intr_strs = [str(k) for k in sample['cam_intrinsics']]
    T = len(gt_views)
    print(f"sample {args.sample_index} of {args.dataset}: {T} views")

    # ---- model by name ----
    entry = MODEL_REGISTRY[args.model]
    cfg = Config.fromfile(entry['config'])
    cfg.model.pretrained_pth = None                 # weights come from --checkpoint
    cfg.model.use_activation_checkpointing = False  # inference only
    model = BUILDER.build(cfg.model)
    state_dict = guess_load_checkpoint(args.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=args.device).to(model.dtype)
    model.eval()

    if args.geometry != 'config':
        want = args.geometry == 'on'
        if not want and getattr(model, 'geometry_state', False) \
                and not getattr(model, 'rgb_blind_to_depth', False):
            print("[warn] --geometry off on a checkpoint trained WITHOUT "
                  "rgb_blind_to_depth: RGB attended depth tokens in "
                  "training, RGB-only inference is NOT equivalent.")
        model.geometry_state = want
    geometry = bool(getattr(model, 'geometry_state', False))
    print(f"geometry_state = {geometry}")

    # ---- generate the novel views ----
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    cam_values = [[c.to(args.device) for c in sample['cam_values']]]
    with torch.no_grad():
        output = model.generate_multi_view(
            prompt=[sample.get('text', '') or ''],
            cfg_prompt=[''],
            pixel_values_init=[[v.to(args.device) for v in gt_views]],
            cam_values=cam_values,
            cam_pose=[pose_strs],
            cam_intrinsics=[intr_strs],
            gt_cam_params=[[str(p) for p in sample['gt_cam_params']]]
            if sample.get('gt_cam_params') else None,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            generator=generator,
            height=args.height,
            width=args.width,
            K=1,
            progress_bar=True,
        )
    gen = normalize_gen_outputs_multi(
        output['images'], B=1, T_tgt=T - 1).detach().cpu().float()[0]

    # ---- save artifacts ----
    out = args.output
    os.makedirs(out, exist_ok=True)
    mp4_fps = args.mp4_fps if args.mp4_fps else args.gif_fps

    save_image_tensor(gt_views[0], f"{out}/00_input_gt.png")
    gen_frames = [gt_views[0]] + [gen[i] for i in range(T - 1)]
    for v in range(1, T):
        save_image_tensor(gt_views[v], f"{out}/{v:02d}_target_gt.png")

    keys_frames = annotate_motion_keys(gen_frames, pose_strs)
    save_uint8_image(keys_frames[0], f"{out}/00_novel_view_gen.png")
    for v in range(1, T):
        save_uint8_image(keys_frames[v], f"{out}/{v:02d}_novel_view_gen.png")
    save_frames_as_gif(gt_views, f"{out}/target_gt.gif", fps=args.gif_fps)
    save_frames_as_gif(keys_frames, f"{out}/novel_view_gen.gif", fps=args.gif_fps)
    save_frames_as_mp4(keys_frames, f"{out}/novel_view_gen.mp4", fps=mp4_fps)
    visualize_camera_trajectory(pose_strs[:T], f"{out}/camera_trajectory.png")

    depths = output.get('depths', None)
    if geometry and depths is not None and depths[0]:
        depth_vis = output.get('depths_vis', None)
        if depth_vis is not None:
            for v, dv in enumerate(depth_vis[0]):
                save_image_tensor(dv, f"{out}/{v:02d}_depth.png")
            save_frames_as_gif(depth_vis[0], f"{out}/novel_view_depth.gif",
                               fps=args.gif_fps)
            save_frames_as_mp4(depth_vis[0], f"{out}/novel_view_depth.mp4",
                               fps=mp4_fps)
        export_reconstruction_glb(
            gen_frames, depths[0], pose_strs[:T], intr_strs[:T],
            f"{out}/reconstruction.glb")
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
