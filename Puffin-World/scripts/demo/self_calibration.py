"""Self-calibration demo: VLM camera estimate -> cascaded roll/pitch correction.

Given ONE image (or several), the merged und+gen model first estimates the
camera (roll, pitch, vfov, k1) with its VLM branch. Each axis whose absolute
angle exceeds a threshold (default 5 deg) is corrected — roll alone, pitch
alone, or both in cascade:

  1. It derives the ACTION SEQUENCE(S), each an 8-view unit with the per-view
     step adapted to the residual angle (step = |angle| / 7 per view):
       - roll stage (when |roll| > thr):  rotate roll  -> 0 over 8 views,
       - pitch stage (when |pitch| > thr): rotate pitch -> 0 over 8 views,
         anchored on the previous stage's last frame when both run,
     and writes them out (actions.json + a plain-text description).
  2. It generates the correction with the multi-view world model, one chunk
     per triggered stage (num_views=8; chunk=2 -> 15 poses when both axes
     trigger, chunk=1 -> 8 poses for a single axis; cascade linkage via the
     standard chunked-AR latent/PF relay).
  3. It saves per-stage frames, perspective-field visualizations and a
     combined GIF (rgb | pf-up | pf-lat) of the whole correction.

SEQUENCE MODE: a DIRECTORY argument is treated as an ordered image sequence
(frames sorted by name). Every frame gets the VLM gravity-latitude check in
order; the FIRST frame whose |roll| or |pitch| exceeds the threshold triggers
the correction above (actions + corrected images), after which monitoring
stops (one anomaly per sequence by convention). The per-frame check log goes
to seq_check.json.

Checkpoint: the MERGED und+gen weight (scripts/merge_und_gen_ckpt.py output),
same requirement as freeview_world_exp.py -- stage-3/4 checkpoints alone have
no llm/projector/visual_encoder weights and cannot run the VLM estimate.

Example (Puffin-Traj-1M-Bench frames):
  python scripts/demo/self_calibration.py <scene>/000001.jpg \
      --checkpoint work_dirs/final_stage_4_world_all_asym_attn_qwen2_5_1_5b_radiov4H_sd3p5L/demo_freeview_und_gen_1_5b.pth \
      --output output/demo_self_calib
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from einops import rearrange
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.demo.freeview_world_exp import (          # noqa: E402
    build_model, image_to_text, split_caption, intrinsics_from_vfov, run_i2i)
from scripts.evaluation.modules.trajectories import (  # noqa: E402
    build_custom_trajectory, _compose_ypr_c2w)
from scripts.evaluation.modules import (               # noqa: E402
    save_frames_as_gif, compose_row_gif, save_image_tensor,
    save_pf_visualizations, visualize_camera_trajectory, annotate_motion_keys)

NUM_VIEWS = 8   # views per action unit (stage); 7 inter-frame steps


# ---------------------------------------------------------------------------
# input image
# ---------------------------------------------------------------------------
def load_input_image(path, res=640):
    """Center-crop to square + resize -> [3,res,res] in [-1,1]."""
    pil = Image.open(path).convert('RGB')
    w, h = pil.size
    s = min(w, h)
    pil = pil.crop(((w - s) // 2, (h - s) // 2,
                    (w - s) // 2 + s, (h - s) // 2 + s))
    if pil.size != (res, res):
        pil = pil.resize((res, res), Image.LANCZOS)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return rearrange(torch.from_numpy(arr * 2.0 - 1.0), 'h w c -> c h w'), pil


# ---------------------------------------------------------------------------
# action sequences
# ---------------------------------------------------------------------------
def build_action(axis, start_deg, stage_no=None, num_views=NUM_VIEWS):
    """One 8-view correction unit: rotate <axis> from start_deg to 0 with the
    per-view step adapted to the residual angle (linear, view 0 = anchor).
    stage_no=None marks a below-threshold axis (no generation stage)."""
    steps = num_views - 1
    step_deg = -start_deg / steps if steps else 0.0
    per_view = [start_deg + step_deg * t for t in range(num_views)]
    if axis == 'roll':
        direction = 'counter-clockwise' if step_deg > 0 else 'clockwise'
    else:
        direction = 'upward' if step_deg > 0 else 'downward'
    if stage_no is None:
        desc = (f"({axis}): {start_deg:+.2f} deg is within the threshold -- "
                f"no correction stage.")
    else:
        desc = (f"Stage {stage_no} ({axis}): rotate the camera {axis} from "
                f"{start_deg:+.2f} deg to 0 deg ({direction}) over {steps} "
                f"steps of {abs(step_deg):.2f} deg/view ({num_views} views).")
    return dict(axis=axis, triggered=stage_no is not None,
                start_deg=round(start_deg, 4), end_deg=0.0,
                num_views=num_views, steps=steps,
                step_deg=round(step_deg, 4),
                per_view_deg=[round(v, 4) for v in per_view],
                description=desc)


def build_correction_poses(roll_deg, pitch_deg, stages):
    """Cascaded pose list for the triggered stages ('roll' and/or 'pitch').

    Anchored `build_custom_trajectory` advances ONLY the selected euler angle
    of the anchor pose (loader convention Ry@Rx@Rz, translation frozen), so a
    later stage anchored on the previous stage's last pose keeps the already-
    corrected angle at 0. Returns 8 + 7*(len(stages)-1) world-anchored c2w
    poses (stage boundaries shared).
    """
    anchor = np.eye(4, dtype=np.float64)
    anchor[:3, :3] = _compose_ypr_c2w(roll_deg, pitch_deg, 0.0)
    poses = None
    for axis in stages:
        ang = roll_deg if axis == 'roll' else pitch_deg
        combo = ('r' if axis == 'roll' else 'p') + ('-' if ang > 0 else '+')
        seg = build_custom_trajectory(combo, NUM_VIEWS,
                                      step_deg=abs(ang) / (NUM_VIEWS - 1),
                                      base_c2w=anchor)
        poses = list(seg) if poses is None else poses + list(seg)[1:]
        anchor = seg[-1]
    return poses


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def save_stage(out_dir, anchor_frame, gen_frames, gif_fps):
    os.makedirs(out_dir, exist_ok=True)
    frames = [anchor_frame] + [f.float().cpu() for f in gen_frames]
    for v, fr in enumerate(frames):
        name = '00_anchor.png' if v == 0 else f'{v:02d}_novel_view_gen.png'
        save_image_tensor(fr, os.path.join(out_dir, name))
    save_frames_as_gif(frames, os.path.join(out_dir, 'stage.gif'), fps=gif_fps)
    return frames


def save_outputs(out_dir, ext_img, results, pose_strs, intr_strs,
                 vlm_text, caption, cam_params, actions, calibrated, args):
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(vlm_text=vlm_text, caption=caption,
                cam_params=dict(zip(('roll', 'pitch', 'vfov', 'k1'),
                                    [float(v) for v in cam_params])),
                cam_params_deg=dict(
                    roll=round(math.degrees(cam_params[0]), 3),
                    pitch=round(math.degrees(cam_params[1]), 3),
                    vfov=round(math.degrees(cam_params[2]), 3)),
                threshold_deg=args.threshold_deg, calibrated=calibrated,
                seed=args.seed, cfg_scale=args.cfg_scale,
                num_steps=args.num_steps)
    if actions is not None:
        with open(os.path.join(out_dir, 'actions.json'), 'w') as f:
            json.dump(actions, f, indent=1)
        with open(os.path.join(out_dir, 'actions.txt'), 'w') as f:
            f.write(actions['summary'] + '\n\n')
            f.write(actions['roll_action']['description'] + '\n')
            f.write(actions['pitch_action']['description'] + '\n')
        meta['actions_summary'] = actions['summary']
        meta['stages'] = actions['stages']
    save_image_tensor(ext_img, os.path.join(out_dir, '00_input.png'))
    if results is None:                       # below threshold: no generation
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        return

    stages = actions['stages']
    gen = [f.float().cpu() for f in results['gen_images_tensor'][0]]
    n = NUM_VIEWS - 1                          # 7 generated frames per stage
    full, anchor = None, ext_img
    for i, axis in enumerate(stages):
        frames = save_stage(os.path.join(out_dir, f'stage{i + 1}_{axis}'),
                            anchor, gen[i * n:(i + 1) * n], args.gif_fps)
        full = frames if full is None else full + frames[1:]
        anchor = frames[-1]
    motion_strs = (results.get('motion_pose_strs') or [None])[0] or pose_strs
    keys_frames = annotate_motion_keys(full, motion_strs)
    save_frames_as_gif(keys_frames, os.path.join(out_dir, 'self_calibration.gif'),
                       fps=args.gif_fps)

    # pf panels driven by the ACTUALLY propagated per-view camera params;
    # combined.gif = rgb | pf-up | pf-lat (the diagonal pf-mix stays in
    # pf_vis/ only, matching the eval convention)
    pp = (results.get('pp_cam_params_full') or [None])[0]
    panels = [keys_frames]
    if pp:
        pf_up, pf_lat, _pf_mix = save_pf_visualizations(
            full, pp, list(intr_strs), os.path.join(out_dir, 'pf_vis'),
            fps=args.gif_fps)
        panels += [pf_up, pf_lat]
        meta['pp_cam_params_full'] = pp
    compose_row_gif(panels, os.path.join(out_dir, 'combined.gif'),
                    fps=args.gif_fps,
                    mp4_path=os.path.join(out_dir, 'combined.mp4'),
                    mp4_fps=args.gif_fps)
    visualize_camera_trajectory(list(pose_strs),
                                os.path.join(out_dir, 'camera_trajectory.png'))
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images', nargs='+', help='input image path(s)')
    p.add_argument('--config', default=(
        'configs/pipelines/'
        'final_stage_4_world_all_asym_attn_qwen2_5_1_5b_radiov4H_'
        'sd3p5L.py'))
    p.add_argument('--checkpoint', required=True,
                   help='MERGED und+gen .pth (merge_und_gen_ckpt.py)')
    p.add_argument('--output', default='output/demo_self_calib')
    p.add_argument('--threshold_deg', type=float, default=5.0,
                   help='|roll| or |pitch| above this (deg) triggers '
                        'the correction generation')
    p.add_argument('--force', action='store_true',
                   help='generate even below the threshold')
    p.add_argument('--cfg_scale', type=float, default=2.0,
                   help='I2I cfg (2 = the sweep-validated multi-view default)')
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gif_fps', type=int, default=4)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def estimate_camera(model, pil):
    """VLM gravity-latitude check: caption + (roll, pitch, vfov, k1) rad."""
    vlm_text = image_to_text(model, pil)
    try:
        caption, cam = split_caption(vlm_text)
    except ValueError:
        print('  !! VLM output carried no camera block; assuming default cam')
        caption, cam = vlm_text, (0.0, 0.0, 1.0, 0.0)
    return vlm_text, caption, cam


def calibrate_image(model, path, out_dir, args, precomputed=None):
    """Check ONE image and, when |roll| or |pitch| exceeds the threshold,
    generate the per-axis correction. `precomputed` optionally carries an
    already-run (vlm_text, caption, cam) estimate (sequence mode).
    Returns (corrected, cam)."""
    ext_img, pil = load_input_image(path, args.height)
    vlm_text, caption, cam = precomputed or estimate_camera(model, pil)
    roll, pitch, vfov, k1 = cam
    roll_deg, pitch_deg = math.degrees(roll), math.degrees(pitch)
    print(f'  VLM estimate: roll={roll_deg:+.2f} deg '
          f'pitch={pitch_deg:+.2f} deg vfov={math.degrees(vfov):.2f} deg')

    stages = [ax for ax, v in (('roll', roll_deg), ('pitch', pitch_deg))
              if abs(v) > args.threshold_deg]
    need = bool(stages)
    if args.force and not stages:
        stages = ['roll', 'pitch']
    stage_no = {ax: i + 1 for i, ax in enumerate(stages)}
    roll_action = build_action('roll', roll_deg, stage_no.get('roll'))
    pitch_action = build_action('pitch', pitch_deg, stage_no.get('pitch'))
    plan = ' then '.join(f'{ax} -> 0 (8 views)' for ax in stages)
    actions = dict(
        threshold_deg=args.threshold_deg, stages=stages,
        trigger=dict(roll=abs(roll_deg) > args.threshold_deg,
                     pitch=abs(pitch_deg) > args.threshold_deg),
        summary=(f'Self-calibration '
                 f'{"TRIGGERED" if need else "not needed"}: '
                 f'estimated roll {roll_deg:+.2f} deg / pitch '
                 f'{pitch_deg:+.2f} deg vs threshold '
                 f'{args.threshold_deg:.1f} deg.'
                 + (f' Correction: {plan}.' if stages else '')),
        roll_action=roll_action, pitch_action=pitch_action)
    print('  ' + actions['summary'])

    if not (need or args.force):
        save_outputs(out_dir, ext_img, None, None, None,
                     vlm_text, caption, cam, actions, False, args)
        return False, cam

    args.chunk = len(stages)                   # one AR chunk per stage
    poses = build_correction_poses(roll_deg, pitch_deg, stages)
    pose_strs = [np.array2string(p) for p in poses]
    K = intrinsics_from_vfov(vfov, args.height)
    intr_strs = [np.array2string(K)] * len(poses)
    anchor_str = ' '.join(f'{v:.8f}' for v in (roll, pitch, vfov, k1))

    results = run_i2i(model, ext_img, pose_strs, intr_strs, anchor_str, args)
    save_outputs(out_dir, ext_img, results, pose_strs, intr_strs,
                 vlm_text, caption, cam, actions, True, args)
    print(f'  saved -> {out_dir}')
    return True, cam


SEQ_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def monitor_sequence(model, seq_dir, out_dir, args):
    """Ordered-sequence mode: gravity-latitude check every frame in name
    order; on the FIRST frame whose |roll| or |pitch| exceeds the threshold,
    emit the correction actions + corrected images, then STOP monitoring
    (one anomaly per sequence by convention). Writes seq_check.json with the
    per-frame check log."""
    frames = sorted(f for f in os.listdir(seq_dir)
                    if f.lower().endswith(SEQ_EXTS))
    os.makedirs(out_dir, exist_ok=True)
    print(f'===== sequence {seq_dir}: {len(frames)} frames, '
          f'gravity-latitude check (threshold {args.threshold_deg:.1f} deg)')
    checks, anomaly = [], None
    for idx, fn in enumerate(frames):
        path = os.path.join(seq_dir, fn)
        _, pil = load_input_image(path, args.height)
        vlm_text, caption, cam = estimate_camera(model, pil)
        roll_deg = math.degrees(cam[0])
        pitch_deg = math.degrees(cam[1])
        bad = (abs(roll_deg) > args.threshold_deg
               or abs(pitch_deg) > args.threshold_deg)
        checks.append(dict(index=idx, frame=fn,
                           roll_deg=round(roll_deg, 3),
                           pitch_deg=round(pitch_deg, 3), anomaly=bad))
        print(f'  [{idx:03d}] {fn}: roll={roll_deg:+.2f} '
              f'pitch={pitch_deg:+.2f} -> {"ANOMALY" if bad else "ok"}')
        if bad:
            anomaly = fn
            sub = os.path.join(
                out_dir, f'{idx:03d}_{os.path.splitext(fn)[0]}_corrected')
            calibrate_image(model, path, sub, args,
                            precomputed=(vlm_text, caption, cam))
            print(f'  anomaly at frame {idx} ({fn}) corrected; '
                  f'monitoring stopped.')
            break
    with open(os.path.join(out_dir, 'seq_check.json'), 'w') as f:
        json.dump(dict(sequence=os.path.abspath(seq_dir),
                       threshold_deg=args.threshold_deg,
                       num_frames=len(frames), frames_checked=len(checks),
                       anomaly_frame=anomaly, checks=checks), f, indent=1)


def main():
    args = parse_args()
    # run_i2i reads these from args (chunk is set per run = #stages)
    args.num_views, args.chunk = NUM_VIEWS, 1
    args.height = args.width = 640

    model = build_model(args.config, args.checkpoint, args.device,
                        geometry='off')
    for path in args.images:
        if os.path.isdir(path):                # ordered image sequence
            name = os.path.basename(os.path.normpath(path))
            monitor_sequence(model, path,
                             os.path.join(args.output, f'seq_{name}'), args)
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        parent = os.path.basename(os.path.dirname(path))
        out_dir = os.path.join(args.output, f'{parent}_{name}'
                               if parent else name)
        print(f'===== {path} -> {out_dir}')
        calibrate_image(model, path, out_dir, args)


if __name__ == '__main__':
    main()
