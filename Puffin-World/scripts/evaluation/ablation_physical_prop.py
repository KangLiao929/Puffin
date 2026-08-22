"""Trajectory-conditioned generation evaluation on a pano-trajectory test set.

Given a test root containing many segment subfolders (each produced by
`dataset/generation/GeoCalib/siclib/datasets/create_dataset_from_pano_trajectory.py`,
i.e. perspective frames + cameras.json with per-frame roll/pitch/yaw/vfov +
4x4 c2w + a segment-level `motion_type` in {roll, pitch, yaw}):

  1. Pick N folders (--num_folders; -1 = all).
  2. In each folder, deterministically select 8 frames using the SAME puffin_omni
     video sampling (min_interval/max_interval, fixed-interval video), seeded so
     the same frames are chosen on every run (--seed). The first selected frame
     is the input/anchor.
  3. Generate the remaining 7 target frames from (frame-0 image + all 8 frames'
     camera conditions), exactly as scripts/evaluation/generation_multi_view.py.
     Supports model.load_gt_camera_params=True (gt_cam_params are always passed;
     the model decides whether to use them).
  4. Compute PSNR / SSIM / LPIPS (generated vs GT), grouped by motion_type and
     overall; print a summary table.
  5. Save the input frame + generated frames under <output>/<segment>/<framename>
     (same structure/naming as the test set).
  6. Save GT camera params of the generated frames as 3 CSVs (one per motion_type)
     in eval_understanding.py's column format.
  7. Estimate camera params from the GENERATED images (understanding.py style),
     saving 3 JSONs (one per motion_type) in understanding.py's [{id, output_text}]
     format. CSV `fname` and JSON `id` use the same key "<segment>/<framename>"
     so they correspond one-to-one.

Run once per model:
    python scripts/evaluation/eval_traj_generation.py \\
        configs/pipelines/final_stage_3_world_puffin_cam_puffin_traj_qwen2_5_7b_radiov3H_sd3p5M.py \\
        --checkpoint work_dirs/<run>/iter_xxxx.pth \\
        --test_root /mnt/.../Trajectory_test/360swiss_8 \\
        --output output/traj_eval/base

    python scripts/evaluation/eval_traj_generation.py \\
        configs/pipelines/final_stage_3_world_puffin_cam_puffin_traj_phys_qwen2_5_7b_radiov3H_sd3p5M.py \\
        --checkpoint work_dirs/<run_phys>/iter_xxxx.pth \\
        --test_root /mnt/.../Trajectory_test/360swiss_8 \\
        --output output/traj_eval/phys --load_gt_camera_params
"""
import argparse
import csv
import json
import math
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from src.dust3r.datasets.puffin_omni import PuffinOmni_Multi
from scripts.evaluation.generation_multi_view import (
    normalize_gen_outputs_multi,
    save_image_tensor,
)

from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:
    LearnedPerceptualImagePatchSimilarity = None

from scripts.camera.utils.text import parse_camera_params


UND_PROMPT = (
    "Describe the image in detail. Then reason its spatial distribution "
    "and estimate its camera parameters (roll, pitch, field-of-view, and "
    "radial distortion)."
)


# ---------------------------------------------------------------------------
# Test dataset: single-level pano-trajectory folders, deterministic selection
# ---------------------------------------------------------------------------
class PuffinTrajTestDataset(PuffinOmni_Multi):
    """Reuses PuffinOmni_Multi's image / ray_map / cam_pose-transpose / gt-angle
    pipeline, but indexes a SINGLE-level test root (each subdir = one segment)
    with exactly ONE entry per segment that starts at frame 0. With a fixed
    `seed`, base __getitem__ reseeds per-idx so frame selection is identical
    on every run.
    """

    def _load_data(self, cache_path=None, index_workers=8):
        root = self.ROOT
        seg_names = sorted([
            d for d in os.listdir(root)
            if osp.isdir(osp.join(root, d))
            and osp.isfile(osp.join(root, d, "cameras.json"))
        ])

        scenes, sceneids, images = [], [], []
        scene_img_list, start_img_ids, scene_dirs = [], [], []
        offset, j = 0, 0
        cut_off = self.num_views  # need at least num_views frames per segment

        for seg in seg_names:
            seg_dir = osp.join(root, seg)
            try:
                with open(osp.join(seg_dir, "cameras.json"), "r") as f:
                    meta = json.load(f)
                frame_names = sorted(fr["file_path"] for fr in meta["frames"])
            except Exception as e:
                print(f"[TrajTest] skip {seg}: {e}")
                continue
            n = len(frame_names)
            if n < cut_off:
                continue
            img_ids = list(np.arange(n) + offset)
            scenes.append(seg)
            scene_dirs.append(seg_dir)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * n)
            images.extend(frame_names)
            start_img_ids.append((seg, img_ids[0]))   # one start, frame 0
            offset += n
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.start_img_ids = start_img_ids
        self.scene_dirs = scene_dirs
        self.scene_to_idx = {s: i for i, s in enumerate(scenes)}
        self.invalid_scenes = {s: False for s in scenes}
        print(f"[TrajTest] indexed {len(self.scenes)} segments, {len(self.images)} frames "
              f"under {root}")

    def __getitem__(self, idx):
        # Base multiview __getitem__ -> _get_views (deterministic via self.seed)
        views = super(PuffinOmni_Multi, self).__getitem__(idx)

        pixel_values = [v["img"] for v in views]
        cam_values = [rearrange(torch.from_numpy(v["ray_map"]), 'h w c -> c h w') for v in views]
        cam_pose = [str(v["camera_pose"]) for v in views]
        gt_cam_params = [
            " ".join(f"{float(x):.8f}" for x in v["gt_cam_angles"])
            for v in views
        ]
        # label = "<scene_key>_<basename>"; recover scene + per-frame basenames.
        scene_key = self.start_img_ids[idx][0]
        frame_names = [v["label"][len(scene_key) + 1:] for v in views]
        # Per-frame GT angles (radians) for CSV: (roll, pitch, vfov, k1)
        gt_angles = [v["gt_cam_angles"].tolist() for v in views]

        seg_dir = self.scene_dirs[self.scene_to_idx[scene_key]]
        with open(osp.join(seg_dir, "cameras.json"), "r") as f:
            motion_type = json.load(f).get("motion_type", "unknown")

        return dict(
            pixel_values=pixel_values,
            pixel_values_init=pixel_values,
            cam_values=cam_values,
            cam_pose=cam_pose,
            cam_intrinsics=[str(v["camera_intrinsics"]) for v in views],
            gt_cam_params=gt_cam_params,
            scene_id=scene_key,
            frame_names=frame_names,
            gt_angles=gt_angles,
            motion_type=motion_type,
            type='image2image',
            text="",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to01(x):
    """[-1, 1] CHW (or BCHW) -> [0, 1], clamped, float32 on CPU."""
    return ((x.float() + 1.0) / 2.0).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='Model/pipeline config path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--test_root', required=True, type=str,
                        help='Root with segment subfolders (frames + cameras.json).')
    parser.add_argument('--output', default='output/traj_eval', type=str)
    parser.add_argument('--num_folders', type=int, default=-1,
                        help='Number of segment folders to test (-1 = all).')
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--min_interval', type=int, default=1)
    parser.add_argument('--max_interval', type=int, default=12)
    parser.add_argument('--resolution', type=int, default=512,
                        help='Square model resolution for generation.')
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=180)
    parser.add_argument('--load_gt_camera_params', action='store_true',
                        help="Force physical_propagation='offline' at test time "
                             "(deprecated flag name kept for compatibility; PF "
                             "from the precomputed per-view VLM annotations).")
    parser.add_argument('--lpips_net', default='alex', choices=('alex', 'vgg'))
    parser.add_argument('--pp_mode', default=None,
                        choices=('offline_prop', 'offline', 'const_pf', 'off'),
                        help="Override physical_propagation at test time. "
                             "'offline_prop' = anchor GT params propagated to "
                             "all views via relative poses (WITH propagation); "
                             "'const_pf' = physical_propagation='offline' but "
                             "every view receives the ANCHOR frame's params, "
                             "i.e. a constant PF (propagation ablated).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    # ---- model ----
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Unexpected parameters: {unexpected}")
    if args.load_gt_camera_params:
        model.physical_propagation = 'offline'
        print("[eval] forced model.physical_propagation = 'offline'")
    if args.pp_mode is not None:
        model.physical_propagation = ('offline' if args.pp_mode == 'const_pf'
                                      else args.pp_mode)
        print(f"[eval] pp_mode={args.pp_mode} -> "
              f"model.physical_propagation = '{model.physical_propagation}'")
    model = model.to(device=device).to(model.dtype)
    model.eval()
    gen_device = next(model.parameters()).device

    # ---- metrics ----
    lpips_metric = None
    if LearnedPerceptualImagePatchSimilarity is not None:
        try:
            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type=args.lpips_net, normalize=True).to(gen_device)
            lpips_metric.eval()
        except Exception as e:
            print(f"[eval] LPIPS unavailable ({e}); reporting N/A.")

    # ---- dataset ----
    dataset = PuffinTrajTestDataset(
        ROOT=args.test_root,
        resolution=args.resolution,
        num_views=args.num_views,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        seed=args.seed,             # deterministic per-folder frame selection
    )
    total = len(dataset)
    n_folders = total if args.num_folders < 0 else min(args.num_folders, total)
    indices = list(range(n_folders))
    print(f"[eval] testing {n_folders} / {total} segments")

    K = 1
    # per-motion-type metric accumulators
    metrics = defaultdict(lambda: defaultdict(list))   # motion -> {psnr/ssim/lpips: [vals]}
    # per-motion-type rows for CSV (GT params) and JSON (estimation)
    csv_rows = defaultdict(list)     # motion -> [dict(fname, roll, pitch, vfov, k1, width, height)]
    json_items = defaultdict(list)   # motion -> [dict(id, output_text)]

    for idx in tqdm(indices, desc="segments"):
        # Reseed per folder so generation noise is reproducible and independent
        # of folder count / iteration order.
        generator = torch.Generator(device=gen_device).manual_seed(args.seed + idx)
        sample = dataset[idx]
        scene_id = str(sample['scene_id']).replace('/', '_')
        motion = sample['motion_type']
        frame_names = sample['frame_names']
        T_all = len(sample['cam_values'])
        T_tgt = T_all - K

        # ---- generation (single-sample batch), mirrors generation_multi_view ----
        cam_pose = [[str(p) for p in sample['cam_pose']]]
        gt_cam_params = [[str(p) for p in sample['gt_cam_params']]]
        if args.pp_mode == 'const_pf':
            # propagation ablated: every view gets the ANCHOR frame's params,
            # so the PF conditioning is constant across the whole window
            gt_cam_params = [[gt_cam_params[0][0]] * T_all]
        # Per-view post-crop intrinsics: physical_propagation derives the PF
        # focal / principal point from these (crop-consistent) instead of the
        # VLM-estimated vfov.
        cam_intrinsics = None
        if sample.get('cam_intrinsics') is not None:
            cam_intrinsics = [[str(k) for k in sample['cam_intrinsics']]]
        output = model.generate_multi_view(
            prompt=[sample['text']],
            cfg_prompt=[""],
            pixel_values_init=[sample['pixel_values_init']],
            cam_values=[sample['cam_values']],
            cam_pose=cam_pose,
            cam_intrinsics=cam_intrinsics,
            gt_cam_params=gt_cam_params,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            progress_bar=False,
            generator=generator,
            height=args.resolution,
            width=args.resolution,
            K=K,
        )
        gen = normalize_gen_outputs_multi(output["images"], B=1, T_tgt=T_tgt)[0]  # [T_tgt, C, H, W]
        gen = gen.detach().cpu().to(torch.float32)

        gt_views = sample['pixel_values_init']   # list[T_all] of [3,H,W] in [-1,1]

        # ---- save input frame (view 0, GT) ----
        seg_out = osp.join(args.output, scene_id)
        os.makedirs(seg_out, exist_ok=True)
        save_image_tensor(gt_views[0], osp.join(seg_out, frame_names[0]))

        # ---- per target frame: save, metrics, accumulate CSV + estimation ----
        for t in range(T_tgt):
            v_idx = t + K
            gen_t = gen[t]                       # [3,H,W] in [-1,1]
            gt_t = gt_views[v_idx]               # [3,H,W] in [-1,1]
            save_image_tensor(gen_t, osp.join(seg_out, frame_names[v_idx]))

            g01 = to01(gen_t).unsqueeze(0).to(gen_device)
            t01 = to01(gt_t).unsqueeze(0).to(gen_device)
            psnr = float(peak_signal_noise_ratio(g01, t01, data_range=1.0).item())
            ssim = float(structural_similarity_index_measure(g01, t01, data_range=1.0).item())
            metrics[motion]['psnr'].append(psnr)
            metrics[motion]['ssim'].append(ssim)
            if lpips_metric is not None:
                with torch.no_grad():
                    lp = float(lpips_metric(g01, t01).item())
                metrics[motion]['lpips'].append(lp)

            # GT params (radians) of this generated frame -> CSV row
            roll, pitch, vfov, k1 = sample['gt_angles'][v_idx]
            key = f"{scene_id}/{frame_names[v_idx]}"
            csv_rows[motion].append({
                "fname": key,
                "roll": f"{roll:.6f}",
                "pitch": f"{pitch:.6f}",
                "vfov": f"{vfov:.6f}",
                "k1": f"{k1:.6f}",
                "width": args.resolution,
                "height": args.resolution,
            })

            # estimate camera params on the GENERATED image (understanding style)
            with torch.no_grad():
                texts = model.understand(
                    prompt=[UND_PROMPT],
                    pixel_values=[gen_t.to(gen_device)],
                    max_new_tokens=args.max_new_tokens,
                    progress_bar=False,
                )
            json_items[motion].append({"id": key, "output_text": texts[0]})

    # ---- write CSVs (GT params) + JSONs (estimation), split by motion_type ----
    for motion in sorted(csv_rows.keys()):
        csv_path = osp.join(args.output, f"cameras_{motion}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "fname", "roll", "pitch", "vfov", "k1", "width", "height"])
            writer.writeheader()
            writer.writerows(csv_rows[motion])
        json_path = osp.join(args.output, f"estimate_{motion}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_items[motion], f, indent=2, ensure_ascii=False)
        print(f"[eval] motion={motion}: wrote {csv_path} ({len(csv_rows[motion])} rows) "
              f"and {json_path}")

    # ---- print metric summary (per motion_type + overall) ----
    def _avg(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    print("\n=== Generation metrics (generated vs GT) ===")
    header = f"{'motion':<10}{'count':>8}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}"
    print(header)
    print("-" * len(header))
    overall = defaultdict(list)
    for motion in sorted(metrics.keys()):
        m = metrics[motion]
        cnt = len(m['psnr'])
        for k in ('psnr', 'ssim', 'lpips'):
            overall[k].extend(m[k])
        lp = _avg(m['lpips']) if m['lpips'] else float('nan')
        print(f"{motion:<10}{cnt:>8}{_avg(m['psnr']):>10.4f}{_avg(m['ssim']):>10.4f}{lp:>10.4f}")
    print("-" * len(header))
    lp_all = _avg(overall['lpips']) if overall['lpips'] else float('nan')
    print(f"{'ALL':<10}{len(overall['psnr']):>8}{_avg(overall['psnr']):>10.4f}"
          f"{_avg(overall['ssim']):>10.4f}{lp_all:>10.4f}")

    # also dump the metric summary to JSON for record
    summary = {
        motion: {
            "count": len(m['psnr']),
            "psnr": _avg(m['psnr']),
            "ssim": _avg(m['ssim']),
            "lpips": _avg(m['lpips']) if m['lpips'] else None,
        }
        for motion, m in metrics.items()
    }
    summary["ALL"] = {
        "count": len(overall['psnr']),
        "psnr": _avg(overall['psnr']),
        "ssim": _avg(overall['ssim']),
        "lpips": lp_all if overall['lpips'] else None,
    }
    with open(osp.join(args.output, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] saved metrics_summary.json + images under {args.output}")


if __name__ == "__main__":
    main()
