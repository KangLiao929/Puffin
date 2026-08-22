"""Annotate dense depth with Depth Anything 3 and align it to sparse depth.

This script reproduces the DA3-aligned dense-depth labels used by Puffin2. It
reads RGB and native sparse depth from local dataset mirrors, predicts monocular
DA3 depth, fits a robust scale+shift on valid sparse pixels, and writes local
.npy depth maps that mirror the source dataset layout.

Supported datasets and output layouts:

    dl3dv:
        <save_root>/<part>/<scene_hash>/dense/depth_da3/<frame>.npy

    scannet:
        <save_root>/scans[_test]/<scene>/depth_da3/<frame>.npy

By default, output_mode=fill_sparse keeps valid sparse depth pixels unchanged
and uses the aligned DA3 prediction only for sparse holes / invalid regions.
Use output_mode=aligned_da3 to save the aligned DA3 prediction everywhere.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


def _find_repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "dust3r").is_dir():
            return parent
    return None


REPO_ROOT = _find_repo_root()
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_env_da3 = os.environ.get("DA3_CODE_ROOT")
if _env_da3:
    for _p in (Path(_env_da3).expanduser().resolve() / "src", Path(_env_da3).expanduser().resolve()):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
elif REPO_ROOT is not None:
    _sibling_da3 = REPO_ROOT.parent / "depth-anything-3"
    for _p in (_sibling_da3 / "src", _sibling_da3):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from depth_anything_3.api import DepthAnything3  # noqa: E402


cv2.setNumThreads(0)

DEFAULT_ROOTS = {
    "dl3dv": "data/dl3dv",
    "scannet": "data/scannet",
}


@dataclass
class FrameData:
    rgb: np.ndarray
    sparse_depth: np.ndarray
    valid_mask: np.ndarray
    fill_mask: np.ndarray


def ensure_rgb(rgb_bgr: np.ndarray, what: str) -> np.ndarray:
    if rgb_bgr is None:
        raise IOError(f"cv2 failed to read image: {what}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"cv2 failed to read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask >= 127


def resize_rgb_to_depth(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if rgb.shape[:2] == depth.shape[:2]:
        return rgb
    return cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)


def robust_scale_shift(
    pred: np.ndarray,
    sparse: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_points: int = 32,
    n_iter: int = 5,
    clip: float = 3.0,
    pct_clip: float = 98.0,
) -> tuple[float, float, int]:
    """Fit s,t so s * pred + t matches sparse depth on valid sparse pixels."""
    mask = valid_mask & np.isfinite(pred) & np.isfinite(sparse) & (sparse > 0)
    if mask.any() and pct_clip is not None:
        high = np.percentile(sparse[mask], pct_clip)
        mask = mask & (sparse <= high)

    p = pred[mask].astype(np.float64)
    g = sparse[mask].astype(np.float64)
    if p.size < min_points:
        return 1.0, 0.0, int(p.size)

    inlier = np.ones_like(p, dtype=bool)
    scale, shift = 1.0, 0.0
    for _ in range(n_iter):
        pp = p[inlier]
        gg = g[inlier]
        if pp.size < min_points:
            break
        a = np.stack([pp, np.ones_like(pp)], axis=1)
        sol, *_ = np.linalg.lstsq(a, gg, rcond=None)
        scale, shift = float(sol[0]), float(sol[1])

        residual = np.abs(scale * p + shift - g)
        mad = np.median(residual[inlier]) + 1e-8
        new_inlier = residual < clip * 1.4826 * mad
        if new_inlier.sum() == inlier.sum():
            inlier = new_inlier
            break
        inlier = new_inlier

    return scale, shift, int(inlier.sum())


def align_to_sparse(
    pred: np.ndarray,
    sparse: np.ndarray,
    valid_mask: np.ndarray,
    fill_mask: np.ndarray,
    *,
    output_mode: str,
    min_depth: float,
    pct_clip: float,
) -> tuple[np.ndarray, float, float, int]:
    scale, shift, n_align = robust_scale_shift(
        pred, sparse, valid_mask, pct_clip=pct_clip
    )
    aligned = np.clip(scale * pred + shift, a_min=min_depth, a_max=None).astype(np.float32)

    if output_mode == "aligned_da3":
        return aligned, scale, shift, n_align

    output = aligned.copy()
    keep_sparse = valid_mask & np.isfinite(sparse) & (sparse > 0) & ~fill_mask
    output[keep_sparse] = sparse[keep_sparse].astype(np.float32)
    output[fill_mask] = aligned[fill_mask]
    return output, scale, shift, n_align


@torch.inference_mode()
def predict_depth_batch(model, rgb_list: list[np.ndarray], process_res: int) -> list[np.ndarray]:
    """Run DA3 as independent monocular views and resize back with nearest."""
    device = model.device or next(model.parameters()).device
    sizes = [(im.shape[0], im.shape[1]) for im in rgb_list]

    proc, _, _ = model.input_processor(
        rgb_list, None, None, process_res, "upper_bound_resize"
    )
    batch = proc.shape[0]
    inp = proc.reshape(batch, 1, *proc.shape[1:]).to(device).float()

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        out = model.model(inp, None, None, [], False, False, "saddle_balanced")
    depth = out["depth"][:, 0].float().cpu().numpy()

    results = []
    for d, (height, width) in zip(depth, sizes):
        if d.shape != (height, width):
            d = cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST)
        results.append(d.astype(np.float32))
    return results


def predict_depth_bucketed(model, rgb_list: list[np.ndarray], process_res: int) -> list[np.ndarray]:
    """Batch same-shaped images together while preserving original order."""
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, image in enumerate(rgb_list):
        buckets[image.shape].append(idx)

    results: list[np.ndarray | None] = [None] * len(rgb_list)
    for idxs in buckets.values():
        preds = predict_depth_batch(model, [rgb_list[i] for i in idxs], process_res)
        for i, pred in zip(idxs, preds):
            results[i] = pred
    if any(r is None for r in results):
        raise RuntimeError("DA3 prediction returned incomplete results")
    return [r for r in results if r is not None]


class DatasetAdapter:
    name = "base"

    def list_scenes(self, args) -> list[str]:
        raise NotImplementedError

    def frame_names(self, scene: str, args) -> list[str]:
        raise NotImplementedError

    def load_frame(self, scene: str, frame: str, args) -> FrameData:
        raise NotImplementedError

    def out_path(self, save_root: str, scene: str, frame: str) -> str:
        raise NotImplementedError

    def local_scene_from_path(self, path: str) -> str:
        return osp.basename(path.rstrip("/"))


class DL3DVAdapter(DatasetAdapter):
    name = "dl3dv"

    def list_scenes(self, args) -> list[str]:
        scenes = []
        for top in sorted(os.listdir(args.root)):
            top_path = osp.join(args.root, top)
            if not osp.isdir(top_path):
                continue
            for sub in sorted(os.listdir(top_path)):
                dense = osp.join(top_path, sub, "dense")
                if osp.isdir(osp.join(dense, "rgb")) and osp.isdir(osp.join(dense, "depth")):
                    scenes.append(osp.join(top, sub))
        return scenes

    def frame_names(self, scene: str, args) -> list[str]:
        dense = self._dense_dir(scene, args)
        return sorted(n for n in os.listdir(osp.join(dense, "rgb")) if n.endswith(".png"))

    def load_frame(self, scene: str, frame: str, args) -> FrameData:
        dense = self._dense_dir(scene, args)
        base = frame[:-4]
        rgb_path = osp.join(dense, "rgb", frame)
        rgb = ensure_rgb(cv2.imread(rgb_path, cv2.IMREAD_COLOR), rgb_path)
        depth = np.load(osp.join(dense, "depth", base + ".npy")).astype(np.float32)
        sky = read_mask(osp.join(dense, "sky_mask", frame))
        outlier = read_mask(osp.join(dense, "outlier_mask", frame))

        rgb = resize_rgb_to_depth(rgb, depth)
        invalid_region = sky | outlier
        valid = np.isfinite(depth) & (depth > 0) & ~invalid_region
        fill = (~np.isfinite(depth)) | (depth <= 0) | invalid_region
        return FrameData(rgb=rgb, sparse_depth=depth, valid_mask=valid, fill_mask=fill)

    def out_path(self, save_root: str, scene: str, frame: str) -> str:
        return osp.join(save_root, scene, "dense", "depth_da3", frame[:-4] + ".npy")

    def local_scene_from_path(self, path: str) -> str:
        p = Path(path.rstrip("/"))
        if p.name == "dense":
            p = p.parent
        if p.parent.name.endswith("K"):
            return osp.join(p.parent.name, p.name)
        return p.name

    def _dense_dir(self, scene: str, args) -> str:
        if args.local_scene:
            local = args.local_scene.rstrip("/")
            return local if osp.basename(local) == "dense" else osp.join(local, "dense")
        return osp.join(args.root, scene, "dense")


class ScanNetAdapter(DatasetAdapter):
    name = "scannet"

    def list_scenes(self, args) -> list[str]:
        scenes = []
        for folder in split_csv(args.folders):
            folder_root = osp.join(args.root, folder)
            if not osp.isdir(folder_root):
                continue
            for scene in sorted(os.listdir(folder_root)):
                scene_dir = osp.join(folder_root, scene)
                if osp.isdir(osp.join(scene_dir, "color")) and osp.isdir(osp.join(scene_dir, "depth")):
                    scenes.append(osp.join(folder, scene))
        return scenes

    def frame_names(self, scene: str, args) -> list[str]:
        scene_dir = self._scene_dir(scene, args)
        return sorted(
            n[:-4] for n in os.listdir(osp.join(scene_dir, "color")) if n.endswith(".jpg")
        )

    def load_frame(self, scene: str, frame: str, args) -> FrameData:
        scene_dir = self._scene_dir(scene, args)
        rgb_path = osp.join(scene_dir, "color", frame + ".jpg")
        rgb = ensure_rgb(cv2.imread(rgb_path, cv2.IMREAD_COLOR), rgb_path)
        depth_path = osp.join(scene_dir, "depth", frame + ".png")
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise IOError(f"cv2 failed to read depth: {depth_path}")
        depth = depth_png.astype(np.float32) / 1000.0
        depth[~np.isfinite(depth)] = 0.0
        rgb = resize_rgb_to_depth(rgb, depth)
        valid = np.isfinite(depth) & (depth > 0)
        fill = (~np.isfinite(depth)) | (depth <= 0)
        return FrameData(rgb=rgb, sparse_depth=depth, valid_mask=valid, fill_mask=fill)

    def out_path(self, save_root: str, scene: str, frame: str) -> str:
        return osp.join(save_root, scene, "depth_da3", frame + ".npy")

    def local_scene_from_path(self, path: str) -> str:
        p = Path(path.rstrip("/"))
        parent = p.parent.name
        return osp.join(parent, p.name) if parent in {"scans", "scans_test"} else p.name

    def _scene_dir(self, scene: str, args) -> str:
        if args.local_scene:
            return args.local_scene.rstrip("/")
        return osp.join(args.root, scene)


ADAPTERS = {
    "dl3dv": DL3DVAdapter(),
    "scannet": ScanNetAdapter(),
}


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def default_model_path() -> str:
    if REPO_ROOT is not None:
        sibling = REPO_ROOT.parent / "depth-anything-3" / "develop" / "weights" / "DA3-GIANT-1.1"
        if sibling.exists():
            return str(sibling)
    return "DA3-GIANT-1.1"


def load_model(args, device: torch.device):
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    model.device = device
    model.eval()
    return model


def process_scene(model, adapter: DatasetAdapter, scene: str, args, tag: str = ""):
    frames = adapter.frame_names(scene, args)
    if args.limit_frames:
        frames = frames[: args.limit_frames]

    todo = [
        f for f in frames
        if args.overwrite or not osp.exists(adapter.out_path(args.save_root, scene, f))
    ]
    if not todo:
        print(f"{tag}{scene}: already complete", flush=True)
        return []

    stats = []
    done = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        loaded = []
        for frame in batch:
            try:
                loaded.append((frame, adapter.load_frame(scene, frame, args)))
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}skip frame {scene}/{frame}: {exc!r}", flush=True)

        if not loaded:
            continue

        preds = predict_depth_bucketed(model, [item.rgb for _, item in loaded], args.process_res)
        for (frame, item), pred in zip(loaded, preds):
            if pred.shape != item.sparse_depth.shape:
                pred = cv2.resize(
                    pred,
                    (item.sparse_depth.shape[1], item.sparse_depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            out, scale, shift, n_align = align_to_sparse(
                pred,
                item.sparse_depth,
                item.valid_mask,
                item.fill_mask,
                output_mode=args.output_mode,
                min_depth=args.min_depth,
                pct_clip=args.align_pct_clip,
            )
            out_path = adapter.out_path(args.save_root, scene, frame)
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            np.save(out_path, out.astype(np.float32))
            stats.append((frame, scale, shift, n_align))

        done += len(loaded)
        if stats:
            last = stats[-1]
            print(
                f"{tag}{scene} [{done}/{len(todo)}] last={last[0]} "
                f"s={last[1]:.4f} t={last[2]:.4f} n_align={last[3]}",
                flush=True,
            )

    return stats


def merge_failure_files(save_root: str, num_ranks: int):
    merged = osp.join(save_root, "failed_scenes.txt")
    lines = []
    for rank in range(num_ranks):
        path = osp.join(save_root, f".failed_scenes.rank{rank}.txt")
        if osp.exists(path):
            with open(path, encoding="utf-8") as f:
                lines.extend(f.readlines())
            os.remove(path)
    if lines:
        with open(merged, "a", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"{len(lines)} failed scene(s) -> {merged}", flush=True)
    else:
        print("no failed scenes", flush=True)


def worker(rank: int, gpus: list[int], scenes: list[str], args):
    gpu = gpus[rank]
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    tag = f"[gpu{gpu}] "

    adapter = ADAPTERS[args.dataset]
    model = load_model(args, device)
    assigned = scenes[rank::len(gpus)]
    fail_path = osp.join(args.save_root, f".failed_scenes.rank{rank}.txt")
    print(f"{tag}{len(assigned)} scenes assigned", flush=True)

    for idx, scene in enumerate(assigned):
        out_probe_dir = osp.dirname(adapter.out_path(args.save_root, scene, "__probe__"))
        if args.stop_on_done and osp.isdir(out_probe_dir) and any(
            name.endswith(".npy") for name in os.listdir(out_probe_dir)
        ):
            print(f"{tag}stop-on-done: hit existing scene {scene}", flush=True)
            break

        print(f"{tag}[{idx + 1}/{len(assigned)}] {scene}", flush=True)
        t0 = time.time()
        try:
            process_scene(model, adapter, scene, args, tag=tag)
            print(f"{tag}{scene} done in {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            os.makedirs(args.save_root, exist_ok=True)
            with open(fail_path, "a", encoding="utf-8") as f:
                f.write(f"{scene}\t{exc!r}\n")
            print(f"{tag}FAILED {scene}: {exc!r}\n{tb}", flush=True)


def discover_scenes(args) -> list[str]:
    adapter = ADAPTERS[args.dataset]
    if args.local_scene:
        return [adapter.local_scene_from_path(args.local_scene)]

    cache = args.scene_list or osp.join(args.save_root, "scene_list.txt")
    if osp.exists(cache):
        with open(cache, encoding="utf-8") as f:
            scenes = [line.strip() for line in f if line.strip()]
        print(f"loaded {len(scenes)} scenes from {cache}", flush=True)
    else:
        print("listing local scenes", flush=True)
        scenes = adapter.list_scenes(args)
        os.makedirs(osp.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            f.write("\n".join(scenes) + "\n")
        print(f"listed {len(scenes)} scenes -> {cache}", flush=True)

    if args.num_scenes:
        scenes = scenes[: args.num_scenes]
    if args.reverse:
        scenes = list(reversed(scenes))
    return scenes


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    parser.add_argument(
        "--root",
        default=None,
        help="Local source dataset root. Defaults to data/<dataset>.",
    )
    parser.add_argument(
        "--save-root",
        default=None,
        help="Local output root. Defaults to data/depth_da3/<dataset>.",
    )
    parser.add_argument(
        "--model",
        default=default_model_path(),
        help="Local DA3 weights directory or a model id available to DepthAnything3.",
    )
    parser.add_argument("--folders", default="scans", help="ScanNet top folders.")
    parser.add_argument("--process-res", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids.")
    parser.add_argument("--scene-list", default=None)
    parser.add_argument("--num-scenes", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--local-scene", default=None, help="Local scene or dense dir for debugging.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--align-pct-clip", type=float, default=98.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument(
        "--output-mode",
        choices=("fill_sparse", "aligned_da3"),
        default="fill_sparse",
        help="fill_sparse reproduces Puffin2 labels; aligned_da3 saves DA3 everywhere.",
    )
    args = parser.parse_args(argv)
    if args.root is None:
        args.root = DEFAULT_ROOTS[args.dataset]
    if args.save_root is None:
        args.save_root = osp.join("data", "depth_da3", args.dataset)
    return args


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    os.makedirs(args.save_root, exist_ok=True)

    gpus = [int(g) for g in split_csv(args.gpus)]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    scenes = discover_scenes(args)
    print(f"{len(scenes)} scenes over {len(gpus)} gpu(s): {gpus}", flush=True)

    if len(gpus) == 1:
        worker(0, gpus, scenes, args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, nprocs=len(gpus), args=(gpus, scenes, args), join=True)

    merge_failure_files(args.save_root, len(gpus))


if __name__ == "__main__":
    main()
