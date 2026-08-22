"""Camera-caption annotation for AOSS multi-view datasets.

For ONE dataset per run (dl3dv / re10k / hypersim / mvs_synth / tartanair /
scannet / arkitscenes_highres), iterate over ALL images using exactly the same index the dust3r
loaders use (the cache pkl built by src/dust3r/datasets/<dataset>.py), run the
Puffin VLM (`model.understand`) on each image (longest edge resized to 640,
same preprocessing as scripts/evaluation/understanding.py), parse
(roll, pitch, vfov, k1) from the caption text, and write one JSON per image
under --camera_root with a layout that mirrors the AOSS structure — the same
pattern as the local depth-anything-3 dumps (`depth_da3_path` in dl3dv.py):

    dl3dv     : <camera_root>/<scene>/dense/camera/<frame>.json
    re10k     : <camera_root>/<scene>/camera/<basename>.json
    hypersim  : <camera_root>/<scene>/<fname '...rgb.png' -> '...camera.json'>
    mvs_synth : <camera_root>/<scene>/camera/<basename>.json
    tartanair : <camera_root>/<rel_seq>/<basename>_camera.json
    scannet   : <camera_root>/scans[_test]/<scene>/camera/<basename>.json
    arkitscenes_highres :
                <camera_root>/Training[|Validation]/<scene>/camera/<basename>.json

Each JSON: {"roll": float, "pitch": float, "vfov": float, "k1": float,
            "parse_ok": bool}   (angles in RADIANS; caption text not stored)

These files are later read by the dust3r loaders (camera_caption_root arg)
into per-view `gt_cam_angles`, enabling training with
model.physical_propagation='offline' (per-view perspective field from the
captioned params — the "physical propagation without VLM at train time").

Multi-GPU / multi-node: launch with torchrun or accelerate; images are
sharded by `process_index :: num_processes`. Already-written JSONs are
skipped, so runs are resumable.

Debug mode (--debug_vis N): after captioning, sample N scenes; from each take
8 frames; read view-0's captioned (roll, pitch, vfov); propagate (roll, pitch)
to the other views via the GT relative camera rotations
(R_rel = R_t^T @ R_0, g_t = R_rel @ g_0 — same as model._physical_propagation);
build each view's perspective field and overlay it on the RGB
(make_perspective_figures — same as model._debug_visualize_camera_fields);
save under --debug_dir. Use --debug_only to skip captioning.

Examples:
    # 8-GPU captioning of DL3DV
    torchrun --nproc_per_node=8 scripts/annotation/camera_caption_aoss.py \\
        configs/pipelines/<vlm_or_full>.py --checkpoint <ckpt> \\
        --dataset dl3dv --camera_root /data/.../dl3dv_camera

    # debug 5 sequences after captioning (single process)
    python scripts/annotation/camera_caption_aoss.py \\
        configs/pipelines/<vlm_or_full>.py --checkpoint <ckpt> \\
        --dataset dl3dv --camera_root /data/.../dl3dv_camera \\
        --debug_only --debug_vis 5 --debug_dir output/dl3dv_pf_debug
"""
import argparse
import json
import math
import os
import os.path as osp
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dust3r.oss_file_client import FileClient
from scripts.camera.geometry.camera import SimpleRadial
from scripts.camera.geometry.gravity import Gravity
from scripts.camera.geometry.perspective_fields import get_perspective_field
from scripts.camera.utils.conversions import fov2focal
from scripts.camera.utils.text import parse_camera_params
from scripts.camera.visualization.visualize_batch import make_perspective_figures


PROMPT = (
    "Describe the image in detail. Then reason its spatial distribution "
    "and estimate its camera parameters (roll, pitch, field-of-view, and "
    "radial distortion)."
)


# ---------------------------------------------------------------------------
# Per-dataset adapters: path composition mirrors src/dust3r/datasets/<ds>.py
# ---------------------------------------------------------------------------
def _np_cam_npz(path, k_intr, k_pose):
    cam = np.load(path)
    return cam[k_intr].astype(np.float64), cam[k_pose].astype(np.float64)


def _np_cam_safetensor(path, k_intr, k_pose):
    from safetensors.numpy import load_file
    cam = load_file(path)
    return cam[k_intr].astype(np.float64), cam[k_pose].astype(np.float64)


class DatasetAdapter:
    """Bundles per-dataset path rules. `scene` / `entry` follow exactly the
    (scenes[sceneids[i]], images[i]) structure in the cache pkl built by the
    corresponding dust3r loader."""

    # Datasets without per-frame cam files on AOSS (poses live in the cache
    # pkl instead) set this True; debug-vis then reads cache["trajectories"].
    cam_from_cache = False

    def __init__(self, name, root, cache_path):
        self.name = name
        self.ROOT = root
        self.cache_path = cache_path

    # -- override per dataset --
    def rgb_aoss_path(self, scene, entry):
        raise NotImplementedError

    def cam_aoss_path(self, scene, entry):
        raise NotImplementedError

    def load_cam(self, local_path):
        raise NotImplementedError

    def caption_path(self, cam_root, scene, entry):
        raise NotImplementedError


class DL3DVAdapter(DatasetAdapter):
    # scenes: '1K/<hash>'; entry: 'frame_XXXXX.png'
    def rgb_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, "dense", "rgb", entry)

    def cam_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, "dense", "cam", entry[:-4] + ".npz")

    def load_cam(self, p):
        return _np_cam_npz(p, "intrinsic", "pose")

    def caption_path(self, cam_root, scene, entry):
        return osp.join(cam_root, scene, "dense", "camera", entry[:-4] + ".json")


class RE10KAdapter(DatasetAdapter):
    # scenes: '<scene>'; entry: basename (no ext)
    def rgb_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, "rgb", entry + ".png")

    def cam_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, "cam", entry + ".npz")

    def load_cam(self, p):
        return _np_cam_npz(p, "intrinsics", "pose")

    def caption_path(self, cam_root, scene, entry):
        return osp.join(cam_root, scene, "camera", entry + ".json")


class HypersimAdapter(DatasetAdapter):
    # scenes: '<scene>/<subdir>'; entry: filename ending 'rgb.png' (flat dir)
    def rgb_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, entry)

    def cam_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, entry.replace("rgb.png", "cam.npz"))

    def load_cam(self, p):
        return _np_cam_npz(p, "intrinsics", "pose")

    def caption_path(self, cam_root, scene, entry):
        return osp.join(cam_root, scene, entry.replace("rgb.png", "camera.json"))


class MVSSynthAdapter(DatasetAdapter):
    # scenes: '<scene>'; entry: basename
    def rgb_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, "rgb", entry + ".jpg")

    def cam_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, scene, "cam", entry + ".safetensor")

    def load_cam(self, p):
        return _np_cam_safetensor(p, "intrinsics", "pose")

    def caption_path(self, cam_root, scene, entry):
        return osp.join(cam_root, scene, "camera", entry + ".json")


class TartanAirAdapter(DatasetAdapter):
    # scenes: FULL seq_dir paths '<ROOT><scene>/<Easy|Hard>/<seq>'; entry: basename
    def _rel(self, scene):
        root = self.ROOT if self.ROOT.endswith("/") else self.ROOT + "/"
        return scene[len(root):] if scene.startswith(root) else scene.lstrip("/")

    def rgb_aoss_path(self, scene, entry):
        return osp.join(scene, entry + "_rgb.png")

    def cam_aoss_path(self, scene, entry):
        return osp.join(scene, entry + "_cam.safetensor")

    def load_cam(self, p):
        return _np_cam_safetensor(p, "camera_intrinsics", "camera_pose")

    def caption_path(self, cam_root, scene, entry):
        return osp.join(cam_root, self._rel(scene), entry + "_camera.json")


class ScanNetAdapter(DatasetAdapter):
    # scenes: '<scene>' under scans(_test); entry: basename
    def __init__(self, name, root, cache_path, split="train"):
        super().__init__(name, root, cache_path)
        self.split_subdir = "scans" if split == "train" else "scans_test"

    def rgb_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, self.split_subdir, scene, "color", entry + ".jpg")

    def cam_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, self.split_subdir, scene, "cam", entry + ".safetensor")

    def load_cam(self, p):
        return _np_cam_safetensor(p, "intrinsics", "pose")

    def caption_path(self, cam_root, scene, entry):
        return osp.join(cam_root, self.split_subdir, scene, "camera", entry + ".json")


class ARKitScenesHighResAdapter(DatasetAdapter):
    # Mirrors src/dust3r/datasets/arkitscenes_highres.py:
    #   scenes: '<video_id>' under <Training|Validation>;
    #   entry:  '<video_id>_<timestamp>.png' (cache `images` keeps .png, the
    #           actual RGB lives under vga_wide/ as .jpg).
    # Intrinsics/poses come from each scene's scene_metadata.npz and are
    # stored PER-IMAGE in the cache pkl (no per-frame cam files on AOSS),
    # so debug-vis poses are read from cache["trajectories"].
    cam_from_cache = True

    def __init__(self, name, root, cache_path, split="train"):
        super().__init__(name, root, cache_path)
        self.split_subdir = "Training" if split == "train" else "Validation"

    def rgb_aoss_path(self, scene, entry):
        return osp.join(self.ROOT, self.split_subdir, scene, "vga_wide",
                        entry[:-4] + ".jpg")

    def caption_path(self, cam_root, scene, entry):
        return osp.join(cam_root, self.split_subdir, scene, "camera",
                        entry[:-4] + ".json")


SUMMARY_DIR = "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary"
ADAPTERS = {
    "dl3dv": lambda split: DL3DVAdapter(
        "dl3dv",
        "aoss:s3://yhluo_sgacer/data/tracking/processed_dl3dv_ours_parts/processed_dl3dv_ours/",
        osp.join(SUMMARY_DIR, "dl3dv_index_all.pkl")),
    "re10k": lambda split: RE10KAdapter(
        "re10k",
        "aoss:s3://yhluo_sgacer/data/tracking/processed_re10k/",
        osp.join(SUMMARY_DIR, "re10k_index.pkl")),
    "hypersim": lambda split: HypersimAdapter(
        "hypersim",
        "aoss:s3://yhluo_sgacer/data/tracking/processed_hypersim_new/",
        osp.join(SUMMARY_DIR, "hypersim_index_all.pkl")),
    "mvs_synth": lambda split: MVSSynthAdapter(
        "mvs_synth",
        "aoss:s3://yhluo_sgacer/data/tracking/processed_mvs_synth",
        osp.join(SUMMARY_DIR, "mvs_synth_index_all.pkl")),
    "tartanair": lambda split: TartanAirAdapter(
        "tartanair",
        "aoss:s3://yhluo_sgacer/data/tracking/processed_tartanair/",
        osp.join(SUMMARY_DIR, "tartanair_index_all.pkl")),
    "scannet": lambda split: ScanNetAdapter(
        "scannet",
        "aoss:s3://yhluo_sgacer/data/tracking/processed_scannet/",
        osp.join(SUMMARY_DIR, "scannet_index_all.pkl"),
        split=split),
    # cache_path follows configs/datasets/multi_view/gen_arkitscenes_highres.py
    "arkitscenes_highres": lambda split: ARKitScenesHighResAdapter(
        "arkitscenes_highres",
        "aoss:s3://yhluo_sgacer/data/tracking/processed_arkitscene_highres",
        osp.join(SUMMARY_DIR, "arkitscenes_highres_index.pkl"),
        split=split),
}


# ---------------------------------------------------------------------------
# Index loading + preprocessing (same as scripts/evaluation/understanding.py)
# ---------------------------------------------------------------------------
def load_index(adapter):
    """Load the cache pkl produced by the dust3r loader and flatten to a list
    of (scene, image_entry) over ALL images."""
    if not osp.exists(adapter.cache_path):
        raise FileNotFoundError(
            f"Index cache not found: {adapter.cache_path}. Build it first by "
            f"instantiating the corresponding src/dust3r/datasets loader once "
            f"(it scans AOSS and writes the cache).")
    with open(adapter.cache_path, "rb") as f:
        cache = pickle.load(f)
    scenes, sceneids, images = cache["scenes"], cache["sceneids"], cache["images"]
    return [(scenes[sceneids[i]], images[i]) for i in range(len(images))], cache


def pad_square_tensor(image, pad_value=0):
    h, w = image.shape[-2:]
    if h == w:
        return image
    if h > w:
        pad_left = (h - w) // 2
        p2d = (pad_left, h - w - pad_left, 0, 0)
    else:
        pad_top = (w - h) // 2
        p2d = (0, 0, pad_top, w - h - pad_top)
    return F.pad(image, p2d, "constant", pad_value)


def process_for_model(image, image_size):
    """Resize longest edge to image_size, normalize to [-1, 1], pad square —
    identical to understanding.py::TestDataset._process_image (no ratio)."""
    w, h = image.size
    if w >= h:
        new_w, new_h = image_size, int(h * (image_size / w))
    else:
        new_h, new_w = image_size, int(w * (image_size / h))
    image = image.resize(size=(new_w, new_h))
    pv = torch.from_numpy(np.array(image)).float() / 255.0
    pv = 2.0 * pv - 1.0
    pv = rearrange(pv, 'h w c -> c h w')
    return pad_square_tensor(pv, pad_value=0)


class CaptionItemDataset(torch.utils.data.Dataset):
    """Downloads + preprocesses one image per item in dataloader workers."""

    def __init__(self, items, adapter, image_size):
        self.items = items          # list of (scene, entry)
        self.adapter = adapter
        self.image_size = image_size
        self._fc = None             # lazy per-worker FileClient

    def __len__(self):
        return len(self.items)

    @property
    def fc(self):
        if self._fc is None:
            self._fc = FileClient()
        return self._fc

    def __getitem__(self, idx):
        scene, entry = self.items[idx]
        try:
            rgb_path = self.adapter.rgb_aoss_path(scene, entry)
            temp = self.fc.download_file(rgb_path)
            image = Image.open(temp).convert("RGB")
            pv = process_for_model(image, self.image_size)
            self.fc.cleanup_temp_files()
            return dict(scene=scene, entry=entry, pixel_values=pv, ok=True)
        except Exception as e:
            print(f"[caption] failed to load {scene}/{entry}: {e}")
            return dict(scene=scene, entry=entry,
                        pixel_values=torch.zeros(3, self.image_size, self.image_size),
                        ok=False)


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------
def run_captioning(args, adapter, accelerator):
    items, _ = load_index(adapter)
    accelerator.print(f"[caption] {adapter.name}: {len(items)} images total")

    # Resume: drop already-written captions.
    if not args.overwrite:
        items = [
            (s, e) for (s, e) in items
            if not osp.exists(adapter.caption_path(args.camera_root, s, e))
        ]
    accelerator.print(f"[caption] {len(items)} images remaining after resume filter")

    # Shard across processes (covers multi-node: global process_index).
    shard = items[accelerator.process_index::accelerator.num_processes]
    print(f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
          f"{len(shard)} images in shard", flush=True)
    if not shard:
        return

    # ---- model ----
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device).to(model.dtype)
    model.eval()

    ds = CaptionItemDataset(shard, adapter, args.image_size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, collate_fn=lambda x: x,
    )

    n_done, n_fail = 0, 0
    for batch in tqdm(loader, disable=not accelerator.is_local_main_process,
                      desc=f"caption[{adapter.name}]"):
        good = [b for b in batch if b["ok"]]
        if not good:
            n_fail += len(batch)
            continue
        pixel_values = [b["pixel_values"] for b in good]
        with torch.no_grad():
            texts = model.understand(
                prompt=[PROMPT] * len(good),
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                progress_bar=False,
            )
        for b, text in zip(good, texts):
            try:
                roll, pitch, vfov, k1 = parse_camera_params(text, mode='radial')
                parse_ok = True
            except ValueError:
                roll, pitch, vfov, k1 = 0.0, 0.0, math.radians(90.0), 0.0
                parse_ok = False
            # Sanity ranges (radians): the regex can still latch onto stray
            # numbers in the description (years, room counts, ...). Anything
            # outside physically plausible bounds is marked parse_ok=False so
            # downstream consumers skip it.
            if parse_ok and not (
                abs(roll) <= math.pi / 2 + 1e-3
                and abs(pitch) <= math.pi / 2 + 1e-3
                and 0.0 < vfov < math.pi
                and abs(k1) <= 2.0
            ):
                parse_ok = False
            out_path = adapter.caption_path(args.camera_root, b["scene"], b["entry"])
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "roll": float(roll),
                    "pitch": float(pitch),
                    "vfov": float(vfov),
                    "k1": float(k1),
                    "parse_ok": parse_ok,
                }, f, ensure_ascii=False)
            n_done += 1
        n_fail += len(batch) - len(good)

    print(f"[rank {accelerator.process_index}] wrote {n_done} captions, "
          f"{n_fail} failures", flush=True)


# ---------------------------------------------------------------------------
# Debug visualization: caption -> propagated PF over GT camera poses
# (mirrors model._physical_propagation Step 2-3 + _debug_visualize_camera_fields)
# ---------------------------------------------------------------------------
def run_debug_vis(args, adapter):
    _, cache = load_index(adapter)
    scenes = cache["scenes"]
    sceneids = cache["sceneids"]
    images = cache["images"]
    scene_img_list = cache["scene_img_list"]

    os.makedirs(args.debug_dir, exist_ok=True)
    rng = random.Random(args.seed)
    fc = FileClient()

    n_views = 8
    candidates = [si for si, ids in enumerate(scene_img_list) if len(ids) >= n_views]
    picks = rng.sample(candidates, min(args.debug_vis, len(candidates)))

    for si in tqdm(picks, desc="debug sequences"):
        img_ids = scene_img_list[si]
        scene = scenes[sceneids[img_ids[0]]]
        # evenly spread 8 frames across the sequence
        sel = np.linspace(0, len(img_ids) - 1, n_views).round().astype(int)
        sel_ids = [img_ids[int(k)] for k in sel]   # global image indices
        entries = [images[gid] for gid in sel_ids]

        # view-0 caption must exist
        cap_path = adapter.caption_path(args.camera_root, scene, entries[0])
        if not osp.exists(cap_path):
            print(f"[debug] missing caption for view0 of {scene}, skip "
                  f"({cap_path})")
            continue
        with open(cap_path, "r", encoding="utf-8") as f:
            cap = json.load(f)
        roll0, pitch0, vfov0, k1_0 = cap["roll"], cap["pitch"], cap["vfov"], cap["k1"]

        # download rgbs + cams (poses come from the cache pkl for datasets
        # without per-frame cam files on AOSS, e.g. arkitscenes_highres)
        rgbs, c2ws = [], []
        try:
            for gid, e in zip(sel_ids, entries):
                t_rgb = fc.download_file(adapter.rgb_aoss_path(scene, e))
                rgbs.append(Image.open(t_rgb).convert("RGB"))
                if adapter.cam_from_cache:
                    pose = np.asarray(cache["trajectories"][gid],
                                      dtype=np.float64)
                else:
                    t_cam = fc.download_file(adapter.cam_aoss_path(scene, e))
                    _, pose = adapter.load_cam(t_cam)
                c2w = np.eye(4, dtype=np.float64)
                c2w[:pose.shape[0], :pose.shape[1]] = pose[:4, :4]
                c2ws.append(c2w)
        except Exception as ex:
            print(f"[debug] failed to fetch {scene}: {ex}")
            continue
        finally:
            fc.cleanup_temp_files()

        # propagate (roll, pitch) via relative GT rotations
        # (same formula as model._physical_propagation Step 2)
        R0 = c2ws[0][:3, :3]
        g0 = Gravity.from_rp(torch.tensor(roll0), torch.tensor(pitch0)).vec3d.numpy()
        per_view_rp = []
        for t in range(n_views):
            R_rel = c2ws[t][:3, :3].T @ R0
            g_t = R_rel @ g0
            grav = Gravity(torch.tensor(g_t).float())
            per_view_rp.append((grav.roll.item(), grav.pitch.item()))

        scene_tag = scene.replace("/", "_").strip("_")
        for t in range(n_views):
            img = rgbs[t]
            # resize longest edge to image_size for a manageable figure
            w, h = img.size
            if w >= h:
                nw, nh = args.image_size, int(h * args.image_size / w)
            else:
                nh, nw = args.image_size, int(w * args.image_size / h)
            img_r = img.resize((nw, nh))
            W, H = img_r.size

            roll_t, pitch_t = per_view_rp[t]
            f = fov2focal(torch.tensor(float(vfov0)), torch.tensor(float(H)))
            params = torch.tensor([W, H, float(f), float(f),
                                   W / 2.0, H / 2.0, float(k1_0), 0.0]).float()
            camera = SimpleRadial(params).float().scale(torch.Tensor([1, 1]))
            grav_obj = Gravity.from_rp(torch.tensor(roll_t).float(),
                                       torch.tensor(pitch_t).float())
            up_field, lat_field = get_perspective_field(
                camera, grav_obj, use_up=True, use_latitude=True)

            img_t = torch.from_numpy(
                np.array(img_r).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)

            single = {
                "image": img_t.clamp(0, 1),
                "up_field": up_field.float().cpu(),
                "latitude_field": lat_field.float().cpu(),
            }
            figs = make_perspective_figures(single, single, n_pairs=1)
            for k, fig in figs.items():
                suffix = "_up" if "up" in k else "_lat" if "lat" in k else f"_{k}"
                out = osp.join(
                    args.debug_dir,
                    f"{adapter.name}_{scene_tag}_v{t:02d}{suffix}.png")
                fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

        print(f"[debug] {scene}: view0 caption roll={roll0:.3f} pitch={pitch0:.3f} "
              f"vfov={vfov0:.3f}; propagated rp per view: "
              + " ".join(f"({r:+.3f},{p:+.3f})" for r, p in per_view_rp))

    print(f"[debug] figures saved under {args.debug_dir}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='VLM / pipeline config (must provide model.understand).')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--dataset', required=True, choices=sorted(ADAPTERS.keys()),
                        help='One dataset per run.')
    parser.add_argument('--camera_root', required=True, type=str,
                        help='Local root for caption JSONs (mirrors AOSS layout).')
    parser.add_argument('--split', default='train', choices=('train', 'test'),
                        help='Used by scannet (scans vs scans_test) and '
                             'arkitscenes_highres (Training vs Validation).')
    parser.add_argument('--cache_path', default=None, type=str,
                        help='Override the dataset index cache pkl path.')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--image_size', default=640, type=int)
    parser.add_argument('--max_new_tokens', default=200, type=int)
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-caption even if the JSON already exists.')
    # debug
    parser.add_argument('--debug_vis', type=int, default=0,
                        help='Sample N scenes (8 views each) and visualize the '
                             'captioned + propagated perspective fields.')
    parser.add_argument('--debug_dir', default='output/camera_caption_debug', type=str)
    parser.add_argument('--debug_only', action='store_true',
                        help='Skip captioning; only run the debug visualization.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    adapter = ADAPTERS[args.dataset](args.split)
    if args.cache_path is not None:
        adapter.cache_path = args.cache_path

    accelerator = Accelerator()
    accelerator.print(f"processes: {accelerator.num_processes}")

    if not args.debug_only:
        run_captioning(args, adapter, accelerator)
        accelerator.wait_for_everyone()

    if args.debug_vis > 0 and accelerator.is_main_process:
        run_debug_vis(args, adapter)


if __name__ == "__main__":
    main()
