"""Script to create a trajectory dataset from panorama images.

Each panorama produces M segments. Each segment is a continuous camera motion
following one of three schemes:
  1) pitch-only motion (constant roll): up, down, up-down, down-up
  2) roll-only motion  (constant pitch): cw, ccw, cw-ccw, ccw-cw
  3) yaw-only motion   (constant roll & pitch): right-left, left-right,
     full_circle (monotonic right / left sweeps are currently disabled)

Per-segment, ``images_per_segment`` frames are rendered at a constant angular
step ``angle_step_deg`` (default 1 deg -> 90 frames cover ~90 deg of motion).
Initial roll/pitch are sampled in [-45, 45] deg; sub-patterns that monotonically
traverse the full span start at +/-45 deg.

Each segment is saved under ``<name>/<pano_stem>_<segment_idx>/`` containing:
  - 000001.jpg ... 00000N.jpg (perspective frames)
  - cameras.json              (per-frame intrinsics/extrinsics + Euler params)
  - <segment>.mp4             (if ``save_video=true``)
"""

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from siclib.geometry.camera import camera_models
from siclib.geometry.gravity import Gravity
from siclib.utils.conversions import deg2rad, fov2focal, rad2rotmat
from siclib.utils.image import load_image, write_image

logger = logging.getLogger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

MOTION_TYPES = ("pitch", "roll", "yaw")
SUB_PATTERNS = ("mono_pos", "mono_neg", "bi_pos_neg", "bi_neg_pos")


def _linear_offset(n: int, step: float, sign: float) -> np.ndarray:
    """Monotonic angular offsets in degrees for ``n`` frames."""
    return sign * step * np.arange(n, dtype=np.float64)


def _triangular_offset(n: int, step: float, sign: float) -> np.ndarray:
    """Triangular-wave angular offsets: 0 -> peak -> back down over ``n`` frames.

    With n=90 and step=1, offsets rise 0..44, peak at 45 mid-segment, then
    fall back to ``step`` (1 deg) on the last frame -- not exactly 0.
    """
    half = n // 2
    up = step * np.arange(half, dtype=np.float64)                      # 0..half-1
    down = step * (n - np.arange(half, n, dtype=np.float64))           # half..1
    return sign * np.concatenate([up, down])


def build_trajectory(
    motion_type: str,
    sub_pattern: str,
    n_frames: int,
    step_deg: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-frame (roll, pitch, yaw) arrays in degrees for one segment."""
    peak = (n_frames // 2) * step_deg         # triangular peak amplitude
    span = (n_frames - 1) * step_deg          # monotonic total span

    rolls = np.zeros(n_frames, dtype=np.float64)
    pitches = np.zeros(n_frames, dtype=np.float64)
    yaws = np.zeros(n_frames, dtype=np.float64)

    # A random base yaw so segments from the same pano sweep different regions.
    base_yaw = rng.uniform(-180.0, 180.0)
    yaws += base_yaw

    if motion_type == "pitch":
        # Constant roll; pitch follows the pattern within [-45, 45].
        init_roll = rng.uniform(-45.0, 45.0)
        rolls[:] = init_roll
        if sub_pattern == "mono_pos":           # down -> up (pitch: -span/2 -> +span/2 -> clamp to -45..+45)
            start = -span / 2.0
            pitches[:] = start + _linear_offset(n_frames, step_deg, +1.0)
        elif sub_pattern == "mono_neg":         # up -> down
            start = +span / 2.0
            pitches[:] = start + _linear_offset(n_frames, step_deg, -1.0)
        elif sub_pattern == "bi_pos_neg":       # up -> down (triangular + direction)
            # init in [-45, 45 - peak] so init + peak stays <= 45
            init = rng.uniform(-45.0, 45.0 - peak)
            pitches[:] = init + _triangular_offset(n_frames, step_deg, +1.0)
        elif sub_pattern == "bi_neg_pos":       # down -> up (triangular, negative side)
            init = rng.uniform(-45.0 + peak, 45.0)
            pitches[:] = init + _triangular_offset(n_frames, step_deg, -1.0)
        else:
            raise ValueError(f"Unknown sub_pattern: {sub_pattern}")

    elif motion_type == "roll":
        init_pitch = rng.uniform(-45.0, 45.0)
        pitches[:] = init_pitch
        if sub_pattern == "mono_pos":           # ccw monotonic
            start = -span / 2.0
            rolls[:] = start + _linear_offset(n_frames, step_deg, +1.0)
        elif sub_pattern == "mono_neg":         # cw monotonic
            start = +span / 2.0
            rolls[:] = start + _linear_offset(n_frames, step_deg, -1.0)
        elif sub_pattern == "bi_pos_neg":
            init = rng.uniform(-45.0, 45.0 - peak)
            rolls[:] = init + _triangular_offset(n_frames, step_deg, +1.0)
        elif sub_pattern == "bi_neg_pos":
            init = rng.uniform(-45.0 + peak, 45.0)
            rolls[:] = init + _triangular_offset(n_frames, step_deg, -1.0)
        else:
            raise ValueError(f"Unknown sub_pattern: {sub_pattern}")

    elif motion_type == "yaw":
        # Constant roll and pitch; yaw sweeps freely (no [-45, 45] constraint).
        rolls[:] = rng.uniform(-25.0, 25.0)
        pitches[:] = rng.uniform(-25.0, 25.0)
        # Monotonic yaw sweeps are disabled (also excluded in
        # _sub_pattern_choice); re-enable both together if needed:
        # if sub_pattern == "mono_pos":           # right (increase yaw)
        #     yaws += _linear_offset(n_frames, step_deg, +1.0)
        # elif sub_pattern == "mono_neg":         # left
        #     yaws += _linear_offset(n_frames, step_deg, -1.0)
        if sub_pattern == "bi_pos_neg":       # right -> left
            yaws += _triangular_offset(n_frames, step_deg, +1.0)
        elif sub_pattern == "bi_neg_pos":       # left -> right
            yaws += _triangular_offset(n_frames, step_deg, -1.0)
        elif sub_pattern == "full_circle":      # 360° orbit
            circle_step = 360.0 / n_frames
            sign = rng.choice([+1.0, -1.0])
            yaws += sign * circle_step * np.arange(n_frames, dtype=np.float64)
        else:
            raise ValueError(f"Unknown sub_pattern: {sub_pattern}")

    else:
        raise ValueError(f"Unknown motion_type: {motion_type}")

    return rolls, pitches, yaws


def euler_to_c2w(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix (rotation-only, zero translation).

    Uses the same Euler convention as GeoCalib's rendering path:
    ``R = rad2rotmat(roll, pitch, yaw)`` maps camera-frame bearings to the
    world/panorama frame.
    """
    r = torch.tensor(roll_rad)
    p = torch.tensor(pitch_rad)
    y = torch.tensor(yaw_rad)
    R = rad2rotmat(r, p, y).numpy().astype(np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    return c2w


def intrinsics_matrix(vfov_rad: float, h: int, w: int) -> np.ndarray:
    """Pinhole intrinsics for the given vertical FoV and image size."""
    f = float(fov2focal(torch.tensor(vfov_rad), torch.tensor(float(h))))
    return np.array(
        [[f, 0.0, w / 2.0],
         [0.0, f, h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def save_mp4(images: List[torch.Tensor], out_path: Path, fps: int) -> None:
    """Encode a list of CHW uint8/float images to an mp4.

    Prefers imageio for simplicity; falls back to cv2 if unavailable.
    """
    frames = []
    for img in images:
        arr = img.detach().cpu()
        if arr.dim() == 4:
            arr = arr.squeeze(0)
        arr = arr.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).numpy()
        frames.append(arr)

    try:
        import imageio.v2 as imageio  # type: ignore

        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                    quality=8, macro_block_size=1)
        for f in frames:
            writer.append_data(f)
        writer.close()
        return
    except Exception as e:
        logger.debug(f"imageio failed ({e}); trying cv2.")

    import cv2  # type: ignore

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def _process_single_pano(args):
    """Process a single pano (used by ThreadPoolExecutor within each GPU worker)."""
    gen, pano_path, out_root, seed = args
    rng = random.Random(seed)
    return gen.process_pano(pano_path, out_root, rng)


class TrajectoryGenerator:
    """Panorama -> multi-segment trajectory dataset generator."""

    default_conf = {
        "name": "???",
        "base_dir": "???",
        "pano_dir": "${.base_dir}/${.name}",
        "pano_train": "${.pano_dir}",
        "out_dir": "${.base_dir}/Trajectory/${.name}",
        # trajectory options
        "num_panos": None,          # int N -> randomly pick N panos; None = all
        "segments_per_pano": 4,     # used only when segments_per_motion_type is null
        "segments_per_motion_type": None,  # int M -> M segments for EACH of pitch/roll/yaw
        "images_per_segment": 90,
        "angle_step_deg": 1.0,
        "vfov_min_deg": 60.0,
        "vfov_max_deg": 90.0,
        "image_height": 512,
        "image_width": 512,
        "camera_model": "pinhole",
        # video
        "save_video": False,
        "video_fps": 30,
        # runtime
        "seed": 0,
        "n_gpus": 1,
        "workers_per_gpu": 8,
        "overwrite": False,
    }

    def __init__(self, conf, device=None):
        self.conf = OmegaConf.merge(
            OmegaConf.create(self.default_conf),
            OmegaConf.create(conf),
        )
        logger.info(f"Config:\n{OmegaConf.to_yaml(self.conf)}")
        self.device = device or "cuda:0"
        self.camera_model = camera_models[self.conf.camera_model]

    def _sub_pattern_choice(self, motion_type: str, rng: random.Random) -> str:
        """Randomly pick a sub-pattern for a fixed motion type."""
        if motion_type == "yaw":
            return rng.choice(("bi_pos_neg", "bi_neg_pos", "full_circle"))
        return rng.choice(SUB_PATTERNS)

    def _segment_choice(self, rng: random.Random) -> Tuple[str, str]:
        motion_type = rng.choice(MOTION_TYPES)
        sub_pattern = self._sub_pattern_choice(motion_type, rng)
        return motion_type, sub_pattern

    def _motion_plan(self) -> List[str]:
        """Per-pano ordered list of motion types, one entry per segment.

        When ``segments_per_motion_type`` is set, generate that many segments
        for EACH of pitch/roll/yaw (equal counts). Otherwise fall back to
        ``segments_per_pano`` segments whose motion type is chosen at random.
        """
        per_motion = self.conf.get("segments_per_motion_type", None)
        if per_motion is not None and int(per_motion) > 0:
            return [mt for mt in MOTION_TYPES for _ in range(int(per_motion))]
        return [None] * int(self.conf.segments_per_pano)

    def _render_segment(
        self,
        pano: torch.Tensor,
        rolls_deg: np.ndarray,
        pitches_deg: np.ndarray,
        yaws_deg: np.ndarray,
        vfov_deg: float,
    ) -> List[torch.Tensor]:
        """Render one segment's frames. All frames share vfov and image size."""
        n = len(rolls_deg)
        h, w = int(self.conf.image_height), int(self.conf.image_width)
        vfov = float(deg2rad(torch.tensor(vfov_deg)))

        params = {
            "roll": [torch.tensor(float(deg2rad(torch.tensor(r)))) for r in rolls_deg],
            "pitch": [torch.tensor(float(deg2rad(torch.tensor(p)))) for p in pitches_deg],
            "vfov": [torch.tensor(vfov) for _ in range(n)],
            "height": [h] * n,
            "width": [w] * n,
        }
        cam = self.camera_model.from_dict(params).float().to(self.device)
        gravity = Gravity.from_rp(
            torch.tensor([float(deg2rad(torch.tensor(r))) for r in rolls_deg]),
            torch.tensor([float(deg2rad(torch.tensor(p))) for p in pitches_deg]),
        ).float().to(self.device)

        yaws_rad = torch.tensor(
            [float(deg2rad(torch.tensor(y))) for y in yaws_deg],
            dtype=torch.float32, device=self.device,
        )

        imgs = cam.get_img_from_pano(pano_img=pano, gravity=gravity, yaws=yaws_rad)
        return list(imgs)

    def _build_camera_json(
        self,
        rolls_deg: np.ndarray,
        pitches_deg: np.ndarray,
        yaws_deg: np.ndarray,
        vfov_deg: float,
        motion_type: str,
        sub_pattern: str,
    ) -> Dict:
        h, w = int(self.conf.image_height), int(self.conf.image_width)
        vfov_rad = float(deg2rad(torch.tensor(vfov_deg)))
        K = intrinsics_matrix(vfov_rad, h, w)
        focal_px = float(K[0, 0])  # fx = fy for a pinhole with square pixels

        frames = []
        for i in range(len(rolls_deg)):
            r = float(deg2rad(torch.tensor(float(rolls_deg[i]))))
            p = float(deg2rad(torch.tensor(float(pitches_deg[i]))))
            y = float(deg2rad(torch.tensor(float(yaws_deg[i]))))
            c2w = euler_to_c2w(r, p, y)
            frames.append({
                "file_path": f"{i + 1:06d}.jpg",
                "roll_deg": float(rolls_deg[i]),
                "pitch_deg": float(pitches_deg[i]),
                "yaw_deg": float(yaws_deg[i]),
                "vfov_deg": vfov_deg,
                "focal_px": focal_px,
                "camera_intrinsics": K.tolist(),
                "camera_pose": c2w.tolist(),  # 4x4 c2w, DL3DV-style
            })

        return {
            "motion_type": motion_type,
            "sub_pattern": sub_pattern,
            "image_height": h,
            "image_width": w,
            "vfov_deg": vfov_deg,
            "focal_px": focal_px,
            "angle_step_deg": float(self.conf.angle_step_deg),
            "images_per_segment": int(self.conf.images_per_segment),
            "frames": frames,
        }

    def process_pano(self, pano_path: Path, out_root: Path, rng: random.Random) -> int:
        """Generate M segments for a single panorama."""
        pano = load_image(pano_path).to(self.device)
        n_frames = int(self.conf.images_per_segment)
        step_deg = float(self.conf.angle_step_deg)

        motion_plan = self._motion_plan()

        n_done = 0
        for seg_idx, planned_motion in enumerate(motion_plan, start=1):
            seg_dir = out_root / f"{pano_path.stem}_{seg_idx:06d}"
            if seg_dir.exists() and not self.conf.overwrite:
                logger.info(f"Skip existing segment: {seg_dir}")
                n_done += 1
                continue
            seg_dir.mkdir(parents=True, exist_ok=True)

            if planned_motion is None:
                motion_type, sub_pattern = self._segment_choice(rng)
            else:
                motion_type = planned_motion
                sub_pattern = self._sub_pattern_choice(motion_type, rng)
            vfov_deg = rng.uniform(
                float(self.conf.vfov_min_deg), float(self.conf.vfov_max_deg),
            )
            rolls_deg, pitches_deg, yaws_deg = build_trajectory(
                motion_type, sub_pattern, n_frames, step_deg, rng,
            )

            try:
                imgs = self._render_segment(pano, rolls_deg, pitches_deg, yaws_deg, vfov_deg)
            except Exception as e:
                logger.warning(f"Render failed for {seg_dir.name}: {e}")
                continue

            for i, img in enumerate(imgs):
                write_image(img, seg_dir / f"{i + 1:06d}.jpg")

            cam_meta = self._build_camera_json(
                rolls_deg, pitches_deg, yaws_deg, vfov_deg, motion_type, sub_pattern,
            )
            with open(seg_dir / "cameras.json", "w") as f:
                json.dump(cam_meta, f, indent=2)

            if bool(self.conf.save_video):
                save_mp4(imgs, seg_dir / f"{seg_dir.name}.mp4",
                         int(self.conf.video_fps))

            n_done += 1

        return n_done

    def run(self) -> None:
        out_root = Path(self.conf.out_dir) / self.conf.name
        out_root.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.conf, out_root / "config.yaml")

        pano_paths = sorted([
            p for p in Path(self.conf.pano_train).glob("*")
            if p.suffix.lower() in VALID_EXTS
        ])
        logger.info(f"Found {len(pano_paths)} panoramas in {self.conf.pano_train}")

        rng_master = random.Random(int(self.conf.seed))

        # Optional sub-sampling: pick a deterministic random subset of N panos.
        # `num_panos` null / non-positive / >= len(pano_paths) -> use all.
        num_panos = self.conf.get("num_panos", None)
        if num_panos is not None and int(num_panos) > 0 and int(num_panos) < len(pano_paths):
            n_pick = int(num_panos)
            pano_paths = sorted(rng_master.sample(pano_paths, n_pick))
            logger.info(
                f"Randomly sampled {n_pick} panoramas (seed={int(self.conf.seed)})"
            )

        n_gpus = int(self.conf.n_gpus)
        workers_per_gpu = int(self.conf.workers_per_gpu)

        # Initialize CUDA in the main thread so worker threads can share the context
        for i in range(n_gpus):
            torch.zeros(1, device=f"cuda:{i}")

        # Create one TrajectoryGenerator per GPU and round-robin assign panos
        generators = [
            TrajectoryGenerator(self.conf, device=f"cuda:{i}")
            for i in range(n_gpus)
        ]

        task_args = [
            (generators[i % n_gpus], p, out_root, rng_master.randint(0, 2**31 - 1))
            for i, p in enumerate(pano_paths)
        ]

        total_segments = 0
        total_workers = workers_per_gpu * n_gpus
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            for n in tqdm(
                executor.map(_process_single_pano, task_args),
                total=len(pano_paths), desc="Panos", ncols=80,
            ):
                total_segments += n

        logger.info(f"Generated {total_segments} segments under {out_root}")


@hydra.main(version_base=None, config_path="configs", config_name="pano_trajectory")
def main(cfg: DictConfig) -> None:
    TrajectoryGenerator(cfg).run()


if __name__ == "__main__":
    main()
