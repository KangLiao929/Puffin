import math
import os
import os.path as osp
import json
import pickle
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from src.dust3r.utils.image import imread_cv2

cv2.setNumThreads(0)


def euler_to_textbook_c2w(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a 4x4 c2w with standard right-hand textbook rotations so puffin
    aligns with DL3DV / RealEstate10K (which both store c2w from COLMAP under
    the same convention).

    R_c2w = Rz(roll) @ Rx(pitch) @ Ry(yaw)
    where each R is a right-hand active rotation around the corresponding axis.

    The stored "camera_pose" in cameras.json is rad2rotmat output (Rx/Ry/Rz
    with sign-flipped off-diagonals), which doesn't match the textbook
    composition order/sign and therefore wouldn't align with DL3DV under joint
    training. Recomputing from raw angles bypasses that mismatch.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Rz @ Rx @ Ry
    return c2w


class PuffinOmni_Multi(BaseMultiViewDataset):
    """Multi-view dataset for Puffin-Omni trajectory segments (local filesystem).

    Expected layout under ROOT:
        ROOT/
            <dataset_A>/
                <trajectory_001>/
                    000001.jpg  ...  cameras.json
                <trajectory_002>/
                    ...
            <dataset_B>/
                ...
    """

    def __init__(self, *args,
                 ROOT,
                 cache_path=None,
                 min_interval=1,
                 max_interval=20,
                 **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.min_interval = min_interval
        self.max_interval = max_interval

        super().__init__(*args, **kwargs)
        self._load_data(cache_path=cache_path)

    def _load_data(self, cache_path=None, index_workers=16):
        if cache_path is not None and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("schema_version") == "full_list":
                self.scenes = cache["scenes"]
                self.sceneids = cache["sceneids"]
                self.images = cache["images"]
                self.scene_img_list = cache["scene_img_list"]
                self.scene_dirs = cache["scene_dirs"]
                self.scene_to_idx = {s: i for i, s in enumerate(self.scenes)}
                self.invalid_scenes = {scene: False for scene in self.scenes}
                print(f"[PuffinOmni_Multi] Loaded {len(self.scenes)} trajectories, "
                      f"{len(self.images)} images from cache (full_list): {cache_path}")
                self._compute_start_img_ids(tuple_with_scene=True)  # num_views-dependent, recomputed every run
                return
            print(f"[PuffinOmni_Multi] Cache at {cache_path} is legacy (num_views-baked); rebuilding as full index.")

        dataset_dirs = sorted([
            d for d in os.listdir(self.ROOT)
            if osp.isdir(osp.join(self.ROOT, d))
        ])

        # Collect all (ds_name, traj_name, traj_path) pairs first
        all_trajs = []
        print(f"[PuffinOmni_Multi] Listing trajectory dirs under {len(dataset_dirs)} datasets ...")
        for ds_name in tqdm(dataset_dirs, desc="Listing datasets"):
            ds_path = osp.join(self.ROOT, ds_name)
            traj_dirs = sorted([
                t for t in os.listdir(ds_path)
                if osp.isdir(osp.join(ds_path, t))
                   and osp.isfile(osp.join(ds_path, t, "cameras.json"))
            ])
            for traj_name in traj_dirs:
                all_trajs.append((ds_name, traj_name, osp.join(ds_path, traj_name)))

        print(f"[PuffinOmni_Multi] Reading cameras.json from {len(all_trajs)} trajectories "
              f"with {index_workers} threads ...")

        def _read_one(args):
            ds_name, traj_name, traj_path = args
            cam_path = osp.join(traj_path, "cameras.json")
            try:
                with open(cam_path, "r") as f:
                    cam_meta = json.load(f)
                frame_names = sorted([fr["file_path"] for fr in cam_meta["frames"]])
                return (ds_name, traj_name, traj_path, frame_names)
            except Exception as e:
                return None

        with ThreadPoolExecutor(max_workers=index_workers) as pool:
            results = list(tqdm(
                pool.map(_read_one, all_trajs),
                total=len(all_trajs),
                desc="Reading cameras.json",
            ))

        # Assemble FULL raw index from results: keep EVERY trajectory and ALL
        # its frames (no num_views cut_off, no dropping short scenes beyond the
        # 1-frame case). The num_views-dependent start positions are derived
        # later by _compute_start_img_ids(), so this cache is num_views-agnostic.
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        scene_dirs_list = []
        offset = 0
        j = 0

        for res in results:
            if res is None:
                continue
            ds_name, traj_name, traj_path, frame_names = res
            num_imgs = len(frame_names)
            if num_imgs < 2:                     # 1-frame scene can never form a multi-view sample
                continue

            scene_key = osp.join(ds_name, traj_name)
            img_ids = list(np.arange(num_imgs) + offset)

            sceneids.extend([j] * num_imgs)
            images.extend(frame_names)
            scenes.append(scene_key)
            scene_dirs_list.append(traj_path)
            scene_img_list.append(img_ids)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.scene_dirs = scene_dirs_list
        self.scene_to_idx = {s: i for i, s in enumerate(scenes)}
        self.invalid_scenes = {scene: False for scene in self.scenes}

        print(f"[PuffinOmni_Multi] Indexed {len(self.scenes)} trajectories, "
              f"{len(self.images)} images from {len(dataset_dirs)} datasets.")

        if cache_path is not None:
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            print(f"[PuffinOmni_Multi] Saving full index cache to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "schema_version": "full_list",
                    "scenes": self.scenes,
                    "sceneids": self.sceneids,
                    "images": self.images,
                    "scene_img_list": self.scene_img_list,
                    "scene_dirs": self.scene_dirs,
                }, f)

        self._compute_start_img_ids(tuple_with_scene=True)  # num_views-dependent, derived from the full index

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene, start_id = self.start_img_ids[idx]

        while invalid_seq:
            while self.invalid_scenes.get(scene, False):
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]

            scene_idx = self.scene_to_idx[scene]
            all_image_ids = self.scene_img_list[scene_idx]
            pos, ordered_video = self.get_seq_from_start_id(
                num_views, start_id, all_image_ids, rng,
                min_interval=self.min_interval, max_interval=self.max_interval,
                video_prob=1, fix_interval_prob=1.0
            )
            image_idxs = np.array(all_image_ids)[pos]

            seg_dir = self.scene_dirs[scene_idx]
            cam_path = osp.join(seg_dir, "cameras.json")
            try:
                with open(cam_path, "r") as f:
                    cam_meta = json.load(f)
            except Exception as e:
                print(f"[PuffinOmni_Multi] Error reading {cam_path}: {e}")
                self.invalid_scenes[scene] = True
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]
                continue

            frames_by_name = {fr["file_path"]: fr for fr in cam_meta["frames"]}

            views = []
            for view_idx in image_idxs:
                basename = self.images[view_idx]

                try:
                    img_path = osp.join(seg_dir, basename)
                    rgb_image = imread_cv2(img_path, cv2.IMREAD_COLOR)

                    depthmap = np.ones_like(rgb_image[..., 0], dtype=np.float32)

                    fr = frames_by_name[basename]
                    intrinsics = np.array(fr["camera_intrinsics"], dtype=np.float64)
                    # The matrix stored as "camera_pose" in cameras.json is
                    # rad2rotmat(r, p, y), which is the INVERSE (= transpose
                    # for rotations) of the actual c2w under the puffin
                    # rendering convention.
                    pose_raw = np.array(fr["camera_pose"], dtype=np.float64)
                    camera_pose = np.eye(4, dtype=np.float64)
                    camera_pose[:3, :3] = pose_raw[:3, :3].T
                    camera_pose[:3, 3] = pose_raw[:3, 3]

                    # Ground-truth camera angles for the perspective field.
                    # cameras.json stores per-frame roll/pitch/yaw/vfov in
                    # degrees.
                    gt_cam_angles = np.array(
                        [
                            math.radians(float(fr.get("roll_deg", 0.0))),
                            math.radians(float(fr.get("pitch_deg", 0.0))),
                            math.radians(float(fr.get("vfov_deg", 90.0))),
                            0.0,
                        ],
                        dtype=np.float32,
                    )
                except Exception as e:
                    print(f"[PuffinOmni_Multi] Error loading {scene}/{basename}: {e}")
                    self.invalid_scenes[scene] = True
                    break

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    gt_cam_angles=gt_cam_angles,
                    dataset="puffin_omni",
                    label=scene + "_" + basename,
                    instance=f"{idx}_{view_idx}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=True,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                ))

            if len(views) == num_views:
                invalid_seq = False

        return views