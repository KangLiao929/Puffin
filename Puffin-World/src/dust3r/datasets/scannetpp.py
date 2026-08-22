import os.path as osp
import cv2
import numpy as np
import os
import sys
import zipfile
import torch.distributed as dist

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from src.dust3r.oss_file_client import FileClient
from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from src.dust3r.utils.image import imread_cv2
from tqdm import tqdm
from torch.utils.data import DataLoader


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _get_rank():
    return dist.get_rank() if _is_distributed() else 0


def _is_master():
    return (not _is_distributed()) or _get_rank() == 0


def _broadcast_object(obj, src=0):
    """Broadcast a Python object from src to all ranks."""
    if not _is_distributed():
        return obj
    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def _validate_npz(path, tag="npz"):
    """Strong validation to avoid BadZipFile/partial downloads."""
    if (not osp.exists(path)) or osp.getsize(path) == 0:
        raise RuntimeError(f"Corrupted {tag}: missing/empty file: {path}")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Corrupted {tag}: not a valid zip/npz: {path}")


class ScanNetpp_Multi(BaseMultiViewDataset):
    # ROOT/scene1/depth or images
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 3
        super().__init__(*args, **kwargs)
        assert self.split == "train"
        self.loaded_data = self._load_data()

    def _load_data(self):
        file_client = FileClient()
        rank = _get_rank()

        # ---------- Step 1: Download/load global metadata (rank0 does refresh) ----------
        global_meta_key = osp.join(self.ROOT, "all_metadata.npz")

        if _is_master():
            # rank0 refreshes to ensure cache is consistent
            file_client.download_file(global_meta_key, force_redownload=True)

        if _is_distributed():
            dist.barrier()  # ✅ only once globally

        # everyone reads from cache
        temp_meta = file_client.download_file(global_meta_key, force_redownload=False)
        _validate_npz(temp_meta, tag="global_meta")

        with np.load(temp_meta, allow_pickle=True) as data:
            all_scenes = list(data["scenes"])

        # ---------- Step 2: rank0 decides valid scenes then broadcast ----------
        if _is_master():
            valid_scenes = []
            bad_scenes = 0

            for scene in tqdm(all_scenes, desc="[ScanNet++] validating scenes (rank0)"):
                scene_dir = osp.join(self.ROOT, scene)
                scene_meta_key = osp.join(scene_dir, "new_scene_metadata.npz")

                try:
                    tmp = file_client.download_file(scene_meta_key, force_redownload=False)
                    _validate_npz(tmp, tag=f"scene_meta:{scene}")

                    with np.load(tmp, allow_pickle=True) as d:
                        # minimal required keys
                        if ("images" not in d) or ("intrinsics" not in d) or ("trajectories" not in d) or ("image_collection" not in d):
                            bad_scenes += 1
                            continue
                        # ensure image_collection can be unpacked
                        _ = d["image_collection"].item()

                    valid_scenes.append(scene)
                except Exception:
                    bad_scenes += 1
                    continue

            print(f"[Rank {rank}] ScanNet++ scenes total={len(all_scenes)}, valid={len(valid_scenes)}, bad={bad_scenes}")
        else:
            valid_scenes = None

        valid_scenes = _broadcast_object(valid_scenes, src=0)

        if valid_scenes is None or len(valid_scenes) == 0:
            raise RuntimeError(f"[Rank {rank}] No valid ScanNet++ scenes found.")

        # ---------- Step 3: all ranks build index from SAME valid_scenes ----------
        offset = 0
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        groups = []
        id_ranges = []
        j = 0
        self.image_num = 0

        for scene in valid_scenes:
            scene_dir = osp.join(self.ROOT, scene)
            scene_meta_key = osp.join(scene_dir, "new_scene_metadata.npz")

            temp_scene_meta = file_client.download_file(scene_meta_key, force_redownload=False)
            _validate_npz(temp_scene_meta, tag=f"scene_meta:{scene}")

            with np.load(temp_scene_meta, allow_pickle=True) as data:
                imgs = data["images"]
                intrins = data["intrinsics"]
                traj = data["trajectories"]
                image_collection = data["image_collection"].item()

            self.image_num += len(imgs)
            img_ids = np.arange(len(imgs)).tolist()

            # list images on disk
            imgs_on_disk = sorted(file_client.list_dir(osp.join(scene_dir, "images")))
            imgs_on_disk = [x[:-4] for x in imgs_on_disk]

            dslr_ids = [
                i + offset
                for i in img_ids
                if imgs[i].startswith("DSC") and imgs[i] in imgs_on_disk
            ]
            iphone_ids = [
                i + offset
                for i in img_ids
                if imgs[i].startswith("frame") and imgs[i] in imgs_on_disk
            ]

            img_groups = []
            img_id_ranges = []

            # build groups
            for ref_id, group in image_collection.items():
                if len(group) + 1 < self.num_views:
                    continue
                group = list(group)  # ensure mutable
                group.insert(0, (ref_id, 1.0))
                sorted_group = sorted(group, key=lambda x: x[1], reverse=True)
                g = [int(x[0] + offset) for x in sorted_group]
                img_groups.append(sorted(g))

                # NOTE: keep your original branch behavior (even if naming looks reversed)
                if imgs[ref_id].startswith("frame"):
                    img_id_ranges.append(dslr_ids)
                else:
                    img_id_ranges.append(iphone_ids)

            # IMPORTANT: DO NOT "continue skip" here across ranks in a way that changes offsets differently.
            # We will still register the scene's images/intrinsics/trajectories and advance offset/j
            scenes.append(scene)
            sceneids.extend([j] * len(imgs))
            images.extend(imgs)
            intrinsics.append(intrins)
            trajectories.append(traj)

            if len(img_groups) > 0:
                groups.extend(img_groups)
                id_ranges.extend(img_id_ranges)

            offset += len(imgs)
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images

        if len(intrinsics) > 0:
            self.intrinsics = np.concatenate(intrinsics, axis=0)
            self.trajectories = np.concatenate(trajectories, axis=0)
        else:
            self.intrinsics = np.array([])
            self.trajectories = np.array([])

        self.id_ranges = id_ranges
        self.groups = groups

    def __len__(self):
        return len(self.groups) * 10

    def get_image_num(self):
        return self.image_num

    def _get_views(self, idx, resolution, rng, num_views):
        self.file_client = FileClient(auto_cleanup=False)
        file_client = self.file_client

        idx = idx // 10
        image_idxs = self.groups[idx]
        rand_val = rng.random()

        image_idxs_video = self.id_ranges[idx]
        cut_off = num_views if not self.allow_repeat else max(num_views // 3, 3)
        start_image_idxs = image_idxs_video[: len(image_idxs_video) - cut_off + 1]

        if rand_val < 0.7 and len(start_image_idxs) > 0:
            start_id = rng.choice(start_image_idxs)
            pos, ordered_video = self.get_seq_from_start_id(
                num_views,
                start_id,
                image_idxs_video,
                rng,
                max_interval=self.max_interval,
                video_prob=0.8,
                fix_interval_prob=0.5,
                block_shuffle=16,
            )
            image_idxs = np.array(image_idxs_video)[pos]
        else:
            ordered_video = True
            num_candidates = len(image_idxs)
            max_id = min(num_candidates, int(num_views * (2 + 2 * rng.random())))
            image_idxs = sorted(rng.permutation(image_idxs[:max_id])[:num_views])
            if rand_val > 0.75:
                ordered_video = False
                image_idxs = rng.permutation(image_idxs)

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            temp_rgb = file_client.download_file(
                osp.join(scene_dir, "images", basename + ".jpg"),
                force_redownload=False,
            )
            temp_depth = file_client.download_file(
                osp.join(scene_dir, "depth", basename + ".png"),
                force_redownload=False,
            )

            rgb_image = imread_cv2(temp_rgb, cv2.IMREAD_COLOR)
            depthmap = imread_cv2(temp_depth, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="ScanNet++",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.99, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )

        assert len(views) == num_views
        return views
