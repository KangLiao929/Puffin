import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset, load_caption_cam_angles
from src.dust3r.utils.image import imread_cv2
from src.dust3r.oss_file_client import FileClient

from aoss_client.client import Client as AOSSClient


def _ensure_trailing_slash(s: str) -> str:
    """Ensure a path-like prefix ends with '/'."""
    return s if s.endswith("/") else (s + "/")


def get_aoss_file_iterator(root_aoss: str, cluster: str, conf_path=None):
    """
    Build an iterator over keys within an aoss:s3 bucket.
    iterator_spec: '{cluster}:s3://bucket/prefix/'
    """
    s3_prefix = root_aoss[len("aoss:"):] if root_aoss.startswith("aoss:") else root_aoss
    s3_prefix = _ensure_trailing_slash(s3_prefix)
    iterator_spec = f"{cluster}:{s3_prefix}"

    client = AOSSClient(conf_path)
    return client.get_file_iterator(iterator_spec)


class ARKitScenesHighRes_Multi(BaseMultiViewDataset):
    def __init__(
        self,
        *args,
        split,
        ROOT,
        cache_path=None,
        cluster_name="aoss",
        conf_path="~/aoss.conf",
        camera_caption_root=None,
        **kwargs,
    ):
        # AOSS & basic fields
        self.ROOT = _ensure_trailing_slash(ROOT)
        self.cluster_name = cluster_name
        self.conf_path = conf_path
        # Local root for VLM camera captions (scripts/annotation/camera_caption_aoss.py):
        #   <camera_caption_root>/<Training|Validation>/<scene>/camera/<basename>.json
        # When set, each view gets `gt_cam_angles` = [roll, pitch, vfov, k1]
        # (radians) for model.load_gt_camera_params=True training.
        self.camera_caption_root = camera_caption_root
        self.video = True
        self.max_interval = 8
        self.is_metric = True

        # File client for actual downloads
        self.file_client = FileClient()
        self._s3_prefix = (
            self.ROOT[len("aoss:"):] if self.ROOT.startswith("aoss:") else self.ROOT
        )
        self._s3_prefix = _ensure_trailing_slash(self._s3_prefix)

        super().__init__(*args, **kwargs)

        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Validation"
        else:
            raise ValueError(f"Unknown split: {split}")

        # Build / load index (with optional cache)
        self._load_data(self.split, cache_path=cache_path)

    def _load_data(self, split, cache_path=None):
        """
        Load dataset index from cache if available; otherwise build from AOSS
        and (optionally) save cache to `cache_path`.

        The index includes:
            scenes, sceneids, images, scene_img_list,
            start_img_ids, timestamps, intrinsics, trajectories
        """
        # 1) Load from cache if available
        if cache_path is not None and osp.exists(cache_path):
            print(f"[ARKitScenesHighRes_Multi] Loading cached index from {cache_path}")
            import pickle

            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.scenes = cache["scenes"]
            self.sceneids = cache["sceneids"]
            self.images = cache["images"]
            self.scene_img_list = cache["scene_img_list"]
            self.start_img_ids = cache["start_img_ids"]
            self.timestamps = cache["timestamps"]
            self.intrinsics = cache["intrinsics"]
            self.trajectories = cache["trajectories"]
            assert len(self.images) == len(self.intrinsics) == len(self.trajectories)
            return

        # 2) Otherwise, stream bucket keys from AOSS to collect scenes for this split
        files_iter = get_aoss_file_iterator(
            self.ROOT, self.cluster_name, conf_path=self.conf_path
        )

        # 's3://bucket/prefix/' -> 'bucket/prefix/' for easier matching
        bucket_prefix_clean = self._s3_prefix.replace("s3://", "")
        bucket_prefix_clean = _ensure_trailing_slash(bucket_prefix_clean)

        scenes_set = set()

        print(
            f"[ARKitScenesHighRes_Multi] Streaming bucket keys from AOSS to collect scenes for split={split}..."
        )
        for p, _k in tqdm(files_iter, desc="Scanning keys for scenes"):
            # Accept both with and without trailing slash variants
            if not (p.startswith(bucket_prefix_clean) or p.startswith(bucket_prefix_clean[:-1])):
                continue

            if p.startswith(bucket_prefix_clean):
                rel = p[len(bucket_prefix_clean):]
            else:
                rel = p[len(bucket_prefix_clean[:-1]):]
                if rel.startswith("/"):
                    rel = rel[1:]

            if not rel:
                continue

            parts = rel.split("/")
            # Expected: split/scene/scene_metadata.npz
            # e.g. Training/48458251/scene_metadata.npz
            if len(parts) < 3:
                continue
            if parts[0] != split:
                continue
            if parts[2] != "scene_metadata.npz":
                # Filter out things like Training/scene_list.json
                continue

            scene = parts[1]
            scenes_set.add(scene)

        all_scenes = sorted(list(scenes_set))

        # 3) For each scene, load scene_metadata.npz and build index
        file_client = self.file_client

        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        timestamps = []
        intrinsics = []
        trajectories = []

        scene_id = 0
        for scene in tqdm(all_scenes, desc="Indexing scenes"):
            scene_dir = osp.join(self.ROOT, split, scene)
            metadata_oss_path = osp.join(scene_dir, "scene_metadata.npz")

            try:
                # Put download_file in try to guard against None returns
                metadata_path = file_client.download_file(metadata_oss_path)

                with np.load(metadata_path) as data:
                    imgs_with_indices = sorted(
                        enumerate(data["images"]), key=lambda x: x[1]
                    )
                    imgs = [x[1] for x in imgs_with_indices]
                    cut_off = (
                        self.num_views
                        if not self.allow_repeat
                        else max(self.num_views // 3, 3)
                    )
                    if len(imgs) < cut_off:
                        print(
                            f"[ARKitScenesHighRes_Multi] Skipping {scene}, "
                            f"len(imgs)={len(imgs)} < cut_off={cut_off}"
                        )
                        continue

                    indices = [x[0] for x in imgs_with_indices]
                    tsps = np.array(
                        [float(img_name.split("_")[1][:-4]) for img_name in imgs]
                    )
                    assert all(
                        img[:8] == scene for img in imgs
                    ), f"{scene}, {imgs}"

                    num_imgs = data["images"].shape[0]
                    img_ids = list(np.arange(num_imgs) + offset)
                    start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                    scenes.append(scene)
                    scene_img_list.append(img_ids)
                    sceneids.extend([scene_id] * num_imgs)
                    images.extend(imgs)
                    start_img_ids.extend(start_img_ids_)
                    timestamps.extend(tsps)

                    # Build K from intrinsics
                    K = np.expand_dims(np.eye(3), 0).repeat(num_imgs, 0)
                    intrins = data["intrinsics"][indices]
                    K[:, 0, 0] = [fx for _, _, fx, _, _, _ in intrins]
                    K[:, 1, 1] = [fy for _, _, _, fy, _, _ in intrins]
                    K[:, 0, 2] = [cx for _, _, _, _, cx, _ in intrins]
                    K[:, 1, 2] = [cy for _, _, _, _, _, cy in intrins]
                    intrinsics.extend(list(K))
                    trajectories.extend(list(data["trajectories"][indices]))

            except Exception as e:
                print(
                    f"[ARKitScenesHighRes_Multi] Failed processing scene {scene}: {e}, skipping."
                )
                continue

            # offset groups
            offset += num_imgs
            scene_id += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.intrinsics = intrinsics
        self.trajectories = trajectories
        self.start_img_ids = start_img_ids
        self.timestamps = timestamps
        assert len(self.images) == len(self.intrinsics) == len(self.trajectories)

        # 4) Save cache if requested
        if cache_path is not None:
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            print(f"[ARKitScenesHighRes_Multi] Saving index cache to {cache_path}")
            import pickle

            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "scenes": self.scenes,
                        "sceneids": self.sceneids,
                        "images": self.images,
                        "scene_img_list": self.scene_img_list,
                        "start_img_ids": self.start_img_ids,
                        "timestamps": self.timestamps,
                        "intrinsics": self.intrinsics,
                        "trajectories": self.trajectories,
                    },
                    f,
                )

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        file_client = FileClient(auto_cleanup=False)
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]
            assert (
                basename[:8] == self.scenes[scene_id]
            ), f"{basename}, {self.scenes[scene_id]}"

            rgb_path = osp.join(
                scene_dir, "vga_wide", basename.replace(".png", ".jpg")
            )
            temp_rgb = file_client.download_file(rgb_path)
            rgb_image = imread_cv2(temp_rgb, cv2.IMREAD_COLOR)

            depth_path = osp.join(scene_dir, "highres_depth", basename)
            temp_depth = file_client.download_file(depth_path)
            depthmap = imread_cv2(temp_depth, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.7, 0.25, 0.05]
            )

            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="arkitscenes_highres",
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

            # Per-view GT camera angles from the VLM caption JSONs:
            #   <camera_caption_root>/<Training|Validation>/<scene>/camera/<basename>.json
            # NOTE: the caption vfov refers to the original image; at PF-build
            # time the model uses the crop-consistent focal from
            # `cam_intrinsics` instead (see _apply_gt_camera_params).
            if self.camera_caption_root is not None:
                cap_path = osp.join(
                    self.camera_caption_root, self.split,
                    self.scenes[scene_id], "camera", basename[:-4] + ".json")
                gt_cam_angles = load_caption_cam_angles(cap_path)
                if gt_cam_angles is not None:
                    view_dict["gt_cam_angles"] = gt_cam_angles

            views.append(view_dict)
        assert len(views) == num_views
        return views