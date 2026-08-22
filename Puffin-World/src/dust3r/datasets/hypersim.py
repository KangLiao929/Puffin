import os.path as osp
import os
import sys
import itertools
import pickle

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset, load_caption_cam_angles
from src.dust3r.utils.image import imread_cv2, rgb
from src.dust3r.oss_file_client import FileClient
from src.dust3r.viz import colorize_np


class HyperSim_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, cache_path=None,
                 min_interval=1, max_interval=1,
                 camera_caption_root=None, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.file_client = FileClient()
        # Local root for absolute camera captions:
        #   <camera_caption_root>/<scene>/<fname 'rgb.png' -> 'camera.json'>
        self.camera_caption_root = camera_caption_root
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data(cache_path=cache_path)

    def _load_data(self, cache_path=None):
        if cache_path is not None and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("schema_version") == "full_list":
                self.scenes = cache["scenes"]
                self.sceneids = cache["sceneids"]
                self.images = cache["images"]
                self.scene_img_list = cache["scene_img_list"]
                print(
                    f"[HyperSim_Multi] Loaded {len(self.scenes)} scenes, "
                    f"{len(self.images)} images from cache (full_list)."
                )
                self._compute_start_img_ids(tuple_with_scene=False)
                return
            print(
                f"[HyperSim_Multi] Cache at {cache_path} is legacy "
                f"(num_views-baked); rebuilding as full index."
            )

        self.all_scenes = sorted(
            [
                f
                for f in self.file_client.list_dir(self.ROOT, return_only_dir=True)
                if f
            ]
        )
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            scene_root = osp.join(self.ROOT, scene)
            sub_dirs = self.file_client.list_dir(scene_root, return_only_dir=True)
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in sub_dirs
                    if f and len(self.file_client.list_dir(osp.join(scene_root, f))) > 0
                ]
            )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        j = 0
        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_paths = sorted(
                [f for f in self.file_client.list_dir(scene_dir) if f.endswith(".png")]
            )
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            num_imgs = len(rgb_paths)
            if num_imgs < 2:
                print(f"Skipping {scene}")
                continue
            img_ids = list(np.arange(num_imgs) + offset)

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list

        if cache_path is not None:
            cache_dir = osp.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            print(f"Saving full index cache to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "schema_version": "full_list",
                        "scenes": self.scenes,
                        "sceneids": self.sceneids,
                        "images": self.images,
                        "scene_img_list": self.scene_img_list,
                    },
                    f,
                )

        self._compute_start_img_ids(tuple_with_scene=False)

    def __len__(self):
        return len(self.start_img_ids) * 10

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        idx = idx // 10
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]
        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            temp_rgb = self.file_client.download_file(osp.join(scene_dir, rgb_path))
            temp_depth = self.file_client.download_file(osp.join(scene_dir, depth_path))
            temp_cam = self.file_client.download_file(osp.join(scene_dir, cam_path))

            rgb_image = imread_cv2(temp_rgb, cv2.IMREAD_COLOR)
            depthmap = np.load(temp_depth).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(temp_cam)
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="hypersim",
                label=self.scenes[scene_id] + "_" + rgb_path,
                instance=f"{str(idx)}_{str(view_idx)}",
                is_metric=self.is_metric,
                is_video=ordered_video,
                quantile=np.array(1.0, dtype=np.float32),
                img_mask=img_mask,
                ray_mask=ray_mask,
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            )

            # VLM camera caption -> per-view gt angles (radians).
            # (vfov is original-image; PF focal comes from cam_intrinsics
            # at model side — see _apply_gt_camera_params.)
            if self.camera_caption_root is not None:
                cap_path = osp.join(
                    self.camera_caption_root, self.scenes[scene_id],
                    rgb_path.replace("rgb.png", "camera.json"))
                gt_cam_angles = load_caption_cam_angles(cap_path)
                if gt_cam_angles is not None:
                    view_dict["gt_cam_angles"] = gt_cam_angles

            views.append(view_dict)
        assert len(views) == num_views
        return views