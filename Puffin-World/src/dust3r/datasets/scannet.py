import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys

import pickle
from safetensors.numpy import load_file as load_safetensors
sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset, load_caption_cam_angles
from src.dust3r.utils.image import imread_cv2
from src.dust3r.utils.image import rgb
from src.dust3r.oss_file_client import FileClient
from src.dust3r.viz import colorize_np

import torch

class ScanNet_Multi(BaseMultiViewDataset):
    # ROOT/scans_(test)/scene0000_00/cam or color or depth
    def __init__(self, *args, ROOT, cache_path=None, min_interval=1, max_interval=20, depth_da3_root=None,
                 camera_caption_root=None, scene_subdir=None, test_sceneids_path=None, **kwargs):
        self.test_sceneids_path = test_sceneids_path
        self.scene_subdir = scene_subdir
        self.ROOT = ROOT
        # Local root for the depth-anything-3 depth maps, layout:
        #   <depth_da3_root>/scans(_test)/<scene>/depth_da3/<basename>.npy.
        # No hardcoded default -- set it in the gen config (gen_scannet.py); every
        # sample loads depth from it, so fail fast when missing.
        assert depth_da3_root is not None, (
            "ScanNet_Multi requires depth_da3_root (set it in the gen config, "
            "e.g. configs/datasets/multi_view/gen_scannet.py)."
        )
        self.depth_da3_root = depth_da3_root
        # Local root for absolute camera captions:
        #   <camera_caption_root>/scans[_test]/<scene>/camera/<basename>.json
        self.camera_caption_root = camera_caption_root
        self.video = True
        self.is_metric = True

        self.min_interval = min_interval
        self.max_interval = max_interval
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data(self.split, cache_path=cache_path)

    def _load_data(self, split, cache_path=None):
        # These two roots only depend on (ROOT, depth_da3_root, split); always
        # set them regardless of cache so _get_views works after a cached load.
        split_subdir = self.scene_subdir or ("scans" if split == "train" else "scans_test")
        self.scene_root = osp.join(self.ROOT, split_subdir)
        self.depth_da3_scene_root = osp.join(self.depth_da3_root, split_subdir)

        if cache_path is not None and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("schema_version") == "full_list":
                self.scenes = cache["scenes"]
                self.sceneids = cache["sceneids"]
                self.images = cache["images"]
                self.scene_img_list = cache["scene_img_list"]
                print(
                    f"[ScanNet_Multi] Loaded {len(self.scenes)} scenes, "
                    f"{len(self.images)} images from cache (full_list)."
                )
                self._apply_split_filter()
                self._compute_start_img_ids(tuple_with_scene=False)  # num_views-dependent
                return
            print(f"[ScanNet_Multi] Cache at {cache_path} is legacy (num_views-baked); rebuilding as full index.")

        file_client = FileClient()
        self.scenes = [
            scene for scene in file_client.list_dir(self.scene_root, return_only_dir=True) if scene.startswith("scene")
        ]

        # FULL raw index: keep EVERY scene (>=2 frames) and ALL its frames (no
        # num_views cut_off). The num_views-dependent start positions are derived
        # later by _compute_start_img_ids(), so this cache is num_views-agnostic.
        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []

        j = 0
        for scene_idx, scene in tqdm(enumerate(self.scenes)):
            scene_dir = osp.join(self.scene_root, scene)
            try:
                temp_meta = file_client.download_file(
                    osp.join(scene_dir, "new_scene_metadata.pkl"),
                    force_redownload=True
                )
                with open(temp_meta, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[ScanNet_Multi] Skipping {scene}: failed to load metadata ({e})")
                continue
            basenames = data["images"]
            num_imgs = len(basenames)
            if num_imgs < 2:  # 1-frame scene can never form a multi-view sample
                print(f"Skipping {scene}")
                continue

            img_ids = list(np.arange(num_imgs) + offset)

            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            # offset groups
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

        self._apply_split_filter()
        self._compute_start_img_ids(tuple_with_scene=False)  # num_views-dependent

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)


    def _load_test_scenes(self):
        if self.split not in ("train", "test"):
            return None
        if not self.test_sceneids_path or not osp.exists(self.test_sceneids_path):
            return set()
        with open(self.test_sceneids_path, "rb") as f:
            test_scenes = pickle.load(f)
        return set(test_scenes)

    def _apply_split_filter(self):
        test_scenes = self._load_test_scenes()
        if test_scenes is None:
            return
        if not test_scenes:
            print(f"[ScanNet_Multi] No test scene split found at {self.test_sceneids_path}; using all scenes for split={self.split}.")
            return

        if self.split == "test":
            keep_old_scene_ids = [i for i, scene in enumerate(self.scenes) if scene in test_scenes]
        else:
            keep_old_scene_ids = [i for i, scene in enumerate(self.scenes) if scene not in test_scenes]

        scenes, sceneids, images, scene_img_list = [], [], [], []
        for new_scene_id, old_scene_id in enumerate(keep_old_scene_ids):
            new_img_ids = []
            for old_img_id in self.scene_img_list[old_scene_id]:
                old_img_id_int = int(old_img_id)
                new_img_ids.append(len(images))
                images.append(self.images[old_img_id_int])
                sceneids.append(new_scene_id)
            scenes.append(self.scenes[old_scene_id])
            scene_img_list.append(new_img_ids)

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        print(f"[ScanNet_Multi] Applied split={self.split}: {len(self.scenes)} scenes, {len(self.images)} images.")

    def _get_views(self, idx, resolution, rng, num_views):
        self.file_client = FileClient(auto_cleanup=False)
        file_client = self.file_client
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            video_prob=0.6,
            fix_interval_prob=0.6,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]
        # import ipdb; ipdb.set_trace()
        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_name = self.scenes[scene_id]
            scene_dir = osp.join(self.scene_root, scene_name)
            rgb_dir = osp.join(scene_dir, "color")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Depth now comes from the local depth-anything-3 dump (not AOSS):
            #   <depth_da3_scene_root>/<scene>/depth_da3/<basename>.npy
            depth_da3_path = osp.join(
                self.depth_da3_scene_root, scene_name, "depth_da3", basename + ".npy"
            )
            temp_rgb = file_client.download_file(
                osp.join(rgb_dir, basename + ".jpg"),
                force_redownload=True
            )
            temp_cam = file_client.download_file(
                osp.join(cam_dir, basename + ".safetensor"),
                force_redownload=True
            )
            
            rgb_image = imread_cv2(temp_rgb, cv2.IMREAD_COLOR)

            # Depth read directly from the local depth-anything-3 .npy (already float metric).
            depthmap = np.load(depth_da3_path).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            
            cam = load_safetensors(temp_cam)
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
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
                dataset="ScanNet",
                label=self.scenes[scene_id] + "_" + basename,
                instance=f"{str(idx)}_{str(view_idx)}",
                is_metric=self.is_metric,
                is_video=ordered_video,
                quantile=np.array(0.98, dtype=np.float32),
                img_mask=img_mask,
                ray_mask=ray_mask,
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            )

            # VLM camera caption -> per-view gt angles (radians).
            # Mirrors the depth_da3 layout: <root>/scans[_test]/<scene>/camera/<b>.json
            if self.camera_caption_root is not None:
                split_subdir = osp.basename(self.scene_root.rstrip("/"))
                cap_path = osp.join(
                    self.camera_caption_root, split_subdir, scene_name,
                    "camera", basename + ".json")
                # (vfov is original-image; PF focal comes from cam_intrinsics
                # at model side — see _apply_gt_camera_params.)
                gt_cam_angles = load_caption_cam_angles(cap_path)
                if gt_cam_angles is not None:
                    view_dict["gt_cam_angles"] = gt_cam_angles

            views.append(view_dict)
        assert len(views) == num_views
        return views
