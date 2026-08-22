import os.path as osp
import cv2
import numpy as np
import os
import pickle

from tqdm import tqdm
from safetensors.numpy import load_file

from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset, load_caption_cam_angles
from src.dust3r.utils.image import imread_cv2, rgb
from src.dust3r.oss_file_client import FileClient
from src.dust3r.viz import colorize_np


class MVS_Synth_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, cache_path=None,
                 min_interval=1, max_interval=2,
                 camera_caption_root=None, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.min_interval = min_interval
        self.max_interval = max_interval
        # Local root for absolute camera captions:
        #   <camera_caption_root>/<scene>/camera/<basename>.json
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
                    f"[MVS_Synth_Multi] Loaded {len(self.scenes)} scenes, "
                    f"{len(self.images)} images from cache (full_list)."
                )
                self._compute_start_img_ids(tuple_with_scene=False)  # num_views-dependent, recomputed every run
                return
            print(
                f"[MVS_Synth_Multi] Cache at {cache_path} is legacy (num_views-baked); "
                f"rebuilding as full index."
            )

        file_client = FileClient()
        self.scenes = sorted(
            [d for d in file_client.list_dir(self.ROOT, return_only_dir=True)]
        )

        # FULL raw index: keep EVERY scene (>=2 frames) and ALL its frames (no
        # num_views cut_off). The num_views-dependent start positions are derived
        # later by _compute_start_img_ids(), so this cache is num_views-agnostic.
        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [
                    f[:-4]
                    for f in file_client.list_dir(rgb_dir)
                    if f.endswith(".jpg")
                ]
            )
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

        self._compute_start_img_ids(tuple_with_scene=False)  # num_views-dependent, derived from the full index

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
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_path = osp.join(rgb_dir, basename + ".jpg")
            temp_rgb = file_client.download_file(rgb_path)
            rgb_image = imread_cv2(temp_rgb)
            # Load depthmap
            depth_path = osp.join(depth_dir, basename + ".npy")
            temp_depth = file_client.download_file(depth_path)
            depthmap = np.load(temp_depth)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            cam_path = osp.join(cam_dir, basename + ".safetensor")
            temp_cam = file_client.download_file(cam_path)
            cam = load_file(temp_cam)
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.8, 0.15, 0.05]
            )

            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="MVS_Synth",
                label=self.scenes[scene_id] + "_" + basename,
                instance=osp.join(rgb_dir, basename + ".jpg"),
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
                    "camera", basename + ".json")
                gt_cam_angles = load_caption_cam_angles(cap_path)
                if gt_cam_angles is not None:
                    view_dict["gt_cam_angles"] = gt_cam_angles

            views.append(view_dict)
        assert len(views) == num_views
        return views
