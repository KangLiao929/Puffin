import os.path as osp
import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from safetensors.numpy import load_file as load_safetensors
from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset, load_caption_cam_angles
from src.dust3r.utils.image import imread_cv2, rgb
from src.dust3r.viz import colorize_np
from src.dust3r.oss_file_client import FileClient


class TartanAir_Multi(BaseMultiViewDataset):

    def __init__(self, ROOT, *args, cache_path=None, max_samples=None,
                 camera_caption_root=None, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 10
        self.max_samples = max_samples
        self.file_client = FileClient()
        # Local root for absolute camera captions:
        #   <camera_caption_root>/<rel_seq>/<basename>_camera.json
        self.camera_caption_root = camera_caption_root
        super().__init__(*args, **kwargs)
        # loading all
        assert self.split is None
        self._load_data(cache_path=cache_path)

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
                    f"[TartanAir_Multi] Loaded {len(self.scenes)} scenes, "
                    f"{len(self.images)} images from cache (full_list)."
                )
                self._compute_start_img_ids(tuple_with_scene=False)  # num_views-dependent, recomputed every run
                if self.max_samples:
                    self.start_img_ids = self.start_img_ids[:self.max_samples]
                return
            print(
                f"[TartanAir_Multi] Cache at {cache_path} is legacy "
                f"(num_views-baked); rebuilding as full index."
            )

        scene_dirs = sorted(
            [
                d
                for d in self.file_client.list_dir(self.ROOT, return_only_dir=True)
                if d
            ]
        )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        j = 0

        for scene in tqdm(scene_dirs, desc="Indexing scenes"):
            for mode in ["Easy", "Hard"]:
                mode_dir = os.path.join(self.ROOT, scene, mode)
                seq_dirs = sorted(
                    [
                        os.path.join(mode_dir, d)
                        for d in self.file_client.list_dir(mode_dir, return_only_dir=True)
                        if d
                    ]
                )
                for seq_dir in seq_dirs:
                    basenames = sorted(
                        [
                            f[:-8]
                            for f in self.file_client.list_dir(seq_dir)
                            if f.endswith("_rgb.png")
                        ]
                    )
                    num_imgs = len(basenames)
                    if num_imgs < 2:  # 1-frame scene can never form a multi-view sample
                        print(f"Skipping {scene}")
                        continue
                    img_ids = list(np.arange(num_imgs) + offset)

                    scenes.append(seq_dir)
                    scene_img_list.append(img_ids)
                    sceneids.extend([j] * num_imgs)
                    images.extend(basenames)
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
        if self.max_samples:
            self.start_img_ids = self.start_img_ids[:self.max_samples]

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(self, idx, resolution, rng, num_views):
        file_client = self.file_client
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = self.scenes[scene_id]
            basename = self.images[view_idx]

            img = basename + "_rgb.png"
            temp_img = file_client.download_file(osp.join(scene_dir, img))
            temp_depth = file_client.download_file(osp.join(scene_dir, basename + "_depth.npy"))
            temp_cam = file_client.download_file(osp.join(scene_dir, basename + "_cam.safetensor"))
            image = imread_cv2(temp_img)
            depthmap = np.load(temp_depth)
            camera_params = load_safetensors(temp_cam)

            intrinsics = camera_params["camera_intrinsics"]
            camera_pose = camera_params["camera_pose"]

            sky_mask = depthmap >= 1000
            depthmap[sky_mask] = -1.0  # sky
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, img)
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            view_dict = dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset="TartanAir",
                label=scene_dir,
                is_metric=self.is_metric,
                instance=scene_dir + "_" + img,
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
            # scenes store ROOT-prefixed seq dirs; strip ROOT for the local path.
            if self.camera_caption_root is not None:
                root = self.ROOT if self.ROOT.endswith("/") else self.ROOT + "/"
                rel_seq = scene_dir[len(root):] if scene_dir.startswith(root) \
                    else scene_dir.lstrip("/")
                cap_path = osp.join(
                    self.camera_caption_root, rel_seq, basename + "_camera.json")
                # (vfov is original-image; PF focal comes from cam_intrinsics
                # at model side — see _apply_gt_camera_params.)
                gt_cam_angles = load_caption_cam_angles(cap_path)
                if gt_cam_angles is not None:
                    view_dict["gt_cam_angles"] = gt_cam_angles

            views.append(view_dict)
        assert len(views) == num_views
        return views
