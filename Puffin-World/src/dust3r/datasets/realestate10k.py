import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys
import pickle

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
    Build an iterator over keys within an aoss: s3 bucket.
    iterator_spec: '{cluster}:s3://bucket/prefix/'
    """
    s3_prefix = root_aoss[len("aoss:"):] if root_aoss.startswith("aoss:") else root_aoss
    s3_prefix = _ensure_trailing_slash(s3_prefix)
    iterator_spec = f"{cluster}:{s3_prefix}"

    client = AOSSClient(conf_path)
    return client.get_file_iterator(iterator_spec)


class RE10K_Multi(BaseMultiViewDataset):
    def __init__(
        self,
        *args,
        ROOT,
        cache_path=None,
        min_interval=1,
        max_interval=128,
        cluster_name="aoss",
        conf_path="~/aoss.conf",
        camera_caption_root=None,
        test_sceneids_path=None,
        **kwargs,
    ):
        # Basic fields
        self.ROOT = _ensure_trailing_slash(ROOT)
        # Local root for absolute camera captions:
        #   <camera_caption_root>/<scene>/camera/<basename>.json
        self.camera_caption_root = camera_caption_root
        self.cluster_name = cluster_name
        self.conf_path = conf_path
        self.video = True
        self.is_metric = False
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.test_sceneids_path = test_sceneids_path

        # File client for actual downloads
        self.file_client = FileClient()

        # For computing relative paths from iterator keys
        self._s3_prefix = self.ROOT[len("aoss:"):] if self.ROOT.startswith("aoss:") else self.ROOT
        self._s3_prefix = _ensure_trailing_slash(self._s3_prefix)

        super().__init__(*args, **kwargs)

        # Build / load index (with optional cache)
        self._load_data(cache_path=cache_path)

    def _load_data(self, cache_path=None):
        """
        Load dataset index from cache if available; otherwise build from AOSS
        and (optionally) save cache to `cache_path`.
        """
        # 1) Load from cache if available
        if cache_path is not None and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("schema_version") == "full_list":
                self.scenes = cache["scenes"]
                self.sceneids = cache["sceneids"]
                self.images = cache["images"]
                self.scene_img_list = cache["scene_img_list"]
                print(f"[RE10K_Multi] Loaded {len(self.scenes)} scenes, {len(self.images)} images from cache (full_list).")
                self._apply_split_filter()
                self._compute_start_img_ids(tuple_with_scene=True)   # num_views-dependent, recomputed every run
                self.invalid_scenes = {scene: False for scene in self.scenes}
                return
            print(f"[RE10K_Multi] Cache at {cache_path} is legacy (num_views-baked); rebuilding as full index.")

        # 2) Otherwise, stream bucket keys from AOSS to collect scenes
        files_iter = get_aoss_file_iterator(self.ROOT, self.cluster_name, conf_path=self.conf_path)

        # 's3://bucket/prefix/' -> 'bucket/prefix/' for easier matching
        bucket_prefix_clean = self._s3_prefix.replace("s3://", "")
        bucket_prefix_clean = _ensure_trailing_slash(bucket_prefix_clean)

        scenes_set = set()

        print("[RE10K_Multi] Streaming bucket keys from AOSS to collect scenes...")
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
            # Expect: scene/rgb/xxxx.png or scene/cam/xxxx.npz etc.
            if len(parts) < 2:
                continue

            scene = parts[0]
            scenes_set.add(scene)

        # 3) For each scene, list rgb_dir and build index
        file_client = self.file_client
        self.scenes = sorted(list(scenes_set))

        # FULL raw index: keep EVERY scene and ALL its frames (no num_views cut_off,
        # no dropping short scenes beyond 1-frame). The num_views-dependent start
        # positions are derived later by _compute_start_img_ids(), so this cache is
        # num_views-agnostic.
        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []

        j = 0
        for scene in tqdm(self.scenes, desc="Indexing scenes"):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")

            try:
                rgb_files = [
                    f for f in file_client.list_dir(rgb_dir) if f.endswith(".png")
                ]
            except Exception as e:
                print(f"[RE10K_Multi] Failed to list_dir for {rgb_dir}: {e}, skipping this scene.")
                continue

            basenames = sorted(
                [f[:-4] for f in rgb_files],
                key=lambda x: int(x),
            )

            num_imgs = len(basenames)
            if num_imgs < 2:                     # 1-frame scene can never form a multi-view sample
                continue

            img_ids = list(np.arange(num_imgs) + offset)

            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list

        self.invalid_scenes = {scene: False for scene in self.scenes}

        # 4) Save cache if requested
        if cache_path is not None:
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            print(f"[RE10K_Multi] Saving index cache to {cache_path}")
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
        self._compute_start_img_ids(tuple_with_scene=True)   # num_views-dependent, derived from the full index
        self.invalid_scenes = {scene: False for scene in self.scenes}

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
            print(f"[RE10K_Multi] No test scene split found at {self.test_sceneids_path}; using all scenes for split={self.split}.")
            return

        if self.split == "test":
            keep_old_scene_ids = [i for i, scene in enumerate(self.scenes) if scene in test_scenes]
        else:
            keep_old_scene_ids = [i for i, scene in enumerate(self.scenes) if scene not in test_scenes]

        scenes, sceneids, images, scene_img_list = [], [], [], []

        for new_scene_id, old_scene_id in enumerate(keep_old_scene_ids):
            scene = self.scenes[old_scene_id]
            new_img_ids = []
            for old_img_id in self.scene_img_list[old_scene_id]:
                old_img_id_int = int(old_img_id)
                new_img_ids.append(len(images))
                images.append(self.images[old_img_id_int])
                sceneids.append(new_scene_id)
            scenes.append(scene)
            scene_img_list.append(new_img_ids)

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        # start_img_ids are (re)derived by _compute_start_img_ids() AFTER this filter.
        print(f"[RE10K_Multi] Applied split={self.split}: {len(self.scenes)} scenes, {len(self.images)} images.")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        file_client = self.file_client
        invalid_seq = True
        scene, start_id = self.start_img_ids[idx]

        while invalid_seq:
            # Skip scenes that have been marked invalid
            while self.invalid_scenes[scene]:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]

            all_image_ids = self.scene_img_list[self.sceneids[start_id]]
            pos, ordered_video = self.get_seq_from_start_id(
                num_views, start_id, all_image_ids, rng,
                min_interval=self.min_interval, max_interval=self.max_interval,
                video_prob=1, fix_interval_prob=1.0
            )
            image_idxs = np.array(all_image_ids)[pos]

            views = []
            for view_idx in image_idxs:
                scene_id = self.sceneids[view_idx]
                scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
                rgb_dir = osp.join(scene_dir, "rgb")
                cam_dir = osp.join(scene_dir, "cam")

                basename = self.images[view_idx]

                try:
                    # Load RGB image
                    rgb_oss_path = osp.join(rgb_dir, basename + ".png")
                    temp_rgb = file_client.download_file(rgb_oss_path)
                    rgb_image = imread_cv2(temp_rgb, cv2.IMREAD_COLOR)

                    # RE10K: no depth -> all ones
                    depthmap = np.ones_like(rgb_image[..., 0], dtype=np.float32)

                    # Load camera parameters
                    cam_oss_path = osp.join(cam_dir, basename + ".npz")
                    temp_cam = file_client.download_file(cam_oss_path)
                    cam = np.load(temp_cam)
                    intrinsics = cam["intrinsics"]
                    camera_pose = cam["pose"]
                except Exception as e:
                    print(f"[RE10K_Multi] Error loading {scene} {basename}, skipping. Error: {e}")
                    self.invalid_scenes[scene] = True
                    break

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                view_dict = dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="realestate10k",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=True,
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

            if len(views) == num_views:
                invalid_seq = False

        return views
