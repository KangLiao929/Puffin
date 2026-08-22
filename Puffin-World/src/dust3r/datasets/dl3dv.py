import os.path as osp
import os
import pickle
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

from src.dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset, load_caption_cam_angles
from src.dust3r.utils.image import imread_cv2
from src.dust3r.utils.image import rgb
from src.dust3r.oss_file_client import FileClient
from src.dust3r.viz import colorize_np

import torch
from torch.utils.data import DataLoader
from aoss_client.client import Client as AOSSClient

cv2.setNumThreads(0)


def _ensure_trailing_slash(s: str) -> str:
    """Ensure a path-like prefix ends with '/'."""
    return s if s.endswith("/") else (s + "/")


def _rel_from_prefix(p: str, bucket_prefix_clean: str) -> str:
    """
    Robustly compute 'rel' from an absolute key 'p', even if upstream keys omit
    the slash between prefix and the next token.

    Examples:
        bucket_prefix_clean = 'bucket/a/b/c/'
        p = 'bucket/a/b/c/1K/hash/dense/rgb/xxx.png'  -> '1K/hash/dense/rgb/xxx.png'
        p = 'bucket/a/b/c1K/hash/dense/rgb/xxx.png'   -> '1K/hash/dense/rgb/xxx.png'  (no slash case)
    """
    if p.startswith(bucket_prefix_clean):
        rel = p[len(bucket_prefix_clean):]
    else:
        prefix_no_slash = bucket_prefix_clean[:-1] if bucket_prefix_clean.endswith('/') else bucket_prefix_clean
        if p.startswith(prefix_no_slash):
            rel = p[len(prefix_no_slash):]
            if rel.startswith('/'):
                rel = rel[1:]
        else:
            return None

    # Safety fix: if first segment is 'K', recover to '1K' (typical off-by-one)
    parts = rel.split('/')
    if parts and parts[0] == 'K':
        start_idx = len(bucket_prefix_clean)
        if start_idx <= len(p) and p[start_idx - 1:start_idx] == '1':
            parts[0] = '1K'
            rel = '/'.join(parts)
    return rel


def get_aoss_file_iterator(root_aoss: str, cluster: str, file_client_fallback: FileClient, conf_path=None):
    """
    Build an iterator over keys within an aoss: s3 bucket.
    iterator_spec: '{cluster}:s3://bucket/prefix/'
    """
    s3_prefix = root_aoss[len("aoss:"):] if root_aoss.startswith("aoss:") else root_aoss
    s3_prefix = _ensure_trailing_slash(s3_prefix)

    iterator_spec = f"{cluster}:{s3_prefix}"

    client = AOSSClient(conf_path)
    return client.get_file_iterator(iterator_spec)


class DL3DV_Multi(BaseMultiViewDataset):
    """
    Multi-view dataset reader for DL3DV using AOSS streaming.

    Expected layout under ROOT (aoss:s3://.../processed_dl3dv_ours/):
        1K/<hash>/dense/rgb/frame_00001.png
        1K/<hash>/dense/depth/frame_00001.npy
        1K/<hash>/dense/cam/frame_00001.npz
        1K/<hash>/dense/sky_mask/frame_00001.png
        1K/<hash>/dense/outlier_mask/frame_00001.png
    """

    def __init__(self, *args,
                 split,
                 ROOT,
                 cache_path=None,
                 debug_dl3dv=False,
                 max_samples=None,
                 cluster_name="aoss",
                 conf_path="~/aoss.conf",
                 debug_num_files=None,
                 min_interval=1,
                 max_interval=20,
                 depth_da3_root=None,
                 camera_caption_root=None,
                 test_sceneids_path=None,
                 **kwargs):
        self.ROOT = _ensure_trailing_slash(ROOT)  # keep trailing slash for consistent joins
        # Local root for the depth-anything-3 depth maps, layout mirrors the AOSS
        # scene structure: <depth_da3_root>/<scene>/dense/depth_da3/frame_XXXXX.npy.
        assert depth_da3_root is not None, (
            "DL3DV_Multi requires depth_da3_root (set it in the gen config, "
            "e.g. configs/datasets/multi_view/gen_dl3dv.py)."
        )
        self.depth_da3_root = depth_da3_root
        # Local root for absolute camera captions:
        #   <camera_caption_root>/<scene>/dense/camera/frame_XXXXX.json
        self.camera_caption_root = camera_caption_root
        self.cluster_name = cluster_name
        self.video = True
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.is_metric = False
        self.debug_dl3dv = debug_dl3dv
        self.max_samples = max_samples
        self.conf_path = conf_path
        self.debug_num_files = debug_num_files  # only index first N files for debugging
        self.test_sceneids_path = test_sceneids_path
        self.file_client = FileClient()

        # Normalize 'aoss:' → keep 's3://...' and ensure trailing '/'
        self._s3_prefix = self.ROOT[len("aoss:"):] if self.ROOT.startswith("aoss:") else self.ROOT
        self._s3_prefix = _ensure_trailing_slash(self._s3_prefix)

        super().__init__(*args, **kwargs)
        self.split = split

        # Auto build/load index
        self._load_data(cache_path=cache_path)

    def _scene_dir_main(self, scene: str) -> str:
        """Compose the canonical 'scene/dense' path under ROOT."""
        return osp.join(self.ROOT, scene, "dense")

    def _scene_dir_noslash(self, scene: str) -> str:
        """
        Compose a fallback path for the case where the object store omitted
        the slash between ROOT and scene (rare but seen in some buckets).
        """
        root_no_slash = self.ROOT[:-1] if self.ROOT.endswith('/') else self.ROOT
        scene_no_leading = scene.lstrip('/')
        return root_no_slash + scene_no_leading + "/dense"

    def _download_with_fallback(self, file_client: FileClient, main_path: str, alt_path: str, force_redownload: bool = False):
        """
        Try downloading 'main_path'; if it fails or returns None, try 'alt_path'.
        Return a local file path or raise IOError if both failed.
        """
        temp = file_client.download_file(main_path, force_redownload=force_redownload)
        if temp is None or not osp.exists(temp):
            temp2 = file_client.download_file(alt_path, force_redownload=force_redownload)
            if temp2 is None or not osp.exists(temp2):
                raise IOError(f"download_file returned None or missing path for both: {main_path} and {alt_path}")
            return temp2
        return temp

    def _load_data(self, cache_path=None):
        """
        Load dataset index from cache if available; otherwise stream from AOSS and (optionally) save cache.
        """
        if cache_path is not None and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("schema_version") == "full_list":
                self.scenes = cache["scenes"]
                self.sceneids = cache["sceneids"]
                self.images = cache["images"]
                self.scene_img_list = cache["scene_img_list"]
                print(f"[DL3DV_Multi] Loaded {len(self.scenes)} scenes, {len(self.images)} images from cache (full_list).")
                self._apply_split_filter()
                self._compute_start_img_ids()          # num_views-dependent, recomputed every run
                if self.max_samples:
                    self.start_img_ids = self.start_img_ids[:self.max_samples]
                return
            print(f"[DL3DV_Multi] Cache at {cache_path} is legacy (num_views-baked); rebuilding as full index.")

        files_iter = get_aoss_file_iterator(self.ROOT, self.cluster_name, self.file_client, conf_path=self.conf_path)

        # 's3://bucket/prefix/' -> 'bucket/prefix/' for comparing with iterator keys
        bucket_prefix_clean = self._s3_prefix.replace("s3://", "")
        bucket_prefix_clean = _ensure_trailing_slash(bucket_prefix_clean)

        scenes_to_rgbs = {}
        print("Streaming bucket keys from AOSS...")
        file_count = 0
        debug_printed = 0

        for p, _k in tqdm(files_iter, desc="Scanning keys"):
            # Accept both with and without the trailing slash variant
            if not (p.startswith(bucket_prefix_clean) or p.startswith(bucket_prefix_clean[:-1])):
                continue

            rel = _rel_from_prefix(p, bucket_prefix_clean)
            if rel is None:
                continue

            if debug_printed < 3:
                debug_printed += 1

            # Only RGB pngs in .../dense/rgb/
            if "/dense/rgb/" not in rel or not rel.endswith(".png"):
                continue

            parts = rel.split("/")
            # Expect: ['1K','<hash>','dense','rgb','frame_XXXXX.png']
            if len(parts) < 5 or parts[2] != "dense" or parts[3] != "rgb":
                continue

            scene = "/".join(parts[:2])   # '1K/<hash>'
            fname = parts[-1]
            scenes_to_rgbs.setdefault(scene, []).append(fname)

            file_count += 1
            if self.debug_num_files is not None and file_count >= self.debug_num_files:
                break

        all_subscenes = sorted(scenes_to_rgbs.keys())

        # FULL raw index: keep EVERY scene and ALL its frames (no num_views cut_off,
        # no dropping short scenes). The num_views-dependent start positions are
        # derived later by _compute_start_img_ids(), so this cache is num_views-agnostic.
        scenes, sceneids, images, scene_img_list = [], [], [], []
        offset, j = 0, 0
        for scene in tqdm(all_subscenes, desc="Indexing subscenes"):
            rgb_paths = sorted(scenes_to_rgbs[scene])
            num_imgs = len(rgb_paths)
            if num_imgs < 2:                     # 1-frame scene can never form a multi-view sample
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
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            print(f"Saving full index cache to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "schema_version": "full_list",
                    "scenes": self.scenes,
                    "sceneids": self.sceneids,
                    "images": self.images,
                    "scene_img_list": self.scene_img_list,
                }, f)

        self._apply_split_filter()
        self._compute_start_img_ids()          # num_views-dependent, derived from the full index
        if self.max_samples:
            self.start_img_ids = self.start_img_ids[:self.max_samples]

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
            print(f"[DL3DV_Multi] No test scene split found at {self.test_sceneids_path}; using all scenes for split={self.split}.")
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
        # start_img_ids are (re)derived by _compute_start_img_ids() AFTER this filter.
        print(f"[DL3DV_Multi] Applied split={self.split}: {len(self.scenes)} scenes, {len(self.images)} images.")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        """
        Fetch a sequence of views; each file is downloaded to a temp path via FileClient before reading.
        """

        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]

        pos, ordered_video = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, rng,
            min_interval=self.min_interval, max_interval=self.max_interval,
            block_shuffle=25, video_prob=1, fix_interval_prob=1.0
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for view_idx in image_idxs:
            scene_id = self.sceneids[view_idx]
            scene = self.scenes[scene_id]                 # e.g., '1K/<hash>'
            scene_dir_main = self._scene_dir_main(scene)  # canonical
            scene_dir_alt  = self._scene_dir_noslash(scene)  # fallback for no-slash buckets

            rgb_fname = self.images[view_idx]             # e.g., 'frame_00001.png'
            basename = rgb_fname[:-4]

            # Compose object keys (both main and noslash variants)
            rgb_main = osp.join(scene_dir_main, "rgb", rgb_fname)
            rgb_alt  = osp.join(scene_dir_alt,  "rgb", rgb_fname)

            # Depth now comes from the local depth-anything-3 dump (not AOSS):
            #   <depth_da3_root>/<scene>/dense/depth_da3/frame_XXXXX.npy
            depth_da3_path = osp.join(self.depth_da3_root, scene, "dense", "depth_da3", basename + ".npy")

            cam_main = osp.join(scene_dir_main, "cam", basename + ".npz")
            cam_alt  = osp.join(scene_dir_alt,  "cam", basename + ".npz")

            sky_main = osp.join(scene_dir_main, "sky_mask", rgb_fname)
            sky_alt  = osp.join(scene_dir_alt,  "sky_mask", rgb_fname)

            outlier_main = osp.join(scene_dir_main, "outlier_mask", rgb_fname)
            outlier_alt  = osp.join(scene_dir_alt,  "outlier_mask", rgb_fname)

            # RGB with retry + noslash fallback
            rgb_image = None
            last_err = None
            for attempt in range(3):
                try:
                    temp_rgb = self._download_with_fallback(
                        self.file_client, rgb_main, rgb_alt, force_redownload=(attempt > 0)
                    )
                    rgb_image = imread_cv2(temp_rgb, cv2.IMREAD_COLOR)
                    if rgb_image is None:
                        raise IOError(f"cv2 failed to read {temp_rgb}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"[DL3DV_Multi] Failed to load {rgb_main} try {attempt+1}/3: {e}")
            if rgb_image is None:
                raise RuntimeError(f"Failed to load {rgb_main} after 3 attempts. Last error: {last_err}")

            # Depth (read directly from the local depth-anything-3 path)
            depthmap = np.load(depth_da3_path).astype(np.float32)
            # depthmap[~np.isfinite(depthmap)] = 0

            # Camera
            temp_cam = self._download_with_fallback(self.file_client, cam_main, cam_alt)
            cam_file = np.load(temp_cam)
            intrinsics = cam_file["intrinsic"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            # Masks
            temp_sky = self._download_with_fallback(self.file_client, sky_main, sky_alt)
            sky_mask = np.array(Image.open(temp_sky).convert("L")) >= 127

            non_sky_invalid = (np.isnan(depthmap) | np.isinf(depthmap) | (depthmap <= 0)) & (~sky_mask)
            if np.any(non_sky_invalid):
                # Stage 1: local dilation-based fill, repeated until convergence or max iters.
                valid_for_pool = (~non_sky_invalid) & (~sky_mask) & np.isfinite(depthmap) & (depthmap > 0)
                if np.any(valid_for_pool):
                    filled = depthmap.astype(np.float32, copy=True)
                    filled[~valid_for_pool] = -np.inf
                    filled = np.ascontiguousarray(filled)

                    kernel = np.ones((5, 5), dtype=np.uint8)
                    max_iters = 24
                    for _ in range(max_iters):
                        before_mask = (~np.isfinite(filled)) | (filled <= 0)
                        dilated = cv2.dilate(filled, kernel)
                        can_fill = non_sky_invalid & before_mask & np.isfinite(dilated) & (dilated > 0)
                        if not np.any(can_fill):
                            break
                        filled[can_fill] = dilated[can_fill]

                    depthmap[non_sky_invalid & np.isfinite(filled) & (filled > 0)] = filled[
                        non_sky_invalid & np.isfinite(filled) & (filled > 0)
                    ]

                rem_invalid = (np.isnan(depthmap) | np.isinf(depthmap) | (depthmap <= 0)) & (~sky_mask)
                depthmap[rem_invalid] = 0.0

            depthmap[sky_mask] = 0

            # Resize / crop consistently
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="dl3dv",
                label=scene + "_" + rgb_fname,
                instance=osp.join(scene_dir_main, "rgb", rgb_fname),
                is_metric=self.is_metric,
                is_video=ordered_video,
                quantile=np.array(0.9, dtype=np.float32),
                img_mask=True,
                ray_mask=False,
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            )

            # Attach VLM-captioned camera angles when a caption root is set.
            #   <camera_caption_root>/<scene>/dense/camera/<basename>.json
            # NOTE: the caption vfov refers to the original image; at PF-build
            # time the model uses the crop-consistent focal from
            # `cam_intrinsics` instead (see _apply_gt_camera_params).
            if self.camera_caption_root is not None:
                cap_path = osp.join(
                    self.camera_caption_root, scene, "dense", "camera",
                    basename + ".json")
                gt_cam_angles = load_caption_cam_angles(cap_path)
                if gt_cam_angles is not None:
                    view_dict["gt_cam_angles"] = gt_cam_angles

            views.append(view_dict)

        assert len(views) == num_views
        return views