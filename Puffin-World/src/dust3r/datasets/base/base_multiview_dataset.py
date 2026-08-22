import PIL
import numpy as np
import torch
import random
import itertools
from src.dust3r.datasets.base.easy_dataset import EasyDataset
from src.dust3r.datasets.utils.transforms import ImgNorm, SeqColorJitter
from src.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import src.dust3r.datasets.utils.cropping as cropping
from src.dust3r.datasets.utils.corr import extract_correspondences_from_pts3d


def get_ray_map(c2w1, c2w2, intrinsics, h, w):
    rel = np.linalg.inv(c2w1) @ c2w2
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i + 0.5, j + 0.5, np.ones_like(i)], axis=-1)
    ro = rel[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (rel[:3, :3] @ rd).T.reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    ray_map = np.concatenate([ro, rd], axis=-1)
    return ray_map


def load_caption_cam_angles(caption_path):
    """Read a camera-caption JSON written by scripts/annotation/camera_caption_aoss.py
    and return per-view GT angles [roll, pitch, vfov, k1] (radians, np.float32).

    NOTE: the caption's vfov refers to the ORIGINAL image. The training crop
    (`_crop_resize_if_necessary`) changes the effective FOV; the model's
    `_apply_gt_camera_params` therefore derives the PF focal from the
    crop-consistent `cam_intrinsics` in the data dict and only uses
    roll / pitch (crop-invariant) plus k1 from these angles.

    Returns None when the file is missing or unreadable, or when the VLM
    caption failed to parse (parse_ok=False) — callers should then omit
    `gt_cam_angles` from the view so downstream code falls back gracefully.
    """
    import json as _json
    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            cap = _json.load(f)
        if not cap.get("parse_ok", True):
            return None
        roll = float(cap["roll"])
        pitch = float(cap["pitch"])
        vfov = float(cap["vfov"])
        k1 = float(cap.get("k1", 0.0))
        # Sanity guard against caption-parse mishaps that slipped through
        # (e.g. years from the description matched as angles): reject
        # physically implausible values so training falls back gracefully.
        if not (abs(roll) <= np.pi / 2 + 1e-3
                and abs(pitch) <= np.pi / 2 + 1e-3
                and 0.0 < vfov < np.pi
                and abs(k1) <= 2.0):
            return None
        return np.array([roll, pitch, vfov, k1], dtype=np.float32)
    except Exception:
        return None


class BaseMultiViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseMultiViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        num_views=None,
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=ImgNorm,
        aug_crop=False,
        n_corres=0,
        nneg=0,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
    ):
        assert num_views is not None, "undefined num_views"
        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)

        self.n_corres = n_corres
        self.nneg = nneg
        assert (
            self.n_corres == "all"
            or isinstance(self.n_corres, int)
            or (
                isinstance(self.n_corres, list) and len(self.n_corres) == self.num_views
            )
        ), f"Error, n_corres should either be 'all', a single integer or a list of length {self.num_views}"
        assert (
            self.nneg == 0 or self.n_corres != "all"
        ), "nneg should be 0 if n_corres is all"

        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            transform = eval(transform)
        if transform == SeqColorJitter:
            transform = SeqColorJitter()
            self.is_seq_color_jitter = True
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.seq_aug_crop = seq_aug_crop

    def _compact_index(self):
        """Convert the big per-frame index containers from Python lists to numpy
        arrays. Numpy keeps the payload in refcount-free buffers, so the COW pages
        stay shared forever. ``self.scenes`` (per-scene, small) stays a list.
        Idempotent; runs pre-fork from __init__ via _compute_start_img_ids."""
        if getattr(self, 'sceneids', None) is not None:
            self.sceneids = np.asarray(self.sceneids, dtype=np.int64)
        images = getattr(self, 'images', None)
        if images is not None and not isinstance(images, np.ndarray):
            # fixed-width unicode array: one contiguous buffer, no per-element
            # PyObjects (np.str_ elements are str subclasses -> drop-in).
            self.images = np.asarray(images)
        if getattr(self, 'scene_img_list', None) is not None:
            self.scene_img_list = [np.asarray(ids, dtype=np.int64)
                                   for ids in self.scene_img_list]

    def _compute_start_img_ids(self, tuple_with_scene=False):
        """Derive valid window START ids from the FULL per-scene frame index
        (``self.scene_img_list``) using the CURRENT ``self.num_views`` /
        ``self.allow_repeat``.
        tuple_with_scene=False -> flat global start ids (dl3dv / hypersim / ...).
        tuple_with_scene=True  -> (scene, global_start_id) tuples (re10k / puffin_omni).
        """
        # Every dataset calls this on BOTH the cache-load and fresh-build paths,
        # always after any split filter -- so compacting here guarantees the
        # index is numpy (COW-shareable) before the dataloader forks workers.
        self._compact_index()
        cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        if tuple_with_scene:
            # (scene, global_id) pairs (re10k / puffin_omni). Per-scene Python loop;
            # these datasets are smaller / used at modest num_views.
            start_img_ids = []
            for s, img_ids in enumerate(self.scene_img_list):
                n = len(img_ids)
                if n < cut_off:
                    continue
                scene = self.scenes[s]
                start_img_ids.extend((scene, int(i)) for i in img_ids[: n - cut_off + 1])
            self.start_img_ids = start_img_ids
        else:
            # flat global ids (dl3dv / hypersim / mvs_synth / tartanair / scannet).
            # VECTORIZED: loop is over SCENES (thousands), the per-frame work is
            # numpy slice + concatenate (C-level) -> sub-second even for tens of
            # millions of frames, so recomputing this every startup is negligible.
            chunks = [np.asarray(img_ids)[: len(img_ids) - cut_off + 1]
                      for img_ids in self.scene_img_list if len(img_ids) >= cut_off]
            self.start_img_ids = (np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64))
        return self.start_img_ids

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def efficient_random_intervals(
        start,
        num_elements,
        interval_range,
        fixed_interval_prob=0.8,
        weights=None,
        seed=42,
    ):
        if random.random() < fixed_interval_prob:
            intervals = random.choices(interval_range, weights=weights) * (
                num_elements - 1
            )
        else:
            intervals = [
                random.choices(interval_range, weights=weights)[0]
                for _ in range(num_elements - 1)
            ]
        return list(itertools.accumulate([start] + intervals))

    def sample_based_on_timestamps(self, i, timestamps, num_views, interval=1):
        time_diffs = np.abs(timestamps - timestamps[i])
        ids_candidate = np.where(time_diffs < interval)[0]
        ids_candidate = np.sort(ids_candidate)
        if (self.allow_repeat and len(ids_candidate) < num_views // 3) or (
            len(ids_candidate) < num_views
        ):
            return []
        ids_sel_list = []
        ids_candidate_left = ids_candidate.copy()
        while len(ids_candidate_left) >= num_views:
            ids_sel = np.random.choice(ids_candidate_left, num_views, replace=False)
            ids_sel_list.append(sorted(ids_sel))
            ids_candidate_left = np.setdiff1d(ids_candidate_left, ids_sel)

        if len(ids_candidate_left) > 0 and len(ids_candidate) >= num_views:
            ids_sel = np.concatenate(
                [
                    ids_candidate_left,
                    np.random.choice(
                        np.setdiff1d(ids_candidate, ids_candidate_left),
                        num_views - len(ids_candidate_left),
                        replace=False,
                    ),
                ]
            )
            ids_sel_list.append(sorted(ids_sel))

        if self.allow_repeat:
            ids_sel_list.append(
                sorted(np.random.choice(ids_candidate, num_views, replace=True))
            )

        # add sequences with fixed intervals (all possible intervals)
        pos_i = np.where(ids_candidate == i)[0][0]
        curr_interval = 1
        stop = len(ids_candidate) < num_views
        while not stop:
            pos_sel = [pos_i]
            count = 0
            while len(pos_sel) < num_views:
                if count % 2 == 0:
                    curr_pos_i = pos_sel[-1] + curr_interval
                    if curr_pos_i >= len(ids_candidate):
                        stop = True
                        break
                    pos_sel.append(curr_pos_i)
                else:
                    curr_pos_i = pos_sel[0] - curr_interval
                    if curr_pos_i < 0:
                        stop = True
                        break
                    pos_sel.insert(0, curr_pos_i)
                count += 1
            if not stop and len(pos_sel) == num_views:
                ids_sel = sorted([ids_candidate[pos] for pos in pos_sel])
                if ids_sel not in ids_sel_list:
                    ids_sel_list.append(ids_sel)
            curr_interval += 1
        return ids_sel_list

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        # find id_ref's position; works for BOTH Python lists and the numpy
        # arrays produced by _compact_index (ndarray has no .index()).
        _matches = np.nonzero(np.asarray(ids_all) == id_ref)[0]
        assert _matches.size > 0, f"id_ref {id_ref} not in ids_all"
        pos_ref = int(_matches[0])
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            effective_min = min(min_interval, max_interval)
            intervals = [
                rng.choice(range(effective_min, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video

    def get_img_and_ray_masks(self, is_metric, v, rng, p=[0.8, 0.15, 0.05]):
        # generate img mask and raymap mask
        if v == 0 or (not is_metric):
            img_mask = True
            raymap_mask = False
        else:
            rand_val = rng.random()
            if rand_val < p[0]:
                img_mask = True
                raymap_mask = False
            elif rand_val < p[0] + p[1]:
                img_mask = False
                raymap_mask = True
            else:
                img_mask = True
                raymap_mask = True
        return img_mask, raymap_mask

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # print("Receiving:" , idx)
        if isinstance(idx, (tuple, list, np.ndarray)):
            # the idx is specifying the aspect-ratio
            idx, ar_idx, nview = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            nview = self.num_views

        assert nview >= 1 and nview <= self.num_views
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        try:
            views = self._get_views(idx, resolution, self._rng, nview)
        except Exception as e:
            print(f"get_views error: {e}")
            if hasattr(self, "file_client"):
                self.file_client.cleanup_temp_files()
            return BaseMultiViewDataset.__getitem__(self, ((idx + 1) % self.__len__(), ar_idx, nview))
        assert len(views) == nview

        if "camera_pose" not in views[0]:
            views[0]["camera_pose"] = np.ones((4, 4), dtype=np.float32)
        first_view_camera_pose = views[0]["camera_pose"]
        transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

        for v, view in enumerate(views):
            assert (
                "pts3d" not in view
            ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view["idx"] = (idx, ar_idx, v)

            # encode the image
            width, height = view["img"].size

            view["true_shape"] = np.int32((height, width))
            view["img"] = transform(view["img"])
            view["sky_mask"] = view["depthmap"] < 0

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(
                    view["camera_pose"]
                ).all(), f"NaN in camera pose for view {view_name(view)}"

            ray_map = get_ray_map(
                first_view_camera_pose,
                view["camera_pose"],
                view["camera_intrinsics"],
                height,
                width,
            )
            view["ray_map"] = ray_map.astype(np.float32)

            assert "pts3d" not in view
            assert "valid_mask" not in view
            assert np.isfinite(
                view["depthmap"]
            ).all(), f"NaN in depthmap for view {view_name(view)}"
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view["camera_intrinsics"]

        if self.n_corres > 0:
            ref_view = views[0]
            for view in views:
                corres1, corres2, valid = extract_correspondences_from_pts3d(
                    ref_view, view, self.n_corres, self._rng, nneg=self.nneg
                )
                view["corres"] = (corres1, corres2)
                view["valid_corres"] = valid

        try:
            self._normalize_all_geometry_visionbana(views)
        except Exception as e:
            print(f"_normalize_all_geometry_visionbana error: {e}")
            if hasattr(self, "file_client"):
                self.file_client.cleanup_temp_files()
            return BaseMultiViewDataset.__getitem__(self, ((idx + 1) % self.__len__(), ar_idx, nview))

        # last thing done!
        for view in views:
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        return views

    def _normalize_raymap_only(self, views):
        # This method only normalizes the raymaps and camera poses.
        # The depth and point clouds are not normalized.

        translations = []
        for view in views:
            T = view["camera_pose"][:3, 3]
            translations.append(T)

        translations = np.array(translations)
        norms = np.linalg.norm(translations, axis=-1)
        mean_norm = np.mean(norms)
        for view in views:
            view["camera_pose"][:3, 3] /= mean_norm

        first_view_camera_pose = views[0]["camera_pose"]

        for view in views:
            height, width = view["true_shape"]
            ray_map = get_ray_map(
                first_view_camera_pose,
                view["camera_pose"],
                view["camera_intrinsics"],
                height,
                width,
            )
            view["ray_map"] = ray_map.astype(np.float32)

    def _normalize_all_geometry(self, views):
        # Normalize all valid depths so that the minimum valid depth across the
        # sequence becomes 1.0, and apply the same metric scale to all geometry info (including raymaps).
        min_depth = float('inf')
        for view in views:
            depthmap = view["depthmap"]
            valid_mask = view["valid_mask"]

            if np.any(valid_mask):
                cur_min_depth = float(depthmap[valid_mask].min())
                min_depth = min(min_depth, cur_min_depth)

        if min_depth is None or not np.isfinite(min_depth) or min_depth <= 0:
            raise ValueError(f"Invalid minimum depth: {min_depth}")

        scale = float(min_depth)
        for view in views:
            depthmap = view["depthmap"].astype(np.float32, copy=True)
            valid_mask = view["valid_mask"]

            depthmap[valid_mask] /= scale
            view["depthmap"] = depthmap
            view["disparity_for_vae"] = depth_to_vaedisparsity(depthmap, valid_mask)

            view["pts3d"] = (view["pts3d"] / scale).astype(np.float32)
            view["camera_pose"][:3, 3] /= scale

        first_view_camera_pose = views[0]["camera_pose"]
        for view in views:
            height, width = view["true_shape"]
            ray_map = get_ray_map(
                first_view_camera_pose,
                view["camera_pose"],
                view["camera_intrinsics"],
                height,
                width,
            )
            view["ray_map"] = ray_map.astype(np.float32)

        return scale

    def _normalize_all_geometry_visionbana(self, views):
        sum_depth = 0.0
        count_depth = 0
        for view in views:
            sum_depth += float(np.einsum("ij,ij->", view["depthmap"], view["valid_mask"], optimize=True))
            count_depth += int(np.count_nonzero(view["valid_mask"]))

        if count_depth == 0:
            raise ValueError("No valid depths across the sequence")

        scale = sum_depth / count_depth
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid mean depth scale: {scale}")

        for view in views:
            depthmap = view["depthmap"].astype(np.float32, copy=True)
            valid_mask = view["valid_mask"]

            depthmap[valid_mask] /= scale
            view["depthmap"] = depthmap
            view["depth_visionbanana"] = depth_to_visionbanana(depthmap, valid_mask)

            view["pts3d"] = (view["pts3d"] / scale).astype(np.float32)
            view["camera_pose"][:3, 3] /= scale

        first_view_camera_pose = views[0]["camera_pose"]
        for view in views:
            height, width = view["true_shape"]
            ray_map = get_ray_map(
                first_view_camera_pose,
                view["camera_pose"],
                view["camera_intrinsics"],
                height,
                width,
            )
            view["ray_map"] = ray_map.astype(np.float32)

        return scale

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(
                width, int
            ), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(
                height, int
            ), f"Bad type for {height=} {type(height)=}, should be int"
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]

        assert view["ray_map"].shape == (height, width, 6)
        view["ray_map"] = view["ray_map"].swapaxes(0, 1)

        assert view["sky_mask"].shape == (height, width)
        view["sky_mask"] = view["sky_mask"].swapaxes(0, 1)

        if "corres" in view:
            # transpose correspondences x and y
            view["corres"][0] = view["corres"][0][:, [1, 0]]
            view["corres"][1] = view["corres"][1][:, [1, 0]]


def depth_to_vaedisparsity(depth, valid_mask):
    # the input depth here must be >= 1.0 for valid pixels, so disparity = 1/depth is in (0, 1],
    # then mapped to (-1, 1] for VAE.
    if torch.is_tensor(depth):
        # We now only train on DL3DV, TartanAir all depth are valid, the only invalid area is the sky, so we set depth as inf
        # (VAE disparity is in [-1, 1], where -1 corresponds to disparity=0.)
        disparity = torch.full_like(depth, -1.0)

        # assert depth[valid_mask].min() >= 1.0, f"Bad depth: {depth[valid_mask].min()}"
        disparity[valid_mask] = 1.0 / depth[valid_mask]  # in (0, 1]
        disparity[valid_mask] = 2.0 * disparity[valid_mask] - 1.0  # in [-1, 1]
        
    else:
        # We now only train on DL3DV, all depth are valid, the only invalid area is the sky, so we set depth as inf
        # (VAE disparity is in [-1, 1], where -1 corresponds to disparity=0.)
        disparity = np.full_like(depth, -1.0, dtype=np.float32)

        # assert depth[valid_mask].min() >= 1.0, f"Bad depth: {depth[valid_mask].min()}"
        disparity[valid_mask] = 1.0 / depth[valid_mask]  # in (0, 1]
        disparity[valid_mask] = 2.0 * disparity[valid_mask] - 1.0  # in (-1, 1]

    return disparity

_VISIONBANANA_LAMBDA = -3.0
_VISIONBANANA_C = 10.0 / 3.0
_VISIONBANANA_HILBERT_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],  # black
        [0.0, 0.0, 1.0],  # blue
        [0.0, 1.0, 1.0],  # cyan
        [0.0, 1.0, 0.0],  # green
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 0.0],  # red
        [1.0, 0.0, 1.0],  # magenta
        [1.0, 1.0, 1.0],  # white
    ],
    dtype=np.float32,
)


def depth_to_visionbanana(depth, valid_mask):
    is_tensor = torch.is_tensor(depth)
    if is_tensor:
        device = depth.device
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        mask_np = valid_mask.detach().cpu().numpy().astype(bool)
    else:
        depth_np = np.asarray(depth, dtype=np.float32)
        mask_np = np.asarray(valid_mask, dtype=bool)

    out = np.full(depth_np.shape + (3,), 1.0, dtype=np.float32)

    if np.any(mask_np):
        d = np.clip(depth_np[mask_np], 0.0, None)
        lam = _VISIONBANANA_LAMBDA
        c = _VISIONBANANA_C
        # f(d) = 1 - (1 - d / (λ·c))^(λ+1); with λ=-3, c=10/3 ⇒ 1 - (1 + d/10)^(-2)
        f = 1.0 - np.power(1.0 - d / (lam * c), lam + 1.0)
        f = np.clip(f, 0.0, 1.0 - 1e-7)

        n_segs = _VISIONBANANA_HILBERT_VERTICES.shape[0] - 1  # 7
        pos = f * n_segs
        seg = np.minimum(np.floor(pos).astype(np.int64), n_segs - 1)
        t = (pos - seg).astype(np.float32)[:, None]
        v0 = _VISIONBANANA_HILBERT_VERTICES[seg]
        v1 = _VISIONBANANA_HILBERT_VERTICES[seg + 1]
        rgb = v0 + t * (v1 - v0)  # [0, 1]
        out[mask_np] = 2.0 * rgb - 1.0  # [-1, 1]

    if is_tensor:
        return torch.from_numpy(out).to(device)
    return out


def visionbanana_to_depth(depth_visionbanana, valid_mask=None):
    is_tensor = torch.is_tensor(depth_visionbanana)
    if is_tensor:
        device = depth_visionbanana.device
        rgb = depth_visionbanana.detach().cpu().numpy().astype(np.float32)
    else:
        rgb = np.asarray(depth_visionbanana, dtype=np.float32)

    rgb_unit = (rgb + 1.0) * 0.5  # back to [0, 1]
    flat = rgb_unit.reshape(-1, 3)

    verts = _VISIONBANANA_HILBERT_VERTICES
    n_segs = verts.shape[0] - 1
    seg_v0 = verts[:-1]                       # (n_segs, 3)
    seg_v1 = verts[1:]                        # (n_segs, 3)
    seg_dir = seg_v1 - seg_v0                 # (n_segs, 3)
    seg_len_sq = np.sum(seg_dir ** 2, axis=1) # (n_segs,)

    diff = flat[:, None, :] - seg_v0[None, :, :]            # (N, n_segs, 3)
    t = np.sum(diff * seg_dir[None, :, :], axis=2) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = seg_v0[None, :, :] + t[:, :, None] * seg_dir[None, :, :]
    dist_sq = np.sum((flat[:, None, :] - proj) ** 2, axis=2) # (N, n_segs)
    best = np.argmin(dist_sq, axis=1)                        # (N,)
    rows = np.arange(flat.shape[0])
    f = (best.astype(np.float32) + t[rows, best]) / n_segs
    f = np.clip(f, 0.0, 1.0 - 1e-7)

    lam = _VISIONBANANA_LAMBDA
    c = _VISIONBANANA_C
    # invert f(d) = 1 - (1 - d/(λc))^(λ+1)  ⇒  d = λc * (1 - (1 - f)^(1/(λ+1)))
    d = lam * c * (1.0 - np.power(1.0 - f, 1.0 / (lam + 1.0)))
    depth = d.reshape(rgb.shape[:-1]).astype(np.float32)

    if valid_mask is not None:
        mask_np = (
            valid_mask.detach().cpu().numpy().astype(bool)
            if torch.is_tensor(valid_mask)
            else np.asarray(valid_mask, dtype=bool)
        )
        depth = np.where(mask_np, depth, 0.0).astype(np.float32)

    if is_tensor:
        return torch.from_numpy(depth).to(device)
    return depth


def vaedisparsity_to_depth(vae_disparity, valid_mask=None):
    # disparity (-1, 1] to depth (0, inf)
    if torch.is_tensor(vae_disparity):
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
        disparity = (vae_disparity + 1.0) * 0.5  # back to (0, 1]
        depth = torch.zeros_like(vae_disparity)
        mask = valid_mask if valid_mask is not None else (disparity > 0)
        depth[mask] = 1.0 / disparity[mask]
    else:
        vae_disparity = np.asarray(vae_disparity, dtype=np.float32)
        if valid_mask is not None:
            valid_mask = valid_mask.astype(bool)
        disparity = (vae_disparity + 1.0) * 0.5  # back to (0, 1]
        depth = np.zeros_like(vae_disparity)
        mask = valid_mask if valid_mask is not None else (disparity > 0)
        depth[mask] = 1.0 / disparity[mask]

    return depth