"""Evaluation dataset registry + builder for the multi-view eval scripts."""

DATASET_REGISTRY = {
    # camera_caption_root mirrors the training configs
    # (configs/datasets/multi_view/gen_*.py). Without it the loaders never
    # set gt_cam_angles, gt_cam_params degrades to "" and BOTH the offline
    # PF model conditioning and the pf_vis overlays silently fall back to
    # the default camera (roll=0, pitch=0, vfov=90 deg) on every view.
    "re10k": {
        "import": "src.datasets.generation.caption_datasets_realestate10k",
        "root": "aoss:s3://yhluo_sgacer/data/tracking/processed_re10k/",
        "cache_path": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/re10k_index.pkl",
        "extra": {"split": "train", "min_interval": 48, "max_interval": 144,
                  "camera_caption_root":
                      "/mnt/afs_100t/NTU_slab/kliao/data/camera_caption/re10k"},
    },
    "dl3dv": {
        "import": "src.datasets.generation.caption_datasets_dl3dv",
        "root": "aoss:s3://yhluo_sgacer/data/tracking/processed_dl3dv_ours_parts/processed_dl3dv_ours/",
        "cache_path": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/dl3dv_index_all.pkl",
        "extra": {"split": "train", "min_interval": 1, "max_interval": 1,
                  # 2026-08-10: /data_cavan is not mounted on this server; DA3
                  # depth restored per-scene from HF KangLiao/DL3DV-Depth-DA3-
                  # Aligned into the AFS path below (same <scene>/dense/depth_da3
                  # layout).
                  "depth_da3_root": "/mnt/afs_100t/NTU_slab/kliao/data/da3_anno/dl3dv_da3",
                  "camera_caption_root":
                      "/mnt/afs_100t/NTU_slab/kliao/data/camera_caption/dl3dv"},
    },
    "hypersim": {
        "import": "src.datasets.generation.caption_datasets_hypersim",
        "root": "aoss:s3://yhluo_sgacer/data/tracking/processed_hypersim_new/",
        "cache_path": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/hypersim_index_all.pkl",
        "extra": {"split": "train",
                  "camera_caption_root":
                      "/mnt/afs_100t/NTU_slab/kliao/data/camera_caption/hypersim"},
    },
    "mvs_synth": {
        "import": "src.datasets.generation.caption_datasets_mvs_synth",
        "root": "aoss:s3://yhluo_sgacer/data/tracking/processed_mvs_synth",
        "cache_path": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/mvs_synth_index_all.pkl",
        "extra": {"camera_caption_root":
                      "/mnt/afs_100t/NTU_slab/kliao/data/camera_caption/mvs_synth"},
    },
    "tartanair": {
        "import": "src.datasets.generation.caption_datasets_tartanair",
        "root": "aoss:s3://yhluo_sgacer/data/tracking/processed_tartanair/",
        "cache_path": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/tartanair_index_all.pkl",
        "extra": {"split": None,
                  "camera_caption_root":
                      "/mnt/afs_100t/NTU_slab/kliao/data/camera_caption/tartanair"},
    },
    "scannet": {
        # 2026-08-11: added for the train16 scannet eval. Deterministic first
        # window: interval fixed to 1 (gen config trains with 1-8). Captions +
        # DA3 restored per-scene from HF KangLiao/ScanNet-{Absolute-Camera,
        # Depth-DA3-Aligned} (scans/<scene>.zip) into the AFS roots below.
        "import": "src.datasets.generation.caption_datasets_scannet",
        "root": "aoss:s3://yhluo_sgacer/data/tracking/processed_scannet/",
        "cache_path": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/scannet_index.pkl",
        "extra": {"split": "train", "min_interval": 1, "max_interval": 1,
                  "scene_subdir": "scans",
                  "depth_da3_root": "/mnt/afs_100t/NTU_slab/kliao/data/da3_anno/scannet_da3",
                  "camera_caption_root":
                      "/mnt/afs_100t/NTU_slab/kliao/data/camera_caption/scannet"},
    },
    "puffin_omni": {
        "import": "src.datasets.generation.caption_datasets_puffin_omni",
        "root": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/Trajectory",
        "cache_path": "/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/trajectory_afs_index.pkl",
        "extra": {"min_interval": 1, "max_interval": 12},
    },
}


def build_eval_dataset(dataset_name, args):
    import importlib
    entry = DATASET_REGISTRY[dataset_name]
    mod = importlib.import_module(entry["import"])
    cls = mod.CaptionDatasetGen

    # For chunked autoregressive inference, the dataset must supply enough
    # views to cover (chunk - 1) overlapping windows of length num_views with
    # stride num_views - 1 (each chunk's first view overlaps the previous
    # chunk's last view).
    chunk = max(1, int(getattr(args, "chunk", 1)))
    if getattr(args, "anchor_k3v", False):
        # velocity-pinned anchoring: anchor stride is num_views-2
        total_num_views = 1 + chunk * (args.num_views - 2)
    else:
        total_num_views = args.num_views + (chunk - 1) * (args.num_views - 1)

    kwargs = dict(
        data_type='image2image',
        ROOT=args.data_root or entry["root"],
        resolution=args.height,
        num_views=total_num_views,
        debug=False,
    )
    cache = args.cache_path or entry.get("cache_path", "")
    if cache:
        kwargs["cache_path"] = cache
    kwargs.update(entry["extra"])
    if getattr(args, "min_interval", None) is not None:
        kwargs["min_interval"] = args.min_interval
    if getattr(args, "max_interval", None) is not None:
        kwargs["max_interval"] = args.max_interval
    if getattr(args, "test_sceneids_path", None):
        # Evaluate on the held-out scene list: keep exactly those scenes.
        # A '{dataset}' placeholder resolves per dataset, so one multi-
        # dataset launch can use per-dataset split files.
        kwargs["test_sceneids_path"] = args.test_sceneids_path.format(
            dataset=dataset_name)
        kwargs["split"] = "test"
    if args.control is not None:
        kwargs["control"] = args.control
    return cls(**kwargs)
