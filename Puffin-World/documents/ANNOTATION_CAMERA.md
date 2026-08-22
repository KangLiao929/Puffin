# 📷 Camera Annotation

Camera annotation runs the Puffin-World understanding branch (any config whose model
provides `model.understand`, e.g. the [understanding-only VLM
track](TRAINING.md#understanding-only-vlm-track) or a merged full model) over a
dataset's images and writes one JSON per image:

```json
{"roll": 0.031, "pitch": -0.118, "vfov": 0.982, "k1": 0.000, "parse_ok": true}
```

(angles in **radians**; the caption text itself is not stored).
Both annotators shard work across ranks (`torchrun` / `accelerate`), skip
already-written JSONs, and are therefore resumable and multi-node friendly.


## Sharded / generic datasets — `camera_caption_shards.py`

One script for every non-AOSS source, switched by `--type`:

| `--type` | source |
|---|---|
| `tar` / `zip` / `tgz` | local archive shards (webdataset-style), read in-memory without extracting |
| `img` | loose image files under `--data_root` (recursive) |
| `parquet` | online URL-list parquet shards (e.g. megalith-10m); images stream through the DataLoader workers, nothing but JSONs touch disk |

Archive/parquet shards are distributed across ranks; `--start_idx/--end_idx`
split the shard list across separate jobs; `--chunk_size` sets the rows per
parquet chunk. With `--pack` (default) each shard's JSONs are bundled into one
`.tar` mirror; `--no-pack` keeps them loose.

```shell
# local tar shards
torchrun --nproc_per_node=8 scripts/annotation/camera/camera_caption_shards.py \
    configs/pipelines/<vlm_or_full>.py --checkpoint <ckpt> \
    --type tar --data_root /data/.../my_dataset \
    --camera_root /data/.../my_dataset_camera

# online parquet URL dataset
torchrun --nproc_per_node=8 scripts/annotation/camera/camera_caption_shards.py \
    configs/pipelines/<full>.py --checkpoint <ckpt> \
    --type parquet --data_root /data/.../megalith-10m/data \
    --camera_root /data/.../camera/megalith-10m \
    --batch_size 128 --num_workers 32
```

## AOSS multi-view datasets — `camera_caption_aoss.py`

One dataset per run (`dl3dv / re10k / hypersim / mvs_synth / tartanair /
scannet / arkitscenes_highres`). It iterates images with **exactly the same
index the training dataloaders use** (the per-dataset cache pkl, see
[TRAINING.md, dataset index caches](TRAINING.md#before-you-start-dataset-index-caches-optional)),
and mirrors each dataset's AOSS layout under `--camera_root`, e.g. for DL3DV:

```
<camera_root>/<scene>/dense/camera/<frame>.json
```

```shell
torchrun --nproc_per_node=8 scripts/annotation/camera/camera_caption_aoss.py \
    configs/pipelines/<vlm_or_full>.py --checkpoint <ckpt> \
    --dataset dl3dv --camera_root /data/.../dl3dv_camera
```

Point the dataset's `camera_caption_root` (in
`configs/datasets/multi_view/gen_<dataset>.py`) at `--camera_root` and the
loaders pick the JSONs up as per-view `gt_cam_angles`.

**Sanity-check the propagation** with the built-in debug mode: sample N
scenes, take view-0's captioned (roll, pitch, vfov), propagate it to the other
views through the GT relative rotations (the same math as
`model._physical_propagation`), and overlay the resulting perspective fields
on the RGB frames:

```shell
python scripts/annotation/camera/camera_caption_aoss.py \
    configs/pipelines/<vlm_or_full>.py --checkpoint <ckpt> \
    --dataset dl3dv --camera_root /data/.../dl3dv_camera \
    --debug_only --debug_vis 5 --debug_dir output/dl3dv_pf_debug
```

## Statistics & visualization — `stat_camera_captions.py`

Recursively reads ALL caption JSONs under a `--camera_root`, converts to
degrees, and plots the roll / pitch / FoV distributions as a 1×3 histogram
figure (fractions of valid samples; fixed ranges roll/pitch ∈ [-45°,
45°], FoV ∈ [20°, 105°]; bin width via `--bin_width_deg`):

```shell
python scripts/annotation/camera/stat_camera_captions.py \
    --camera_root /mnt/.../camera_caption/mvs_synth \
    --output output/mvs_synth_cam_stats.png
```

Works on any annotator output that leaves **loose JSONs** on disk — all of
`camera_caption_aoss.py`, and `camera_caption_shards.py` when run with
`--no-pack` (or before `--remove_loose`); packed `.tar` bundles are not
scanned. JSON reading is multi-threaded (`--num_workers`).
