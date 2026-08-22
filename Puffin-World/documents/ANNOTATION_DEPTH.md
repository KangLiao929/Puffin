# 🌊 Depth Annotation (DA3-aligned dense depth)

`scripts/annotation/depth/annotate_da3_sparse_depth.py` reproduces the
**DA3-aligned dense-depth labels** used by Puffin-World (the `depth_da3` npys, e.g.
`da3_anno/dl3dv_da3`, `da3_anno/scannet_da3`). For every frame it:

1. reads the RGB and the dataset's **native sparse/imperfect depth**,
2. predicts monocular dense depth with **Depth Anything 3 (DA3)**,
3. fits a robust **scale + shift** on valid sparse pixels so the DA3 prediction
   lives in the dataset's metric/COLMAP scale,
4. writes one `float32 .npy` depth map per frame, mirroring the source layout.

```
dl3dv:    <save_root>/<part>/<scene_hash>/dense/depth_da3/<frame>.npy
scannet:  <save_root>/scans[_test]/<scene>/depth_da3/<frame>.npy
```

We release our captioned datasets in 🤗 [DL3DV-Depth-DA3-Aligned](https://huggingface.co/datasets/KangLiao/DL3DV-Depth-DA3-Aligned)
and 🤗 [ScanNet-Depth-DA3-Aligned](https://huggingface.co/datasets/KangLiao/ScanNet-Depth-DA3-Aligned).

---

## 1. Environment (fresh `da3` conda env)

DA3 needs a newer `torch` stack than the training env; keep it isolated:

```bash
conda create -n da3 python=3.10 -y
conda activate da3

# 1) DA3 (pulls torch 2.13+cu130 by default)
pip install depth-anything-3          # tested: 0.1.1

# 2) pin torch to the cluster's CUDA 12.x driver
pip install "torch==2.7.0" "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu126

# 3) sanity
python -c "import torch, depth_anything_3; \
           print(torch.__version__, torch.cuda.is_available())"
# expect: 2.7.0+cu126 True
```

### Model weights

`--model` accepts either a **HF model id** or a **local weights directory**:

| option | how |
|---|---|
| `depth-anything/da3-large` | `HF_HUB_OFFLINE=0` once (proxy on) to cache, offline afterwards |
| `DA3-GIANT-1.1` (default) | expects a sibling checkout `<repo>/../depth-anything-3/develop/weights/DA3-GIANT-1.1`; override with `--model /path/to/weights` |

Registry names: `da3-small/base/large/giant`, `da3metric-large`,
`da3mono-large`, `da3nested-giant-large`. A source checkout can be pointed to
with `DA3_CODE_ROOT=/path/to/depth-anything-3`.

---

## 2. Input layout

The script reads **local mirrors** (not AOSS). Defaults: `--root data/<dataset>`.

```
dl3dv (per scene, all REQUIRED):
  <root>/<part>/<scene_hash>/dense/rgb/frame_XXXXX.png
  <root>/<part>/<scene_hash>/dense/depth/frame_XXXXX.npy      # sparse depth
  <root>/<part>/<scene_hash>/dense/sky_mask/frame_XXXXX.png   # ≥127 = sky
  <root>/<part>/<scene_hash>/dense/outlier_mask/frame_XXXXX.png

scannet:
  <root>/scans[_test]/<scene>/color/<frame>.jpg
  <root>/scans[_test]/<scene>/depth/<frame>.png               # uint16 mm → /1000
```

- dl3dv: RGB is resized to the depth resolution; `sky ∪ outlier` counts as
  invalid and becomes fill region.
- scannet: `depth==0` pixels are the fill region; `--folders scans,scans_test`
  selects the top folders.

## 3. Usage

```bash
conda activate da3
cd Puffin-World   # repo root (the script auto-inserts it into sys.path)

# dl3dv, 8 GPUs, default fill_sparse output
env -u http_proxy -u https_proxy HF_HUB_OFFLINE=1 \
python scripts/annotation/depth/annotate_da3_sparse_depth.py \
    --dataset dl3dv \
    --root /path/to/processed_dl3dv_ours \
    --save-root /path/to/da3_anno/dl3dv_da3 \
    --model depth-anything/da3-large \
    --gpus 0,1,2,3,4,5,6,7 --batch-size 8 --process-res 896

# scannet (train + test scans)
python scripts/annotation/depth/annotate_da3_sparse_depth.py \
    --dataset scannet --folders scans,scans_test \
    --root /path/to/scannet --save-root /path/to/da3_anno/scannet_da3 \
    --model depth-anything/da3-large --gpus 0,1,2,3

# debug a single local scene (no discovery walk)
python scripts/annotation/depth/annotate_da3_sparse_depth.py \
    --dataset dl3dv --local-scene /path/to/1K/<scene_hash>/dense \
    --save-root /tmp/da3_dbg --limit-frames 4 --batch-size 2
```

### All arguments

| arg | default | meaning |
|---|---|---|
| `--dataset` | (required) | `dl3dv` \| `scannet` |
| `--root` | `data/<dataset>` | local source root |
| `--save-root` | `data/depth_da3/<dataset>` | output root (layout mirrored) |
| `--model` | `DA3-GIANT-1.1` | HF id or local weights dir |
| `--folders` | `scans` | ScanNet top folders (csv) |
| `--process-res` | 896 | DA3 processing resolution (upper-bound resize) |
| `--batch-size` | 8 | frames per forward (bucketed by resolution) |
| `--gpus` | `0` | csv GPU ids → one worker process per GPU |
| `--scene-list` | — | text file, one scene per line (subset/sharding) |
| `--num-scenes` | — | cap the number of scenes |
| `--limit-frames` | — | cap frames per scene (debug) |
| `--local-scene` | — | single scene/dense dir (debug, skips discovery) |
| `--overwrite` | off | recompute even if the output npy exists |
| `--reverse` | off | process scene list back-to-front (2-job ends-meet) |
| `--stop-on-done` | off | stop a reverse job when it meets finished work |
| `--align-pct-clip` | 98.0 | drop sparse pixels above this percentile before fitting |
| `--min-depth` | 1e-3 | clamp floor of the aligned depth |
| `--output-mode` | `fill_sparse` | see below |

### Output modes

- **`fill_sparse`** (default, reproduces Puffin2 labels): valid sparse pixels
  are kept **unchanged**; the aligned DA3 prediction only fills sparse holes,
  sky and outlier regions.
- **`aligned_da3`**: the scale/shift-aligned DA3 prediction everywhere
  (smooth dense map, no sparse pass-through).

### Alignment details

`s, t = argmin Σ‖s·pred + t − sparse‖²` on valid sparse pixels, made robust by:
p98 percentile clip on sparse (`--align-pct-clip`) → up to 5 IRLS iterations
with MAD-based inlier gating (`|res| < 3 × 1.4826 × MAD`) → needs ≥32 points,
otherwise falls back to `s=1, t=0` (fallbacks are counted in the per-scene log
line as `fallback=`). Output is clamped at `--min-depth`.

### Resume / multi-GPU / failures

- Already-written npys are **skipped** (resumable) unless `--overwrite`.
- One worker per GPU id; scenes are interleaved (`scenes[rank::n]`).
- Per-frame failures are logged and skipped; failing scenes go to
  `<save_root>/.failed_scenes.rankN.txt`, merged into
  `<save_root>/failed_scenes.txt` at the end.

