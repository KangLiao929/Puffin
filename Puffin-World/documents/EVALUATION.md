# 🖼️ Evaluation

This document describes how to evaluate **Puffin-World** on the two single-view
capabilities: **camera understanding** (physical-world perception) and
**free-viewpoint spatial simulation** (camera-controllable image generation).
Both evaluations are multi-GPU–ready through `accelerate`, and both re-use
**GeoCalib** to score the recovered camera geometry.

All commands are run from the repository root with the model conda environment
(e.g. `Puffin-World`) activated:

```bash
export PYTHONPATH=./
```

Checkpoints follow the training layout: `work_dirs/<EXP_NAME>/model.pth`
or the latest `work_dirs/<EXP_NAME>/iter_*.pth`. The config is
`configs/pipelines/<EXP_NAME>.py`.

---

## Camera Understanding

Puffin-World estimates the absolute camera state of a single image — roll, pitch,
and vertical field-of-view (vFoV) — by *reasoning about the
scene* and emitting the parameters as text. Evaluation is a two-phase process:
**(1) predict** the camera parameters with the model, then **(2) score** them against
the ground truth with GeoCalib.

### Phase 1 — predict

```bash
accelerate launch scripts/evaluation/understanding.py \
    configs/pipelines/<EXP_NAME>.py \
    --checkpoint work_dirs/<EXP_NAME>/model.pth \
    --image_dir <DATA_ROOT>/<dataset>/images \
    --image_size 640 \
    --batch_size 8 \
    --output output/<EXP_NAME>_<dataset>.json
```

Each image is padded to a square and resized so that its longest edge is
`--image_size`. The model is prompted to describe the scene and then estimate the
camera parameters; the raw response is stored per image in the output JSON as a
list of `{"id": <fname>, "output_text": <str>}`.

| Argument        | Default                | Description                                              |
|-----------------|------------------------|----------------------------------------------------------|
| `config`        | (required)             | pipeline config, `configs/pipelines/<EXP_NAME>.py`       |
| `--checkpoint`  | `None`                 | checkpoint (`guess_load_checkpoint`, `.pth` or DeepSpeed)|
| `--image_dir`   | stanford2d3d           | folder of test images (`<DATA_ROOT>/<dataset>/images`)   |
| `--image_size`  | `640`                  | target size of the longest edge                          |
| `--ratio`       | `None`                 | optional central crop, e.g. `1_1`, `16_9`, `3_4`, `4_3`  |
| `--batch_size`  | `4`                    | per-GPU batch size                                        |
| `--thinking`    | off                    | use the *thinking* prompt (spatial reasoning first)      |
| `--output`      | `output`               | output JSON path                                         |

### Phase 2 — score (GeoCalib)

#### GeoCalib environment installation

The scorer runs in its own `geocalib` environment (**Python 3.10, PyTorch
2.5.0, CUDA 12.4** — a different torch stack from the main `puffin-world`
env). `dataset/generation/GeoCalib/requirements.txt` pins the exact working
versions; install in this order (torch first, editable installs last):

```bash
conda create -n geocalib python=3.10 -y
conda activate geocalib

# 1. torch stack (cu124 index)
pip install torch==2.5.0 torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. pinned dependencies (kornia, opencv, pytorch-fid, ...)
cd dataset/generation/GeoCalib
pip install -r requirements.txt

# 3. the vendored geocalib + siclib packages, editable
pip install -e .
pip install -e siclib
```

Sanity check: `python -c "import geocalib, siclib, kornia; print('ok')"`.

Switch to the `geocalib` environment and score the predictions against the GT CSV
by running the evaluator **as a module inside** `dataset/generation/GeoCalib`:

```bash
conda activate geocalib
cd dataset/generation/GeoCalib
python -m siclib.eval.eval_understanding \
    --input_json  <repo>/output/<EXP_NAME>_<dataset>.json \
    --gt_csv      <DATA_ROOT>/<dataset>/images.csv \
    --output_dir  <repo>/output/<EXP_NAME>_<dataset>_eval \
    --thresholds  1 5 10 \
    --advance_metrics
```

This reports the **median error** and **AUC @ 1°/5°/10°** for roll, pitch, and
vFoV, and (with `--advance_metrics`) the perspective-field metrics (up-vector,
gravity, and latitude). A `summary_metrics.txt` is written per dataset. The GT
CSV (`images.csv`) has columns `fname, roll, pitch, vfov, [k1], width, height`.

### Benchmarks

We evaluate on four public benchmarks, each laid out as
`<DATA_ROOT>/<dataset>/{images/, images.csv}`:

- [MegaDepth](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/megadepth2k.zip)
- [TartanAir](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/tartanair.zip)
- [LaMAR](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/lamar2k.zip)
- [Stanford2D3D](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/stanford2d3d.zip)

---

## Free-Viewpoint Spatial Simulation

Given a text prompt and a target camera, Puffin-World generates an image whose
realized viewpoint and intrinsics adhere to the specified camera. We evaluate on
**Puffin-World-Gen** (650 text–camera specification pairs spanning diverse scenes,
viewpoints, poses, FoVs, and aspect ratios).

### Phase 1 — generate

```bash
accelerate launch scripts/evaluation/generation.py \
    configs/pipelines/<EXP_NAME>.py \
    --checkpoint work_dirs/<EXP_NAME>/model.pth \
    --prompt_path <PROMPT_ROOT> \
    --num -1 \
    --cfg_scale 4.5 \
    --num_steps 50 \
    --seed 42 \
    --output output/<EXP_NAME>_gen
```

Each prompt is a JSON with a `caption` and (optionally) a target `height`/`width`;
the target camera is built from the caption by `Cam_Generator` (radial model) and
converted into the Omni-Camera condition. The script writes one PNG per prompt,
named by its sample id.

| Argument        | Default | Description                                                     |
|-----------------|---------|-----------------------------------------------------------------|
| `config`        | (required) | pipeline config, `configs/pipelines/<EXP_NAME>.py`           |
| `--checkpoint`  | `None`  | checkpoint (`guess_load_checkpoint`)                            |
| `--prompt_path` | —       | folder of prompt `*.json` files (caption + optional H/W)        |
| `--num`         | `100`   | number of prompts to sample; `-1` = use all                     |
| `--cfg_scale`   | `4.5`   | classifier-free guidance scale                                  |
| `--num_steps`   | `50`    | flow-matching sampling steps                                    |
| `--cfg_prompt`  | `""`    | negative/unconditional prompt for CFG                           |
| `--batch_size`  | `1`     | per-GPU batch size                                              |
| `--seed`        | `42`    | RNG seed                                                        |
| `--output`      | `output`| output image folder                                            |

> Sizes must be divisible by 16; prompts carrying an incompatible embedded size
> (e.g. 426/360) are filtered out, and prompts without a size fall back to a random
> valid resolution.

### Phase 2 — score (camera controllability + fidelity)

Camera controllability is measured by re-estimating the camera of each **generated**
image and comparing it to the **target** camera used for generation:

1. Run **Camera Understanding** (above) on `output/<EXP_NAME>_gen` to recover the
   camera parameters of every generated image.
2. Derive the pixel-wise perspective fields and score them against the target
   fields with GeoCalib's `siclib.eval.eval_understanding`, reporting the mean and
   median angular errors of the **up-vector**, **gravity**, and **latitude** maps.
3. Report **FID** against the reference distribution to assess visual fidelity and
   realism.

---

## Notes

- **Multi-GPU.** Both scripts use `accelerate`; launch with `accelerate launch`
  (or plain `python` for a single GPU). Results are gathered across processes.
- **Two environments.** The model runs in the `Puffin-World` env; GeoCalib scoring runs
  in the `geocalib` env as a module inside `dataset/generation/GeoCalib`.
