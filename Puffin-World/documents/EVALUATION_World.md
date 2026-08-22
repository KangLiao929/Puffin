# 🌍 World Modeling Evaluation

This document describes `scripts/evaluation/generation_multi_view.py`, the
evaluation entry for the **world model stages**: given ONE initial view and a
camera trajectory, the model generates the remaining views (Stage III: RGB;
Stage IV: RGB + depth jointly). One script covers both stages — the extra
Stage IV outputs (depth maps, 3D reconstruction) appear automatically when
the config enables `model.geometry_state`.

```shell
cd /path/to/Puffin-World
export PYTHONPATH=./:$PYTHONPATH
```

**Single-sample demo**: `scripts/demo/world_modeling.py` wraps the same
pipeline as a one-sample, model-by-NAME, single-GPU demo (registry:
`Puffin-World` = Stage IV 1.5b asym, `Puffin-World-7B`, `Puffin-World-RGB`
= Stage III). Its `--geometry off` runs a Stage IV checkpoint RGB-only:
with `rgb_blind_to_depth=True` the RGB/text tokens never attend depth
tokens, so dropping the depth block leaves the RGB **attention pathway**
unchanged (no train/test conflict) while skipping the depth/.glb
artifacts (~2x faster). The sampler's dynamic timestep shift does adapt
to the halved token count (by design, sequence-length-dependent
scheduling; verified visually equivalent on re10k — pin it with the
model's `max_shift_override` if exact parity is wanted). For checkpoints
trained with bidirectional RGB-depth attention the equivalence does NOT
hold and the demo warns.

---

## Stage III — RGB 3D world model

### Basic run

```shell
python scripts/evaluation/generation_multi_view.py \
    configs/pipelines/final_stage_3_world_dl3dv_re10k_omni_cam_inject_qwen2_5_7b_radiov3H_sd3p5M.py \
    --checkpoint work_dirs/final_stage_3_world_dl3dv_re10k_omni_cam_inject_qwen2_5_7b_radiov3H_sd3p5M/iter_XXXX.pth \
    --dataset re10k \
    --num_views 8 \
    --num_test_samples 5 \
    --output output/world_stage3
```

- `config` is the **training** pipeline config (the model is rebuilt from it);
  `--checkpoint` accepts a merged `.pth` file or a DeepSpeed `iter_*.pth`
  directory.
- `--dataset` takes one or more of `re10k / dl3dv / hypersim / mvs_synth /
  tartanair / puffin_omni`. With several datasets the weights are loaded once
  and each dataset writes to `<output>_<dataset>`. Per-dataset roots, index
  caches, frame intervals and the `camera_caption_root` (offline VLM camera
  annotations) are pre-wired in
  `scripts/evaluation/modules/datasets.py::DATASET_REGISTRY`; override with
  `--data_root / --cache_path / --min_interval / --max_interval`.
- `--num_test_samples` random scenes are drawn reproducibly from `--seed`.
- Generation knobs: `--cfg_scale` (default 4.5), `--num_steps` (50),
  `--height/--width` (512), `--gif_fps` (4).

### Perspective-field source (`--physical_propagation`)

Overrides the training config's physical-propagation mode at eval time,
without retraining:

| value | meaning |
|---|---|
| `config` (default) | keep whatever the training config used |
| `offline` | per-view PF from the precomputed Puffin-World-und annotations (`camera_caption_root`) |
| `online` | run the und branch on frame 0 and propagate via relative poses (needs usable VLM weights, e.g. a merged ckpt) |
| `off` | constant default PF |

`--cam_cfg fix` (default) nulls the camera on the CFG unconditional branch so
`--cfg_scale` amplifies camera-following; `legacy` reproduces the old
behavior for A/B comparison.

### Per-sample outputs

Each scene writes to `<output>/<scene_id><mode_suffix>/`:

| file | content |
|---|---|
| `00_input_gt.png` / `00_novel_view_gen.png` | the input anchor view: clean original / with the keyboard panel for the motion to frame 1 |
| `NN_target_gt.png` / `NN_novel_view_gen.png` | per-view GT / generated RGB |
| `target_gt.gif` / `novel_view_gen.gif` | GT vs. generated playback, frame-aligned (both start at the anchor) |
| `novel_view_gen.mp4` | mp4 twin of the generated-RGB playback (H.264; `--mp4_fps`, default = `--gif_fps`). Depth and PF panels get mp4 twins too (`novel_view_depth.mp4`, `pf_vis/pf_{up,lat}.mp4`) |
| `camera_trajectory.png` | 3D render of the poses the model conditioned on: rainbow wireframe frustums along the path + an offset red trajectory line with an endpoint arrowhead |
| `pf_vis/` | per-view **up-field / latitude** overlays (`NN_pf_up.png`, `NN_pf_lat.png`, and `NN_pf_mix.png` — latitude upper-left / up-field lower-right, split seamlessly along the top-right→bottom-left diagonal) + `pf_{up,lat,mix}.{gif,mp4}`, built from the SAME offline annotations as `physical_propagation='offline'` — including frame 0 |
| `combined.gif` / `combined.mp4` | RGB \| up \| latitude side by side (Stage IV adds a depth panel); the mp4 twin is composed from the original frames, never from the palette-quantized GIF |
| `vlm_prediction.json`, `perspective_fields/` | only with `--physical_propagation online` (the VLM text + the PF the model actually used) |

### Keyboard-control visualization (on by default)

Every generated RGB PNG — including the frame-0 anchor `00_input_gt.png` —
and every GIF/mp4 built from them carries a Genie-style key panel showing
the motion **to the next frame** (the final frame shows the idle panel; GT
outputs stay clean). Disable with `--no_keyboard_vis`.

- **Bottom-left, blue — rotation**: arrow keys (`↑↓` pitch, `←→` yaw) plus
  two curved-arrow keycaps flanking `↑` for roll (left = CCW, right = CW,
  following the on-screen content rotation).
- **Bottom-right, red — translation**: `W/S` forward/backward, `A/D` strafe,
  plus two triangle keycaps flanking `W` for vertical flight (descend /
  ascend).
- Motions are decomposed as **gravity-referenced** Euler rates (azimuth /
  elevation / gravity-roll) rather than raw camera-axis rates, so a pan held
  at a tilted or rolled pose still lights exactly one key — the command, not
  its camera-frame smear.
- A dimension only lights up when it exceeds 30% of the sequence's dominant
  motion (plus a 0.3°/view absolute floor for rotation), so negligible
  jitter stays dark — matching the "don't visualize tiny differences" rule.

### Chunked autoregressive inference (`--chunk`)

`--chunk C` (default 1) chains C windows of `--num_views` views with a
1-frame overlap, extending the trajectory far beyond one window:

- The dataset automatically fetches `num_views + (C-1) * (num_views-1)`
  views total (e.g. `--num_views 8 --chunk 4` → 29 views).
- Each window's ray maps are rebuilt relative to its own first view; from
  chunk 1 on, the anchor is the previous chunk's last generated view — its
  **latent** is handed over directly (no VAE round-trip, avoiding compounding
  blur).
- Physical propagation with `online` runs the VLM **once**, on chunk 0's
  anchor: the model returns the propagated per-view absolute params, and the
  script feeds each window's LAST-view params back in as the next window's
  anchor (`pp_anchor_params`), so later chunks skip the VLM entirely and the
  single chunk-0 estimate propagates through the whole trajectory — no
  re-estimation on generated pixels. With `offline`, the per-view
  annotations are simply sliced per window and the VLM never runs.
- All Stage III/IV outputs (GIFs, pf_vis, keyboard panels, depth, .glb) work
  transparently over the stitched timeline; the keyboard overlay uses the
  effective per-chunk poses, so combo motions (below) are labeled correctly.
- Both trajectory replacements compose with chunking: `--combo_traj` assigns
  one motion PER chunk, while `--cus_traj` builds ONE continuous synthetic
  trajectory over the whole timeline and lets the AR windows walk it.

```shell
# 4-chunk autoregressive run on the GT trajectory
python scripts/evaluation/generation_multi_view.py <config> \
    --checkpoint <ckpt> --dataset re10k --num_views 8 --chunk 4 \
    --output output/world_stage3_chunk4
```

### Custom synthetic trajectories (`--cus_traj`)

Replaces the dataset camera with synthetic motions to probe individual
control dimensions. Per sample, `--cus_traj N` randomly picks N combos from
the pool selected by `--cus_traj_type`; each pick runs a full pass and writes
to `<scene_id>_cus<ii>_<combo>/`:

| `--cus_traj_type` | pool | combo syntax |
|---|---|---|
| `all` (default) | 24 | rotation + translation, e.g. `r+tl`, `p-tf`, `y-tb` |
| `only_rot` | 6 | pure rotation, e.g. `r+`, `y-` |
| `only_trans` | 4 | pure translation, e.g. `tf`, `tl` |
| `360` | 2 | yaw full-circle orbit (`y+360`, `y-360`) |

Tokens: `r/p/y` = roll/pitch/yaw, `+/-` = direction, `t<f|b|l|r>` =
translate forward/back/left/right. View t rotates by `sign * step * t` and/or
translates by `step_dist * t`. Steps: `--cus_step_deg` (default 2°/view;
`360` type defaults to a uniform full-circle orbit over the whole generated
timeline) and `--cus_step_dist` (default 2.0, RE10K-scale units).

Combining with `--chunk > 1` builds ONE continuous synthetic trajectory over
all `num_views + (chunk-1)*(num_views-1)` views and splits it into AR
windows.

```shell
# 3 random pure-rotation probes per sample (single window)
python scripts/evaluation/generation_multi_view.py <config> \
    --checkpoint <ckpt> --dataset puffin_omni \
    --cus_traj 3 --cus_traj_type only_rot --output output/probe_rot
```

### Combo (compound) trajectories (`--combo_traj`, chunked only)

Injects in-place **go-and-return rotations** into chosen chunks while every
other chunk follows the dataset's original GT trajectory — "walk, turn and
look, walk on". Specs are given inline (no predefined trajectory file);
requires `--chunk > 1`.

- Spec format `<chunk><r|p|y><+|->`, 1-based chunk index. E.g.
  `--combo_traj 2r+ 4p- 6y-`: the 2nd chunk rolls + and back, the 4th chunk
  pitches − and back, the 6th chunk yaws − and back; chunks 1/3/5/7/…
  follow the GT path. Each rotation rises at a constant step per view to its
  peak at the window middle and returns to the anchor by the window's last
  view, so the trajectory always resumes exactly where it left off.
- `--combo_deg` sets each spec's per-view step in degrees and must match the
  spec count exactly (e.g. `--combo_deg 1 2 3`); default is 4°/view for
  every spec. Peak deviation = `deg * (num_views-1)/2`.
- `--combo_pause M` (default 1.0 s): every frame-aligned GIF (RGB, GT,
  depth, pf_vis, combined) holds each rotation's maximum-deviation frame for
  M seconds, so the peak view is easy to inspect.
- The single CLI combo applies to every sample; outputs go to
  `<scene_id>_combo_<specs>/` with the resolved per-chunk motions written to
  `combo.txt` (e.g. `t, r+(4 deg/view), t, p-(2 deg/view), ...`).

```shell
# 8 chunks: rotate in place at chunks 2/4/6 (1/2/3 deg per view), GT elsewhere
python scripts/evaluation/generation_multi_view.py <config> \
    --checkpoint <ckpt> --dataset dl3dv --num_views 8 --chunk 8 \
    --combo_traj 2r+ 4p- 6y- --combo_deg 1 2 3 --combo_pause 1.5 \
    --output output/world_stage3_combo
```

---

## Stage IV — joint RGB + depth

Everything above applies unchanged (chunking, keyboard panels, custom /
combo trajectories, physical propagation). The differences:

### Run

```shell
python scripts/evaluation/generation_multi_view.py \
    configs/pipelines/final_stage_4_world_all_asym_attn_qwen2_5_7b_radiov3H_sd3p5M.py \
    --checkpoint work_dirs/final_stage_4_world_all_asym_attn_qwen2_5_7b_radiov3H_sd3p5M/iter_XXXX.pth \
    --dataset dl3dv --num_views 8 --output output/world_stage4
```

- The Stage IV configs enable `model.geometry_state = True`, which switches
  on all depth artifacts below — no extra flag needed.
- `--depth_attn_closed` keeps the RGB←depth attention isolation CLOSED at
  inference (RGB queries never attend to depth tokens; depth still follows
  RGB). Use it with the asymmetric-attention checkpoints trained with
  `model.rgb_blind_to_depth = True`.
- Chunked runs align depth scales ACROSS chunks by default: each chunk
  regenerates the overlap view's depth under its own scale gauge, and the
  median ratio against the previously kept copy of that SAME frame rescales
  the whole chunk onto chunk 0's scale (the overlap-frame relay, like
  `pp_anchor_params` for the PF). Without it the fused .glb splits into one
  shell per chunk. On top of the median-ratio relay, the factor is refined
  by the same cross-view reprojection objective used at export time (below)
  against the last kept views — the single-frame median pins the overlap
  frame itself but can leave a few-percent jump on the first NEW views of a
  chunk, which the refinement absorbs. Printed as
  `[depth-align] chunk k: s=... refined=...`; the raw factor is itself a
  per-checkpoint drift measurement. `--no_depth_align_chunks` disables the
  relay entirely, `--glb_no_align` keeps the relay but skips the
  refinement.

### Additional outputs

| file | content |
|---|---|
| `NN_input_depth.png` / `NN_novel_view_depth.png` | per-view generated depth in the dataloader's **visionbanana** encoding (scalar depths are re-encoded the same way, so both decode paths look identical) |
| `novel_view_depth.gif` | depth playback at `--gif_fps` |
| `NN_gt_depth.png` | GT visionbanana depths for comparison |
| `combined.gif` | now four panels: RGB \| up \| latitude \| depth |
| `reconstruction.glb` | fused colored point cloud (view-0 GT + generated views, generated depths, run poses), viewable in any glTF viewer; camera frusta are omitted by default (the trajectory lives in camera_trajectory.png) |
| `reconstruction_gt.glb` | reference cloud from GT frames + GT depths on the GT trajectory |

### 3D reconstruction export

#### Depth-to-pose gauge alignment (on by default)

Generated depths live in the dataloader's **normalized, sequence-relative
units**, while the dataset's camera translations are in **per-scene SfM
units**. Whenever the two gauges differ (dataset-dependent; most visible on
re10k-style scenes), raw fusion offsets each view's backprojection along the
trajectory by an amount proportional to its baseline — the cloud splits into
parallel "multi-shell" copies of the same surface even though every per-view
depth map is smooth.

As a visualization post-process, the export therefore solves **one global
depth-to-pose scale per window** before unprojection: a coarse-to-fine log
search for the factor that minimizes cross-view reprojected-depth
disagreement (median `|log z_pred / (s·d_ref)|` over view pairs). The solve
is a guarded no-op — pure-rotation windows (no baseline, nothing to align),
windows with too few cross-view correspondences, and windows whose gauges
already agree all pass through with `s = 1` — so it never disturbs runs that
don't need it. The factor is printed as `[glb-align] ... s=...`; on
re10k-style scenes it typically lands around 2–3.5 and collapses the shells
into a single coherent surface (measured cross-view misalignment drops
1.6–10×). This only rescales the fused cloud for display — the depth
PNGs/GIFs and every other output are untouched. Disable with
`--glb_no_align` to reproduce the raw fusion.

Note when re-fusing offline from saved outputs: build the cloud from the
CLEAN frames (`NN_target_gt.png` anchor + in-memory generated tensors), not
from the saved generated PNGs — those carry the keyboard-panel overlay and
would bake the keycaps into the point colors. The pipeline itself always
fuses the clean tensors.

#### Denoising stages

The `.glb` fusion then runs three denoising stages, each with its own flag
(`--no_glb` disables the export entirely):

| flag | default | stage |
|---|---|---|
| `--glb_no_align` | off (align on) | disable the global depth-to-pose gauge solve above |
| `--glb_grad_thresh` | 0.05 | flying-pixel filter: drop pixels near relative depth discontinuities before unprojection (0 = off) |
| `--glb_consistency` | 0 (off) | cross-view vote: keep a point only if a neighboring view (±2) agrees within this relative depth tolerance (e.g. `0.08`); strong ghost/double-surface removal and a consistency probe |
| `--glb_voxel_res` / `--glb_voxel_min_pts` | 512 / 2 | voxel denoise + mean-downsample: drop isolated specks, average survivors (0 = off) |
| `--glb_max_points` / `--glb_depth_percentile` | 1M / 98 | point budget / per-view far cutoff |

Tip: with gauge alignment on, systematic shells are already handled; if
residual double surfaces remain (per-view depth-shape scatter rather than a
shared scale), add `--glb_consistency 0.08` — how many points it removes is
itself a measure of the generated depths' cross-view consistency.
