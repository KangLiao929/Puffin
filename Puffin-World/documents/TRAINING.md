# ✈️ Training

Puffin-World is trained in four stages: two single-view stages that build the
unified physics understanding + spatial generation base (alignment → full SFT), followed by
two world model stages that turn it into a trajectory-controlled multi-view 3D 
generator (RGB → joint RGB + depth) with physics propagation (gravity field 
and latitude), while keeping the capabilities learned in the first two stages. 
All stages share one entry point:

```shell
cd /path/to/Puffin-World
export PYTHONPATH=./:$PYTHONPATH
bash scripts/train_ddp.sh configs/pipelines/<stage_config>.py
```

`scripts/train_ddp.sh` wraps `torchrun + scripts/train.py --deepspeed
deepspeed_zero2` and reads the usual cluster env vars
(`MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK / NPROC_PER_NODE`, all
defaulting to a single-process local run). Checkpoints and logs go to
`work_dirs/<config_name>/` (DeepSpeed ZeRO checkpoints `iter_*.pth`, saved
every `save_steps` iterations, `save_total_limit` kept).


## Before you start: dataset index caches (optional)

The multi-view dataloaders (stages III/IV) use a num_views-agnostic full-index
cache per dataset. Pre-build them once on the cluster (needs the real data
mounts / AOSS credentials), from the repo root:

```shell
python scripts/build_dataset_caches.py --force --jobs 7
```

A single cache works for any `num_views`; do not overwrite the cache of a job
that is still running on the old format.

**Where the caches are used.** Every multi-view dataset config
(`configs/datasets/multi_view/gen_*.py`) points at its cache via `cache_path`.
At training startup the dataset class loads it in `_load_data` instead of
re-scanning the remote storage (a full bucket walk that can take a long
time per dataset). The cache stores only the num_views-agnostic FULL per-scene
frame index (`scenes / sceneids / images / scene_img_list`, schema
`full_list`); the `num_views`-dependent window start positions are then derived
in memory from the current config
(`BaseMultiViewDataset._compute_start_img_ids`).

Example — `configs/datasets/multi_view/gen_dl3dv.py` reads:

```python
dataset = dict(type=CaptionDataset,
               ...
               cache_path=".../dataset_summary/dl3dv_cache.pkl",
               num_views=8,
               ...)
```

When Stage III starts (its data config imports `gen_dl3dv`), the dataloader
logs

```
[DL3DV_Multi] Loaded <n_scenes> scenes, <n_images> images from cache (full_list).
```

and computes the 8-view start windows on the fly. A 16-view experiment (e.g.
`concat_datasets_dl3dv_re10k_omni_puffin_cam_view16.py`, which only overrides
`num_views`) reuses the SAME `dl3dv_cache.pkl` with no rebuild.

## Stage I — Alignment

Aligns the modalities around the frozen backbones: the vision encoder, LLM,
and diffusion transformer are frozen; only the projector (understanding side)
and the learnable queries + connector that map LLM hidden states into
conditioning signals for the DiT (generation side) are trained. Data is the
single-view mix (`configs/datasets/basic/concat_datasets.py`) from [Puffin-Cam-15M](https://huggingface.co/datasets/KangLiao/Puffin-16M): captioned
images with camera parameters for both image2text and text/cam2image
directions.

```shell
bash scripts/train_ddp.sh configs/pipelines/final_stage_1_alignment_qwen2_5_7b_radiov3H_sd3p5M.py
```

Weights land in
`work_dirs/final_stage_1_alignment_qwen2_5_7b_radiov3H_sd3p5M/`.

## Stage II — Base SFT

Unfreezes everything except the VAE and fine-tunes the whole framework on the
same [Puffin-Cam-15M](https://huggingface.co/datasets/KangLiao/Puffin-16M). Point `model.pretrained_pth` at the Stage-I weights
(uncomment / edit the line in the config):

```python
model.pretrained_pth = 'work_dirs/final_stage_1_alignment_qwen2_5_7b_radiov3H_sd3p5M/model.pth'
```

```shell
bash scripts/train_ddp.sh configs/pipelines/final_stage_2_base_qwen2_5_7b_radiov3H_sd3p5M.py
```

The result
(`work_dirs/final_stage_2_base_.../model.pth`) is the full base model used to
initialize both world stages AND as the understanding donor when merging
checkpoints later (see [Checkpoint utilities](#checkpoint-utilities)).

Note: training saves DeepSpeed `iter_*.pth` directories — produce the
`model.pth` referenced between stages with `scripts/deepspeed2torch.py`
(see [Checkpoint utilities](#checkpoint-utilities)); `model.pretrained_pth`
also accepts an `iter_*.pth` directory directly.

## Stage III — World Modeling: trajectory-controlled 3D generation

Turns the base model into a multi-view 3D world generator. The understanding
backbone (vision encoder, LLM, projector) is frozen; only the diffusion
transformer (+ camera pathway) trains. Camera control uses the Omni-Camera
representation — absolute physics states anchoring to the world + per-view relative ray
maps — fused and added to the DiT input latent; `model.cam_inject_multi = True`
additionally re-injects the camera signal at deeper DiT blocks through
zero-init projections. RGB only in this
stage (`model.geometry_state = False`); `model.unconditional = 0.1` drops the
conditioning for CFG training.

The absolute physics states come from **physical propagation**, one knob with
three modes:

```python
model.physical_propagation = 'offline'   # training default
```

- `'offline'` — per-view perspective fields from Puffin-World-und annotations
  precomputed offline (the dataloader's `gt_cam_params`). Same propagation,
  just without running the VLM at train time, which is why training uses it.
- `'online'` — Puffin-World-und estimates the first frame at runtime and
  propagates it to all views via the relative poses.
- `'off'` — constant default perspective field (ablation).

The offline absolute-camera annotations used by Stages III and IV (labelled by
Puffin-World-und) are released on HuggingFace, one per source dataset:
[DL3DV-Absolute-Camera](https://huggingface.co/datasets/KangLiao/DL3DV-Absolute-Camera),
[RealEstate10K-Absolute-Camera](https://huggingface.co/datasets/KangLiao/RealEstate10K-Absolute-Camera),
[HyperSim-Absolute-Camera](https://huggingface.co/datasets/KangLiao/HyperSim-Absolute-Camera),
[MVS-Synth-Absolute-Camera](https://huggingface.co/datasets/KangLiao/MVS-Synth-Absolute-Camera),
[TartanAir-Absolute-Camera](https://huggingface.co/datasets/KangLiao/TartanAir-Absolute-Camera)
and
[ScanNet-Absolute-Camera](https://huggingface.co/datasets/KangLiao/ScanNet-Absolute-Camera).
Each mirrors the directory structure of its source dataset, so before training
download them alongside the raw data and point the dataset's
`camera_caption_root` (`configs/datasets/multi_view/gen_<dataset>.py`) at them
— the loaders read the per-view JSONs as `gt_cam_angles`, which is exactly
what `physical_propagation = 'offline'` consumes. This keeps training on the
fast `'offline'` path — no per-batch VLM forward, as `'online'` would require —
and also saves you from running the annotation pass yourself (GPU-days on the
large sources; see [ANNOTATION_CAMERA.md](ANNOTATION_CAMERA.md)).

Evaluation can override the trained setting:
`scripts/evaluation/generation_multi_view.py --physical_propagation
{offline,online,off}` (`online` needs usable understanding weights, e.g. a
merged checkpoint).

Two further conditioning / noise-schedule knobs:

- `model.uncond_pf = 0.15` — PF-drop: with this probability, zero the
  perspective-field channels but keep the ray maps, forcing camera motion to
  be read from the relative ray maps instead of the absolute PF (prevents PF
  dominance and the resulting translation collapse); it also trains a PF-null
  branch for optional PF-specific CFG at inference. Complements
  `model.unconditional = 0.1` (nulls ray + PF + text jointly — the CFG
  unconditional sample); no-op when `physical_propagation = 'off'`.
- `model.max_shift_override = 1.5` — biases the flow-matching timestep
  sampling of the multi-view path toward higher noise than the scheduler
  default: multi-view sequences are long, and the high-noise steps are where
  the cross-view layout and geometry are decided, so they get proportionally
  more training.

Data (`configs/datasets/multi_view/concat_datasets_dl3dv_re10k_omni_puffin_cam.py`):
DL3DV + RealEstate10K trajectories, [Puffin-Traj-1M](https://huggingface.co/datasets/KangLiao/Puffin-16M), and
[Puffin-Cam-15M](https://huggingface.co/datasets/KangLiao/Puffin-16M) mix. Initialized from Stage II
(`model.pretrained_pth = 'work_dirs/final_stage_2_base_.../model.pth'`).

```shell
bash scripts/train_ddp.sh configs/pipelines/final_stage_3_world_dl3dv_re10k_omni_cam_inject_qwen2_5_7b_radiov3H_sd3p5M.py
```

## Stage IV — World Modeling: joint RGB + depth with asymmetric attention

Adds geometry to the world model: `model.geometry_state = True` switches to
joint RGB + depth multi-view generation and reconstruction with asymmetric attention between the
two modality streams (physical propagation stays `'offline'` as in Stage III).

The asymmetry is the single knob `model.rgb_blind_to_depth = True`: depth
tokens attend to RGB (so depth follows the scene), but RGB (and text) tokens
never attend to depth — in training AND inference — so the colorized depth
stream can never leak into RGB appearance. A zero-init
`model.depth_modality_embed` gives depth tokens a learned identity (they share
their RGB view's position and view-id).

To keep the RGB quality of Stage III intact while depth
ramps up, the stage warm-starts from a Stage-III checkpoint and uses a gradual
transition: `model.depth_loss_split` ramps the depth-loss weight 0 → 1 over
`model.depth_loss_warmup_iters` (driven by `DepthTransitionHook`), and
`NoAdvanceTrainLoop` makes crash-resume instant (skips the dataloader
fast-forward, re-seeds the sampler for fresh data).

Data (`configs/datasets/multi_view/concat_datasets_8.py`): eight sources —
DL3DV, RealEstate10K, Hypersim, MVS-Synth, TartanAir, ScanNet, [Puffin-Traj-1M](https://huggingface.co/datasets/KangLiao/Puffin-16M), and
[Puffin-Cam-15M](https://huggingface.co/datasets/KangLiao/Puffin-16M) mix (depth-labelled sources feed the depth
branch).

**Dense depth labels.** For DL3DV and ScanNet, whose native depth is sparse, we
re-annotated dense depth with Depth Anything 3 (DA3) and aligned it to the
original sparse depth, so the dense maps stay consistent with the source
scene's geometry; Stage IV trains the depth branch on this aligned dense depth
(the other depth sources ship dense GT natively). The labels are released as
[DL3DV-Depth-DA3-Aligned](https://huggingface.co/datasets/KangLiao/DL3DV-Depth-DA3-Aligned)
and
[ScanNet-Depth-DA3-Aligned](https://huggingface.co/datasets/KangLiao/ScanNet-Depth-DA3-Aligned),
mirroring the source layouts. After downloading, point `depth_da3_root` in
`configs/datasets/multi_view/gen_dl3dv.py` / `gen_scannet.py` at them.

```python
model.pretrained_pth = 'work_dirs/final_stage_3_world_dl3dv_re10k_omni_cam_inject_.../model.pth'
```

```shell
bash scripts/train_ddp.sh configs/pipelines/final_stage_4_world_all_asym_attn_qwen2_5_7b_radiov3H_sd3p5M.py
```

## Checkpoint utilities

**DeepSpeed → plain torch.** Training saves DeepSpeed ZeRO checkpoints
(`iter_*.pth` directories). Convert one into a single loadable `.pth`:

```shell
python scripts/deepspeed2torch.py --input work_dirs/<exp>/iter_20000.pth --output work_dirs/<exp>/model_20k.pth
```

**Merge understanding + generation.** Stage III/IV freeze the understanding
backbone, so their checkpoints contain ONLY the trained generation weights —
loading one alone leaves the LLM/vision/projector at base weights and breaks
understanding. Rebuild a complete model by combining a full Stage-II
checkpoint (understanding donor) with the latest world model checkpoint (generation
donor):

```shell
python scripts/merge_und_gen_ckpt.py \
    --und_ckpt work_dirs/final_stage_2_base_.../model.pth \
    --gen_ckpt work_dirs/final_stage_4_world_all_asym_attn_.../iter_6300.pth \
    --output checkpoints/model.pth
```

Both inputs may be a plain `.pth` or a DeepSpeed `iter_*.pth` directory.

## Understanding-only VLM track

`configs/pipelines/vlm_qwen3_5_0_8b_radiov3H_stage_1.py / _stage_2.py` train a
lightweight understanding-only VLM (Qwen3.5-0.8B + RADIOv3-H) on the
[Puffin-Cam-15M](https://huggingface.co/datasets/KangLiao/Puffin-16M). It is trained with the same launcher as above
and is used where a small camera-understanding model suffices (e.g. dataset
annotation and evaluation).

Stage 1 aligns with both the visual encoder and the LLM frozen (only the
projector trains):

```shell
bash scripts/train_ddp.sh configs/pipelines/vlm_qwen3_5_0_8b_radiov3H_stage_1.py
```

Stage 2 fine-tunes end-to-end from the stage-1 weights (uncomment / edit the
line in the config):

```python
model.pretrained_pth = 'work_dirs/vlm_qwen3_5_0_8b_radiov3H_stage_1/model.pth'
```

```shell
bash scripts/train_ddp.sh configs/pipelines/vlm_qwen3_5_0_8b_radiov3H_stage_2.py
```

## Resuming

With `resume = True` in a config, every
launch auto-resumes from the latest `iter_*.pth` in the work_dir — model,
optimizer, lr schedule and iteration count — and cold-starts from
`pretrained_pth` when the work_dir is empty; combined with a restart loop this
makes long runs self-healing — no CLI flag needed.

Only when a SPECIFIC checkpoint is required (roll back to an earlier iter, the
latest save is corrupted, the checkpoint lives in another work_dir, or the
config ships `resume = False` and you don't want to edit it) pass it
explicitly to `train.py` — the CLI overrides the config, and the wrapper does
not forward extra args:

```shell
torchrun --nproc_per_node=8 scripts/train.py configs/pipelines/<stage_config>.py \
    --launcher pytorch --deepspeed deepspeed_zero2 --resume work_dirs/<exp>/iter_XXXX.pth
```