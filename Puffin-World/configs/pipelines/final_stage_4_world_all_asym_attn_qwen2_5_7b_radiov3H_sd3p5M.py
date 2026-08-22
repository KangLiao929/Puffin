from mmengine.config import read_base
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from src.runners.no_advance_train_loop import NoAdvanceTrainLoop
from src.optimisers.custom_adamw import CustomAdamW
from src.engine.depth_transition_hook import DepthTransitionHook

with read_base():
    from ..models.qwen2_5_7b_radiov3H_sd3p5M import model
    from ..datasets.multi_view.concat_datasets_8 import train_dataloader, repeats

model.freeze_visual_encoder = True
model.freeze_llm = True
model.freeze_projector = True
model.freeze_transformer = False
model.use_activation_checkpointing = True

model.initial_view_num = 1
model.max_view_num = 8
model.geometry_state = True
# Physical propagation -- the perspective-field (PF) source:
#   'offline': per-view PF from Puffin-World-und annotations precomputed
#              OFFLINE (the dataloader's gt_cam_params). Same propagation,
#              without running the VLM at train time for saving time.
#   'online' : Puffin-World-und estimates frame 0 at runtime, then
#              propagates it to all views via the relative poses.
#   'off'    : constant default PF.
model.physical_propagation = 'offline'

model.cam_inject_multi = True
model.unconditional = 0.1
model.uncond_pf = 0.15
model.max_shift_override = 1.5

model.pretrained_pth = 'work_dirs/final_stage_3_world_dl3dv_re10k_omni_cam_inject_qwen2_5_7b_radiov3H_sd3p5M/model_20k.pth'

model.rgb_blind_to_depth = True # False = full bidirectional attention
model.depth_loss_split = True
model.depth_loss_warmup_iters = 2000
model.depth_loss_weight_max = 1.0
model.depth_modality_embed = True
custom_hooks = [dict(type=DepthTransitionHook)]

# Scheduler & Optimizer
accumulative_counts = sum(repeats)  
max_iters = 3000 * accumulative_counts
optim_type = CustomAdamW
lr = 2e-5
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_iters = 2000

# Save
save_steps = 400
save_total_limit = 1

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    paramwise_cfg=dict(custom_keys={
        'depth_modality_embed': dict(lr_mult=4.0),
    }),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=warmup_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=False,
        begin=warmup_iters,
        end=max_iters)
]

# train, val, test setting.
train_cfg = dict(type=NoAdvanceTrainLoop, max_iters=max_iters)

# configure default hooks
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# other settings
visualizer = None
log_level = 'INFO'
load_from = None
resume = True
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
