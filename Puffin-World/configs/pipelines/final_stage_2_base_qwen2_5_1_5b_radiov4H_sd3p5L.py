from mmengine.config import read_base
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from src.optimisers.custom_adamw import CustomAdamW

with read_base():
    from ..models.qwen2_5_1_5b_radiov4H_sd3p5L import model
    from ..datasets.basic.concat_datasets import train_dataloader, repeats

model.freeze_visual_encoder = False
model.freeze_llm = False
model.freeze_transformer = False
model.use_activation_checkpointing = True
model.freeze_projector=False
model.visual_encoder_grad_scale = 0.1
#model.pretrained_pth = 'work_dirs/final_stage_1_alignment_qwen2_5_1_5b_radiov4H_sd3p5L/model.pth'

# Scheduler & Optimizer
accumulative_counts = sum(repeats)
dataloader_num_workers = 4
max_iters = 22000 * accumulative_counts
optim_type = CustomAdamW
lr = 2e-5 * accumulative_counts
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.03

# Save
save_steps = 1000
save_start = 1000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)


# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=warmup_ratio * max_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=False,
        begin=warmup_ratio * max_iters,
        end=max_iters)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_iters=max_iters)

# configure default hooks
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        save_begin=save_start,
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
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
