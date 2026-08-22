from mmengine.config import read_base
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from src.optimisers.custom_adamw import CustomAdamW

with read_base():
    from ..models.qwen3_5_0_8b_radiov3H import model
    from ..datasets.basic.image2text import train_dataloader, repeats

model.freeze_visual_encoder = True
model.freeze_llm = True
model.freeze_projector = False
model.use_activation_checkpointing = True

# Scheduler & Optimizer 
accumulative_counts = sum(repeats)
dataloader_num_workers = 4
max_iters = 18000 * accumulative_counts
optim_type = CustomAdamW
lr = 1e-4 * accumulative_counts
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.1

# Save
save_steps = 2000
save_total_limit = 1

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-5, by_epoch=False,
         begin=0, end=warmup_ratio * max_iters),
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=False,
         begin=warmup_ratio * max_iters, end=max_iters),
]

train_cfg = dict(type=TrainLoop, max_iters=max_iters)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
