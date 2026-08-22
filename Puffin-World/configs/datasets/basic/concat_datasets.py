from mmengine.config import read_base
from xtuner.dataset import ConcatDataset
from src.datasets.collate_functions import CollateFuncUnd, CollateFuncGen, CollateConcat
from src.datasets.samplers.multi_source_sampler import MultiSourceSampler, MultiSourceBatchSampler

with read_base():
    from .image2text import dataset as und_data
    from .cam2image import dataset as gen_data

_mix = [
    # dataset    task           collate         repeat  bs
    (und_data,  'image2text',   CollateFuncUnd,  1,     32),
    (gen_data,  'cam2image',    CollateFuncGen,  1,     32),
]

dataset = dict(
    type=ConcatDataset,
    datasets=[ds for ds, _, _, _, _ in _mix],
)

group_keys = [task for _, task, _, _, _ in _mix]
repeats = [repeat for _, _, _, repeat, _ in _mix]
batch_sizes = [bs for _, _, _, _, bs in _mix]

# nominal per-GPU batch size = samples per optimizer step / accum
batch_size = sum(repeat * bs for _, _, _, repeat, bs in _mix) // sum(repeats)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=MultiSourceSampler,
                 repeats=repeats,
                 batch_sizes=batch_sizes,
                 shuffle=True),
    batch_sampler=dict(type=MultiSourceBatchSampler,
                       repeats=repeats,
                       batch_sizes=batch_sizes),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=collate, data_type=task)
                                 for _, task, collate, _, _ in _mix],
                    keys=group_keys),
)
