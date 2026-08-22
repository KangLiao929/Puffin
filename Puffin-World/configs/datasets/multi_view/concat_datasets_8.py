from mmengine.config import read_base
from xtuner.dataset import ConcatDataset
from src.datasets.collate_functions import CollateFuncGen, CollateConcat
from src.datasets.samplers.multi_source_sampler import MultiSourceSampler, MultiSourceBatchSampler

with read_base():
    from .gen_dl3dv import dataset as dl3dv
    from .gen_realestate import dataset as re10k
    from .gen_hypersim import dataset as hypersim
    from .gen_mvs_synth import dataset as mvs_synth
    from .gen_tartanair import dataset as tartanair
    from .gen_scannet import dataset as scannet
    from .gen_puffin_omni import dataset as puffin_omni
    from configs.datasets.basic.cam2image import dataset as puffin15m

bs = 4
_mix = [
    # dataset       task            repeat  bs
    (dl3dv,        'image2image',   3,      bs),
    (re10k,        'image2image',   2,      bs),
    (hypersim,     'image2image',   1,      bs),
    (mvs_synth,    'image2image',   1,      bs),
    (tartanair,    'image2image',   1,      bs),
    (scannet,      'image2image',   2,      bs),
    (puffin_omni,  'image2image',   1,      bs),
    (puffin15m,    'cam2image',     1,      bs),
]

dataset = dict(
    type=ConcatDataset,
    datasets=[ds for ds, _, _, _ in _mix],
)

group_keys = [task for _, task, _, _ in _mix]
repeats = [repeat for _, _, repeat, _ in _mix]
batch_sizes = [bs for _, _, _, bs in _mix]

# nominal per-GPU batch size = samples per optimizer step / accum
batch_size = sum(repeat * bs for _, _, repeat, bs in _mix) // sum(repeats)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=False,
    dataset=dataset,
    sampler=dict(type=MultiSourceSampler,
                 repeats=repeats,
                 batch_sizes=batch_sizes,
                 shuffle=True),
    batch_sampler=dict(type=MultiSourceBatchSampler,
                       repeats=repeats,
                       batch_sizes=batch_sizes),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=CollateFuncGen, data_type=task)
                                 for _, task, _, _ in _mix],
                    keys=group_keys),
)
