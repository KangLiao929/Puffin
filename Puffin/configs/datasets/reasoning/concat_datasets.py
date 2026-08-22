from mmengine.config import read_base
from xtuner.dataset import ConcatDataset
from src.datasets.collate_functions import CollateFuncUnd, CollateFuncGen, CollateConcat
from src.datasets.samplers.multi_source_sampler import MultiSourceSampler, MultiSourceBatchSampler

with read_base():
    from .image2text import dataset as und_data
    from .cam2image import dataset as gen_data
    from .text2text import dataset as und_data_text


dataset = dict(
    type=ConcatDataset,
    datasets=[und_data, gen_data, und_data_text]
)

group_keys = ['image2text', 'cam2image', 'text2text']
repeats = [1, 1, 1]
batch_sizes = [8, 8, 8]

batch_size = sum([repeat * batch_size for repeat, batch_size in zip(repeats, batch_sizes)]) // sum(repeats)

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
                       batch_sizes=batch_sizes,
                       ),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=CollateFuncUnd, data_type='image2text'),
                                 dict(type=CollateFuncGen, data_type='cam2image'),
                                 dict(type=CollateFuncUnd, data_type='text2text'),
                                 ],
                    keys=group_keys
                    )
)