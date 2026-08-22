from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncGen, CollateConcat
from src.datasets.generation.caption_datasets import CaptionDatasetGen as CaptionDataset

with read_base():
    from ..processors import image_size, image_process

repeats = [1,]
group_keys = ['text2image']

dataset = dict(type=CaptionDataset,
               data_type='text2image',
               image_size=image_size,
               image_process=image_process,
               cap_folder='Puffin-4M/training_data/cap_folder/',
               data_path='Puffin-4M/training_data/summary.json',
               image_folder='Puffin-4M/training_data/local_folder/',
               ceph_folder=None,
               ceph_config=None,)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=CollateFuncGen, data_type='text2image'),
                                 ],
                    keys=group_keys
                    )
)