from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncGen
from src.datasets.generation.caption_datasets import CaptionDatasetGen as CaptionDataset

with read_base():
    from ..processors import image_size, image_process

dataset = dict(type=CaptionDataset,
               data_type='image2image',
               image_size=image_size,
               image_process=image_process,
               cap_folder='Puffin-4M/training_data/cross_view/cap_folder/',
               data_path='Puffin-4M/training_data/cross_view/summary.json',
               image_folder='Puffin-4M/training_data/cross_view/local_folder/',
               image_folder_init='Puffin-4M/training_data/cross_view/local_folder_init/',
               cam_folder='Puffin-4M/training_data/cross_view/cam_folder/', # need to be locally generated from captions using scripts/camera/cam_dataset.py
               ceph_folder=None,
               ceph_config=None,
               mixed_motion=True,
               mixed_prob=0.5,)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateFuncGen, data_type='image2image')
)