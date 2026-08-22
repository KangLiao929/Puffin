from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncGen
from src.datasets.generation.caption_datasets import CaptionDatasetGen as CaptionDataset

with read_base():
    from ..processors import image_size, image_process

dataset = dict(type=CaptionDataset,
               data_type='cam2image',
               image_size=image_size,
               image_process=image_process,
               cap_folder='Puffin-4M/training_data/cap_folder/',
               cap_folder_cot='Puffin-4M/training_data/cap_folder_cot/',
               data_path='Puffin-4M/training_data/summary.json',
               image_folder='Puffin-4M/training_data/local_folder/',
               cam_folder='Puffin-4M/training_data/cam_folder/', # need to be locally generated from captions using scripts/camera/cam_dataset.py
               max_length=512,
               ceph_folder=None,
               ceph_config=None,)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateFuncGen, data_type='cam2image')
)