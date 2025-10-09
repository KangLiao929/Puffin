from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncUnd
from src.datasets.understanding.caption_datasets import CaptionDatasetUnd as CaptionDataset

with read_base():
    from ..processors import image_size, image_process

caption_prompts = [
    "Estimate the camera parameters (roll, pitch, and field-of-view) of this image. And then predict the deviation camera yaw angle and pitch angle of the target view with high photographic aesthetics.",
  ]

dataset = dict(type=CaptionDataset,
               data_type='image2text',
               image_size=image_size,
               image_process=image_process,
               caption_prompts=caption_prompts,
               cap_folder='Puffin-4M/training_data/photography/cap_folder/',
               data_path='Puffin-4M/training_data/photography/summary.json',
               image_folder='Puffin-4M/training_data/photography/local_folder/',
               ceph_folder=None,
               ceph_config=None,)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateFuncUnd, data_type='image2text')
)