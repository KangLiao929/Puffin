from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncUnd, CollateConcat
from src.datasets.understanding.caption_datasets import CaptionDatasetUnd as CaptionDataset

with read_base():
    from ..processors import image_size, image_process

caption_prompts = [
    "Reason the spatial distribution of this image in a thinking mode, and then estimate its camera parameters (roll, pitch, and field-of-view).",
 ]

dataset = dict(type=CaptionDataset,
               data_type='image2text',
               image_size=image_size,
               image_process=image_process,
               caption_prompts=caption_prompts,
               data_path='Puffin-4M/training_data/summary.json',
               image_folder='Puffin-4M/training_data/local_folder/',
               cap_folder='Puffin-4M/training_data/cap_folder_cot/',
               max_length=512,
               ceph_folder=None,
               ceph_config=None,)

group_keys = ['image2text',]
repeat = [1,]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=CollateFuncUnd, data_type='image2text'),
                                 ],
                    keys=group_keys
                    )
)