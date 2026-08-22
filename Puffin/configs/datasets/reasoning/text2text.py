from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncUnd
from src.datasets.understanding.caption_datasets import CaptionDatasetUnd as CaptionDataset

caption_prompts = [
    "Given a scene description and corresponding camera parameters, merge them into a coherent prompt and generate an accurate visualization that highlights visual cues for spatial reasoning.",
 ]

dataset = dict(type=CaptionDataset,
               data_type='text2text',
               cap_folder='Puffin-4M/training_data/cap_folder/',
               cap_folder_cot='Puffin-4M/training_data/cap_folder_cot/',
               data_path='Puffin-4M/training_data/summary.json',
               max_length=512,
               caption_prompts=caption_prompts,
               ceph_folder=None,
               ceph_config=None,)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateFuncUnd, data_type='text2text')
)