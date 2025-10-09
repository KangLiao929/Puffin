from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncUnd, CollateConcat
from src.datasets.understanding.caption_datasets import CaptionDatasetUnd as CaptionDataset

with read_base():
    from ..processors import image_size, image_process

suffix = " Then reason its spatial distribution and estimate its camera parameters (roll, pitch, and field-of-view)."
heads = [
    "Describe the image in detail.",
    "Provide a comprehensive description of everything you see in the picture.",
    "Explain the scene depicted in the image as if you were describing it to someone who cannot see it.",
    "List all the objects and activities taking place in this image.",
    "What is the story being told by this image? Describe in detail.",
    "Imagine you are giving a detailed tour of the image's scene. How would you describe it?",
    "Describe the foreground, background, and any notable features of the image.",
    "How would you describe this image to build a replica of the scene?",
    "Write a paragraph detailing the setting, characters, and actions visible in this image.",
    "Describe every aspect of the image, including the environment, objects, and any people present.",
    "Provide a detailed analysis of the composition and elements of the image.",
    "What are the main focal points of this image? Describe them in detail.",
    "Catalog all visible elements in the image and describe their significance to the overall scene."
]
caption_prompts = [h + suffix for h in heads]


repeats = [1,]
group_keys = ['image2text']

dataset = dict(type=CaptionDataset,
               data_type='image2text',
               image_size=image_size,
               image_process=image_process,
               caption_prompts=caption_prompts,
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
                    collate_fns=[dict(type=CollateFuncUnd, data_type='image2text'),
                                 ],
                    keys=group_keys
                    )
)