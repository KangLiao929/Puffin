from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncUnd
from src.datasets.understanding.caption_datasets import CaptionDatasetUnd as CaptionDataset

with read_base():
    from ..processors import image_size, image_process

prefix = "Given the initial view and the camera parameters of the target view with the deviation yaw angle, "
tails = [
    "describe the target image in detail.",
    "provide a comprehensive description of everything you see in the target image.",
    "explain the scene depicted in the target image as if you were describing it to someone who cannot see it.",
    "list all the objects and activities taking place in the target image.",
    "what is the story being told by the target image? Describe in detail.",
    "imagine you are giving a detailed tour of the target image's scene. How would you describe it?",
    "describe the foreground, background, and any notable features of the target image.",
    "how would you describe the target image to build a replica of the scene?",
    "write a paragraph detailing the setting, characters, and actions visible in the target image.",
    "describe every aspect of the target image, including the environment, objects, and any people present.",
    "provide a detailed analysis of the composition and elements of the target image.",
    "what are the main focal points of the target image? Describe them in detail.",
    "catalog all visible elements in the target image and describe their significance to the overall scene."
]
caption_prompts = [prefix + t for t in tails]

dataset = dict(type=CaptionDataset,
               data_type='image2text',
               image_size=image_size,
               image_process=image_process,
               caption_prompts=caption_prompts,
               cap_folder='Puffin-4M/training_data/cross_view/cap_folder_scene/',
               data_path='Puffin-4M/training_data/cross_view/summary.json',
               image_folder='Puffin-4M/training_data/cross_view/local_folder_init/',
               cap_folder_cam='Puffin-4M/training_data/cross_view/cap_folder_cam/',
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