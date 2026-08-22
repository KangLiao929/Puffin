from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncGen
from src.datasets.generation.caption_datasets_scannet import CaptionDatasetGen as CaptionDataset
with read_base():
    from ..processors import image_size

dataset = dict(type=CaptionDataset,
               data_type='image2image',
               split='train',
               ROOT="aoss:s3://yhluo_sgacer/data/tracking/processed_scannet/",
               cache_path="/mnt/afs_100t/NTU_slab/kliao/data/Puffin2/dataset_summary/scannet_cache.pkl",
               camera_caption_root="/mnt/afs_100t/NTU_slab/kliao/data/camera_caption/scannet",
               depth_da3_root="/data_cavan/NTU_slab/yhluo/Data/da3_anno/scannet_da3",
               resolution=image_size,
               num_views=8,
               min_interval=1,
               max_interval=8,
               debug=False, 
               )

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateFuncGen, data_type='image2image')
)

repeats = [1,]