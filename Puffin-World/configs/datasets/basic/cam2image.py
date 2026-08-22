from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import CollateFuncGen, CollateConcat
from src.datasets.generation.caption_datasets import CaptionDatasetGen as CaptionDataset

with read_base():
    from ..processors import image_size, image_process, camera_model
    
repeats = [1,]
group_keys = ['cam2image']

dataset = dict(type=CaptionDataset,
               data_type='cam2image',
               image_size=image_size,
               image_process=image_process,
               camera_model=camera_model,
               cap_folder='/mnt/afs_100t/NTU_slab/kliao/data/Puffin-Pro/Projection',
               image_folder='/mnt/afs_100t/NTU_slab/kliao/data/Puffin-Pro/Projection',
               data_path='/mnt/afs_100t/NTU_slab/kliao/data/Puffin-Pro/summary_gen_scene_cam.json',
               ceph_folder=None,
               ceph_config=None,
               )

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=CollateFuncGen, data_type='cam2image'),
                                 ],
                    keys=group_keys
                    )
)