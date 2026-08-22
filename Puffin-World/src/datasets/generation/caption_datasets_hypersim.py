import torch
import math
import os
import random
import re
from einops import rearrange
from src.dust3r.datasets.hypersim import HyperSim_Multi


class CaptionDatasetGen(HyperSim_Multi):
    def __init__(self,
                 data_type='image2image',
                 debug=False,
                 **kwargs):
        super(CaptionDatasetGen, self).__init__(**kwargs)
        self.data_type = data_type
        self.debug = debug

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        
        # get multi-view data from HyperSim and detach selected ones for training
        views = super().__getitem__(idx)
        data = dict()
        pixel_values = [view["img"] for view in views]
        pixel_values_init = [view["img"] for view in views]
        
        cam_values = [rearrange(torch.from_numpy(view["ray_map"]), 'h w c -> c h w') for view in views]
        
        cam_pose = [str(view["camera_pose"]) for view in views]
        
        cam_intrinsics = [str(view["camera_intrinsics"]) for view in views]
        cam_para = [pose + "\n" + intr for pose, intr in zip(cam_pose, cam_intrinsics)]

        # GT per-frame perspective-field angles from VLM camera captions
        # as space-separated "roll pitch vfov k1" strings.
        gt_cam_params = []
        for view in views:
            angles = view.get("gt_cam_angles", None)
            if angles is None:
                gt_cam_params.append("")
            else:
                gt_cam_params.append(
                    " ".join(f"{float(x):.8f}" for x in angles)
                )

        if self.data_type == 'image2image':
            data.update(pixel_values=pixel_values, cam_values=cam_values,
                        pixel_values_init=pixel_values_init,
                        cam_intrinsics=cam_intrinsics, cam_pose=cam_pose,
                        gt_cam_params=gt_cam_params,
                        type=self.data_type, text="")

            depth_values = [rearrange(torch.from_numpy(view["depth_visionbanana"]), 
                                      'h w c -> c h w') for view in views]
            data.update(depth_values=depth_values)
            # data.update(disparity_values=disparity_values)
            self.file_client.cleanup_temp_files()
            return data
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
