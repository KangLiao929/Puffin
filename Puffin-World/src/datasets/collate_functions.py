import torch
from typing import Dict, Sequence, Optional

def collate_func_gen(instances: Sequence[Dict], data_type: Optional[str]='text2image'):
    pixel_values, texts = [], []
    cam_values = []
    pixel_values_init = []
    depth_values = []
    cam_pose = []
    cam_intrinsics = []
    gt_cam_params = []
    for example in instances:
        # singe-view branch
        if data_type == 'cam2image':
            pixel_values.append(example.pop('pixel_values'))
            texts.append(example.pop('text'))
            cam_values_init_ = example.pop('cam_values')
            if isinstance(cam_values_init_, torch.Tensor):
                cam_values_init_ = [cam_values_init_]
            cam_values.append(cam_values_init_)
        
        # multi-view branch
        if data_type == 'image2image':
            text_ = example.pop('text')
            if isinstance(text_, torch.Tensor):
                text_ = [text_]
            texts.append(text_)
            pixel_values_ = example.pop('pixel_values')
            if isinstance(pixel_values_, torch.Tensor):
                pixel_values_ = [pixel_values_]
            pixel_values.append(pixel_values_)
            cam_values_init_ = example.pop('cam_values')
            if isinstance(cam_values_init_, torch.Tensor):
                cam_values_init_ = [cam_values_init_]
            cam_values.append(cam_values_init_)
            pixel_values_init_ = example.pop('pixel_values_init')
            if isinstance(pixel_values_init_, torch.Tensor):
                pixel_values_init_ = [pixel_values_init_]
            pixel_values_init.append(pixel_values_init_)
            
            # optional 3D clues for both the 3D generation and reconstruction
            depth_values_ = example.pop('depth_values', None)
            if depth_values_ is not None:
                if isinstance(depth_values_, torch.Tensor):
                    depth_values_ = [depth_values_]
                depth_values.append(depth_values_)

            # optional camera pose and intrinsics (for physical propagation)
            cam_pose_ = example.pop('cam_pose', None)
            if cam_pose_ is not None:
                cam_pose.append(cam_pose_)
            cam_intrinsics_ = example.pop('cam_intrinsics', None)
            if cam_intrinsics_ is not None:
                cam_intrinsics.append(cam_intrinsics_)

            # optional GT per-frame perspective-field angles:
            # list[T] of stringified "roll pitch vfov k1" floats per view.
            gt_cam_params_ = example.pop('gt_cam_params', None)
            if gt_cam_params_ is not None:
                gt_cam_params.append(gt_cam_params_)

    if data_type == 'cam2image':
        data_dict = dict(pixel_values=pixel_values, texts=texts, cam_values=cam_values)
    elif data_type == 'image2image':
        data_dict = dict(pixel_values=pixel_values, texts=texts, cam_values=cam_values,
                         pixel_values_init=pixel_values_init)
        data_dict['depth_values'] = depth_values
        if cam_pose:
            data_dict['cam_pose'] = cam_pose
        if cam_intrinsics:
            data_dict['cam_intrinsics'] = cam_intrinsics
        if gt_cam_params:
            data_dict['gt_cam_params'] = gt_cam_params
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return {'data': data_dict, 'data_samples': None}

def collate_func_und(instances: Sequence[Dict], data_type: Optional[str]='text2text'):
    conversations = []
    pixel_values = []
    for example in instances:
        conversations.append(example.pop('conversation'))
        if data_type == 'image2text':
            pixel_values.append(example.pop('pixel_values'))

    data_dict = dict(conversations=conversations)
    if data_type == 'image2text':
        data_dict['pixel_values'] = pixel_values

    return {'data': data_dict, 'data_samples': None}


class CollateFuncGen:
    def __init__(self, data_type='text2image'):
        self.data_type = data_type

    def __call__(self, instances):
        return collate_func_gen(instances, data_type=self.data_type)

class CollateFuncUnd:
    def __init__(self, data_type='image2text'):
        self.data_type = data_type

    def __call__(self, instances):
        return collate_func_und(instances, data_type=self.data_type)

class CollateConcat(object):
    def __init__(self, collate_fns, keys):
        self.keys = keys
        self.collate_fns = {}
        for key, collate_fn in zip(keys, collate_fns):
            func_class = collate_fn.pop('type')
            self.collate_fns[key] = func_class(**collate_fn)

    def __call__(self, data_samples):
        data_samples = [data_sample for data_sample in data_samples if len(data_sample) > 0]
        data_dict = {}
        key = data_samples[0]['type']
        data_dict[key] = self.collate_fns[key](data_samples)['data']

        return {'data': data_dict, 'data_samples': None}