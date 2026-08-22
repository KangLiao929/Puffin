import torch
from typing import Dict, Sequence, Optional

def collate_func_gen(instances: Sequence[Dict], data_type: Optional[str]='text2image'):
    pixel_values, texts = [], []
    cam_values = []
    pixel_values_init = []
    for example in instances:
        pixel_values.append(example.pop('pixel_values'))
        texts.append(example.pop('text'))
        
        # for cam2image and image2image
        if data_type != 'text2image':
            cam_values_init_ = example.pop('cam_values')
            if isinstance(cam_values_init_, torch.Tensor):  # single ref case
                cam_values_init_ = [cam_values_init_]
            cam_values.append(cam_values_init_)
            if data_type == 'image2image':
                pixel_values_init_ = example.pop('pixel_values_init')
                if isinstance(pixel_values_init_, torch.Tensor):  # single ref case
                    pixel_values_init_ = [pixel_values_init_]
                pixel_values_init.append(pixel_values_init_)

    data_dict = dict(pixel_values=pixel_values, texts=texts)
    if data_type != 'text2image':
        data_dict['cam_values'] = cam_values
        if data_type == 'image2image':
            data_dict['pixel_values_init'] = pixel_values_init

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