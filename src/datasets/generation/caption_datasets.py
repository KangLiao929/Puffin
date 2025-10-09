import torch
import math
import os
import random
import re

from src.datasets.base_datasets import CaptionDataset

class CaptionDatasetGen(CaptionDataset):
    def __init__(self,
                 data_type='text2image',
                 cam_folder=None,
                 cap_folder_cot=None,
                 image_folder_init=None,
                 mixed_motion=False,
                 mixed_prob=0.5,
                 **kwargs):
        super(CaptionDatasetGen, self).__init__(**kwargs)
        self.data_type = data_type
        self.cam_folder = cam_folder
        self.cap_folder_cot = cap_folder_cot
        self.image_folder_init = image_folder_init
        self.mixed_motion = mixed_motion
        self.mixed_prob = mixed_prob

    def _process_image(self, image):
        return super()._process_image(image)['pixel_values']

    def _read_camera(self, camera_file, cam_folder):
        camera = torch.load(os.path.join(cam_folder, camera_file))
        return camera

    def _process_camera(self, camera):
        return camera / (math.pi / 2)

    def _process_text(self, text):
        if self.tokenizer is None:
            return dict()

        if random.uniform(0, 1) < self.unconditional:
            prompt = self.prompt_template['CFG']
        else:
            prompt = self.prompt_template['GENERATION'].format(input=text.strip())

        # add image token as prompt prefix
        image_tokens = self.prompt_template['IMG_START_TOKEN'] + \
                       self.prompt_template['IMG_CONTEXT_TOKEN'] * self.image_length + \
                       self.prompt_template['IMG_END_TOKEN']
        prompt = f'{image_tokens}\n{prompt}'
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        prompt += self.prompt_template['IMG_START_TOKEN']

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]
        return dict(input_ids=input_ids)

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image'], self.image_folder).convert('RGB')
            pixel_values = self._process_image(image)
            caption = self._read_json(data_sample['annotation'], self.cap_folder)[self.cap_source].strip()
            
            # build the base caption with the thinking caption when thinking with camera
            if self.cap_folder_cot is not None:
                caption_cot = self._read_json(data_sample['annotation'], self.cap_folder_cot)[self.cap_source].strip()
                caption_cot = (m.group(1).strip() if (m := re.search(r'<think>(.*?)</think>', caption_cot, re.S)) else caption_cot)
                caption = caption + ' ' + caption_cot
            
            data = self._process_text(caption)
            
            # text-to-image generation (single-view)
            if self.data_type == 'text2image':
                data.update(pixel_values=pixel_values,image_dir=self.image_folder, 
                            image_file=data_sample['image'], type=self.data_type, text=caption)
                return data

            # text-to-image generation (single-view) with camera map
            elif self.data_type == 'cam2image':
                camera = self._read_camera(data_sample['camera'], self.cam_folder)
                cam_values = self._process_camera(camera)
                
                data.update(pixel_values=pixel_values, cam_values=cam_values,
                            image_dir=self.image_folder, image_file=data_sample['image'],
                            type=self.data_type, text=caption)
                return data
            
            # image-to-image generation (cross-view)
            elif self.data_type == 'image2image':
                image_init = self._read_image(data_sample['image_init'], self.image_folder_init).convert('RGB')
                pixel_values_init = self._process_image(image_init)
                camera = self._read_camera(data_sample['camera'], self.cam_folder)
                cam_values = self._process_camera(camera)
                
                # mix the text+motion and pure motion as conditions to support both text-conditioned and text-free generation
                if self.mixed_motion:
                    if random.random() < self.mixed_prob:  
                        caption = (m.group(1).strip() if (m := re.search(r"(The camera parameters.*)$", caption)) else caption)

                data.update(pixel_values=pixel_values, cam_values=cam_values, 
                            pixel_values_init=pixel_values_init, image_dir=self.image_folder, 
                            image_file=data_sample['image'], type=self.data_type, text=caption)
                return data

            else:
                raise ValueError(f"Unsupported data_type: {self.data_type}")

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
