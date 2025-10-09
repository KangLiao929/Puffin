import random
import re
from xtuner.utils import DEFAULT_IMAGE_TOKEN

from src.datasets.base_datasets import CaptionDataset

class CaptionDatasetUnd(CaptionDataset):
    def __init__(self,
                 caption_prompts,
                 data_type='image2text',
                 cap_folder_cot=None,
                 cap_folder_cam=None,
                 **kwargs):
        super(CaptionDatasetUnd, self).__init__(**kwargs)
        self.caption_prompts = caption_prompts
        self.data_type = data_type
        self.cap_folder_cot = cap_folder_cot
        self.cap_folder_cam = cap_folder_cam

    def _process_text(self, text, text_in=None, data_type='image2text'):
        if data_type == 'image2text':
            data_dict = dict(conversation=[{
                'input': f"{DEFAULT_IMAGE_TOKEN}\n{random.choice(self.caption_prompts)}" 
                        if self.cap_folder_cam is None else f"{DEFAULT_IMAGE_TOKEN}\n{random.choice(self.caption_prompts)} {text_in}",
                'output': text
            }])
        elif data_type == 'text2text':
            data_dict = dict(conversation=[{
                'input': f"{random.choice(self.caption_prompts)} {text_in}",
                'output': text
            }])
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
        
        return data_dict

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            caption = self._read_json(data_sample['annotation'], self.cap_folder)[self.cap_source].strip()

            # image-to-text(camera) understanding, mixed base, thinking, and instruction tuning
            if self.data_type == 'image2text':
                image = self._read_image(data_sample['image'], self.image_folder).convert('RGB')
                data = self._process_image(image)
                
                if self.cap_folder_cam is not None:
                    # for the cross-view understanding task (given an initial image and target camera parameters)
                    caption_cam = self._read_json(data_sample['annotation'], self.cap_folder_cam)[self.cap_source].strip()
                    data.update(self._process_text(caption, caption_cam))
                    data['type'] = 'image2text_cross_view'
                else:
                    data.update(self._process_text(caption))
                    data['type'] = 'image2text'
                return data

            # text-to-text understanding, offering the enhanced caption for the generation
            elif self.data_type == 'text2text':
                caption_cot = self._read_json(data_sample['annotation'], self.cap_folder_cot)[self.cap_source].strip()
                caption_cot = (m.group(1).strip() if (m := re.search(r'<think>(.*?)</think>', caption_cot, re.S)) else caption_cot)
                data = self._process_text(caption_cot, caption, data_type='text2text')
                data.update(type='text2text')
                return data

            else:
                raise ValueError(f"Unsupported data_type: {self.data_type}")

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
