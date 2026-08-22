import math
from torch.utils.data import Dataset
from PIL import Image
import os
import io
import json
import random
import torch
try:
    from aoss_client.client import Client
except:
    try:
        from petrel_client.client import Client
    except:
        Client = None
from glob import glob
from xtuner.registry import BUILDER
from src.datasets.utils import crop2square
from einops import rearrange
import numpy as np

from scripts.camera.cam_dataset import Cam_Generator


class _ColumnarDataList:
    """Fork-safe drop-in for a huge JSON list-of-dicts index.

    This stores each dict key as ONE fixed-width numpy unicode array (refcount-free
    buffers -> COW pages stay shared forever) and rebuilds a small dict on
    access. Supports exactly the two operations the codebase uses:
    ``data_list[idx]`` and ``len(data_list)``.
    """

    def __init__(self, columns, n):
        self._columns = columns          # {key: np.ndarray('<U..')}
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return {k: str(col[idx]) for k, col in self._columns.items()}
        raise TypeError(f"_ColumnarDataList only supports int indexing, got {type(idx)}")

    @classmethod
    def build(cls, data_list):
        """Columnarize `data_list`; return the ORIGINAL list unchanged when the
        rows are not uniform all-string dicts (behavior-preserving fallback)."""
        if not data_list or not isinstance(data_list[0], dict):
            return data_list
        keys = list(data_list[0].keys())
        try:
            # uniform keys (same count + every key present below) and all-string
            # values -- otherwise numpy could silently coerce (e.g. int -> str).
            if any(len(row) != len(keys) for row in data_list):
                return data_list
            columns = {}
            for k in keys:
                if not all(type(row[k]) is str for row in data_list):
                    return data_list
                columns[k] = np.asarray([row[k] for row in data_list])
            return cls(columns, len(data_list))
        except (KeyError, TypeError, AttributeError):
            return data_list


class CaptionDataset(Dataset):
    def __init__(self,
                 data_path,
                 image_folder=None,
                 debug=False,
                 ceph_folder=None,
                 image_ceph_folder=None,
                 cap_ceph_folder=None,
                 latents_ceph_folder=None,
                 ceph_config=None,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=128,
                 min_image_size=80,
                 unit_image_size=32,
                 image_size=256,
                 image_length=256,
                 image_process='identity',
                 image_tokens_folder=None,
                 image_latents_folder=None,
                 cap_folder=None,
                 cap_source='caption',
                 camera_model='radial',
                 tokenizer_kwargs=dict(add_special_tokens=True),
                 unconditional=0.1
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.image_folder = image_folder
        self.cap_folder = cap_folder
        self.cap_source = cap_source
        self.debug = debug
        self.image_process = image_process
        self.unit_image_size = unit_image_size

        if tokenizer is not None:
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = None
        self.prompt_template = prompt_template

        self.max_length = max_length
        self.image_length = image_length
        self.image_tokens_folder = image_tokens_folder
        self.image_latents_folder = image_latents_folder
        self.min_image_size = min_image_size
        self.image_size = image_size
        self.unconditional = unconditional
        self.tokenizer_kwargs = tokenizer_kwargs
        self.cam_generator = Cam_Generator(mode=camera_model)

        self.FILE_CLIENT = None
        self.ceph_folder = ceph_folder
        self.ceph_config = ceph_config
        self.latents_ceph_folder = latents_ceph_folder
        self.image_ceph_folder = ceph_folder if image_ceph_folder is None else image_ceph_folder
        self.cap_ceph_folder = ceph_folder if cap_ceph_folder is None else cap_ceph_folder

        self.use_ceph = ((Client is not None) and (ceph_config is not None) and os.path.exists(ceph_config))

    def _load_data(self, data_path: str):
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            json_files = glob(f"{data_path}/*.json")
            data_list = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data_list += json.load(f)

            self.data_list = data_list

        # Fork-safety: columnarize huge index lists (see _ColumnarDataList).
        # Small lists are left as-is (their COW footprint is negligible and the
        # one-time validation scan is not worth it).
        if len(self.data_list) > 100_000:
            self.data_list = _ColumnarDataList.build(self.data_list)

        print(f"Load {len(self.data_list)} data samples from {data_path} "
              f"(columnar={isinstance(self.data_list, _ColumnarDataList)})", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_ceph(self, ceph_path):
        if self.FILE_CLIENT is None:
            self.FILE_CLIENT = Client(self.ceph_config)
        data_bytes = self.FILE_CLIENT.get(ceph_path)

        return io.BytesIO(data_bytes)

    def _read_image(self, image_file, image_folder=None):
        if image_folder is None:
            assert self.use_ceph
            assert self.image_ceph_folder is not None
            image = Image.open(
                self._read_ceph(
                    os.path.join(self.image_ceph_folder, image_file)
                )
            )
        else:
            image = Image.open(
                os.path.join(image_folder, image_file)
            )
        assert image.width > self.min_image_size and image.height > self.min_image_size, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image.convert('RGB')

    def _read_json(self, annotation_file, cap_folder=None):
        if cap_folder is None:
            assert self.use_ceph
            assert self.cap_ceph_folder is not None
            annotation = json.load(
                self._read_ceph(
                    os.path.join(self.cap_ceph_folder, annotation_file)
                )
            )
        else:
            with open(os.path.join(cap_folder, annotation_file), 'r') as f:
                annotation = json.load(f)

        return annotation
    
    def _read_camera(self, caption, h, w):
        return self.cam_generator.get_cam(caption, h, w)

    def _process_image(self, image):
        data = dict()
        if self.image_process == 'crop2square':
            image = crop2square(image)
            image = image.resize(size=(self.image_size, self.image_size))
        elif self.image_process == 'dynamic':
            w, h = image.size
            if w >= h and w >= self.image_size:
                target_w = self.image_size
                target_h = h * (target_w / w)
                target_h = math.ceil(target_h / self.unit_image_size) * self.unit_image_size

            elif h >= w and h >= self.image_size:
                target_h = self.image_size
                target_w = w * (target_h / h)
                target_w = math.ceil(target_w / self.unit_image_size) * self.unit_image_size

            else:
                target_h = math.ceil(h / self.unit_image_size) * self.unit_image_size
                target_w = math.ceil(w / self.unit_image_size) * self.unit_image_size

            image = image.resize(size=(target_w, target_h))
        elif self.image_process == 'identity':
            image = image
        else:
            raise NotImplementedError

        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)
        return data

    def _process_text(self, text):
        if self.tokenizer is None:
            return dict()

        if random.uniform(0, 1) < self.unconditional:
            prompt = self.prompt_template['CFG']
        else:
            prompt = self.prompt_template['GENERATION'].format(input=text.strip())

        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        prompt += self.prompt_template.get('IMG_START_TOKEN', '')
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]

        return dict(input_ids=input_ids[:self.max_length])
    
    def _process_camera(self, camera):
        return camera / (math.pi / 2)

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]

            if self.image_tokens_folder is not None:
                image_tokens = torch.load(os.path.join(self.image_tokens_folder,
                                                       data_sample['image'] + '.pt')).long()
                data = dict(image_tokens=image_tokens)
            elif self.latents_ceph_folder is not None:
                image_latents = torch.load(
                    self._read_ceph(
                        os.path.join(
                            self.latents_ceph_folder, data_sample['image'] + '.pt'
                        )
                    )
                )
                data = dict(image_latents=image_latents)
            elif self.image_latents_folder is not None:
                image_latents = torch.load(os.path.join(self.image_latents_folder,
                                                        data_sample['image'] + '.pt'))
                data = dict(image_latents=image_latents)
            else:
                image = self._read_image(data_sample['image'], self.image_folder).convert('RGB')
                data = self._process_image(image)

            caption = self._read_json(data_sample['annotation'], self.cap_folder)[self.cap_source].strip()

            data.update(self._process_text(caption))
            data.update(image_dir=self.image_folder, image_file=data_sample['image'],
                        type='text2image', text=caption)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
