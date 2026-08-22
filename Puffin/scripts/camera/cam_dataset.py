import os
import json
import argparse
import torch
from tqdm import tqdm

from scripts.camera.geometry.camera import SimpleRadial
from scripts.camera.geometry.gravity import Gravity
from scripts.camera.geometry.perspective_fields import get_perspective_field
from scripts.camera.utils.conversions import fov2focal
from scripts.camera.utils.text import parse_camera_params

class Cam_Generator:
    def __init__(self, mode="base"):
        self.mode = mode

    def _load_text(self, caption, h=512, w=512, k1=0, k2=0):
        # Parse camera params from caption
        roll, pitch, vfov = parse_camera_params(caption, self.mode)
        
        # Convert vertical FoV to focal length
        f = fov2focal(torch.tensor(vfov), h)
        px, py = w / 2, h / 2
        params = torch.tensor([w, h, f, f, px, py, k1, k2]).float()
        gravity = torch.tensor([roll, pitch]).float()
        return params, gravity

    def _read_param(self, parameters, gravity):
        # Build camera and gravity objects
        camera = SimpleRadial(parameters).float()
        roll, pitch = gravity.unbind(-1)
        gravity_obj = Gravity.from_rp(roll, pitch)
        camera = camera.scale(torch.Tensor([1, 1]))
        return {"camera": camera, "gravity": gravity_obj}

    def _get_perspective(self, data):
        # Generate up and latitude fields
        camera = data["camera"]
        gravity_obj = data["gravity"]
        up_field, lat_field = get_perspective_field(
            camera, gravity_obj, use_up=True, use_latitude=True
        )
        del camera, gravity_obj
        return torch.cat([up_field[0], lat_field[0]], dim=0)

    def get_cam(self, caption):
        params, gravity = self._load_text(caption)
        data = self._read_param(params, gravity)
        return self._get_perspective(data)

def process_folders(input_root, output_root, start_idx=0, num_folders=None, mode="base"):
    gen = Cam_Generator(mode=mode)
    all_dirs = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])
    if num_folders is None:
        num_folders = len(all_dirs) - start_idx
    selected = all_dirs[start_idx:start_idx + num_folders]

    for sub in tqdm(selected, desc="Subfolders"):
        in_sub = os.path.join(input_root, sub)
        out_sub = os.path.join(output_root, sub)
        os.makedirs(out_sub, exist_ok=True)

        json_files = sorted([
            f for f in os.listdir(in_sub)
            if f.lower().endswith('.json')
        ])

        for jf in tqdm(json_files, desc=f"Processing {sub}", leave=False):
            in_path = os.path.join(in_sub, jf)
            with open(in_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            caption = data.get('caption', '')
            cam = gen.get_cam(caption)
            out_name = os.path.splitext(jf)[0] + '.pt'
            out_path = os.path.join(out_sub, out_name)
            torch.save(cam, out_path)

def main():
    parser = argparse.ArgumentParser(
        description="Batch process the captions to the camera maps and save as .pt"
    )
    parser.add_argument('--input_root', type=str,
                        help='Root directory of JSON subfolders')
    parser.add_argument('--output_root', type=str,
                        help='Root directory to save .pt files')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index of subfolders (0-based, default=0)')
    parser.add_argument('--num_folders', type=int, default=None,
                        help='Number of subfolders to process (default: all)')
    parser.add_argument('--mode', type=str, default='base',
                        help='parse_camera_params mode')
    args = parser.parse_args()

    process_folders(
        args.input_root,
        args.output_root,
        start_idx=args.start_idx,
        num_folders=args.num_folders,
        mode=args.mode
    )


if __name__ == '__main__':
    main()
