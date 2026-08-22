import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from scripts.camera.geometry.camera import SimpleRadial
from scripts.camera.geometry.gravity import Gravity
from scripts.camera.geometry.perspective_fields import get_perspective_field
from scripts.camera.utils.conversions import fov2focal
from scripts.camera.utils.text import parse_camera_params
from src.dust3r.datasets.base.base_multiview_dataset import get_ray_map

def str2bool(v):
    """Parse a bool from a CLI string so `--add_ray False/True/0/1/yes/no` works."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('true', 't', 'yes', 'y', '1'):
        return True
    if s in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {v!r}.")


class Cam_Generator:
    def __init__(self, mode="radial"):
        self.mode = mode

    def get_cam(self, caption, h=640, w=640, k2=0, add_ray=True):
        # Parse camera params from caption
        roll, pitch, vfov, k1 = parse_camera_params(caption, self.mode)
        
        # Convert vertical FoV to focal length
        f = fov2focal(torch.tensor(vfov), h)
        px, py = w / 2, h / 2
        params = torch.tensor([w, h, f, f, px, py, k1, k2]).float()
        gravity = torch.tensor([roll, pitch]).float()
        
        # Build camera and gravity objects
        camera = SimpleRadial(params).float()
        roll, pitch = gravity.unbind(-1)
        gravity_obj = Gravity.from_rp(roll, pitch)
        camera = camera.scale(torch.Tensor([1, 1]))
        
        # Generate up and latitude fields for perspective fields
        up_field, lat_field = get_perspective_field(
            camera, gravity_obj, use_up=True, use_latitude=True
        )
        camera_field = torch.cat([up_field[0], lat_field[0]], dim=0) 

        # Generate 6-channel Ray Map (Origin + Direction)
        if add_ray:
            intrinsics = np.array(
                [[float(f), 0.0, float(px)], [0.0, float(f), float(py)], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            eye = np.eye(4, dtype=np.float32)
            ray_map_np = get_ray_map(eye, eye, intrinsics, h, w)  # [H, W, 6]
            ray_map = torch.from_numpy(ray_map_np).to(
                device=camera_field.device, dtype=camera_field.dtype
            )
            ray_map = ray_map.permute(2, 0, 1).contiguous()  # [6, H, W]
            
            # Concatenate Ray Map (6ch) + Perspective Field (3ch) -> [9, H, W]
            camera_field = torch.cat([ray_map, camera_field], dim=0)
        
        del camera, gravity_obj
        return camera_field

def process_dir(gen, in_dir, out_dir, add_ray=True, desc="Processing",
                h=512, w=512):
    """Read every *.json directly under `in_dir` (no recursion), build a camera
    map per file and save it as a same-named .pt under `out_dir`.

    Camera-map size is always (h, w) as specified by the caller."""
    os.makedirs(out_dir, exist_ok=True)
    json_files = sorted([
        f for f in os.listdir(in_dir)
        if f.lower().endswith('.json')
    ])
    for jf in tqdm(json_files, desc=desc, leave=False):
        in_path = os.path.join(in_dir, jf)
        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        caption = data.get('caption', '')
        cam = gen.get_cam(caption, h=h, w=w, add_ray=add_ray)
        out_name = os.path.splitext(jf)[0] + '.pt'
        torch.save(cam, os.path.join(out_dir, out_name))


def process_flat(input_root, output_root, mode="radial", add_ray=True,
                 h=512, w=512):
    """Flat mode: JSONs live directly under `input_root` (one level, no
    subfolders); .pt files are written directly under `output_root`."""
    gen = Cam_Generator(mode=mode)
    process_dir(gen, input_root, output_root, add_ray=add_ray,
                desc="Processing files", h=h, w=w)


def process_folders(input_root, output_root, start_idx=0, num_folders=None,
                    mode="radial", add_ray=True, h=512, w=512):
    """Subfolder mode: each subfolder of `input_root` holds JSONs; the output
    mirrors the subfolder structure under `output_root`."""
    gen = Cam_Generator(mode=mode)
    all_dirs = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])
    if num_folders is None:
        num_folders = len(all_dirs) - start_idx
    selected = all_dirs[start_idx:start_idx + num_folders]

    for sub in tqdm(selected, desc="Subfolders"):
        process_dir(gen, os.path.join(input_root, sub),
                    os.path.join(output_root, sub), add_ray=add_ray,
                    desc=f"Processing {sub}", h=h, w=w)

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
    parser.add_argument('--mode', type=str, default='radial',
                        choices=['radial', 'pinhole'],
                        help='parse_camera_params mode (must match how the '
                             'captions were produced; default radial).')
    parser.add_argument('--add_ray', type=str2bool, default=False,
                        metavar='BOOL',
                        help='Whether to prepend the 6-ch ray map to the '
                             'perspective field, e.g. --add_ray True / '
                             '--add_ray False. True -> [9, H, W]; '
                             'False -> [3, H, W] (default False).')
    parser.add_argument('--flat', action='store_true',
                        help='Flat mode: read all *.json directly under '
                             'input_root (one level, no subfolders) and write '
                             '.pt directly under output_root. Without this, '
                             'each subfolder of input_root is processed and '
                             'the structure is mirrored to output_root.')
    parser.add_argument('--height', type=int, default=512,
                        help='Camera-map height (default 512).')
    parser.add_argument('--width', type=int, default=512,
                        help='Camera-map width (default 512).')
    args = parser.parse_args()

    if args.flat:
        process_flat(
            args.input_root,
            args.output_root,
            mode=args.mode,
            add_ray=args.add_ray,
            h=args.height,
            w=args.width,
        )
        return

    process_folders(
        args.input_root,
        args.output_root,
        start_idx=args.start_idx,
        num_folders=args.num_folders,
        mode=args.mode,
        add_ray=args.add_ray,
        h=args.height,
        w=args.width,
    )


if __name__ == '__main__':
    main()
