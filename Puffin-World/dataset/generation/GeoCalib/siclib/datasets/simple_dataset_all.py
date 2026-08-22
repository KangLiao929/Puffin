"""
Dataset for images created with 'create_dataset_from_pano.py'.
This version processes each image on the fly, saves up_field & latitude_field,
optionally visualizes them, and uses a progress bar to show overall progress.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import pandas as pd
import torch
from omegaconf import DictConfig

from siclib.datasets.augmentations import IdentityAugmentation
from siclib.datasets.base_dataset import BaseDataset
from siclib.geometry.camera import SimpleRadial
from siclib.geometry.gravity import Gravity
from siclib.geometry.perspective_fields import get_perspective_field
from siclib.utils.conversions import fov2focal
from siclib.utils.image import ImagePreprocessor, load_image
from siclib.utils.tools import fork_rng
from siclib.visualization.visualize_batch import make_perspective_figures
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def load_csv(
    csv_file: Path, img_root: Path
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    """
    Load a CSV file containing image information.

    Args:
        csv_file (Path): Path to the CSV file.
        img_root (Path): Root directory containing the images.

    Returns:
        (infos, params, gravity):
            - infos (list of dict): Each dict has {'name': str, 'file_name': str}.
            - params (torch.Tensor): Stacked camera parameters for each image.
            - gravity (torch.Tensor): Stacked roll/pitch values for each image.
    """
    df = pd.read_csv(csv_file)

    infos, params, gravity = [], [], []
    for _, row in df.iterrows():
        h = row["height"]
        w = row["width"]
        px = row.get("px", w / 2)
        py = row.get("py", h / 2)
        vfov = row["vfov"]
        f = fov2focal(torch.tensor(vfov), h)
        k1 = row.get("k1", 0)
        k2 = row.get("k2", 0)
        params.append(torch.tensor([w, h, f, f, px, py, k1, k2]))

        roll = row["roll"]
        pitch = row["pitch"]
        gravity.append(torch.tensor([roll, pitch]))

        infos.append({"name": row["fname"], "file_name": str(img_root / row["fname"])})

    params = torch.stack(params).float()
    gravity = torch.stack(gravity).float()
    return infos, params, gravity


class SimpleDataset(BaseDataset):
    """
    Dataset for images created with 'create_dataset_from_pano.py'.
    """

    default_conf = {
        # paths
        "dataset_dir": "???",
        "train_img_dir": "${.dataset_dir}/train",
        "val_img_dir": "${.dataset_dir}/val",
        "test_img_dir": "${.dataset_dir}/test",
        "train_csv": "${.dataset_dir}/train.csv",
        "val_csv": "${.dataset_dir}/val.csv",
        "test_csv": "${.dataset_dir}/test.csv",
        # data options
        "use_up": True,
        "use_latitude": True,
        "use_prior_focal": False,
        "use_prior_gravity": False,
        "use_prior_k1": False,
        # image options
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "augmentations": {"name": "geocalib", "verbose": False},
        "p_rotate": 0.0,  # Probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        # data loader options
        "num_workers": 8,
        "prefetch_factor": 2,
        "train_batch_size": 32,
        "val_batch_size": 32,
        "test_batch_size": 32,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split: str) -> torch.utils.data.Dataset:
        """
        Return a dataset for a given split.

        Args:
            split (str): 'train', 'val', or 'test'

        Returns:
            A torch.utils.data.Dataset object for that split
        """
        return _SimpleDataset(self.conf, split)


class _SimpleDataset(torch.utils.data.Dataset):
    """
    Internal Dataset class for images created with 'create_dataset_from_pano.py'.
    """

    def __init__(self, conf: DictConfig, split: str):
        """
        Initialize the dataset.

        Args:
            conf (DictConfig): Configuration dictionary
            split (str): Split name ('train', 'val', or 'test')
        """
        self.conf = conf
        self.split = split
        self.img_dir = Path(conf.get(f"{split}_img_dir"))

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        # Load image information from CSV
        assert f"{split}_csv" in conf, f"Missing {split}_csv in conf"
        infos_path = self.conf.get(f"{split}_csv")
        self.infos, self.parameters, self.gravity = load_csv(infos_path, self.img_dir)
        self.augmentation = IdentityAugmentation()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        if not self.conf.reseed:
            return self.getitem(idx)
        with fork_rng(self.conf.seed + idx, False):
            return self.getitem(idx)

    def getitem(self, idx: int) -> Dict[str, Any]:
        """
        Return a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Dictionary with keys:
                - "name": filename
                - "path": full path to the image
                - "camera": camera parameters
                - "gravity": gravity object
                - "image": the preprocessed image
                - "up_field": if self.conf.use_up = True
                - "latitude_field": if self.conf.use_latitude = True
                - plus prior fields if configured
        """
        infos = self.infos[idx]
        parameters = self.parameters[idx]
        gravity = self.gravity[idx]
        data = self._read_image(infos, parameters, gravity)

        if self.conf.use_up or self.conf.use_latitude:
            data |= self._get_perspective(data)

        return data

    def _read_image(
        self, infos: Dict[str, Any], parameters: torch.Tensor, gravity: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Read the image file, apply augmentation, set up camera, etc.

        Args:
            infos (dict): Dictionary with 'name', 'file_name'
            parameters (torch.Tensor): The camera parameters
            gravity (torch.Tensor): roll, pitch

        Returns:
            Dictionary containing the sample data
        """
        path = Path(str(infos["file_name"]))

        # Load image as uint8 (HWC), then augment -> returns a tensor
        image = load_image(path, self.conf.grayscale, return_tensor=False)
        image = self.augmentation(image, return_tensor=True)

        # Create radial camera
        camera = SimpleRadial(parameters[None]).float()

        roll, pitch = gravity[None].unbind(-1)
        gravity_obj = Gravity.from_rp(roll, pitch)

        # Preprocess (scale, crop, etc.)
        data = self.preprocessor(image)
        camera = camera.scale(data["scales"])
        if "crop_pad" in data:
            camera = camera.crop(data["crop_pad"])

        # Prepare optional priors
        priors = {}
        if self.conf.use_prior_gravity:
            priors["prior_gravity"] = gravity_obj
        if self.conf.use_prior_focal:
            priors["prior_focal"] = camera.f[..., 1]
        if self.conf.use_prior_k1:
            priors["prior_k1"] = camera.k1

        return {
            "name": infos["name"],
            "path": str(path),
            "camera": camera[0],
            "gravity": gravity_obj[0],
            "image": data["image"],  # preprocessed image tensor
            **priors,
            **data,
        }

    def _get_perspective(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute perspective fields (up_field, latitude_field) if configured.

        Args:
            data (dict): Must contain 'camera' and 'gravity' keys.

        Returns:
            dict: possibly containing "up_field" and "latitude_field" keys.
        """
        camera = data["camera"]
        gravity_obj = data["gravity"]

        up_field, lat_field = get_perspective_field(
            camera,
            gravity_obj,
            use_up=self.conf.use_up,
            use_latitude=self.conf.use_latitude
        )

        out = {}
        if self.conf.use_up:
            out["up_field"] = up_field[0]      # shape: [2, H, W]
        if self.conf.use_latitude:
            out["latitude_field"] = lat_field[0]  # shape: [2, H, W]
        return out


if __name__ == "__main__":
    """
    Main script:
    - For each sample, immediately compute up_field & latitude_field (via dataset),
      concatenate them, save .pt, optionally visualize & save .png.
    - A tqdm progress bar is shown to indicate the progress across all images.
    """
    import argparse
    from tqdm import tqdm  # For progress bar

    parser = argparse.ArgumentParser(description="Compute and save up/lat fields for images.")
    parser.add_argument("--data_dir", type=str, default="/mnt/sfs-common/kliao/Dataset/Puffins/panorama-source/train_dataset/Projection_Temp/cam_und_real/",
                        help="Root directory of the dataset containing train/val/test subfolders and CSVs.")
    parser.add_argument("--split", type=str, default="train",
                        help="Which dataset split to process: train, val, or test.")
    parser.add_argument("--save_tensor_dir", type=str, default="/mnt/sfs-common/kliao/Dataset/Puffins/panorama-source/train_dataset/Projection_Temp/cam_und_real/pt/",
                        help="Directory to save concatenated up/lat .pt files.")
    parser.add_argument("--save_vis_dir", type=str, default="/mnt/sfs-common/kliao/Dataset/Puffins/panorama-source/train_dataset/Projection_Temp/cam_und_real/vis/",
                        help="Directory to save PNG visualizations. Omit if not visualizing.")
    parser.add_argument("--visualize", default="True",
                        help="If set, generate _up.png and _lat.png visualizations.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for loading images. Usually 1 for full coverage.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Whether to shuffle the data.")
    args = parser.parse_args()

    # Set up default config
    dconf = SimpleDataset.default_conf
    dconf["dataset_dir"] = args.data_dir
    dconf[f"{args.split}_batch_size"] = args.batch_size
    dconf["num_workers"] = args.num_workers

    dataset = SimpleDataset(dconf)
    loader = dataset.get_data_loader(args.split, shuffle=args.shuffle)

    # Ensure output directories
    os.makedirs(args.save_tensor_dir, exist_ok=True)
    if args.visualize and args.save_vis_dir is not None:
        os.makedirs(args.save_vis_dir, exist_ok=True)

    torch.set_grad_enabled(False)  # No need for autograd for saving fields

    # We'll track progress for the entire split. Usually dataset len = number of images.
    n_total = len(loader.dataset)

    with tqdm(total=n_total, desc="Processing images", unit="img") as pbar:
        for batch in loader:
            B = len(batch["name"])
            for i in range(B):
                img_name = batch["name"][i]

                # Check if up_field or latitude_field is present
                if "up_field" not in batch or "latitude_field" not in batch:
                    # It's possible the dataset is configured not to use them.
                    print(f"Warning: up_field or latitude_field missing in batch, skipping '{img_name}'.")
                    pbar.update(1)
                    continue

                up_field = batch["up_field"][i]     # shape: [2, H, W]
                lat_field = batch["latitude_field"][i]  # shape: [2, H, W]

                # Concatenate -> [3, H, W]
                out_tensor = torch.cat([up_field, lat_field], dim=0)

                # Save .pt file using the image's base name
                pt_filename = os.path.splitext(img_name)[0] + ".pt"
                pt_save_path = os.path.join(args.save_tensor_dir, pt_filename)
                torch.save(out_tensor, pt_save_path)

                # Optional visualization
                if args.visualize and args.save_vis_dir is not None:
                    # Single-sample "batch"
                    single_batch = {}
                    if "image" in batch:
                        single_batch["image"] = batch["image"][i].unsqueeze(0)
                    # up_field in batch dimension: [1, 2, H, W]
                    single_batch["up_field"] = up_field.unsqueeze(0)
                    single_batch["latitude_field"] = lat_field.unsqueeze(0)
                    single_batch["name"] = [img_name]

                    figs_dict = make_perspective_figures(single_batch, single_batch, n_pairs=1)

                    for fig_key, fig_obj in figs_dict.items():
                        import matplotlib.pyplot as plt
                        if not isinstance(fig_obj, plt.Figure):
                            continue

                        # Decide suffix
                        if "up_field" in fig_key:
                            suffix = "_up"
                        elif "latitude_field" in fig_key:
                            suffix = "_lat"
                        else:
                            suffix = f"_{fig_key}"

                        png_name = os.path.splitext(img_name)[0] + suffix + ".png"
                        out_png_path = os.path.join(args.save_vis_dir, png_name)
                        plt.tight_layout()
                        fig_obj.savefig(out_png_path, dpi=1000, bbox_inches='tight', pad_inches=0)
                        plt.close(fig_obj)

                pbar.update(1)  # increment progress by 1 sample

    print("All images processed on the fly. up_field + lat_field have been saved.")
    if args.visualize and args.save_vis_dir:
        print("Visualization images have also been saved.")