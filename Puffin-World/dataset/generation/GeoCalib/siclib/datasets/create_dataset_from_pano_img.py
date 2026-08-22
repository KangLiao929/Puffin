"""Create a perspective-image dataset from panorama images.

For every equirectangular panorama, sample `images_per_pano` yaws uniformly
around the circle and, per view, sample camera parameters (roll / pitch /
vfov / optional k1_hat distortion / output shape) from the configured
distributions, crop the perspective image with the chosen camera model, and
record everything to <perspective_dir>/train/ plus a train.csv metadata file
(fname, roll, pitch, vfov, height, width, ...).

Configured via hydra (see configs/ next to this file), e.g.:

    python -m siclib.datasets.create_dataset_from_pano_img \
        --config-name pano_img dataset_dir=/path/panos

Notes:
  - Existing output images are skipped unless `overwrite` is set (resume).
  - Views with >1% black pixels (outside the pano coverage) are discarded.
  - `num_panos` optionally sub-samples the panorama list deterministically
    (seeded by `seed`) for smoke tests or restricted subsets.
"""

import logging
import random
from concurrent import futures
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from tqdm import tqdm

from siclib.geometry.camera import camera_models
from siclib.geometry.gravity import Gravity
from siclib.utils.conversions import deg2rad, focal2fov, fov2focal, rad2deg
from siclib.utils.image import load_image, write_image

logger = logging.getLogger(__name__)
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# mypy: ignore-errors


def max_radius(a, b):
    """Compute the maximum radius of a Brown distortion model."""
    discrim = a * a - 4 * b
    valid = torch.isfinite(discrim) & (discrim >= 0.0)
    discrim = torch.sqrt(discrim) - a
    valid &= discrim > 0.0
    # Avoid division by zero by replacing 0 with 1
    denom = torch.where(valid, discrim, torch.ones_like(discrim))
    return 2.0 / denom


def brown_max_radius(k1, k2):
    """Compute the maximum radius of a Brown distortion model."""
    a = k1 * 3
    b = k2 * 5
    return torch.sqrt(max_radius(a, b))


class ParallelProcessor:
    """Process-pool wrapper with a per-split tqdm progress bar."""

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.executor = futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.pbars = {}

    def update_pbar(self, pbar_key):
        """Advance the progress bar registered under pbar_key."""
        pbar = self.pbars.get(pbar_key)
        if pbar:
            pbar.update(1)

    def submit_tasks(self, task_func, task_args, pbar_key):
        """Submit task_func(*args) for each args tuple; returns the futures."""
        pbar = tqdm(total=len(task_args), desc=f"Processing {pbar_key}", ncols=80)
        self.pbars[pbar_key] = pbar

        def update_pbar_callback(future):
            self.update_pbar(pbar_key)

        fs = []
        for args in task_args:
            future = self.executor.submit(task_func, *args)
            future.add_done_callback(update_pbar_callback)
            fs.append(future)

        return fs

    def wait_for_completion(self, futures_list):
        """Collect all task results (flattened) and close the progress bars."""
        results = []
        for f in futures_list:
            try:
                res = f.result()
                if res is not None:
                    results += res
            except Exception as e:
                logger.error(f"Task failed with error: {e}")

        for key in list(self.pbars.keys()):
            self.pbars[key].close()
            del self.pbars[key]

        return results

    def shutdown(self):
        self.executor.shutdown()


class DatasetGenerator:
    """Dataset generator class to create perspective datasets from panoramas."""

    default_conf = {
        "name": "???",
        "base_dir": "???",
        "pano_dir": "${.base_dir}/{.name}",
        "pano_train": "${.pano_dir}",
        "perspective_dir": "${.save_dir}/${.name}",
        "perspective_train": "${.perspective_dir}/train",
        "perspective_val": "${.perspective_dir}/val",
        "perspective_test": "${.perspective_dir}/test",
        "train_csv": "${.perspective_dir}/train.csv",
        "val_csv": "${.perspective_dir}/val.csv",
        "test_csv": "${.perspective_dir}/test.csv",
        "camera_model": "pinhole",
        "parameter_dists": {
            "roll": {"type": "uniform", "options": {"loc": deg2rad(-45), "scale": deg2rad(90)}},
            "pitch": {"type": "uniform", "options": {"loc": deg2rad(-45), "scale": deg2rad(90)}},
            "vfov": {"type": "uniform", "options": {"loc": deg2rad(20), "scale": deg2rad(85)}},
            "resize_factor": {"type": "uniform", "options": {"loc": 1.0, "scale": 1.0}},
            "shape": {"type": "fix", "value": (640, 640)},
        },
        "num_panos": None,        # int N -> randomly pick N panos; None = all
        "seed": 0,                # only used for the pano sub-sampling
        "images_per_pano": 16,
        "n_workers": 10,
        "device": "cpu",
        "overwrite": False,
    }

    def __init__(self, conf):
        self.conf = OmegaConf.merge(
            OmegaConf.create(self.default_conf),
            OmegaConf.create(conf),
        )
        logger.info(f"Config:\n{OmegaConf.to_yaml(self.conf)}")
        self.infos = {}
        self.device = self.conf.device
        self.camera_model = camera_models[self.conf.camera_model]

    def sample_value(self, parameter_name):
        """Sample one value for parameter_name from its configured distribution."""
        param_conf = self.conf["parameter_dists"][parameter_name]

        # Optional probability gate (e.g. k1_hat: distorted only with prob p)
        if "prob" in param_conf:
            p = float(param_conf.prob)
            if random.random() > p:
                return torch.tensor(0.0)

        # Custom resolution sampling: mixed aspect ratios at 640 base size
        if param_conf.type == "custom_resolution_640":
            r = random.random()
            if r < 0.50:
                choices = [(640, 640)]  # 1:1
            elif r < 0.70:
                choices = [(640, 480), (480, 640)]  # 4:3 / 3:4
            elif r < 0.85:
                choices = [(640, 426), (426, 640)]  # 3:2 / 2:3
            else:
                choices = [(640, 360), (360, 640)]  # 16:9 / 9:16

            # To pin a single fixed ratio instead, replace choices here, e.g.:
            # choices = [(640, 360), (360, 640)]
            selected_shape = random.choice(choices)
            return torch.tensor(selected_shape, dtype=torch.int64)

        # Custom resolution sampling: mixed aspect ratios at 1024 base size
        if param_conf.type == "custom_resolution_1024":
            r = random.random()
            if r < 0.25:
                choices = [(1024, 1024)]  # 1:1
            elif r < 0.50:
                choices = [(1024, 768), (768, 1024)]  # 4:3 / 3:4
            elif r < 0.75:
                choices = [(1024, 688), (688, 1024)]  # 3:2 / 2:3
            else:
                choices = [(1024, 576), (576, 1024)]  # 16:9 / 9:16

            selected_shape = random.choice(choices)
            return torch.tensor(selected_shape, dtype=torch.int64)

        # Fixed value
        if param_conf.type == "fix":
            return torch.tensor(param_conf.value)

        # Anything else resolves to a scipy.stats distribution by name
        rng = np.random.default_rng()
        sampler = getattr(stats, param_conf.type)
        return torch.tensor(sampler.rvs(random_state=rng, **param_conf.options))

    def plot_distributions(self):
        """Save roll/pitch/vfov histograms of the generated split(s) as PDF."""
        if "train" not in self.infos or not self.infos["train"]:
            return
        splits = [s for s in ["train"] if s in self.infos]
        fig, ax = plt.subplots(3, 3, figsize=(15, 10))
        for i, split in enumerate(splits):
            if i > 2:
                break
            data = self.infos[split]
            if not data:
                continue

            # Helper to extract degrees from tensor or float entries
            def to_deg(key):
                return [rad2deg(row[key]) if isinstance(row[key], (float, int))
                        else rad2deg(row[key].item()) for row in data]

            ax[i, 0].hist(to_deg("roll"), bins=100)
            ax[i, 0].set_xlabel("Roll (°)")
            ax[i, 0].set_ylabel(f"Count {split}")

            ax[i, 1].hist(to_deg("pitch"), bins=100)
            ax[i, 1].set_xlabel("Pitch (°)")
            ax[i, 1].set_ylabel(f"Count {split}")

            ax[i, 2].hist(to_deg("vfov"), bins=100)
            ax[i, 2].set_xlabel("vFoV (°)")
            ax[i, 2].set_ylabel(f"Count {split}")

        plt.tight_layout()
        plt.savefig(Path(self.conf.perspective_dir) / "distributions.pdf")
        plt.close(fig)

    def generate_images_from_pano(self, pano_path: Path, out_dir: Path):
        """Generate perspective images from a single panorama."""
        infos = []
        try:
            pano = load_image(pano_path).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load image {pano_path}: {e}")
            return []

        # Target yaws, uniform around the circle
        yaws_target = np.linspace(0, 2 * np.pi, self.conf.images_per_pano, endpoint=False)

        for i, yaw in enumerate(yaws_target):
            perspective_name = f"{pano_path.stem}_{i}.jpg"
            out_path = out_dir / perspective_name

            # Resume: skip views that already exist
            if out_path.exists() and not self.conf.overwrite:
                continue

            # 1. Sample the camera parameters for this single view
            params = {}
            shape = self.sample_value("shape")
            params["height"] = shape[0]
            params["width"] = shape[1]

            for k in self.conf.parameter_dists:
                if k == "shape":
                    continue
                val = self.sample_value(k)
                params[k] = val

            # 2. k1 from k1_hat, with focal correction so the distorted image
            #    still covers the full sensor (raises f / shrinks vfov if not)
            if "k1_hat" in params:
                h = params["height"]
                w = params["width"]
                k1_hat = params["k1_hat"]
                vfov = params["vfov"]

                focal = fov2focal(vfov, h)

                # k = k_hat * vFoV
                k1 = k1_hat * vfov

                if k1.abs() > 1e-6:
                    min_permissible_rmax = torch.sqrt((h / 2) ** 2 + (w / 2) ** 2)
                    r_max = brown_max_radius(k1=k1, k2=0)
                    denom = r_max * (1 + k1 * r_max**2)

                    # Avoid NaN/Inf
                    if denom > 0:
                        lowest_possible_f_px = min_permissible_rmax / denom
                        if lowest_possible_f_px > focal:
                            focal = lowest_possible_f_px
                            vfov = focal2fov(focal, h)

                params["vfov"] = vfov
                params["k1"] = k1

            # 3. Instantiate camera + gravity (batch size 1)
            cam_params = {k: torch.tensor([v]).to(self.device) if not torch.is_tensor(v)
                          else v.unsqueeze(0).to(self.device)
                          for k, v in params.items()}

            cam = self.camera_model.from_dict(cam_params).float().to(self.device)

            roll_t = cam_params["roll"]
            pitch_t = cam_params["pitch"]
            gravity = Gravity.from_rp(roll_t, pitch_t).float().to(self.device)

            # 4. Render the view from the panorama
            current_yaw = torch.tensor([yaw]).to(self.device)

            perspective_images = cam.get_img_from_pano(
                pano_img=pano,
                gravity=gravity,
                yaws=current_yaw,
                resize_factor=None,
            )
            perspective_image = perspective_images[0]

            # 5. Discard views with >1% black pixels (outside pano coverage)
            n_pixels = perspective_image.shape[-2] * perspective_image.shape[-1]
            valid_img = (torch.sum(perspective_image.sum(0) == 0) / n_pixels) < 0.01
            if not valid_img:
                continue

            write_image(perspective_image, out_path)

            info = {"fname": perspective_name} | {
                k: v.item() if torch.is_tensor(v) else v for k, v in params.items()
            }
            infos.append(info)

        return infos

    def generate_split(self, split: str, parallel_processor: ParallelProcessor):
        """Generate all perspective images of one split and write its CSV."""
        self.infos[split] = []

        pano_dir_key = f"pano_{split}"
        if pano_dir_key not in self.conf:
            return

        pano_dir = Path(self.conf[pano_dir_key])
        if not pano_dir.exists():
            return

        panorama_paths = sorted([
            path
            for path in pano_dir.glob("*")
            if path.suffix.lower() in valid_extensions
        ])

        # Optional deterministic sub-sampling: pick N panos (seeded), useful
        # for smoke-tests or restricted training subsets.
        num_panos = self.conf.get("num_panos", None)
        if num_panos is not None and int(num_panos) > 0 and int(num_panos) < len(panorama_paths):
            n_pick = int(num_panos)
            rng = random.Random(int(self.conf.get("seed", 0)))
            panorama_paths = sorted(rng.sample(panorama_paths, n_pick))
            logger.info(
                f"[{split}] randomly sampled {n_pick} panoramas "
                f"(seed={int(self.conf.get('seed', 0))})"
            )

        out_dir = Path(self.conf[f"perspective_{split}"])
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        futures_list = parallel_processor.submit_tasks(
            self.generate_images_from_pano, [(f, out_dir) for f in panorama_paths], split
        )
        self.infos[split] = parallel_processor.wait_for_completion(futures_list)

        if self.infos[split]:
            metadata = pd.DataFrame(data=self.infos[split])
            metadata.to_csv(self.conf[f"{split}_csv"], index=False)

    def generate_dataset(self):
        """Generate the train split, save the config, and plot distributions."""
        out_dir = Path(self.conf.perspective_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        OmegaConf.save(self.conf, out_dir / "config.yaml")

        processor = ParallelProcessor(self.conf.n_workers)
        for split in ["train"]:
            self.generate_split(split=split, parallel_processor=processor)

        processor.shutdown()

        if "train" in self.infos:
            logger.info(f"Generated {len(self.infos['train'])} train images.")

        self.plot_distributions()


@hydra.main(version_base=None, config_path="configs", config_name="SUN360")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: build the generator from the config and run it."""
    generator = DatasetGenerator(conf=cfg)
    generator.generate_dataset()


if __name__ == "__main__":
    main()
