import logging
import resource
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from siclib.datasets import get_dataset
from siclib.eval.eval_pipeline import EvalPipeline
from siclib.eval.io import get_eval_parser, load_model, parse_eval_args
from siclib.eval.utils import download_and_extract_benchmark, plot_scatter_grid
from siclib.geometry.base_camera import BaseCamera
from siclib.geometry.camera import Pinhole
from siclib.geometry.gravity import Gravity
from siclib.models.cache_loader import CacheLoader
from siclib.settings import EVAL_PATH
from siclib.utils.conversions import rad2deg
from siclib.utils.export_predictions import export_predictions
from siclib.utils.tensor import add_batch_dim
from siclib.utils.tools import AUCMetric, set_seed
from siclib.visualization import visualize_batch, viz2d
from siclib.models.utils.metrics import (
    gravity_error,
    latitude_error,
    pitch_error,
    roll_error,
    up_error,
    vfov_error,
)

logger = logging.getLogger(__name__)
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
torch.set_grad_enabled(False)


def calculate_pixel_projection_error(
    camera_pred: BaseCamera, camera_gt: BaseCamera, N: int = 500, distortion_only: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W = camera_gt.size.unbind(-1)
    H, W = H.int(), W.int()
    assert torch.allclose(camera_gt.size, camera_pred.size), \
        f"Cameras must have the same size: {camera_gt.size} != {camera_pred.size}"

    if distortion_only:
        params = camera_gt._data.clone()
        params[..., -2:] = camera_pred._data[..., -2:]
        CameraModel = type(camera_gt)
        camera_pred = CameraModel(params)

    x_gt, y_gt = torch.meshgrid(
        torch.linspace(0, H - 1, N), torch.linspace(0, W - 1, N), indexing="xy"
    )
    xy = torch.stack((x_gt, y_gt), dim=-1).reshape(-1, 2)
    camera_pin_gt = camera_gt.pinhole()
    uv_pin, _ = camera_pin_gt.image2world(xy)

    xy_undist_gt, valid_dist_gt = camera_gt.world2image(uv_pin)
    xy_undist, valid_dist = camera_pred.world2image(uv_pin)
    valid = valid_dist_gt & valid_dist

    dist = (xy_undist - xy_undist_gt).pow(2).sum(-1).sqrt()
    return dist[valid_dist_gt], valid[valid_dist_gt]


def compute_camera_metrics(
    camera_pred: BaseCamera, camera_gt: BaseCamera, thresholds: List[float]
) -> Dict[str, float]:
    results = defaultdict(list)
    results["vfov"].append(rad2deg(camera_pred.vfov).item())
    results["vfov_error"].append(vfov_error(camera_pred, camera_gt).item())
    results["focal"].append(camera_pred.f[..., 1].item())
    focal_error = torch.abs(camera_pred.f[..., 1] - camera_gt.f[..., 1])
    results["focal_error"].append(focal_error.item())
    rel_focal_error = focal_error / camera_gt.f[..., 1]
    results["rel_focal_error"].append(rel_focal_error.item())
    if hasattr(camera_pred, "k1"):
        results["k1"].append(camera_pred.k1.item())
        k1_error = torch.abs(camera_pred.k1 - camera_gt.k1)
        results["k1_error"].append(k1_error.item())
        if thresholds is None:
            return results
        # pixel projection & distortion errors...
        for mode in (False, True):
            err, valid = calculate_pixel_projection_error(camera_pred, camera_gt, distortion_only=mode)
            key_prefix = "pixel_distortion_error" if mode else "pixel_projection_error"
            for th in thresholds:
                frac = (err[valid] < th).sum().float() / len(valid)
                results[f"{key_prefix}@{th}"].append(frac.item())
    return results


def compute_gravity_metrics(gravity_pred: Gravity, gravity_gt: Gravity) -> Dict[str, float]:
    results = defaultdict(list)
    results["roll"].append(rad2deg(gravity_pred.roll).item())
    results["pitch"].append(rad2deg(gravity_pred.pitch).item())
    results["roll_error"].append(roll_error(gravity_pred, gravity_gt).item())
    results["pitch_error"].append(pitch_error(gravity_pred, gravity_gt).item())
    results["gravity_error"].append(gravity_error(gravity_pred[None], gravity_gt[None]).item())
    return results


def evaluate_from_csv(
    gt_csv: Path,
    pred_csv: Path,
    output_dir: Path,
    thresholds: List[float] = [1.0, 5.0, 10.0]
):
    df_gt = pd.read_csv(gt_csv, usecols=["fname", "roll", "pitch", "vfov"])
    df_pred = pd.read_csv(pred_csv, usecols=["fname", "roll", "pitch", "vfov"])
    df = pd.merge(df_gt, df_pred, on="fname", suffixes=("_gt", "_pred"), how="inner")
    if df.shape[0] == 0:
        raise ValueError("No matching 'fname' between gt and pred CSVs.")

    for field in ("roll", "pitch", "vfov"):
        df[f"{field}_error"] = ((df[f"{field}_pred"] - df[f"{field}_gt"]).abs()) / np.pi * 180 

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / "detailed_results.csv"
    df.to_csv(detailed_csv, index=False)
    logger.info(f"Detailed results saved to {detailed_csv}")

    metrics = {}
    for field in ("roll", "pitch", "vfov"):
        errs = df[f"{field}_error"].to_numpy()
        metrics[f"mean_{field}_error"] = errs.mean()
        metrics[f"median_{field}_error"] = np.median(errs)
        aucs = AUCMetric(elements=errs, thresholds=thresholds, min_error=1).compute()
        for i, thr in enumerate(thresholds):
            metrics[f"auc_{field}_error@{thr}"] = float(aucs[i])

    summary_txt = output_dir / "summary_metrics.txt"
    with open(summary_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    logger.info(f"Summary metrics saved to {summary_txt}")

    print("\n=== Summary Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


class SimplePipeline(EvalPipeline):
    default_conf = {
        "data": {},
        "model": {},
        "eval": {
            "thresholds": [1, 5, 10],
            "pixel_thresholds": [0.5, 1, 3, 5],
            "num_vis": 10,
            "verbose": True,
        },
        "url": None,
    }
    export_keys = ["camera", "gravity"]
    optional_export_keys = [
        "focal_uncertainty", "vfov_uncertainty", "roll_uncertainty", "pitch_uncertainty",
        "gravity_uncertainty", "up_field", "up_confidence",
        "latitude_field", "latitude_confidence",
    ]

    def _init(self, conf):
        super()._init(conf)

    @classmethod
    def get_dataloader(cls, data_conf=None, batch_size=None):
        return super().get_dataloader(data_conf, batch_size)

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        return super().get_predictions(experiment_dir, model, overwrite)

    def get_figures(self, results):
        return super().get_figures(results)

    def run_eval(self, loader, pred_file):
        return super().run_eval(loader, pred_file)


if __name__ == "__main__":
    parser = get_eval_parser()
    parser.add_argument(
        "--gt_csv", type=str, default="/mnt/sfs-common/kliao/Dataset/Puffins/panorama-source/test_dataset/stanford2d3d/images.csv",
        help="Path to ground-truth CSV (cols: fname, roll, pitch, vfov in degrees)"
    )
    parser.add_argument(
        "--pred_csv", type=str, default="/mnt/sfs-common/kliao/Dataset/Puffins/panorama-source/test_dataset/stanford2d3d/puffin_base/captions.csv",
        help="Path to predictions CSV (same format as gt_csv)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/mnt/sfs-common/kliao/Dataset/Puffins/panorama-source/test_dataset/stanford2d3d/puffin_base",
        help="Directory to write detailed_results.csv and summary_metrics.txt"
    )
    args = parser.parse_intermixed_args()

    if args.gt_csv and args.pred_csv and args.output_dir:
        evaluate_from_csv(
            gt_csv=Path(args.gt_csv),
            pred_csv=Path(args.pred_csv),
            output_dir=Path(args.output_dir),
            thresholds=SimplePipeline.default_conf["eval"]["thresholds"]
        )
    else:
        dataset_name = Path(__file__).stem
        default_conf = OmegaConf.create(SimplePipeline.default_conf)
        output_dir = Path(EVAL_PATH, dataset_name)
        output_dir.mkdir(exist_ok=True, parents=True)

        name, conf = parse_eval_args(dataset_name, args, "configs/", default_conf)
        experiment_dir = output_dir / name
        experiment_dir.mkdir(exist_ok=True)

        pipeline = SimplePipeline(conf)
        summaries, figures, results = pipeline.run(
            experiment_dir,
            overwrite=args.overwrite,
            overwrite_eval=args.overwrite_eval
        )
        pprint(summaries)
        if args.plot:
            for name, fig in figures.items():
                fig.canvas.manager.set_window_title(name)
            plt.show()
