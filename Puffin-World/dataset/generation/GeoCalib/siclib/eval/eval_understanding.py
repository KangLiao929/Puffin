import argparse
import csv
import json
import logging
import re
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from siclib.utils.tools import AUCMetric

_rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, _rlimit[1]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_params(caption: str):
    """
    Extract roll, pitch, vertical field-of-view (vfov), and radial distortion (k1) 
    from a float-format caption.
    
    Expected format: "roll, pitch, vfov, k1"
    Defaults to (0.0, 0.0, 0.9163, 0.0) if not found.
    """
    # Updated pattern to look for 4 float numbers separated by commas
    pattern = re.compile(
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)'
    )
    m = pattern.search(caption)
    if not m:
        # Return default values including k1=0.0
        return 0.0, 0.0, 0.9163, 0.0
    
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))

def detect_image_suffix(gt_csv: Path) -> str:
    """Auto-detect the single image extension used by this dataset.

    A dataset is assumed to use exactly one image format (jpg / png / ...),
    so we read the GT CSV's `fname` column and return its extension (without
    the leading dot). Using the GT extension guarantees the predictions CSV
    `fname` will match GT at merge time. If multiple extensions are present
    (mixed dataset), the most common one is used with a warning.
    """
    df = pd.read_csv(gt_csv, usecols=["fname"])
    exts = [
        Path(str(f)).suffix.lstrip(".").lower()
        for f in df["fname"]
        if str(f).strip() and Path(str(f)).suffix
    ]
    if not exts:
        raise ValueError(
            f"Could not detect any image extension from 'fname' column of {gt_csv}."
        )
    counts = Counter(exts)
    if len(counts) > 1:
        logger.warning(
            f"GT CSV contains multiple image extensions {dict(counts)}; "
            f"using the most common one."
        )
    suffix = counts.most_common(1)[0][0]
    logger.info(f"Auto-detected image suffix '{suffix}' from {gt_csv}")
    return suffix


def write_predictions_csv(input_json: Path, pred_csv: Path,
                          suffix: str, width: int, height: int):
    """
    Read `input_json` (list of {id, output_text}), extract camera params (including k1),
    and write predictions CSV at `pred_csv`.
    """
    records = json.loads(input_json.read_text(encoding='utf-8'))
    
    # Added 'k1' to the CSV header
    with open(pred_csv, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "fname", "roll", "pitch", "vfov", "k1", "width", "height"
        ])
        writer.writeheader()
        count = 0
        for item in records:
            id_str = item.get("id")
            caption = item.get("output_text", "")
            if not id_str:
                continue

            # Extract 4 parameters now
            roll, pitch, vfov, k1 = extract_params(caption)

            # Normalize the id to the GT-csv `fname` form: keep any directory
            # prefix intact (ids may be relative paths like "segment/000013.jpg")
            # and only swap the extension to the detected `suffix`. Using
            # `with_suffix` (not `.stem`) preserves the directory component so
            # path-style ids match the GT csv; plain basenames still work.
            fname = Path(id_str).with_suffix(f".{suffix}").as_posix()
            writer.writerow({
                "fname":  fname,
                "roll":   f"{roll:.6f}",
                "pitch":  f"{pitch:.6f}",
                "vfov":   f"{vfov:.6f}",
                "k1":     f"{k1:.6f}",
                "width":  width,
                "height": height,
            })
            count += 1

    logger.info(f"Wrote {count} prediction rows to {pred_csv}")

def compute_advanced_metrics(df) -> dict:
    """Compute perspective-field metrics (same definitions as simple_pipeline):
    latitude_error / latitude_med_error, up_error / up_med_error, and
    gravity_mean / gravity_median.

    For each matched row, build the GT and predicted camera (SimpleRadial) +
    gravity from (roll, pitch, vfov, k1, width, height), render their up &
    latitude fields, and compute per-pixel up/latitude angular errors plus a
    per-sample gravity error. Aggregate across samples:
        latitude_error     = mean over samples of per-image MEAN  latitude err
        latitude_med_error = mean over samples of per-image MEDIAN latitude err
        up_error           = mean over samples of per-image MEAN  up err
        up_med_error       = mean over samples of per-image MEDIAN up err
        gravity_mean       = mean   over samples of per-image gravity err
        gravity_median     = median over samples of per-image gravity err
    All errors are in degrees. Angles in the CSV are assumed in radians.
    """
    import torch
    from siclib.geometry.camera import SimpleRadial
    from siclib.geometry.gravity import Gravity
    from siclib.geometry.perspective_fields import get_perspective_field
    from siclib.models.utils.metrics import up_error, latitude_error, gravity_error

    @torch.no_grad()
    def _cam_grav(roll, pitch, vfov, k1, w, h):
        cam = SimpleRadial.from_dict({
            "height": torch.tensor(float(h)),
            "width": torch.tensor(float(w)),
            "vfov": torch.tensor(float(vfov)),
            "k1": torch.tensor(float(k1)),
        }).float()
        grav = Gravity.from_rp(torch.tensor(float(roll)).float(),
                               torch.tensor(float(pitch)).float())
        return cam, grav

    up_m, up_md, lat_m, lat_md, grav = [], [], [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            w = int(row.get("width", 512))
            h = int(row.get("height", 512))
            cam_p, g_p = _cam_grav(row["roll_pred"], row["pitch_pred"],
                                   row["vfov_pred"], row.get("k1_pred", 0.0), w, h)
            cam_g, g_g = _cam_grav(row["roll_gt"], row["pitch_gt"],
                                   row["vfov_gt"], row.get("k1_gt", 0.0), w, h)

            up_p, lat_p = get_perspective_field(cam_p, g_p, use_up=True, use_latitude=True)
            up_g, lat_g = get_perspective_field(cam_g, g_g, use_up=True, use_latitude=True)

            ue = up_error(up_p, up_g)            # [1, H, W] degrees
            le = latitude_error(lat_p, lat_g)    # [1, H, W] degrees
            ge = gravity_error(g_p[None], g_g[None])  # scalar degrees

            up_m.append(float(ue.mean()))
            up_md.append(float(ue.median()))
            lat_m.append(float(le.mean()))
            lat_md.append(float(le.median()))
            grav.append(float(ge))

    return {
        "latitude_error":     float(np.mean(lat_m)),
        "latitude_med_error": float(np.mean(lat_md)),
        "up_error":           float(np.mean(up_m)),
        "up_med_error":       float(np.mean(up_md)),
        "gravity_mean":       float(np.mean(grav)),
        "gravity_median":     float(np.median(grav)),
    }


def evaluate_from_csv(gt_csv: Path, pred_csv: Path,
                      output_dir: Path, thresholds, advance_metrics: bool = False):

    # Read prediction CSV (ensure k1 is included)
    df_pred = pd.read_csv(pred_csv, usecols=["fname", "roll", "pitch", "vfov", "k1"])

    # Read GT CSV. We don't specify usecols strictly for k1 yet because it might be missing.
    # We read expected columns first.
    gt_cols_needed = ["fname", "roll", "pitch", "vfov", "width", "height"]
    
    # Check what columns are actually in the file to avoid KeyError if k1 is missing
    df_gt_raw = pd.read_csv(gt_csv)
    
    # Filter only the columns we need if they exist
    existing_cols = [c for c in gt_cols_needed if c in df_gt_raw.columns]
    df_gt = df_gt_raw[existing_cols].copy()
    
    # If 'k1' exists in GT, use it; otherwise fill with 0.0
    if "k1" in df_gt_raw.columns:
        df_gt["k1"] = df_gt_raw["k1"]
    else:
        logger.info("Column 'k1' not found in GT CSV. Filling GT k1 with 0.0.")
        df_gt["k1"] = 0.0

    # Merge GT and Pred
    df = pd.merge(df_gt, df_pred, on="fname", suffixes=("_gt","_pred"), how="inner")
    if df.empty:
        raise ValueError("No matching 'fname' between GT and predictions.")
    
    # Restore the original vertical FOV for predictions based on crop logic
    '''
    mask = df["height"] > df["width"]
    if mask.any():
        ratio = (df["height"] / df["width"])[mask]
        vf_crop = df.loc[mask, "vfov_pred"]
        df.loc[mask, "vfov_pred"] = 2 * np.arctan(np.tan(vf_crop / 2) * ratio)
    '''
    # Compute absolute errors
    # For roll, pitch, vfov (angles), we convert radians to degrees for error metric if inputs are radians
    # (Assuming inputs are radians based on original code's * 180 / pi conversion)
    for f in ("roll", "pitch", "vfov"):
        df[f"{f}_error"] = ((df[f"{f}_pred"] - df[f"{f}_gt"]).abs()
                             / np.pi * 180.0)

    # For k1, it is typically a coefficient (unitless), so we just take abs difference.
    # We do NOT convert to degrees.
    df["k1_error"] = (df["k1_pred"] - df["k1_gt"]).abs()

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed = output_dir / "detailed_results.csv"
    df.to_csv(detailed, index=False)
    logger.info(f"Saved detailed results to {detailed}")

    # Compute and save summary metrics
    metrics = {}
    # Iterate over all 4 parameters
    for f in ("roll", "pitch", "vfov", "k1"):
        errs = df[f"{f}_error"].to_numpy()
        metrics[f"mean_{f}_error"]   = errs.mean()
        metrics[f"median_{f}_error"] = np.median(errs)
        
        # Calculate AUC. 
        # Note: For k1, 'thresholds' (in degrees) might not be physically meaningful 
        # if k1 is small (e.g. < 1.0), but we compute it consistent with the pipeline.
        aucs = AUCMetric(elements=errs, thresholds=thresholds, min_error=1).compute()
        for thr, auc in zip(thresholds, aucs):
            metrics[f"auc_{f}_error@{thr}"] = float(auc)

    summary = output_dir / "summary_metrics.txt"
    with open(summary, 'w', encoding='utf-8') as wf:
        for k,v in metrics.items():
            wf.write(f"{k}: {v:.4f}\n")
    logger.info(f"Saved summary metrics to {summary}")

    print("\n=== Summary Metrics ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Optional perspective-field / gravity metrics (simple_pipeline-style).
    if advance_metrics:
        adv = compute_advanced_metrics(df)
        with open(summary, 'a', encoding='utf-8') as wf:
            wf.write("\n# advanced metrics\n")
            for k, v in adv.items():
                wf.write(f"{k}: {v:.4f}\n")
        print("\n=== Advanced Metrics (perspective field / gravity) ===")
        for k, v in adv.items():
            print(f"{k}: {v:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="JSON → predictions CSV → evaluate vs GT CSV"
    )
    parser.add_argument(
        "--input_json", default="Puffin-Und.json",
        help="Input JSON file (list of {id, output_text})"
    )
    parser.add_argument(
        "--output_dir",  default="Puffin-Und_eval/",
        help="Directory to save predictions CSV, detailed_results.csv, summary_metrics.txt"
    )
    parser.add_argument(
        "--gt_csv",  default="Puffin-Und/cameras.csv",
        help="Path to GT CSV (cols: fname, roll, pitch, vfov, [k1], width, height)"
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Image width for predictions CSV (default: 512)"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Image height for predictions CSV (default: 512)"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[1,5,10],
        help="Error thresholds (for angles in degrees). k1 uses same thresholds for AUC logic."
    )
    parser.add_argument(
        "--advance_metrics", action="store_true",
        help="Also compute perspective-field metrics (latitude_error, "
             "latitude_med_error, up_error, up_med_error) and gravity_mean / "
             "gravity_median, as defined in simple_pipeline.py."
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / f"{input_json.stem}_predictions.csv"

    # Auto-detect the image extension from the GT CSV (one format per dataset).
    suffix = detect_image_suffix(Path(args.gt_csv))

    write_predictions_csv(
        input_json=input_json,
        pred_csv=pred_csv,
        suffix=suffix,
        width=args.width,
        height=args.height
    )

    evaluate_from_csv(
        gt_csv=Path(args.gt_csv),
        pred_csv=pred_csv,
        output_dir=out_dir,
        thresholds=args.thresholds,
        advance_metrics=args.advance_metrics,
    )

if __name__ == "__main__":
    main()