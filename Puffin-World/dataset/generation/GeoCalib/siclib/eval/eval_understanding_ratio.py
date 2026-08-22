import argparse
import csv
import json
import logging
import re
import resource
from pathlib import Path

import numpy as np
import pandas as pd
# Assuming siclib is available in your environment as per original code
from siclib.utils.tools import AUCMetric

# Set resource limits
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
    pattern = re.compile(
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)'
    )
    m = pattern.search(caption)
    if not m:
        return 0.0, 0.0, 0.9163, 0.0
    
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))

def detect_suffix_from_gt(gt_csv: Path) -> str:
    """
    Reads the first row of the GT CSV to detect the file extension (jpg, png, etc.).
    """
    try:
        # Read only the first row to check the filename
        df_head = pd.read_csv(gt_csv, nrows=1)
        if "fname" not in df_head.columns:
            logger.warning(f"'fname' column not found in {gt_csv}. Defaulting to 'jpg'.")
            return "jpg"
        
        fname = str(df_head.iloc[0]["fname"])
        suffix = fname.split('.')[-1]
        logger.info(f"Auto-detected file extension from GT: {suffix}")
        return suffix
    except Exception as e:
        logger.warning(f"Failed to detect suffix from GT: {e}. Defaulting to 'jpg'.")
        return "jpg"

def write_predictions_csv(input_json: Path, pred_csv: Path,
                          suffix: str, width: int, height: int):
    """
    Read `input_json` (list of {id, output_text}), extract camera params,
    and write predictions CSV at `pred_csv` using the auto-detected suffix.
    """
    records = json.loads(input_json.read_text(encoding='utf-8'))
    
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

            roll, pitch, vfov, k1 = extract_params(caption)
            
            # Construct filename using the detected suffix
            fname = f"{Path(id_str).stem}.{suffix}"
            
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

def restore_original_vfov(df: pd.DataFrame, ratio_str: str):
    """
    Adjusts the 'vfov_pred' column in-place.
    
    Logic:
    The model predicts vFOV based on a Central Crop defined by 'ratio_str'.
    - If the original image is 'taller' than the target ratio, the crop cut off the top/bottom.
      We must scale the vFOV up to match the original height.
    - If the original image is 'wider' than the target ratio, the crop cut off sides.
      The height (and thus vFOV) remained unchanged.
    """
    if not ratio_str or ratio_str.lower() == "none":
        return

    try:
        w_r, h_r = map(float, ratio_str.split('_'))
        target_aspect = w_r / h_r
    except ValueError:
        logger.warning(f"Invalid ratio format: {ratio_str}. Skipping FOV restoration.")
        return

    # Calculate aspect ratio of original images
    # Avoid division by zero
    df["aspect_orig"] = df["width"] / df["height"]

    # Condition: The image is relatively "taller" (narrower) than the target crop box.
    # In Central Crop logic (keep shortest edge), this means Width was kept, Height was cropped.
    # We need to restore the full vertical FOV.
    mask = df["aspect_orig"] < target_aspect

    if mask.any():
        logger.info(f"Restoring vFOV for {mask.sum()} images based on crop ratio {ratio_str}...")
        
        # Derivation:
        # H_orig = Original Height
        # H_crop = Height of the crop seen by the model
        # For 'tall' images (mask=True): H_crop = Width_orig / target_aspect
        # Scale Factor S = H_orig / H_crop 
        #                = H_orig / (Width_orig / target_aspect)
        #                = (H_orig / Width_orig) * target_aspect
        #                = (1 / aspect_orig) * target_aspect
        
        scale_factor = (1.0 / df.loc[mask, "aspect_orig"]) * target_aspect
        
        vf_crop = df.loc[mask, "vfov_pred"]
        
        # Apply formula: new_vfov = 2 * arctan( tan(old_vfov/2) * scale )
        df.loc[mask, "vfov_pred"] = 2 * np.arctan(np.tan(vf_crop / 2) * scale_factor)
    else:
        logger.info(f"No images required vFOV restoration for ratio {ratio_str} (all fit within crop height).")

def evaluate_from_csv(gt_csv: Path, pred_csv: Path,
                      output_dir: Path, thresholds, ratio: str):

    # Read prediction CSV
    df_pred = pd.read_csv(pred_csv, usecols=["fname", "roll", "pitch", "vfov", "k1"])

    # Read GT CSV
    gt_cols_needed = ["fname", "roll", "pitch", "vfov", "width", "height"]
    df_gt_raw = pd.read_csv(gt_csv)
    
    # Filter columns
    existing_cols = [c for c in gt_cols_needed if c in df_gt_raw.columns]
    df_gt = df_gt_raw[existing_cols].copy()
    
    # Handle optional k1 column in GT
    if "k1" in df_gt_raw.columns:
        df_gt["k1"] = df_gt_raw["k1"]
    else:
        logger.info("Column 'k1' not found in GT CSV. Filling GT k1 with 0.0.")
        df_gt["k1"] = 0.0

    # Merge GT and Pred
    df = pd.merge(df_gt, df_pred, on="fname", suffixes=("_gt","_pred"), how="inner")
    if df.empty:
        raise ValueError("No matching 'fname' between GT and predictions.")
    
    # --- Restore vFOV based on Crop Ratio ---
    restore_original_vfov(df, ratio)
    
    # Compute absolute errors (Angles in degrees, k1 unitless)
    for f in ("roll", "pitch", "vfov"):
        df[f"{f}_error"] = ((df[f"{f}_pred"] - df[f"{f}_gt"]).abs()
                             / np.pi * 180.0)

    df["k1_error"] = (df["k1_pred"] - df["k1_gt"]).abs()

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed = output_dir / "detailed_results.csv"
    df.to_csv(detailed, index=False)
    logger.info(f"Saved detailed results to {detailed}")

    # Compute and save summary metrics
    metrics = {}
    for f in ("roll", "pitch", "vfov", "k1"):
        errs = df[f"{f}_error"].to_numpy()
        metrics[f"mean_{f}_error"]   = errs.mean()
        metrics[f"median_{f}_error"] = np.median(errs)
        
        # Calculate AUC
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

def main():
    parser = argparse.ArgumentParser(
        description="JSON -> predictions CSV -> evaluate vs GT CSV"
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
    # Removed --suffix, it is now auto-detected
    
    parser.add_argument(
        "--width", type=int, default=512,
        help="Image width for predictions CSV (default: 512)"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Image height for predictions CSV (default: 512)"
    )
    parser.add_argument(
        "--ratio", type=str, default=None,
        help="Crop ratio used during inference, e.g., '1_1', '3_4', '16_9'. Default: None (no crop)."
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[1,5,10],
        help="Error thresholds (for angles in degrees). k1 uses same thresholds for AUC logic."
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir    = Path(args.output_dir)
    pred_csv = out_dir / f"{input_json.stem}_predictions.csv"
    gt_csv_path = Path(args.gt_csv)

    # 1. Auto-detect suffix
    suffix = detect_suffix_from_gt(gt_csv_path)

    # 2. Write Prediction CSV
    write_predictions_csv(
        input_json=input_json,
        pred_csv=pred_csv,
        suffix=suffix,
        width=args.width,
        height=args.height
    )

    # 3. Evaluate with Ratio-aware FOV restoration
    evaluate_from_csv(
        gt_csv=gt_csv_path,
        pred_csv=pred_csv,
        output_dir=out_dir,
        thresholds=args.thresholds,
        ratio=args.ratio
    )

if __name__ == "__main__":
    main()