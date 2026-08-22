import argparse
import csv
import json
import logging
import re
import resource
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
    Extract roll, pitch, and field-of-view from a float-format caption.
    Defaults to (0.0, 0.0, 0.9163) if not found.
    """
    pattern = re.compile(
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)\s*,\s*'
        r'([+-]?\d+\.\d+)'
    )
    m = pattern.search(caption)
    if not m:
        return 0.0, 0.0, 0.9163
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

def write_predictions_csv(input_json: Path, pred_csv: Path,
                          suffix: str, width: int, height: int):
    """
    Read `input_json` (list of {id, output_text}), extract camera params,
    and write predictions CSV at `pred_csv`.
    """
    records = json.loads(input_json.read_text(encoding='utf-8'))
    with open(pred_csv, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "fname", "roll", "pitch", "vfov", "width", "height"
        ])
        writer.writeheader()
        count = 0
        for item in records:
            id_str = item.get("id")
            caption = item.get("output_text", "")
            if not id_str:
                continue

            roll, pitch, vfov = extract_params(caption)

            fname = f"{Path(id_str).stem}.{suffix}"
            writer.writerow({
                "fname":  fname,
                "roll":   f"{roll:.6f}",
                "pitch":  f"{pitch:.6f}",
                "vfov":   f"{vfov:.6f}",
                "width":  width,
                "height": height,
            })
            count += 1

    logger.info(f"Wrote {count} prediction rows to {pred_csv}")

def evaluate_from_csv(gt_csv: Path, pred_csv: Path,
                      output_dir: Path, thresholds):

    df_gt   = pd.read_csv(gt_csv, usecols=["fname", "roll", "pitch", "vfov", "width", "height"])
    df_pred = pd.read_csv(pred_csv, usecols=["fname", "roll", "pitch", "vfov"])
    df = pd.merge(df_gt, df_pred, on="fname", suffixes=("_gt","_pred"), how="inner")
    if df.empty:
        raise ValueError("No matching 'fname' between GT and predictions.")
    
    # Restore the original vertical FOV
    mask = df["height"] > df["width"]
    ratio = (df["height"] / df["width"])[mask]
    vf_crop = df.loc[mask, "vfov_pred"]
    df.loc[mask, "vfov_pred"] = 2 * np.arctan(np.tan(vf_crop / 2) * ratio)

    # Compute absolute errors in degrees
    for f in ("roll","pitch","vfov"):
        df[f"{f}_error"] = ((df[f"{f}_pred"] - df[f"{f}_gt"]).abs()
                             / np.pi * 180.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed = output_dir / "detailed_results.csv"
    df.to_csv(detailed, index=False)
    logger.info(f"Saved detailed results to {detailed}")

    # Compute and save summary metrics
    metrics = {}
    for f in ("roll","pitch","vfov"):
        errs = df[f"{f}_error"].to_numpy()
        metrics[f"mean_{f}_error"]   = errs.mean()
        metrics[f"median_{f}_error"] = np.median(errs)
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
        help="Path to GT CSV (cols: fname, roll, pitch, vfov, width, height)"
    )
    parser.add_argument(
        "--suffix", default="jpg",
        help="File extension (without dot) for output filenames (default: jpg)"
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
        help="Error thresholds in degrees for AUC computation (default: 1,5,10)"
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir    = Path(args.output_dir)
    pred_csv = out_dir / f"{input_json.stem}_predictions.csv"

    write_predictions_csv(
        input_json=input_json,
        pred_csv=pred_csv,
        suffix=args.suffix,
        width=args.width,
        height=args.height
    )

    evaluate_from_csv(
        gt_csv=Path(args.gt_csv),
        pred_csv=pred_csv,
        output_dir=out_dir,
        thresholds=args.thresholds
    )

if __name__ == "__main__":
    main()