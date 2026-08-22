"""Statistics for a camera-caption annotated dataset.

Recursively reads ALL caption JSONs under --camera_root (the layout written by
scripts/annotation/camera_caption_aoss.py, any dataset), collects roll / pitch
/ vfov (stored in radians, converted to degrees here), and plots three
histograms in a 1x3 academic-style figure:

    roll  : [-45, 45] deg
    pitch : [-45, 45] deg
    fov   : [ 20, 105] deg

Bin width is user-defined (--bin_width_deg, default 10 deg — e.g. [-5, 5) for
roll/pitch, [20, 30) for fov). Bar height = fraction of all valid samples.
JSON reading is multi-threaded (--num_workers).

Example:
    python scripts/annotation/stat_camera_captions.py \\
        --camera_root /mnt/.../camera_caption/mvs_synth \\
        --output output/mvs_synth_cam_stats.png
"""
import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


# Fixed histogram ranges (degrees), per the dataset construction conventions.
PARAM_SPECS = [
    # (key, pretty label, range lo, range hi, color)
    ("roll",  "Roll (°)",  -45.0,  45.0, "#4C72B0"),   # muted blue
    ("pitch", "Pitch (°)", -45.0,  45.0, "#DD8452"),   # muted orange
    ("vfov",  "FoV (°)",    20.0, 105.0, "#55A868"),   # muted green
]


def read_one(path):
    """Read a single caption JSON -> (roll_deg, pitch_deg, vfov_deg) or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cap = json.load(f)
        if not cap.get("parse_ok", True):
            return None
        return (
            math.degrees(float(cap["roll"])),
            math.degrees(float(cap["pitch"])),
            math.degrees(float(cap["vfov"])),
        )
    except Exception:
        return None


def collect_values(camera_root, num_workers):
    """Find all caption JSONs under camera_root and read them in parallel."""
    paths = [str(p) for p in Path(camera_root).rglob("*.json")]
    if not paths:
        raise FileNotFoundError(f"No .json files found under {camera_root}")
    print(f"Found {len(paths)} caption JSONs under {camera_root}")

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for r in tqdm(pool.map(read_one, paths, chunksize=256),
                      total=len(paths), desc="Reading JSONs"):
            results.append(r)

    n_bad = sum(1 for r in results if r is None)
    vals = np.array([r for r in results if r is not None], dtype=np.float64)
    print(f"Valid: {len(vals)}  |  skipped (parse_ok=False / unreadable): {n_bad}")
    return vals  # [N, 3] -> roll, pitch, vfov in degrees


def make_bin_edges(lo, hi, width):
    """Edges from lo with the given width; last bin clipped at hi."""
    edges = list(np.arange(lo, hi, width))
    edges.append(hi)
    return np.array(edges, dtype=np.float64)


def rounded_top_bar(x0, w, h, rx, ry):
    """Bar outline with quarter-bezier rounded TOP corners only.

    rx / ry are the corner radii in x / y data units; the bottom stays square
    so bars sit flat on the axis.
    """
    x1 = x0 + w
    if rx <= 0 or ry <= 0:
        verts = [(x0, 0.0), (x0, h), (x1, h), (x1, 0.0), (x0, 0.0)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
                 MplPath.LINETO, MplPath.CLOSEPOLY]
        return MplPath(verts, codes)
    ry = min(ry, h)  # short bars: don't let the arc dip below the baseline
    verts = [
        (x0, 0.0),
        (x0, h - ry),
        (x0, h), (x0 + rx, h),      # top-left quarter arc
        (x1 - rx, h),
        (x1, h), (x1, h - ry),      # top-right quarter arc
        (x1, 0.0),
        (x0, 0.0),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def plot_histograms(vals, bin_width, out_path, corner_round=0.35):
    n_total = len(vals)
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (key, label, lo, hi, color), col in zip(
            axes, PARAM_SPECS, range(3)):
        data = vals[:, col]
        in_range = (data >= lo) & (data <= hi)
        n_out = int((~in_range).sum())

        edges = make_bin_edges(lo, hi, bin_width)
        counts, _ = np.histogram(data[in_range], bins=edges)
        fracs = counts / max(n_total, 1)

        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges) * 0.92          # small gap between bars

        x_lo, x_hi = lo - bin_width * 0.3, hi + bin_width * 0.3
        y_max = max(float(fracs.max()), 1e-6) * 1.18
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, y_max)

        # Convert the x-radius into a y-radius that LOOKS the same on screen
        # (data units differ wildly between the two axes).
        pos = ax.get_position()
        ax_aspect = (pos.width * fig.get_figwidth()) / \
                    (pos.height * fig.get_figheight())
        y_per_x = (y_max / (x_hi - x_lo)) * ax_aspect

        for x_c, h, w in zip(centers, fracs, widths):
            if h <= 0:
                continue
            rx = corner_round * w / 2.0
            ax.add_patch(PathPatch(
                rounded_top_bar(x_c - w / 2.0, w, h, rx, rx * y_per_x),
                facecolor=color, alpha=0.85,
                edgecolor="white", linewidth=0.8, zorder=3))

        ax.set_xlabel(label)
        ax.set_ylabel("Proportion")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        # Summary stats over IN-RANGE values only: out-of-range entries are
        # caption-parse failures (e.g. years matched as angles), not real
        # samples — a couple of them would otherwise blow up the std.
        d_in = data[in_range]
        mean, med, std = d_in.mean(), np.median(d_in), d_in.std()
        ax.set_title(f"{label.split(' ')[0]}  "
                     f"(μ={mean:.1f}°, med={med:.1f}°, σ={std:.1f}°)")

        print(f"[{key}] (in-range) mean={mean:.2f}°  median={med:.2f}°  "
              f"std={std:.2f}°  min={d_in.min():.2f}°  max={d_in.max():.2f}°  "
              f"out-of-range={n_out} ({n_out / n_total:.2%}, "
              f"raw min={data.min():.2f}° max={data.max():.2f}°)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--camera_root", required=True, type=str,
                        help="Root of an annotated dataset (caption JSONs).")
    parser.add_argument("--output", default="output/camera_caption_stats.png",
                        type=str, help="Output PNG path.")
    parser.add_argument("--bin_width_deg", type=float, default=10.0,
                        help="Histogram bin width in degrees (default 10).")
    parser.add_argument("--corner_round", type=float, default=0.35,
                        help="Roundness of bar tops in [0, 1]: 0 = square "
                             "corners, 1 = fully rounded (half-bar-width "
                             "radius). Default 0.35 (slight arc).")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Threads for JSON reading.")
    args = parser.parse_args()

    vals = collect_values(args.camera_root, args.num_workers)
    corner_round = min(max(args.corner_round, 0.0), 1.0)
    plot_histograms(vals, args.bin_width_deg, args.output, corner_round)


if __name__ == "__main__":
    main()
