"""Camera-trajectory visualization: rainbow wireframe frustums along the
path plus an offset red trajectory line with an endpoint arrowhead, on a
clean white 3D canvas. Ported from the standalone camera_pose_vis.py
(replacing the old scatter/dashed-line plot); the pipeline-facing entry is
``visualize_camera_trajectory`` which accepts the eval scripts' stringified
poses directly."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def _validate_c2w_poses(poses: Iterable[np.ndarray]) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape (N, 4, 4), got {poses.shape}.")
    if poses.shape[0] == 0:
        raise ValueError("Expected at least one camera pose.")
    if not np.all(np.isfinite(poses)):
        raise ValueError("Poses contain NaN or infinite values.")
    return poses


def _frustum_corners(scale: float, z_sign: float = 1.0) -> np.ndarray:
    """Return camera-space points for a small wireframe frustum."""
    depth = scale
    half_w = scale * 0.55
    half_h = scale * 0.38
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [-half_w, -half_h, z_sign * depth],
            [half_w, -half_h, z_sign * depth],
            [half_w, half_h, z_sign * depth],
            [-half_w, half_h, z_sign * depth],
        ],
        dtype=np.float64,
    )


def _set_axes_equal(ax, points: np.ndarray, padding_ratio: float = 0.16) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5)
    if radius <= 1e-9:
        radius = 1.0
    radius *= 1.0 + padding_ratio

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0, pad=-2)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.72, 0.72, 0.72, 1.0))
        axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.85


def _axis_radius(points: np.ndarray, padding_ratio: float = 0.16) -> float:
    """The half-extent of the (cubic) plot the points will be framed in --
    mirrors _set_axes_equal so sizes derived from it are screen-stable."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    radius = float(np.max(maxs - mins) * 0.5)
    if radius <= 1e-9:
        radius = 1.0
    return radius * (1.0 + padding_ratio)


def _auto_frustum_scale(points: np.ndarray) -> float:
    """Frustum size as a FIXED fraction of the displayed axis cube (not of
    the trajectory length): every sample's cameras render at the same
    apparent size, whether the trajectory is near-static (where a
    span-based floor used to blow the frustums up past the whole plot) or
    long and single-axis (where span-based sizing shrank them)."""
    return _axis_radius(points) * 0.16


def _trajectory_offset_direction(poses: np.ndarray,
                                 view_direction: np.ndarray | None = None
                                 ) -> np.ndarray:
    """Direction to shift the trajectory line off the frustum row.

    Preferred: perpendicular to BOTH the mean path direction and the view
    direction -- maximal on-screen separation regardless of the camera
    layout (a camera-down offset used to land the line right on top of
    horizontal trajectories). Sign follows mean camera-down so the line
    stays on the 'below' side; falls back to camera-down when degenerate."""
    down = poses[:, :3, 1].mean(axis=0)
    if np.linalg.norm(down) <= 1e-9:
        down = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    down = down / np.linalg.norm(down)

    centers = poses[:, :3, 3]
    path = centers[-1] - centers[0]
    if view_direction is not None and np.linalg.norm(path) > 1e-9:
        perp = np.cross(path / np.linalg.norm(path), view_direction)
        if np.linalg.norm(perp) > 1e-6:
            perp = perp / np.linalg.norm(perp)
            if np.dot(perp, down) < 0:
                perp = -perp
            return perp
    return down


def _view_direction(elev: float, azim: float) -> np.ndarray:
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    direction = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ],
        dtype=np.float64,
    )
    return direction / np.linalg.norm(direction)


TRAJ_COLOR = "#c90000"
TRAJ_GLOW = "#ff6a55"


def _catmull_rom(points: np.ndarray, samples_per_seg: int = 16) -> np.ndarray:
    """Smooth the polyline with a centripetal-flavored Catmull-Rom spline
    (endpoint-clamped). Returns a densely sampled curve through all points."""
    n = len(points)
    if n < 3:
        return points.astype(np.float64)
    p = np.vstack([points[0], points, points[-1]]).astype(np.float64)
    curve = []
    for i in range(1, n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        m1 = (p2 - p0) * 0.5
        m2 = (p3 - p1) * 0.5
        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=(i == n - 1))
        t2, t3 = t * t, t * t * t
        h = (2 * t3 - 3 * t2 + 1)[:, None] * p1 + (t3 - 2 * t2 + t)[:, None] * m1 \
            + (-2 * t3 + 3 * t2)[:, None] * p2 + (t3 - t2)[:, None] * m2
        curve.append(h)
    return np.concatenate(curve, axis=0)


def _draw_comet_trajectory(ax, points: np.ndarray, head_size: float,
                           view_direction: np.ndarray) -> None:
    """Comet-styled trajectory: a smooth spline whose width and opacity ramp
    from a faint thin tail to a bold head, over a soft glow underlay, ending
    in a solid shaded 3D cone arrowhead."""
    curve = _catmull_rom(points)
    # leave room for the cone so the line doesn't poke past its tip
    tip = curve[-1]
    direction = curve[-1] - curve[-8 if len(curve) > 8 else -2]
    norm = np.linalg.norm(direction)
    if norm <= 1e-9:
        direction = view_direction
    else:
        direction = direction / norm
    cone_len = head_size * 1.45
    keep = np.linalg.norm(curve - tip, axis=1) > cone_len * 0.6
    keep[0] = True
    body = curve[keep] if keep.sum() >= 2 else curve

    segs = np.stack([body[:-1], body[1:]], axis=1)
    t = np.linspace(0.0, 1.0, len(segs))
    base = np.array(to_rgb(TRAJ_COLOR))
    glow = np.array(to_rgb(TRAJ_GLOW))

    # glow underlay
    glow_rgba = np.concatenate(
        [np.tile(glow, (len(segs), 1)), (0.06 + 0.22 * t)[:, None]], axis=1)
    ax.add_collection3d(Line3DCollection(
        segs, colors=glow_rgba, linewidths=3.0 + 8.0 * t,
        capstyle="round", zorder=9))
    # core comet body: hue drifts warm-orange tail -> deep-crimson head
    core_rgb = glow[None] * (1.0 - t)[:, None] * 0.85 + base[None] * t[:, None]
    core_rgba = np.concatenate(
        [core_rgb, (0.15 + 0.85 * t)[:, None]], axis=1)
    ax.add_collection3d(Line3DCollection(
        segs, colors=np.clip(core_rgba, 0, 1), linewidths=0.6 + 3.2 * t,
        capstyle="round", zorder=10))

    # origin dot: a soft pad marking where the journey starts
    ax.scatter(*points[0], s=26, c=[base], alpha=0.9,
               edgecolors="white", linewidths=0.8, zorder=11)

    # solid cone arrowhead with simple lambert shading
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(direction, up)
    if np.linalg.norm(side) <= 1e-9:
        side = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    side = side / np.linalg.norm(side)
    side2 = np.cross(direction, side)
    center = tip - direction * cone_len
    radius = head_size * 0.52
    ang = np.linspace(0.0, 2.0 * np.pi, 18)
    ring = (center[None] + radius * (np.cos(ang)[:, None] * side[None]
                                     + np.sin(ang)[:, None] * side2[None]))
    light = view_direction + up * 0.6
    light = light / np.linalg.norm(light)
    faces, face_colors = [], []
    for i in range(len(ang) - 1):
        a, b = ring[i], ring[i + 1]
        n_vec = np.cross(b - a, tip - a)
        nn = np.linalg.norm(n_vec)
        shade = 0.5
        if nn > 1e-12:
            shade = 0.5 + 0.6 * max(0.0, float(np.dot(n_vec / nn, light)))
        faces.append([a, b, tip])
        face_colors.append(np.clip(base * shade + 0.16, 0, 1))
        faces.append([a, b, center])     # base disc (two fans close the cone)
        face_colors.append(np.clip(base * 0.45, 0, 1))
    cone = Poly3DCollection(faces, facecolors=face_colors, edgecolors="none",
                            zorder=12)
    ax.add_collection3d(cone)


def visualize_camera_poses(
    poses: Iterable[np.ndarray],
    output_path: str | Path,
    *,
    stride: int = 1,
    frustum_scale: float | None = None,
    trajectory_offset: float | None = None,
    show_trajectory: bool = True,
    elev: float = 22.0,
    azim: float = -58.0,
    dpi: int = 200,
    figsize: tuple[float, float] = (3.0, 3.0),
    camera_forward: str = "+z",
) -> Path:
    """Render a camera pose sequence ((N, 4, 4) c2w) to a PNG image."""
    poses = _validate_c2w_poses(poses)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    if camera_forward not in {"+z", "-z"}:
        raise ValueError("camera_forward must be '+z' or '-z'.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    centers = poses[:, :3, 3]
    if frustum_scale is None:
        frustum_scale = _auto_frustum_scale(centers)
    if frustum_scale <= 0:
        raise ValueError(f"frustum_scale must be positive, got {frustum_scale}.")
    if trajectory_offset is None:
        trajectory_offset = frustum_scale * 2.1
    if trajectory_offset < 0:
        raise ValueError(f"trajectory_offset must be non-negative, got {trajectory_offset}.")

    z_sign = 1.0 if camera_forward == "+z" else -1.0
    corners_cam = _frustum_corners(frustum_scale, z_sign=z_sign)
    edges = (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    )

    frustum_indices = np.arange(0, len(poses), stride)
    if frustum_indices[-1] != len(poses) - 1:
        frustum_indices = np.append(frustum_indices, len(poses) - 1)

    segments = []
    segment_colors = []
    cmap = plt.get_cmap("rainbow")
    denom = max(len(frustum_indices) - 1, 1)
    for color_idx, pose_idx in enumerate(frustum_indices):
        pose = poses[pose_idx]
        corners_world = (pose[:3, :3] @ corners_cam.T).T + pose[:3, 3]
        color = cmap(color_idx / denom)
        for start, end in edges:
            segments.append([corners_world[start], corners_world[end]])
            segment_colors.append(color)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _style_axis(ax)

    if show_trajectory and len(centers) > 1:
        offset_direction = _trajectory_offset_direction(
            poses, view_direction=_view_direction(elev, azim))
        trajectory_centers = centers + offset_direction[None] * trajectory_offset
        _draw_comet_trajectory(
            ax,
            trajectory_centers,
            head_size=frustum_scale * 1.0,
            view_direction=_view_direction(elev, azim),
        )

    ax.add_collection3d(
        Line3DCollection(segments, colors=segment_colors, linewidths=1.0, alpha=0.9)
    )
    _set_axes_equal(ax, centers)
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path.resolve()


def _coerce_pose(pose) -> np.ndarray | None:
    """One pipeline pose (str / tensor / array, 4x4 or 3x4) -> 4x4 or None."""
    if isinstance(pose, str):
        clean = pose.replace("[", " ").replace("]", " ").replace("\n", " ")
        vals = [float(x) for x in clean.split()]
        if len(vals) == 16:
            pose = np.array(vals).reshape(4, 4)
        elif len(vals) == 12:
            pose = np.array(vals).reshape(3, 4)
        else:
            return None
    elif isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0.0, 0.0, 0.0, 1.0]])
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose


def visualize_camera_trajectory(poses, save_path):
    """Pipeline entry: render the trajectory of a pose LIST (stringified 4x4
    c2w / tensors / arrays, 3x4 accepted) as frustums + trajectory line."""
    mats = [m for m in (_coerce_pose(p) for p in poses) if m is not None]
    if not mats:
        print(f"[traj-vis] no parseable poses; skipped {save_path}")
        return
    visualize_camera_poses(np.stack(mats), save_path)
