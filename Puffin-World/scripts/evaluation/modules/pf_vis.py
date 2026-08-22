"""Per-view perspective-field (PF) visualization built from the OFFLINE VLM
camera annotations — the same construction as the model's
physical_propagation='offline' path (model._apply_gt_camera_params):
"roll pitch vfov k1" strings + crop-consistent intrinsics -> up / latitude
fields, rendered as overlays on the corresponding views."""
import io
import math
import os

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.camera.geometry.camera import SimpleRadial
from scripts.camera.geometry.gravity import Gravity
from scripts.camera.geometry.perspective_fields import get_perspective_field
from scripts.camera.utils.conversions import fov2focal
from scripts.camera.visualization.viz2d import (
    plot_images,
    plot_latitudes,
    plot_vector_fields,
)

from .media import _tensor_frame_to_uint8_hwc, save_frames_as_gif
from .parsing import _parse_3x3


def build_offline_pf(gt_cam_param_str, H, W, cam_intr_str=None):
    """(up [2,H,W], lat [H,W]) in radians from one "roll pitch vfov k1" string.

    Mirrors model._apply_gt_camera_params: focal / principal point come from
    the per-view post-crop intrinsics when available (crop-consistent),
    otherwise from the annotated vfov with a centered principal point;
    missing / unparsable captions fall back to the default camera
    (roll=0, pitch=0, vfov=90 deg, k1=0).
    """
    tokens = (str(gt_cam_param_str).strip().split()
              if gt_cam_param_str is not None else [])
    if len(tokens) == 4:
        roll, pitch, vfov, k1 = (float(x) for x in tokens)
    else:
        roll, pitch, vfov, k1 = 0.0, 0.0, math.radians(90.0), 0.0

    if cam_intr_str is not None:
        K = _parse_3x3(str(cam_intr_str))
        f = float(K[1, 1])
        px, py = float(K[0, 2]), float(K[1, 2])
    else:
        f = fov2focal(torch.tensor(vfov), H)
        px, py = W / 2.0, H / 2.0

    params = torch.tensor([W, H, f, f, px, py, k1, 0.0]).float()
    camera = SimpleRadial(params).float().scale(torch.Tensor([1, 1]))
    gravity = Gravity.from_rp(torch.tensor(roll).float(),
                              torch.tensor(pitch).float())
    up_field, lat_field = get_perspective_field(
        camera, gravity, use_up=True, use_latitude=True)
    return up_field[0], lat_field[0, 0]


def render_pf_overlay_frame(image_tensor, pf_field, kind, out_size=None):
    """Rasterize ONE PF overlay (up vectors / latitude contours) on one view.

    image_tensor: [3, H, W] in [-1, 1]. Returns uint8 [H', W', 3]; when
    out_size=(W, H) is given the raster is resized to it so GIF frames stay
    uniform across views.
    """
    img = _tensor_frame_to_uint8_hwc(image_tensor).astype(np.float32) / 255.0
    fig = plot_images([img])
    if kind == 'up':
        plot_vector_fields([pf_field], axes=fig.axes)
    else:
        plot_latitudes([pf_field], is_radians=True, axes=fig.axes)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    pil = Image.open(buf).convert('RGB')
    if out_size is not None:
        pil = pil.resize(out_size, Image.LANCZOS)
    return np.asarray(pil)


def diagonal_mix(lat_frame, up_frame, blend_px=2.0):
    """Seamless diagonal composite of the two PF overlays for ONE view:
    the latitude panel fills the UPPER-LEFT half and the up-field panel the
    LOWER-RIGHT half, split by the top-right -> bottom-left diagonal with a
    ~2 px anti-aliased transition (no visible seam)."""
    lat = lat_frame.astype(np.float32)
    up = np.asarray(up_frame, dtype=np.float32)
    if up.shape != lat.shape:
        up = np.asarray(
            Image.fromarray(up_frame).resize((lat.shape[1], lat.shape[0]),
                                             Image.LANCZOS), dtype=np.float32)
    H, W = lat.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # signed distance (in px) to the line x/W + y/H = 1 (TR -> BL diagonal)
    d = (xx / W + yy / H - 1.0) * (H * W) / np.hypot(H, W)
    alpha = np.clip(0.5 - d / max(blend_px, 1e-3), 0.0, 1.0)[..., None]
    return np.clip(lat * alpha + up * (1.0 - alpha), 0, 255).astype(np.uint8)


def save_pf_visualizations(frames, gt_cam_param_strs, cam_intr_strs,
                           out_dir, fps=4, durations=None):
    """Offline-PF up / latitude overlays for EVERY view (incl. frame 0).

    Saves per-view PNGs (<v>_pf_up.png / <v>_pf_lat.png) and two GIFs
    (pf_up.gif / pf_lat.gif) into out_dir.

    Args:
        frames:            list[T] of [3, H, W] tensors in [-1, 1]
                           (frame 0 = GT anchor, the rest = generated views).
        gt_cam_param_strs: list[>=T] of "roll pitch vfov k1" strings.
        cam_intr_strs:     list[>=T] of 3x3 intrinsics strings, or None.
        out_dir:           output directory (created if missing).
        fps:               GIF frame rate.

    Returns:
        (up_frames, lat_frames): per-view uint8 [H, W, 3] arrays, for
        downstream composition into the combined row GIF.
    """
    os.makedirs(out_dir, exist_ok=True)
    T = min(len(frames), len(gt_cam_param_strs))
    _, H, W = frames[0].shape

    up_frames, lat_frames, mix_frames = [], [], []
    for v in range(T):
        intr = cam_intr_strs[v] if cam_intr_strs is not None else None
        up, lat = build_offline_pf(gt_cam_param_strs[v], H, W, intr)
        up_frame = render_pf_overlay_frame(frames[v], up, 'up', out_size=(W, H))
        lat_frame = render_pf_overlay_frame(frames[v], lat, 'lat', out_size=(W, H))
        mix_frame = diagonal_mix(lat_frame, up_frame)
        Image.fromarray(up_frame).save(os.path.join(out_dir, f"{v:02d}_pf_up.png"))
        Image.fromarray(lat_frame).save(os.path.join(out_dir, f"{v:02d}_pf_lat.png"))
        Image.fromarray(mix_frame).save(os.path.join(out_dir, f"{v:02d}_pf_mix.png"))
        up_frames.append(up_frame)
        lat_frames.append(lat_frame)
        mix_frames.append(mix_frame)

    save_frames_as_gif(up_frames, os.path.join(out_dir, "pf_up.gif"),
                       fps=fps, durations=durations)
    save_frames_as_gif(lat_frames, os.path.join(out_dir, "pf_lat.gif"),
                       fps=fps, durations=durations)
    save_frames_as_gif(mix_frames, os.path.join(out_dir, "pf_mix.gif"),
                       fps=fps, durations=durations)
    return up_frames, lat_frames, mix_frames
