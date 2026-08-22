"""Image / GIF / heatmap / trajectory-plot saving and generation-output
normalization for the multi-view eval scripts."""
import os

import numpy as np
import torch
from PIL import Image
from einops import rearrange

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_image_tensor(tensor, path):
    tensor = tensor.to(torch.float32)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    img_np = rearrange(tensor, 'c h w -> h w c').detach().cpu().numpy()
    img_np = (img_np * 127.5 + 128.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def _tensor_frame_to_uint8_hwc(tensor):
    """Convert a single image tensor [C,H,W] in [-1, 1] to uint8 [H,W,C]."""
    t = tensor.to(torch.float32)
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = rearrange(t, 'c h w -> h w c').detach().cpu().numpy()
    return (arr * 127.5 + 128.0).clip(0, 255).astype(np.uint8)


def save_frames_as_gif(frames, path, fps=10, durations=None):
    """Save a list of [C,H,W] tensors / arrays as an animated GIF.

    Args:
        frames:    iterable of image tensors (or HxWx3 uint8 arrays).
        path:      output .gif path.
        fps:       target frames per second (PIL stores per-frame ms duration).
        durations: optional per-frame durations in ms (overrides fps for the
                   frames it covers; padded with the fps duration when
                   shorter than the frame list).
    """
    if not frames:
        return
    pil_frames = []
    for fr in frames:
        if isinstance(fr, torch.Tensor):
            arr = _tensor_frame_to_uint8_hwc(fr)
        elif isinstance(fr, np.ndarray):
            arr = fr.astype(np.uint8) if fr.dtype != np.uint8 else fr
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
        else:
            raise TypeError(f"unsupported frame type: {type(fr)}")
        pil_frames.append(Image.fromarray(arr))

    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    if durations is not None:
        dur = [max(1, int(d)) for d in durations[:len(pil_frames)]]
        dur += [duration_ms] * (len(pil_frames) - len(dur))
    else:
        dur = duration_ms
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=dur,
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_frames_as_mp4(frames, path, fps=10):
    """Write frames ([3,H,W] tensors in [-1,1] or uint8 HWC arrays) as an
    H.264 mp4 (imageio-ffmpeg backend; OpenCV fallback). Even dimensions are
    enforced by a 1-px crop when needed (yuv420p requirement)."""
    imgs = []
    for fr in frames:
        arr = (_tensor_frame_to_uint8_hwc(fr) if isinstance(fr, torch.Tensor)
               else np.asarray(fr))
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        h, w = arr.shape[:2]
        imgs.append(arr[: h - h % 2, : w - w % 2])
    if not imgs:
        return
    try:
        import imageio
        imageio.mimwrite(path, imgs, fps=fps, codec="libx264", quality=8,
                         pixelformat="yuv420p", macro_block_size=1)
    except Exception:
        import cv2
        h, w = imgs[0].shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
        for arr in imgs:
            vw.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        vw.release()


def _save_single_heatmap(data, path, cmap='turbo'):
    valid = np.isfinite(data) & (data > 0)
    if valid.any():
        vmin = float(np.percentile(data[valid], 2))
        vmax = float(np.percentile(data[valid], 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_depth_heatmap(depth_hw, path):
    """Save both depth and disparity heatmaps side by side.

    Saves two files:
      - <name>_depth.png  — log-depth, percentile-clamped to suppress sky noise
      - <name>_disp.png   — disparity (1/depth), naturally smooth from the network

    Args:
        depth_hw: torch.Tensor or np.ndarray of shape [H, W], depth values.
        path:     output image path (the '_depth' suffix is auto-managed).
    """
    if isinstance(depth_hw, torch.Tensor):
        depth_hw = depth_hw.detach().cpu().float().numpy()
    depth = np.asarray(depth_hw, dtype=np.float32)

    valid = np.isfinite(depth) & (depth > 0)

    base, ext = os.path.splitext(path)
    if base.endswith('_depth'):
        base = base[:-6]

    disp = np.zeros_like(depth)
    disp[valid] = 1.0 / depth[valid]
    _save_single_heatmap(disp, f"{base}_disp{ext}")

    depth_vis = depth.copy()
    if valid.any():
        p98 = float(np.percentile(depth[valid], 98))
        depth_vis = np.clip(depth_vis, 1e-3, p98)
    depth_vis[valid] = np.log(depth_vis[valid])
    _save_single_heatmap(depth_vis, f"{base}_depth{ext}")


def depth_to_visionbanana_frame(depth_hw):
    """Encode one [H, W] depth map as a visionbanana RGB uint8 frame
    [H, W, 3] — the dataloader's depth encoding (invalid pixels render
    white), so scalar depths visualize exactly like `depth_visionbanana`."""
    # Local import: keeps this module importable without the dust3r deps.
    from src.dust3r.datasets.base.base_multiview_dataset import (
        depth_to_visionbanana,
    )
    if isinstance(depth_hw, torch.Tensor):
        depth_hw = depth_hw.detach().cpu().float().numpy()
    depth = np.asarray(depth_hw, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    vb = depth_to_visionbanana(depth, valid)   # [H, W, 3] in [-1, 1]
    return (vb * 127.5 + 128.0).clip(0, 255).astype(np.uint8)


def save_uint8_image(arr, path):
    """Save a uint8 [H, W, 3] array as an image file."""
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


def compose_row_gif(panel_frame_lists, path, fps=10, gap=8, durations=None,
                    mp4_path=None, mp4_fps=None):
    """Compose N per-panel frame sequences into ONE side-by-side row GIF.

    Args:
        panel_frame_lists: list of frame lists (one per panel, left to right);
            frames may be [C,H,W] tensors in [-1,1] or uint8 [H,W,3] arrays.
            Panels after the first are resized to the first panel's frame
            size; all sequences are truncated to the shortest one.
        path: output .gif path.
        fps:  GIF frame rate.
        gap:  white spacing (px) between panels.
        durations: optional per-frame durations in ms (see save_frames_as_gif).
        mp4_path: optionally also write the SAME row frames as an mp4.
        mp4_fps:  mp4 frame rate (default: fps).
    """
    panel_frame_lists = [fl for fl in panel_frame_lists if fl]
    if not panel_frame_lists:
        return
    n_frames = min(len(fl) for fl in panel_frame_lists)

    def to_uint8(fr):
        if isinstance(fr, torch.Tensor):
            return _tensor_frame_to_uint8_hwc(fr)
        arr = np.asarray(fr)
        return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr

    base = to_uint8(panel_frame_lists[0][0])
    Hb, Wb = base.shape[:2]
    gap_col = np.full((Hb, gap, 3), 255, dtype=np.uint8)

    frames = []
    for i in range(n_frames):
        panels = []
        for fl in panel_frame_lists:
            arr = to_uint8(fl[i])
            if arr.shape[:2] != (Hb, Wb):
                arr = np.asarray(
                    Image.fromarray(arr).resize((Wb, Hb), Image.LANCZOS))
            panels.append(arr)
        row = panels[0]
        for p in panels[1:]:
            row = np.concatenate([row, gap_col, p], axis=1)
        frames.append(row)

    save_frames_as_gif(frames, path, fps=fps, durations=durations)
    if mp4_path is not None:
        save_frames_as_mp4(frames, mp4_path, fps=mp4_fps or fps)


def normalize_gen_outputs_multi(images, B, T_tgt):
    """
    Normalize the outputs of model.generate_multi_view into a tensor
    of shape [B, T_tgt, C, H, W].

    We support the following formats returned by generate_multi_view:
      1) Tensor [B * T_tgt, C, H, W]
      2) Tensor [B, T_tgt, C, H, W]
      3) list[B][T_tgt] of [C, H, W] tensors
    """
    # Case 3: list[B][T_tgt] of [C,H,W]
    if isinstance(images, list):
        # stack -> [B, T_tgt, C, H, W]
        images = torch.stack(
            [torch.stack(views, dim=0) for views in images],
            dim=0,
        )

    assert torch.is_tensor(images), "generate_multi_view() must return a Tensor or a nested list of Tensors."

    if images.dim() == 4:
        # Assume shape is [B * T_tgt, C, H, W]
        N, C, H, W = images.shape
        assert N == B * T_tgt, f"Expected {B*T_tgt} images, but got {N}."
        images = images.view(B, T_tgt, C, H, W)
    elif images.dim() == 5:
        # Assume shape is [B, T_tgt, C, H, W]
        assert images.shape[0] == B, f"Batch size mismatch: expected {B}, got {images.shape[0]}."
        assert images.shape[1] == T_tgt, f"Number of views mismatch: expected {T_tgt}, got {images.shape[1]}."
    else:
        raise ValueError(f"Unexpected output dimension: {images.dim()}.")

    return images
