"""Synthetic / combo camera trajectories and chunk-relative ray-map rebuild
for the multi-view eval scripts."""
import math

import numpy as np
import torch
from einops import rearrange

from src.dust3r.datasets.base.base_multiview_dataset import get_ray_map
from .parsing import _parse_3x3, _parse_4x4


# ---------------------------------------------------------------------------
# Custom trajectories (--cus_traj)
# ---------------------------------------------------------------------------
# 24 combinations: {roll/pitch/yaw} x {+/-} x {forward/back/left/right}.
# Encoding: '<rot_letter><sign>t<trans_letter>' (e.g. 'r+tl', 'p-tf', 'y-tb').
ROT_AXIS_MAP = {'r': 'z', 'p': 'x', 'y': 'y'}  # roll=Z, pitch=X, yaw=Y (camera frame)
TRANS_DIR_MAP = {
    'f': np.array([0.0, 0.0,  1.0]),  # forward  (+Z in OpenCV camera frame)
    'b': np.array([0.0, 0.0, -1.0]),  # backward
    'l': np.array([-1.0, 0.0, 0.0]),  # left
    'r': np.array([ 1.0, 0.0, 0.0]),  # right
}
CUSTOM_COMBOS_ALL = [
    f"{r}{s}t{d}"
    for r in ('r', 'p', 'y')
    for s in ('+', '-')
    for d in ('f', 'b', 'l', 'r')
]  # 24 entries: rotation + translation simultaneously
CUSTOM_COMBOS_ROT = [
    f"{r}{s}"
    for r in ('r', 'p', 'y')
    for s in ('+', '-')
]  # 6 entries: rotation only (translation = 0)
CUSTOM_COMBOS_TRANS = [
    f"t{d}"
    for d in ('f', 'b', 'l', 'r')
]  # 4 entries: translation only (rotation = 0)
CUSTOM_COMBOS_360 = ['y+360', 'y-360']
# 2 entries: yaw-only full-circle orbit. Step defaults to 360/num_views to
# uniformly sample the circle (matches `full_circle` in pano dataset
# construction: scripts/.../create_dataset_from_pano_trajectory.py).
# Suffix '360' is a label only — build_custom_trajectory parses combo[:2].
CUSTOM_COMBOS_RETURN = [
    f"{r}{s}{o}"
    for r in ('r', 'p', 'y')
    for s, o in (('+', '-'), ('-', '+'))
]  # 6 entries: in-place GO-AND-RETURN rotation ('r+-' = roll out in +, back
#    to the anchor by the last view). First sign = outbound direction; the
#    trailing opposite sign is a label. Same triangular constant-step profile
#    as --combo_traj chunks (_triangular_step_offset).
CUSTOM_COMBOS_POOL = {
    'all':        CUSTOM_COMBOS_ALL,
    'only_rot':   CUSTOM_COMBOS_ROT,
    'only_trans': CUSTOM_COMBOS_TRANS,
    '360':        CUSTOM_COMBOS_360,
    'return_rot': CUSTOM_COMBOS_RETURN,
}


def _axis_rot(axis, angle_rad):
    """3x3 right-hand rotation matrix around the named camera-frame axis."""
    import math as _math
    c, s = _math.cos(angle_rad), _math.sin(angle_rad)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"unknown axis {axis!r}")


def _compose_ypr_c2w(roll_deg, pitch_deg, yaw_deg):
    """c2w rotation under the puffin_omni LOADER's emitted convention:
    R = Ry(yaw) @ Rx(pitch) @ Rz(roll) — the transpose of the cameras.json
    matrix, numerically verified against the dataset (pitch = elevation about
    the post-azimuth horizontal axis, roll = innermost in-plane spin). NOTE:
    puffin_omni.euler_to_textbook_c2w (Rz@Rx@Ry) is a DEAD helper the loader
    never calls; using it here mislabels pitch as view-axis roll at large yaw."""
    r, p, y = (math.radians(roll_deg), math.radians(pitch_deg),
               math.radians(yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return Ry @ Rx @ Rz


def _decompose_rpy_c2w(R):
    """Invert _compose_ypr_c2w: R = Ry(yaw)@Rx(pitch)@Rz(roll).
    Returns (roll, pitch, yaw) in radians."""
    import math as _math
    p = _math.asin(max(-1.0, min(1.0, -float(R[1, 2]))))
    r = _math.atan2(float(R[1, 0]), float(R[1, 1]))
    y = _math.atan2(float(R[0, 2]), float(R[2, 2]))
    return r, p, y


def build_custom_trajectory(combo, num_views, step_deg=2.0, step_dist=2.0,
                            base_c2w=None):
    """Build a list of T 4x4 c2w matrices for a custom trajectory.

    Three combo formats are accepted (one per `--cus_traj_type`):
      - rot+trans   : '<r|p|y>{+|-} t <f|b|l|r>' e.g. 'r+tl' (4 chars)
      - rot only    : '<r|p|y>{+|-}'             e.g. 'r+'   (2 chars)
      - trans only  : 't<f|b|l|r>'               e.g. 'tl'   (2 chars)

    View 0 is identity; view t rotates by sign * step_deg * t degrees (if rot
    present) and translates by step_dist * t along the chosen camera-frame
    direction (if trans present).

    Rotation axis semantics: when `base_c2w` (the sample's view-0 GT c2w) is
    given, the rotation follows the LOADER-emitted euler convention
    R = Ry(yaw) @ Rx(pitch) @ Rz(roll) (see _compose_ypr_c2w; NOT the dead
    euler_to_textbook_c2w helper): view-0's orientation is decomposed and ONLY
    the selected euler angle is advanced, exactly like the GT mono/full_circle
    trajectories — pitch stays a true elevation sweep and yaw a gravity-axis
    pan even when the base view has roll/pitch/yaw offsets. Poses are emitted
    ABSOLUTE (world-anchored at base_c2w) so the gravity-referenced keyboard
    overlay classifies them correctly; ray maps are built relative to poses[0],
    leaving the model conditioning unchanged. Without base_c2w it falls back
    to the legacy identity-anchored camera-axis behavior.

    Rotation default 2°/view × 8 views = 16° — within puffin_omni's
    roll/pitch/yaw distribution. Translation default 2/view × 8 views = 16 —
    RE10K-scale (its c2w translations span ~20).
    """
    has_rot   = combo[0] in ('r', 'p', 'y')
    has_trans = 't' in combo
    # go-and-return ('r+-' / 'p-+' ...): triangular profile instead of linear
    is_return = (has_rot and not has_trans and len(combo) == 3
                 and combo[2] in ('+', '-'))

    if has_rot:
        rot_letter, sign_ch = combo[0], combo[1]
        sign = +1.0 if sign_ch == '+' else -1.0
        axis = ROT_AXIS_MAP[rot_letter]
    if has_trans:
        trans_letter = combo[combo.index('t') + 1]
        trans_dir = TRANS_DIR_MAP[trans_letter]
    ret_offsets = (_triangular_step_offset(num_views, step_deg, sign)
                   if is_return else None)

    anchored = base_c2w is not None
    if anchored:
        B = np.asarray(base_c2w, dtype=np.float64)
        R0, t0 = B[:3, :3], B[:3, 3]
        r0, p0, y0 = _decompose_rpy_c2w(R0)
        base_deg = {'r': math.degrees(r0), 'p': math.degrees(p0),
                    'y': math.degrees(y0)}

    poses = []
    for t in range(num_views):
        if is_return:
            theta = float(ret_offsets[t])
        elif has_rot:
            theta = sign * step_deg * t
        if anchored and has_rot:
            # GT-euler-axis sweep, WORLD-ANCHORED at the sample's view-0 pose:
            # advance ONLY the selected euler angle of the base orientation.
            # Absolute poses keep the gravity-referenced keyboard overlay
            # correct (same reason combo poses are world-anchored); ray maps
            # are built relative to poses[0] so conditioning is unaffected.
            ang = dict(base_deg)
            ang[rot_letter] += theta
            R = _compose_ypr_c2w(ang['r'], ang['p'], ang['y'])
        elif anchored:
            R = R0.copy()
        elif is_return:
            R = _axis_rot(axis, math.radians(theta))
        else:
            R = _axis_rot(axis, math.radians(theta)) if has_rot \
                else np.eye(3, dtype=np.float64)
        T_cam = (trans_dir * (step_dist * t)) if has_trans \
            else np.zeros(3, dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        # camera-frame translation, composed onto the base pose when anchored
        c2w[:3, 3] = (t0 + R0 @ T_cam) if anchored else T_cam
        poses.append(c2w)
    return poses


# ---------------------------------------------------------------------------
# Multi-chunk combo trajectories (--combo_traj): one motion per chunk.
# ---------------------------------------------------------------------------
def _triangular_step_offset(n, step_deg, sign):
    """Per-frame angles for a GO-AND-RETURN rotation at a CONSTANT step of
    step_deg degrees per view: 0 at frame 0, rising +step_deg per view to the
    middle, then falling step_deg per view back to 0 at frame n-1 (so the chunk
    returns exactly to its anchor). Peak = step_deg * (n-1)/2 (with even n the
    two middle frames plateau one step below it)."""
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    half = (n - 1) / 2.0
    tri = half - np.abs(np.arange(n, dtype=np.float64) - half)   # 0..half..0, +/-1 per step
    return sign * step_deg * tri


def build_combo_chunk_pose(motion, gt_window_strs, num_views, step_deg):
    """Per-chunk c2w pose strings for one combo entry.

    motion 't'          -> the GT window poses unchanged (original trajectory).
    motion '<r|p|y><s>' -> an in-place GO-AND-RETURN rotation at a CONSTANT
                           step of step_deg degrees per view: rises
                           step_deg/view to the middle, then falls back to 0
                           at the last view (returns to the anchor). Poses
                           are WORLD-ANCHORED at the window's first GT pose
                           (camera-frame rotation composed onto it, zero
                           translation), so the stitched timeline stays in
                           ONE reference frame — window-relative poses blew
                           up the keyboard overlay's sequence-max thresholds
                           through garbage cross-frame pairs and every key
                           went dark. Ray maps are window-relative anyway,
                           so the model conditioning is unchanged.
                           Peak = step_deg * (num_views-1)/2.
    """
    if motion == "t":
        return list(gt_window_strs)

    kind = motion[0]
    sign = +1.0 if motion[1] == "+" else -1.0
    axis = ROT_AXIS_MAP[kind]
    offsets = _triangular_step_offset(num_views, step_deg, sign)  # degrees, fixed step
    from scripts.evaluation.modules.parsing import _parse_4x4
    anchor = np.asarray(_parse_4x4(str(gt_window_strs[0])), dtype=np.float64)
    poses = []
    for t in range(num_views):
        rel = np.eye(4, dtype=np.float64)
        rel[:3, :3] = _axis_rot(axis, math.radians(float(offsets[t])))
        poses.append(np.array2string(anchor @ rel))
    return poses


def build_custom_inputs(combo, data_samples, cam_values, step_deg, step_dist):
    """Build cam_values and cam_pose strings for a custom synthetic trajectory.

    Reuses each sample's view-0 intrinsics to render ray maps; the image
    resolution is inferred from the original cam_values shape. Returns the
    (cam_values_new, cam_pose_strs_new): per-sample ray-map tensors plus the
    matching 4x4 pose strings (what the model actually conditions on).
    """
    new_cam_values = []
    new_cam_pose = []
    for b, views in enumerate(cam_values):
        cam_intr_strs = data_samples[b].get('cam_intrinsics', None)
        assert cam_intr_strs is not None, (
            "--cus_traj requires the dataset to emit cam_intrinsics."
        )
        K_intr = _parse_3x3(cam_intr_strs[0])
        T_views = len(views)
        _, H, W = views[0].shape
        # GT-euler-axis rotations: anchor the synthetic sweep on the sample's
        # view-0 orientation so pitch/yaw follow the dataset's original axes.
        base_pose_strs = data_samples[b].get('cam_pose', None)
        base_c2w = (_parse_4x4(str(base_pose_strs[0]))
                    if base_pose_strs else None)
        poses = build_custom_trajectory(combo, T_views, step_deg, step_dist,
                                        base_c2w=base_c2w)
        c2w_0 = poses[0]  # identity (legacy) or the sample's anchored GT base
        sample_views, sample_poses = [], []
        for t in range(T_views):
            ray_map = get_ray_map(c2w_0, poses[t], K_intr, H, W)   # (H, W, 6)
            tens = torch.from_numpy(ray_map).to(
                dtype=views[0].dtype, device=views[0].device,
            )
            tens = rearrange(tens, 'h w c -> c h w').contiguous()
            sample_views.append(tens)
            sample_poses.append(np.array2string(poses[t]))
        new_cam_values.append(sample_views)
        new_cam_pose.append(sample_poses)
    return new_cam_values, new_cam_pose


def rebuild_chunk_ray_maps(cam_pose_strs_window, cam_intr_strs_window,
                            H, W, dtype, device):
    """Rebuild ray_maps for a chunk window relative to the window's first view.

    The dataset emits ray_maps relative to the GLOBAL view-0. When chunking,
    each chunk's first view becomes the new reference, so we recompute ray
    maps using `get_ray_map(c2w_window_0, c2w_window_t, K, H, W)`.

    Args:
        cam_pose_strs_window: list[T_chunk] of 4x4 c2w strings.
        cam_intr_strs_window: list[T_chunk] of 3x3 K strings.
        H, W:                 spatial resolution of the ray_map tensors.
        dtype, device:        target tensor dtype / device.

    Returns:
        list[T_chunk] of [6, H, W] tensors (window-relative ray maps).
    """
    c2w_mats = [_parse_4x4(s) for s in cam_pose_strs_window]
    K_mats = [_parse_3x3(s) for s in cam_intr_strs_window]
    c2w_0 = c2w_mats[0]
    out = []
    for t in range(len(c2w_mats)):
        rm = get_ray_map(c2w_0, c2w_mats[t], K_mats[t], H, W)   # (H, W, 6)
        tens = torch.from_numpy(rm).to(dtype=dtype, device=device)
        tens = rearrange(tens, 'h w c -> c h w').contiguous()
        out.append(tens)
    return out
