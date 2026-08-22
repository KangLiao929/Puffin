"""3D reconstruction export (point cloud + camera frusta)
with three optional denoising stages: flying-pixel filter, cross-view
consistency vote, voxel denoise + downsample."""
import math

import numpy as np
import torch
import trimesh
import matplotlib
from PIL import Image

from .media import _tensor_frame_to_uint8_hwc
from .parsing import _parse_3x3, _parse_4x4


def _dilate3x3(mask):
    """One 3x3 binary dilation pass (numpy-only, separable)."""
    v = mask.copy()
    v[1:, :] |= mask[:-1, :]
    v[:-1, :] |= mask[1:, :]
    out = v.copy()
    out[:, 1:] |= v[:, :-1]
    out[:, :-1] |= v[:, 1:]
    return out


def _depth_edge_mask(depth, grad_thresh):
    """True near depth discontinuities (the pixels that unproject to 'flying
    points' streaking between foreground and background). Relative forward
    differences, dilated by 1 px so both sides of an edge are dropped."""
    gx = np.zeros_like(depth)
    gy = np.zeros_like(depth)
    gx[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
    gy[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
    rel = np.maximum(gx, gy) / np.maximum(np.abs(depth), 1e-6)
    return _dilate3x3(rel > grad_thresh)


def _voxel_denoise(pts, rgb, voxel_size, min_pts):
    """Voxel-grid denoise + downsample: drop voxels with < min_pts points
    (isolated specks), average points/colors inside surviving voxels (also
    thins doubled surfaces and evens out near/far view density)."""
    q = np.floor(pts / voxel_size).astype(np.int64)
    q -= q.min(axis=0)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)

    n_vox = counts.size
    centers = np.stack([
        np.bincount(inv, weights=pts[:, a], minlength=n_vox) for a in range(3)
    ], axis=1) / counts[:, None]
    colors = np.stack([
        np.bincount(inv, weights=rgb[:, a].astype(np.float64), minlength=n_vox)
        for a in range(3)
    ], axis=1) / counts[:, None]

    keep = counts >= min_pts
    return centers[keep], np.clip(np.round(colors[keep]), 0, 255).astype(np.uint8)


def _rescale_K(K, W, H):
    """Rescale post-crop intrinsics to the depth-map resolution (principal
    point assumed centered, as in the loaders)."""
    K = K.copy()
    ref_w, ref_h = 2.0 * K[0, 2], 2.0 * K[1, 2]
    if ref_w > 0 and ref_h > 0 and (abs(ref_w - W) > 1 or abs(ref_h - H) > 1):
        K[0] *= W / ref_w
        K[1] *= H / ref_h
    return K


def _pair_gauge_error(ref, mov, s, stride=6, max_pts=3000):
    """Median |log depth ratio| of view `mov` reprojected into view `ref`
    with mov's (and ref's) depths scaled by s. Views are dicts with
    depth [H,W] (0 = invalid), K, c2w. Returns (err, n_pts)."""
    depth = mov['depth']
    H, W = depth.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[yy, xx]
    ok = d > 0
    if ok.sum() < 50:
        return None, 0
    xs, ys, d = xx[ok].ravel(), yy[ok].ravel(), d[ok].ravel()
    if len(d) > max_pts:
        sel = np.random.default_rng(0).choice(len(d), max_pts, replace=False)
        xs, ys, d = xs[sel], ys[sel], d[sel]
    Km, Kr = mov['K'], ref['K']
    rays = np.stack([(xs + 0.5 - Km[0, 2]) / Km[0, 0],
                     (ys + 0.5 - Km[1, 2]) / Km[1, 1],
                     np.ones_like(d)], axis=1)
    pts_w = (rays * (s * d)[:, None]) @ mov['c2w'][:3, :3].T + mov['c2w'][:3, 3]
    Rr, tr = ref['c2w'][:3, :3], ref['c2w'][:3, 3]
    p = (pts_w - tr) @ Rr                       # w2c
    z = p[:, 2]
    front = z > 1e-9
    zs = np.where(front, z, 1.0)
    u = np.floor(p[:, 0] / zs * Kr[0, 0] + Kr[0, 2]).astype(np.int64)
    v = np.floor(p[:, 1] / zs * Kr[1, 1] + Kr[1, 2]).astype(np.int64)
    Hr, Wr = ref['depth'].shape
    inb = front & (u >= 0) & (u < Wr) & (v >= 0) & (v < Hr)
    if inb.sum() < 50:
        return None, 0
    d_ref = ref['depth'][v[inb], u[inb]]
    good = d_ref > 0
    if good.sum() < 50:
        return None, 0
    err = np.abs(np.log(z[inb][good] / (s * d_ref[good])))
    return float(np.median(err)), int(good.sum())


def _pair_gauge_ratio(ref, mov, stride=6, max_pts=3000):
    """Median z_pred / d_ref of view `mov` reprojected into view `ref`
    (raw depths): the multiplicative gauge of mov's depth RELATIVE to
    ref's. Scale-invariant to any global depth multiplier (both z and
    d_ref scale with it). Returns (ratio, n_pts) or (None, 0)."""
    depth = mov['depth']
    H, W = depth.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[yy, xx]
    ok = d > 0
    if ok.sum() < 50:
        return None, 0
    xs, ys, d = xx[ok].ravel(), yy[ok].ravel(), d[ok].ravel()
    if len(d) > max_pts:
        sel = np.random.default_rng(0).choice(len(d), max_pts, replace=False)
        xs, ys, d = xs[sel], ys[sel], d[sel]
    Km, Kr = mov['K'], ref['K']
    rays = np.stack([(xs + 0.5 - Km[0, 2]) / Km[0, 0],
                     (ys + 0.5 - Km[1, 2]) / Km[1, 1],
                     np.ones_like(d)], axis=1)
    pts_w = (rays * d[:, None]) @ mov['c2w'][:3, :3].T + mov['c2w'][:3, 3]
    Rr, tr = ref['c2w'][:3, :3], ref['c2w'][:3, 3]
    p = (pts_w - tr) @ Rr
    z = p[:, 2]
    front = z > 1e-9
    zs = np.where(front, z, 1.0)
    u = np.floor(p[:, 0] / zs * Kr[0, 0] + Kr[0, 2]).astype(np.int64)
    v = np.floor(p[:, 1] / zs * Kr[1, 1] + Kr[1, 2]).astype(np.int64)
    Hr, Wr = ref['depth'].shape
    inb = front & (u >= 0) & (u < Wr) & (v >= 0) & (v < Hr)
    if inb.sum() < 50:
        return None, 0
    d_ref = ref['depth'][v[inb], u[inb]]
    good = d_ref > 0
    if good.sum() < 50:
        return None, 0
    return float(np.median(z[inb][good] / d_ref[good])), int(good.sum())


def solve_per_view_depth_scales(views, max_link=1.25, max_total=1.5,
                                min_spread=0.02, iters=1):
    """Chained PER-VIEW depth gauges for one window: the generated views'
    depths carry small independent scale errors (3-10%/pair on re10k) that
    ONE global depth-to-pose scale cannot remove — at wide baselines each
    view then unprojects a shifted copy of the surface (multi-shell). For
    every consecutive pair the relative gauge is the geometric mean of the
    forward median reprojected-depth ratio and the reciprocal backward one
    (_pair_gauge_ratio, scale-invariant), links are chained from view 0 and
    normalized by their median so the window's global depth-to-pose gauge
    is untouched. Returns np.ones(n) when the window is already consistent
    (max |log gauge| < min_spread) or unmeasurable; per-link correction is
    clamped to max_link."""
    n = len(views)
    m = np.ones(n, dtype=np.float64)
    if n < 2:
        return m
    # The link estimator is only meaningful when depths and poses share a
    # gauge (raw re10k depths are ~3x off their pose units, and the pose
    # translation term then contaminates every reprojected-depth comparison
    # and compounds along the chain). Pre-scale with the GLOBAL solve for
    # MEASUREMENT only — the returned multipliers stay unit-median so the
    # caller's own global solve keeps full authority over the final gauge.
    s0 = solve_depth_pose_scale(views)
    meas = [None if v is None else dict(v, depth=v['depth'] * s0)
            for v in views]
    for _ in range(iters):
        step = np.ones(n, dtype=np.float64)
        for i in range(n - 1):
            a, b = meas[i], meas[i + 1]
            if a is None or b is None:
                step[i + 1] = step[i]
                continue
            am = dict(a, depth=a['depth'] * m[i])
            bm = dict(b, depth=b['depth'] * m[i + 1])
            # per-link scale on the LATER view minimizing the symmetric
            # reprojection error. min_gain=0.95: accept only clear (>=5%)
            # error reductions — weak-overlap links can otherwise "improve"
            # spuriously as correspondences slide out of frame, and a single
            # runaway link poisons the whole chain downstream of it.
            r = solve_chunk_boundary_scale([am], [bm], s_init=1.0,
                                           span=0.2, steps=11, min_gain=0.95)
            r = float(np.clip(r, 1.0 / max_link, max_link))
            step[i + 1] = step[i] * r
        m = m * step
        if np.abs(np.log(step)).max() < 1e-3:
            break
    m = np.clip(m, 1.0 / max_total, max_total)   # bound cumulative drift
    if np.abs(np.log(m)).max() < min_spread:
        return np.ones(n, dtype=np.float64)
    m = m / np.median(m)

    # Window-level acceptance: the correction must clearly reduce the mean
    # cross-view reprojection error of the whole window, else return
    # identity (protects already-consistent windows from spurious per-link
    # "improvements" that survive the local gates).
    def _window_err(mults):
        errs = []
        pairs = [(i, i + 1) for i in range(n - 1)]
        pairs += [(i, i + 2) for i in range(n - 2)]
        for i, j in pairs:
            if meas[i] is None or meas[j] is None:
                continue
            a = dict(meas[i], depth=meas[i]['depth'] * mults[i])
            b = dict(meas[j], depth=meas[j]['depth'] * mults[j])
            for e, _ in (_pair_gauge_error(a, b, 1.0),
                         _pair_gauge_error(b, a, 1.0)):
                if e is not None:
                    errs.append(e)
        return float(np.mean(errs)) if errs else None

    e_id = _window_err(np.ones(n))
    e_m = _window_err(m)
    if e_id is None or e_m is None or e_m > 0.98 * e_id:
        return np.ones(n, dtype=np.float64)
    return m


def solve_depth_pose_scale(views, s_lo=0.2, s_hi=10.0, coarse=33,
                           min_gain=0.98, min_baseline_frac=0.02):
    """Solve ONE global depth-to-pose scale s for a window of views: the
    factor on all depths that minimizes cross-view reprojected-depth
    disagreement (median |log z_pred / (s*d_ref)| averaged over view pairs).

    Fixes the unanchored depth gauge of scenes whose depth was never
    supervised against their SfM pose units (e.g. re10k): fusing raw depths
    there backprojects each view to X/s + (1-1/s)*c_v -- one shifted copy of
    the surface per view, i.e. the multi-shell artifact. Returns 1.0 when
    the problem is unidentifiable (near-zero baseline / too few
    correspondences) or when the best s brings no real improvement
    (err(s_opt) > min_gain * err(1)), so supervised-domain windows
    (dl3dv-style, already consistent) pass through untouched.
    """
    vs = [v for v in views if v is not None]
    if len(vs) < 2:
        return 1.0
    # Typical scene depth = median of per-view medians. A raster-prefix
    # subsample here once inflated med_d ~200x on an outdoor scene (sky at
    # the TOP of the frame saturates visionbanana to huge values, and a
    # boolean-mask prefix IS the top rows), misfiring the pure-rotation
    # guard below and skipping the solve entirely.
    med_list = [float(np.median(v['depth'][v['depth'] > 0]))
                for v in vs if (v['depth'] > 0).any()]
    if not med_list:
        return 1.0
    med_d = float(np.median(med_list))
    centers = np.stack([v['c2w'][:3, 3] for v in vs])
    baseline = np.linalg.norm(centers - centers[0], axis=1).max()
    if not np.isfinite(med_d) or med_d <= 0 or \
            baseline < min_baseline_frac * med_d:
        return 1.0                    # pure-rotation window: s unidentifiable

    n = len(vs)
    pairs = [(i, i + 1) for i in range(n - 1)]
    pairs += [(i, i + 2) for i in range(n - 2)]
    if n > 2:
        pairs.append((0, n - 1))

    def err_at(s):
        errs = []
        for i, j in pairs:
            e1, _ = _pair_gauge_error(vs[i], vs[j], s)
            e2, _ = _pair_gauge_error(vs[j], vs[i], s)
            for e in (e1, e2):
                if e is not None:
                    errs.append(e)
        return float(np.mean(errs)) if errs else None

    grid = np.exp(np.linspace(np.log(s_lo), np.log(s_hi), coarse))
    errs = [err_at(s) for s in grid]
    if any(e is None for e in errs):
        return 1.0
    k = int(np.argmin(errs))
    lo = grid[max(0, k - 1)]
    hi = grid[min(coarse - 1, k + 1)]
    for _ in range(2):                # two refinement sweeps (~1% precision)
        grid_r = np.exp(np.linspace(np.log(lo), np.log(hi), 9))
        errs_r = [err_at(s) for s in grid_r]
        if any(e is None for e in errs_r):
            return 1.0
        k = int(np.argmin(errs_r))
        lo = grid_r[max(0, k - 1)]
        hi = grid_r[min(len(grid_r) - 1, k + 1)]
        s_best, e_best = float(grid_r[k]), errs_r[k]
    e_unit = err_at(1.0)
    if e_unit is None or e_best > min_gain * e_unit:
        return 1.0                    # no real improvement: keep raw gauge
    return s_best


def solve_chunk_boundary_scale(ref_views, new_views, s_init=1.0,
                               span=0.35, steps=15, min_gain=0.98):
    """Refine the depth scale of a NEW chunk against the already-kept views
    of the previous chunks: search s around s_init (the single-overlap-frame
    median ratio) minimizing cross-pair reprojection error between up to 3
    ref views and 3 new views. Returns s_init when unidentifiable or when
    the refinement brings no improvement over s_init."""
    refs = [v for v in ref_views if v is not None][-3:]
    news = [v for v in new_views if v is not None][:3]
    if not refs or not news:
        return s_init

    def err_at(s):
        errs = []
        for r in refs:
            for m in news:
                sm = dict(m, depth=m['depth'] * s)
                e1, _ = _pair_gauge_error(r, sm, 1.0)
                e2, _ = _pair_gauge_error(sm, r, 1.0)
                for e in (e1, e2):
                    if e is not None:
                        errs.append(e)
        return float(np.mean(errs)) if errs else None

    grid = s_init * np.exp(np.linspace(-span, span, steps))
    errs = [err_at(s) for s in grid]
    if any(e is None for e in errs):
        return s_init
    k = int(np.argmin(errs))
    # Walk the bracket outward while the minimum sits on its edge (a badly
    # off s_init would otherwise clamp the refinement to the window rim).
    for _ in range(4):
        if 0 < k < len(grid) - 1:
            break
        center = grid[k] * (np.exp(span) if k == len(grid) - 1
                            else np.exp(-span))
        grid = center * np.exp(np.linspace(-span, span, steps))
        errs = [err_at(s) for s in grid]
        if any(e is None for e in errs):
            return s_init
        k = int(np.argmin(errs))
    lo, hi = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    grid_r = np.exp(np.linspace(np.log(lo), np.log(hi), 9))
    errs_r = [err_at(s) for s in grid_r]
    if any(e is None for e in errs_r):
        return s_init
    k = int(np.argmin(errs_r))
    e_init = err_at(s_init)
    if e_init is not None and errs_r[k] > min_gain * e_init:
        return s_init
    return float(grid_r[k])


def _consistency_filter(views, rel_tol, window=2, min_support=1):
    """Cross-view depth-consistency vote (in place).

    A point from view v survives only if, reprojected into at least
    `min_support` neighboring views (v +- window), the neighbor's depth map
    agrees within `rel_tol` (relative). Kills ghost double-surfaces from
    per-view depth scale disagreement; points seen by no other view (out of
    frustum / occluded everywhere) are dropped too.
    """
    T = len(views)
    for v, view in enumerate(views):
        if view is None or len(view['pts']) == 0:
            continue
        pts_v = view['pts']
        support = np.zeros(len(pts_v), dtype=np.int32)
        for u in range(max(0, v - window), min(T, v + window + 1)):
            if u == v or views[u] is None:
                continue
            ref = views[u]
            R, t = ref['c2w'][:3, :3], ref['c2w'][:3, 3]
            p_cam = (pts_v - t) @ R          # w2c rotation = R.T applied row-wise
            z = p_cam[:, 2]
            front = z > 1e-6
            K = ref['K']
            zs = np.where(front, z, 1.0)
            x = p_cam[:, 0] / zs * K[0, 0] + K[0, 2]
            y = p_cam[:, 1] / zs * K[1, 1] + K[1, 2]
            Hr, Wr = ref['depth'].shape
            xi = np.floor(x).astype(np.int64)
            yi = np.floor(y).astype(np.int64)
            inb = front & (xi >= 0) & (xi < Wr) & (yi >= 0) & (yi < Hr)
            d_ref = np.zeros_like(z)
            d_ref[inb] = ref['depth'][yi[inb], xi[inb]]
            ok = inb & (d_ref > 0) & (np.abs(z - d_ref) <= rel_tol * d_ref)
            support += ok.astype(np.int32)
        keep = support >= min_support
        view['pts'] = pts_v[keep]
        view['rgb'] = view['rgb'][keep]


def export_reconstruction_glb(frames, depths, cam_pose_strs, cam_intr_strs,
                              out_path, max_points=1_000_000,
                              depth_percentile=98.0, frustum_frac=0.05,
                              grad_thresh=0.05, voxel_res=512,
                              voxel_min_pts=2, consistency_tol=0.0,
                              align_scale=True, per_view_align=True,
                              camera_frusta=False,
                              adaptive_stride=True, max_stride=4):
    """3D reconstruction export: colored point cloud + camera frusta.

    Unprojects every frame's depth map through its (post-crop) intrinsics and
    c2w pose into a single world-frame point cloud colored by the RGB frames,
    adds one pyramid frustum mesh per camera (colormap along the trajectory,
    view 0 in red), and writes a .glb viewable in any glTF viewer.

    Depth and cam_pose only share the dataloader's scene normalization on
    domains whose training data carries real depth (the sequence-mean
    normalizer divides both); elsewhere (re10k / puffin_omni) generated depth
    lives in an unanchored gauge relative to the SfM pose units and raw
    fusion shells the surface. align_scale=True (default) therefore solves
    one global depth-to-pose scale per window (solve_depth_pose_scale) and
    multiplies all depths by it before unprojection; the solve is a no-op
    (returns 1.0) on already-consistent windows and pure-rotation windows.

    Denoising stages (each individually disableable):
      1. flying-pixel filter (grad_thresh > 0): drop pixels near relative
         depth discontinuities before unprojection.
      2. cross-view consistency vote (consistency_tol > 0): keep a point only
         if a neighboring view's depth agrees at its reprojection.
      3. voxel denoise + downsample (voxel_res > 0): remove isolated-speck
         voxels and average the rest (replaces raw points with voxel means).

    Args:
        frames:        list[T] of [3, H, W] tensors in [-1, 1]
                       (view 0 = GT input, the rest = generated views).
        depths:        list[T] of [H, W] depth tensors/arrays (model output).
        cam_pose_strs: list[T] of stringified 4x4 c2w matrices (OpenCV, z fwd).
        cam_intr_strs: list[T] of stringified 3x3 intrinsics.
        out_path:      output .glb path.
        max_points:    total point budget (uniform random subsample above it).
        depth_percentile: per-view depth cutoff to drop sky / far outliers.
        frustum_frac:  frustum size as a fraction of the cloud's diagonal.
        grad_thresh:   relative depth-gradient threshold of stage 1 (0 = off).
        voxel_res:     voxel grid resolution of stage 3: voxel size = robust
                       cloud diagonal / voxel_res (0 = off).
        voxel_min_pts: minimum points per voxel to keep it in stage 3.
        consistency_tol: relative depth tolerance of stage 2 (0 = off).
    """

    T = min(len(frames), len(depths), len(cam_pose_strs), len(cam_intr_strs))

    # Materialize depths once (numpy float64) so the gauge solve and the
    # unprojection below see the same arrays.
    depths_np = []
    for v in range(T):
        depth = depths[v]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().float().numpy()
        depths_np.append(np.asarray(depth, dtype=np.float64))

    # Gauge solves (see docstring). Per-view FIRST (scale-invariant chained
    # relative gauges — removes the per-view scale scatter that a single
    # global factor cannot), then the global depth-to-pose scale (better
    # posed once the views agree with each other).
    if (align_scale or per_view_align) and T >= 2:
        solver_views = []
        for v in range(T):
            d = depths_np[v]
            H, W = d.shape
            solver_views.append(dict(
                depth=np.where(np.isfinite(d) & (d > 0), d, 0.0),
                K=_rescale_K(_parse_3x3(cam_intr_strs[v]), W, H),
                c2w=_parse_4x4(cam_pose_strs[v]),
            ))
        if per_view_align:
            m_pv = solve_per_view_depth_scales(solver_views)
            if np.abs(np.log(m_pv)).max() > 1e-3:
                print("[glb-align] per-view depth gauges m="
                      + "/".join(f"{x:.3f}" for x in m_pv)
                      + f" -> {out_path}")
                depths_np = [d * m for d, m in zip(depths_np, m_pv)]
                for v in range(T):
                    solver_views[v]['depth'] = solver_views[v]['depth'] * m_pv[v]
        if align_scale:
            s_gauge = solve_depth_pose_scale(solver_views)
            if abs(s_gauge - 1.0) > 1e-3:
                print(f"[glb-align] global depth-to-pose scale s={s_gauge:.3f} "
                      f"-> {out_path}")
                depths_np = [d * s_gauge for d in depths_np]

    # ---- adaptive frame subsampling for the fused cloud ----
    # When inter-frame motion is small, consecutive views contribute nearly
    # identical geometry: fusing all of them only thickens shells and
    # inflates the file. Pick a UNIFORM stride from the mean per-frame
    # motion — rotation angle plus baseline as a fraction of the scene's
    # median (pose-gauge) depth — capped at max_stride so coverage never
    # gets sparse. First and last frames are always kept. Runs after the
    # gauge solve so the baseline normalization sees pose-unit depths.
    if adaptive_stride and max_stride > 1 and T >= 3:
        c2w_all = [_parse_4x4(cam_pose_strs[v]) for v in range(T)]
        per_view_med = []
        for d in depths_np:
            vmask = np.isfinite(d) & (d > 0)
            if vmask.any():
                dv = d[vmask]
                dv = dv[dv <= 20.0 * np.median(dv)]   # drop saturated sky
                if dv.size:
                    per_view_med.append(float(np.median(dv)))
        med_scene = float(np.median(per_view_med)) if per_view_med else 1.0
        if not (med_scene > 0):
            med_scene = 1.0
        rot_deg, base = [], []
        for v in range(T - 1):
            R0, R1 = c2w_all[v][:3, :3], c2w_all[v + 1][:3, :3]
            cosang = (np.trace(R0.T @ R1) - 1.0) / 2.0
            rot_deg.append(float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))))
            base.append(float(np.linalg.norm(
                c2w_all[v + 1][:3, 3] - c2w_all[v][:3, 3])))
        # Motion quantum per KEPT frame: ~6 deg rotation or ~5%-of-scene
        # baseline counts as "one frame's worth" of new content.
        m = float(np.mean([r / 6.0 + (b / med_scene) / 0.05
                           for r, b in zip(rot_deg, base)]))
        stride = int(np.clip(round(1.0 / max(m, 1e-6)), 1, int(max_stride)))
        if stride > 1:
            keep = list(range(0, T, stride))
            if keep[-1] != T - 1:
                keep.append(T - 1)
            print(f"[glb-subsample] mean motion/frame={m:.3f} "
                  f"(rot {float(np.mean(rot_deg)):.2f} deg, baseline "
                  f"{float(np.mean(base)) / med_scene:.1%} of scene depth) "
                  f"-> stride {stride}: {len(keep)}/{T} views -> {out_path}")
            frames = [frames[v] for v in keep]
            depths_np = [depths_np[v] for v in keep]
            cam_pose_strs = [cam_pose_strs[v] for v in keep]
            cam_intr_strs = [cam_intr_strs[v] for v in keep]
            T = len(keep)

    views = []   # per-view records (None when a view has no valid depth)
    c2w_mats = []

    for v in range(T):
        depth = depths_np[v]
        H, W = depth.shape

        img = _tensor_frame_to_uint8_hwc(frames[v])  # [H, W, 3] uint8
        if img.shape[:2] != (H, W):
            img = np.array(Image.fromarray(img).resize((W, H)))

        K = _rescale_K(_parse_3x3(cam_intr_strs[v]), W, H)
        c2w = _parse_4x4(cam_pose_strs[v])
        c2w_mats.append(c2w)

        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            # Two-stage far cutoff: the percentile alone fails when the sky
            # fraction exceeds (100 - percentile)%, so also cap at a multiple
            # of the median (sky decodes to ~1/disparity with disparity -> 0,
            # i.e. orders of magnitude beyond the scene median).
            d_valid = depth[valid]
            cutoff = 20.0 * float(np.median(d_valid))
            if 0 < depth_percentile < 100:
                cutoff = min(cutoff, float(np.percentile(d_valid, depth_percentile)))
            valid &= depth <= cutoff
        # Stage 1: flying-pixel filter at depth discontinuities.
        if grad_thresh and grad_thresh > 0:
            valid &= ~_depth_edge_mask(depth, grad_thresh)
        if not valid.any():
            views.append(None)
            continue

        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        pix = np.stack([i + 0.5, j + 0.5, np.ones_like(i)], axis=-1)  # [H,W,3]
        rays = pix.reshape(-1, 3) @ np.linalg.inv(K).T                # [HW,3]
        pts_cam = rays[valid.reshape(-1)] * depth.reshape(-1, 1)[valid.reshape(-1)]
        pts_world = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]

        views.append(dict(
            pts=pts_world,
            rgb=img.reshape(-1, 3)[valid.reshape(-1)],
            depth=np.where(valid, depth, 0.0),  # reference map for stage 2
            K=K,
            c2w=c2w,
        ))

    # Stage 2: cross-view consistency vote (ghost / double-surface removal).
    if consistency_tol and consistency_tol > 0:
        _consistency_filter(views, consistency_tol)

    kept = [w for w in views if w is not None and len(w['pts']) > 0]
    if not kept:
        print(f"[glb] no valid depth points; skipped {out_path}")
        return

    pts = np.concatenate([w['pts'] for w in kept], axis=0)
    rgb = np.concatenate([w['rgb'] for w in kept], axis=0)

    # Stage 3: voxel denoise + mean-downsample. Voxel size derives from a
    # ROBUST diagonal (1-99 percentile bounds) so residual far outliers can't
    # inflate the grid.
    if voxel_res and voxel_res > 0 and len(pts) > 1:
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
        diag_r = float(np.linalg.norm(hi - lo))
        if diag_r > 0:
            pts, rgb = _voxel_denoise(pts, rgb, diag_r / voxel_res,
                                      max(1, voxel_min_pts))
    if len(pts) == 0:
        print(f"[glb] all points filtered out; skipped {out_path}")
        return

    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, rgb = pts[sel], rgb[sel]

    # OpenCV +y-down -> glTF +y-up: rotate the whole scene 180 deg about X
    # (same trick as viser/glb export so the scene isn't upside down).
    flip = np.diag([1.0, -1.0, -1.0])
    pts = pts @ flip.T

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(pts, colors=rgb), node_name="points")

    # Camera frusta (off by default: the exported cloud stays clean; the
    # trajectory is visualized separately in camera_trajectory.png).
    if camera_frusta:
      diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
      size = max(diag * frustum_frac, 1e-3)
      cmap = matplotlib.colormaps["viridis"]
      for v, c2w in enumerate(c2w_mats):
          K = _parse_3x3(cam_intr_strs[v])
          # Frustum base half-extents from the FoV (z = size at the base plane).
          hx = size * K[0, 2] / K[0, 0]
          hy = size * K[1, 2] / K[1, 1]
          corners_cam = np.array([
              [0.0, 0.0, 0.0],          # apex (camera center)
              [-hx, -hy, size],
              [+hx, -hy, size],
              [+hx, +hy, size],
              [-hx, +hy, size],
          ])
          verts = (corners_cam @ c2w[:3, :3].T + c2w[:3, 3]) @ flip.T
          faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                            [1, 2, 3], [1, 3, 4]])
          color = (255, 32, 32, 255) if v == 0 else tuple(
              int(255 * c) for c in cmap(v / max(T - 1, 1))[:3]) + (255,)
          mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
          mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8),
                                            (len(faces), 1))
          scene.add_geometry(mesh, node_name=f"cam_{v:02d}")

    scene.export(out_path)
    print(f"[glb] saved {len(pts)} pts + {len(c2w_mats)} cams -> {out_path}")
