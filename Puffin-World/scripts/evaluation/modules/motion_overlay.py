"""Genie-style keyboard-control overlay for the generated RGB frames.

Each frame shows the motion TO the next frame, derived from the poses the
model was actually conditioned on (the final frame has no next motion and
shows the panel with every key inactive):

  bottom-left  (blue) : rotation cluster -- arrow keys (up/down = pitch,
                        left/right = yaw) plus two curved-arrow keycaps
                        flanking the up key for roll (CCW left, CW right;
                        the classic Q/E lean position). Roll follows the
                        ON-SCREEN content rotation (what the viewer sees).
  bottom-right (red)  : translation cluster -- W/S forward/backward,
                        A/D strafe left/right, plus two triangle keycaps
                        flanking W for vertical flight (left = descend,
                        right = ascend).

Motions are decomposed as GRAVITY-REFERENCED Euler-angle rates (world +y
down, the OpenCV/COLMAP convention shared by re10k / dl3dv / puffin_omni),
not around the raw camera axes: yaw = azimuth rate, pitch = elevation rate,
roll = gravity-roll rate (screen-signed), translation = horizontal forward /
strafe / vertical rise. A camera that pans or tilts while HOLDING a nonzero
roll, or pans while tilted up/down (both common in puffin_omni), therefore
still lights a single key, matching how the trajectory was commanded ("it
just pans left") instead of smearing it across camera axes.

Active dimensions render as solid keycaps, inactive ones stay faint; a
dimension whose per-frame motion is negligible relative to the sequence's
dominant motion (30% gate, plus a 0.3 deg/view absolute floor for rotation)
is never lit. Translation has no absolute floor: its scale is the arbitrary
per-scene SfM unit, so only the relative gate is meaningful.
"""
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .media import _tensor_frame_to_uint8_hwc
from .parsing import _parse_4x4

ROT_COLOR = (66, 133, 244)    # blue (rotation cluster)
TRANS_COLOR = (235, 87, 87)   # red  (translation cluster)


def _font(size):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)
    except Exception:
        return ImageFont.load_default()


WORLD_DOWN = np.array([0.0, 1.0, 0.0])  # OpenCV-style world: gravity = +y


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _gravity_state(M):
    """(azimuth, elevation, gravity-roll, right, up, fwd) of one c2w pose,
    or None at gimbal lock (view within ~0.06 deg of straight up/down)."""
    R = M[:3, :3]
    x, z = R[:, 0], R[:, 2]
    up = -WORLD_DOWN
    zu = float(z @ up)
    u_perp = up - zu * z                     # vertical axis, made ⟂ view
    n = float(np.linalg.norm(u_perp))
    if n < 1e-6:
        return None
    u_perp = u_perp / n
    right = np.cross(z, u_perp)              # horizontal right
    fwd = np.cross(u_perp, right)            # horizontal forward
    az = math.atan2(float(z[0]), float(z[2]))
    el = math.asin(max(-1.0, min(1.0, zu)))
    level_down = np.cross(z, right)          # camera-down of a no-roll view
    rg = math.atan2(float(x @ level_down), float(x @ right))
    return az, el, rg, right, u_perp, fwd


def relative_motions(pose_strs):
    """Per consecutive frame pair: gravity-referenced Euler-angle rates plus
    gravity-decomposed translation (world +y down, the OpenCV/COLMAP
    convention shared by re10k / dl3dv / puffin_omni).

    Each c2w pose is reduced to (azimuth, elevation, gravity-roll); the
    per-step motion is their wrapped difference: yaw = azimuth rate (>0
    turns RIGHT), pitch = elevation rate (>0 looks UP), roll = gravity-roll
    rate signed to the ON-SCREEN content rotation (>0 = the image visibly
    turns CW). Differencing ANGLES -- rather than projecting the rotation
    vector onto the view axis -- keeps a pure pan at a tilted-down view from
    leaking into roll (spinning about the vertical while looking down does
    rotate the screen content, but as a camera COMMAND it is yaw only, and
    the keys visualize commands). Translation is projected onto the
    horizontal-right / vertical / horizontal-forward axes: tx strafe (>0
    RIGHT), ty rise (>0 ascends), tz advance (>0 FORWARD). Near straight
    up/down views the azimuth/roll split is ill-defined: yaw/pitch/tx/tz
    are zeroed and roll falls back to the view-axis rotation, which IS the
    visible spin there.
    """
    mats = [_parse_4x4(str(s)) for s in pose_strs]
    states = [_gravity_state(m) for m in mats]
    out = []
    for a, b, sa, sb in zip(mats[:-1], mats[1:], states[:-1], states[1:]):
        rel = np.linalg.inv(a) @ b
        Ra = a[:3, :3]
        tw = Ra @ rel[:3, 3]                 # translation, world coords
        up = -WORLD_DOWN
        if sa is None or sb is None:         # gimbal lock on either end
            R = rel[:3, :3]
            w_cam = 0.5 * np.array([R[2, 1] - R[1, 2],
                                    R[0, 2] - R[2, 0],
                                    R[1, 0] - R[0, 1]])
            w = Ra @ w_cam
            out.append(dict(pitch=0.0, yaw=0.0,
                            roll=float(-(w @ Ra[:, 2])),
                            tx=0.0, ty=float(tw @ up), tz=0.0))
            continue
        az_a, el_a, rg_a, right, _, fwd = sa
        az_b, el_b, rg_b = sb[:3]
        out.append(dict(
            pitch=el_b - el_a,               # >0 looks UP
            yaw=_wrap(az_b - az_a),          # >0 turns RIGHT
            roll=-_wrap(rg_b - rg_a),        # >0 content turns CW on screen
            tx=float(tw @ right),            # >0 strafes RIGHT (D)
            ty=float(tw @ up),               # >0 rises (ascend key)
            tz=float(tw @ fwd),              # >0 moves FORWARD (W)
        ))
    # Numeric-noise gate: pose strings round-trip through np.array2string
    # (~8 significant digits, format chosen per matrix), so an exactly
    # constant translation can come back with ~1e-8 * |t| per-frame jitter.
    # The activation threshold is relative with no absolute floor (SfM scale
    # is arbitrary), so that jitter would otherwise light translation keys
    # on pure-rotation trajectories. Anything six orders below the pose
    # translation scale cannot be a camera command: zero it so _thresholds
    # sees a truly static translation (thr -> inf, keys stay dark).
    t_scale = max((float(np.max(np.abs(m[:3, 3]))) for m in mats), default=0.0)
    eps = 1e-6 * max(1.0, t_scale)
    for m in out:
        for k in ('tx', 'ty', 'tz'):
            if abs(m[k]) < eps:
                m[k] = 0.0
    return out


def _thresholds(motions, rot_frac=0.3, trans_frac=0.3,
                rot_floor=math.radians(0.3)):
    """Per-group activation thresholds: a fraction of the sequence's dominant
    motion, so sub-dominant jitter never lights its key (a 0.2 deg roll
    riding on a 2 deg pan stays dark).

    Rotation additionally gets an absolute floor (its unit is physical, and
    below ~0.3 deg/view nothing is perceptible), but translation does NOT:
    its unit is the arbitrary per-scene SfM scale (re10k-style c2w units),
    where 0.002/view can be a large visible dolly in a small room, so any
    absolute floor would wrongly kill real motion in close-range scenes."""
    rot_max = max((max(abs(m['pitch']), abs(m['yaw']), abs(m['roll']))
                   for m in motions), default=0.0)
    trans_max = max((max(abs(m['tx']), abs(m['ty']), abs(m['tz']))
                     for m in motions), default=0.0)
    rot_thr = max(rot_floor, rot_frac * rot_max)
    trans_thr = (trans_frac * trans_max if trans_max > 0 else float('inf'))
    return rot_thr, trans_thr


def _keycap(draw, x, y, s, color, active):
    r = max(3, s // 6)
    if active:
        draw.rounded_rectangle([x, y, x + s, y + s], radius=r,
                               fill=color + (235,),
                               outline=(255, 255, 255, 255),
                               width=max(1, s // 18))
    else:
        draw.rounded_rectangle([x, y, x + s, y + s], radius=r,
                               fill=(10, 10, 10, 60),
                               outline=color + (80,),
                               width=max(1, s // 18))


def _glyph_fill(color, active):
    return (255, 255, 255, 255) if active else color + (95,)


def _arrow(draw, x, y, s, direction, fill):
    """Solid arrow (triangle head + shaft) centered in the keycap."""
    c = s / 2.0
    h = s * 0.26      # head half-width
    l = s * 0.30      # head length
    w = s * 0.10      # shaft half-width
    tip, base = -s * 0.30, s * 0.30
    pts = [(0, tip), (-h, tip + l), (-w, tip + l), (-w, base),
           (w, base), (w, tip + l), (h, tip + l)]
    rot = {'up': 0, 'right': 90, 'down': 180, 'left': 270}[direction]
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    draw.polygon([(x + c + px * ca - py * sa, y + c + px * sa + py * ca)
                  for px, py in pts], fill=fill)


def _roll_tile(s, fill, cw):
    """Curved-arrow roll glyph on a transparent tile (CW; mirrored for CCW)."""
    scale = 4  # supersample for a smooth arc
    S = s * scale
    tile = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    r = S * 0.28
    cx = cy = S / 2.0
    width = max(scale, int(S * 0.09))
    # 270-degree arc, open at the top-right; PIL angles run clockwise.
    d.arc([cx - r, cy - r, cx + r, cy + r], start=45, end=315,
          fill=fill, width=width)
    # arrowhead at the arc end (315 deg), tangent = clockwise direction
    ang = math.radians(315)
    ex, ey = cx + r * math.cos(ang), cy + r * math.sin(ang)
    tx, ty = -math.sin(ang), math.cos(ang)      # cw tangent
    nx, ny = math.cos(ang), math.sin(ang)       # outward normal
    h = S * 0.16
    d.polygon([
        (ex + tx * h, ey + ty * h),
        (ex + nx * h * 0.6, ey + ny * h * 0.6),
        (ex - nx * h * 0.6, ey - ny * h * 0.6),
    ], fill=fill)
    tile = tile.resize((s, s), Image.LANCZOS)
    if not cw:
        tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
    return tile


def _triangle(draw, x, y, s, up, fill):
    """Solid triangle glyph for the vertical-flight keys (ascend/descend)."""
    c = s / 2.0
    h = s * 0.24
    wd = s * 0.28
    if up:
        pts = [(c, c - h), (c - wd, c + h), (c + wd, c + h)]
    else:
        pts = [(c, c + h), (c - wd, c - h), (c + wd, c - h)]
    draw.polygon([(x + px, y + py) for px, py in pts], fill=fill)


def _letter(draw, x, y, s, ch, fill, font):
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (s - w) / 2 - bbox[0], y + (s - h) / 2 - bbox[1]),
              ch, font=font, fill=fill)


def draw_motion_keys(frame_uint8, motion, rot_thr, trans_thr):
    """Composite the two keyboard clusters onto ONE uint8 [H, W, 3] frame."""
    H, W = frame_uint8.shape[:2]
    s = max(22, H // 14)                    # keycap size
    g = max(3, s // 8)                      # gap
    m = max(10, H // 40)                    # margin
    inset = max(12, H // 32)                # extra inset (~20px at 640): keep
    mx = m + inset                          # the clusters off the frame edges
    my = m + inset
    font = _font(int(s * 0.62))

    base = Image.fromarray(frame_uint8).convert('RGBA')
    ov = Image.new('RGBA', base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    act = {
        'up':    motion['pitch'] > rot_thr,
        'down':  motion['pitch'] < -rot_thr,
        'right': motion['yaw'] > rot_thr,
        'left':  motion['yaw'] < -rot_thr,
        'cw':    motion['roll'] > rot_thr,
        'ccw':   motion['roll'] < -rot_thr,
        'w':     motion['tz'] > trans_thr,
        's':     motion['tz'] < -trans_thr,
        'd':     motion['tx'] > trans_thr,
        'a':     motion['tx'] < -trans_thr,
        'asc':   motion['ty'] > trans_thr,
        'desc':  motion['ty'] < -trans_thr,
    }

    # ---- bottom-left: rotation cluster (blue) ----
    x0 = mx
    y1 = H - my - 2 * s - g                 # top row
    y2 = H - my - s                         # bottom row
    for i, key in enumerate(('ccw', 'up', 'cw')):
        x = x0 + i * (s + g)
        _keycap(d, x, y1, s, ROT_COLOR, act[key])
        fill = _glyph_fill(ROT_COLOR, act[key])
        if key == 'up':
            _arrow(d, x, y1, s, 'up', fill)
        else:
            ov.alpha_composite(_roll_tile(s, fill, cw=(key == 'cw')), (x, y1))
    for i, key in enumerate(('left', 'down', 'right')):
        x = x0 + i * (s + g)
        _keycap(d, x, y2, s, ROT_COLOR, act[key])
        _arrow(d, x, y2, s, key, _glyph_fill(ROT_COLOR, act[key]))

    # ---- bottom-right: translation cluster (red) ----
    x0 = W - mx - 3 * s - 2 * g
    for i, key in enumerate(('desc', 'w', 'asc')):
        x = x0 + i * (s + g)
        _keycap(d, x, y1, s, TRANS_COLOR, act[key])
        fill = _glyph_fill(TRANS_COLOR, act[key])
        if key == 'w':
            _letter(d, x, y1, s, 'W', fill, font)
        else:
            _triangle(d, x, y1, s, up=(key == 'asc'), fill=fill)
    for i, key in enumerate(('a', 's', 'd')):
        x = x0 + i * (s + g)
        _keycap(d, x, y2, s, TRANS_COLOR, act[key])
        _letter(d, x, y2, s, key.upper(),
                _glyph_fill(TRANS_COLOR, act[key]), font)

    return np.asarray(Image.alpha_composite(base, ov).convert('RGB'))


_NO_MOTION = dict(pitch=0.0, yaw=0.0, roll=0.0, tx=0.0, ty=0.0, tz=0.0)


def annotate_motion_keys(frames, pose_strs, rot_frac=0.3, trans_frac=0.3):
    """Overlay the control clusters on every frame: frame i shows the motion
    i -> i+1, and trailing frames without a next motion (normally just the
    last one) show the panel with every key inactive. Returns uint8
    [H, W, 3] frames.

    Args:
        frames:    list[T] of [3, H, W] tensors in [-1, 1] (or uint8 HWC).
        pose_strs: list[>=T] of 4x4 c2w strings the model conditioned on
                   (consecutive entries must share a reference frame).
    """
    def to_uint8(fr):
        if isinstance(fr, torch.Tensor):
            return _tensor_frame_to_uint8_hwc(fr)
        arr = np.asarray(fr)
        return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr

    out = [to_uint8(fr) for fr in frames]
    n_pairs = max(0, min(len(out) - 1, len(pose_strs) - 1))
    motions = relative_motions(pose_strs[:n_pairs + 1]) if n_pairs else []
    rot_thr, trans_thr = _thresholds(motions, rot_frac, trans_frac)
    for i in range(len(out)):
        motion = motions[i] if i < n_pairs else _NO_MOTION
        out[i] = draw_motion_keys(out[i], motion, rot_thr, trans_thr)
    return out
