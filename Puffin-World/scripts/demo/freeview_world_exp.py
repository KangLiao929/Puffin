"""Free-view world exploration demo: image/text -> caption + camera (I2T via
the merged understanding weights) -> single-view generation (T2I, 640x640)
-> multi-view 3D generation along a given camera trajectory (I2I, chunked AR).

Input modes (--input):
  * an image file        -> I2T (VLM caption + camera params) -> T2I -> I2I
  * a .txt file / text   -> (parse trailing camera sentence if present) -> T2I -> I2I
  * a directory          -> each image/.txt inside is one sample; trajectories
                            come from --scene_dir (single scene) or --scene_root
                            (scene dirs matched to sorted inputs by order).

The trajectory is a re10k-style scene dir (rgb/*.png + cam/*.npz with
'intrinsics'/'pose' c2w): T_all = num_views + (chunk-1)*(num_views-1) frames
are sampled at a fixed stride (default: the scene's maximum feasible), the GT
frames are center-crop-rescaled to 640x640 with crop-consistent intrinsics and
saved as the target-GT reference. physical_propagation='offline_prop' anchors
the perspective field at the VLM-estimated (or text-parsed) roll/pitch/vfov/k1
of the START frame; all later views derive theirs through the GT relative
poses. Depth is disabled (geometry off) -- RGB + trajectory outputs only.

Checkpoint: a MERGED und+gen weight (scripts/merge_und_gen_ckpt.py of the
stage-2 full model and the stage-4 model_itr*.pth) -- stage-3/4 checkpoints
alone have no llm/projector/visual_encoder weights and cannot run the I2T step.

Example (one re10k scene, first frame as input):
  python scripts/demo/freeview_world_exp.py \
      --checkpoint /root/weights/demo_freeview_und_gen_1_5b.pth \
      --input  /root/dataset/World-Test/re10k/<scene>/rgb/<first>.png \
      --scene_dir /root/dataset/World-Test/re10k/<scene> \
      --chunk 3 --output output/freeview_demo/<scene>
"""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from scripts.camera.cam_dataset import Cam_Generator
from scripts.camera.utils.text import parse_camera_params
from scripts.evaluation.generation_multi_view import (
    collect_batch_inputs, run_chunked_generation,
)
from scripts.evaluation.modules import (
    annotate_motion_keys, compose_row_gif, save_frames_as_gif,
    save_frames_as_mp4, save_image_tensor, save_pf_visualizations,
    visualize_camera_trajectory,
)
from src.dust3r.datasets.base.base_multiview_dataset import get_ray_map
import src.dust3r.datasets.utils.cropping as cropping

UND_PROMPT = ("Describe the image in detail. Then reason its spatial "
              "distribution and estimate its camera parameters (roll, pitch, "
              "field-of-view, and radial distortion).")
CAM_SENTENCE = ("The camera parameters (roll, pitch, field-of-view, and "
                "radial distortion) are: {roll}, {pitch}, {vfov}, {k1}.")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(config_path, checkpoint, device, geometry='off'):
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained_pth = None
    cfg.model.use_activation_checkpointing = False
    model = BUILDER.build(cfg.model)
    state = guess_load_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[model] loaded {checkpoint}: {len(unexpected)} unexpected keys")
    model = model.to(device).to(model.dtype).eval()
    # RGB-only inference (valid: rgb_blind_to_depth checkpoint) + VLM-anchored
    # perspective-field propagation from the start frame.
    model.geometry_state = (geometry == 'on')
    model.physical_propagation = 'offline_prop'
    model.cam_cfg_mode = 'fix'
    return model


# ---------------------------------------------------------------------------
# I2T: VLM caption + camera params
# ---------------------------------------------------------------------------
def preprocess_for_vlm(pil_img, res=640):
    """Longest edge -> res, [-1, 1], CHW (model.understand pads to square)."""
    w, h = pil_img.size
    s = res / max(w, h)
    img = pil_img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return rearrange(t * 2.0 - 1.0, 'h w c -> c h w')


def split_caption(text):
    """Split a VLM output into (description, (roll, pitch, vfov, k1))."""
    roll, pitch, vfov, k1 = parse_camera_params(text, mode='radial')
    # cut the description at the sentence carrying the trailing number block
    desc = text
    idx = text.rfind('The camera parameters')
    if idx > 0:
        desc = text[:idx].strip()
    return desc, (float(roll), float(pitch), float(vfov), float(k1))


def image_to_text(model, pil_img):
    pixel_values = preprocess_for_vlm(pil_img).to(model.device, model.dtype)
    with torch.no_grad():
        text = model.understand(prompt=[UND_PROMPT],
                                pixel_values=[pixel_values],
                                max_new_tokens=512, progress_bar=False)[0]
    return text


# ---------------------------------------------------------------------------
# T2I: caption + camera text -> one 640x640 image
# ---------------------------------------------------------------------------
def text_to_image(model, caption, cam_params, res=640, cfg_scale=4.5,
                  num_steps=50, seed=42):
    roll, pitch, vfov, k1 = cam_params
    full = f"{caption} " + CAM_SENTENCE.format(
        roll=f"{roll:.4f}", pitch=f"{pitch:.4f}",
        vfov=f"{vfov:.4f}", k1=f"{k1:.4f}")
    cam = Cam_Generator(mode='radial').get_cam(full, res, res)  # [9,H,W] rad
    cam = cam.to(model.device, model.dtype)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    with torch.no_grad():
        imgs = model.generate(prompt=[full], cfg_prompt=[''],
                              cfg_scale=cfg_scale, num_steps=num_steps,
                              cam_values=[[cam / (math.pi / 2.0)]],
                              generator=generator, height=res, width=res,
                              progress_bar=False)
    return imgs[0].float().cpu(), full   # [3,H,W] in [-1,1]


# ---------------------------------------------------------------------------
# trajectory: re10k-style scene dir -> GT frames + poses + intrinsics
# ---------------------------------------------------------------------------
def crop_resize_frame_K(pil_img, K0, res=640):
    """Dataset-parity port of _crop_resize_if_necessary (no rng, no depth):
    principal-point-centered crop -> rescale (scale = max ratio) -> centered
    res x res crop, intrinsics tracked through every step."""
    W, H = pil_img.size
    K = K0.copy()
    cx, cy = int(round(K[0, 2])), int(round(K[1, 2]))
    mx, my = min(cx, W - cx), min(cy, H - cy)
    l, t = cx - mx, cy - my
    img = pil_img.crop((l, t, cx + mx, cy + my))
    K[0, 2] -= l
    K[1, 2] -= t
    img, _, K = cropping.rescale_image_depthmap(
        img, None, K, (res, res), force=True)
    K_target = cropping.camera_matrix_of_crop(
        K, img.size, (res, res), offset_factor=0.5)
    bbox = cropping.bbox_from_intrinsics_in_out(K, K_target, (res, res))
    img = img.crop(bbox)          # crop_image_depthmap needs a real depthmap;
    K = K_target                  # the cropped intrinsics ARE K_target
    arr = np.asarray(img, dtype=np.float32) / 255.0
    frame = rearrange(torch.from_numpy(arr * 2.0 - 1.0), 'h w c -> c h w')
    return frame, K


def intrinsics_from_vfov(vfov, res=640):
    """K from the VLM-estimated (or text-parsed) vertical FOV: focal from
    vfov, principal point at the image center. This is the DEMO's single
    camera model for every generated view -- the GT scene only supplies
    the pose trajectory."""
    f = res / (2.0 * math.tan(max(vfov, 1e-3) / 2.0))
    return np.array([[f, 0.0, res / 2.0],
                     [0.0, f, res / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def load_scene_window(scene_dir, t_all, stride=None, start=0, res=640):
    """GT poses (trajectory) + display-cropped GT frames. The GT npz
    intrinsics are used ONLY to crop the reference frames for display;
    the generation intrinsics come from the VLM vfov."""
    # re10k names are bare timestamps ('128261000.npz'), dl3dv frames carry a
    # prefix ('frame_00001.npz'): order by the digits in the stem either way.
    cams = sorted(os.listdir(os.path.join(scene_dir, 'cam')),
                  key=lambda f: int(''.join(ch for ch in f.split('.')[0]
                                            if ch.isdigit())))
    n = len(cams)
    max_stride = (n - 1 - start) // (t_all - 1)
    assert max_stride >= 1, f"{scene_dir}: {n} frames < {t_all} views"
    s = min(stride, max_stride) if stride else max_stride
    picks = [cams[start + i * s] for i in range(t_all)]
    gt_frames, pose_strs = [], []
    for f in picks:
        z = np.load(os.path.join(scene_dir, 'cam', f))
        K0 = np.asarray(z['intrinsics'] if 'intrinsics' in z else z['intrinsic'],
                        dtype=np.float64)
        pose = np.asarray(z['pose'], dtype=np.float64)
        pil = Image.open(os.path.join(
            scene_dir, 'rgb', f.replace('.npz', '.png'))).convert('RGB')
        frame, _ = crop_resize_frame_K(pil, K0, res)
        gt_frames.append(frame)
        pose_strs.append(np.array2string(pose))
    return gt_frames, pose_strs, s


# ---------------------------------------------------------------------------
# I2I: hand-built batch -> chunked multi-view generation
# ---------------------------------------------------------------------------
def build_data_sample(ext_img, pose_strs, intr_strs, anchor_str, res=640):
    from scripts.evaluation.modules.parsing import _parse_3x3, _parse_4x4
    t_all = len(pose_strs)
    c2w0 = _parse_4x4(pose_strs[0])
    cam_values = []
    for t in range(t_all):
        rm = get_ray_map(c2w0, _parse_4x4(pose_strs[t]),
                         _parse_3x3(intr_strs[t]), res, res)
        cam_values.append(rearrange(
            torch.from_numpy(rm).float(), 'h w c -> c h w').contiguous())
    return dict(
        pixel_values=[ext_img] * t_all,
        pixel_values_init=[ext_img] * t_all,
        cam_values=cam_values,
        cam_pose=list(pose_strs),
        cam_intrinsics=list(intr_strs),
        gt_cam_params=[anchor_str] + [''] * (t_all - 1),
        type='image2image', text='',
    )


def run_i2i(model, ext_img, pose_strs, intr_strs, anchor_str, args):
    sample = build_data_sample(ext_img, pose_strs, intr_strs, anchor_str,
                               res=args.height)
    ns = SimpleNamespace(
        num_views=args.num_views, chunk=args.chunk, cfg_scale=args.cfg_scale,
        num_steps=args.num_steps, height=args.height, width=args.width,
        cfg_prompt='', no_depth_align_chunks=False, depth_align_mode='scale',
        glb_no_align=False, no_depth_global_align=False,
    )
    batch = collect_batch_inputs([sample], ns)
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    with torch.no_grad():
        results = run_chunked_generation(
            model, ns, batch, [sample], batch['cam_values'],
            [list(pose_strs)], None, generator)
    return results


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def save_outputs(out_dir, results, ext_img, gt_frames, pose_strs, intr_strs,
                 vlm_text, caption, cam_params, stride, args):
    os.makedirs(out_dir, exist_ok=True)
    gen = [ext_img] + [f.float().cpu()
                       for f in results['gen_images_tensor'][0]]
    motion_strs = (results.get('motion_pose_strs') or [None])[0] or pose_strs
    keys_frames = annotate_motion_keys(gen, motion_strs)

    for v, fr in enumerate(keys_frames):
        Image.fromarray(fr).save(f"{out_dir}/{v:02d}_novel_view_gen.png")
    for v, fr in enumerate(gt_frames):
        save_image_tensor(fr, f"{out_dir}/{v:02d}_target_gt.png")
    save_image_tensor(ext_img, f"{out_dir}/t2i_input_view.png")
    depth_vis = (results.get('depth_vis_samples') or [None])[0]
    depth_raw = (results.get('depth_samples') or [None])[0]

    save_frames_as_gif(keys_frames, f"{out_dir}/novel_view_gen.gif",
                       fps=args.gif_fps)
    save_frames_as_gif(gt_frames, f"{out_dir}/target_gt.gif",
                       fps=args.gif_fps)
    compose_row_gif([keys_frames, gt_frames], f"{out_dir}/compare.gif",
                    fps=args.gif_fps, mp4_path=f"{out_dir}/compare.mp4",
                    mp4_fps=args.gif_fps)
    visualize_camera_trajectory(list(pose_strs),
                                f"{out_dir}/camera_trajectory.png")
    # eval-style combined.gif: rgb | pf-up | pf-lat (| depth)
    pp = (results.get('pp_cam_params_full') or [None])[0]
    panels = [keys_frames]
    if pp:
        pf_up, pf_lat, _ = save_pf_visualizations(
            gen, pp, list(intr_strs), f"{out_dir}/pf_vis", fps=args.gif_fps)
        panels += [pf_up, pf_lat]
    if depth_vis:
        panels.append([d.float().cpu() for d in depth_vis])
    compose_row_gif(panels, f"{out_dir}/combined.gif", fps=args.gif_fps,
                    mp4_path=f"{out_dir}/combined.mp4", mp4_fps=args.gif_fps)
    if depth_vis:
        dv = [d.float().cpu() for d in depth_vis]
        for v, d in enumerate(dv):
            save_image_tensor(d, f"{out_dir}/{v:02d}_novel_view_depth.png")
        save_frames_as_gif(dv, f"{out_dir}/novel_view_depth.gif",
                           fps=args.gif_fps)
        from scripts.evaluation.modules import export_reconstruction_glb
        n_glb = min(len(gen), len(depth_raw))
        export_reconstruction_glb(
            gen[:n_glb], depth_raw[:n_glb], list(pose_strs)[:n_glb],
            list(intr_strs)[:n_glb], f"{out_dir}/reconstruction.glb")
    with open(f"{out_dir}/meta.json", 'w') as f:
        json.dump(dict(vlm_text=vlm_text, caption=caption,
                       cam_params=dict(zip(('roll', 'pitch', 'vfov', 'k1'),
                                           cam_params)),
                       stride=stride, chunk=args.chunk, seed=args.seed,
                       cfg_scale=args.cfg_scale, num_steps=args.num_steps),
                  f, indent=1)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def gather_inputs(args):
    """Yield (name, kind, payload, scene_dir): kind in {image, text}."""
    inp = args.input
    if os.path.isdir(inp):
        entries = sorted(os.listdir(inp))
        files = [f for f in entries
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.txt'))]
        if args.scene_root:
            scenes = sorted(
                d for d in os.listdir(args.scene_root)
                if os.path.isdir(os.path.join(args.scene_root, d)))
            assert len(scenes) >= len(files), "not enough scene dirs"
            scene_dirs = [os.path.join(args.scene_root, s) for s in scenes]
        else:
            assert args.scene_dir, "--scene_dir or --scene_root required"
            scene_dirs = [args.scene_dir] * len(files)
        for f, sd in zip(files, scene_dirs):
            p = os.path.join(inp, f)
            name = os.path.splitext(f)[0]
            if f.lower().endswith('.txt'):
                yield name, 'text', open(p).read().strip(), sd
            else:
                yield name, 'image', p, sd
    elif os.path.isfile(inp):
        name = os.path.splitext(os.path.basename(inp))[0]
        assert args.scene_dir, "--scene_dir required"
        if inp.lower().endswith('.txt'):
            yield name, 'text', open(inp).read().strip(), args.scene_dir
        else:
            yield name, 'image', inp, args.scene_dir
    else:  # raw text on the command line
        assert args.scene_dir, "--scene_dir required"
        yield 'text_prompt', 'text', inp, args.scene_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=(
        'configs/pipelines/'
        'final_stage_4_world_all_asym_attn_qwen2_5_1_5b_radiov4H_'
        'sd3p5L.py'))
    p.add_argument('--checkpoint', required=True,
                   help='MERGED und+gen weight (merge_und_gen_ckpt.py).')
    p.add_argument('--input', required=True,
                   help='image / .txt / raw text / directory of samples')
    p.add_argument('--scene_dir', default=None,
                   help='re10k-style scene dir (rgb/ + cam/) = trajectory + GT')
    p.add_argument('--scene_root', default=None,
                   help='directory of scene dirs (paired with a folder input)')
    p.add_argument('--output', default='output/freeview_world_exp')
    p.add_argument('--num_views', type=int, default=8)
    p.add_argument('--chunk', type=int, default=3)
    p.add_argument('--stride', type=int, default=None,
                   help='trajectory frame stride (default: scene maximum)')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--height', type=int, default=640)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--cfg_scale', type=float, default=4.0)
    p.add_argument('--t2i_cfg_scale', type=float, default=4.5)
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gif_fps', type=int, default=6)
    p.add_argument('--geometry', choices=('off', 'on'), default='off',
                   help="'on' adds depth maps + reconstruction.glb "
                        "(rgb_blind_to_depth checkpoint: RGB output "
                        "unchanged either way)")
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.config, args.checkpoint, device,
                        geometry=args.geometry)
    t_all = args.num_views + (args.chunk - 1) * (args.num_views - 1)

    for name, kind, payload, scene_dir in gather_inputs(args):
        out_dir = os.path.join(args.output, name)
        print(f"===== [{name}] kind={kind} scene={scene_dir}")

        # ---- I2T (image inputs) / parse (text inputs) ----
        if kind == 'image':
            pil = Image.open(payload).convert('RGB')
            vlm_text = image_to_text(model, pil)
            caption, cam_params = split_caption(vlm_text)
            os.makedirs(out_dir, exist_ok=True)
            pil.save(f"{out_dir}/input.png")
        else:
            vlm_text = None
            try:
                caption, cam_params = split_caption(payload)
            except Exception:
                caption, cam_params = payload, (0.0, 0.0, 1.0, 0.0)
        print(f"  caption: {caption[:120]}...")
        print(f"  cam(rad): roll={cam_params[0]:.4f} pitch={cam_params[1]:.4f} "
              f"vfov={cam_params[2]:.4f} k1={cam_params[3]:.4f}")

        # ---- T2I ----
        ext_img, full_prompt = text_to_image(
            model, caption, cam_params, res=args.height,
            cfg_scale=args.t2i_cfg_scale, num_steps=args.num_steps,
            seed=args.seed)

        # ---- trajectory + GT reference ----
        gt_frames, pose_strs, stride = load_scene_window(
            scene_dir, t_all, stride=args.stride, start=args.start,
            res=args.height)
        # intrinsics come from the VLM/text vfov (focal from vfov, principal
        # point at the image center), shared by all views
        K_vlm = intrinsics_from_vfov(cam_params[2], res=args.height)
        intr_strs = [np.array2string(K_vlm)] * t_all
        print(f"  trajectory: {t_all} views @ stride {stride} | "
              f"f={K_vlm[0, 0]:.1f}px from vfov={cam_params[2]:.4f}")

        # ---- I2I ----
        anchor = " ".join(f"{v:.8f}" for v in cam_params)
        results = run_i2i(model, ext_img, pose_strs, intr_strs, anchor, args)

        save_outputs(out_dir, results, ext_img, gt_frames, pose_strs,
                     intr_strs, vlm_text, full_prompt, cam_params, stride,
                     args)
        print(f"  -> {out_dir}")


if __name__ == '__main__':
    main()
