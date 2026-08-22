"""Shared helpers for the multi-view evaluation scripts.

Split out of generation_multi_view.py by category:
    datasets       -- eval dataset registry + builder
    media          -- image / GIF / heatmap / trajectory-plot saving
    parsing        -- stringified 4x4 / 3x3 matrix parsing
    reconstruction -- .glb point-cloud export + denoising stages
    trajectories   -- synthetic / combo camera trajectories + ray-map rebuild
"""
from .datasets import DATASET_REGISTRY, build_eval_dataset
from .media import (
    compose_row_gif,
    depth_to_visionbanana_frame,
    normalize_gen_outputs_multi,
    save_depth_heatmap,
    save_frames_as_gif,
    save_frames_as_mp4,
    save_image_tensor,
    save_uint8_image,
)
from .camera_vis import visualize_camera_poses, visualize_camera_trajectory
from .motion_overlay import annotate_motion_keys
from .pf_vis import build_offline_pf, save_pf_visualizations
from .reconstruction import (export_reconstruction_glb,
                             solve_chunk_boundary_scale,
                             solve_depth_pose_scale)
from .trajectories import (
    CUSTOM_COMBOS_POOL,
    build_combo_chunk_pose,
    build_custom_inputs,
    build_custom_trajectory,
    rebuild_chunk_ray_maps,
)

__all__ = [
    'DATASET_REGISTRY', 'build_eval_dataset',
    'compose_row_gif', 'depth_to_visionbanana_frame',
    'normalize_gen_outputs_multi', 'save_depth_heatmap', 'save_frames_as_gif',
    'save_image_tensor', 'save_uint8_image', 'save_frames_as_mp4',
    'visualize_camera_poses', 'visualize_camera_trajectory',
    'annotate_motion_keys',
    'build_offline_pf', 'save_pf_visualizations',
    'export_reconstruction_glb',
    'solve_chunk_boundary_scale',
    'solve_depth_pose_scale',
    'CUSTOM_COMBOS_POOL', 'build_combo_chunk_pose',
    'build_custom_inputs', 'build_custom_trajectory',
    'rebuild_chunk_ray_maps',
]
