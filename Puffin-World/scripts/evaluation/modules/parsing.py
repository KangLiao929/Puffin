"""Parsing of stringified camera matrices (the dataloaders emit
`np.array2string` 4x4 c2w poses and 3x3 intrinsics)."""
import re

import numpy as np


def _parse_4x4(s):
    """Parse a numpy 4x4 string back into ndarray (matches model._parse_cam_pose_str)."""
    nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)]
    return np.array(nums, dtype=np.float64).reshape(4, 4)


def _parse_3x3(s):
    """Parse a numpy 3x3 string back into ndarray."""
    nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)]
    return np.array(nums, dtype=np.float64).reshape(3, 3)
