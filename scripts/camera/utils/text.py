import re
from typing import Tuple

def parse_camera_params(
    text: str,
    mode: str = "base"
) -> Tuple[float, float, float]:
    """
    Extract roll, pitch, fov from text using one of two patterns:
      - 'base'   mode: ... are: roll, pitch, fov.
      - 'cot'    mode: <answer>roll, pitch, fov</answer>

    Args:
        text: The full text to search.
        mode: One of {"base", "cot"}.

    Returns:
        roll, pitch, fov as floats.

    Raises:
        ValueError if the chosen pattern is not found, or mode is invalid.
    """
    # compile both regexes
    pat_base = re.compile(
        r"are:\s*([+-]?\d+(?:\.\d+)?)\s*,\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
        r"([+-]?\d+(?:\.\d+)?)[\.\s]*$"
    )
    pat_cot = re.compile(
        r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*,\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*</answer>"
    )

    m = None
    if mode == "base":
        m = pat_base.search(text)
    elif mode == "cot":
        m = pat_cot.search(text)
    else:
        raise ValueError(f"Invalid mode: {mode!r}. Choose 'base', 'cot', or 'auto'.")

    if not m:
        raise ValueError(f"No camera parameters found using mode '{mode}'.")

    roll_s, pitch_s, fov_s = m.group(1), m.group(2), m.group(3)
    return float(roll_s), float(pitch_s), float(fov_s)