import re
from typing import Tuple

def parse_camera_params(
    text: str,
    mode: str = "pinhole"
) -> Tuple[float, float, float, float]:
    """
    Extract camera parameters from text. Always returns 4 parameters.

    Args:
        text: The full text containing the parameters.
        mode: 
            - 'pinhole': Parses 3 params (roll, pitch, fov), sets k1 = 0.0.
            - 'radial':  Parses 4 params (roll, pitch, fov, k1).

    Returns:
        (roll, pitch, fov, k1) as floats.

    Raises:
        ValueError if the pattern is not found or mode is invalid.
    """
    
    # Common number pattern: optional sign, digits, optional decimal part
    num_pat = r"([+-]?\d+(?:\.\d+)?)"
    
    # Regex for Pinhole (3 parameters found in text)
    pat_pinhole = re.compile(
        r"are:\s*" + num_pat + r"\s*,\s*"  # roll
        + num_pat + r"\s*,\s*"             # pitch
        + num_pat +                        # fov
        r"[\.\s]*$"                        # End of sentence
    )

    # Regex for Radial (4 parameters found in text)
    pat_radial = re.compile(
          num_pat + r"\s*,\s*"  # roll #r"are:\s*" + num_pat + r"\s*,\s*"
        + num_pat + r"\s*,\s*"             # pitch
        + num_pat + r"\s*,\s*"             # fov
        + num_pat# +                        # k1
        #r"[\.\s]*$"                        # End of sentence
    )

    m = None
    if mode == "pinhole":
        m = pat_pinhole.search(text)
        if not m:
            raise ValueError(f"No 3-parameter pinhole config found in text using mode '{mode}'.")
        
        roll_s, pitch_s, fov_s = m.group(1), m.group(2), m.group(3)
        # Return 4 values: extracted 3 + hardcoded 0.0 for k1
        return float(roll_s), float(pitch_s), float(fov_s), 0.0

    elif mode == "radial":
        # The camera parameters are appended at the END of the caption
        # ("... are: r, p, f, k1."), but free-form descriptions may also
        # contain comma-separated numbers (years, room counts, prices, ...).
        # `search` would grab the FIRST such group; take the LAST match so the
        # trailing parameter block always wins.
        matches = list(pat_radial.finditer(text))
        if not matches:
            raise ValueError(f"No 4-parameter radial config found in text using mode '{mode}'.")
        m = matches[-1]

        roll_s, pitch_s, fov_s, k1_s = m.group(1), m.group(2), m.group(3), m.group(4)
        return float(roll_s), float(pitch_s), float(fov_s), float(k1_s)

    else:
        raise ValueError(f"Invalid mode: {mode!r}. Choose 'pinhole' or 'radial'.")


if __name__ == "__main__":
    txt_3 = "The view is beautiful. The camera parameters are: 0.1, 0.2, 1.5."
    print(f"Pinhole: {parse_camera_params(txt_3, mode='pinhole')}")

    txt_4 = "The view is beautiful. The camera parameters are: 0.1, 0.2, 1.5, 0.1."
    print(f"Radial:  {parse_camera_params(txt_4, mode='radial')}")