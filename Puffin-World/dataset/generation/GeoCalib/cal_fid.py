import argparse
import torch
from pytorch_fid import fid_score
import os

def calculate_fid_metric(path_a, path_b, batch_size=50, device=None, dims=2048):
    """
    Compute the FID score between the image distributions of two folders.

    Args:
        path_a (str): Folder of generated images.
        path_b (str): Folder of ground-truth / real images.
        batch_size (int): Batch size for Inception inference.
        device (torch.device): Run device (CPU/GPU).
        dims (int): Inception feature dimensionality (default 2048, the final
            average-pooling layer).

    Returns:
        float: FID score (lower is better).
    """

    # 1. Check that the paths exist.
    if not os.path.exists(path_a):
        raise FileNotFoundError(f"Path not found: {path_a}")
    if not os.path.exists(path_b):
        raise FileNotFoundError(f"Path not found: {path_b}")

    # 2. Pick the device.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Computing FID...")
    print(f"   - Generated path: {path_a}")
    print(f"   - Real path:      {path_b}")
    print(f"   - Device:         {device}")

    # 3. Compute FID (num_workers can be tuned to the CPU core count).
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[path_a, path_b],
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=8
        )
        return fid_value
    except Exception as e:
        print(f"Error during computation: {e}")
        return None

# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the FID score between two image folders.")
    parser.add_argument("--path_gen", required=True, type=str,
                        help="Folder of generated images.")
    parser.add_argument("--path_gt", required=True, type=str,
                        help="Folder of ground-truth / real images.")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Inception inference batch size (default 50).")
    parser.add_argument("--dims", type=int, default=2048,
                        help="Inception feature dimensionality (default 2048).")
    args = parser.parse_args()

    # Make sure the folders contain images (png, jpg, ...).
    fid = calculate_fid_metric(args.path_gen, args.path_gt,
                               batch_size=args.batch_size, dims=args.dims)

    if fid is not None:
        print("\n" + "="*40)
        print(f"FID Score: {fid:.4f}")
        print("="*40)
