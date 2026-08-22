"""Merge per-video YouTube frame folders into one flat directory.

Each subfolder of --src (numeric names like "00000057") holds the extracted
frames of one video, plus an optional .txt sidecar containing "y: <int>" -- a
row threshold below which the frames are blacked out (e.g. to mask a channel
watermark / UI bar at the bottom of the panorama). For every subfolder in the
optional --range: images are moved (or blackened + re-saved) into --dst, then
the subfolder is DELETED. Same-name files from different folders overwrite
each other in --dst.
"""
import os
import shutil
import argparse
from PIL import Image

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')


def read_y_threshold(subpath):
    """Parse the optional "y: <int>" sidecar in `subpath`.

    Only the FIRST .txt (listing order) is considered; returns None when there
    is no .txt or it cannot be parsed.
    """
    for fname in sorted(os.listdir(subpath)):
        if not fname.lower().endswith('.txt'):
            continue
        txt_path = os.path.join(subpath, fname)
        y_threshold = None
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if 'y:' in content:
                # tolerate surrounding text: take the number right after "y:"
                parts = content.split('y:')
                if len(parts) > 1:
                    y_threshold = int(parts[1].split()[0].strip())
        except Exception as e:  # noqa: BLE001 -- a bad sidecar must not stop the batch
            print(f"Warning reading sidecar {fname}: {e}")
        return y_threshold  # first .txt only, parsed or not
    return None


def process_and_move_images(src_root, dst_dir, target_range=None):
    """Flatten frame subfolders of `src_root` into `dst_dir` (see module doc).

    target_range: optional (start, end) inclusive filter on NUMERIC subfolder
    names; out-of-range and non-numeric folders are skipped untouched.
    Processed subfolders are DELETED after their images are moved/saved.
    """
    os.makedirs(dst_dir, exist_ok=True)

    # sorted -> deterministic processing order, so on same-name collisions the
    # surviving file is stable across runs
    all_subnames = sorted(os.listdir(src_root))
    print(f"Scanning {src_root}: {len(all_subnames)} entries...")
    if target_range:
        print(f"Only processing folder numbers {target_range[0]} to {target_range[1]}")

    processed_count = 0
    for subname in all_subnames:
        subpath = os.path.join(src_root, subname)
        if not os.path.isdir(subpath):
            continue

        # range filter on numeric folder names (e.g. "00000057" -> 57);
        # out-of-range / non-numeric folders are skipped, NOT deleted
        if target_range:
            try:
                folder_num = int(subname)
            except ValueError:
                print(f"Skipping non-numeric folder: {subname}")
                continue
            start_num, end_num = target_range
            if not (start_num <= folder_num <= end_num):
                continue

        y_threshold = read_y_threshold(subpath)

        for fname in os.listdir(subpath):
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue
            src_img_path = os.path.join(subpath, fname)
            # NOTE: same-name files from different folders overwrite each other
            # here; derive a unique name from `subname` if that ever matters.
            dst_img_path = os.path.join(dst_dir, fname)

            if y_threshold is not None:
                try:
                    img = Image.open(src_img_path).convert('RGB')
                    width, height = img.size
                    # black out everything below the y threshold (paste on a
                    # box region -- much faster than per-pixel loops)
                    if 0 <= y_threshold < height:
                        img.paste((0, 0, 0), (0, y_threshold, width, height))
                    img.save(dst_img_path)
                except Exception as e:  # noqa: BLE001 -- keep the batch going
                    print(f"Failed to process image {src_img_path}: {e}")
            else:
                shutil.move(src_img_path, dst_img_path)

        # delete the whole subfolder (incl. any leftover .txt); only folders
        # that passed the range filter ever reach this point
        try:
            shutil.rmtree(subpath)
            processed_count += 1
            print(f"[{processed_count}] Processed and removed: {subname}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed to remove folder {subname}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten per-video frame subfolders into one directory "
                    "(optionally blacking out rows below a per-folder 'y:' "
                    "threshold), then DELETE the processed subfolders."
    )
    parser.add_argument('--src', '-s', required=True,
                        help='source root containing the frame subfolders (e.g. ./data)')
    parser.add_argument('--dst', '-d', required=True,
                        help='destination directory for all images')
    parser.add_argument('--range', '-r',
                        help='numeric folder range to process, "start-end" '
                             '(e.g. 57-120); omit to process everything')
    args = parser.parse_args()

    range_tuple = None
    if args.range:
        try:
            start_s, end_s = args.range.split('-')
            range_tuple = (int(start_s), int(end_s))
        except ValueError:
            print('Error: bad --range format, use start-end (e.g. 57-120)')
            raise SystemExit(1)

    process_and_move_images(args.src, args.dst, range_tuple)


if __name__ == '__main__':
    main()
