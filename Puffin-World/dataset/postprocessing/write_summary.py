"""Generate summary.json linking images with their annotation JSONs.

For each scene folder under root_path, match images in <folder>/<img_folder>/
with same-stem JSON files in <folder>/<annot_folder>/, optionally filtering by:
  - dimension divisibility (--gen_unit, e.g. 16 for the generation pipeline)
  - aspect ratio whitelist (--aspect_ratios, e.g. "9:16" "1:1", tol --ratio_tol)

The result is written to root_path/<output> as a list of records:

    [{"image": "Scene/train/x.jpg", "annotation": "Scene/train_scene_cam/x.json"}, ...]

ordered deterministically (folders in CLI/sorted order, files sorted by name).

Examples:
    python dataset/postprocessing/write_summary.py --root_path /path/Projection
    python dataset/postprocessing/write_summary.py --root_path /path/Projection \
        --annot_folder train_thinking_cam --gen_unit 16 --aspect_ratios 1:1 4:3 \
        --excluded_folders OmniBlender --output summary_gen.json
"""

import os
import json
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def parse_aspect_ratios(ratio_list):
    """Parse ratio strings (e.g. ['9:16', '4:3'] or ['1.77']) into floats."""
    target_ratios = []
    if not ratio_list:
        return target_ratios

    for r in ratio_list:
        try:
            if ':' in r:
                w, h = map(float, r.split(':'))
                target_ratios.append(w / h)
            else:
                target_ratios.append(float(r))
        except ValueError:
            print(f"Warning: Could not parse aspect ratio '{r}', ignoring.")

    return target_ratios


def image_passes_filters(img_path, gen_unit, target_ratios, ratio_tol):
    """Open the image header and apply the gen_unit / aspect-ratio filters."""
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception as e:
        print(f"Warning: Could not read image properties for {img_path}: {e}")
        return False

    # Dimension divisibility
    if gen_unit > 1 and (w % gen_unit != 0 or h % gen_unit != 0):
        return False

    # Aspect-ratio whitelist
    if target_ratios:
        current_ratio = w / h
        if not any(abs(current_ratio - t) < ratio_tol for t in target_ratios):
            return False

    return True


def process_subfolder(folder_name, root_path, img_folder_name, annot_folder_name,
                      gen_unit, target_ratios, ratio_tol):
    """
    Process a single subfolder: match images with annotation files by stem and
    apply the optional filters. Returns a list of {"image", "annotation"}
    records with relative paths, sorted by annotation filename.
    """
    matched_items = []

    folder_path = os.path.join(root_path, folder_name)
    img_dir = os.path.join(folder_path, img_folder_name)
    annot_dir = os.path.join(folder_path, annot_folder_name)

    if not os.path.exists(img_dir) or not os.path.exists(annot_dir):
        return []

    # 1. Build image index: { filename_stem: full_filename }
    image_map = {}
    try:
        for f in os.listdir(img_dir):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                image_map[Path(f).stem] = f
    except Exception as e:
        print(f"Error reading image dir {img_dir}: {e}")
        return []

    # 2. Match each annotation JSON to its image
    try:
        json_files = sorted(glob.glob(os.path.join(annot_dir, '*.json')))

        for json_path in json_files:
            json_filename = os.path.basename(json_path)
            stem = Path(json_filename).stem

            if stem not in image_map:
                continue
            img_filename = image_map[stem]
            full_img_path = os.path.join(img_dir, img_filename)

            # Only open the image when a filter actually needs its size
            if gen_unit > 1 or target_ratios:
                if not image_passes_filters(full_img_path, gen_unit,
                                            target_ratios, ratio_tol):
                    continue

            matched_items.append({
                "image": f"{folder_name}/{img_folder_name}/{img_filename}",
                "annotation": f"{folder_name}/{annot_folder_name}/{json_filename}",
            })

    except Exception as e:
        print(f"Error reading annot dir {annot_dir}: {e}")
        return []

    return matched_items


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary.json linking images and annotations.")

    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path containing all data folders.')

    parser.add_argument('--folders', type=str, nargs='+', default=None,
                        help='List of specific folder names to process. '
                             'Default: scan all subfolders (sorted).')
    parser.add_argument('--excluded_folders', type=str, nargs='+', default=[],
                        help='List of folder names to exclude.')
    parser.add_argument('--img_folder', type=str, default='train',
                        help='Name of the folder containing images.')
    parser.add_argument('--annot_folder', type=str, default='train_scene_cam',
                        help='Name of the folder containing JSON annotations.')
    parser.add_argument('--output', type=str, default='summary.json',
                        help='Output JSON filename (written under root_path).')
    parser.add_argument('--gen_unit', type=int, default=1,
                        help='Keep only images whose width and height are '
                             'divisible by this unit (16 for the generation).')
    parser.add_argument('--aspect_ratios', type=str, nargs='+', default=None,
                        help='Keep only images matching one of these aspect '
                             'ratios (e.g. "9:16" "3:4" "1:1"). Default: keep all.')
    parser.add_argument('--ratio_tol', type=float, default=0.01,
                        help='Tolerance for the aspect-ratio comparison.')

    args = parser.parse_args()

    # 1. Parse aspect ratios
    target_ratios = parse_aspect_ratios(args.aspect_ratios)
    if target_ratios:
        print(f"Filtering for aspect ratios: {args.aspect_ratios} "
              f"(Calculated: {[round(r, 3) for r in target_ratios]})")
    else:
        print("No aspect ratio filter applied (keeping all).")

    # 2. Resolve the folder list (user order kept; scanned folders sorted)
    if args.folders:
        folder_list = args.folders
        print(f"Targeting {len(folder_list)} specific folders.")
    else:
        if not os.path.exists(args.root_path):
            print(f"Error: Root path does not exist: {args.root_path}")
            return

        all_folders = sorted(d for d in os.listdir(args.root_path)
                             if os.path.isdir(os.path.join(args.root_path, d)))

        excluded_set = set(args.excluded_folders)
        folder_list = [f for f in all_folders if f not in excluded_set]
        if excluded_set:
            print(f"Scanning root path... Found {len(all_folders)} folders, "
                  f"excluded {len(excluded_set)}, remaining {len(folder_list)}.")
        else:
            print(f"Scanning root path... Found {len(folder_list)} folders.")

    if not folder_list:
        print("No folders found to process.")
        return

    # 3. Multi-threaded processing; results assembled in folder_list order
    results_by_folder = {}
    max_workers = min(32, len(folder_list))
    print(f"Generating summary with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {
            executor.submit(process_subfolder, folder, args.root_path,
                            args.img_folder, args.annot_folder,
                            args.gen_unit, target_ratios, args.ratio_tol): folder
            for folder in folder_list
        }

        for future in tqdm(as_completed(future_to_folder), total=len(folder_list),
                           desc="Summarizing"):
            folder = future_to_folder[future]
            try:
                results_by_folder[folder] = future.result()
            except Exception as exc:
                print(f'{folder} generated an exception: {exc}')

    all_results = []
    for folder in folder_list:
        all_results.extend(results_by_folder.get(folder, []))

    # 4. Write summary.json
    output_path = os.path.join(args.root_path, args.output)
    print(f"Writing {len(all_results)} records to {output_path}...")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("Done.")
    except Exception as e:
        print(f"Failed to write output file: {e}")


if __name__ == "__main__":
    main()
