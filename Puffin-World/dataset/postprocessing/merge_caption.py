"""Merge camera parameters and captions into per-image JSON files.

For each scene folder, join path_a/<folder>/train.csv (camera params:
fname, roll, pitch, vfov, optional k1/width/height) with the caption CSV in
path_b/<folder>/ (columns: file_name, label) by filename, then write one
JSON per image to path_a/<folder>/<folder_target>/:

    {"caption": "<label> The camera parameters (roll, pitch, field-of-view,
                 and radial distortion) are: r, p, v, k."}

With --thinking the caption is composed in <think>/<answer> format instead:

    {"caption": "<think> <label> </think><answer>r, p, v, k</answer>"}

Folders are processed in parallel threads; --write_size embeds width/height
from train.csv into each JSON.

Examples:
    python dataset/postprocessing/merge_caption.py \
        --path_a /path/Projection --path_b /path/captions/scenes
    python dataset/postprocessing/merge_caption.py --thinking \
        --path_a /path/Projection --path_b /path/captions/thinking \
        --folder_target train_thinking_cam --folders OmniPhotos
"""

import os
import glob
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def pick_caption_csv(folder_path_B, folder_name):
    """
    Pick the caption CSV in path_b/<folder>/ deterministically:
    prefer <folder>.csv; otherwise take the alphabetically first CSV
    (with a warning when several candidates exist).
    """
    preferred = os.path.join(folder_path_B, f"{folder_name}.csv")
    if os.path.exists(preferred):
        return preferred

    csv_files = sorted(glob.glob(os.path.join(folder_path_B, '*.csv')))
    if not csv_files:
        return None
    if len(csv_files) > 1:
        tqdm.write(f"[Warning] {folder_name}: {len(csv_files)} CSVs found in B, "
                   f"using {os.path.basename(csv_files[0])}")
    return csv_files[0]


def process_subfolder(folder_name, root_path_A, root_path_B, thinking=False,
                      folder_target="train_scene_cam", write_size=False):
    """
    Process a single subfolder: merge CSV data and generate JSON captions.
    """
    try:
        # 1. Construct full paths
        folder_path_A = os.path.join(root_path_A, folder_name)
        folder_path_B = os.path.join(root_path_B, folder_name)

        csv_path_A = os.path.join(folder_path_A, 'train.csv')

        # Validation: Check if paths exist
        if not os.path.exists(folder_path_A):
            return f"[Skip] Folder not found in A: {folder_name}"
        if not os.path.exists(folder_path_B):
            return f"[Skip] Folder not found in B: {folder_name}"

        # 2. Find caption CSV in B
        csv_path_B = pick_caption_csv(folder_path_B, folder_name)
        if csv_path_B is None:
            return f"[Skip] No CSV found in B for: {folder_name}"

        # 3. Read and process CSV A (camera params)
        if not os.path.exists(csv_path_A):
            return f"[Skip] train.csv not found in A: {folder_name}"

        df_A = pd.read_csv(csv_path_A)

        # Handle missing 'k1' column
        if 'k1' not in df_A.columns:
            df_A['k1'] = 0.0
        else:
            df_A['k1'] = df_A['k1'].fillna(0.0)

        # Check required columns in A
        required_cols_A = ['fname', 'roll', 'pitch', 'vfov']
        if write_size:
            required_cols_A = required_cols_A + ['width', 'height']
        if not all(col in df_A.columns for col in required_cols_A):
            return f"[Error] Missing columns in train.csv: {folder_name}"

        # 4. Read and process CSV B (captions)
        df_B = pd.read_csv(csv_path_B)

        if 'file_name' not in df_B.columns or 'label' not in df_B.columns:
            return f"[Error] Missing columns in B csv: {folder_name}"

        # Match by basename (B stores relative paths like "Scene/train/x.jpg")
        df_B['fname_match'] = df_B['file_name'].apply(lambda x: os.path.basename(str(x)))

        # 5. Merge the two tables on filename
        merged_df = pd.merge(df_A, df_B, left_on='fname', right_on='fname_match', how='inner')

        if merged_df.empty:
            return f"[Warning] No matching filenames found in: {folder_name}"

        # 6. Create output directory
        output_dir = os.path.join(folder_path_A, folder_target)
        os.makedirs(output_dir, exist_ok=True)

        # 7. Generate JSON files
        files_written = 0
        for _, row in merged_df.iterrows():
            fname = row['fname']
            label = str(row['label']).strip()

            roll = row['roll']
            pitch = row['pitch']
            vfov = row['vfov']
            k1 = row['k1']

            if thinking:
                new_caption_text = (
                    f"<think> {label} </think><answer>"
                    f"{roll:.4f}, {pitch:.4f}, {vfov:.4f}, {k1:.4f}</answer>"
                )
            else:
                new_caption_text = (
                    f"{label} The camera parameters (roll, pitch, field-of-view, and radial distortion) are: "
                    f"{roll:.4f}, {pitch:.4f}, {vfov:.4f}, {k1:.4f}."
                )

            json_content = {
                "caption": new_caption_text
            }

            # Optionally embed image size from train.csv (path_a)
            if write_size:
                json_content["width"] = int(row['width'])
                json_content["height"] = int(row['height'])

            file_stem = Path(fname).stem
            json_output_path = os.path.join(output_dir, f"{file_stem}.json")

            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, ensure_ascii=False)

            files_written += 1

        return f"[Success] {folder_name}: {files_written} files."

    except Exception as e:
        return f"[Exception] {folder_name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Merge camera params and captions into per-image JSON files.")

    parser.add_argument('--path_a', type=str, required=True,
                        help='Root path A containing image folders and train.csv')
    parser.add_argument('--path_b', type=str, required=True,
                        help='Root path B containing caption CSVs')

    parser.add_argument('--folders', type=str, nargs='+', default=None,
                        help='List of specific folder names to process. '
                             'Default: scan all subfolders of path A.')

    parser.add_argument('--thinking', action='store_true',
                        help='Compose captions in <think>/<answer> format.')

    parser.add_argument('--folder_target', type=str, default='train_scene_cam',
                        help='Target folder name to save JSON captions.')

    parser.add_argument('--write_size', action='store_true',
                        help='Also write image width/height from path_a train.csv '
                             'into each JSON.')

    args = parser.parse_args()

    # Resolve target folders
    if args.folders:
        folder_list = args.folders
        print(f"Targeting {len(folder_list)} specific folders provided by user.")
    else:
        if not os.path.exists(args.path_a):
            print(f"Error: Path A does not exist: {args.path_a}")
            return

        folder_list = [d for d in os.listdir(args.path_a)
                       if os.path.isdir(os.path.join(args.path_a, d))]
        print(f"No specific folders provided. Scanning Path A... Found {len(folder_list)} folders.")

    if not folder_list:
        print("No folders found to process.")
        return

    # Multi-threaded processing over folders
    max_workers = min(32, len(folder_list))
    print(f"Starting processing with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {
            executor.submit(process_subfolder, folder, args.path_a, args.path_b,
                            args.thinking, args.folder_target, args.write_size): folder
            for folder in folder_list
        }

        for future in tqdm(as_completed(future_to_folder), total=len(folder_list),
                           desc="Processing"):
            result = future.result()
            if not result.startswith("[Success]"):
                tqdm.write(result)

    print("All tasks completed.")


if __name__ == "__main__":
    main()
