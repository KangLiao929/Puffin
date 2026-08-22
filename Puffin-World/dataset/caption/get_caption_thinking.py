import os
import csv
import argparse
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor

# Allow loading very large / truncated images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ----------------------------------------------------------------------------
# 1. Parameter Mapping Functions
# ----------------------------------------------------------------------------

def map_fov(x):
    try:
        x = float(x)
    except:
        return "unknown"
    
    if 0.3490 <= x < 0.6109:
        return "close-up"
    elif 0.6109 <= x < 1.1345:
        return "medium shot"
    elif 1.1345 <= x < 1.5708:
        return "wide-angle"
    elif 1.5708 <= x <= 1.8326:
        return "ultra wide-angle"
    else:
        return "unknown"

def map_pitch(x):
    try:
        x = float(x)
    except:
        return "unknown"

    if -0.7854 <= x < -0.3491:
        return "large tilt-down"
    elif -0.3491 <= x < -0.0873:
        return "small tilt-down"
    elif -0.0873 <= x <= 0.0873:
        return "near straight-on shot"
    elif 0.0873 < x <= 0.3491:
        return "small tilt-up"
    elif 0.3491 < x <= 0.7854:
        return "large tilt-up"
    else:
        return "unknown"

def map_roll(x):
    try:
        x = float(x)
    except:
        return "unknown"

    if -0.7854 <= x < -0.3491:
        return "large clockwise Dutch angle"
    elif -0.3491 <= x < -0.0873:
        return "small clockwise Dutch angle"
    elif -0.0873 <= x <= 0.0873:
        return "near level shot"
    elif 0.0873 < x <= 0.3491:
        return "small counterclockwise Dutch angle"
    elif 0.3491 < x <= 0.7854:
        return "large counterclockwise Dutch angle"
    else:
        return "unknown"

def map_k1(x):
    try:
        x = float(x)
    except:
        x = 0.0
        
    if x == 0.0:
        return "nearly undistorted"
    elif 0 < x < 0.1:
        return "small distortion"
    elif 0.1 <= x <= 0.5:
        return "large distortion"
    else:
        return "unknown"

# ----------------------------------------------------------------------------
# 2. Model & Processing Logic
# ----------------------------------------------------------------------------

def resolve_model_path(model_root: str, model_id_or_path: str) -> str:
    """Allow passing either an absolute local path or an HF model id stored under model_root."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    return os.path.join(model_root, model_id_or_path)


def load_qwen_model(model_type: str, model_path: str, device: str, use_flash_attn: bool):
    """
    model_type:
      - "moe": Qwen3VLMoeForConditionalGeneration
      - "vl" : Qwen3VLForConditionalGeneration
    """
    if model_type == "moe":
        from transformers import Qwen3VLMoeForConditionalGeneration as ModelCls
    elif model_type == "vl":
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    else:
        raise ValueError(f"Unknown model_type={model_type}, choose from ['moe','vl'].")

    kwargs = dict(
        torch_dtype="auto",
        device_map=None,
    )
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    print(f"[INFO] Loading model from {model_path} on {device}...")
    model = ModelCls.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def build_tasks_for_one_folder(input_root: str, folder_name: str, args):
    """
    Revised logic:
    1. Print total images in CSV.
    2. Handle start_idx / end_idx logic where end_idx=None means 'to the end'.
    3. Print clear range info.
    """
    tasks = []
    
    base_dir = os.path.join(input_root, folder_name)
    csv_path = os.path.join(base_dir, "train.csv")
    img_dir = os.path.join(base_dir, "train")

    if not os.path.isdir(img_dir):
        print(f"[WARN] Image directory not found: {img_dir}, skipping.")
        return tasks

    if not os.path.exists(csv_path):
        print(f"[WARN] Metadata CSV not found: {csv_path}, skipping.")
        return tasks

    print(f"[INFO] Reading metadata from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Ensure deterministic order
        rows.sort(key=lambda r: r.get('fname', ''))

    total_rows = len(rows)
    
    # --- Index Logic ---
    # 1. Default start is 0
    start_i = args.start_idx if args.start_idx is not None else 0
    
    # 2. Default end is total_rows (Process until the end)
    if args.end_idx is None:
        end_i = total_rows
    else:
        end_i = args.end_idx

    # 3. Clamp values
    start_i = max(0, start_i)
    end_i = min(total_rows, end_i)

    # 4. Info Display (Runs on Main Process/Rank 0)
    print(f"┌── Folder: {folder_name} ──────────────────────────")
    print(f"│ Total images in CSV : {total_rows}")
    print(f"│ Processing Range    : [{start_i} : {end_i}]")
    print(f"│ Actual Task Count   : {max(0, end_i - start_i)}")
    print(f"└─────────────────────────────────────────────────")

    if start_i >= total_rows:
        print(f"[WARN] Start index {start_i} out of bounds (total {total_rows}). Skipping.")
        return []

    # Slice the rows
    target_rows = rows[start_i:end_i]

    for i, row in enumerate(target_rows):
        fname = row.get('fname')
        if not fname:
            continue

        full_img_path = os.path.join(img_dir, fname)
        
        if not os.path.exists(full_img_path):
            continue

        try:
            roll_val = row.get('roll')
            pitch_val = row.get('pitch')
            vfov_val = row.get('vfov')
            k1_val = row.get('k1', 0.0)
            if k1_val == '' or k1_val is None:
                k1_val = 0.0
        except Exception as e:
            print(f"[ERR] Error parsing row for {fname}: {e}")
            continue

        roll_term = map_roll(roll_val)
        pitch_term = map_pitch(pitch_val)
        fov_term = map_fov(vfov_val)
        k1_term = map_k1(k1_val)

        # Updated Prompt with explicit ordering and formatting
        prompt_text = (
            f"Analyze the visual-spatial cues of this image based on the following known camera parameters:\n"
            f"- Horizontal orientation: {roll_term}\n"
            f"- Vertical orientation: {pitch_term}\n"
            f"- Field-of-view: {fov_term}\n"
            f"- Radial geometry: {k1_term}\n\n"
            "Your task is to generate a coherent reasoning paragraph enclosed within <think> and </think> tags. "
            "You must explain the visual evidence supporting these parameters in the **exact order** listed below:\n"
            "1. Discuss **Horizontal Orientation**. \n"
            "2. Discuss **Vertical Orientation**.\n"
            "3. Discuss **Field-of-View**.\n"
            "4. Discuss **Radial Geometry**.\n\n"
            "Combine these observations into exactly four concise sentences without explicitly labeling the steps (e.g., do not say 'First... Second...'). "
            "Focus strictly on how the visual content aligns with the given parameters."
        )

        tasks.append({
            "folder": folder_name,
            "idx": start_i + i, # Global index relative to the CSV
            "image_path": full_img_path,
            "file_name": f"{folder_name}/train/{fname}",
            "prompt": prompt_text
        })

    return tasks


@torch.inference_mode()
def infer_one_image(model, processor, device: str, image_path: str, prompt_text: str, max_new_tokens: int):
    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    out = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return out


def worker_main(rank: int, world_size: int, tasks: list[dict], args, return_queue: mp.Queue):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    model_path = resolve_model_path(args.model_root, args.model_id)
    
    try:
        model, processor = load_qwen_model(
            model_type=args.model_type,
            model_path=model_path,
            device=device,
            use_flash_attn=args.flash_attn,
        )
    except Exception as e:
        print(f"[FATAL] GPU{rank} failed to load model: {e}")
        return

    local_results = []
    my_tasks = tasks[rank::world_size]

    iterator = my_tasks
    # Only show progress bar on Rank 0
    if rank == 0:
        iterator = tqdm(my_tasks, desc=f"GPU{rank} processing", unit="img", leave=False)

    for t in iterator:
        try:
            label = infer_one_image(
                model=model,
                processor=processor,
                device=device,
                image_path=t["image_path"],
                prompt_text=t["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
            local_results.append({
                "folder": t["folder"],
                "idx": t["idx"],
                "file_name": t["file_name"],
                "label": label,
            })
        except Exception as e:
            # Only print errors, standard logs suppressed
            print(f"[ERR] GPU{rank} failed on {t['file_name']}: {e}")
            local_results.append({
                "folder": t["folder"],
                "idx": t["idx"],
                "file_name": t["file_name"],
                "label": f"ERROR: {type(e).__name__}: {e}",
            })

    return_queue.put(local_results)


def write_one_folder_csv(folder: str, results: list[dict], output_root: str, args):
    """
    Write: output_root/<folder>/<folder>_{start}_{end}.csv
    """
    out_dir = os.path.join(output_root, folder)
    os.makedirs(out_dir, exist_ok=True)
    
    # Determine suffix based on args to avoid overwrite
    s = args.start_idx if args.start_idx is not None else 0
    e = args.end_idx if args.end_idx is not None else "end"
    suffix = f"_{s}_{e}"
        
    csv_path = os.path.join(out_dir, f"{folder}{suffix}.csv")

    rows = sorted(results, key=lambda x: x["idx"])
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num", "file_name", "label"])
        for r in rows:
            # Use 6-digit zero padded index
            serial_num = f"{r['idx']:06d}" 
            w.writerow([serial_num, r["file_name"], r["label"]])

    print(f"[OK] Wrote: {csv_path}  (rows={len(rows)})")


def parse_args():
    p = argparse.ArgumentParser("Multi-GPU Qwen3-VL camera reasoning")
    p.add_argument("--model_root", type=str, default="/data/kliao/data/models/",
                   help="Local root that contains downloaded HF models.")
    p.add_argument("--model_id", type=str, default="Qwen3-VL-32B-Instruct",
                   help="HF model id or absolute path.")
    p.add_argument("--model_type", type=str, default="vl", choices=["moe", "vl"])
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
    p.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2")

    p.add_argument("--max_new_tokens", type=int, default=256, help="Increased for thinking process")

    p.add_argument("--input_root", type=str, default="/data/kliao/data/Puffin-Pro/Projection/",
                   help="Root dir containing subfolders.")
    p.add_argument("--output_root", type=str, default="/data/kliao/data/Puffin-Pro/captions/thinking/")
    p.add_argument("--folders", type=str, default="OmniPhotos",
                   help="Comma-separated folder names or 'ALL'.")

    # Modified help text
    p.add_argument("--start_idx", type=int, default=None, 
                   help="Start index (inclusive). Default: 0")
    p.add_argument("--end_idx", type=int, default=None, 
                   help="End index (exclusive). Default: Process until the end of the CSV.")

    return p.parse_args()


def run_one_folder(folder: str, tasks: list[dict], args):
    if len(tasks) == 0:
        print(f"[WARN] No tasks created for folder: {folder}, skip.")
        return

    print(f"[INFO] Launching workers for folder: {folder} | GPUs={args.gpus}")
    q = mp.Queue()

    procs = []
    for rank in range(args.gpus):
        p = mp.Process(target=worker_main, args=(rank, args.gpus, tasks, args, q))
        p.start()
        procs.append(p)

    all_results = []
    # Collect results
    for _ in range(args.gpus):
        all_results.extend(q.get())

    for p in procs:
        p.join()

    # Filter results for this folder (sanity check)
    all_results = [r for r in all_results if r.get("folder") == folder]
    write_one_folder_csv(folder, all_results, args.output_root, args)


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is not available."
    total_gpus = torch.cuda.device_count()
    if args.gpus <= 0:
        args.gpus = 1
    if args.gpus > total_gpus:
        print(f"[WARN] Requested gpus={args.gpus} but only {total_gpus} available; using {total_gpus}.")
        args.gpus = total_gpus

    # Set spawn method safely
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass 

    folders_arg = args.folders.strip()
    if folders_arg.upper() == "ALL":
        if not os.path.isdir(args.input_root):
             print(f"[ERR] Input root {args.input_root} does not exist.")
             return
        folder_names = sorted([d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))])
    else:
        folder_names = [x.strip() for x in folders_arg.split(",") if x.strip()]

    print(f"[INFO] Using GPUs: {args.gpus} | model_type={args.model_type} | flash_attn={args.flash_attn}")
    print(f"[INFO] Folders to process: {folder_names}")

    # Process folders sequentially
    for folder in folder_names:
        # Build tasks on Rank 0
        tasks = build_tasks_for_one_folder(args.input_root, folder, args)
        run_one_folder(folder, tasks, args)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()