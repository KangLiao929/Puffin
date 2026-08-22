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

PROMPT = """
Describe this image in one-two concise sentences.
"""


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
        device_map=None,   # IMPORTANT: one process -> one GPU
    )
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = ModelCls.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def list_all_subfolders(input_root: str) -> list[str]:
    """List first-level subfolders under input_root, sorted."""
    if not os.path.isdir(input_root):
        return []
    folders = []
    for name in os.listdir(input_root):
        full = os.path.join(input_root, name)
        if os.path.isdir(full):
            folders.append(name)
    folders.sort()
    return folders


def build_tasks_for_one_folder(input_root: str, folder: str):
    """
    Return tasks list for ONE folder:
      {
        "folder": folder_name,
        "idx": i (0-based within that folder sorted list),
        "image_path": full path,
        "file_name": folder/filename_with_ext
      }
    """
    tasks = []
    in_folder = os.path.join(input_root, folder)
    if not os.path.isdir(in_folder):
        print(f"[WARN] Folder not found, skip: {in_folder}")
        return tasks

    image_files = [f for f in os.listdir(in_folder) if f.lower().endswith(IMG_EXTS)]
    image_files = sorted(image_files)

    for i, fn in enumerate(image_files):
        tasks.append({
            "folder": folder,
            "idx": i,
            "image_path": os.path.join(in_folder, fn),
            "file_name": f"{folder}/{fn}",
        })
    return tasks


@torch.inference_mode()
def infer_one_image(model, processor, device: str, image_path: str, max_new_tokens: int):
    """Run one image inference without any resizing."""
    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
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

    # Trim the prompt tokens from the generated sequence
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
    """One GPU process: load model on its GPU and process its shard of tasks."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    model_path = resolve_model_path(args.model_root, args.model_id)
    model, processor = load_qwen_model(
        model_type=args.model_type,
        model_path=model_path,
        device=device,
        use_flash_attn=args.flash_attn,
    )

    local_results = []
    my_tasks = tasks[rank::world_size]

    iterator = my_tasks
    if rank == 0:
        iterator = tqdm(my_tasks, desc=f"GPU{rank} processing", unit="img", leave=False)

    for t in iterator:
        try:
            label = infer_one_image(
                model=model,
                processor=processor,
                device=device,
                image_path=t["image_path"],
                max_new_tokens=args.max_new_tokens,
            )
            local_results.append({
                "folder": t["folder"],
                "idx": t["idx"],
                "file_name": t["file_name"],
                "label": label,
            })
        except Exception as e:
            local_results.append({
                "folder": t["folder"],
                "idx": t["idx"],
                "file_name": t["file_name"],
                "label": f"ERROR: {type(e).__name__}: {e}",
            })

    return_queue.put(local_results)


def write_one_folder_csv(folder: str, results: list[dict], output_root: str):
    """
    Write: output_root/<folder>/<folder>.csv
    Columns: num, file_name, label
    """
    out_dir = os.path.join(output_root, folder)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{folder}.csv")

    rows = sorted(results, key=lambda x: x["idx"])
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num", "file_name", "label"])
        for i, r in enumerate(rows):
            serial_num = f"{i + 1:06d}"
            w.writerow([serial_num, r["file_name"], r["label"]])

    print(f"[OK] Wrote: {csv_path}  (rows={len(rows)})")


def parse_args():
    p = argparse.ArgumentParser("Multi-GPU Qwen3-VL image captioning")
    p.add_argument(
        "--model_root",
        type=str,
        default="/data/kliao/data/models/",
        help="Local root that contains downloaded HF models (your layout).",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="Qwen3-VL-8B-Instruct",
        help="HF model id (relative under model_root) OR an absolute local path.",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="vl",
        choices=["moe", "vl"],
        help="moe: Qwen3VLMoeForConditionalGeneration; vl: Qwen3VLForConditionalGeneration",
    )
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (data-parallel).")
    p.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2")
    p.add_argument("--max_new_tokens", type=int, default=128)

    p.add_argument("--input_root", type=str, default="/data/kliao/data/Puffin-Pro/Projection/")
    p.add_argument("--output_root", type=str, default="/data/kliao/data/Puffin-Pro/captions/scenes/")

    # Usage examples:
    #   --folders OmniPhotos Flickr360
    #   --folders ALL
    p.add_argument(
        "--folders",
        nargs="+",
        default=["OmniPhotos"],
        help="Folder names under input_root. Use 'ALL' to process every subfolder (sorted).",
    )

    return p.parse_args()


def run_one_folder(folder: str, tasks: list[dict], args):
    if len(tasks) == 0:
        print(f"[WARN] No images found in folder: {folder}, skip.")
        return

    print(f"\n[INFO] Folder: {folder} | images={len(tasks)} | GPUs={args.gpus}")
    q = mp.Queue()

    procs = []
    for rank in range(args.gpus):
        p = mp.Process(target=worker_main, args=(rank, args.gpus, tasks, args, q))
        p.start()
        procs.append(p)

    all_results = []
    for _ in range(args.gpus):
        all_results.extend(q.get())

    for p in procs:
        p.join()

    # Only keep current folder's results (safety)
    all_results = [r for r in all_results if r.get("folder") == folder]
    write_one_folder_csv(folder, all_results, args.output_root)


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is not available."
    total_gpus = torch.cuda.device_count()
    if args.gpus <= 0:
        args.gpus = 1
    if args.gpus > total_gpus:
        print(f"[WARN] Requested gpus={args.gpus} but only {total_gpus} available; using {total_gpus}.")
        args.gpus = total_gpus

    # Set spawn once (safe for CUDA multiprocessing)
    mp.set_start_method("spawn", force=True)

    # Resolve folders list
    folders_upper = [x.strip().upper() for x in args.folders]
    if len(folders_upper) == 1 and folders_upper[0] == "ALL":
        folder_names = list_all_subfolders(args.input_root)
        if len(folder_names) == 0:
            print("[WARN] No subfolders found under input_root.")
            return
    else:
        folder_names = [x.strip() for x in args.folders if x.strip()]

    print(f"[INFO] Using GPUs: {args.gpus} | model_type={args.model_type} | flash_attn={args.flash_attn}")
    print(f"[INFO] model_id={args.model_id} | input_root={args.input_root} | output_root={args.output_root}")
    print(f"[INFO] folders={folder_names}")

    # Process folders sequentially
    for folder in folder_names:
        folder = f"{folder}/train/" # Only process train/ subfolder
        tasks = build_tasks_for_one_folder(args.input_root, folder)
        run_one_folder(folder, tasks, args)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
