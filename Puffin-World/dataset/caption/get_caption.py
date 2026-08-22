"""Unified multi-GPU VLM captioning / labeling for Puffin datasets.

One script for all caption tasks; switch with --task. Each task ships the
task-model-prompt defaults of the original standalone scripts, and any CLI
flag overrides its preset:

  scene         : 1-2 sentence scene caption          (Qwen2.5-VL-7B, train/ only)
  camera_height : 5-class camera-height label         (Qwen3-VL-32B,  640x320 resize)
  filter        : binary low-quality-image filter 0/1 (Qwen2.5-VL-7B, train/ only)
  thinking      : camera-parameter reasoning <think>  (Qwen3-VL-32B,  prompt built
                  per image from <folder>/train.csv roll/pitch/vfov/k1)

Data-parallel over GPUs: one process per GPU, tasks sharded round-robin,
results merged and written as CSV (num, file_name, label) per scene:
  output_root/<scene>/<scene>[_<start>_<end>].csv

Examples:
    python dataset/caption/get_caption.py --task scene --gpus 4 --folders ALL
    python dataset/caption/get_caption.py --task camera_height \
        --folders OmniPhotos,Flickr360 --model_id Qwen3-VL-8B-Instruct
    python dataset/caption/get_caption.py --task thinking \
        --folders OmniPhotos --start_idx 0 --end_idx 2000
"""

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
# 1. Prompts
# ----------------------------------------------------------------------------

SCENE_PROMPT = """
Describe this image in one-two concise sentences.
"""

CAMERA_HEIGHT_PROMPT = """
Analyze the physical camera height (elevation) assuming a strictly HORIZONTAL camera angle.
Do not consider looking-up or looking-down angles.

Choose exactly ONE label from the following five options:
Underwater shot
Low-position shot
Eye-level shot
High-position shot
Aerial shot

Classify based on the position of scene objects relative to the horizon line and geometric and semantic cues (perspective geometry, visible ground coverage, and object scale).:

   [Underwater shot]
   - Physical Height: Submerged in water (refraction, blue/green cast, turbidity). Distinct from air.
   - Key Cue: The entire visual field is immersed in a **volumetric liquid medium**, characterized by optical attenuation and refraction. The concept of a sharp "horizon line" is often replaced by a gradient of water visibility.

   [Low-position shot]
   - Physical Height: Ground level (< 1.0m).
   - Horizontal View Geometry: The camera is closer to the floor than the objects.
   - Key Cue: Most foreground objects (chairs, pedestrians, cars) tower *above* the image's center line.

   [Eye-level shot]
   - Physical Height: Human standing height (~1.5m - 2m).
   - Horizontal View Geometry: Camera height is approximately at human eye level with a near-horizontal viewing direction. The camera is at the similar height as the subjects.
   - The image center line intersects the physical "middle band" of the environment, capturing objects (furniture, buildings) in **neutral side-profile** (revealing neither top surfaces nor undersides). It geometrically bisects the scene, creating an equal balance between the ground plane and the ceiling/sky.

   [High-position shot]
   - Physical Height: Elevated but terrestrial (~3m - 10m).
   - Horizontal View Geometry: Camera is positioned high (e.g., on a ladder or tall vehicle) looking straight forward.
   - Key Cue: Most scene objects (furniture, crowds, cars) sit *below* the image's center line.

   [Aerial shot]
   - Physical Height: Sky and space level (> 20m).
   - Horizontal View Geometry: Drone or skyscraper view looking out at the skyline or the space.
   - Key Cue: The scene consists of rooftops, distant horizons, clouds, or starscape. Ground objects are insignificant in scale.

Output ONLY the label.
"""

FILTER_PROMPT = """
Analyze the image based on the following criteria. Determine if ANY of the conditions below are met:

1. Composition: A single face, person, or head dominates the image (occupies more than 50% of the area).
2. Low Information (Natural): The image consists solely of pure-color sky or ground with no other distinct objects.
3. Low Information (Artificial): The image consists solely of pure-color walls, roads, or floors lacking geometric structure/pattern or other content.
4. Artifacts: Presence of obvious striped artifacts or interference patterns (e.g., moiré patterns, banding).
5. Obscuration: Large areas of the image are covered by mosaic, pixelation, or censoring.
6. Watermarks: Presence of heavy, dense, or intrusive watermarks covering the image.

Output Rule:
- Return '1' if at least one condition is true.
- Return '0' otherwise.

Output ONLY the label (0 or 1).
"""

# ----------------------------------------------------------------------------
# 2. Task presets (the original per-script defaults; CLI flags override them)
# ----------------------------------------------------------------------------

TASK_PRESETS = {
    "scene": dict(
        prompt=SCENE_PROMPT,
        model_root="/mnt/afs_100t/NTU_slab/kliao/models/",
        model_id="Qwen2.5-VL-7B-Instruct",
        model_type="qwen2p5",
        max_new_tokens=128,
        input_root="/mnt/afs_100t/NTU_slab/kliao/data/Puffin-Pro/Projection/",
        output_root="/mnt/afs_100t/NTU_slab/kliao/data/Puffin-Pro/captions/scenes/",
        folders=["OmniPhotos", "OmniBlender"],
        train_only=True,
        resize_w=0, resize_h=0,
    ),
    "camera_height": dict(
        prompt=CAMERA_HEIGHT_PROMPT,
        model_root="/data/kliao/data/models/",
        model_id="Qwen3-VL-32B-Instruct",
        model_type="qwen3",
        max_new_tokens=16,
        input_root="/data/kliao/data/360_dataset/",
        output_root="/data/kliao/data/Puffin-Pro/captions/camera_height/",
        folders=["OmniPhotos"],
        train_only=False,
        resize_w=640, resize_h=320,
    ),
    "filter": dict(
        prompt=FILTER_PROMPT,
        model_root="/data/kliao/data/models/",
        model_id="Qwen2.5-VL-7B-Instruct",
        model_type="qwen2p5",
        max_new_tokens=16,
        input_root="/data/kliao/data/Puffin-Pro/Projection/",
        output_root="/data/kliao/data/Puffin-Pro/captions/marks/",
        folders=["OmniPhotos", "OmniBlender"],
        train_only=True,
        resize_w=0, resize_h=0,
    ),
    "thinking": dict(
        prompt=None,  # built per image from <folder>/train.csv camera params
        model_root="/data/kliao/data/models/",
        model_id="Qwen3-VL-32B-Instruct",
        model_type="qwen3",
        max_new_tokens=256,
        input_root="/data/kliao/data/Puffin-Pro/Projection/",
        output_root="/data/kliao/data/Puffin-Pro/captions/thinking/",
        folders=["OmniPhotos"],
        train_only=True,
        resize_w=0, resize_h=0,
    ),
}

# ----------------------------------------------------------------------------
# 3. Camera-parameter -> language mapping (thinking task)
# ----------------------------------------------------------------------------

def map_fov(x):
    try:
        x = float(x)
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
        x = 0.0

    if x == 0.0:
        return "nearly undistorted"
    elif 0 < x < 0.1:
        return "small distortion"
    elif 0.1 <= x <= 0.5:
        return "large distortion"
    else:
        return "unknown"


def build_thinking_prompt(roll_val, pitch_val, vfov_val, k1_val) -> str:
    roll_term = map_roll(roll_val)
    pitch_term = map_pitch(pitch_val)
    fov_term = map_fov(vfov_val)
    k1_term = map_k1(k1_val)

    return (
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

# ----------------------------------------------------------------------------
# 4. Model loading
# ----------------------------------------------------------------------------

def resolve_model_path(model_root: str, model_id_or_path: str) -> str:
    """Allow passing either an absolute local path or an HF model id stored under model_root."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    return os.path.join(model_root, model_id_or_path)


def load_qwen_model(model_type: str, model_path: str, device: str, use_flash_attn: bool):
    """
    model_type:
      - "moe"     : Qwen3VLMoeForConditionalGeneration
      - "qwen3"   : Qwen3VLForConditionalGeneration
      - "qwen2p5" : Qwen2_5_VLForConditionalGeneration
    """
    if model_type == "moe":
        from transformers import Qwen3VLMoeForConditionalGeneration as ModelCls
    elif model_type == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    elif model_type == "qwen2p5":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    else:
        raise ValueError(f"Unknown model_type={model_type}, choose from ['moe','qwen3','qwen2p5'].")

    kwargs = dict(
        torch_dtype="auto",
        device_map=None,   # IMPORTANT: one process -> one GPU
    )
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    print(f"[INFO] Loading model from {model_path} on {device}...")
    model = ModelCls.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

# ----------------------------------------------------------------------------
# 5. Task building
# ----------------------------------------------------------------------------

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


def slice_range(total: int, start_idx, end_idx):
    """Clamp [start_idx, end_idx) against total; None means full range."""
    start_i = 0 if start_idx is None else max(0, start_idx)
    end_i = total if end_idx is None else min(total, end_idx)
    return start_i, end_i


def build_image_tasks(input_root: str, scene: str, prompt: str, train_only: bool,
                      start_idx, end_idx):
    """
    Image-list tasks (scene / camera_height / filter).

    Each task:
      {
        "folder": scene name,
        "idx": global index within the sorted image list,
        "image_path": absolute path,
        "file_name": relative path stored in CSV (e.g. "OmniPhotos/train/xxx.jpg"),
        "prompt": the task prompt,
      }
    """
    tasks = []
    folder_rel = os.path.join(scene, "train") if train_only else scene
    in_folder = os.path.join(input_root, folder_rel)
    if not os.path.isdir(in_folder):
        print(f"[WARN] Folder not found, skip: {in_folder}")
        return tasks

    image_files = sorted(f for f in os.listdir(in_folder) if f.lower().endswith(IMG_EXTS))

    start_i, end_i = slice_range(len(image_files), start_idx, end_idx)
    if start_i >= len(image_files) and len(image_files) > 0:
        print(f"[WARN] Start index {start_i} out of bounds (total {len(image_files)}). Skipping.")
        return tasks

    # Normalize to forward slashes for CSV consistency
    folder_rel_norm = folder_rel.replace("\\", "/").strip("/")

    for i, fn in enumerate(image_files[start_i:end_i]):
        tasks.append({
            "folder": scene,
            "idx": start_i + i,
            "image_path": os.path.join(in_folder, fn),
            "file_name": f"{folder_rel_norm}/{fn}",
            "prompt": prompt,
        })
    return tasks


def build_thinking_tasks(input_root: str, scene: str, start_idx, end_idx):
    """
    CSV-driven tasks (thinking): read <scene>/train.csv and build a per-image
    reasoning prompt from its roll/pitch/vfov/k1 columns.
    """
    tasks = []
    base_dir = os.path.join(input_root, scene)
    csv_path = os.path.join(base_dir, "train.csv")
    img_dir = os.path.join(base_dir, "train")

    if not os.path.isdir(img_dir):
        print(f"[WARN] Image directory not found: {img_dir}, skipping.")
        return tasks
    if not os.path.exists(csv_path):
        print(f"[WARN] Metadata CSV not found: {csv_path}, skipping.")
        return tasks

    print(f"[INFO] Reading metadata from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        rows.sort(key=lambda r: r.get("fname", ""))  # deterministic order

    total_rows = len(rows)
    start_i, end_i = slice_range(total_rows, start_idx, end_idx)

    print(f"┌── Folder: {scene} ──────────────────────────")
    print(f"│ Total images in CSV : {total_rows}")
    print(f"│ Processing Range    : [{start_i} : {end_i}]")
    print(f"│ Actual Task Count   : {max(0, end_i - start_i)}")
    print(f"└─────────────────────────────────────────────────")

    if start_i >= total_rows:
        print(f"[WARN] Start index {start_i} out of bounds (total {total_rows}). Skipping.")
        return tasks

    for i, row in enumerate(rows[start_i:end_i]):
        fname = row.get("fname")
        if not fname:
            continue
        full_img_path = os.path.join(img_dir, fname)
        if not os.path.exists(full_img_path):
            continue

        k1_val = row.get("k1", 0.0)
        if k1_val == "" or k1_val is None:
            k1_val = 0.0

        tasks.append({
            "folder": scene,
            "idx": start_i + i,  # global index relative to the CSV
            "image_path": full_img_path,
            "file_name": f"{scene}/train/{fname}",
            "prompt": build_thinking_prompt(row.get("roll"), row.get("pitch"),
                                            row.get("vfov"), k1_val),
        })
    return tasks

# ----------------------------------------------------------------------------
# 6. Inference workers
# ----------------------------------------------------------------------------

@torch.inference_mode()
def infer_one_image(model, processor, device: str, image_path: str, prompt_text: str,
                    resize_to, max_new_tokens: int):
    img = Image.open(image_path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.BILINEAR)

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
    try:
        model, processor = load_qwen_model(
            model_type=args.model_type,
            model_path=model_path,
            device=device,
            use_flash_attn=args.flash_attn,
        )
    except Exception as e:
        print(f"[FATAL] GPU{rank} failed to load model: {e}")
        return_queue.put([])  # keep the main process from hanging on q.get()
        return

    resize_to = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)  # (W,H)

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
                prompt_text=t["prompt"],
                resize_to=resize_to,
                max_new_tokens=args.max_new_tokens,
            )
            local_results.append({
                "folder": t["folder"],
                "idx": t["idx"],
                "file_name": t["file_name"],
                "label": label,
            })
        except Exception as e:
            print(f"[ERR] GPU{rank} failed on {t['file_name']}: {e}")
            local_results.append({
                "folder": t["folder"],
                "idx": t["idx"],
                "file_name": t["file_name"],
                "label": f"ERROR: {type(e).__name__}: {e}",
            })

    return_queue.put(local_results)

# ----------------------------------------------------------------------------
# 7. Output
# ----------------------------------------------------------------------------

def write_one_scene_csv(scene: str, results: list[dict], args):
    """
    Write: output_root/<scene>/<scene>[_<start>_<end>].csv
    Columns: num, file_name, label

    The range suffix is always present for the thinking task (original
    behavior) and added to other tasks when --start_idx/--end_idx is given,
    so parallel range jobs never overwrite each other. Serial numbers use
    the global index for the thinking task and a 1-based row counter
    otherwise (original behaviors).
    """
    out_dir = os.path.join(args.output_root, scene)
    os.makedirs(out_dir, exist_ok=True)

    suffix = ""
    if args.task == "thinking" or args.start_idx is not None or args.end_idx is not None:
        s = args.start_idx if args.start_idx is not None else 0
        e = args.end_idx if args.end_idx is not None else "end"
        suffix = f"_{s}_{e}"
    csv_path = os.path.join(out_dir, f"{scene}{suffix}.csv")

    rows = sorted(results, key=lambda x: x["idx"])
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num", "file_name", "label"])
        for i, r in enumerate(rows):
            serial = r["idx"] if args.task == "thinking" else i + 1
            w.writerow([f"{serial:06d}", r["file_name"], r["label"]])

    print(f"[OK] Wrote: {csv_path}  (rows={len(rows)})")


def run_one_scene(scene: str, tasks: list[dict], args):
    """Run multi-GPU captioning for a single scene and write its CSV."""
    if len(tasks) == 0:
        print(f"[WARN] No tasks created for scene: {scene}, skip.")
        return

    print(f"\n[INFO] Scene: {scene} | images={len(tasks)} | GPUs={args.gpus}")
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

    # Keep only this scene's results (safety)
    all_results = [r for r in all_results if r.get("folder") == scene]
    write_one_scene_csv(scene, all_results, args)

# ----------------------------------------------------------------------------
# 8. CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        "Unified multi-GPU VLM captioning (--task selects task/model/prompt preset; "
        "any flag overrides its preset default)")
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_PRESETS),
                   help="Which captioning task to run.")

    # Preset-backed options (default None -> filled from TASK_PRESETS[task])
    p.add_argument("--model_root", type=str, default=None,
                   help="Local root that contains downloaded HF models.")
    p.add_argument("--model_id", type=str, default=None,
                   help="HF model id (relative under model_root) OR an absolute local path.")
    p.add_argument("--model_type", type=str, default=None,
                   choices=["moe", "qwen3", "qwen2p5"],
                   help="moe: Qwen3VLMoe; qwen3: Qwen3VL; qwen2p5: Qwen2_5_VL.")
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--input_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--folders", type=str, nargs="+", default=None,
                   help="Scene folder names under input_root (space- and/or "
                        "comma-separated). Use 'ALL' for every subfolder (sorted).")
    p.add_argument("--prompt", type=str, default=None,
                   help="Override the preset prompt (ignored by the thinking task, "
                        "whose prompt is built per image from train.csv).")
    p.add_argument("--resize_w", type=int, default=None,
                   help="Resize width before inference; <=0 means no resize.")
    p.add_argument("--resize_h", type=int, default=None,
                   help="Resize height before inference; <=0 means no resize.")

    # Shared options
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs (data-parallel).")
    p.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2")
    p.add_argument("--start_idx", type=int, default=None,
                   help="Start index (inclusive) into the sorted image/CSV list. Default: 0")
    p.add_argument("--end_idx", type=int, default=None,
                   help="End index (exclusive). Default: process until the end.")

    args = p.parse_args()

    # Fill unset options from the task preset
    preset = TASK_PRESETS[args.task]
    for key in ("model_root", "model_id", "model_type", "max_new_tokens",
                "input_root", "output_root", "folders", "prompt",
                "resize_w", "resize_h"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    args.train_only = preset["train_only"]

    if args.task == "thinking" and args.prompt is not None:
        print("[WARN] --prompt is ignored for the thinking task (prompt is built per image).")
    return args


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
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Resolve scenes: split comma-separated tokens, 'ALL' expands to all subfolders
    scene_names = [s for tok in args.folders for s in tok.split(",") if s.strip()]
    scene_names = [s.strip() for s in scene_names]
    if len(scene_names) == 1 and scene_names[0].upper() == "ALL":
        scene_names = list_all_subfolders(args.input_root)
        if len(scene_names) == 0:
            print("[WARN] No subfolders found under input_root.")
            return

    print(f"[INFO] task={args.task} | GPUs={args.gpus} | model_type={args.model_type} | flash_attn={args.flash_attn}")
    print(f"[INFO] model_id={args.model_id} | input_root={args.input_root} | output_root={args.output_root}")
    if args.resize_w > 0 and args.resize_h > 0:
        print(f"[INFO] resize_to=({args.resize_w}, {args.resize_h})")
    print(f"[INFO] scenes={scene_names}")
    if args.train_only and args.task != "thinking":
        print("[INFO] Will ONLY process each scene's train/ subfolder.")

    # Process scenes sequentially
    for scene in scene_names:
        if args.task == "thinking":
            tasks = build_thinking_tasks(args.input_root, scene, args.start_idx, args.end_idx)
        else:
            tasks = build_image_tasks(args.input_root, scene, args.prompt,
                                      args.train_only, args.start_idx, args.end_idx)
        run_one_scene(scene, tasks, args)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
