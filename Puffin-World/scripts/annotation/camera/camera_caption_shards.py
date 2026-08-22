"""Camera-caption annotation for SHARDED datasets: local archives or online parquet.

One script, two source kinds, switched by --type:

  --type tar | zip | tgz   LOCAL archive shards (webdataset-style). Images are
                           read straight from the archive WITHOUT extracting to
                           disk (tarfile/zipfile + BytesIO). tgz covers both
                           *.tgz and *.tar.gz.
  --type img               LOCAL loose image files: every image under
                           --data_root (recursively) is captioned directly.
  --type parquet           ONLINE dataset stored as parquet URL shards (e.g.
                           madebyollin/megalith-10m): rows hold image URLs,
                           images are downloaded on-the-fly inside DataLoader
                           workers and captioned in memory.

Each JSON: {"roll", "pitch", "vfov", "k1", "parse_ok"}  (radians; no text).

Output layout
-------------
Archive mode -- each archive becomes a directory (its archive suffix dropped)
mirroring the member's internal path, image extension replaced by .json:

    <data_root>/train/shard_0001.tar :: images/0042.jpg
        -> <camera_root>/train/shard_0001/images/0042.json

and, with --pack (default), the shard's JSONs are bundled into
<camera_root>/train/shard_0001.tar (packed output is ALWAYS a .tar, whatever
the source archive format).

Img mode -- one JSON per image, mirroring the image's path relative to
--data_root (image extension replaced by .json):

    <data_root>/scene/0042.jpg -> <camera_root>/scene/0042.json

No packing in this mode (--pack is ignored): the loose JSONs stay addressable
by their mirrored relative path. The sorted image list is split into
--chunk_size groups that act as the shards for rank-sharding / progress /
--start_idx & --end_idx.

Parquet mode -- each parquet file is split into fixed-size row CHUNKS
(--chunk_size rows); a chunk plays the role of one shard and packs into

    <camera_root>/<parquet_stem>/chunk_<cstart>-<cend>.tar

The JSON is named after the image id derived from the canonical `url` column
stem, so the id is stable regardless of which URL column (--url_column) is
downloaded.

Sharding / resume
-----------------
Shards (archives or chunks) are sharded across ranks
(process_index :: num_processes), so the job scales over GPUs/nodes with
torchrun/accelerate. --start_idx/--end_idx restrict the run to a contiguous
index range of the globally-sorted shard list (for splitting across jobs).
Runs are resumable: a shard is skipped when its packed .tar already exists,
and within a shard any image whose JSON already exists is skipped.

Examples:
    # 8-GPU captioning of all local tar shards
    torchrun --nproc_per_node=8 scripts/annotation/camera/camera_caption_shards.py \\
        configs/pipelines/<vlm_or_full>.py --checkpoint <ckpt> \\
        --type tar \\
        --data_root /data/.../my_dataset \\
        --camera_root /data/.../my_dataset_camera

    # 8-GPU captioning of loose local images
    torchrun --nproc_per_node=8 scripts/annotation/camera/camera_caption_shards.py \\
        configs/pipelines/<vlm_or_full>.py --checkpoint <ckpt> \\
        --type img \\
        --data_root /data/.../my_images \\
        --camera_root /data/.../my_images_camera

    # 8-GPU captioning of an online parquet URL dataset
    torchrun --nproc_per_node=8 scripts/annotation/camera/camera_caption_shards.py \\
        configs/pipelines/<full>.py --checkpoint <ckpt> \\
        --type parquet \\
        --data_root /data/.../megalith-10m/data \\
        --camera_root /data/.../camera/megalith-10m \\
        --batch_size 128 --num_workers 32
"""
import argparse
import io
import json
import math
import os
import os.path as osp
import shutil
import tarfile
import zipfile
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator

from scripts.camera.utils.text import parse_camera_params


PROMPT = (
    "Describe the image in detail. Then reason its spatial distribution "
    "and estimate its camera parameters (roll, pitch, field-of-view, and "
    "radial distortion)."
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

USER_AGENT = "Mozilla/5.0 (compatible; PuffinCameraAnnotator/1.0)"

# --type value -> filename suffixes scanned under --data_root
ARCHIVE_TYPES = {
    "tar": (".tar",),
    "zip": (".zip",),
    "tgz": (".tgz", ".tar.gz"),
}


# ---------------------------------------------------------------------------
# Preprocessing (identical to camera_caption_aoss.py / understanding.py)
# ---------------------------------------------------------------------------
def pad_square_tensor(image, pad_value=0):
    h, w = image.shape[-2:]
    if h == w:
        return image
    if h > w:
        pad_left = (h - w) // 2
        p2d = (pad_left, h - w - pad_left, 0, 0)
    else:
        pad_top = (w - h) // 2
        p2d = (0, 0, pad_top, w - h - pad_top)
    return F.pad(image, p2d, "constant", pad_value)


def process_for_model(image, image_size):
    """Resize longest edge to image_size, normalize to [-1, 1], pad square."""
    w, h = image.size
    if w >= h:
        new_w, new_h = image_size, int(h * (image_size / w))
    else:
        new_h, new_w = image_size, int(w * (image_size / h))
    image = image.resize(size=(new_w, new_h))
    pv = torch.from_numpy(np.array(image)).float() / 255.0
    pv = 2.0 * pv - 1.0
    pv = rearrange(pv, 'h w c -> c h w')
    return pad_square_tensor(pv, pad_value=0)


# ---------------------------------------------------------------------------
# Archive abstraction: uniform read-only view over tar / zip shards
# ---------------------------------------------------------------------------
class ArchiveReader:
    """Read-only access to one archive shard (tar[.gz] via tarfile, zip via
    zipfile) behind a single interface: image_members() and read(member)."""

    def __init__(self, path):
        self.path = path
        if path.lower().endswith(".zip"):
            self._zip = zipfile.ZipFile(path, "r")
            self._tar = None
        else:
            # "r:*" transparently handles plain tar and compressed tar.gz/tgz
            self._tar = tarfile.open(path, "r:*")
            self._zip = None

    def image_members(self):
        """Names of all regular-file members with an image extension."""
        if self._tar is not None:
            return [m.name for m in self._tar.getmembers()
                    if m.isfile() and m.name.lower().endswith(IMAGE_EXTS)]
        return [i.filename for i in self._zip.infolist()
                if not i.is_dir() and i.filename.lower().endswith(IMAGE_EXTS)]

    def read(self, member):
        """Raw bytes of one member."""
        if self._tar is not None:
            return self._tar.extractfile(member).read()
        return self._zip.read(member)

    def close(self):
        if self._tar is not None:
            self._tar.close()
        if self._zip is not None:
            self._zip.close()


def strip_archive_ext(path):
    """Drop the archive suffix (longest match first: .tar.gz > .tgz/.tar/.zip)."""
    low = path.lower()
    for ext in (".tar.gz", ".tgz", ".tar", ".zip"):
        if low.endswith(ext):
            return path[:-len(ext)]
    return path


def list_archive_shards(data_root, exts):
    """All archive shards with the given suffixes under data_root, sorted."""
    base = Path(data_root)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {base}")
    shards = sorted(str(p) for p in base.rglob("*")
                    if p.is_file() and str(p).lower().endswith(exts))
    if not shards:
        raise FileNotFoundError(f"No {'/'.join(exts)} shards found under {base}")
    return shards


def caption_dir_for(camera_root, data_root, shard_path):
    """Loose-JSON output dir for a shard: <camera_root>/<rel shard w/o ext>."""
    rel = osp.relpath(shard_path, data_root)
    return osp.join(camera_root, strip_archive_ext(rel))


def caption_tar_for(camera_root, data_root, shard_path):
    """Packed-output tar path (always .tar, whatever the source format):
    <camera_root>/<rel shard w/o ext>.tar"""
    return caption_dir_for(camera_root, data_root, shard_path) + ".tar"


def caption_path_for(camera_root, data_root, shard_path, member_name):
    """Mirror structure: <shard rel to root w/o ext>/<member path -> .json>."""
    member_stem = osp.splitext(member_name)[0]
    return osp.join(caption_dir_for(camera_root, data_root, shard_path),
                    member_stem + ".json")


def build_items_for_shard(shard_path, data_root, camera_root, overwrite):
    """Scan one archive's members (no image bytes read), filter images, apply
    the resume skip, and return the flat (shard, member, out_path) work list."""
    items = []
    try:
        ar = ArchiveReader(shard_path)
        try:
            for name in ar.image_members():
                out_path = caption_path_for(camera_root, data_root,
                                            shard_path, name)
                if not overwrite and osp.exists(out_path):
                    continue
                items.append((shard_path, name, out_path))
        finally:
            ar.close()
    except Exception as e:
        print(f"[caption] cannot open archive {shard_path}: {e}", flush=True)
    return items


class ArchiveCaptionDataset(torch.utils.data.Dataset):
    """Yields one preprocessed image per (shard, member). Each worker keeps a
    single open archive handle (lazy, post-fork) so consecutive members of the
    same shard don't re-read the archive index."""

    def __init__(self, items, image_size):
        # items: list of (shard_path, member_name, out_json_path)
        self.items = items
        self.image_size = image_size
        self._cur_path = None
        self._cur_ar = None

    def __len__(self):
        return len(self.items)

    def _archive(self, shard_path):
        if shard_path != self._cur_path:
            if self._cur_ar is not None:
                self._cur_ar.close()
            self._cur_ar = ArchiveReader(shard_path)
            self._cur_path = shard_path
        return self._cur_ar

    def __getitem__(self, idx):
        shard_path, member, out_path = self.items[idx]
        try:
            data = self._archive(shard_path).read(member)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            pv = process_for_model(image, self.image_size)
            return dict(out_path=out_path, pixel_values=pv, ok=True)
        except Exception as e:
            print(f"[caption] failed {shard_path}::{member}: {e}", flush=True)
            return dict(out_path=out_path,
                        pixel_values=torch.zeros(3, self.image_size, self.image_size),
                        ok=False)


# ---------------------------------------------------------------------------
# Loose local images (--type img)
# ---------------------------------------------------------------------------
def list_images(data_root):
    """All loose image files under data_root, recursively, sorted."""
    base = Path(data_root)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {base}")
    imgs = sorted(str(p) for p in base.rglob("*")
                  if p.is_file() and str(p).lower().endswith(IMAGE_EXTS))
    if not imgs:
        raise FileNotFoundError(f"No images found under {base}")
    return imgs


def img_json_for(camera_root, data_root, img_path):
    """Mirror structure: <camera_root>/<image rel path with ext -> .json>."""
    rel = osp.relpath(img_path, data_root)
    return osp.join(camera_root, osp.splitext(rel)[0] + ".json")


def build_items_for_images(img_paths, data_root, camera_root, overwrite):
    """Return the flat (image_path, out_json_path) work list for a group of
    images, applying the per-image resume skip."""
    items = []
    for p in img_paths:
        out_path = img_json_for(camera_root, data_root, p)
        if not overwrite and osp.exists(out_path):
            continue
        items.append((p, out_path))
    return items


class ImgCaptionDataset(torch.utils.data.Dataset):
    """Yields one preprocessed image per local image file."""

    def __init__(self, items, image_size):
        # items: list of (image_path, out_json_path)
        self.items = items
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, out_path = self.items[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            pv = process_for_model(image, self.image_size)
            return dict(out_path=out_path, pixel_values=pv, ok=True)
        except Exception as e:
            print(f"[caption] failed {img_path}: {e}", flush=True)
            return dict(out_path=out_path,
                        pixel_values=torch.zeros(3, self.image_size, self.image_size),
                        ok=False)


# ---------------------------------------------------------------------------
# Parquet abstraction: row chunks of URL shards
# ---------------------------------------------------------------------------
def list_parquets(data_root):
    """All *.parquet under data_root, recursively, sorted."""
    pqs = sorted(glob(osp.join(data_root, "**", "*.parquet"), recursive=True))
    if not pqs:
        raise FileNotFoundError(f"No .parquet files found under {data_root}")
    return pqs


def list_chunks(data_root, chunk_size):
    """Build the global, ordered list of chunks across all parquet files.

    Each chunk is a dict: {parquet, stem, cstart, cend (exclusive)}."""
    import pyarrow.parquet as pq
    chunks = []
    for p in list_parquets(data_root):
        nrows = pq.ParquetFile(p).metadata.num_rows
        stem = osp.splitext(osp.basename(p))[0]
        for cs in range(0, nrows, chunk_size):
            ce = min(cs + chunk_size, nrows)
            chunks.append(dict(parquet=p, stem=stem, cstart=cs, cend=ce))
    return chunks


def chunk_dir_for(camera_root, ch):
    """Loose-JSON output dir for a chunk."""
    return osp.join(camera_root, ch["stem"],
                    f"chunk_{ch['cstart']:09d}-{ch['cend']:09d}")


def chunk_tar_for(camera_root, ch):
    """Packed-output tar path for a chunk (mirrors the loose dir name)."""
    return chunk_dir_for(camera_root, ch) + ".tar"


def url_to_id(url):
    """Canonical image id = basename stem of the `url` column value."""
    return osp.splitext(osp.basename(url))[0]


# Tiny per-process cache: keep the most-recently-read parquet's columns so the
# ~chunks-per-file consecutive chunks owned by a rank don't re-read the file.
_PARQUET_CACHE = {"path": None, "cols": None}


def _read_url_columns(parquet_path, url_column):
    import pyarrow.parquet as pq
    if _PARQUET_CACHE["path"] != parquet_path:
        cols = ["url"] if url_column == "url" else ["url", url_column]
        tbl = pq.read_table(parquet_path, columns=cols)
        _PARQUET_CACHE["path"] = parquet_path
        _PARQUET_CACHE["cols"] = {
            "url": tbl.column("url").to_pylist(),
            "dl": tbl.column(url_column).to_pylist(),
        }
    return _PARQUET_CACHE["cols"]


def build_items_for_chunk(ch, camera_root, url_column, overwrite):
    """Return the flat (download_url, out_json_path) work list for a chunk,
    applying the per-image resume skip."""
    cols = _read_url_columns(ch["parquet"], url_column)
    base_urls = cols["url"][ch["cstart"]:ch["cend"]]
    dl_urls = cols["dl"][ch["cstart"]:ch["cend"]]
    loose_dir = chunk_dir_for(camera_root, ch)
    items = []
    for base_url, dl_url in zip(base_urls, dl_urls):
        if not base_url or not dl_url:
            continue
        out_path = osp.join(loose_dir, url_to_id(base_url) + ".json")
        if not overwrite and osp.exists(out_path):
            continue
        items.append((dl_url, out_path))
    return items


class UrlCaptionDataset(torch.utils.data.Dataset):
    """Downloads one image per item from its URL and preprocesses it in memory.
    A requests.Session is created lazily per worker (post-fork)."""

    def __init__(self, items, image_size, timeout, retries):
        # items: list of (download_url, out_json_path)
        self.items = items
        self.image_size = image_size
        self.timeout = timeout
        self.retries = retries
        self._session = None

    def __len__(self):
        return len(self.items)

    def _sess(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT})
            self._session = s
        return self._session

    def __getitem__(self, idx):
        url, out_path = self.items[idx]
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                r = self._sess().get(url, timeout=self.timeout)
                r.raise_for_status()
                image = Image.open(io.BytesIO(r.content)).convert("RGB")
                pv = process_for_model(image, self.image_size)
                return dict(out_path=out_path, pixel_values=pv, ok=True)
            except Exception as e:  # network / decode / HTTP error
                last_err = e
        print(f"[caption] failed {url}: {last_err}", flush=True)
        return dict(out_path=out_path,
                    pixel_values=torch.zeros(3, self.image_size, self.image_size),
                    ok=False)


# ---------------------------------------------------------------------------
# Packing (shared): bundle a loose-JSON dir into one .tar, atomically
# ---------------------------------------------------------------------------
def pack_json_dir(loose_dir, out_tar, remove_loose=False):
    """Bundle all *.json under loose_dir (recursively; arcnames relative to the
    dir, so nested member paths are preserved) into out_tar. Written atomically
    (temp + rename). Optionally removes the loose dir. Returns True if packed."""
    if not osp.isdir(loose_dir):
        return False
    jsons = sorted(str(p) for p in Path(loose_dir).rglob("*.json"))
    if not jsons:
        return False
    os.makedirs(osp.dirname(out_tar) or ".", exist_ok=True)
    tmp_tar = out_tar + ".tmp"
    with tarfile.open(tmp_tar, "w") as tf:
        for jp in jsons:
            tf.add(jp, arcname=osp.relpath(jp, loose_dir))
    os.replace(tmp_tar, out_tar)
    if remove_loose:
        shutil.rmtree(loose_dir, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Captioning (shared)
# ---------------------------------------------------------------------------
def caption_items(model, ds, args):
    """Caption a prepared dataset of items, writing loose JSONs.
    Returns (n_done, n_fail)."""
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, collate_fn=lambda x: x,
    )
    n_done, n_fail = 0, 0
    for batch in loader:
        good = [b for b in batch if b["ok"]]
        if not good:
            n_fail += len(batch)
            continue
        pixel_values = [b["pixel_values"] for b in good]
        with torch.no_grad():
            texts = model.understand(
                prompt=[PROMPT] * len(good),
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                progress_bar=False,
            )
        for b, text in zip(good, texts):
            try:
                roll, pitch, vfov, k1 = parse_camera_params(text, mode='radial')
                parse_ok = True
            except ValueError:
                roll, pitch, vfov, k1 = 0.0, 0.0, math.radians(90.0), 0.0
                parse_ok = False
            # Range sanity (radians): reject stray numbers (years, etc.).
            if parse_ok and not (
                abs(roll) <= math.pi / 2 + 1e-3
                and abs(pitch) <= math.pi / 2 + 1e-3
                and 0.0 < vfov < math.pi
                and abs(k1) <= 2.0
            ):
                parse_ok = False
            os.makedirs(osp.dirname(b["out_path"]), exist_ok=True)
            with open(b["out_path"], "w", encoding="utf-8") as f:
                json.dump({
                    "roll": float(roll),
                    "pitch": float(pitch),
                    "vfov": float(vfov),
                    "k1": float(k1),
                    "parse_ok": parse_ok,
                }, f, ensure_ascii=False)
            n_done += 1
        n_fail += len(batch) - len(good)
    return n_done, n_fail


def run_captioning(args, accelerator):
    # ---- per-type source hooks: list units + build items + locate outputs ----
    if args.type == "parquet":
        units = list_chunks(args.data_root, args.chunk_size)
        unit_word = "chunk"
        extra_info = f"chunk_size={args.chunk_size}, url_column={args.url_column}, "

        def packed_path(u):
            return chunk_tar_for(args.camera_root, u)

        def loose_dir(u):
            return chunk_dir_for(args.camera_root, u)

        def build_items(u):
            return build_items_for_chunk(u, args.camera_root,
                                         args.url_column, args.overwrite)

        def make_dataset(items):
            return UrlCaptionDataset(items, args.image_size,
                                     args.download_timeout, args.download_retries)
    elif args.type == "img":
        # Loose images: chunk the sorted list so the unit loop provides rank
        # sharding, range slicing, and the progress bar. No packing (the loose
        # JSONs must stay addressable by their mirrored relative path).
        images = list_images(args.data_root)
        units = [dict(cstart=cs, cend=min(cs + args.chunk_size, len(images)))
                 for cs in range(0, len(images), args.chunk_size)]
        unit_word = "chunk"
        extra_info = f"images={len(images)}, chunk_size={args.chunk_size}, "

        def packed_path(u):
            return ""  # never used: pack is forced off for --type img

        def loose_dir(u):
            return ""  # never used: pack is forced off for --type img

        def build_items(u):
            return build_items_for_images(images[u["cstart"]:u["cend"]],
                                          args.data_root, args.camera_root,
                                          args.overwrite)

        def make_dataset(items):
            return ImgCaptionDataset(items, args.image_size)
    else:
        units = list_archive_shards(args.data_root, ARCHIVE_TYPES[args.type])
        unit_word = "shard"
        extra_info = ""

        def packed_path(u):
            return caption_tar_for(args.camera_root, args.data_root, u)

        def loose_dir(u):
            return caption_dir_for(args.camera_root, args.data_root, u)

        def build_items(u):
            return build_items_for_shard(u, args.data_root,
                                         args.camera_root, args.overwrite)

        def make_dataset(items):
            return ArchiveCaptionDataset(items, args.image_size)

    # Restrict to a unit-index range [start_idx, end_idx] (inclusive, into the
    # globally-sorted unit list). end_idx == -1 means "until the end". This lets
    # you split the dataset across separate jobs, e.g. 0-2000, 2001-4000, ...
    n_total = len(units)
    start_idx = max(0, args.start_idx)
    end_excl = n_total if args.end_idx < 0 else min(args.end_idx + 1, n_total)
    units = units[start_idx:end_excl]
    accelerator.print(
        f"[caption] {n_total} {args.type} {unit_word}s total "
        f"(root={args.data_root}; {extra_info}pack={args.pack}); "
        f"range [{start_idx}, {end_excl - 1}] -> {len(units)} {unit_word}s this run")

    # Shard units across ranks (covers multi-node via global process_index).
    my_units = units[accelerator.process_index::accelerator.num_processes]
    print(f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
          f"{len(my_units)} {unit_word}s", flush=True)
    if not my_units:
        return

    # ---- model ----
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device).to(model.dtype)
    model.eval()

    # ---- process one unit at a time, then pack it ----
    # Progress bar is unit-based and shown on the global rank-0 only. Each rank
    # owns a near-equal slice of the range, so rank-0's bar tracks the run well.
    n_done, n_fail, n_packed = 0, 0, 0
    for unit in tqdm(my_units, desc=f"{unit_word}s (rank0)", unit=unit_word,
                     disable=not accelerator.is_main_process):
        # Whole-unit resume: skip if this unit's packed output already exists.
        if args.pack and not args.overwrite and osp.exists(packed_path(unit)):
            continue

        # Caption only the images that still lack a loose JSON.
        items = build_items(unit)
        if items:
            d, f = caption_items(model, make_dataset(items), args)
            n_done += d
            n_fail += f

        # Pack this unit's JSONs (gathers all loose JSONs, incl. resumed ones).
        if args.pack:
            if pack_json_dir(loose_dir(unit), packed_path(unit),
                             remove_loose=args.remove_loose):
                n_packed += 1

    print(f"[rank {accelerator.process_index}] wrote {n_done} captions, "
          f"{n_fail} failures, packed {n_packed} {unit_word}s", flush=True)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='VLM / pipeline config (provides model.understand).')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--type', default='tar', type=str,
                        choices=sorted(ARCHIVE_TYPES) + ['img', 'parquet'],
                        help='Source kind: local archive shards (tar / zip / '
                             'tgz, read without extraction), loose local '
                             'images (img), or online parquet URL shards '
                             '(images streamed from their URLs).')
    parser.add_argument('--data_root', required=True, type=str,
                        help='Local dataset directory; all shards of --type '
                             'under it are captioned (searched recursively).')
    parser.add_argument('--camera_root', required=True, type=str,
                        help='Output root for caption JSONs / packed tars.')
    parser.add_argument('--start_idx', default=0, type=int,
                        help='First shard/chunk index (inclusive) into the '
                             'sorted list to process this run (default 0).')
    parser.add_argument('--end_idx', default=-1, type=int,
                        help='Last shard/chunk index (INCLUSIVE) to process; '
                             '-1 means until the end. E.g. 0/2000 then '
                             '2001/4000 splits the work into contiguous chunks.')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=None, type=int,
                        help='DataLoader workers. Default: 4 for archives, 16 '
                             'for parquet (workers also drive the downloads; '
                             'network is the bottleneck -- use 32-64).')
    parser.add_argument('--image_size', default=640, type=int)
    parser.add_argument('--max_new_tokens', default=200, type=int)
    # parquet-only options
    parser.add_argument('--url_column', default='url_highres', type=str,
                        help='[parquet] Column to download (default url_highres; '
                             'use "url" for the smaller medium-res image).')
    parser.add_argument('--chunk_size', default=10000, type=int,
                        help='[parquet/img] Rows (parquet) or images (img) per '
                             'chunk; for parquet one chunk == one packed tar.')
    parser.add_argument('--download_timeout', default=15, type=float,
                        help='[parquet] Per-image HTTP timeout in seconds.')
    parser.add_argument('--download_retries', default=2, type=int,
                        help='[parquet] Retries per image on download/decode '
                             'failure.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-caption even if the JSON already exists.')
    parser.add_argument('--pack', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='After captioning each shard/chunk, bundle its '
                             'JSONs into a .tar mirroring the source name/'
                             'structure (--pack / --no-pack). Default: pack. '
                             'Ignored for --type img (loose JSONs only).')
    parser.add_argument('--remove_loose', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='After packing, delete the loose per-image JSON dir '
                             'so only the .tar remains (--remove_loose / '
                             '--no-remove_loose; only used with --pack). '
                             'Default: remove. Note: with the loose dir gone, '
                             'resume is whole-shard only (skips a shard when '
                             'its packed .tar exists).')
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = 16 if args.type == "parquet" else 4
    if args.type == "img":
        args.pack = False  # loose JSONs mirror the image paths; no packing

    accelerator = Accelerator()
    accelerator.print(f"processes: {accelerator.num_processes}")
    run_captioning(args, accelerator)


if __name__ == "__main__":
    main()
