#!/usr/bin/env python
"""Build (write) the FULL-INDEX cache for the multi-view datasets in ONE run.

The datasets now use a num_views/interval-AGNOSTIC cache (schema_version="full_list"):
the cache stores only the raw per-scene frame index (scenes / sceneids / images /
scene_img_list); the num_views-dependent start positions are derived at runtime by
BaseMultiViewDataset._compute_start_img_ids(). So a single cache works for ANY
num_views -- building at 8 or 16 produces the IDENTICAL cache file, and changing
num_views later needs NO rebuild. (There is therefore no --num_views flag here.)

This builds each cache from its TRAINING gen config (configs/datasets/multi_view/
gen_*.py) -- the EXACT same class / cache_path the dataloader uses -- so the
on-disk cache matches what training loads. Building a dataset triggers its
__init__ -> _load_data, which writes the cache when it is missing (or rebuilds
with --force / when an old num_views-baked cache is detected as legacy).

RUN ON THE CLUSTER (needs the real data mounts / aoss + petreloss configured),
from the repo root, e.g.:
    python scripts/build_dataset_caches.py --force                         # rebuild all (sequential)
    python scripts/build_dataset_caches.py --force --jobs 7                 # all 7 in parallel (I/O-bound -> big speedup)
    python scripts/build_dataset_caches.py --datasets hypersim scannet --force
    python scripts/build_dataset_caches.py --cache-dir /path/to/caches --force --jobs 7   # -> <name>_cache.pkl

NOTE on a running job: an OLD-code process crashes if it loads a NEW-format cache
(KeyError 'start_img_ids'). So do NOT overwrite a running job's cache_path; build
new caches to a fresh --cache-dir and point only NEW experiments at them.
"""
import argparse
import os
import os.path as osp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from mmengine.config import Config
from xtuner.registry import BUILDER

# dataset name -> the TRAINING gen config (same args the dataloader uses).
# These are exactly the 7 multi-view (image2image) sources in concat_datasets_8
# (puffin15m / cam2image is single-view and has no such index cache).
GEN_CONFIGS = {
    'dl3dv':       'configs/datasets/multi_view/gen_dl3dv.py',
    're10k':       'configs/datasets/multi_view/gen_realestate.py',
    'hypersim':    'configs/datasets/multi_view/gen_hypersim.py',
    'mvs_synth':   'configs/datasets/multi_view/gen_mvs_synth.py',
    'tartanair':   'configs/datasets/multi_view/gen_tartanair.py',
    'scannet':     'configs/datasets/multi_view/gen_scannet.py',
    'puffin_omni': 'configs/datasets/multi_view/gen_puffin_omni.py',
}


def build_one(name, cfg_path, force=False, cache_dir=None):
    print(f"\n{'=' * 72}\n[{name}] from {cfg_path}")
    cfg = Config.fromfile(cfg_path)
    ds_cfg = cfg.dataset                       # the dict(type=..., cache_path=..., num_views=...)

    if cache_dir is not None:
        # explicit dir + canonical name "<dataset>_cache.pkl" (overrides gen config path)
        ds_cfg['cache_path'] = osp.join(cache_dir, f"{name}_cache.pkl")

    cache_path = ds_cfg.get('cache_path', None)
    print(f"   cache_path={cache_path}   (full-index, num_views-agnostic)")

    if cache_path and osp.exists(cache_path):
        if force:
            os.remove(cache_path)
            print("   removed existing cache (--force) -> will rebuild")
        else:
            print("   cache already exists -> will LOAD (pass --force to rebuild)")

    if cache_path:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)

    ds = BUILDER.build(ds_cfg)                 # __init__ -> _load_data -> writes cache if missing
    n_scenes = len(getattr(ds, 'scenes', []) or [])
    n_images = getattr(ds, 'get_image_num', lambda: -1)()
    print(f"   OK: scenes={n_scenes}  images={n_images}  cache={cache_path}")
    return ('OK', n_scenes, n_images, cache_path)


def _safe_build(name, force, cache_dir):
    """build_one wrapped so a single dataset failing never aborts the others.
    Top-level (picklable) so it can run inside a ProcessPoolExecutor worker."""
    try:
        return build_one(name, GEN_CONFIGS[name], force=force, cache_dir=cache_dir)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"   FAILED: {name}: {e}")
        return ('FAILED', -1, -1, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=list(GEN_CONFIGS),
                    choices=list(GEN_CONFIGS),
                    help='which datasets to build (default: all 7)')
    ap.add_argument('--cache-dir', type=str, default=None,
                    help='save all caches into this dir as "<dataset>_cache.pkl" '
                         '(overrides each gen config cache_path)')
    ap.add_argument('--force', action='store_true',
                    help='delete an existing cache first to force a rebuild')
    ap.add_argument('--jobs', '-j', type=int, default=1,
                    help='build this many datasets IN PARALLEL (one process each). '
                         'The work is AOSS-I/O-bound, so concurrency overlaps the waits and '
                         'total time drops toward the slowest single dataset. Each job holds '
                         'one dataset index in RAM, so lower this if you hit host-RAM limits. '
                         'Use up to the number of --datasets (default 1 = sequential).')
    args = ap.parse_args()

    results = {}
    jobs = min(max(args.jobs, 1), len(args.datasets))

    # Aggregate progress bar counted BY SCENES (total unknown until each dataset
    # finishes scanning, so this is a running scene counter; it advances by a
    # dataset's scene count when that dataset completes). Each dataset class also
    # shows its own live per-scene "Indexing subscenes" bar during the scan.
    n_done = [0]
    pbar = tqdm(total=None, unit=' scene', desc='cache build', dynamic_ncols=True)

    def _record(name, res):
        results[name] = res
        n_done[0] += 1
        if res[0] == 'OK' and res[1] > 0:
            pbar.update(res[1])                       # res[1] = n_scenes for this dataset
        pbar.set_postfix_str(f"{n_done[0]}/{len(args.datasets)} datasets | {name}: {res[1]} scenes")

    if jobs > 1:
        tqdm.write(f"Building {len(args.datasets)} datasets with {jobs} parallel processes...")
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_safe_build, name, args.force, args.cache_dir): name
                    for name in args.datasets}
            for fut in as_completed(futs):
                _record(futs[fut], fut.result())
    else:
        for name in args.datasets:
            _record(name, _safe_build(name, args.force, args.cache_dir))
    pbar.close()

    print(f"\n{'=' * 72}\nSUMMARY")
    for name in args.datasets:
        status, n_scenes, n_images, cp = results[name]
        print(f"  {name:12s} {status:7s} scenes={n_scenes:<8} images={n_images:<10} {cp or ''}")
    failed = [n for n in args.datasets if results[n][0] != 'OK']
    if failed:
        print(f"\n{len(failed)} FAILED: {failed}")
        raise SystemExit(1)
    print("\nAll caches built OK.")


if __name__ == '__main__':
    main()
