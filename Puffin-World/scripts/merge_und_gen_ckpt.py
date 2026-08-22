#!/usr/bin/env python
"""Merge two checkpoints into ONE that keeps BOTH capabilities:

  * understanding  -> frozen backbone: LLM + visual_encoder + projector
  * generation     -> diffusion transformer + connectors + cond_fuser + ...

Why this is needed
------------------
Later training stages (e.g. final_stage_3 / final_stage_4) FREEZE the
understanding backbone and only train the diffusion transformer. Their saved
checkpoints (merged model_*.pth or DeepSpeed iter_*.pth) therefore contain ONLY
the transformer and DROP the frozen backbone. Loading such a checkpoint alone
leaves the LLM/vision/projector at the base (un-tuned) weights, so understanding
(e.g. camera-parameter estimation) is broken even though the weights were never
meant to change.

This script rebuilds a complete model: the understanding backbone is taken from
`--und_ckpt` (a full checkpoint that still has it, e.g. final_stage_2_base
model.pth) and everything else (the trained generation weights) is taken from
`--gen_ckpt` (e.g. final_stage_4 iter_6300.pth). The result has good
understanding AND the latest generation transformer.

Both inputs may be a plain `.pth` file OR a DeepSpeed zero checkpoint dir
(iter_*.pth) -- `guess_load_checkpoint` handles either.

Example
-------
  python scripts/merge_und_gen_ckpt.py \
      --und_ckpt work_dirs/final_stage_2_base_qwen2_5_7b_radiov3H_sd3p5M/model.pth \
      --gen_ckpt work_dirs/final_stage_4_world_all_asym_attn_qwen2_5_7b_radiov3H_sd3p5M/iter_6300.pth \
      --output   work_dirs/.../merged_und_gen.pth
"""
import argparse
import os

import torch

# PyTorch >= 2.6 defaults torch.load(weights_only=True); DeepSpeed optimizer
# states pickle non-tensor globals, so force weights_only=False (same trick as
# scripts/deepspeed2torch.py).
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from xtuner.model.utils import guess_load_checkpoint  # noqa: E402

# Key prefixes that make up the understanding backbone (kept from --und_ckpt).
DEFAULT_BACKBONE_PREFIXES = "llm.,visual_encoder.,projector."


def to_state_dict(obj):
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return obj


def load_ckpt(path):
    """Load a plain .pth or a DeepSpeed zero checkpoint dir into a flat dict."""
    return to_state_dict(guess_load_checkpoint(path))


def summarize(sd):
    from collections import defaultdict
    groups = defaultdict(lambda: [0, 0])
    for k, v in sd.items():
        top = k.split(".")[0]
        groups[top][0] += 1
        groups[top][1] += (v.numel() if hasattr(v, "numel") else 0)
    return groups


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--und_ckpt", required=True,
                    help="full checkpoint that still holds the understanding "
                         "backbone (LLM/vision/projector), e.g. stage_2 model.pth")
    ap.add_argument("--gen_ckpt", required=True,
                    help="checkpoint with the trained generation weights "
                         "(transformer, ...), e.g. stage_4 iter_*.pth")
    ap.add_argument("--output", required=True, help="output merged .pth path")
    ap.add_argument("--backbone_prefixes", default=DEFAULT_BACKBONE_PREFIXES,
                    help="comma-separated key prefixes kept from --und_ckpt "
                         "(default: %(default)s)")
    ap.add_argument("--verify", action="store_true",
                    help="verify every backbone tensor in the merge equals the "
                         "one in --und_ckpt (slower)")
    args = ap.parse_args()

    prefixes = tuple(p for p in args.backbone_prefixes.split(",") if p)

    print(f"[merge] loading und_ckpt (backbone source): {args.und_ckpt}", flush=True)
    und = load_ckpt(args.und_ckpt)
    print(f"[merge] loading gen_ckpt (generation source): {args.gen_ckpt}", flush=True)
    gen = load_ckpt(args.gen_ckpt)

    # Base = und (the complete model). Overlay gen's NON-backbone keys so the
    # understanding backbone stays from und and the generation weights come from
    # gen. gen-only keys (new modules trained in the later stage) are added.
    merged = dict(und)
    overlaid = added = skipped = 0
    for k, v in gen.items():
        if k.startswith(prefixes):
            skipped += 1          # gen's (stale/empty) backbone copy -> keep und's
            continue
        if k not in merged:
            added += 1
        merged[k] = v
        overlaid += 1

    def nparams(sd, pred):
        return sum(v.numel() for k, v in sd.items() if pred(k) and hasattr(v, "numel"))
    bb_params = nparams(merged, lambda k: k.startswith(prefixes))
    gen_params = nparams(merged, lambda k: not k.startswith(prefixes))

    print(f"[merge] und keys={len(und)} gen keys={len(gen)} | "
          f"overlaid(non-backbone)={overlaid} (new={added}) "
          f"skipped(backbone-in-gen)={skipped}")
    print(f"[merge] merged keys={len(merged)} | "
          f"backbone (from und)={bb_params/1e9:.2f}B  generation (from gen)={gen_params/1e9:.2f}B")

    # Sanity checks: backbone came from und.
    bb_keys = [k for k in merged if k.startswith(prefixes)]
    if not bb_keys:
        raise SystemExit(f"[merge] ERROR: no backbone keys ({prefixes}) found in "
                         f"--und_ckpt; is it the right (full) checkpoint?")
    check_keys = bb_keys if args.verify else bb_keys[:1]
    for k in check_keys:
        same = (k in und) and torch.equal(merged[k], und[k])
        if not same:
            raise SystemExit(f"[merge] ERROR: backbone key {k} does not match und_ckpt.")
    print(f"[merge] backbone-source check passed on {len(check_keys)} key(s) "
          f"(e.g. {check_keys[0]}).")

    print("[merge] merged top-level groups:")
    for top, (c, n) in sorted(summarize(merged).items(), key=lambda x: -x[1][1]):
        print(f"    {top:<22} keys={c:<6d} params={n/1e9:.2f}B")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[merge] saving -> {args.output}", flush=True)
    torch.save(merged, args.output)
    print(f"[merge] DONE size={os.path.getsize(args.output)} bytes", flush=True)


if __name__ == "__main__":
    main()
