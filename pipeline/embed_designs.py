"""
Encode all .aig files to 256-d DeepGate2 embeddings, cached to disk.

Walks all DATA_SOURCES; first .aig found per design name is used (priority
order from config.DATA_SOURCES). Skips designs whose name is in SKIP_DESIGNS
or whose AIG > AIG_SIZE_CAP.

Smart cache:
  - existing cache file is read; designs already in it are reused
  - new designs are encoded; cache file rewritten with the union
  - --force re-encodes everything from scratch

Run:
    python -m pipeline.embed_designs
    python -m pipeline.embed_designs --force
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.config import EMBED_PATH, AIG_SIZE_CAP, SKIP_DESIGNS
from pipeline.encoders import build_encoder
from pipeline.data import list_all_designs, resolve_aig_path


def _eligible_designs():
    return [d for d in list_all_designs() if d not in SKIP_DESIGNS]


def main(device=None, force=False):
    embeds, meta = {}, None

    if EMBED_PATH.exists() and not force:
        blob = torch.load(EMBED_PATH, map_location="cpu", weights_only=False)
        embeds, meta = blob["embeds"], blob["meta"]
    elif force and EMBED_PATH.exists():
        print("[embed] --force: re-encoding everything")

    designs = _eligible_designs()
    missing = designs if force else [d for d in designs if d not in embeds]

    if not missing:
        print(f"[embed] cache hit: {len(embeds)} designs "
              f"(encoder={meta['encoder']}, dim={meta['out_dim']})")
        print(f"[embed] skipping. Delete {EMBED_PATH} or pass --force to re-encode.")
        return

    print(f"[embed] {len(embeds)}/{len(designs)} cached; encoding {len(missing)} more")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = build_encoder(device=device)

    meta = {"encoder": enc.name, "out_dim": enc.out_dim}

    for name in missing:
        aig = resolve_aig_path(name)
        if aig is None:
            print(f"[embed] {name}: no .aig file in any source, skipping")
            continue
        size = os.path.getsize(aig)
        if size > AIG_SIZE_CAP:
            print(f"[embed] {name}: {size:,} > cap ({AIG_SIZE_CAP:,}), skipping")
            continue
        try:
            tag = aig.parent.parent.name
            print(f"[embed] {name} ({size:,} B from {tag})...", end="", flush=True)
            embeds[name] = enc.encode(aig)
            print(f" {tuple(embeds[name].shape)}")
            # Incremental save so Ctrl+C doesn't lose progress
            torch.save({"meta": meta, "embeds": embeds}, EMBED_PATH)
        except KeyboardInterrupt:
            torch.save({"meta": meta, "embeds": embeds}, EMBED_PATH)
            print(f"\n[embed] interrupted; saved {len(embeds)} embeddings -> {EMBED_PATH}")
            raise
        except Exception as e:
            print(f" FAILED: {type(e).__name__}: {e}")
            continue

    print(f"[embed] saved {len(embeds)} embeddings -> {EMBED_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-encode all designs even if cache exists")
    args = ap.parse_args()
    main(force=args.force)
