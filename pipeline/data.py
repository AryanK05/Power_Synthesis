"""
Multi-source variable-length data loader.

Reads from all DATA_SOURCES. Each source has its own scripts/ and power/
folders; the same `sid` in different sources refers to a different recipe.

For embedding, one canonical .aig per design name (priority: data > 12-15 > 8-9).

For each (source, design, sid) sample:
  - recipe is read from <source>/scripts/script<sid>.txt (length 8-9 or 12-15
    or 20 depending on source)
  - power label is from <source>/power/<design>_power.csv row sid
  - recipe is padded to RECIPE_LEN_MAX with PAD_IDX
  - power is z-scored per (design, length)
"""
import os
import sys
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.config import (
    DATA_SOURCES, EMBED_PATH, NORMS_PATH, ABCSTATS_DIR,
    VOCAB, RECIPE_LEN_MAX, RECIPE_LEN_VALID, PAD_IDX,
)


# ---------------- discovery ----------------

def _list_designs_in_source(src_dir: Path):
    designs_dir = src_dir / "designs"
    if not designs_dir.exists():
        return []
    return sorted(p.stem for p in designs_dir.glob("*.aig"))


def list_all_designs():
    """Sorted list of unique design names across all sources."""
    designs = set()
    for src in DATA_SOURCES:
        designs.update(_list_designs_in_source(src["dir"]))
    return sorted(designs)


def resolve_aig_path(design):
    """Return the .aig file path for `design`. Priority order from DATA_SOURCES."""
    for src in DATA_SOURCES:
        p = src["dir"] / "designs" / f"{design}.aig"
        if p.exists():
            return p
    return None


# ---------------- recipe loading ----------------

def load_recipe_file(path):
    """Read script*.txt; return (op_id list, length) or (None, 0) on parse failure."""
    with open(path) as f:
        ops = [ln.strip() for ln in f if ln.strip()]
    try:
        ids = [VOCAB[op] for op in ops]
    except KeyError:
        return None, 0
    return ids, len(ids)


def load_recipes_for_source(src_dir: Path):
    """Return {sid: (op_id list, length)} for one source. Filter to valid lengths."""
    out = {}
    scripts_dir = src_dir / "scripts"
    if not scripts_dir.exists():
        return out
    for path in sorted(scripts_dir.glob("script*.txt")):
        try:
            sid = int(path.stem.replace("script", ""))
        except ValueError:
            continue
        ids, L = load_recipe_file(path)
        if ids is None or L not in RECIPE_LEN_VALID:
            continue
        out[sid] = (ids, L)
    return out


# ---------------- power & embeddings ----------------

def load_power_csv(src_dir: Path, design):
    p = src_dir / "power" / f"{design}_power.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_embeddings():
    blob = torch.load(EMBED_PATH, map_location="cpu", weights_only=False)
    return blob["embeds"], blob["meta"]


def load_abc_baselines(design):
    """Read abcStats_withmap/<design>.csv. Returns {alias: power} or {} if missing."""
    p = ABCSTATS_DIR / f"{design}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["alias"], df["power"]))


# ---------------- normalisation ----------------

def compute_design_length_norms(design_filter):
    """Per-(design, length) mean/std/min/max/n for both power AND area, across
    all sources. Returns dict {(design, length): {power: {...}, area: {...}}}.
    """
    rows = []
    for src in DATA_SOURCES:
        recipes = load_recipes_for_source(src["dir"])
        if not recipes:
            continue
        for design in _list_designs_in_source(src["dir"]):
            if design not in design_filter:
                continue
            df = load_power_csv(src["dir"], design)
            if df is None:
                continue
            for _, row in df.iterrows():
                sid = int(row["sid"])
                if sid not in recipes:
                    continue
                _, L = recipes[sid]
                rows.append((design, L, float(row["power"]), float(row["area"])))
    norms = {}
    if not rows:
        torch.save(norms, NORMS_PATH)
        return norms
    df = pd.DataFrame(rows, columns=["design", "length", "power", "area"])
    for (design, L), grp in df.groupby(["design", "length"]):
        def _stats(s):
            return {
                "mean": float(s.mean()),
                "std":  float(s.std() + 1e-6),
                "min":  float(s.min()),
                "max":  float(s.max()),
            }
        norms[(design, int(L))] = {
            "power": _stats(grp["power"]),
            "area":  _stats(grp["area"]),
            "n":     int(len(grp)),
            # back-compat top-level keys (old code reads norm["mean"] etc. for power)
            "mean":  float(grp["power"].mean()),
            "std":   float(grp["power"].std() + 1e-6),
            "min":   float(grp["power"].min()),
            "max":   float(grp["power"].max()),
        }
    torch.save(norms, NORMS_PATH)
    return norms


# ---------------- dataset ----------------

def _pad(ids, max_len, pad=PAD_IDX):
    out = [pad] * max_len
    out[:len(ids)] = ids
    return out


class PowerDataset(Dataset):
    """
    Each sample is a dict:
      {g, recipe (padded), length, power_z, power_raw, design, source, sid}

    Skips: design not in `design_filter`, design not in `embeds`, or no
    (design, length) entry in `norms`.
    """
    def __init__(self, design_filter, embeds, norms):
        self.samples = []
        ds_set = set(design_filter)
        for src in DATA_SOURCES:
            recipes = load_recipes_for_source(src["dir"])
            if not recipes:
                continue
            for design in _list_designs_in_source(src["dir"]):
                if design not in ds_set or design not in embeds:
                    continue
                df = load_power_csv(src["dir"], design)
                if df is None:
                    continue
                g = embeds[design]
                for _, row in df.iterrows():
                    sid = int(row["sid"])
                    if sid not in recipes:
                        continue
                    ids, L = recipes[sid]
                    norm = norms.get((design, L))
                    if norm is None:
                        continue
                    p = float(row["power"])
                    a = float(row["area"])
                    p_z = (p - norm["power"]["mean"]) / norm["power"]["std"]
                    a_z = (a - norm["area"]["mean"])  / norm["area"]["std"]
                    self.samples.append({
                        "g":         g,
                        "recipe":    torch.tensor(_pad(ids, RECIPE_LEN_MAX), dtype=torch.long),
                        "length":    int(L),
                        "power_z":   float(p_z),
                        "power_raw": p,
                        "area_z":    float(a_z),
                        "area_raw":  a,
                        "design":    design,
                        "source":    src["name"],
                        "sid":       sid,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    g         = torch.stack([b["g"] for b in batch])
    recipe    = torch.stack([b["recipe"] for b in batch])
    length    = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    power_z   = torch.tensor([b["power_z"]   for b in batch], dtype=torch.float).unsqueeze(-1)
    power_raw = torch.tensor([b["power_raw"] for b in batch], dtype=torch.float).unsqueeze(-1)
    area_z    = torch.tensor([b["area_z"]    for b in batch], dtype=torch.float).unsqueeze(-1)
    area_raw  = torch.tensor([b["area_raw"]  for b in batch], dtype=torch.float).unsqueeze(-1)
    designs   = [b["design"] for b in batch]
    sources   = [b["source"] for b in batch]
    sids      = [b["sid"]    for b in batch]
    return (g, recipe, length, power_z, power_raw, area_z, area_raw,
            designs, sources, sids)
