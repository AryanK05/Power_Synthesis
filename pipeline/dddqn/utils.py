"""DDDQN training helpers."""
from __future__ import annotations

import numpy as np

from pipeline.config import RECIPE_LEN_VALID


def sample_episode_length(mode: str, rng: np.random.Generator) -> int:
    """Pick L per `DDDQN_TRAIN_L_MODE`.

    Modes:
      - "fixed_20"      : L = 20 always (interim, length-20-only surrogate)
      - "sample_6_20"   : L ~ Uniform{6..20}
      - "discrete_set"  : L ~ Uniform(RECIPE_LEN_VALID)  ← matches our data
    """
    if mode == "fixed_20":
        return 20
    if mode == "sample_6_20":
        return int(rng.integers(6, 21))
    if mode == "discrete_set":
        return int(rng.choice(RECIPE_LEN_VALID))
    raise ValueError(f"unknown DDDQN_TRAIN_L_MODE: {mode!r}")
