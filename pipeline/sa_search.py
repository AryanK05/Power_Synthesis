"""
Search algorithms over recipes:
  - simulated_annealing       (geometric cooling, 3 move types)
  - late_acceptance_hill_climbing  (LAHC; no temperature)

Both treat the surrogate as a frozen energy function. Recipe is a 1-D numpy
array of op-ids (length L; padded internally to RECIPE_LEN_MAX before scoring).
"""
import torch
import numpy as np

from pipeline.config import RECIPE_LEN_MAX, PAD_IDX


def random_recipe(rng, length, n_ops):
    return rng.integers(low=1, high=n_ops + 1, size=length, dtype=np.int64)


def _propose(rng, recipe, n_ops):
    r = recipe.copy()
    L = len(r)
    move = int(rng.integers(0, 3))
    if move == 0:                                  # replace one op
        i = int(rng.integers(0, L))
        new_op = int(rng.integers(1, n_ops + 1))
        while new_op == r[i] and n_ops > 1:
            new_op = int(rng.integers(1, n_ops + 1))
        r[i] = new_op
    elif move == 1:                                # swap two
        if L >= 2:
            i, j = rng.integers(0, L, size=2)
            r[i], r[j] = r[j], r[i]
    else:                                          # block-shift (size 2-4, length-preserving)
        Lb = int(rng.integers(2, min(5, L + 1)))
        if Lb >= L:
            return r
        i = int(rng.integers(0, L - Lb + 1))
        j = int(rng.integers(0, L - Lb + 1))
        block = r[i:i + Lb].copy()
        r = np.delete(r, range(i, i + Lb))
        r = np.insert(r, j, block)
    return r


def _score(surrogate, g, recipe_np, device):
    """Pad to RECIPE_LEN_MAX with PAD_IDX, query surrogate, return predicted z-power."""
    L = len(recipe_np)
    padded = np.full(RECIPE_LEN_MAX, PAD_IDX, dtype=np.int64)
    padded[:L] = recipe_np
    r = torch.as_tensor(padded, dtype=torch.long, device=device).unsqueeze(0)
    Lt = torch.tensor([L], dtype=torch.long, device=device)
    with torch.no_grad():
        return float(surrogate(g, r, Lt).item())


def simulated_annealing(
    surrogate, g_emb, n_ops, length,
    n_iter=20000, T0=1.0, T_end=0.01,
    init_recipe=None, device="cpu", seed=0,
):
    """Returns (best_recipe_np[L], best_energy_z, info)."""
    rng = np.random.default_rng(seed)
    surrogate.eval()
    g = g_emb.to(device).unsqueeze(0)

    cur = init_recipe.copy() if init_recipe is not None else random_recipe(rng, length, n_ops)
    cur_e = _score(surrogate, g, cur, device)
    best, best_e = cur.copy(), cur_e

    cooling = (T_end / T0) ** (1.0 / max(1, n_iter - 1))
    T = T0
    accepts = 0
    for _ in range(n_iter):
        cand = _propose(rng, cur, n_ops)
        cand_e = _score(surrogate, g, cand, device)
        d = cand_e - cur_e
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur, cur_e = cand, cand_e
            accepts += 1
            if cur_e < best_e:
                best, best_e = cur.copy(), cur_e
        T *= cooling
    return best, best_e, {"accept_rate": accepts / n_iter}


def late_acceptance_hill_climbing(
    surrogate, g_emb, n_ops, length,
    n_iter=20000, history_len=100,
    init_recipe=None, device="cpu", seed=0,
):
    """LAHC: accept if cand < cur OR cand <= history[i % history_len]. No temperature."""
    rng = np.random.default_rng(seed)
    surrogate.eval()
    g = g_emb.to(device).unsqueeze(0)

    cur = init_recipe.copy() if init_recipe is not None else random_recipe(rng, length, n_ops)
    cur_e = _score(surrogate, g, cur, device)
    best, best_e = cur.copy(), cur_e

    history = [cur_e] * history_len

    accepts = 0
    for i in range(n_iter):
        cand = _propose(rng, cur, n_ops)
        cand_e = _score(surrogate, g, cand, device)
        slot = i % history_len
        if cand_e < cur_e or cand_e <= history[slot]:
            cur, cur_e = cand, cand_e
            accepts += 1
            if cur_e < best_e:
                best, best_e = cur.copy(), cur_e
        history[slot] = cur_e
    return best, best_e, {"accept_rate": accepts / n_iter}
