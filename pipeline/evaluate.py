"""
Evaluate on TEST_DESIGNS at lengths in EVAL_LENGTHS (default: only L=20).

Methods reported:
  oracle     — best of N known recipes at length L by ground truth
  DDDQN+SA   — DDDQN policy emits 5 recipes; SA refinement from each
  resyn      — ABC built-in (from abcStats_withmap/<design>.csv)
  resyn2     — ABC built-in

`true_power` for DDDQN+SA is computed by ABC (Nangate45 + map). Other rows
are CSV lookups.

Output:
  results/<design>.csv  with columns: length, method, recipe, pred_power, true_power
"""
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.config import (
    TEST_DESIGNS, SEED, RECIPE_LEN_MAX, EVAL_LENGTHS, N_OPS,
    SA_ITERS, SA_T0, SA_T_END, SA_RESTARTS,
    NORMS_PATH, SURR_PATH, POLICY_DDDQN_PATH,
    ID_TO_OP, RESULTS_DIR, DATA_SOURCES,
)
from pipeline.data import (
    load_embeddings, load_power_csv, load_abc_baselines,
    load_recipes_for_source,
)
from pipeline.surrogate import QoRSurrogate
from pipeline.sa_search import simulated_annealing
from pipeline.abc_runner import abc_power


# ---------- helpers ----------

def decode_recipe(np_array):
    return [ID_TO_OP[int(x)] for x in np_array]


def _gather_known(test_designs):
    """{(design, length): list[(sid, source, ops, true_power)]}."""
    bucket = {}
    for src in DATA_SOURCES:
        recipes = load_recipes_for_source(src["dir"])
        for design in test_designs:
            df = load_power_csv(src["dir"], design)
            if df is None:
                continue
            for _, row in df.iterrows():
                sid = int(row["sid"])
                if sid not in recipes:
                    continue
                ids, L = recipes[sid]
                if L not in EVAL_LENGTHS:
                    continue
                bucket.setdefault((design, L), []).append(
                    (sid, src["name"], ids, float(row["power"]))
                )
    return bucket


def _best_oracle(items):
    return min(items, key=lambda x: x[3])


def _sample_dddqn(policy, g, length, n_recipes, n_ops, recipe_max_len, device, rng):
    out = []
    for _ in range(n_recipes):
        recipe = torch.zeros(recipe_max_len, dtype=torch.long, device=device)
        L_t = torch.tensor(length, dtype=torch.long, device=device)
        for t in range(length):
            t_t = torch.tensor(t, dtype=torch.long, device=device)
            with torch.no_grad():
                Q, _ = policy(
                    g.unsqueeze(0), recipe.unsqueeze(0),
                    t_t.unsqueeze(0), L_t.unsqueeze(0),
                )
            probs = torch.softmax(Q[0], dim=-1).cpu().numpy()
            a = int(rng.choice(n_ops, p=probs))
            recipe[t] = a + 1
        out.append(recipe.cpu().numpy()[:length])
    return out


def _reset_csv(design):
    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / f"{design}.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerow(["length", "method", "recipe", "pred_power", "true_power"])


def _write_row(design, length, method, recipe_strs, pred, true):
    p = RESULTS_DIR / f"{design}.csv"
    with open(p, "a", newline="") as f:
        csv.writer(f).writerow([
            length, method,
            ";".join(recipe_strs) if recipe_strs is not None else "",
            f"{pred:.4f}" if pred is not None else "",
            f"{true:.4f}" if (true == true) else "NaN",
        ])


# ---------- main ----------

def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeds, meta = load_embeddings()
    aig_dim = meta["out_dim"]
    norms = torch.load(NORMS_PATH, map_location="cpu", weights_only=False)

    surrogate = QoRSurrogate(aig_dim=aig_dim, n_ops=N_OPS).to(device)
    surrogate.load_state_dict(
        torch.load(SURR_PATH, map_location=device, weights_only=False)["model"]
    )
    surrogate.eval()

    if not os.path.exists(POLICY_DDDQN_PATH):
        sys.exit(f"[eval] no DDDQN policy at {POLICY_DDDQN_PATH}; train it first")
    from pipeline.dddqn.networks import DDDQNGenerator
    policy_dddqn = DDDQNGenerator(aig_dim=aig_dim, n_ops=N_OPS,
                                  recipe_max_len=RECIPE_LEN_MAX).to(device)
    policy_dddqn.load_state_dict(
        torch.load(POLICY_DDDQN_PATH, map_location=device, weights_only=False)["model"]
    )
    policy_dddqn.eval()
    print(f"[eval] loaded DDDQN policy from {POLICY_DDDQN_PATH}")

    test_designs = [d for d in TEST_DESIGNS if d in embeds]
    print(f"[eval] {len(test_designs)} test designs at lengths {EVAL_LENGTHS}")

    known = _gather_known(test_designs)
    per_restart = max(1, SA_ITERS // SA_RESTARTS)

    print(f"\n{'design':12s} {'L':>3s} {'oracle':>10s} {'DDDQN+SA':>10s} "
          f"{'resyn':>10s} {'resyn2':>10s}")
    print("-" * 60)

    for design in test_designs:
        _reset_csv(design)
        g = embeds[design].to(device)
        baselines = load_abc_baselines(design)

        # Pick the first EVAL_LENGTHS entry the design actually has data for.
        available = {L for (d, L) in known.keys() if d == design}
        lengths_here = []
        for L in EVAL_LENGTHS:
            if L in available:
                lengths_here = [L]
                break

        for L in lengths_here:
            items = known.get((design, L), [])
            norm = norms.get((design, L))
            if not items or norm is None:
                continue
            mu, sd = norm["power"]["mean"], norm["power"]["std"]
            z2w = lambda z: z * sd + mu

            # (1) oracle
            _, _, ops_o, p_oracle = _best_oracle(items)
            _write_row(design, L, "oracle", decode_recipe(ops_o), None, p_oracle)

            # (2) DDDQN + SA
            rl_inits = _sample_dddqn(
                policy_dddqn, g, L, SA_RESTARTS, N_OPS, RECIPE_LEN_MAX, device, rng,
            )
            best_z, best_rec = float("inf"), None
            for j, init in enumerate(rl_inits):
                rec_j, z_j, _ = simulated_annealing(
                    surrogate, embeds[design], N_OPS, L,
                    n_iter=per_restart, T0=SA_T0, T_end=SA_T_END,
                    init_recipe=init, device=device, seed=SEED + j,
                )
                if z_j < best_z:
                    best_z, best_rec = z_j, rec_j
            if best_rec is not None:
                ops_strs = decode_recipe(best_rec)
                try:
                    true_dddqn = abc_power(design, ops_strs)
                except Exception as e:
                    print(f"[eval] ABC failed for {design} L={L} DDDQN+SA: {e}")
                    true_dddqn = float("nan")
                _write_row(design, L, "DDDQN+SA", ops_strs, z2w(best_z), true_dddqn)
            else:
                true_dddqn = float("nan")

            # (3, 4) resyn / resyn2 baselines from abcStats_withmap
            r1 = baselines.get("resyn",  float("nan"))
            r2 = baselines.get("resyn2", float("nan"))
            _write_row(design, L, "resyn",  None, None, r1)
            _write_row(design, L, "resyn2", None, None, r2)

            print(f"{design:12s} {L:>3d} {p_oracle:>10.2f} {true_dddqn:>10.2f} "
                  f"{r1:>10.2f} {r2:>10.2f}")


if __name__ == "__main__":
    main()
