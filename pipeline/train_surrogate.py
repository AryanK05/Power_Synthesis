"""
Train QoR surrogate v2 on multi-source variable-length data.

Loss = MSE_z(power) + α * MSE_z(area) + β * pairwise_rank_loss(power)

  - MSE_z(power)        : primary regression target
  - MSE_z(area)          : auxiliary multi-task regulariser
  - pairwise_rank_loss   : within each (design, length) bucket in a batch,
                           sample pairs with different power_z and require
                           predicted ordering to match. Directly optimises
                           Spearman, which SA / RL care about.

Train on all designs except TEST / SKIP. Eval on TEST_DESIGNS.
Save best checkpoint by mean holdout Spearman on POWER.
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from pipeline.config import (
    TEST_DESIGNS, SKIP_DESIGNS, SEED,
    SURROGATE_EPOCHS, SURROGATE_BATCH, SURROGATE_LR,
    SURR_PATH, N_OPS,
)
from pipeline.data import (
    PowerDataset, list_all_designs, load_embeddings,
    compute_design_length_norms, collate,
)
from pipeline.surrogate import QoRSurrogate


AREA_COEF = 0.3            # weight on auxiliary area regression
RANK_COEF = 1.0            # weight on within-(design, length) pairwise loss
RANK_MARGIN = 0.0          # pairwise margin


def _state_signature(state_dict):
    return {k: tuple(v.shape) for k, v in state_dict.items()}


def _assert_signature_match(saved_sig, current_sig, ckpt_path):
    saved_keys = set(saved_sig.keys())
    current_keys = set(current_sig.keys())
    missing = sorted(current_keys - saved_keys)
    unexpected = sorted(saved_keys - current_keys)
    shape_mismatch = sorted(
        k for k in (saved_keys & current_keys)
        if tuple(saved_sig[k]) != tuple(current_sig[k])
    )
    if missing or unexpected or shape_mismatch:
        lines = [
            "Checkpoint architecture mismatch.",
            f"checkpoint: {ckpt_path}",
            f"missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}",
            f"unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}",
        ]
        if shape_mismatch:
            sample = shape_mismatch[:8]
            details = [
                f"{k}: saved={tuple(saved_sig[k])} current={tuple(current_sig[k])}"
                for k in sample
            ]
            lines.append("shape mismatches: " + "; ".join(details))
        lines.append("Run without --resume for a fresh initialization.")
        raise RuntimeError("\n".join(lines))


def _resume_surrogate_if_requested(resume, model, opt, aig_dim, meta):
    if not resume:
        return 0, -float("inf")
    if not os.path.exists(SURR_PATH):
        raise FileNotFoundError(
            f"--resume set but checkpoint not found: {SURR_PATH}. "
            "Run without --resume first."
        )

    ckpt = torch.load(SURR_PATH, map_location="cpu")
    if int(ckpt.get("aig_dim", -1)) != int(aig_dim):
        raise RuntimeError(
            f"Checkpoint aig_dim mismatch: saved={ckpt.get('aig_dim')} current={aig_dim}. "
            "Run without --resume."
        )
    if int(ckpt.get("n_ops", -1)) != int(N_OPS):
        raise RuntimeError(
            f"Checkpoint n_ops mismatch: saved={ckpt.get('n_ops')} current={N_OPS}. "
            "Run without --resume."
        )

    enc_saved = ckpt.get("encoder_meta", {})
    if isinstance(enc_saved, dict):
        if enc_saved.get("encoder") != meta.get("encoder"):
            raise RuntimeError(
                f"Checkpoint encoder mismatch: saved={enc_saved.get('encoder')} "
                f"current={meta.get('encoder')}. Run without --resume."
            )
        if int(enc_saved.get("out_dim", -1)) != int(meta.get("out_dim", -2)):
            raise RuntimeError(
                f"Checkpoint encoder out_dim mismatch: saved={enc_saved.get('out_dim')} "
                f"current={meta.get('out_dim')}. Run without --resume."
            )

    current_sig = _state_signature(model.state_dict())
    saved_sig = ckpt.get("model_signature")
    if saved_sig is None:
        saved_sig = _state_signature(ckpt["model"])
    _assert_signature_match(saved_sig, current_sig, SURR_PATH)

    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_rho = float(ckpt.get("best_rho", -float("inf")))
    print(f"[train] resumed from {SURR_PATH} at epoch {start_epoch:02d} "
          f"(best_rho={best_rho:+.3f})")
    return start_epoch, best_rho


def _spearman(x, y):
    if _HAS_SCIPY:
        rho = spearmanr(x, y).correlation
        return float(rho) if rho == rho else 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    if n < 2:
        return 0.0
    return float(1 - 6 * np.sum((rx - ry) ** 2) / (n * (n * n - 1)))


def pairwise_rank_loss(pred_pow, target_pow, designs, lengths, margin=0.0):
    """Within each (design, length) group, penalise misordered pairs.

    pred_pow, target_pow: [B] tensors (z-scored power).
    designs, lengths: lists of B identifiers used to bucket samples.
    """
    device = pred_pow.device
    groups = defaultdict(list)
    for i, (d, L) in enumerate(zip(designs, lengths)):
        groups[(d, int(L))].append(i)
    losses = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
        p = pred_pow.index_select(0, idxs_t)
        t = target_pow.index_select(0, idxs_t)
        # diff_t[i, j] = t[i] - t[j];  diff_p analogous
        diff_t = t.unsqueeze(0) - t.unsqueeze(1)
        diff_p = p.unsqueeze(0) - p.unsqueeze(1)
        # Pairs where target ordering says p[i] < p[j].
        mask = diff_t < 0
        if mask.any():
            losses.append(F.relu(margin + diff_p[mask]).mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def per_dl_eval(model, ds, device):
    if len(ds) == 0:
        return {}
    model.eval()
    bucket = {}
    for s in ds.samples:
        bucket.setdefault((s["design"], s["length"]), []).append(s)
    out = {}
    with torch.no_grad():
        for key, items in bucket.items():
            g = torch.stack([s["g"] for s in items]).to(device)
            r = torch.stack([s["recipe"] for s in items]).to(device)
            L = torch.tensor([s["length"] for s in items],
                             dtype=torch.long, device=device)
            yz = np.array([s["power_z"]   for s in items])
            yp = np.array([s["power_raw"] for s in items])
            pz = model(g, r, L).squeeze(-1).cpu().numpy()
            out[key] = {
                "n":        len(items),
                "mse_z":    float(np.mean((pz - yz) ** 2)),
                "spearman": _spearman(pz, yp),
            }
    return out


def main(resume=False, epochs=SURROGATE_EPOCHS):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeds, meta = load_embeddings()
    aig_dim = meta["out_dim"]
    print(f"[train] encoder={meta['encoder']} dim={aig_dim}")

    all_designs   = list_all_designs()
    test_designs  = [d for d in TEST_DESIGNS if d in embeds]
    train_designs = [d for d in all_designs
                     if d not in TEST_DESIGNS
                     and d not in SKIP_DESIGNS
                     and d in embeds]
    print(f"[train] {len(train_designs)} train, {len(test_designs)} test designs")

    norms = compute_design_length_norms(set(train_designs) | set(test_designs))

    train_ds = PowerDataset(train_designs, embeds, norms)
    test_ds  = PowerDataset(test_designs,  embeds, norms)
    print(f"[train] samples — train: {len(train_ds):,}  test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=SURROGATE_BATCH,
                              shuffle=True, collate_fn=collate)

    model = QoRSurrogate(aig_dim=aig_dim, n_ops=N_OPS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=SURROGATE_LR)
    crit = nn.MSELoss()

    start_epoch, best_rho = _resume_surrogate_if_requested(
        resume=resume,
        model=model,
        opt=opt,
        aig_dim=aig_dim,
        meta=meta,
    )

    if start_epoch >= epochs:
        print(f"[train] nothing to do: checkpoint already at epoch {start_epoch:02d} "
              f"with requested epochs={epochs}")
        return

    for epoch in range(start_epoch, epochs):
        model.train()
        run_total, run_mse_p, run_mse_a, run_rank, n_seen = 0.0, 0.0, 0.0, 0.0, 0
        for (g, recipe, length, yz_pow, _, yz_area, _,
             designs, _, _) in train_loader:
            g       = g.to(device)
            recipe  = recipe.to(device)
            length  = length.to(device)
            yz_pow  = yz_pow.to(device)
            yz_area = yz_area.to(device)

            opt.zero_grad()
            pred = model.forward_multi(g, recipe, length)        # [B, 2]
            mse_p = crit(pred[:, 0:1], yz_pow)
            mse_a = crit(pred[:, 1:2], yz_area)
            rank  = pairwise_rank_loss(
                pred[:, 0], yz_pow.squeeze(-1),
                designs, length.cpu().tolist(), margin=RANK_MARGIN,
            )
            loss = mse_p + AREA_COEF * mse_a + RANK_COEF * rank
            loss.backward()
            opt.step()

            B = g.size(0)
            run_total += loss.item()  * B
            run_mse_p += mse_p.item() * B
            run_mse_a += mse_a.item() * B
            run_rank  += rank.item()  * B
            n_seen    += B
        n_seen = max(1, n_seen)

        train_eval = per_dl_eval(model, train_ds, device)
        test_eval  = per_dl_eval(model, test_ds,  device)
        train_rho = (float(np.mean([v["spearman"] for v in train_eval.values()]))
                     if train_eval else 0.0)
        test_rho  = (float(np.mean([v["spearman"] for v in test_eval.values()]))
                     if test_eval else 0.0)
        test_mse  = (float(np.mean([v["mse_z"]   for v in test_eval.values()]))
                     if test_eval else 0.0)

        marker = ""
        if test_rho > best_rho:
            best_rho = test_rho
            torch.save({
                "model":        model.state_dict(),
                "model_signature": _state_signature(model.state_dict()),
                "optimizer":    opt.state_dict(),
                "aig_dim":      aig_dim,
                "n_ops":        N_OPS,
                "encoder_meta": meta,
                "epoch":        epoch,
                "epochs_total": epochs,
                "test_per_dl":  test_eval,
                "best_rho":     best_rho,
            }, SURR_PATH)
            marker = "  *"

        print(f"epoch {epoch:02d} | total {run_total/n_seen:.4f} | "
              f"mse_p {run_mse_p/n_seen:.4f} | mse_a {run_mse_a/n_seen:.4f} | "
              f"rank {run_rank/n_seen:.4f} | "
              f"train_rho {train_rho:+.3f} | test_rho {test_rho:+.3f} | "
              f"test_mse_z {test_mse:.4f}{marker}")

    print("\n[train] final per-(design,length) metrics on test:")
    final = per_dl_eval(model, test_ds, device)
    by_design = {}
    for (d, L), v in sorted(final.items()):
        by_design.setdefault(d, []).append((L, v))
    for d, rows in by_design.items():
        avg = float(np.mean([v["spearman"] for _, v in rows]))
        print(f"  {d:14s} avg_rho={avg:+.3f}")
        for L, v in sorted(rows):
            print(f"     L={L:2d}  rho={v['spearman']:+.3f}  "
                  f"MSE_z={v['mse_z']:.4f}  n={v['n']}")
    print(f"\n[train] best checkpoint -> {SURR_PATH}  (best test_rho={best_rho:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Resume surrogate training from checkpoint with strict architecture checks")
    ap.add_argument("--epochs", type=int, default=SURROGATE_EPOCHS,
                    help="Total number of epochs to train (when resuming, continue until this epoch count)")
    args = ap.parse_args()
    main(resume=args.resume, epochs=args.epochs)
