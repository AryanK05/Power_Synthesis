"""
REINFORCE training of RecipePolicy. Kept for fallback; not run by default.

Sample length per episode from RECIPE_LEN_VALID; pad recipe to RECIPE_LEN_MAX
when querying the surrogate.

Run with:  python -m pipeline.train_rl
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.config import (
    TEST_DESIGNS, SKIP_DESIGNS, SEED, RECIPE_LEN_MAX, RECIPE_LEN_VALID,
    N_OPS, PAD_IDX, RL_EPISODES, RL_BATCH, RL_LR, RL_ENTROPY_COEF,
    SURR_PATH, POLICY_PATH,
)
from pipeline.data import load_embeddings, list_all_designs
from pipeline.surrogate import QoRSurrogate
from pipeline.policy import RecipePolicy


def _pad_recipe_to_max(recipe_BL):
    """recipe_BL [B, L] -> [B, RECIPE_LEN_MAX] padded with PAD_IDX."""
    B, L = recipe_BL.shape
    if L >= RECIPE_LEN_MAX:
        return recipe_BL[:, :RECIPE_LEN_MAX]
    pad = torch.full((B, RECIPE_LEN_MAX - L), PAD_IDX,
                     dtype=recipe_BL.dtype, device=recipe_BL.device)
    return torch.cat([recipe_BL, pad], dim=1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeds, meta = load_embeddings()
    aig_dim = meta["out_dim"]

    surrogate = QoRSurrogate(aig_dim=aig_dim, n_ops=N_OPS).to(device)
    surrogate.load_state_dict(torch.load(SURR_PATH, map_location=device,
                                         weights_only=False)["model"])
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    policy = RecipePolicy(aig_dim=aig_dim, n_ops=N_OPS).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=RL_LR)

    train_designs = [d for d in list_all_designs()
                     if d not in TEST_DESIGNS
                     and d not in SKIP_DESIGNS
                     and d in embeds]
    train_g = torch.stack([embeds[n] for n in train_designs]).to(device)
    print(f"[rl] {train_g.size(0)} training designs, {RL_EPISODES} episodes "
          f"(batch={RL_BATCH})")

    baseline = 0.0
    ema_alpha = 0.05
    log_interval = max(1, RL_EPISODES // 30)

    for episode in range(RL_EPISODES):
        idx = torch.randint(0, train_g.size(0), (RL_BATCH,), device=device)
        g = train_g[idx]
        L = int(rng.choice(RECIPE_LEN_VALID))

        recipe, log_probs, entropies = policy.sample(g, L)         # [B, L]
        recipe_padded = _pad_recipe_to_max(recipe)                 # [B, RECIPE_LEN_MAX]
        L_t = torch.tensor([L] * RL_BATCH, dtype=torch.long, device=device)
        with torch.no_grad():
            pred_z = surrogate(g, recipe_padded, L_t).squeeze(-1)  # [B]
        reward = -pred_z
        baseline = (1 - ema_alpha) * baseline + ema_alpha * reward.mean().item()
        adv = reward - baseline

        loss_pg  = -(log_probs.sum(dim=1) * adv).mean()
        loss_ent = -RL_ENTROPY_COEF * entropies.mean()
        loss = loss_pg + loss_ent

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if episode % log_interval == 0:
            print(f"ep {episode:04d} | L={L:2d} | reward {reward.mean().item():+.3f} | "
                  f"baseline {baseline:+.3f} | ent {entropies.mean().item():.3f} | "
                  f"loss {loss.item():+.3f}")

    torch.save({
        "model":        policy.state_dict(),
        "aig_dim":      aig_dim,
        "n_ops":        N_OPS,
        "encoder_meta": meta,
    }, POLICY_PATH)
    print(f"[rl] saved policy -> {POLICY_PATH}")


if __name__ == "__main__":
    main()
