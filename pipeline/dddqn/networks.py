"""Dueling Q-networks for DDDQN methods 4 and 5."""
from __future__ import annotations

import torch
import torch.nn as nn


class RecipeTrunk(nn.Module):
    """Shared encoder: (g, recipe, ctx) -> (z trunk feature, h_seq per-position).

    `ctx_dim` is the size of the auxiliary scalar/embedding vector
    (e.g. [t/20, L/20] for method 4, or [T, progress, cur_e, L/20] for method 5).
    """

    def __init__(
        self,
        aig_dim: int,
        n_ops: int,
        recipe_max_len: int,
        hidden: int,
        ctx_dim: int,
    ):
        super().__init__()
        self.aig_dim = aig_dim
        self.n_ops = n_ops
        self.recipe_max_len = recipe_max_len
        self.hidden = hidden
        self.ctx_dim = ctx_dim

        self.g_mlp = nn.Sequential(
            nn.Linear(aig_dim, 128),
            nn.ReLU(),
        )
        self.recipe_emb = nn.Embedding(n_ops + 1, 32, padding_idx=0)
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, 32),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(128 + hidden + 32, 256),
            nn.ReLU(),
        )
        self.out_dim = 256

    def forward(self, g: torch.Tensor, recipe: torch.Tensor, ctx: torch.Tensor):
        """g [B, aig_dim], recipe [B, recipe_max_len] long, ctx [B, ctx_dim]."""
        g_feat = self.g_mlp(g)
        recipe_emb = self.recipe_emb(recipe)            # [B, L_max, 32]
        h_seq, h_T = self.gru(recipe_emb)               # h_seq [B, L_max, H], h_T [1, B, H]
        h_T = h_T.squeeze(0)                            # [B, H]
        ctx_feat = self.ctx_mlp(ctx)                    # [B, 32]
        z = self.combine(torch.cat([g_feat, h_T, ctx_feat], dim=-1))   # [B, 256]
        return z, h_seq


class DDDQNGenerator(nn.Module):
    """Method 4: state -> Q over 7 next-op actions. Dueling head."""

    def __init__(
        self,
        aig_dim: int = 256,
        n_ops: int = 7,
        recipe_max_len: int = 20,
        hidden: int = 128,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.recipe_max_len = recipe_max_len
        self.trunk = RecipeTrunk(
            aig_dim=aig_dim,
            n_ops=n_ops,
            recipe_max_len=recipe_max_len,
            hidden=hidden,
            ctx_dim=2,
        )
        self.v_head = nn.Sequential(
            nn.Linear(self.trunk.out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.a_head = nn.Sequential(
            nn.Linear(self.trunk.out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_ops),
        )

    def forward(
        self,
        g: torch.Tensor,
        recipe: torch.Tensor,
        t: torch.Tensor,
        L: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Returns (Q [B, n_ops], None). The None mask matches DDDQNDriver's interface."""
        L_max = float(self.recipe_max_len)
        ctx = torch.stack([t.float() / L_max, L.float() / L_max], dim=-1)
        z, _ = self.trunk(g, recipe, ctx)
        V = self.v_head(z)                              # [B, 1]
        A = self.a_head(z)                              # [B, n_ops]
        Q = V + A - A.mean(dim=-1, keepdim=True)
        return Q, None
