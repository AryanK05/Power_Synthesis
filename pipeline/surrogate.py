"""
QoR predictor v2: f(g, recipe_padded, length) -> (z-power, z-area).

Improvements over v1:
  - Bidirectional LSTM (captures both forward and backward op dependencies)
  - Dropout + LayerNorm in the head (regularisation)
  - Multi-task head: predicts power AND area (free regulariser)

Inference signature is unchanged: forward(g, recipe, lengths) returns
[B, 1] predicted z-power. For multi-task training use forward_multi.
"""
import torch
import torch.nn as nn


class QoRSurrogate(nn.Module):
    def __init__(self, aig_dim, n_ops,
                 emb_dim=32, lstm_dim=64, head_dim=128, dropout=0.1):
        super().__init__()
        self.recipe_emb  = nn.Embedding(n_ops + 1, emb_dim, padding_idx=0)
        self.recipe_lstm = nn.LSTM(emb_dim, lstm_dim,
                                    batch_first=True, bidirectional=True)
        # bidirectional doubles the per-direction hidden, so concat = 2*lstm_dim.
        recipe_feat_dim = 2 * lstm_dim
        self.head = nn.Sequential(
            nn.Linear(aig_dim + recipe_feat_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.LayerNorm(head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, 2),    # [power_z, area_z]
        )
        self.lstm_dim = lstm_dim

    def encode_recipe(self, recipe, lengths):
        """recipe [B, T_pad], lengths [B] long. Returns h [B, 2*lstm_dim]."""
        e = self.recipe_emb(recipe)                            # [B, T_pad, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h, _) = self.recipe_lstm(packed)
        # h is [num_layers*2, B, lstm_dim]; we want forward-final ⊕ backward-final
        h_fwd = h[-2]                                          # [B, lstm_dim]
        h_bwd = h[-1]                                          # [B, lstm_dim]
        return torch.cat([h_fwd, h_bwd], dim=-1)               # [B, 2*lstm_dim]

    def forward_multi(self, g, recipe, lengths):
        """Returns [B, 2] = (z-power, z-area). Used for multi-task training."""
        r = self.encode_recipe(recipe, lengths)
        return self.head(torch.cat([g, r], dim=-1))

    def forward(self, g, recipe, lengths):
        """Returns [B, 1] predicted z-power. Inference / search interface."""
        return self.forward_multi(g, recipe, lengths)[:, 0:1]
