"""
RL policy: autoregressive recipe generator conditioned on AIG embedding.

Architecture
- g (D)            -> Linear -> initial GRU hidden state (H)
- step t: prev op  -> Embedding -> GRUCell -> Linear -> softmax over 7 ops
- start token: 0 (PAD); first sampled op is conditioned only on g
"""
import torch
import torch.nn as nn


class RecipePolicy(nn.Module):
    def __init__(self, aig_dim, n_ops, emb_dim=32, hidden=128):
        super().__init__()
        self.n_ops = n_ops
        self.hidden = hidden
        self.op_emb = nn.Embedding(n_ops + 1, emb_dim, padding_idx=0)
        self.init_h = nn.Linear(aig_dim, hidden)
        self.gru    = nn.GRUCell(emb_dim, hidden)
        self.head   = nn.Linear(hidden, n_ops)

    def sample(self, g, length, temperature=1.0):
        """g: [B, D]. Returns recipe[B,T] (op-ids in [1, n_ops]),
        log_probs[B,T], entropy[B,T]."""
        B = g.size(0)
        h = torch.tanh(self.init_h(g))
        prev = torch.zeros(B, dtype=torch.long, device=g.device)
        recipe, log_probs, entropies = [], [], []
        for _ in range(length):
            x = self.op_emb(prev)
            h = self.gru(x, h)
            logits = self.head(h) / max(temperature, 1e-6)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            op_id = a + 1                      # vocab is 1-indexed
            recipe.append(op_id)
            prev = op_id
        return (
            torch.stack(recipe, dim=1),
            torch.stack(log_probs, dim=1),
            torch.stack(entropies, dim=1),
        )

    @torch.no_grad()
    def greedy(self, g, length):
        B = g.size(0)
        h = torch.tanh(self.init_h(g))
        prev = torch.zeros(B, dtype=torch.long, device=g.device)
        out = []
        for _ in range(length):
            h = self.gru(self.op_emb(prev), h)
            a = self.head(h).argmax(dim=-1)
            op_id = a + 1
            out.append(op_id)
            prev = op_id
        return torch.stack(out, dim=1)
