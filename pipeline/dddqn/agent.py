"""DDDQN+PER agent. Backbone-agnostic: works with any (state -> (Q, mask)) net."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def select_action_eps_greedy(
    Q: torch.Tensor,
    eps: float,
    rng: np.random.Generator,
    valid: Optional[torch.Tensor] = None,
) -> int:
    """ε-greedy selection from a 1-D Q vector. `valid` (bool, same shape) optional."""
    Q = Q.detach()
    if valid is None:
        valid_idx = torch.arange(Q.size(-1))
    else:
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    assert valid_idx.numel() > 0, "no valid actions"
    if rng.random() < eps:
        return int(valid_idx[rng.integers(0, valid_idx.numel())].item())
    if valid is not None:
        Q_masked = Q.masked_fill(~valid, float("-inf"))
        return int(Q_masked.argmax().item())
    return int(Q.argmax().item())


def _stack_state(states: list[dict]) -> dict:
    """Collate a list of dict-states into a dict of batched tensors."""
    keys = states[0].keys()
    return {k: torch.stack([s[k] for s in states], dim=0) for k in keys}


class DDDQNAgent:
    """Backbone-agnostic Dueling Double DQN with PER.

    The Q-net is a callable mapping state-kwargs to (Q [B, n_actions], mask
    [B, n_actions] or None). The agent itself does not care about the state
    structure — it forwards collated batched dicts to the net.
    """

    def __init__(
        self,
        online_net,
        target_net,
        optimizer,
        gamma: float = 0.99,
        grad_clip: float = 1.0,
    ):
        self.online = online_net
        self.target = target_net
        self.opt = optimizer
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.sync_target()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def update(self, batch: list, tree_idxs, is_weights, buffer):
        """One gradient step on a PER mini-batch.

        Each transition is `(state, action, reward, next_state, done)` with
        `state` and `next_state` as dicts of tensors. `action` is an int.
        Returns `(loss_value, td_errors)`. Updates priorities in-place on `buffer`.
        """
        states, actions, rewards, next_states, dones = zip(*batch)
        s = _stack_state(list(states))
        s_next = _stack_state(list(next_states))
        a = torch.tensor(actions, dtype=torch.long)
        r = torch.tensor(rewards, dtype=torch.float32)
        d = torch.tensor(dones, dtype=torch.float32)
        is_w = torch.tensor(is_weights, dtype=torch.float32)

        # Move to the same device as the online net.
        device = next(self.online.parameters()).device
        s = {k: v.to(device) for k, v in s.items()}
        s_next = {k: v.to(device) for k, v in s_next.items()}
        a, r, d, is_w = a.to(device), r.to(device), d.to(device), is_w.to(device)

        # Online Q for chosen actions.
        Q_all, _ = self.online(**s)
        q_a = Q_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)

        # Double-Q target: argmax via online, evaluation via target.
        with torch.no_grad():
            Q_next_online, mask_next = self.online(**s_next)
            if mask_next is not None:
                Q_next_online = Q_next_online.masked_fill(~mask_next, float("-inf"))
            a_next = Q_next_online.argmax(dim=-1)

            Q_next_target, _ = self.target(**s_next)
            q_next = Q_next_target.gather(-1, a_next.unsqueeze(-1)).squeeze(-1)
            target = r + self.gamma * q_next * (1.0 - d)

        td_error = target - q_a
        # IS-weighted Huber loss.
        elementwise = F.smooth_l1_loss(q_a, target, reduction="none")
        loss = (is_w * elementwise).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.opt.step()

        td_errors_np = td_error.detach().abs().cpu().numpy()
        buffer.update_priorities(tree_idxs, td_errors_np)
        return float(loss.item()), td_errors_np
