"""Prioritized Experience Replay buffer (sum-tree based)."""
from __future__ import annotations

import numpy as np


class SumTree:
    """Fixed-capacity binary sum tree.

    Internal nodes hold the sum of their children's priorities. Leaves at
    indices [capacity-1, 2*capacity-1) hold the per-slot priorities. The
    `data_idx` is the slot index in [0, capacity) corresponding to a leaf.
    """

    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """Append `priority` at the next slot (cyclic). Returns the tree index used."""
        tree_idx = self.write + self.capacity - 1
        self.update(tree_idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        return tree_idx

    def update(self, tree_idx: int, priority: float) -> None:
        if priority < 0:
            raise ValueError(f"priority must be >= 0, got {priority}")
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        idx = tree_idx
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float) -> tuple[int, float, int]:
        """Find leaf where cumulative priority crosses `s`. Returns (tree_idx, priority, data_idx)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), data_idx


class PrioritizedReplayBuffer:
    """Proportional PER, sum-tree based.

    Stores arbitrary transition objects; the buffer is agnostic to their
    structure. The caller is responsible for collating the returned batch
    into tensors.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(capacity)
        self.data: list = [None] * capacity
        self.write = 0
        self.size = 0
        self.max_priority = 1.0   # raw priority (pre-alpha) for new transitions

    def __len__(self) -> int:
        return self.size

    def push(self, transition) -> None:
        priority = (self.max_priority + self.eps) ** self.alpha
        self.tree.add(priority)
        self.data[self.write] = transition
        self.write = (self.write + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int, beta: float):
        """Sample `batch_size` transitions proportionally to priority.

        Returns (batch_list, tree_idxs, is_weights_array). IS weights are
        normalised so the max in the batch is 1.0 (Schaul et al., 2016).
        """
        assert self.size > 0, "buffer is empty"
        batch = []
        tree_idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        total = self.tree.total()
        segment = total / batch_size
        rng = np.random.default_rng()
        for i in range(batch_size):
            s = rng.uniform(segment * i, segment * (i + 1))
            # Defensive: ensure s < total so the sum-tree descent never lands on a stale
            # zero-priority leaf when the buffer is partially filled or due to FP rounding.
            s = min(s, total - 1e-12)
            tree_idx, priority, data_idx = self.tree.get(s)
            tree_idxs[i] = tree_idx
            priorities[i] = priority
            batch.append(self.data[data_idx])
        sampling_probs = priorities / max(total, 1e-12)
        is_weights = (self.size * sampling_probs) ** (-beta)
        is_weights = is_weights / is_weights.max()
        return batch, tree_idxs, is_weights.astype(np.float32)

    def update_priorities(self, tree_idxs, td_errors) -> None:
        """Update priorities for previously-sampled tree indices."""
        td_errors = np.asarray(td_errors, dtype=np.float64)
        raw = np.abs(td_errors) + self.eps
        for tree_idx, r in zip(tree_idxs, raw):
            priority = r ** self.alpha
            self.tree.update(int(tree_idx), float(priority))
            if r > self.max_priority:
                self.max_priority = float(r)
