"""
Train DDDQN+PER recipe generator (Method 4).

Same role as REINFORCE: emit a length-L recipe given an AIG embedding.
Reward is terminal-only: -surrogate(g, full_recipe, L).

Run: python -m pipeline.train_dddqn_init [--episodes N]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.config import (
    SEED, N_OPS, RECIPE_LEN_MAX, TEST_DESIGNS, SKIP_DESIGNS,
    DDDQN_TRAIN_L_MODE, DDDQN_EPISODES, DDDQN_BATCH, DDDQN_LR, DDDQN_GAMMA,
    DDDQN_BUFFER_INIT, DDDQN_PER_ALPHA, DDDQN_PER_BETA0, DDDQN_PER_BETA1,
    DDDQN_PER_EPS, DDDQN_TARGET_SYNC, DDDQN_GRAD_CLIP,
    DDDQN_EPS_START, DDDQN_EPS_END, DDDQN_EPS_DECAY_FRAC,
    SURR_PATH, POLICY_DDDQN_PATH,
)
from pipeline.data import load_embeddings, list_all_designs
from pipeline.surrogate import QoRSurrogate
from pipeline.dddqn.networks import DDDQNGenerator
from pipeline.dddqn.buffer import PrioritizedReplayBuffer
from pipeline.dddqn.agent import DDDQNAgent, select_action_eps_greedy
from pipeline.dddqn.utils import sample_episode_length


def _state_signature(state_dict):
    return {k: tuple(v.shape) for k, v in state_dict.items()}


def _assert_signature_match(saved_sig, current_sig, ckpt_path, which):
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
            f"{which} architecture mismatch.",
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


def _resume_dddqn_if_requested(resume, online, target, opt, aig_dim, meta, save_path):
    if not resume:
        return 0
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"--resume set but checkpoint not found: {save_path}. "
            "Run without --resume first."
        )

    ckpt = torch.load(save_path, map_location="cpu")
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
    if int(ckpt.get("recipe_max_len", -1)) != int(RECIPE_LEN_MAX):
        raise RuntimeError(
            f"Checkpoint recipe_max_len mismatch: saved={ckpt.get('recipe_max_len')} "
            f"current={RECIPE_LEN_MAX}. Run without --resume."
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

    saved_online = ckpt.get("online_model", ckpt.get("model"))
    saved_target = ckpt.get("target_model", saved_online)
    if saved_online is None:
        raise RuntimeError(
            f"Checkpoint missing online_model/model weights: {save_path}. "
            "Run without --resume."
        )

    online_sig = ckpt.get("online_signature", _state_signature(saved_online))
    target_sig = ckpt.get("target_signature", _state_signature(saved_target))
    _assert_signature_match(
        online_sig,
        _state_signature(online.state_dict()),
        save_path,
        which="online_model",
    )
    _assert_signature_match(
        target_sig,
        _state_signature(target.state_dict()),
        save_path,
        which="target_model",
    )

    online.load_state_dict(saved_online, strict=True)
    target.load_state_dict(saved_target, strict=True)
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    start_episode = int(ckpt.get("episodes_completed", 0))
    print(f"[dddqn] resumed from {save_path} at episode {start_episode}")
    return start_episode


def _epsilon(step: int, total: int) -> float:
    decay_steps = max(1, int(total * DDDQN_EPS_DECAY_FRAC))
    if step >= decay_steps:
        return DDDQN_EPS_END
    return DDDQN_EPS_START + (DDDQN_EPS_END - DDDQN_EPS_START) * (step / decay_steps)


def _beta(step: int, total: int) -> float:
    if total <= 1:
        return DDDQN_PER_BETA1
    return DDDQN_PER_BETA0 + (DDDQN_PER_BETA1 - DDDQN_PER_BETA0) * (step / (total - 1))


def _state(g: torch.Tensor, recipe: torch.Tensor, t: int, L: int) -> dict:
    """All tensors placed on the same device as `g`."""
    return {
        "g":      g,
        "recipe": recipe.clone(),
        "t":      torch.tensor(t, dtype=torch.long, device=g.device),
        "L":      torch.tensor(L, dtype=torch.long, device=g.device),
    }


def main(episodes: int = DDDQN_EPISODES, save_path=None, resume=False) -> None:
    if save_path is None:
        save_path = POLICY_DDDQN_PATH

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeds, meta = load_embeddings()
    aig_dim = meta["out_dim"]

    surrogate = QoRSurrogate(aig_dim=aig_dim, n_ops=N_OPS).to(device)
    surrogate.load_state_dict(torch.load(SURR_PATH, map_location=device)["model"])
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    online = DDDQNGenerator(aig_dim=aig_dim, n_ops=N_OPS,
                            recipe_max_len=RECIPE_LEN_MAX).to(device)
    target = DDDQNGenerator(aig_dim=aig_dim, n_ops=N_OPS,
                            recipe_max_len=RECIPE_LEN_MAX).to(device)
    opt = torch.optim.Adam(online.parameters(), lr=DDDQN_LR)
    agent = DDDQNAgent(online, target, opt, gamma=DDDQN_GAMMA,
                       grad_clip=DDDQN_GRAD_CLIP)

    start_episode = _resume_dddqn_if_requested(
        resume=resume,
        online=online,
        target=target,
        opt=opt,
        aig_dim=aig_dim,
        meta=meta,
        save_path=save_path,
    )
    if start_episode >= episodes:
        print(f"[dddqn] nothing to do: checkpoint already at episode {start_episode} "
              f"with requested episodes={episodes}")
        return

    buffer = PrioritizedReplayBuffer(
        capacity=DDDQN_BUFFER_INIT,
        alpha=DDDQN_PER_ALPHA,
        eps=DDDQN_PER_EPS,
    )

    train_designs = [d for d in list_all_designs()
                     if d not in TEST_DESIGNS
                     and d not in SKIP_DESIGNS
                     and d in embeds]
    print(f"[dddqn] {len(train_designs)} train designs, {episodes} episodes "
          f"(L-mode={DDDQN_TRAIN_L_MODE})")
    if start_episode > 0:
        print(f"[dddqn] continuing from episode {start_episode} "
              f"(eps now {_epsilon(start_episode, episodes):.3f})")

    grad_step = 0
    log_interval = max(1, (episodes - start_episode) // 30)
    recent_returns: list[float] = []

    for ep in range(start_episode, episodes):
        design = train_designs[int(rng.integers(0, len(train_designs)))]
        g = embeds[design].to(device)
        L = sample_episode_length(DDDQN_TRAIN_L_MODE, rng)

        recipe = torch.zeros(RECIPE_LEN_MAX, dtype=torch.long, device=device)
        traj = []
        for t in range(L):
            s = _state(g, recipe, t, L)
            with torch.no_grad():
                Q, _ = online(
                    s["g"].unsqueeze(0),
                    s["recipe"].unsqueeze(0),
                    s["t"].unsqueeze(0),
                    s["L"].unsqueeze(0),
                )
            a = select_action_eps_greedy(Q[0], eps=_epsilon(ep, episodes), rng=rng)
            op_id = a + 1                                   # vocab is 1-indexed
            recipe[t] = op_id
            traj.append((s, a))

        with torch.no_grad():
            L_t = torch.tensor([L], dtype=torch.long, device=device)
            pred_z = surrogate(g.unsqueeze(0), recipe.unsqueeze(0), L_t).item()
        terminal_reward = -pred_z
        recent_returns.append(terminal_reward)
        if len(recent_returns) > 100:
            recent_returns.pop(0)

        for i, (s, a) in enumerate(traj):
            if i + 1 < len(traj):
                s_next, _ = traj[i + 1]
                r, done = 0.0, False
            else:
                s_next = _state(g, recipe, L, L)
                r, done = float(terminal_reward), True
            buffer.push((s, a, r, s_next, done))

            if len(buffer) >= DDDQN_BATCH:
                batch, idxs, is_w = buffer.sample(DDDQN_BATCH, beta=_beta(ep, episodes))
                agent.update(batch, idxs, is_w, buffer)
                grad_step += 1
                if grad_step % DDDQN_TARGET_SYNC == 0:
                    agent.sync_target()

        if ep % log_interval == 0:
            avg_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
            print(f"ep {ep:04d} | L={L:2d} | reward {terminal_reward:+.3f} | "
                  f"avg100 {avg_ret:+.3f} | eps {_epsilon(ep, episodes):.3f} | "
                  f"buf {len(buffer)}")

    torch.save({
        "model":          online.state_dict(),
        "online_model":   online.state_dict(),
        "target_model":   target.state_dict(),
        "online_signature": _state_signature(online.state_dict()),
        "target_signature": _state_signature(target.state_dict()),
        "optimizer":      opt.state_dict(),
        "episodes_completed": episodes,
        "aig_dim":        aig_dim,
        "n_ops":          N_OPS,
        "recipe_max_len": RECIPE_LEN_MAX,
        "encoder_meta":   meta,
    }, save_path)
    print(f"[dddqn] saved policy -> {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=DDDQN_EPISODES)
    ap.add_argument("--resume", action="store_true",
                    help="Resume DDDQN training from checkpoint with strict architecture checks")
    args = ap.parse_args()
    main(episodes=args.episodes, resume=args.resume)
