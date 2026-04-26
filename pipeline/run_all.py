"""
End-to-end orchestration. Run from project root:

    python -m pipeline.run_all
    python -m pipeline.run_all --resume

Runs (in order):
    1. embed_designs       (cache hit if already done)
    2. train_surrogate     (~45 min)
    3. train_dddqn_init    (~60 min)  ← primary RL agent
    4. evaluate            (~30 min)

`pipeline.train_rl` (REINFORCE) is NOT in this orchestrator. Run it manually
if you want a REINFORCE+SA column in the eval table:
    python -m pipeline.train_rl
then re-run evaluate.
"""
import subprocess
import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(stage, module, module_args=None):
    print(f"\n{'='*72}\n[run_all] {stage}\n{'='*72}")
    cmd = [sys.executable, "-m", module]
    if module_args:
        cmd.extend(module_args)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        sys.exit(f"[run_all] {stage} failed (exit {r.returncode})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Resume surrogate and DDDQN stages from checkpoints")
    args = ap.parse_args()

    _run("1/4 embed designs",   "pipeline.embed_designs")
    train_args = ["--resume"] if args.resume else None
    _run("2/4 train surrogate", "pipeline.train_surrogate", train_args)
    _run("3/4 train DDDQN",     "pipeline.train_dddqn_init", train_args)
    _run("4/4 evaluate",        "pipeline.evaluate")
