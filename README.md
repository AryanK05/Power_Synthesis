# Inverse Recipe Search for Power Optimization

Given an AIG, find a 7-op ABC recipe of length **L = 20** (configurable) that
minimises post-mapping (Nangate45) power. The system is a **frozen DeepGate2
encoder + a learned QoR surrogate (v2)**, used as the energy/reward function
for **simulated annealing** and a **DDDQN+PER** policy. Final recipes are
validated against ABC for true Nangate45-mapped power and compared to ABC's
hand-tuned baselines (`resyn`, `resyn2`).

## Pipeline at a glance

```
                       ┌──────────────────┐
   AIG (.aig)  ──────▶ │ DeepGate2 frozen │ ───▶  g ∈ ℝ²⁵⁶  (cached once)
                       └──────────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────────┐
                                          │  QoR surrogate (v2)        │
                                          │  bi-LSTM + multi-task +    │
                                          │  pairwise ranking loss     │
                                          │  f(g, recipe, L) → z-power │
                                          └────────────────────────────┘
                                                       │ frozen during search
                                                       ▼
                                          ┌────────────────────────────┐
                                          │   DDDQN+PER policy         │
                                          │   samples 5 recipes        │
                                          └────────────────────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────────┐
                                          │   SA refines each (60k iters)
                                          └────────────────────────────┘
                                                       │
                                                       ▼
                                          ABC (Nangate45 + map)
                                                       ↓
                                  results/<design>.csv  ←  vs  abcStats_withmap/
```

## Data layout

```
data/                       (15 designs × 500 length-20 recipes;   7,500 pts)
formatted_data(8-9)/        (26 designs × 1000 recipes len 8 or 9; 26,000)
formatted_data(12-15)/      (34 designs × 2000 recipes len 12-15;  68,000)
abcStats_withmap/           ABC-baseline true_power for resyn / resyn2 (15 designs)
results/<design>.csv        per-design search outputs (filled at eval time)
pipeline/                   all code
```

## Splits

| Bucket | Count | Designs |
|---|---|---|
| **Test** | 7 | `i2c, ss_pcm, usb_phy, sasc, spi, wb_dma, tv80` |
| **Train** | ≈30 | all remaining designs whose AIG fits the 150 KB cap |
| **Skipped** | 9 | `dft, idft, hyp, vga_lcd, picosoc, bp_be, div, ethernet, jpeg` |

Why 9 designs are skipped: DeepGate2's `top_sort` is ~O(N³) in gate count and
hangs on AIGs above ~10 K gates. A 150 KB AIG file size cap (configurable in
`pipeline/config.py`) is the practical workaround. Removing the cap requires
either patching DG2's parser internals or switching to DG3 with windowing —
both out of scope for this project.

Eval-set selection criteria, in priority order:
1. Has an entry in `abcStats_withmap/` (apples-to-apples vs `resyn` / `resyn2`)
2. AIG ≤ 50 KB so per-recipe ABC eval stays fast
3. Has length-20 recipes (so eval is at the project's primary length)

## File layout

```
pipeline/
├── config.py                paths, vocab, splits, hyperparams, length set
├── encoders.py              DeepGate2 wrapper (256-d mean-pool of hs⊕hf)
├── embed_designs.py         one-time encoding pass; incremental cache
├── data.py                  multi-source variable-length loader (padded to 20)
├── surrogate.py             QoRSurrogate v2 — bi-LSTM, dropout, multi-output
├── train_surrogate.py       MSE + pairwise rank + multi-task; save by Spearman
├── policy.py                REINFORCE policy (length-conditioned) — fallback
├── train_rl.py              REINFORCE training (optional)
├── dddqn/
│   ├── agent.py             DDDQN (Double-Q + dueling head + IS-weighted Huber)
│   ├── buffer.py            sum-tree Prioritized Experience Replay
│   ├── networks.py          dueling Q-net over RecipeTrunk
│   └── utils.py             discrete-set length sampler
├── train_dddqn_init.py      DDDQN training loop (primary RL)
├── sa_search.py             SA + LAHC (LAHC kept; unused at eval right now)
├── abc_runner.py            run a recipe through ABC + Nangate45 + map
├── evaluate.py              per-(design,length,method) results table
├── run_all.py               4-stage orchestrator
└── README.md
```

## Surrogate v2 architecture

```python
QoRSurrogate(
    aig_dim=256,                # DG2 mean-pool dimension
    n_ops=7,
    emb_dim=32,                 # op embedding
    lstm_dim=64,                # bi-LSTM hidden (per direction)
    head_dim=128,
    dropout=0.1,
)
# Recipe → Embedding(8, 32) → BiLSTM(32→64×2) → concat fwd+bwd hidden = [B, 128]
# AIG g (256-d) ⊕ recipe_h (128-d) → [Linear→LayerNorm→ReLU→Dropout]×2 → Linear(2)
# Output: [z_power, z_area]   # area used for multi-task regularisation
```

**Loss** = `MSE_z(power) + 0.3·MSE_z(area) + 1.0·pairwise_rank(power)`

The pairwise term groups in-batch samples by `(design, length)` and penalises
mis-ordered pairs — directly optimises the Spearman ρ that SA cares about.

## Recipe vocab

```python
VOCAB = {"refactor -z": 1, "balance": 2, "rewrite": 3, "rewrite -z": 4,
         "resub": 5, "resub -z": 6, "refactor": 7}
PAD_IDX = 0
```

All recipes padded to `RECIPE_LEN_MAX = 20`. Each sample carries its actual
length; LSTM uses `pack_padded_sequence` so PAD never affects hidden state.
DDDQN samples L from `RECIPE_LEN_VALID = (8, 9, 12, 13, 14, 15, 20)` per
episode (`DDDQN_TRAIN_L_MODE = "discrete_set"`).

## Eval methods

For each test design at length L:

| Method | What it does |
|---|---|
| `oracle`   | best of N known recipes at length L by ground truth (CSV lookup) |
| `DDDQN+SA` | DDDQN policy emits 5 recipes → SA from each → pick best |
| `resyn`    | ABC built-in (looked up from `abcStats_withmap/`) |
| `resyn2`   | ABC built-in |

`true_power` for `DDDQN+SA` is computed by running the recipe through ABC
(`read_lib nangate45.lib; read <design>.aig; <recipe>; map; print_stats -p`).
`oracle` / `resyn` / `resyn2` are CSV lookups.

## Hyperparameters (current)

```python
SURROGATE_EPOCHS = 80           # 2.7× the original 30
SURROGATE_BATCH  = 128
SURROGATE_LR     = 1e-3

DDDQN_EPISODES   = 3000
DDDQN_BATCH      = 64
DDDQN_LR         = 3e-4
DDDQN_GAMMA      = 0.99

SA_ITERS         = 60_000       # 3× the original 20k
SA_RESTARTS      = 5
EVAL_LENGTHS     = (20,)        # eval only at L=20

AIG_SIZE_CAP     = 150_000      # bytes; bigger AIGs hang DG2
```

## Install

```bash
pip install torch torch-geometric scipy pandas numpy
# DeepGate2 pretrained encoder
git clone https://github.com/Ironprop-Stone/python-deepgate.git /tmp/python-deepgate
pip install /tmp/python-deepgate
# ABC built in WSL at /usr/local/bin/abc; nangate45.lib at data/designs/
```

## Run

```bash
python -m pipeline.run_all              # full pipeline end-to-end
python -m pipeline.run_all --resume     # resume surrogate + DDDQN if compatible
# OR step-by-step:
python -m pipeline.embed_designs        # cache hit if already done
python -m pipeline.train_surrogate      # ~25 min @ 80 epochs
python -m pipeline.train_surrogate --resume
python -m pipeline.train_dddqn_init     # ~60 min
python -m pipeline.train_dddqn_init --resume
python -m pipeline.evaluate             # ~15 min @ 60k SA iters

# Optional REINFORCE fallback (not in run_all):
python -m pipeline.train_rl
```

## Caching behaviour

- `embed_designs.py` is **idempotent and incremental**. Existing entries reused;
  new designs added on the fly; **saves after every successful encode**, so
  Ctrl+C is non-destructive. `--force` re-encodes everything.
- `train_surrogate --resume` continues from `pipeline/checkpoints/surrogate.pt`
   (restores model + optimizer, resumes from `epoch+1`).
- `train_dddqn_init --resume` continues from `pipeline/checkpoints/policy_dddqn.pt`
   (restores online/target/optimizer, resumes from saved episode count so
   epsilon schedule continues at the lower value).
- Resume is **strictly guarded**: if model keys/shapes or core metadata
   (`aig_dim`, `n_ops`, encoder meta, recipe length) mismatch, training aborts
   with a loud error and asks for fresh init.
- `results/<design>.csv` is reset at the start of each `evaluate.py` run.

## Decisions baked in (and where to change them)

| Choice | Where | Why |
|---|---|---|
| Frozen DG2 (no fine-tune) | `encoders.py` | DG2 was pretrained on 68k circuits; fine-tuning on ~30 risks degrading the prior |
| 256-d embedding (mean-pool of hs⊕hf) | `encoders.py` | Simple; can swap to attention pool later |
| Save best surrogate by **Spearman**, not MSE | `train_surrogate.py` | Search cares about ranking, not absolute calibration |
| Discrete length set {8,9,12,13,14,15,20} | `config.py` | Only the lengths we have ground truth for |
| Bi-LSTM + multi-task (area+power) + pairwise rank | `surrogate.py`, `train_surrogate.py` | Bi-LSTM captures forward+backward op deps; multi-task is a free regulariser; pairwise loss directly optimises Spearman |
| LSTM with `pack_padded_sequence` | `surrogate.py` | Cleanest variable-length handling |
| Skip designs >150 KB | `embed_designs.py` | DG2's `top_sort` is O(N³) and hangs |
| Eval only at L=20 | `config.py: EVAL_LENGTHS` | Project's primary length; matches `resyn`/`resyn2` baselines |
| ABC eval per recipe (no batching) | `abc_runner.py` | ~1 s per call; total eval is ~10 min of ABC time |
| **DDDQN as primary RL, REINFORCE optional** | `run_all.py` | DDDQN's PER + dueling + Double-Q address REINFORCE's known weaknesses |
| **Dropped LAHC + parallel tempering from eval** | n/a | Time budget; SA on the new surrogate is sufficient at L=20 |

## Limitations / known issues

1. **AIG size cap.** DG2 cannot encode AIGs above ~10 K gates in tractable time
   (its `top_sort` uses `numpy.isin` in an O(N) loop, making the whole pass
   O(N³)). 9 designs from the dataset are unreachable for this reason —
   notably `div`, `ethernet`, `jpeg`, `hyp`, `dft`, `idft`. Larger circuits
   would likely give bigger absolute power reductions, but require either DG3
   (windowing) or a custom feature encoder.
2. **Surrogate ranking imperfect.** Holdout Spearman varies per design. On
   designs where `resyn2` is already near-optimal (e.g. `ss_pcm`), the search
   has little headroom. On designs where `oracle` is far from `resyn` (e.g.
   `i2c`, `tv80`), the search materially helps.
3. **Beyond-oracle search.** SA can in principle invent recipes outside the
   500 labelled ones that beat oracle, but this only works when the surrogate
   ranks novel recipes correctly. With current ρ values (~0.4) we usually
   land at oracle's level, not beyond it.
