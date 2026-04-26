"""Centralised config for the inverse-recipe-search pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- data sources (ordered by priority for AIG file resolution) ----
DATA_SOURCES = [
    {"name": "len20",     "dir": ROOT / "data"},
    {"name": "len12_15",  "dir": ROOT / "formatted_data(12-15)"},
    {"name": "len8_9",    "dir": ROOT / "formatted_data(8-9)"},
]

PIPE_DIR     = ROOT / "pipeline"
CACHE_DIR    = PIPE_DIR / "cache"
CKPT_DIR     = PIPE_DIR / "checkpoints"
RESULTS_DIR  = ROOT / "results"
ABCSTATS_DIR = ROOT / "abcStats_withmap"

EMBED_PATH        = CACHE_DIR / "aig_embeddings.pt"
NORMS_PATH        = CKPT_DIR  / "design_length_norms.pt"
SURR_PATH         = CKPT_DIR  / "surrogate.pt"
POLICY_PATH       = CKPT_DIR  / "policy.pt"           # REINFORCE (kept; not auto-trained)
POLICY_DDDQN_PATH = CKPT_DIR  / "policy_dddqn.pt"
DRIVER_DDDQN_PATH = CKPT_DIR  / "driver_dddqn.pt"     # method 5, deferred

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---- vocab ----
VOCAB = {
    "refactor -z": 1,
    "balance":     2,
    "rewrite":     3,
    "rewrite -z":  4,
    "resub":       5,
    "resub -z":    6,
    "refactor":    7,
}
PAD_IDX = 0
N_OPS = len(VOCAB)
ID_TO_OP = {v: k for k, v in VOCAB.items()}

# ---- recipe lengths ----
RECIPE_LEN_MAX   = 20
RECIPE_LEN_VALID = (8, 9, 12, 13, 14, 15, 20)
EVAL_LENGTHS     = (20,)            # DG2 can't encode the bigger AIGs in tractable time

# ---- design splits ----
TEST_DESIGNS = ["i2c", "ss_pcm", "usb_phy", "sasc", "spi", "wb_dma", "tv80"]
# DG2's parser hangs on AIGs with O(10k+) gates due to O(N³) top_sort.
# Anything in this list is excluded from both training and evaluation.
SKIP_DESIGNS = [
    "dft", "idft", "hyp", "vga_lcd", "picosoc", "bp_be",
    "div", "ethernet", "jpeg",          # also too slow for DG2 in practice
]

AIG_SIZE_CAP = 150_000   # bytes; bigger AIGs hang DG2's top_sort

# ---- training ----
SEED = 42

SURROGATE_EPOCHS = 30      
SURROGATE_BATCH  = 128
SURROGATE_LR     = 1e-3

# REINFORCE — kept for fallback; not run by default
RL_EPISODES      = 3000
RL_BATCH         = 16
RL_LR            = 3e-4
RL_ENTROPY_COEF  = 0.01

# SA / LAHC
SA_ITERS         = 20000   
SA_T0            = 1.0
SA_T_END         = 0.01
SA_RESTARTS      = 5
TOPK_INIT        = 50
RANDOM_POOL      = 5000
LAHC_HISTORY     = 100

# DDDQN
DDDQN_TRAIN_L_MODE   = "discrete_set"   # {"fixed_20", "sample_6_20", "discrete_set"}
DDDQN_EPISODES       = 3000
DDDQN_BATCH          = 64
DDDQN_LR             = 3e-4
DDDQN_GAMMA          = 0.99
DDDQN_BUFFER_INIT    = 100_000
DDDQN_PER_ALPHA      = 0.6
DDDQN_PER_BETA0      = 0.4
DDDQN_PER_BETA1      = 1.0
DDDQN_PER_EPS        = 1e-6
DDDQN_TARGET_SYNC    = 500
DDDQN_GRAD_CLIP      = 1.0
DDDQN_EPS_START      = 1.0
DDDQN_EPS_END        = 0.05
DDDQN_EPS_DECAY_FRAC = 0.5
