"""
abc_runner.py — run a recipe through ABC + Nangate45 + map, get true power,
log to results/<design>.csv.

Matches the flow in standard_recipie_runner.py:
    cd data/designs
    abc -c "read_lib nangate45.lib; read <design>.aig; <recipe>; map; print_stats -p"
    parse "Power = X" (case-insensitive) from output.

Public API:
    abc_power(design, recipe) -> float
    log_result(design, recipe, pred_power, method="ours") -> dict
        appends row to results/<design>.csv: method, recipe, pred_power, true_power

Requires WSL with abc on $PATH.
"""
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESIGNS_DIR = ROOT / "data" / "designs"
RESULTS_DIR = ROOT / "results"

_POWER_RE = re.compile(r"Power\s*=\s*([0-9\.eE\-\+]+)", re.IGNORECASE)


def _wslpath(win_path):
    """Convert Windows path to /mnt/<drive>/... form."""
    s = str(win_path)
    if len(s) > 2 and s[1:3] == ":\\":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s


def abc_power(design: str, recipe) -> float:
    """Run a recipe through ABC + Nangate45 + map; return Power.

    Args:
        design: design name (without .aig extension), e.g. "i2c".
        recipe: iterable of ABC op strings, e.g. ["balance","rewrite",...].
                Length is unconstrained.

    Raises:
        RuntimeError if ABC's output doesn't contain a Power= line.
    """
    designs_wsl = _wslpath(DESIGNS_DIR)
    abc_cmd = (
        f"read_lib nangate45.lib; "
        f"read {design}.aig; "
        f"{'; '.join(recipe)}; "
        f"map; "
        f"print_stats -p"
    )
    bash_cmd = f"cd '{designs_wsl}' && abc -c '{abc_cmd}'"
    r = subprocess.run(
        ["wsl", "bash", "-c", bash_cmd],
        capture_output=True, text=True, timeout=300,
    )
    out = r.stdout + r.stderr
    m = _POWER_RE.search(out)
    if m is None:
        raise RuntimeError(
            f"[abc_runner] no 'Power=' in ABC output for {design}.\n"
            f"-- last 500 chars of output --\n{out[-500:]}"
        )
    return float(m.group(1))


def log_result(design: str, recipe, pred_power: float,
               method: str = "ours") -> dict:
    """Run ABC for true_power, append a row to results/<design>.csv.

    CSV columns: method, recipe (semicolon-joined), pred_power, true_power.
    """
    true_power = abc_power(design, recipe)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{design}.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["method", "recipe", "pred_power", "true_power"])
        w.writerow([
            method,
            ";".join(recipe),
            f"{pred_power:.4f}",
            f"{true_power:.4f}",
        ])

    return {
        "design": design,
        "method": method,
        "recipe": list(recipe),
        "pred_power": float(pred_power),
        "true_power": true_power,
    }


if __name__ == "__main__":
    # smoke test: python -m pipeline.abc_runner i2c
    import sys
    design = sys.argv[1] if len(sys.argv) > 1 else "i2c"
    recipe = ["balance", "rewrite", "refactor", "rewrite -z"]
    print(f"[abc_runner] design={design}, recipe={recipe}")
    p = abc_power(design, recipe)
    print(f"[abc_runner] true power = {p}")
