# /// script
# requires-python = ">=3.11"
# dependencies = ["wandb", "pandas", "matplotlib", "numpy"]
# ///
"""Pull per-layer grad/update norms from wandb and render layer×step heatmaps + cross-run compare.

Dependencies are declared inline (PEP 723) so `uv run` installs them in an ephemeral env — no
`--with` flags, no torch. Run LOCALLY (needs your WANDB_API_KEY / `wandb login`):

    uv run --no-project training/analysis/viz_dynamics.py \
      --entity saketh-chervu-personal --project pixelpolicy-sft \
      --runs wordle --metric gradnorm --component layer --out ./dynamics_plots

The trainer logs `gradnorm/layer_NN` + `gradnorm/attn_NN` + `gradnorm/mlp_NN` (and updnorm/*, plus
embed / lm_head / final_norm / _total) every `--gradlog-steps`. In the wandb UI that's dozens of lines
— unreadable. This draws, per run:
  - a **layer × step heatmap** (where updates concentrate, and how it moves across training), and
  - a **cross-run per-layer profile** (mean over steps, one line per run — wordle vs full vs curriculum).

`--component`: `layer` = whole block · `attn` = self-attention only · `mlp` = feed-forward only.
(attn/mlp require a run trained AFTER the attn/mlp-split logging was added; older runs only have `layer`.)
Layer index: `layer_00` = FIRST block (input side); highest = LAST (output side, near lm_head).
"""

from __future__ import annotations

import argparse
import os
import re

_LAYER = re.compile(r"_(\d+)$")


def _layer_keys(run, metric: str, component: str) -> list[str]:
    keys = [k for k in run.summary.keys() if k.startswith(f"{metric}/{component}_")
            and _LAYER.search(k)]
    return sorted(keys, key=lambda k: int(_LAYER.search(k).group(1)))


def _fetch(run, metric: str, component: str):
    import pandas as pd

    keys = _layer_keys(run, metric, component)
    if not keys:
        return None, []
    rows = list(run.scan_history(keys=["_step", *keys]))
    df = pd.DataFrame(rows).dropna(how="all", subset=keys).sort_values("_step")
    return df, keys


def _heatmap(df, keys, *, run_name: str, metric: str, component: str, normalize: bool, out_dir: str):
    import matplotlib.pyplot as plt
    import numpy as np

    steps = df["_step"].to_numpy()
    mat = df[keys].to_numpy(dtype=float).T              # (n_layers, n_steps)
    n_layers = mat.shape[0]
    if normalize:                                        # per-step column → fraction of that step's norm
        col = mat.sum(axis=0, keepdims=True)
        mat = np.divide(mat, col, out=np.zeros_like(mat), where=col > 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                   extent=[steps.min(), steps.max(), 0, n_layers])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("layer (0 = first/input → top = last/output)")
    ax.set_title(f"{run_name} — {metric} per {component} block"
                 + (" (per-step normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, label=("fraction of step norm" if normalize else metric))
    fig.tight_layout()
    path = os.path.join(out_dir,
                        f"{run_name}_{metric}_{component}{'_norm' if normalize else ''}_heatmap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {path}")


def _compare(per_run: dict, *, metric: str, component: str, out_dir: str):
    """One line per run: mean-over-steps norm at each layer → where each variant concentrates updates."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, (df, keys) in per_run.items():
        if df is None:
            continue
        layers = [int(_LAYER.search(k).group(1)) for k in keys]
        mean = df[keys].to_numpy(dtype=float).mean(axis=0)
        ax.plot(layers, mean, marker="o", ms=3, label=run_name)
    ax.set_xlabel("layer (0 = first/input → last/output)")
    ax.set_ylabel(f"mean {metric} over training")
    ax.set_title(f"per-layer {metric} ({component} block) — cross-run comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"compare_{metric}_{component}_perlayer.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize per-layer grad/update norms from wandb.")
    ap.add_argument("--entity", required=True, help="wandb entity (e.g. saketh-chervu-personal).")
    ap.add_argument("--project", default="pixelpolicy-sft")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run display names (default: all runs in the project).")
    ap.add_argument("--metric", choices=["updnorm", "gradnorm"], default="updnorm")
    ap.add_argument("--component", choices=["layer", "attn", "mlp"], default="layer",
                    help="layer = whole block; attn = self-attention only; mlp = feed-forward only.")
    ap.add_argument("--normalize", action="store_true",
                    help="per-step column-normalize the heatmap (show relative layer distribution).")
    ap.add_argument("--out", default="./dynamics_plots")
    args = ap.parse_args()

    import wandb

    os.makedirs(args.out, exist_ok=True)
    api = wandb.Api()
    all_runs = list(api.runs(f"{args.entity}/{args.project}"))
    if args.runs:
        wanted = set(args.runs)
        runs = [r for r in all_runs if r.name in wanted or r.id in wanted]
        missing = wanted - {r.name for r in runs} - {r.id for r in runs}
        if missing:
            print(f"[viz] WARNING: not found: {sorted(missing)}; "
                  f"available: {sorted(r.name for r in all_runs)}")
    else:
        runs = all_runs
    if not runs:
        raise SystemExit("no matching runs")

    per_run = {}
    for run in runs:
        df, keys = _fetch(run, args.metric, args.component)
        per_run[run.name] = (df, keys)
        if df is None or df.empty:
            print(f"[viz] {run.name}: no {args.metric}/{args.component}_* data "
                  f"(attn/mlp need a run trained after the split was added; or wordle is tiny — "
                  f"lower --gradlog-steps)")
            continue
        print(f"[viz] {run.name}: {len(df)} logged points, {len(keys)} {args.component} blocks")
        _heatmap(df, keys, run_name=run.name, metric=args.metric, component=args.component,
                 normalize=args.normalize, out_dir=args.out)

    _compare(per_run, metric=args.metric, component=args.component, out_dir=args.out)


if __name__ == "__main__":
    main()
