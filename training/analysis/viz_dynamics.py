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

# Heatmap colormap. magma reads as an intuitive "hotter = more" thermal scale for a general
# audience (dark = low, bright = high), while staying perceptually uniform + colorblind-safe.
# Swap to "inferno" / "YlOrRd" / "viridis" here if you prefer.
_CMAP = "magma"

# Qwen3.5-0.8B is a HYBRID model: 3 of every 4 transformer blocks use linear attention, and every
# 4th block (indices 3, 7, 11, …) uses full softmax attention. Those blocks are a *different kind of
# layer*, so they take different-sized optimizer steps — which shows up as a period-4 ripple in the
# per-layer plots. We flag them so that ripple isn't misread as a depth/learning trend. The marked
# layers are i where (i+1) % interval == 0. Set to 0/None for a uniform (non-hybrid) model.
_FULL_ATTN_INTERVAL = 4
_FULL_ATTN_NOTE = "marked = every 4th layer is a different kind of attention layer (not a depth effect)"


def _full_attn_layers(n_layers: int) -> list[int]:
    if not _FULL_ATTN_INTERVAL:
        return []
    return [i for i in range(n_layers) if (i + 1) % _FULL_ATTN_INTERVAL == 0]


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

    # Clip the color scale to the 2nd–98th percentile so a single hot block (e.g. the last block's
    # large post-clip gradnorm) doesn't dominate the colormap and flatten everything else to one hue.
    vmin, vmax = (None, None) if normalize else tuple(np.nanpercentile(mat, [2, 98]))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=_CMAP, vmin=vmin, vmax=vmax,
                   extent=[steps.min(), steps.max(), 0, n_layers])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("layer (0 = first/input → top = last/output)")
    ax.set_title(f"{run_name} — {metric} per {component} block"
                 + (" (per-step normalized)" if normalize else " (color: 2–98 pct)"))
    fig.colorbar(im, ax=ax, label=("fraction of step norm" if normalize else metric),
                 extend=("neither" if normalize else "both"))

    # Flag the every-4th (full-attention) rows with a marker at the left edge + a caption, so the
    # period-4 banding reads as "these layers are a different type", not as a depth trend.
    fa = _full_attn_layers(n_layers)
    if fa:
        x0 = steps.min()
        for layer in fa:
            ax.plot([x0], [layer + 0.5], marker="<", ms=10, color="white", mec="black",
                    mew=0.7, clip_on=False, zorder=5)
        fig.text(0.5, 0.005, "◄ " + _FULL_ATTN_NOTE, ha="center", va="bottom", fontsize=9,
                 color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1) if fa else None)
    path = os.path.join(out_dir,
                        f"{run_name}_{metric}_{component}{'_norm' if normalize else ''}_heatmap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {path}")


def _extras(run, *, metric: str, out_dir: str):
    """Per-step series for the NON-block params: embed / lm_head / final_norm / _total.
    These have very different param counts (embed/lm_head are huge, final_norm tiny), so the y-axis
    is log-scaled — this is a *within-bucket-over-time* view, not a fair cross-bucket magnitude one."""
    import matplotlib.pyplot as plt
    import pandas as pd

    names = ["embed", "lm_head", "final_norm", "_total"]
    keys = [f"{metric}/{n}" for n in names if f"{metric}/{n}" in run.summary.keys()]
    if not keys:
        return
    rows = list(run.scan_history(keys=["_step", *keys]))
    df = pd.DataFrame(rows).dropna(how="all", subset=keys).sort_values("_step")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in keys:
        ax.plot(df["_step"], df[k], marker="o", ms=4, label=k.split("/", 1)[1])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(f"{metric} (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"{run.name} — {metric}: non-block params (embed / lm_head / final_norm / total)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run.name}_{metric}_extras.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {path}")


def _compare(per_run: dict, *, metric: str, component: str, out_dir: str):
    """One CONTINUOUS line per run (mean-over-steps norm at each layer) so the overall depth trend
    stays intact, with the every-4th (full-attention) layers flagged as open rings — those sit off
    the trend because they're a different kind of layer, not because of their depth."""
    import matplotlib.pyplot as plt

    n_layers = max((int(_LAYER.search(k).group(1)) for _, ks in per_run.values()
                    if ks for k in ks), default=-1) + 1
    fa = set(_full_attn_layers(n_layers))

    fig, ax = plt.subplots(figsize=(10, 6))
    for x in sorted(fa):                                   # faint guides on the special layers
        ax.axvline(x, color="0.85", lw=0.8, zorder=0)
    for run_name, (df, keys) in per_run.items():
        if df is None:
            continue
        layers = [int(_LAYER.search(k).group(1)) for k in keys]
        mean = df[keys].to_numpy(dtype=float).mean(axis=0)
        (line,) = ax.plot(layers, mean, marker="o", ms=3, label=run_name)   # continuous trend
        fx = [l for l in layers if l in fa]                                  # flag the dips
        fy = [mean[layers.index(l)] for l in fx]
        ax.scatter(fx, fy, s=80, facecolors="none", edgecolors=line.get_color(),
                   linewidths=1.8, zorder=5)
    if fa:                                                 # one neutral legend entry for the rings
        ax.scatter([], [], s=80, facecolors="none", edgecolors="0.4", linewidths=1.8,
                   label="every 4th layer (different type)")
    ax.set_xlabel("layer (0 = first/input → last/output)")
    ax.set_ylabel(f"mean {metric} over training")
    ax.set_title(f"per-layer {metric} ({component} block) — cross-run comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    if fa:
        fig.text(0.5, 0.005, "○ " + _FULL_ATTN_NOTE, ha="center", va="bottom", fontsize=9,
                 color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
        if not args.normalize:                    # non-block series (embed/lm_head/final_norm/total)
            _extras(run, metric=args.metric, out_dir=args.out)

    _compare(per_run, metric=args.metric, component=args.component, out_dir=args.out)


if __name__ == "__main__":
    main()
