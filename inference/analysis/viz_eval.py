# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Blog-quality plots of the checkpoint evaluation (per-game accuracy / win-rate across epochs).

Reads the `eval_results/*.json` written by `inference.evaluate` / `inference.run_checkpoints` and
renders, into `eval_plots/`:
  1. game × checkpoint accuracy heatmap
  2. per-game accuracy across checkpoints (small-multiples grid)
  3. aggregate accuracy across checkpoints (macro / single-turn / multi-turn / reasoning)
  4. wordle headline: win-rate across checkpoints (Wilson CI bars) + solved-by-round for the best
  5. base → best-checkpoint delta per game (the transfer story)
  6. format discipline: action-parse-rate + think-rate across checkpoints

Run LOCALLY (no vLLM / torch):

    uv run --no-project inference/analysis/viz_eval.py --results eval_results --out eval_plots
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob


def _ckpt_order(label: str) -> tuple[int, int]:
    """Sort key: base first (0), then by epoch number parsed from '...-eN'."""
    if label == "base":
        return (0, 0)
    try:
        return (1, int(label.rsplit("-e", 1)[1]))
    except (IndexError, ValueError):
        return (2, 0)


def _short(label: str) -> str:
    if label == "base":
        return "base"
    return "e" + label.rsplit("-e", 1)[1] if "-e" in label else label


def _load(results_dir: str):
    runs = []
    for path in glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit(f"no result JSONs in {results_dir}")
    runs.sort(key=lambda r: _ckpt_order(r["label"]))
    # game order by game_no (consistent across checkpoints)
    game_no = {}
    for r in runs:
        for g, m in r["games"].items():
            game_no.setdefault(g, m.get("game_no", 99))
    games = sorted(game_no, key=lambda g: game_no[g])
    return runs, games


def _acc(run, game):
    m = run["games"].get(game)
    return m["accuracy"] if m else float("nan")


def _heatmap(runs, games, out):
    import matplotlib.pyplot as plt
    import numpy as np

    mat = np.array([[_acc(r, g) for r in runs] for g in games])
    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(runs), 0.5 + 0.5 * len(games)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(runs)), [_short(r["label"]) for r in runs])
    ax.set_yticks(range(len(games)), games)
    for i in range(len(games)):
        for j in range(len(runs)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)
    ax.set_title("Accuracy / win-rate per game across checkpoints")
    fig.colorbar(im, ax=ax, label="accuracy", fraction=0.046, pad=0.04)
    _save(fig, out, "heatmap_accuracy")


def _small_multiples(runs, games, out):
    import matplotlib.pyplot as plt
    import numpy as np

    xs = range(len(runs))
    xlabels = [_short(r["label"]) for r in runs]
    cols = 4
    rows = (len(games) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.2 * rows), squeeze=False)
    for idx, g in enumerate(games):
        ax = axes[idx // cols][idx % cols]
        ys = [_acc(r, g) for r in runs]
        lo = [r["games"][g]["ci_lo"] if g in r["games"] else np.nan for r in runs]
        hi = [r["games"][g]["ci_hi"] if g in r["games"] else np.nan for r in runs]
        ax.plot(xs, ys, marker="o", color="#3b7dd8")
        ax.fill_between(xs, lo, hi, alpha=0.2, color="#3b7dd8")
        ax.set_title(g, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xticks(list(xs), xlabels, fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(len(games), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Per-game accuracy across checkpoints (shaded = 95% Wilson CI)", y=1.0)
    _save(fig, out, "per_game_accuracy")


def _aggregate(runs, out):
    import matplotlib.pyplot as plt

    xs = range(len(runs))
    xlabels = [_short(r["label"]) for r in runs]
    series = {"macro_accuracy": "macro (all games)", "single_turn_acc": "single-turn",
              "multi_turn_acc": "multi-turn", "reasoning_acc": "reasoning (<think>)"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lbl in series.items():
        ys = [r["aggregate"].get(key) for r in runs]
        ax.plot(xs, ys, marker="o", label=lbl)
    ax.set_xticks(list(xs), xlabels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro-average accuracy")
    ax.set_title("Aggregate accuracy across checkpoints")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out, "aggregate_accuracy")


def _wordle(runs, out):
    import matplotlib.pyplot as plt
    import numpy as np

    if not any("wordle" in r["games"] for r in runs):
        return
    sub = [r for r in runs if "wordle" in r["games"]]
    xs = range(len(sub))
    xlabels = [_short(r["label"]) for r in sub]
    acc = [r["games"]["wordle"]["accuracy"] for r in sub]
    lo = [r["games"]["wordle"]["ci_lo"] for r in sub]
    hi = [r["games"]["wordle"]["ci_hi"] for r in sub]
    yerr = np.array([np.subtract(acc, lo), np.subtract(hi, acc)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.errorbar(xs, acc, yerr=yerr, marker="o", capsize=4, color="#2a9d4a")
    ax1.set_xticks(list(xs), xlabels)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("win rate")
    ax1.set_title("Wordle win rate across checkpoints (95% Wilson CI)")
    ax1.grid(alpha=0.3)

    best = max(sub, key=lambda r: r["games"]["wordle"]["accuracy"])
    sbr = best["games"]["wordle"].get("solved_by_round", {})
    rounds = sorted(int(k) for k in sbr)
    counts = [sbr[str(k)] if str(k) in sbr else sbr.get(k, 0) for k in rounds]
    ax2.bar([str(r) for r in rounds], counts, color="#2a9d4a")
    ax2.set_xlabel("rounds to win")
    ax2.set_ylabel("games won")
    ax2.set_title(f"Wordle solved-by-round ({_short(best['label'])}, best)")
    ax2.grid(alpha=0.3, axis="y")
    _save(fig, out, "wordle_headline")


def _delta(runs, games, out):
    import matplotlib.pyplot as plt
    import numpy as np

    base = next((r for r in runs if r["label"] == "base"), None)
    trained = [r for r in runs if r["label"] != "base"]
    if base is None or not trained:
        return
    best = max(trained, key=lambda r: r["aggregate"].get("macro_accuracy") or 0)
    deltas = [(_acc(best, g) - _acc(base, g)) for g in games]
    order = sorted(range(len(games)), key=lambda i: deltas[i])
    g_sorted = [games[i] for i in order]
    d_sorted = [deltas[i] for i in order]
    colors = ["#c0504d" if d < 0 else "#2a9d4a" for d in d_sorted]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(games) + 1.5))
    ax.barh(range(len(g_sorted)), d_sorted, color=colors)
    ax.set_yticks(range(len(g_sorted)), g_sorted)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ accuracy (best − base)")
    ax.set_title(f"Transfer: {_short(best['label'])} vs base, per game")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, out, "delta_vs_base")


def _format_discipline(runs, games, out):
    import matplotlib.pyplot as plt

    xs = range(len(runs))
    xlabels = [_short(r["label"]) for r in runs]
    reasoning = {"wordle", "anagram", "crossword", "mistakeid"}

    def mean_over(games_subset, key):
        out_vals = []
        for r in runs:
            vals = [r["games"][g][key] for g in games_subset if g in r["games"]]
            out_vals.append(sum(vals) / len(vals) if vals else None)
        return out_vals

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, mean_over(games, "action_parse_rate"), marker="o", label="action parse-rate (all)")
    ax.plot(xs, mean_over([g for g in games if g in reasoning], "think_rate"),
            marker="s", label="<think> rate (reasoning games)")
    ax.set_xticks(list(xs), xlabels)
    ax.set_ylim(0, 1.02)
    ax.set_title("Format discipline across checkpoints")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out, "format_discipline")


def _table(runs, games, out):
    """Write (and print) a paste-ready markdown accuracy table with a Δ(best−base) column."""
    short = [_short(r["label"]) for r in runs]
    base = next((r for r in runs if r["label"] == "base"), None)
    trained = [r for r in runs if r["label"] != "base"]

    def pct(v):
        return "—" if (v != v) else f"{v * 100:.0f}"

    lines = ["| game | " + " | ".join(short) + " | Δ best−base |",
             "|" + "---|" * (len(runs) + 2)]
    for g in games:
        cells = [pct(_acc(r, g)) for r in runs]
        delta = ""
        if base and trained:
            ba = _acc(base, g)
            bestv = max((_acc(r, g) for r in trained), default=float("nan"))
            if ba == ba and bestv == bestv:
                delta = f"{(bestv - ba) * 100:+.0f}"
        lines.append("| " + g + " | " + " | ".join(cells) + " | " + delta + " |")
    for key, label in [("macro_accuracy", "**macro (all)**"), ("single_turn_acc", "single-turn"),
                       ("multi_turn_acc", "multi-turn"), ("reasoning_acc", "reasoning")]:
        cells = [pct(r["aggregate"].get(key) if r["aggregate"].get(key) is not None else float("nan"))
                 for r in runs]
        lines.append("| " + label + " | " + " | ".join(cells) + " |  |")

    n = runs[0].get("n", "?")
    text = ("### Accuracy / win-rate per game (%)\n\n" + "\n".join(lines) +
            f"\n\n_n={n}/game · identical seeded held-out set per checkpoint · "
            f"95% Wilson CI in the result JSONs · Δ = best epoch − base._\n")
    path = os.path.join(out, "results_table.md")
    with open(path, "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"[viz] wrote {path}")


def _save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = os.path.join(out, f"{name}.{ext}")
        fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[viz] wrote {os.path.join(out, name)}.png/.svg")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blog plots from checkpoint eval results.")
    ap.add_argument("--results", default="eval_results")
    ap.add_argument("--out", default="eval_plots")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs, games = _load(args.results)
    print(f"[viz] {len(runs)} checkpoints {[r['label'] for r in runs]}; {len(games)} games")
    _heatmap(runs, games, args.out)
    _small_multiples(runs, games, args.out)
    _aggregate(runs, args.out)
    _wordle(runs, args.out)
    _delta(runs, games, args.out)
    _format_discipline(runs, games, args.out)
    _table(runs, games, args.out)


if __name__ == "__main__":
    main()
