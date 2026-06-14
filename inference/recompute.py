"""Recompute eval metrics from stored raw predictions — NO re-inference.

The eval harness persists raw per-episode predictions to `eval_results/raw/<label>/<game>.jsonl`
(target, outcome, and every turn's raw reply + parsed action). This reads them back and recomputes
the per-game + aggregate metrics, so a NEW metric (add it to `inference/metrics.py`, or compute it
here over the records) can be derived **without paying for inference again**.

    # recompute every label under the raw dir:
    uv run --package inference python -m inference.recompute --raw eval_results/raw --out eval_results
    # or a single label:
    uv run --package inference python -m inference.recompute --raw eval_results/raw/wordle-e4 --out eval_results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from types import SimpleNamespace as NS

from inference.metrics import aggregate, game_metrics


def _load_game(jsonl: str):
    """Reconstruct lightweight trajectory objects (enough for metrics) from a game's JSONL."""
    trajs, good, game_no = [], None, None
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            good, game_no = r["good_status"], r["game_no"]
            turns = [NS(response=t.get("response"), action=t.get("action")) for t in r["turns"]]
            trajs.append(NS(turns=turns, final=NS(status=r["status"])))
    return trajs, good, game_no


def recompute_label(label_dir: str) -> dict:
    per_game = {}
    for jsonl in sorted(glob.glob(os.path.join(label_dir, "*.jsonl"))):
        name = os.path.splitext(os.path.basename(jsonl))[0]
        trajs, good, game_no = _load_game(jsonl)
        if not trajs:
            continue
        m = game_metrics(trajs, good)
        m["good_status"], m["game_no"] = good, game_no
        per_game[name] = m
    return {"games": per_game, "aggregate": aggregate(per_game)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute eval metrics from stored raw predictions.")
    ap.add_argument("--raw", required=True, help="eval_results/raw (all labels) OR a single label dir.")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    # A label dir contains *.jsonl; a parent contains <label>/ dirs.
    if glob.glob(os.path.join(args.raw, "*.jsonl")):
        label_dirs = [args.raw]
    else:
        label_dirs = [d for d in sorted(glob.glob(os.path.join(args.raw, "*"))) if os.path.isdir(d)]
    if not label_dirs:
        raise SystemExit(f"no raw label dirs / jsonl under {args.raw}")

    os.makedirs(args.out, exist_ok=True)
    for ld in label_dirs:
        label = os.path.basename(ld.rstrip("/"))
        res = recompute_label(ld)
        res.update({"label": label, "recomputed_from_raw": True})
        path = os.path.join(args.out, f"{label}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        macro = res["aggregate"]["macro_accuracy"]
        print(f"[recompute] {label}: macro={macro:.1%} ({len(res['games'])} games) → {path}")


if __name__ == "__main__":
    main()
