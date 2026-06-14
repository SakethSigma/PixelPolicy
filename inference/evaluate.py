"""Evaluation client — play every game on a fixed held-out test set with a served checkpoint.

Reuses the generic per-game wiring from `distillation/registry.py::GAMES` (each game exposes
`make_agent`, `make_env(target)`, `sample_targets(n, mode, rng)`, `good_status`) and the generic
episode driver `agents/rollout.py::run_eval`. So one loop evaluates all 13 games with no per-game
code: sample N seeded `val` instances → play them against the local vLLM server → score
`final.status == good_status`.

    # against an already-running server (see inference.server)
    uv run --package inference python -m inference.evaluate \
        --label wordle-e3 --games all --n 300 --seed 0 --out eval_results/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from distillation.registry import GAME_NUMBERS, GAMES

from inference.metrics import aggregate, game_metrics

# Locked eval sampling (frozen across checkpoints for fairness) — Qwen3.5-recommended.
EVAL_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "enable_thinking": True}


def games_arg(value: str) -> list[str]:
    """'all' or a comma list → ordered list of valid game names."""
    if value == "all":
        return sorted(GAMES, key=lambda g: GAME_NUMBERS[g])
    names = [g.strip() for g in value.split(",") if g.strip()]
    unknown = [g for g in names if g not in GAMES]
    if unknown:
        raise SystemExit(f"unknown game(s): {unknown}; known: {sorted(GAMES)}")
    return names


def build_backend(base_url: str, model: str, *, max_tokens: int = 2048):
    """OpenAI-compatible backend pointed at the local vLLM server, with the locked eval sampling."""
    from agents.backend import OpenAICompatBackend
    from agents.config import AgentConfig

    cfg = AgentConfig.from_env()
    return OpenAICompatBackend(
        base_url=base_url, model=model, api_key=cfg.api_key,
        max_tokens=max_tokens, top_k=cfg.top_k, presence_penalty=cfg.presence_penalty,
        **EVAL_SAMPLING,
    )


def _print_samples(name: str, trajs: list, k: int) -> None:
    """Dump the first k episodes' raw replies so formatting (<think>/<guess>/<answer>) is eyeballable."""
    print(f"\n--- samples: {name} (first {min(k, len(trajs))}) ---", file=sys.stderr)
    for i, tr in enumerate(trajs[:k]):
        status = getattr(tr.final, "status", "?")
        target = getattr(tr.final, "target", None)
        print(f"[{i + 1}] target={target} status={status} ({len(tr.turns)} turn(s))", file=sys.stderr)
        for j, t in enumerate(tr.turns, 1):
            resp = (t.response or "").strip()
            if len(resp) > 1200:
                resp = resp[:1200] + " …[truncated]"
            print(f"  turn {j}  parsed_action={t.action!r}", file=sys.stderr)
            print("    " + resp.replace("\n", "\n    "), file=sys.stderr)


def evaluate_game(name: str, *, n: int, seed: int, generate, concurrency: int, show: int = 0) -> dict:
    """Play N seeded held-out (`val`) instances of one game and return its metrics."""
    from agents.rollout import run_eval

    spec = GAMES[name]()
    targets = spec.sample_targets(n, "val", random.Random(seed))
    agent = spec.make_agent()
    pairs = [(agent, spec.make_env(t)) for t in targets]
    trajs = run_eval(pairs, generate, concurrency=concurrency)
    if show:
        _print_samples(name, trajs, show)
    m = game_metrics(trajs, spec.good_status)
    m["good_status"] = spec.good_status
    m["game_no"] = GAME_NUMBERS[name]
    return m


def evaluate_all(games: list[str], *, n: int, seed: int, generate, concurrency: int,
                 show: int = 0) -> dict:
    per_game: dict[str, dict] = {}
    for name in games:
        m = evaluate_game(name, n=n, seed=seed, generate=generate, concurrency=concurrency, show=show)
        per_game[name] = m
        print(f"  {name:<12} acc={m['accuracy']:.1%}  ({m['solved']}/{m['n']})  "
              f"[{m['ci_lo']:.1%}, {m['ci_hi']:.1%}]", file=sys.stderr)
    return {"games": per_game, "aggregate": aggregate(per_game)}


def run_and_save(*, label: str, model: str, revision: str | None, base_url: str,
                 games: list[str], n: int, seed: int, concurrency: int, max_tokens: int,
                 out: str, show: int = 0) -> dict:
    """Build the backend, evaluate all games, write `out/<label>.json`, return the result."""
    backend = build_backend(base_url, model, max_tokens=max_tokens)
    print(f"[eval] label={label} model={model} rev={revision} n={n} "
          f"games={len(games)} @ {base_url}", file=sys.stderr)
    result = evaluate_all(games, n=n, seed=seed, generate=backend.generate,
                          concurrency=concurrency, show=show)
    result.update({"label": label, "model": model, "revision": revision, "n": n, "seed": seed,
                   "sampling": {**EVAL_SAMPLING, "max_tokens": max_tokens}})
    Path(out).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out, f"{label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] {label}: macro={result['aggregate']['macro_accuracy']:.1%}  → wrote {path}",
          file=sys.stderr)
    return result


def _main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a served checkpoint across all games.")
    ap.add_argument("--label", required=True, help="checkpoint id for the output file (e.g. wordle-e3).")
    ap.add_argument("--games", type=games_arg, default="all", help="all | comma list.")
    ap.add_argument("--n", type=int, default=300, help="held-out instances per game.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--base-url", default=None, help="default: OPENAI_BASE_URL / .env.")
    ap.add_argument("--model", default=None, help="served model id (default: INFERENCE_MODEL / .env).")
    ap.add_argument("--revision", default=None, help="recorded in the result for provenance.")
    ap.add_argument("--show", type=int, default=0,
                    help="print the first N raw episodes per game (eyeball <think>/<guess> format).")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    from dotenv import load_dotenv
    from agents.config import AgentConfig
    load_dotenv()
    cfg = AgentConfig.from_env()

    run_and_save(label=args.label, model=args.model or cfg.model, revision=args.revision,
                 base_url=args.base_url or cfg.base_url, games=args.games, n=args.n, seed=args.seed,
                 concurrency=args.concurrency, max_tokens=args.max_tokens, out=args.out, show=args.show)


if __name__ == "__main__":
    _main()
