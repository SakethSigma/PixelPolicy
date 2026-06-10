"""Play N Wordle games with the teacher, dump per-turn (system, input, output) to JSON,
and price the run from real token usage — a live-call cost estimate for the dataset.

    uv run --package distillation python -m distillation.cost_probe --episodes 10

Token usage isn't kept in the Trajectory, so we grab it off each Completion via the
Observer.on_step(turn, completion) hook (completion.raw["usage"]). Episodes run
sequentially so usage maps cleanly to turns (no concurrency interleaving).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from agents.backend import AnthropicBackend
from agents.rollout import run_episode
from agents.wordle.agent import WordleAgent, WordleEnv
from games.wordle.client import LocalWordleClient
from games.wordle.game import WordBank

# Sonnet 4.6 list price, USD per 1M tokens. Cache read = 0.1x input; 5-min cache write = 1.25x.
PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-8":   {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
}


class UsageRecorder:
    """Captures each turn's token usage (only the Observer sees the Completion)."""

    def __init__(self) -> None:
        self.usages: list[dict] = []

    def on_start(self, state) -> None:  # noqa: D401
        pass

    def on_step(self, turn, completion) -> None:
        self.usages.append((completion.raw or {}).get("usage", {}) or {})

    def on_end(self, state) -> None:
        pass


def price(totals: dict, model: str) -> dict:
    p = PRICING[model]
    cost = (
        totals["input_tokens"] * p["input"]
        + totals["cache_read_input_tokens"] * p["cache_read"]
        + totals["cache_creation_input_tokens"] * p["cache_write"]
        + totals["output_tokens"] * p["output"]
    ) / 1_000_000
    return {**totals, "total_usd": round(cost, 4)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cost probe: play N games, dump turns + price the run.")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--effort", default="high", help="match what real generation will use")
    ap.add_argument("--mode", choices=["train", "val"], default="train")
    ap.add_argument("--words", default=None,
                    help="comma-separated targets to pin (one per game); overrides random sampling")
    ap.add_argument("--out", default="distillation/data/cost_probe.json")
    args = ap.parse_args(argv)

    words = [w.strip() for w in args.words.split(",")] if args.words else None
    if words:
        args.episodes = len(words)  # one game per pinned word, apples-to-apples reruns

    load_dotenv()
    backend = AnthropicBackend(model=args.model, effort=args.effort, max_tokens=4096)
    # Fail fast on a stalled connection (the $$ wedge): 90s/request, NO retries
    # (a retried request that the server already started is billed twice).
    backend._client = backend._client.with_options(timeout=90.0, max_retries=0)
    bank = WordBank()
    agent = WordleAgent()

    totals = {k: 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    episodes = []

    for i in range(args.episodes):
        env = WordleEnv(LocalWordleClient(bank))
        env.reset(mode=args.mode, word=words[i] if words else None)  # pin target if --words given
        rec = UsageRecorder()
        traj = run_episode(agent, env, backend.generate, rec)

        turns = []
        for turn, usage in zip(traj.turns, rec.usages):
            for k in totals:
                totals[k] += usage.get(k, 0) or 0
            turns.append({
                "round": len(turns) + 1,
                "input": turn.messages[1:],   # everything after the system turn (the live prompt)
                "output": turn.response,      # <think>…</think>\n<guess>…</guess>
                "action": turn.action,
                "usage": usage,
            })
        status = getattr(traj.final, "status", None)
        target = getattr(traj.final, "target", None)
        episodes.append({"episode": i, "target": target, "status": status, "turns": turns})
        print(f"game {i + 1}/{args.episodes}: {target} -> {status} in {len(turns)} rounds")

    cost = price(totals, args.model)
    n = args.episodes
    summary = {
        "model": args.model,
        "effort": args.effort,
        "episodes": n,
        "system": agent.system_prompt,        # constant across turns — stored once
        "cost": cost,
        "per_game_usd": round(cost["total_usd"] / n, 4) if n else 0.0,
        "projected_500_games_usd": round(cost["total_usd"] / n * 500, 2) if n else 0.0,
        "games": episodes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n— cost —")
    print(f"  input={cost['input_tokens']}  output={cost['output_tokens']}  "
          f"cache_read={cost['cache_read_input_tokens']}  cache_write={cost['cache_creation_input_tokens']}")
    print(f"  total=${cost['total_usd']}   per game=${summary['per_game_usd']}   "
          f"x500 games≈${summary['projected_500_games_usd']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
