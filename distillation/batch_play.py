"""Lockstep batch play — game-agnostic teacher rollouts via the Anthropic Batch API (~50% off).

Plays N games of a registered game in lockstep: batch all still-active games' current-round
prompts into ONE batch, apply each env's feedback locally, then batch the next round for
whoever is still playing. N games => <= max_rounds batches (not max_rounds * N live calls).

The game itself is wired in `registry.py` (a GameSpec). This driver knows nothing about any
specific game — it only calls agent.build_messages/parse_action/system_prompt and env.step/state.

    uv run --package distillation python -u -m distillation.batch_play --game wordle --episodes 1 --effort low

Resilient to network loss: a batch is durable server-side, so we checkpoint after every round
(and persist the in-flight batch id), and `--resume <checkpoint>` replays the saved guesses to
restore each env, re-attaches to any in-flight batch (no re-submit), and continues.

Writes two formats:
  - raw  (--out-raw):  full Claude response + per-turn usage, for inspection/cost.
  - sft  (--out-sft):  one JSONL line per move {system, messages, completion, completion_no_think, has_think}.

See distillation/batch_play.md for the full design.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from agents.backend import AnthropicBackend
from agents.base import Turn
from distillation.registry import GAMES

# List price, USD per 1M tokens. Cache read = 0.1x input; 5-min cache write = 1.25x.
PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-8":   {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
}
BATCH_DISCOUNT = 0.5  # Anthropic Message Batches API is 50% of standard price.

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop the <think>…</think> block, leaving just the final answer (e.g. <guess>word</guess>)."""
    return _THINK.sub("", text)


def price(totals: dict, model: str, *, discount: float) -> dict:
    p = PRICING[model]
    raw = (
        totals["input_tokens"] * p["input"]
        + totals["cache_read_input_tokens"] * p["cache_read"]
        + totals["cache_creation_input_tokens"] * p["cache_write"]
        + totals["output_tokens"] * p["output"]
    ) / 1_000_000
    return {**totals, "live_equivalent_usd": round(raw, 4), "total_usd": round(raw * discount, 4)}


def apply_round(games: list[dict], active_idx: list[int], prompts: list[list[dict]], completions, agent) -> None:
    """Step each active game with its completion; append the Turn + raw row; update status."""
    for i, prompt, comp in zip(active_idx, prompts, completions):
        g = games[i]
        action = agent.parse_action(comp.text)
        state = g["env"].step(action)
        g["history"].append(Turn(messages=prompt, response=comp.text, action=action, state=state))
        g["turns"].append({
            "round": len(g["turns"]) + 1,
            "prompt": prompt,            # full inference prompt (system + think-stripped history)
            "input": prompt[1:],         # everything after the system turn
            "output": comp.text,
            "action": action,
            "usage": (comp.raw or {}).get("usage", {}) or {},
            "raw": comp.raw,             # full Claude response dump
        })
        g["status"] = getattr(state, "status", "in_progress")


def write_checkpoint(path: Path, args, round_idx: int, games: list[dict], in_flight) -> None:
    """Persist enough to resume: run config, next round, any in-flight batch, and per-game turns."""
    doc = {
        "args": {"game": args.game, "model": args.model, "effort": args.effort, "mode": args.mode,
                 "seed": args.seed, "poll": args.poll, "out_raw": args.out_raw, "out_sft": args.out_sft},
        "round": round_idx,
        "in_flight": in_flight,  # {"batch_id": str, "active": [idx]} or None
        "games": [{"target": g["target"], "status": g["status"], "turns": g["turns"]} for g in games],
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def restore_games(ckpt: dict, spec) -> list[dict]:
    """Rebuild games from a checkpoint: make each env for its target and REPLAY saved guesses
    (deterministic, local, free) to restore env state + the think-stripped history."""
    games = []
    for sg in ckpt["games"]:
        env = spec.make_env(sg["target"])
        history = []
        for row in sg["turns"]:
            state = env.step(row["action"])  # replay → restores env to the exact recorded state
            history.append(Turn(messages=row["prompt"], response=row["output"], action=row["action"], state=state))
        games.append({
            "env": env,
            "history": history,
            "turns": sg["turns"],
            "target": sg["target"],
            "status": getattr(env.state(), "status", sg["status"]),
        })
    return games


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Lockstep batch play: run N games of a registered game, dump raw+sft, price the run.")
    ap.add_argument("--game", default="wordle", choices=sorted(GAMES), help="which registered game to play")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--effort", default="low", help="adaptive-thinking depth: low|medium|high")
    ap.add_argument("--mode", choices=["train", "val"], default="train")
    ap.add_argument("--seed", type=int, default=0, help="seed for random target sampling (reproducible)")
    ap.add_argument("--words", default=None, help="comma-separated targets to pin (overrides sampling)")
    ap.add_argument("--targets-file", default=None, help="file with one target per line (overrides sampling)")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between batch status checks")
    ap.add_argument("--out-raw", default="distillation/data/batch_raw.json")
    ap.add_argument("--out-sft", default="distillation/data/batch_sft.jsonl")
    ap.add_argument("--checkpoint", default="distillation/data/batch_state.json")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args(argv)

    load_dotenv()
    ckpt_path = Path(args.checkpoint)

    # ---- setup: fresh or resumed ----
    if args.resume:
        ckpt = json.loads(Path(args.resume).read_text())
        a = ckpt["args"]  # restore run config so cost/outputs stay self-consistent
        args.game, args.model, args.effort, args.mode = a["game"], a["model"], a["effort"], a["mode"]
        args.seed, args.out_raw, args.out_sft = a["seed"], a["out_raw"], a["out_sft"]
        spec = GAMES[args.game]()
        games = restore_games(ckpt, spec)
        start_round = ckpt["round"]
        in_flight = ckpt.get("in_flight")
        print(f"resumed {args.game}: {len(games)} games, round {start_round}, in_flight={bool(in_flight)}")
    else:
        spec = GAMES[args.game]()
        if args.words:
            targets = [w.strip() for w in args.words.split(",")]
        elif args.targets_file:
            targets = [ln.strip() for ln in Path(args.targets_file).read_text().splitlines() if ln.strip()]
        else:
            targets = spec.sample_targets(args.episodes, args.mode, random.Random(args.seed))
        games = [{"env": spec.make_env(t), "history": [], "turns": [], "target": t, "status": "in_progress"}
                 for t in targets]
        start_round, in_flight = 0, None
        write_checkpoint(ckpt_path, args, 0, games, None)

    agent = spec.make_agent()
    backend = AnthropicBackend(model=args.model, effort=args.effort, max_tokens=4096)
    # Fail fast on a stalled connection; NO auto-retry on the create POST (would double-bill).
    # The poll/results GETs ARE retried internally by batch_generate (idempotent).
    backend._client = backend._client.with_options(timeout=90.0, max_retries=0)

    # ---- if we crashed mid-round, re-attach to the in-flight batch (no re-submit) ----
    if in_flight:
        active_idx = in_flight["active"]
        prompts = [agent.build_messages(games[i]["env"].state(), games[i]["history"]) for i in active_idx]
        completions = backend.batch_generate(prompts, poll_interval=args.poll, resume_batch_id=in_flight["batch_id"])
        apply_round(games, active_idx, prompts, completions, agent)
        start_round += 1
        write_checkpoint(ckpt_path, args, start_round, games, None)
        print(f"re-attached batch {in_flight['batch_id']} -> applied round {ckpt['round'] + 1}")

    # ---- lockstep rounds: one batch per round across all still-active games ----
    for r in range(start_round, spec.max_rounds):
        active_idx = [i for i, g in enumerate(games) if g["status"] == "in_progress"]
        if not active_idx:
            break
        prompts = [agent.build_messages(games[i]["env"].state(), games[i]["history"]) for i in active_idx]

        def _on_created(bid, r=r, active_idx=active_idx):
            # Persist the batch id the instant it exists, BEFORE the failure-prone poll.
            write_checkpoint(ckpt_path, args, r, games, {"batch_id": bid, "active": active_idx})

        completions = backend.batch_generate(prompts, poll_interval=args.poll, on_created=_on_created)
        apply_round(games, active_idx, prompts, completions, agent)
        write_checkpoint(ckpt_path, args, r + 1, games, None)  # round done

        counts = Counter(g["status"] for g in games)
        breakdown = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"round {r + 1}: batched {len(active_idx)} games -> {breakdown}")

    # ---- aggregate cost + write outputs ----
    totals = {k: 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    for g in games:
        for t in g["turns"]:
            for k in totals:
                totals[k] += t["usage"].get(k, 0) or 0
    cost = price(totals, args.model, discount=BATCH_DISCOUNT)
    n = len(games)

    raw_doc = {
        "game": args.game, "model": args.model, "effort": args.effort, "batch": True,
        "system": agent.system_prompt, "cost": cost,
        "per_game_usd": round(cost["total_usd"] / n, 4) if n else 0.0,
        "projected_500_games_usd": round(cost["total_usd"] / n * 500, 2) if n else 0.0,
        "games": [{"episode": i, "target": g["target"], "status": g["status"], "turns": g["turns"]}
                  for i, g in enumerate(games)],
    }
    out_raw = Path(args.out_raw)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_raw.write_text(json.dumps(raw_doc, indent=2, ensure_ascii=False))

    out_sft = Path(args.out_sft)
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with out_sft.open("w") as f:
        for i, g in enumerate(games):
            for t in g["turns"]:
                f.write(json.dumps({
                    "game": i, "round": t["round"], "target": g["target"],
                    "system": agent.system_prompt,
                    "messages": t["prompt"],                          # byte-identical inference prompt
                    "completion": t["output"],                        # full <think>…</think><answer>
                    "completion_no_think": strip_think(t["output"]),  # just the final answer
                    "has_think": "<think>" in t["output"],            # flag: did the teacher reason here?
                }, ensure_ascii=False) + "\n")

    print(f"\n— {args.game} | {args.model} | effort={args.effort} | {n} games —")
    print("  outcomes:", dict(Counter(g["status"] for g in games)))
    print(f"  cost (batch, 50% off): input={cost['input_tokens']} output={cost['output_tokens']} "
          f"cache_read={cost['cache_read_input_tokens']} cache_write={cost['cache_creation_input_tokens']}")
    print(f"  batch total=${cost['total_usd']}  (live-equivalent ${cost['live_equivalent_usd']})")
    print(f"  per game=${raw_doc['per_game_usd']}   x500 games≈${raw_doc['projected_500_games_usd']}")
    print(f"  wrote {out_raw} and {out_sft}")


if __name__ == "__main__":
    main()
