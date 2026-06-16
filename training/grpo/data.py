"""Build the verl training dataset: one row per pinned target word.

verl's ``RLHFDataset`` reads a parquet whose rows carry, at minimum:

* ``prompt``        — the opening chat messages (system + first user turn);
* ``data_source``   — the game name (our reward fn dispatches on it);
* ``reward_model``  — ``{"style": "rule", "ground_truth": <target>}``;
* ``extra_info``    — ``{"target": <target>, "game": <name>, "index": i}``.

The agent loop rebuilds the *full* multi-turn conversation itself (via ``WordleAgent``); the
``prompt`` column only seeds the first turn and satisfies verl's schema. The important payload
is ``extra_info.target`` — the word the loop pins the env to — and ``reward_model.ground_truth``.

Targets are drawn with ``GameSpec.sample_targets(n, "train", rng)`` from the **train** pool only,
which the deterministic sha256 split keeps disjoint from the **val** pool used by
``inference/evaluate.py``. So we never train on an eval word. Diversity within a GRPO step comes
from the ``n`` (group-size) rollouts per target, not from needing every word — a few thousand
distinct train targets is plenty.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

from distillation.registry import GAMES

# The 13-game roster for Arm C (task="all"). Wordle is included and gets the dense reward;
# the rest get sparse solved/not-solved (see reward.compute_score).
ALL_GAMES: list[str] = list(GAMES.keys())


def _opening_messages(agent: Any) -> list[dict]:
    """The genuine first-turn prompt: system + the game's opening user nudge.

    We special-case nothing — every agent exposes ``system_prompt``. For the multi-turn games
    the opening user line is a constant ("Make your first guess." for wordle); for single-turn
    games ``build_messages`` would carry the puzzle, but since the loop rebuilds messages from the
    env anyway, a minimal opener is sufficient to satisfy verl's schema.
    """
    return [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": "Make your first guess."},
    ]


def build_rows(task: str, n: int, seed: int) -> list[dict]:
    """Return the list of dataset rows for ``task`` ("wordle" or "all").

    For "all", ``n`` targets are drawn *per game* (capped at each pool's size), so the total row
    count is up to ``n * len(ALL_GAMES)``. A single shared seed makes the draw reproducible.
    """
    games = ["wordle"] if task == "wordle" else ALL_GAMES
    rng = random.Random(seed)
    rows: list[dict] = []
    idx = 0
    for game in games:
        spec = GAMES[game]()
        agent = spec.make_agent()
        # Don't ask for more distinct targets than the train pool holds.
        want = n
        targets = spec.sample_targets(want, "train", rng)
        opener = _opening_messages(agent)
        for tgt in targets:
            rows.append({
                "data_source": game,
                "prompt": opener,
                "reward_model": {"style": "rule", "ground_truth": tgt},
                "extra_info": {"target": tgt, "game": game, "index": idx},
            })
            idx += 1
    return rows


def build_dataset(task: str, n: int, seed: int, out_path: str) -> int:
    """Write the parquet verl reads; return the number of rows written.

    Requires ``pandas`` + ``pyarrow`` (present in the verl venv). Kept import-local so the rest of
    this module (``build_rows``) is testable in the bare workspace venv with no heavy deps.
    """
    import pandas as pd

    rows = build_rows(task, n, seed)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"[data] wrote {len(rows)} rows ({task}) -> {out_path}")
    return len(rows)


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the verl GRPO training parquet.")
    ap.add_argument("--task", choices=["wordle", "all"], default="wordle")
    ap.add_argument("--n", type=int, default=2000, help="distinct train targets (per game for --task all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .parquet path")
    args = ap.parse_args(argv)
    build_dataset(args.task, args.n, args.seed, args.out)


if __name__ == "__main__":
    _main()
