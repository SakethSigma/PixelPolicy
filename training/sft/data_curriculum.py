"""LOADER #2 — the curriculum data loader (full set, difficulty-aware ordering).

Three selectable strategies (`--strategy`). See `training/CURRICULUM_NOTES.md` for the research and
hypotheses behind them; the short version:

- **widening** (default, recommended) — *competence-widening*: hard games are *introduced*
  progressively (stage 0 from the start, stage 1 after ~30%, stage 2 after ~55%), but once a game
  is introduced it stays in the shuffled mix — NOT a hard block. The 3 chain-of-thought reasoning
  games (anagram, crossword, mistakeid) are kept available **throughout** so that fragile skill is
  never starved, and a small **replay** slice of easy+reasoning rows is spliced into the tail.
  This is the arm most likely to beat a plain shuffle without eroding reasoning.
- **sorted** — strict easy→hard sort (the literal "easy first" curriculum). Kept as a *cautionary
  baseline arm*: the literature says this is the order most likely to forget reasoning over 4
  epochs (each epoch restarts on trivial data). Trained with a no-shuffle SequentialSampler.
- **weighted** — no hard ordering; harder rows are mildly oversampled, then the whole set is
  shuffled (normal sampler). A soft alternative to a hard schedule.

`widening` and `sorted` produce an intentional ORDER → the trainer iterates them with a
SequentialSampler (see `ORDERED_STRATEGIES`). `weighted` is shuffled like the flat loader.

Dry run (no model loaded):

    uv run --package training python -m training.sft.data_curriculum --strategy widening --dry-run
"""

from __future__ import annotations

import argparse
import random

from training.sft.format import GAME_NO, REASONING_GAMES, STAGE, build_example
from training.sft.data_flat import DEFAULT_MODEL, DEFAULT_REPO, load_valid, summarize, to_examples

# Strategies whose row ORDER is meaningful → train with a SequentialSampler (no reshuffle).
ORDERED_STRATEGIES = frozenset({"widening", "sorted"})

# Fraction of the stream at which each difficulty stage becomes eligible (widening).
INTRO = {0: 0.0, 1: 0.30, 2: 0.55}


def difficulty_score(row: dict) -> tuple[int, int, int]:
    """Sort key for `sorted` (lower = easier): (game stage, round, completion length)."""
    return (STAGE[GAME_NO[row["game_name"]]], int(row["round"]),
            len(row["completion_no_think"]))


def _columns(raw):
    """Pull the per-row signals we order on into memory (fast for ~90k rows)."""
    names = raw["game_name"]
    rounds = raw["round"]
    comps = raw["completion_no_think"]
    stages = [STAGE[GAME_NO[n]] for n in names]
    lengths = [len(c) for c in comps]
    is_reason = [n in REASONING_GAMES for n in names]
    return names, rounds, stages, lengths, is_reason


def _order_indices(raw, strategy: str, *, reasoning_throughout: bool = True,
                   replay_frac: float = 0.03, seed: int = 0) -> list[int]:
    """Return the list of row indices (into `raw`) in curriculum order (may contain repeats)."""
    names, rounds, stages, lengths, is_reason = _columns(raw)
    n = len(raw)

    if strategy == "sorted":
        return sorted(range(n), key=lambda i: (stages[i], int(rounds[i]), lengths[i]))

    if strategy == "widening":
        rng = random.Random(seed)
        keyed: list[tuple[float, int]] = []
        for i in range(n):
            intro = 0.0 if (reasoning_throughout and is_reason[i]) else INTRO[stages[i]]
            keyed.append((max(intro, rng.random()), i))
        keyed.sort(key=lambda t: t[0])
        order = [i for _, i in keyed]
        if replay_frac > 0:
            # Splice a small replay slice of easy+reasoning rows into the back half, so they
            # are revisited late in the epoch (cross-epoch replay is automatic — order repeats).
            pool = [i for i in range(n) if stages[i] == 0 or is_reason[i]]
            rng.shuffle(pool)
            replay = pool[: int(replay_frac * n)]
            cut = len(order) // 2
            tail = order[cut:] + replay
            rng.shuffle(tail)
            order = order[:cut] + tail
        return order

    if strategy == "weighted":
        # Mild oversample of harder rows (by stage), then shuffle. Coarse on purpose — we have no
        # model-loss signal at load time; see CURRICULUM_NOTES.md for the "target the frontier" caveat.
        rng = random.Random(seed)
        order: list[int] = []
        for i in range(n):
            reps = 1 + (1 if rng.random() < 0.5 * stages[i] else 0)  # stage 0→1x, 1→~1.5x, 2→~2x
            order.extend([i] * reps)
        rng.shuffle(order)
        return order

    raise ValueError(f"unknown curriculum strategy: {strategy!r} "
                     f"(choose from widening, sorted, weighted)")


def load_curriculum(repo_id: str = DEFAULT_REPO, *, split: str = "train",
                    games: list[str] | None = None, tokenizer, strategy: str = "widening",
                    reasoning_throughout: bool = True, replay_frac: float = 0.03,
                    num_proc: int = 4, seed: int = 0):
    """Load the full valid set and return a prompt/completion Dataset in curriculum order."""
    raw = load_valid(repo_id, split=split, games=games)
    order = _order_indices(raw, strategy, reasoning_throughout=reasoning_throughout,
                           replay_frac=replay_frac, seed=seed)
    ordered = raw.select(order)
    return to_examples(ordered, tokenizer, num_proc=num_proc)


def _order_profile(raw, order: list[int], *, n_buckets: int = 10) -> None:
    """Print the stage / reasoning composition across the ordered stream (visualize the curriculum)."""
    names, _, stages, _, is_reason = _columns(raw)
    print(f"\norder profile over {len(order)} positions ({n_buckets} buckets, % of bucket):")
    print(f"  {'bucket':<8}{'stage0':>8}{'stage1':>8}{'stage2':>8}{'reason':>8}")
    size = max(1, len(order) // n_buckets)
    for b in range(n_buckets):
        chunk = order[b * size: (b + 1) * size] if b < n_buckets - 1 else order[b * size:]
        if not chunk:
            continue
        s = [0, 0, 0]
        r = 0
        for i in chunk:
            s[stages[i]] += 1
            r += is_reason[i]
        tot = len(chunk)
        print(f"  {b:<8}{100*s[0]/tot:>7.0f}%{100*s[1]/tot:>7.0f}%"
              f"{100*s[2]/tot:>7.0f}%{100*r/tot:>7.0f}%")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Curriculum SFT data loader / dry run.")
    ap.add_argument("--strategy", choices=sorted(ORDERED_STRATEGIES | {"weighted"}),
                    default="widening")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-frac", type=float, default=0.03)
    ap.add_argument("--no-reasoning-throughout", action="store_true",
                    help="gate reasoning games to their stage instead of keeping them throughout.")
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--max-show", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = load_valid(args.repo_id, split=args.split, games=None)
    order = _order_indices(raw, args.strategy,
                           reasoning_throughout=not args.no_reasoning_throughout,
                           replay_frac=args.replay_frac, seed=args.seed)
    print(f"[dry-run] curriculum strategy={args.strategy} split={args.split} "
          f"(ordered={args.strategy in ORDERED_STRATEGIES}) positions={len(order)}\n")
    _order_profile(raw, order)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ordered = raw.select(order)
    formatted = to_examples(ordered, tok, num_proc=args.num_proc)
    summarize(raw, formatted, tokenizer=tok, max_show=args.max_show)


if __name__ == "__main__":
    _main()
