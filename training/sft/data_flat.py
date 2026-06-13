"""LOADER #1 — the flat, non-curriculum data loader.

Serves both non-curriculum training variants:
- **wordle-only baseline** → `games=["wordle"]`
- **full set, no curriculum** → `games=None` (every game)

Pipeline: load a split of the Hub dataset → keep `valid==True` (the quality gate; for Wordle this
is format compliance / `has_think`, already re-derived in `distillation/push.py`) → optional game
filter → shuffle → render each row to a `{"prompt", "completion"}` pair via
`format.build_example`. The result is a TRL prompt-completion `Dataset` (completion-only loss).

Run as a script for a **dry run** that loads NO model — only the dataset (and, unless
`--no-tokenize`, the tokenizer for length stats + a rendered sample):

    uv run --package training python -m training.sft.data_flat --variant wordle --dry-run
    uv run --package training python -m training.sft.data_flat --variant full   --dry-run
"""

from __future__ import annotations

import argparse
from collections import Counter

from training.sft.format import GAME_NO, build_example

DEFAULT_REPO = "saketh-chervu/word-games-distillation"
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
KEEP_COLUMNS = ("prompt", "completion")


def load_valid(repo_id: str = DEFAULT_REPO, *, split: str = "train",
               games: list[str] | None = None):
    """Load a split, drop invalid rows, and (optionally) keep only the named games.

    Returns the *raw* unified-schema rows (not yet rendered) so callers can either format them
    (`to_examples`) or inspect them (the curriculum loader sorts on `game_no`/`round`/length).
    """
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split)
    n_total = len(ds)
    ds = ds.filter(lambda r: bool(r["valid"]))
    n_valid = len(ds)
    if games is not None:
        wanted = set(games)
        unknown = wanted - set(GAME_NO)
        if unknown:
            raise ValueError(f"unknown game(s): {sorted(unknown)} (known: {sorted(GAME_NO)})")
        ds = ds.filter(lambda r: r["game_name"] in wanted)
    # Stash the drop counts on the dataset object for the dry-run report.
    ds._pp_counts = {"total": n_total, "valid": n_valid, "dropped_invalid": n_total - n_valid}  # type: ignore[attr-defined]
    return ds


def to_examples(ds, tokenizer, *, num_proc: int = 4):
    """Map raw rows → `{"prompt", "completion"}`, dropping every other column."""
    remove = [c for c in ds.column_names if c not in KEEP_COLUMNS]
    return ds.map(
        lambda r: build_example(r, tokenizer),
        remove_columns=remove,
        num_proc=num_proc,
        desc="render prompt/completion",
    )


def filter_max_tokens(ds, tokenizer, max_tokens: int, *, num_proc: int = 4):
    """Drop rows whose tokenized prompt+completion exceeds `max_tokens`.

    Filtering (vs truncating) keeps SFT targets intact — a truncated completion would teach a
    cut-off answer. Operates on the formatted prompt/completion Dataset.
    """
    def _measure(batch):
        joined = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
        ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
        return {"_ntok": [len(x) for x in ids]}

    ds = ds.map(_measure, batched=True, num_proc=num_proc, desc="measure tokens")
    before = len(ds)
    ds = ds.filter(lambda r: r["_ntok"] <= max_tokens, num_proc=num_proc)
    dropped = before - len(ds)
    if dropped:
        print(f"[filter] dropped {dropped}/{before} rows over {max_tokens} tokens")
    return ds.remove_columns("_ntok")


def load_flat(repo_id: str = DEFAULT_REPO, *, split: str = "train",
              games: list[str] | None = None, tokenizer, seed: int = 0,
              shuffle: bool = True, num_proc: int = 4, max_tokens: int | None = None):
    """Flat loader: valid-filter (+ optional game filter) → shuffle → prompt/completion Dataset.

    If `max_tokens` is set, rows whose prompt+completion exceeds it are dropped (not truncated).
    """
    ds = load_valid(repo_id, split=split, games=games)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = to_examples(ds, tokenizer, num_proc=num_proc)
    if max_tokens is not None:
        ds = filter_max_tokens(ds, tokenizer, max_tokens, num_proc=num_proc)
    return ds


# --------------------------------------------------------------------------------------------
# Dry-run reporting (shared with data_curriculum)
# --------------------------------------------------------------------------------------------

def _percentiles(values: list[int], ps=(50, 90, 99)) -> dict[int, int]:
    if not values:
        return {p: 0 for p in ps}
    s = sorted(values)
    return {p: s[min(len(s) - 1, int(p / 100 * len(s)))] for p in ps}


def summarize(raw_ds, formatted_ds, tokenizer=None, *, max_show: int = 2, length_sample: int = 2000):
    """Print a human-readable dry-run report: counts, per-game breakdown, token lengths, samples."""
    counts = getattr(raw_ds, "_pp_counts", None)
    if counts:
        print(f"rows: total={counts['total']}  valid={counts['valid']}  "
              f"dropped_invalid={counts['dropped_invalid']}")
    print(f"rows after filtering: {len(raw_ds)}")

    by_game = Counter(raw_ds["game_name"])
    print("\nper-game (kept):")
    for name, n in sorted(by_game.items(), key=lambda kv: GAME_NO.get(kv[0], 99)):
        print(f"  {GAME_NO.get(name, '?'):>2}  {name:<12} {n}")

    if tokenizer is not None and len(formatted_ds) > 0:
        n = min(length_sample, len(formatted_ds))
        sample = formatted_ds.select(range(n))
        plens, clens = [], []
        for ex in sample:
            plens.append(len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]))
            clens.append(len(tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]))
        totals = [p + c for p, c in zip(plens, clens)]
        pp, cc, tt = _percentiles(plens), _percentiles(clens), _percentiles(totals)
        print(f"\ntoken lengths over {n} sampled rows (p50/p90/p99):")
        print(f"  prompt     {pp[50]}/{pp[90]}/{pp[99]}")
        print(f"  completion {cc[50]}/{cc[90]}/{cc[99]}")
        print(f"  total      {tt[50]}/{tt[90]}/{tt[99]}   (max prompt+completion = {max(totals)})")

    print(f"\n--- first {max_show} formatted example(s) ---")
    for i in range(min(max_show, len(formatted_ds))):
        ex = formatted_ds[i]
        print(f"\n[example {i}] PROMPT:\n{ex['prompt']}")
        print(f"[example {i}] COMPLETION:\n{ex['completion']}")
        print("-" * 80)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Flat (non-curriculum) SFT data loader / dry run.")
    ap.add_argument("--variant", choices=["wordle", "full"], default="full",
                    help="wordle = wordle-only baseline; full = every game.")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer source for rendering/lengths.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--max-show", type=int, default=2, help="formatted examples to print.")
    ap.add_argument("--no-tokenize", action="store_true",
                    help="skip loading the tokenizer (counts only; no rendered sample/lengths).")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + report only; do not return data to a trainer.")
    args = ap.parse_args()

    games = ["wordle"] if args.variant == "wordle" else None

    if args.no_tokenize:
        raw = load_valid(args.repo_id, split=args.split, games=games)
        print(f"[dry-run] variant={args.variant} split={args.split} (counts only)\n")
        summarize(raw, raw.select(range(0)), tokenizer=None, max_show=0)
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_flat(args.repo_id, split=args.split, games=games, tokenizer=tok,
                   seed=args.seed, shuffle=True, num_proc=args.num_proc)
    raw = load_valid(args.repo_id, split=args.split, games=games)
    print(f"[dry-run] variant={args.variant} split={args.split} model={args.model}\n")
    summarize(raw, ds, tokenizer=tok, max_show=args.max_show)


if __name__ == "__main__":
    _main()
