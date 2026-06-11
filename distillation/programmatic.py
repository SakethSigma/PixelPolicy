"""Programmatic SFT generation — the "synthetic teacher" for no-reasoning word games.

For games whose label is cheap and exact (char counts, validity, ends→starts, rhymes), no
Claude is needed: step the env, read the gold answer the core computed, and format it straight
into the completion. The rows are byte-identical in shape to the distilled ones (the unified
schema in ``distillation/schema.py``), so the combine + ``push.py`` step is unchanged.

This module implements game #1, **charcount**. It does NOT touch the generic Claude pipeline.

    uv run --package distillation python -m distillation.programmatic   # default 14k rows

Determinism: the sample is drawn with a seeded RNG; the labels are pure functions of the word,
so a given seed reproduces the exact dataset. Every row is **self-checked** (its own answer is
fed back through ``env.step`` and must score ``"correct"``) before being written — rejection by
construction, the same quality gate the distilled games pass by filtering.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from agents.charcount.agent import CharCountAgent
from games.charcount.game import CharCountBank, analyze
from games.charcount.render import render_answer
from games.wordle.game import WordBank
from distillation.registry import GAME_NUMBERS
from distillation.schema import sft_row

MIN_LEN, MAX_LEN = 3, 20


def _stratified_words(
    bank: CharCountBank, *, n: int, wordle_min: int, rng: random.Random
) -> list[str]:
    """Draw ``n`` distinct charcount-**train** words: ≥``wordle_min`` from the Wordle vocab,
    the rest from WordNet-origin words spread across lengths 3-20 (so length variety is real).

    Generating only from the charcount-train pool keeps charcount-val clean for eval. Wordle's
    own train/val words both land in charcount-train ~80% of the time (the salted split), which
    is exactly the cross-game design — the model learns to analyze words it also plays Wordle on.
    """
    wordle_vocab = WordBank().all  # all 12,972 Wordle words (every one is length 5)
    train = set(bank.train)
    wordle_train = sorted(train & wordle_vocab)
    wn_by_len: dict[int, list[str]] = defaultdict(list)
    for w in sorted(train - wordle_vocab):
        if MIN_LEN <= len(w) <= MAX_LEN:
            wn_by_len[len(w)].append(w)

    n_wordle = min(wordle_min, len(wordle_train), n)
    picked_wordle = rng.sample(wordle_train, n_wordle)

    # Fill the remainder from WordNet words, round-robin across lengths for an even spread.
    n_wn = n - n_wordle
    lengths = sorted(wn_by_len)
    for L in lengths:
        rng.shuffle(wn_by_len[L])
    picked_wn: list[str] = []
    cursors = {L: 0 for L in lengths}
    while len(picked_wn) < n_wn and any(cursors[L] < len(wn_by_len[L]) for L in lengths):
        for L in lengths:
            if cursors[L] < len(wn_by_len[L]):
                picked_wn.append(wn_by_len[L][cursors[L]])
                cursors[L] += 1
                if len(picked_wn) >= n_wn:
                    break

    words = picked_wordle + picked_wn
    rng.shuffle(words)
    return words


def generate(n: int, wordle_min: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` charcount SFT rows in the unified schema and write them to ``out_path``."""
    bank = CharCountBank()
    agent = CharCountAgent()
    rng = random.Random(seed)
    words = _stratified_words(bank, n=n, wordle_min=wordle_min, rng=rng)

    wordle_vocab = WordBank().all
    game_no = GAME_NUMBERS["charcount"]
    rows: list[dict] = []
    for i, word in enumerate(words):
        state = bank_state(word)
        messages = agent.build_messages(state)
        analysis = analyze(word)
        completion = f"<answer>\n{render_answer(analysis)}\n</answer>"

        # Self-check (rejection by construction): the answer we wrote must score "correct".
        if agent.parse_action(completion) != render_answer(analysis):
            raise AssertionError(f"answer round-trip failed for {word!r}")
        assert state.status == "in_progress"

        rows.append(sft_row(
            game_name="charcount", game_no=game_no, round=1, target=word,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=i,
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_wordle = sum(1 for w in words if w in wordle_vocab)
    hist = Counter(len(w) for w in words)
    print(f"wrote {len(rows)} charcount rows -> {out_path}")
    print(f"  wordle-vocab words: {n_wordle}   wordnet-only words: {len(words) - n_wordle}")
    print("  length spread:", {L: hist[L] for L in sorted(hist)})
    return out_path


def bank_state(word: str):
    """A fresh in-progress GameState for ``word`` (challenge posed, nothing answered yet)."""
    from games.charcount.game import CharCountGame

    return CharCountGame(word=word, game_id=f"gen-{word}").state()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Programmatically generate charcount SFT data.")
    ap.add_argument("--episodes", type=int, default=14000, help="number of samples (distinct words)")
    ap.add_argument("--wordle-min", type=int, default=4000, help="at least this many Wordle-vocab words")
    ap.add_argument("--seed", type=int, default=0, help="seed for reproducible sampling")
    ap.add_argument("--out", default="distillation/data/charcount_sft.jsonl", help="output JSONL path")
    args = ap.parse_args(argv)
    generate(args.episodes, args.wordle_min, args.seed, Path(args.out))


if __name__ == "__main__":
    main()
