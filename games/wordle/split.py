"""Regenerate the committed train/val word-split artifacts.

    python -m games.wordle.split

Writes ``train_words.txt`` and ``val_words.txt`` next to ``words.txt`` using the
deterministic, order-independent assignment in ``game.assign_pool``. These files are
the source of truth the server loads, and they're committed to the repo — so the
split is identical for every run, every model, over any length of time. Only run
this on purpose (e.g. after curating ``words.txt``); a regeneration changes which
words are eval words and invalidates cross-run comparisons.
"""

from __future__ import annotations

from games.wordle.game import _TRAIN_FILE, _VAL_FILE, generate_split_files


def main() -> None:
    n_train, n_val = generate_split_files()
    total = n_train + n_val
    print(f"Wrote {n_train} train words -> {_TRAIN_FILE.name}")
    print(f"Wrote {n_val} val words   -> {_VAL_FILE.name}")
    print(f"Total {total} words ({n_val / total:.1%} val)")


if __name__ == "__main__":
    main()
