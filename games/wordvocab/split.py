"""Shared, game-salted train/val split for the multi-length vocabulary.

The single utility every auxiliary word-skill game reuses. It is a *salted* variant of
:func:`games.wordle.game.assign_pool`: the bucket is hashed over ``f"{game}:{word}"`` instead
of ``word`` alone, so **the same word lands in different pools for different games**. That is
the deliberate cross-game design from ``games/DATA_SOURCING.md`` — a word held out for one
game is trained in another, so the model becomes familiar with every word's spelling, letters,
meaning, and sound even when it is never *played* on that word.

Like the Wordle split, assignment depends only on the (game, word) bytes via sha256, so it is
byte-for-byte identical on every machine, every run, forever — no committed per-game artifact
is needed; a bank can derive its split deterministically at load time.

Wordle keeps its own *unsalted* committed split (``games/wordle``); only these new games use
the salted rule, so Wordle eval stays backward-compatible.
"""

from __future__ import annotations

import hashlib
from typing import Literal

Mode = Literal["train", "val"]

# Match the Wordle split parameters exactly (only the hashed key differs: it is salted by game).
_VAL_FRACTION = 0.2
_SPLIT_BUCKETS = 1000


def assign_pool(game: str, word: str, val_fraction: float = _VAL_FRACTION) -> Mode:
    """Assign ``word`` to ``"train"`` or ``"val"`` *for a specific game*.

    ``bucket = sha256(f"{game}:{word}") % 1000``; the bottom ``val_fraction`` go to val. The
    ``game`` salt is what makes a word's pool game-dependent (the whole point of the shared
    vocabulary). Deterministic and order-independent, exactly like the Wordle split.
    """
    key = f"{game}:{word.strip().lower()}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % _SPLIT_BUCKETS
    return "val" if bucket < val_fraction * _SPLIT_BUCKETS else "train"
