"""Pure Anagrams game logic (Word-skill game #3).

A **single-turn** environment: ``reset`` poses two words, ``step(answer)`` scores a yes/no
verdict and ends the episode. It teaches letter-multiset reasoning. This is a *reasoning* game
distilled from Claude (``<think>…</think><answer>yes|no</answer>``) with **rejection sampling**:
the core's ground truth is the gate, so a trace that reasons to the wrong answer is dropped.

Like ``games.charcount.game`` this module has **no** web/FastAPI dependency and is the single
source of truth. Ground truth is pure Python: two words are anagrams iff their sorted letter
multisets are equal. The single-turn ``status`` convention mirrors charcount/Wordle:

    "in_progress"  -> challenge posed, no answer yet
    "correct"      -> verdict matched the multiset check    (the "good" status)
    "incorrect"    -> verdict wrong or unparseable
"""

from __future__ import annotations

import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from games.wordvocab.build import load_vocab
from games.wordvocab.split import Mode, assign_pool

GAME_NAME = "anagram"  # salt for the per-game vocabulary split

Status = Literal["in_progress", "correct", "incorrect"]


def signature(word: str) -> str:
    """The sorted-letter signature; two words are anagrams iff their signatures match."""
    return "".join(sorted(word.strip().lower()))


def are_anagrams(w1: str, w2: str) -> bool:
    """Ground truth: same letters, each used the same number of times."""
    return signature(w1) == signature(w2)


def encode_pair(w1: str, w2: str) -> str:
    """Encode a pair as a single target string (for the registry/batch driver)."""
    return f"{w1.strip().lower()},{w2.strip().lower()}"


def decode_pair(target: str) -> tuple[str, str]:
    """Split a ``"w1,w2"`` target back into two words."""
    parts = [p.strip().lower() for p in target.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected a 'w1,w2' pair, got {target!r}")
    return parts[0], parts[1]


class Solution(BaseModel):
    """The revealed answer once the episode ends."""

    are_anagrams: bool


class GameState(BaseModel):
    """Serializable snapshot. The solution is revealed only once the episode ends."""

    game_id: str
    word1: str
    word2: str
    status: Status = "in_progress"
    submitted: Optional[str] = None
    solution: Optional[Solution] = None


_YESNO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_answer(text: str) -> Optional[bool]:
    """Pull the yes/no verdict from a submitted answer (last occurrence); ``None`` if absent."""
    tokens = _YESNO.findall(text.lower())
    if not tokens:
        return None
    return tokens[-1] == "yes"


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class AnagramGame:
    """A single Anagrams episode. ``step`` scores once and ends the game."""

    def __init__(self, word1: str, word2: str, game_id: str):
        self.word1 = word1.strip().lower()
        self.word2 = word2.strip().lower()
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score ``answer`` against the multiset check; terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        verdict = parse_answer(answer)
        truth = are_anagrams(self.word1, self.word2)
        self.status = "correct" if (verdict is not None and verdict == truth) else "incorrect"
        return self.state()

    def state(self) -> GameState:
        return GameState(
            game_id=self.game_id,
            word1=self.word1,
            word2=self.word2,
            status=self.status,
            submitted=self.submitted,
            solution=(
                Solution(are_anagrams=are_anagrams(self.word1, self.word2))
                if self.status != "in_progress" else None
            ),
        )


def _overlap(w1: str, w2: str) -> int:
    """Size of the shared letter multiset — how 'near' a near-miss negative is."""
    c1, c2 = Counter(w1), Counter(w2)
    return sum((c1 & c2).values())


class AnagramBank:
    """Loads the shared vocabulary, derives the salted ``anagram`` split, and builds pairs.

    Positives come from sorted-signature groups of size ≥ 2 (real anagram sets). Negatives are
    mostly *hard* near-misses (same length, high letter overlap, but not anagrams) with some easy
    different-length pairs mixed in, so the model can't shortcut on length or letter set alone.
    """

    HARD_NEG_FRACTION = 0.7

    def __init__(self, vocab_path: Optional[Path] = None):
        words = load_vocab(vocab_path) if vocab_path is not None else load_vocab()
        self.train: list[str] = []
        self.val: list[str] = []
        for w in words:
            (self.val if assign_pool(GAME_NAME, w) == "val" else self.train).append(w)
        if not self.train or not self.val:
            raise ValueError("Train and val pools must both be non-empty")
        # Per-pool indexes: anagram groups (for positives) and by-length lists (for negatives).
        self._groups: dict[str, list[list[str]]] = {}
        self._by_len: dict[str, dict[int, list[str]]] = {}
        for name, pool in (("train", self.train), ("val", self.val)):
            sig_map: dict[str, list[str]] = defaultdict(list)
            by_len: dict[int, list[str]] = defaultdict(list)
            for w in pool:
                sig_map[signature(w)].append(w)
                by_len[len(w)].append(w)
            self._groups[name] = [ws for ws in sig_map.values() if len(ws) >= 2]
            self._by_len[name] = by_len
        import random

        self._rng = random.Random()

    def _pool(self, mode: Mode):
        return (self.train if mode == "train" else self.val,
                self._groups[mode], self._by_len[mode])

    def positive_pair(self, mode: Mode, rng) -> tuple[str, str]:
        _, groups, _ = self._pool(mode)
        if not groups:
            raise RuntimeError(f"no anagram groups in the {mode} pool")
        group = rng.choice(groups)
        w1, w2 = rng.sample(group, 2)
        return w1, w2

    def negative_pair(self, mode: Mode, rng, *, hard: bool) -> tuple[str, str]:
        pool, _, by_len = self._pool(mode)
        for _ in range(1000):
            w1 = rng.choice(pool)
            if hard:
                same_len = by_len.get(len(w1), [])
                if len(same_len) < 2:
                    continue
                # Probe a handful of same-length words; take the highest-overlap non-anagram.
                cands = [rng.choice(same_len) for _ in range(8)]
                cands = [c for c in cands if c != w1 and not are_anagrams(w1, c)]
                if not cands:
                    continue
                w2 = max(cands, key=lambda c: _overlap(w1, c))
            else:
                lengths = [L for L in by_len if L != len(w1)]
                if not lengths:
                    continue
                w2 = rng.choice(by_len[rng.choice(lengths)])
            if w1 != w2 and not are_anagrams(w1, w2):
                return w1, w2
        raise RuntimeError("could not synthesize a negative pair")

    def sample_targets(self, n: int, mode: Mode, rng, *, pos_fraction: float = 0.4) -> list[str]:
        """``n`` distinct ``"w1,w2"`` targets with a ``pos_fraction`` positive / rest negative mix.

        Of the negatives, :attr:`HARD_NEG_FRACTION` are hard near-misses; the rest are easy
        different-length pairs. Order is shuffled so positives and negatives interleave.
        """
        n_pos = round(n * pos_fraction)
        n_neg = n - n_pos
        n_hard = round(n_neg * self.HARD_NEG_FRACTION)
        seen: set[frozenset[str]] = set()
        out: list[str] = []

        def _add(maker) -> None:
            for _ in range(10000):
                if len(out) >= target_count:
                    return
                w1, w2 = maker()
                key = frozenset((w1, w2))
                if key in seen:
                    continue
                seen.add(key)
                out.append(encode_pair(w1, w2))
            raise RuntimeError("ran out of distinct pairs")

        target_count = n_pos
        _add(lambda: self.positive_pair(mode, rng))
        target_count = n_pos + n_hard
        _add(lambda: self.negative_pair(mode, rng, hard=True))
        target_count = n
        _add(lambda: self.negative_pair(mode, rng, hard=False))

        rng.shuffle(out)
        return out

    def sample_pair(self, mode: Mode) -> str:
        """One random ``"w1,w2"`` target (50/50 positive/negative) — for terminal play."""
        if self._rng.random() < 0.5:
            return encode_pair(*self.positive_pair(mode, self._rng))
        return encode_pair(*self.negative_pair(mode, self._rng, hard=True))
