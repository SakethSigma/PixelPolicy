"""Pure Character-set game logic (Word-skill game #7).

A **single-turn** environment: ``reset`` poses a few words, ``step(answer)`` scores the submitted
**used** / **unused** letter sets and ends the episode. It teaches the model to aggregate letter
coverage across several words — the skill of tracking which letters of a-z are in play (directly
useful for Wordle, where you reason about remaining letters).

Like ``games.charcount.game`` this module has **no** web/FastAPI dependency and is the single
source of truth. Ground truth is pure Python: ``used`` = the union of letters across all the
words; ``unused`` = the 26-letter alphabet minus ``used``. The single-turn ``status`` convention
mirrors charcount/Wordle:

    "in_progress"  -> challenge posed, no answer yet
    "correct"      -> submitted used AND unused sets match the truth   (the "good" status)
    "incorrect"    -> either set wrong, or the answer was unparseable
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from games.wordvocab.build import load_vocab
from games.wordvocab.split import Mode, assign_pool

GAME_NAME = "charset"  # salt for the per-game vocabulary split
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

Status = Literal["in_progress", "correct", "incorrect"]


def analyze(words: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(used, unused)`` letter lists (sorted) for the union of ``words``."""
    used = {c for w in words for c in w.lower() if c.isalpha()}
    unused = set(ALPHABET) - used
    return sorted(used), sorted(unused)


def encode_words(words: list[str]) -> str:
    """Encode the challenge words as a single comma-joined target string."""
    return ",".join(w.strip().lower() for w in words)


def decode_words(target: str) -> list[str]:
    """Split a comma-joined target back into its words."""
    words = [w.strip().lower() for w in target.split(",") if w.strip()]
    if not words:
        raise ValueError(f"expected a comma-joined word list, got {target!r}")
    return words


class Solution(BaseModel):
    """The computed used/unused letter sets — the episode's ground truth."""

    used: list[str] = Field(default_factory=list)
    unused: list[str] = Field(default_factory=list)


class GameState(BaseModel):
    """Serializable snapshot. The solution is revealed only once the episode ends."""

    game_id: str
    words: list[str]
    status: Status = "in_progress"
    submitted: Optional[str] = None
    solution: Optional[Solution] = None


# "unused" contains "used"; the negative lookbehind keeps the used-line regex from matching it.
_USED = re.compile(r"(?<![a-z])used\b[^:\n]*:\s*([^\n]*)", re.IGNORECASE)
_UNUSED = re.compile(r"unused\b[^:\n]*:\s*([^\n]*)", re.IGNORECASE)
_LETTER = re.compile(r"[a-z]", re.IGNORECASE)


def parse_answer(text: str) -> Optional[tuple[set[str], set[str]]]:
    """Parse a submitted answer into ``(used, unused)`` letter sets; ``None`` if a line is absent.

    Tolerant: reads the canonical ``used (K): …`` / ``unused (M): …`` block (any order/spacing/
    case). Both lines must be present — an unparseable answer scores ``incorrect`` (the
    "malformed costs you the round" contract, as in Wordle/charcount).
    """
    u = _USED.search(text)
    n = _UNUSED.search(text)
    if u is None or n is None:
        return None
    used = {c.lower() for c in _LETTER.findall(u.group(1))}
    unused = {c.lower() for c in _LETTER.findall(n.group(1))}
    return used, unused


def is_correct(words: list[str], used: set[str], unused: set[str]) -> bool:
    """Whether the submitted used/unused sets exactly match the truth."""
    truth_used, truth_unused = analyze(words)
    return used == set(truth_used) and unused == set(truth_unused)


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class CharsetGame:
    """A single Character-set episode. ``step`` scores once and ends the game."""

    def __init__(self, words: list[str], game_id: str):
        self.words = [w.strip().lower() for w in words]
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score ``answer`` against the used/unused truth; terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        parsed = parse_answer(answer)
        self.status = "correct" if (parsed is not None and is_correct(self.words, *parsed)) else "incorrect"
        return self.state()

    def state(self) -> GameState:
        used, unused = analyze(self.words)
        return GameState(
            game_id=self.game_id,
            words=self.words,
            status=self.status,
            submitted=self.submitted,
            solution=Solution(used=used, unused=unused) if self.status != "in_progress" else None,
        )


class CharsetBank:
    """Loads the shared vocabulary and builds multi-word challenges that mix lengths.

    Each challenge has one five-letter **Wordle** word and one or more **non-five-letter** words
    (the "and otherwise" the task asks for), so the model practises aggregating letters across
    words of different lengths. Pools use the salted ``charset`` train/val split.
    """

    def __init__(self, vocab_path: Optional[Path] = None):
        from games.wordle.game import WordBank

        words = load_vocab(vocab_path) if vocab_path is not None else load_vocab()
        wordle = set(WordBank().all)
        five = sorted(wordle)                                  # the Wordle vocab (all 5 letters)
        nonfive = sorted(w for w in words if len(w) != 5)      # "otherwise" — non-5-letter words
        self.five_train = [w for w in five if assign_pool(GAME_NAME, w) == "train"]
        self.five_val = [w for w in five if assign_pool(GAME_NAME, w) == "val"]
        self.nonfive_train = [w for w in nonfive if assign_pool(GAME_NAME, w) == "train"]
        self.nonfive_val = [w for w in nonfive if assign_pool(GAME_NAME, w) == "val"]
        if not (self.five_train and self.nonfive_train):
            raise ValueError("Empty word pool — is vocab.txt built?")
        import random

        self._rng = random.Random()

    def make_words(self, mode: Mode, rng, *, k: Optional[int] = None) -> list[str]:
        """Pick ``k`` words: one five-letter Wordle word + ``k-1`` non-five-letter words."""
        if k is None:
            k = rng.choice([2, 3, 4])
        five = self.five_train if mode == "train" else self.five_val
        nonfive = self.nonfive_train if mode == "train" else self.nonfive_val
        chosen = [rng.choice(five)] + [rng.choice(nonfive) for _ in range(k - 1)]
        rng.shuffle(chosen)
        return chosen

    def sample_words(self, mode: Mode) -> list[str]:
        """A challenge for terminal play (random word count)."""
        return self.make_words(mode, self._rng)
