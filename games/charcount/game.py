"""Pure Character-counts game logic (Word-skill game #1).

A **single-turn** environment, the simplest member of the new word-skill family: ``reset``
poses one word, ``step(answer)`` scores the submitted character analysis and ends the episode.
It teaches the model to map a word to its characters — length, and the vowel/consonant split.

Like ``games.wordle.game`` this module has **no** web/FastAPI dependency and is the single
source of truth: the core owns the ground truth (the analysis) and the scoring, so the
in-process and HTTP paths can't diverge and the distillation rejection filter stays
game-agnostic. The single-turn ``status`` convention mirrors Wordle's ``"won"``:

    "in_progress"  -> challenge posed, no answer yet
    "correct"      -> step()'s answer matched the computed analysis   (the "good" status)
    "incorrect"    -> step()'s answer did not match (or was unparseable)

Ground truth is pure Python (no corpora): classify each letter against ``aeiou``. Counts are
multisets (repeats kept), so the invariant ``length == #vowels + #consonants`` always holds —
a small consistency the student learns. ``y`` is treated as a consonant.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from games.wordvocab.build import load_vocab
from games.wordvocab.split import Mode, assign_pool

VOWELS = "aeiou"
GAME_NAME = "charcount"  # salt for the per-game vocabulary split

Status = Literal["in_progress", "correct", "incorrect"]


class Analysis(BaseModel):
    """The computed character analysis of a word — the episode's ground truth."""

    length: int
    vowels: list[str] = Field(default_factory=list)        # in left-to-right order, repeats kept
    consonants: list[str] = Field(default_factory=list)    # in left-to-right order, repeats kept

    @property
    def vowel_count(self) -> int:
        return len(self.vowels)

    @property
    def consonant_count(self) -> int:
        return len(self.consonants)


def analyze(word: str) -> Analysis:
    """Compute the character analysis: length + the vowel/consonant split (multisets)."""
    letters = [c for c in word.lower() if c.isalpha()]
    vowels = [c for c in letters if c in VOWELS]
    consonants = [c for c in letters if c not in VOWELS]
    return Analysis(length=len(letters), vowels=vowels, consonants=consonants)


class GameState(BaseModel):
    """Serializable snapshot. The solution is revealed only once the episode ends."""

    game_id: str
    word: str                                   # the challenge word the model analyzes
    status: Status = "in_progress"
    submitted: Optional[str] = None             # the raw answer text step() scored
    solution: Optional[Analysis] = None         # revealed only when status != "in_progress"


# Pull a "vowels: a, e" / "consonants (4): p, l, n, t" style line apart into its letters.
_LABEL_LETTERS = {
    "length": re.compile(r"length\D*?(\d+)", re.IGNORECASE),
    "vowels": re.compile(r"vowels?\b[^:]*:\s*(.*)", re.IGNORECASE),
    "consonants": re.compile(r"consonants?\b[^:]*:\s*(.*)", re.IGNORECASE),
}
_LETTER = re.compile(r"[a-z]", re.IGNORECASE)


def parse_answer(text: str) -> Optional[Analysis]:
    """Parse a submitted analysis back into an :class:`Analysis`, tolerantly.

    Accepts the canonical block (see :func:`games.charcount.render.render_answer`) and minor
    variations: any order, optional ``(count)`` parentheticals, commas/spaces between letters,
    any case. Returns ``None`` if the vowel or consonant line is absent — an unparseable answer
    is scored ``incorrect`` (the "malformed costs you the round" contract, as in Wordle).
    """
    v_match = _LABEL_LETTERS["vowels"].search(text)
    c_match = _LABEL_LETTERS["consonants"].search(text)
    if v_match is None or c_match is None:
        return None

    vowels = [c.lower() for c in _LETTER.findall(v_match.group(1))]
    consonants = [c.lower() for c in _LETTER.findall(c_match.group(1))]

    len_match = _LABEL_LETTERS["length"].search(text)
    length = int(len_match.group(1)) if len_match else len(vowels) + len(consonants)
    return Analysis(length=length, vowels=vowels, consonants=consonants)


def is_correct(word: str, submitted: Analysis) -> bool:
    """Whether a submitted analysis matches the word's computed analysis (multiset compare)."""
    truth = analyze(word)
    return (
        submitted.length == truth.length
        and sorted(submitted.vowels) == sorted(truth.vowels)
        and sorted(submitted.consonants) == sorted(truth.consonants)
    )


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class CharCountGame:
    """A single Character-counts episode. ``step`` scores once and ends the game."""

    def __init__(self, word: str, game_id: str):
        self.word = word.strip().lower()
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score ``answer`` against the computed analysis; terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        parsed = parse_answer(answer)
        self.status = "correct" if (parsed is not None and is_correct(self.word, parsed)) else "incorrect"
        return self.state()

    def state(self) -> GameState:
        return GameState(
            game_id=self.game_id,
            word=self.word,
            status=self.status,
            submitted=self.submitted,
            solution=analyze(self.word) if self.status != "in_progress" else None,
        )


class CharCountBank:
    """Loads the shared multi-length vocabulary and derives the per-game train/val split.

    The split is the *salted* rule (:func:`games.wordvocab.split.assign_pool` with ``game =
    "charcount"``), derived deterministically at load time — no committed per-game artifact is
    needed because the hash is byte-stable. ``all`` is the union; every word is a legal target.
    """

    def __init__(self, vocab_path: Optional[Path] = None):
        words = load_vocab(vocab_path) if vocab_path is not None else load_vocab()
        self.train: list[str] = []
        self.val: list[str] = []
        for w in words:
            (self.val if assign_pool(GAME_NAME, w) == "val" else self.train).append(w)
        if not self.train or not self.val:
            raise ValueError("Train and val pools must both be non-empty")
        self.all: set[str] = set(words)
        import random

        self._rng = random.Random()

    def sample(self, mode: Mode) -> str:
        """Pick a random word from the requested pool."""
        pool = self.train if mode == "train" else self.val
        return self._rng.choice(pool)
