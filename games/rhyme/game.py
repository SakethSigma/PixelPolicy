"""Pure Rhymes game logic (Word-skill game #5).

A **single-turn** environment with two variants: ``reset`` poses a word (and, for MCQ, five
options) and ``step(answer)`` scores the reply and ends the episode. It teaches the model the
sound/phonetic mapping of words.

Like ``games.charcount.game`` this module has **no** web/FastAPI dependency and is the single
source of truth. The ground truth is the **CMU Pronouncing Dictionary** via the ``pronouncing``
library (bundled, offline — no download): a word rhymes with ``word`` iff it is in
``pronouncing.rhymes(word)``. The single-turn ``status`` convention mirrors charcount/Wordle:

    "in_progress"  -> challenge posed, no answer yet
    "correct"      -> step()'s answer rhymes (and, for MCQ, is one of the options)
    "incorrect"    -> step()'s answer did not rhyme (or was unparseable / not an option)

Two variants:
  - "mcq"  : a word + 5 options, exactly one of which rhymes. Correct = pick that option.
  - "free" : "name a word that rhymes with X". Correct = any member of the rhyme set.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from games.wordvocab.build import load_vocab
from games.wordvocab.split import Mode, assign_pool

GAME_NAME = "rhyme"  # salt for the per-game vocabulary split
N_OPTIONS = 5        # MCQ: one rhyming option + four distractors
_MAX_EXAMPLES = 10   # how many sample rhymes to reveal in the free-variant solution

Status = Literal["in_progress", "correct", "incorrect"]
Variant = Literal["mcq", "free"]


@lru_cache(maxsize=None)
def rhymes(word: str) -> frozenset[str]:
    """The CMU-dict rhyme set of ``word`` (lowercased), as a frozenset. Empty if unknown.

    This is the full (lenient) scoring oracle — it includes CMU forms with hyphens/apostrophes
    (e.g. ``non-discrimination``). For *generating* clean single-word challenges, use
    :func:`alpha_rhymes`, which keeps only plain alphabetic rhymes.
    """
    import pronouncing  # lazy: only the rhyme core needs it (bundled CMU dict, offline)

    return frozenset(r.lower() for r in pronouncing.rhymes(word.strip().lower()))


@lru_cache(maxsize=None)
def alpha_rhymes(word: str) -> frozenset[str]:
    """The rhyme set restricted to plain alphabetic words (no hyphens/apostrophes/periods).

    The shared vocabulary is all-alpha, so options and gold answers are drawn from here to keep
    generated challenges clean and unambiguous to parse.
    """
    return frozenset(r for r in rhymes(word) if r.isalpha())


def is_rhyme(word: str, candidate: str) -> bool:
    """Whether ``candidate`` rhymes with ``word`` (CMU-dict membership)."""
    return candidate.strip().lower() in rhymes(word)


class Solution(BaseModel):
    """The revealed answer once the episode ends."""

    variant: Variant
    correct_option: Optional[str] = None        # MCQ: the one rhyming option
    examples: list[str] = Field(default_factory=list)  # free: a few accepted rhymes


class GameState(BaseModel):
    """Serializable snapshot. The solution is revealed only once the episode ends."""

    game_id: str
    word: str                                   # the word to rhyme with
    variant: Variant = "free"
    options: Optional[list[str]] = None         # MCQ only: the five choices (shuffled)
    status: Status = "in_progress"
    submitted: Optional[str] = None             # the raw answer text step() scored
    solution: Optional[Solution] = None


def parse_answer(text: str) -> Optional[str]:
    """Pull a single candidate word out of a submitted answer; ``None`` if none is found.

    Tag-robust: if an ``<answer>…</answer>`` tag is present, parse its body (so the closing tag's
    own letters can't be mistaken for the answer); otherwise take the last word of the text.
    """
    import re

    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    src = m.group(1) if m else text
    # Hyphens/apostrophes are kept inside a token so a CMU form like "non-discrimination" parses
    # whole rather than being split into its tail.
    words = re.findall(r"[a-z][a-z'\-]*", src.strip().lower())
    return words[-1] if words else None


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class RhymeGame:
    """A single Rhymes episode. ``step`` scores once and ends the game."""

    def __init__(self, word: str, game_id: str, *, variant: Variant = "free",
                 options: Optional[list[str]] = None):
        self.word = word.strip().lower()
        self.game_id = game_id
        self.variant: Variant = variant
        self.options = [o.strip().lower() for o in options] if options else None
        if self.variant == "mcq" and not self.options:
            raise ValueError("MCQ variant requires options")
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score ``answer`` against the rhyme set; terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        cand = parse_answer(answer)
        self.status = "correct" if (cand is not None and self._is_correct(cand)) else "incorrect"
        return self.state()

    def _is_correct(self, cand: str) -> bool:
        if self.variant == "mcq":
            return cand in (self.options or []) and is_rhyme(self.word, cand)
        return is_rhyme(self.word, cand)

    def _solution(self) -> Solution:
        if self.variant == "mcq":
            correct = next((o for o in (self.options or []) if is_rhyme(self.word, o)), None)
            return Solution(variant="mcq", correct_option=correct)
        examples = sorted(rhymes(self.word))[:_MAX_EXAMPLES]
        return Solution(variant="free", examples=examples)

    def state(self) -> GameState:
        return GameState(
            game_id=self.game_id,
            word=self.word,
            variant=self.variant,
            options=self.options,
            status=self.status,
            submitted=self.submitted,
            solution=self._solution() if self.status != "in_progress" else None,
        )


class RhymeBank:
    """Loads the shared vocabulary, derives the salted ``rhyme`` train/val split, and builds
    challenges using the CMU dictionary.

    The split is the salted rule (:func:`games.wordvocab.split.assign_pool` with ``game =
    "rhyme"``), so a word held out here is trained in another game. Only words with at least one
    rhyme are usable as challenge seeds; ``train_seeds`` / ``val_seeds`` hold those subsets.
    """

    def __init__(self, vocab_path: Optional[Path] = None):
        words = load_vocab(vocab_path) if vocab_path is not None else load_vocab()
        self.train: list[str] = []
        self.val: list[str] = []
        for w in words:
            (self.val if assign_pool(GAME_NAME, w) == "val" else self.train).append(w)
        if not self.train or not self.val:
            raise ValueError("Train and val pools must both be non-empty")
        self.all: list[str] = words
        import random

        self._rng = random.Random()

    def has_rhyme(self, word: str) -> bool:
        # Require a clean single-word rhyme so a usable gold answer / correct option exists.
        return len(alpha_rhymes(word)) > 0

    def _pool(self, mode: Mode) -> list[str]:
        return self.train if mode == "train" else self.val

    def sample_seed(self, mode: Mode) -> str:
        """A random word from the pool that has at least one rhyme."""
        pool = self._pool(mode)
        for _ in range(1000):
            w = self._rng.choice(pool)
            if self.has_rhyme(w):
                return w
        raise RuntimeError("could not find a rhymable word — is the CMU dict available?")

    def a_rhyme(self, word: str, rng) -> Optional[str]:
        """One clean (alphabetic) rhyme for ``word`` — the gold free answer / correct MCQ option."""
        rs = sorted(alpha_rhymes(word))
        return rng.choice(rs) if rs else None

    def mcq_options(self, word: str, rng, *, n: int = N_OPTIONS) -> Optional[list[str]]:
        """One rhyming option + ``n-1`` non-rhyming distractors from the vocab, shuffled.

        Returns ``None`` if ``word`` has no rhyme. Distractors are confirmed **not** to rhyme so
        exactly one option is correct.
        """
        rhyme_set = rhymes(word)
        correct = self.a_rhyme(word, rng)
        if correct is None:
            return None
        options = {correct}
        guard = 0
        while len(options) < n and guard < 10000:
            guard += 1
            cand = rng.choice(self.all)
            if cand != word and cand not in rhyme_set and cand not in options:
                options.add(cand)
        opts = list(options)
        rng.shuffle(opts)
        return opts
