"""Pure Validity + meaning game logic (Word-skill game #2).

A **single-turn** environment: ``reset`` poses one word (a real word, or a generated
pseudo-word), ``step(answer)`` scores the verdict (+ a meaning when valid) and ends the episode.
It teaches vocabulary membership and meaning recall.

Like ``games.charcount.game`` this module has **no** web/FastAPI dependency and is the single
source of truth. The ground truth comes from the committed **meanings asset**
(``games/wordvocab/meanings.jsonl``, built once from WordNet), so there is **no ``nltk`` at
runtime**: a word is *valid* iff it carries a WordNet definition, and that definition is the
gold ``<meaning>``. The single-turn ``status`` convention mirrors charcount/Wordle:

    "in_progress"  -> challenge posed, no answer yet
    "correct"      -> verdict matched membership (and, when valid, a non-empty meaning was given)
    "incorrect"    -> verdict wrong, missing meaning for a valid word, or unparseable

The meaning is checked *loosely* (non-empty) — we teach the model to recall a gloss, not to
reproduce WordNet's exact wording.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from games.wordvocab.build_meanings import load_meanings
from games.wordvocab.split import Mode, assign_pool

GAME_NAME = "validity"  # salt for the per-game vocabulary split
_MIN_PSEUDO_LEN = 3

Status = Literal["in_progress", "correct", "incorrect"]


@lru_cache(maxsize=1)
def _meanings() -> dict[str, str]:
    """The committed word → definition map (loaded once; no nltk)."""
    return load_meanings()


def is_valid_word(word: str) -> bool:
    """The validity oracle: a word is valid iff it carries a WordNet definition."""
    return word.strip().lower() in _meanings()


def lookup_meaning(word: str) -> Optional[str]:
    """The gold definition for ``word`` (``None`` if it has none)."""
    return _meanings().get(word.strip().lower())


class Solution(BaseModel):
    """The revealed answer once the episode ends."""

    valid: bool
    meaning: Optional[str] = None  # the gold definition, revealed only for valid words


class GameState(BaseModel):
    """Serializable snapshot. The solution is revealed only once the episode ends."""

    game_id: str
    word: str                                   # the challenge word the model judges
    status: Status = "in_progress"
    submitted: Optional[str] = None             # the raw answer text step() scored
    solution: Optional[Solution] = None


_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_MEANING = re.compile(r"<meaning>\s*(.*?)\s*</meaning>", re.IGNORECASE | re.DOTALL)


def parse_answer(text: str) -> Optional[tuple[bool, str]]:
    """Parse a submitted answer into ``(claims_valid, meaning)``; ``None`` if no verdict found.

    Tolerant: reads the canonical ``<answer>valid</answer>`` / ``<answer>invalid</answer>`` block
    (with an optional ``<meaning>…</meaning>``) and also bare ``valid``/``invalid`` prose. The
    verdict is read from the ``<answer>`` tag when present — never from the meaning, whose gloss
    may itself contain the word "invalid" (e.g. *annul*: "declare invalid"). ``invalid`` is
    checked before ``valid`` since it contains ``valid`` as a substring.
    """
    ans = _ANSWER.search(text)
    verdict_src = (ans.group(1) if ans is not None else _MEANING.sub("", text)).lower()
    if re.search(r"\binvalid\b", verdict_src):
        claims_valid = False
    elif re.search(r"\bvalid\b", verdict_src):
        claims_valid = True
    else:
        return None
    m = _MEANING.search(text)
    meaning = m.group(1).strip() if m else ""
    return claims_valid, meaning


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class ValidityGame:
    """A single Validity episode. ``step`` scores once and ends the game."""

    def __init__(self, word: str, game_id: str):
        self.word = word.strip().lower()
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score ``answer`` against word membership (+ meaning when valid); terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        self.status = "correct" if self._is_correct(answer) else "incorrect"
        return self.state()

    def _is_correct(self, answer: str) -> bool:
        parsed = parse_answer(answer)
        if parsed is None:
            return False
        claims_valid, meaning = parsed
        truth = is_valid_word(self.word)
        if claims_valid != truth:
            return False
        # A valid word must come with a (non-empty) meaning; an invalid one needs no meaning.
        return (not truth) or bool(meaning.strip())

    def state(self) -> GameState:
        valid = is_valid_word(self.word)
        return GameState(
            game_id=self.game_id,
            word=self.word,
            status=self.status,
            submitted=self.submitted,
            solution=(
                Solution(valid=valid, meaning=lookup_meaning(self.word) if valid else None)
                if self.status != "in_progress" else None
            ),
        )


def perturb(word: str, rng) -> str:
    """Mutate ``word`` by one swap/insert/delete — the seed of a pseudo-word."""
    letters = string.ascii_lowercase
    chars = list(word)
    op = rng.choice(("swap", "insert", "delete")) if len(chars) > _MIN_PSEUDO_LEN else \
        rng.choice(("swap", "insert"))
    if op == "swap":
        i = rng.randrange(len(chars))
        chars[i] = rng.choice(letters)
    elif op == "insert":
        i = rng.randrange(len(chars) + 1)
        chars.insert(i, rng.choice(letters))
    else:  # delete
        i = rng.randrange(len(chars))
        chars.pop(i)
    return "".join(chars)


class ValidityBank:
    """Loads the **Wordle vocabulary** (train + val union) as the word universe and the committed
    meanings asset, and builds valid/invalid challenges.

    Unlike the other word-skill games, validity draws from the Wordle vocab specifically (so the
    student learns the spelling and meaning of every Wordle word). The salted ``assign_pool``
    split is still available for eval, but ``valid_words`` exposes the full set — Wordle's own
    val words are deliberately allowed into this game's training data (meaning recall is a
    different skill, not Wordle-eval leakage).
    """

    def __init__(self):
        from games.wordle.game import WordBank

        self.wordle: set[str] = set(WordBank().all)
        self._meanings = _meanings()
        # Valid challenge candidates: real Wordle words that carry a definition.
        self.valid_words: list[str] = sorted(w for w in self.wordle if w in self._meanings)
        if not self.valid_words:
            raise ValueError("No valid words — is meanings.jsonl built?")
        self.train: list[str] = []
        self.val: list[str] = []
        for w in self.valid_words:
            (self.val if assign_pool(GAME_NAME, w) == "val" else self.train).append(w)
        import random

        self._rng = random.Random()

    def make_pseudo_word(self, rng) -> str:
        """A pseudo-word: perturb a real Wordle word until it is absent from both WordNet (the
        meanings asset) and the Wordle vocab — so the "invalid" label is trustworthy."""
        for _ in range(10000):
            base = rng.choice(self.valid_words)
            cand = perturb(base, rng)
            if (len(cand) >= _MIN_PSEUDO_LEN and cand.isalpha()
                    and cand not in self._meanings and cand not in self.wordle):
                return cand
        raise RuntimeError("could not synthesize a pseudo-word")

    def sample(self, mode: Mode) -> str:
        """A random valid (real) word from the requested pool."""
        pool = self.train if mode == "train" else self.val
        return self._rng.choice(pool)
