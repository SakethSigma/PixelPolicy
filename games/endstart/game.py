"""Pure Ends-with → starts-with game logic (Word-skill game #4).

A **single-turn** MCQ environment: given a word and five candidate words, pick the candidate
whose **first** letter equals the given word's **last** letter. It teaches first/last-character
attention. Exactly one candidate matches (the four distractors start with other letters).

Like ``games.charcount.game`` this module has **no** web/FastAPI dependency and is the single
source of truth. Ground truth is pure Python: ``word1[-1] == candidate[0]``. The single-turn
``status`` convention mirrors the other word-skill games:

    "in_progress"  -> challenge posed, no answer yet
    "correct"      -> the chosen candidate is the one that matches   (the "good" status)
    "incorrect"    -> wrong candidate, or unparseable
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from games.wordvocab.build import load_vocab
from games.wordvocab.split import Mode, assign_pool

GAME_NAME = "endstart"
N_OPTIONS = 5

Status = Literal["in_progress", "correct", "incorrect"]


def correct_option(word1: str, options: list[str]) -> Optional[str]:
    """The option whose first letter equals ``word1``'s last letter (``None`` if none)."""
    last = word1.strip().lower()[-1]
    for o in options:
        o = o.strip().lower()
        if o and o[0] == last:
            return o
    return None


def encode_target(word1: str, options: list[str]) -> str:
    """Encode the challenge as ``word1;opt1,opt2,…``."""
    return f"{word1.strip().lower()};{','.join(o.strip().lower() for o in options)}"


def decode_target(target: str) -> tuple[str, list[str]]:
    """Inverse of :func:`encode_target`."""
    w, _, opts = target.partition(";")
    return w.strip().lower(), [o.strip().lower() for o in opts.split(",") if o.strip()]


def parse_answer(text: str) -> Optional[str]:
    """Pull the chosen word out of a submitted answer; ``None`` if none is found.

    Tag-robust: parse the ``<answer>…</answer>`` body when present, else the last word.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    src = m.group(1) if m else text
    words = re.findall(r"[a-z']+", src.strip().lower())
    return words[-1] if words else None


class GameState(BaseModel):
    """Serializable snapshot. The solution is revealed only once the episode ends."""

    game_id: str
    word1: str
    options: list[str]
    status: Status = "in_progress"
    submitted: Optional[str] = None
    solution: Optional[str] = None       # the correct option


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class EndstartGame:
    """A single Ends-with → starts-with episode. ``step`` scores once and ends the game."""

    def __init__(self, word1: str, options: list[str], game_id: str):
        self.word1 = word1.strip().lower()
        self.options = [o.strip().lower() for o in options]
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score ``answer`` against the matching option; terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        parsed = parse_answer(answer)
        self.status = "correct" if (parsed is not None and parsed == correct_option(self.word1, self.options)) else "incorrect"
        return self.state()

    def state(self) -> GameState:
        return GameState(
            game_id=self.game_id,
            word1=self.word1,
            options=self.options,
            status=self.status,
            submitted=self.submitted,
            solution=correct_option(self.word1, self.options) if self.status != "in_progress" else None,
        )


class EndstartBank:
    """Loads the shared vocabulary and builds MCQ challenges with a unique answer.

    ``word1`` is drawn from the salted ``endstart`` train/val split. One option starts with
    ``word1``'s last letter (the answer); the other four start with four *distinct other* letters,
    so the answer is unique.
    """

    def __init__(self, vocab_path: Optional[Path] = None):
        words = load_vocab(vocab_path) if vocab_path is not None else load_vocab()
        self.by_first: dict[str, list[str]] = defaultdict(list)
        for w in words:
            self.by_first[w[0]].append(w)
        self.train: list[str] = []
        self.val: list[str] = []
        for w in words:
            (self.val if assign_pool(GAME_NAME, w) == "val" else self.train).append(w)
        if not self.train or not self.val:
            raise ValueError("Train and val pools must both be non-empty")
        import random

        self._rng = random.Random()

    def make_challenge(self, mode: Mode, rng) -> tuple[str, list[str]]:
        """Pick ``word1`` + 5 options (one matching, four distinct-other-letter distractors)."""
        pool = self.train if mode == "train" else self.val
        for _ in range(10000):
            word1 = rng.choice(pool)
            last = word1[-1]
            matches = [w for w in self.by_first.get(last, []) if w != word1]
            other_letters = [c for c in self.by_first if c != last]
            if not matches or len(other_letters) < N_OPTIONS - 1:
                continue
            correct = rng.choice(matches)
            distractor_letters = rng.sample(other_letters, N_OPTIONS - 1)
            distractors = [rng.choice(self.by_first[c]) for c in distractor_letters]
            options = [correct] + distractors
            rng.shuffle(options)
            return word1, options
        raise RuntimeError("could not build an endstart challenge")

    def sample(self, mode: Mode) -> tuple[str, list[str]]:
        return self.make_challenge(mode, self._rng)
