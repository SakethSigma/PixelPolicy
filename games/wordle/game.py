"""Pure Wordle game logic.

This module has **no** web/FastAPI dependency. It is the single source of truth
for the environment and is used two ways:

- ``server.py`` wraps it in an HTTP API for inference / eval / interactive play.
- Training rollouts ``from games.wordle.game import WordleGame, WordBank`` and step
  thousands of envs in-process with zero network cost.

The environment knows nothing about models, tokens, or rewards — reward is an
RL-specific concern that lives in ``training/``. A human could play it and get the
same per-letter feedback.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

WORD_LENGTH = 5
DEFAULT_MAX_ROUNDS = 6

_DIR = Path(__file__).parent
_WORDS_FILE = _DIR / "words.txt"          # full allowed-guess list + split input
_TRAIN_FILE = _DIR / "train_words.txt"    # committed split artifact (source of truth)
_VAL_FILE = _DIR / "val_words.txt"        # committed split artifact (source of truth)

# Split parameters. Assignment is a pure function of the word string via sha256,
# so a given word ALWAYS lands in the same pool — independent of list order, list
# size, Python version, or PYTHONHASHSEED. Changing these re-derives the split, so
# they must stay fixed for results to be comparable across runs/models/months.
_VAL_FRACTION = 0.2
_SPLIT_BUCKETS = 1000

Mode = Literal["train", "val"]
Status = Literal["in_progress", "won", "lost"]


def assign_pool(word: str, val_fraction: float = _VAL_FRACTION) -> Mode:
    """Deterministically assign a word to ``"train"`` or ``"val"``.

    The bucket is ``sha256(word) % _SPLIT_BUCKETS``; words whose bucket falls in the
    bottom ``val_fraction`` go to val. Because it depends only on the word's bytes,
    the assignment is identical on every machine, every run, forever — and adding or
    removing other words never moves an existing word's pool.
    """
    digest = hashlib.sha256(word.strip().lower().encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % _SPLIT_BUCKETS
    return "val" if bucket < val_fraction * _SPLIT_BUCKETS else "train"


class GameOverError(Exception):
    """Raised when a guess is submitted to a game that has already ended."""


class LetterFeedback(str, Enum):
    CORRECT = "✓"       # right letter, right position (green)
    WRONG_POS = "-"     # right letter, wrong position (yellow)
    WRONG_LETTER = "x"  # letter not in the word, or all copies accounted for (gray)


class InvalidReason(str, Enum):
    """Why a guess was rejected. An invalid guess still consumes a round, but it
    carries no per-letter feedback (so a non-word can't probe letters for free)."""

    LENGTH = "inadequate length"   # wrong length or non-alphabetic
    VOCAB = "out of vocabulary"    # right shape, but not an allowed word


class RoundResult(BaseModel):
    guess: str                              # the guess, uppercased
    feedback: list[LetterFeedback] = Field(default_factory=list)  # empty when invalid
    error: Optional[InvalidReason] = None   # set iff the guess was invalid


class GameState(BaseModel):
    game_id: str
    max_rounds: int = DEFAULT_MAX_ROUNDS
    current_round: int = 0          # number of guesses made so far
    rounds: list[RoundResult] = Field(default_factory=list)
    status: Status = "in_progress"
    target: Optional[str] = None    # revealed only once the game is over


def compute_feedback(guess: str, target: str) -> list[LetterFeedback]:
    """Per-letter Wordle feedback with correct duplicate handling.

    Two passes so that duplicate letters are scored against the *remaining*
    unmatched copies in the target. Example — guess ``PUPPY`` vs target ``APPLE``:
    index 2 P→CORRECT, index 0 P→WRONG_POS (one P left), the other P→WRONG_LETTER.
    """
    guess = guess.upper()
    target = target.upper()
    feedback: list[Optional[LetterFeedback]] = [None] * len(guess)
    remaining = Counter(target)

    # Pass 1: greens first, consuming a copy from the target for each match.
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback[i] = LetterFeedback.CORRECT
            remaining[g] -= 1

    # Pass 2: yellows/grays for the rest, against whatever copies are left.
    for i, g in enumerate(guess):
        if feedback[i] is not None:
            continue
        if remaining[g] > 0:
            feedback[i] = LetterFeedback.WRONG_POS
            remaining[g] -= 1
        else:
            feedback[i] = LetterFeedback.WRONG_LETTER

    return feedback  # type: ignore[return-value]


class WordleGame:
    """A single Wordle episode. The target is held privately until the game ends.

    Validation is owned here (not by callers) so the in-process and HTTP paths
    cannot diverge: an invalid guess always consumes a round and reports a reason.
    ``validate_word`` is the vocabulary check (e.g. ``WordBank.is_valid``); when it
    is ``None`` only the structural length/alpha tier is enforced — handy for unit
    tests that exercise feedback mechanics without a word bank.
    """

    def __init__(
        self,
        target: str,
        game_id: str,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        validate_word: Optional[Callable[[str], bool]] = None,
    ):
        self.target = target.upper()
        self.game_id = game_id
        self.max_rounds = max_rounds
        self.validate_word = validate_word
        self.rounds: list[RoundResult] = []
        self.status: Status = "in_progress"

    @property
    def current_round(self) -> int:
        return len(self.rounds)

    def guess(self, word: str) -> RoundResult:
        """Score a guess, append it to history, and update win/lose status.

        A guess that is the wrong length / non-alphabetic, or not an allowed word
        (when a validator is configured), still consumes a round but carries no
        feedback — only an ``error`` reason.
        """
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")

        w = word.strip().upper()
        if len(w) != WORD_LENGTH or not w.isalpha():
            result = RoundResult(guess=w, error=InvalidReason.LENGTH)
        elif self.validate_word is not None and not self.validate_word(w):
            result = RoundResult(guess=w, error=InvalidReason.VOCAB)
        else:
            result = RoundResult(guess=w, feedback=compute_feedback(w, self.target))
        self.rounds.append(result)

        # NB: guard on ``result.feedback`` — ``all([])`` is True, so without it an
        # invalid (empty-feedback) round would falsely register as a win.
        if result.feedback and all(f is LetterFeedback.CORRECT for f in result.feedback):
            self.status = "won"
        elif len(self.rounds) >= self.max_rounds:
            self.status = "lost"

        return result

    def state(self) -> GameState:
        """Serializable snapshot. Target is exposed only after the game ends."""
        return GameState(
            game_id=self.game_id,
            max_rounds=self.max_rounds,
            current_round=self.current_round,
            rounds=self.rounds,
            status=self.status,
            target=self.target if self.status != "in_progress" else None,
        )


def _load_words(path: Path) -> list[str]:
    """Load valid 5-letter words, skipping the comment header and any junk lines."""
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if not word or word.startswith("#"):
            continue
        if len(word) == WORD_LENGTH and word.isalpha():
            words.append(word)
    return words


def generate_split_files(
    words_file: Path = _WORDS_FILE,
    train_file: Path = _TRAIN_FILE,
    val_file: Path = _VAL_FILE,
    val_fraction: float = _VAL_FRACTION,
) -> tuple[int, int]:
    """(Re)generate the committed ``train_words.txt`` / ``val_words.txt`` artifacts.

    Run deliberately (see ``split.py``) — not at server startup. The output is the
    source of truth the server loads, so regenerating it is the only way the split
    can ever change. Words are written sorted, one per line. Returns (n_train, n_val).
    """
    train, val = [], []
    for word in sorted(_load_words(words_file)):
        (val if assign_pool(word, val_fraction) == "val" else train).append(word)
    train_file.write_text("\n".join(train) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(val) + "\n", encoding="utf-8")
    return len(train), len(val)


class WordBank:
    """Loads the committed train/val word lists and samples target words.

    The split is read straight from ``train_words.txt`` / ``val_words.txt`` — fixed,
    version-controlled files — so pool membership is byte-for-byte identical on every
    run, every model, every month. (Those files are produced by
    :func:`generate_split_files`, which uses the order-independent :func:`assign_pool`.)
    ``all`` is the union and is used to validate guesses regardless of mode.
    """

    def __init__(self, train_file: Path = _TRAIN_FILE, val_file: Path = _VAL_FILE):
        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(
                f"Split files missing ({train_file}, {val_file}). "
                "Generate them with: python -m games.wordle.split"
            )
        self.train: list[str] = _load_words(train_file)
        self.val: list[str] = _load_words(val_file)
        if not self.train or not self.val:
            raise ValueError("Train and val word lists must both be non-empty")

        self.all: set[str] = set(self.train) | set(self.val)

        # Per-game target sampling is intentionally random (a fresh target each
        # episode); only pool *membership* is fixed.
        self._rng = random.Random()

    def sample(self, mode: Mode) -> str:
        """Pick a random target word from the requested pool."""
        pool = self.train if mode == "train" else self.val
        return self._rng.choice(pool)

    def is_valid(self, word: str) -> bool:
        """Whether ``word`` is an allowed guess (valid in both modes)."""
        return word.strip().lower() in self.all
