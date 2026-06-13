"""Pure Crossword-fill game logic (Word-skill game #6).

A **single-turn** environment: ``reset`` poses a clue — a WordNet definition, the word length,
and a partially-masked letter pattern (some letters revealed, the rest hidden as ``_``) — and
``step(answer)`` scores the guess and ends the episode. It teaches meaning + partial-pattern →
word retrieval (the core crossword skill). This is a *reasoning* game distilled from Claude
(``<think>…</think><answer>word</answer>``) with **rejection sampling**: a trace is kept only if
it recovers the exact seed word (and so is consistent with the revealed letters).

Like ``games.charcount.game`` this module has **no** web/FastAPI dependency and is the single
source of truth. The definition comes from the committed meanings asset
(``games/wordvocab/meanings.jsonl``); the mask is **derived deterministically from the word**, so
pinning a target word fully reconstructs the clue. The single-turn ``status`` convention mirrors
charcount/Wordle:

    "in_progress"  -> clue posed, no answer yet
    "correct"      -> answer == the seed word (and matches the revealed letters)   (the "good" status)
    "incorrect"    -> answer wrong or unparseable

The seed word is **never** exposed in the in-progress ``GameState`` — only the clue is — so the
observation a model reads cannot leak the answer; the word is revealed only in the terminal
``solution``.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from games.wordvocab.build import load_vocab
from games.wordvocab.build_meanings import load_meanings

GAME_NAME = "crossword"  # salt for the per-game vocabulary split (unused here; bank pools by source)

Status = Literal["in_progress", "correct", "incorrect"]


def make_pattern(word: str) -> str:
    """The masked pattern for ``word``: ~half the letters revealed, the rest ``_``.

    Deterministic in the word (the revealed positions are chosen by a word-seeded RNG), so the
    same target always yields the same clue — pinning a target reconstructs the whole challenge.
    """
    w = word.strip().lower()
    n = len(w)
    hidden = max(1, n // 2)          # hide ~half; keep at least one hidden and one revealed
    reveal = max(1, n - hidden)
    seed = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
    revealed = set(random.Random(seed).sample(range(n), reveal))
    return "".join(w[i] if i in revealed else "_" for i in range(n))


def matches_pattern(word: str, pattern: str) -> bool:
    """Whether ``word`` is consistent with the revealed letters of ``pattern``."""
    w = word.strip().lower()
    if len(w) != len(pattern):
        return False
    return all(p == "_" or p == c for c, p in zip(w, pattern))


def parse_answer(text: str) -> Optional[str]:
    """Pull a single candidate word out of a submitted answer; ``None`` if none is found.

    Tag-robust: parse the ``<answer>…</answer>`` body when present (so the closing tag's letters
    can't be mistaken for the answer); otherwise take the last alphabetic word.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    src = m.group(1) if m else text
    words = re.findall(r"[a-z]+", src.lower())
    return words[-1] if words else None


class Solution(BaseModel):
    """The revealed answer once the episode ends."""

    word: str
    definition: str


class GameState(BaseModel):
    """Serializable snapshot. The seed word is revealed only once the episode ends."""

    game_id: str
    definition: str
    length: int
    pattern: str                                # revealed letters + "_" for hidden, e.g. "c_a_e"
    status: Status = "in_progress"
    submitted: Optional[str] = None
    solution: Optional[Solution] = None


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class CrosswordGame:
    """A single Crossword-fill episode. ``step`` scores once and ends the game."""

    def __init__(self, word: str, definition: str, game_id: str):
        self.word = word.strip().lower()
        self.definition = definition
        self.pattern = make_pattern(self.word)
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score ``answer`` — must equal the seed word (and match the revealed letters)."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        cand = parse_answer(answer)
        correct = cand is not None and cand == self.word and matches_pattern(cand, self.pattern)
        self.status = "correct" if correct else "incorrect"
        return self.state()

    def state(self) -> GameState:
        return GameState(
            game_id=self.game_id,
            definition=self.definition,
            length=len(self.word),
            pattern=self.pattern,
            status=self.status,
            submitted=self.submitted,
            solution=(
                Solution(word=self.word, definition=self.definition)
                if self.status != "in_progress" else None
            ),
        )


class CrosswordBank:
    """Loads the meanings asset and builds clues from two word sources.

    Every crossword seed needs a definition, so candidates are the words that carry one in the
    committed ``meanings.jsonl``. Two pools: the **Wordle vocabulary** (train + val union) and
    **general** multi-length words (the rest of the shared vocab, lengths 3–20). ``sample_targets``
    draws half from each, so a run mixes familiar five-letter words with varied-length vocabulary.
    """

    def __init__(self, vocab_path: Optional[Path] = None):
        from games.wordle.game import WordBank

        self._meanings = load_meanings()
        self.wordle: set[str] = set(WordBank().all)
        vocab = load_vocab(vocab_path) if vocab_path is not None else load_vocab()
        self.wordle_words: list[str] = sorted(w for w in self.wordle if w in self._meanings)
        self.general_words: list[str] = sorted(
            w for w in vocab if w in self._meanings and w not in self.wordle
        )
        if not self.wordle_words or not self.general_words:
            raise ValueError("Empty word pool — is meanings.jsonl built?")
        self._rng = random.Random()

    def definition(self, word: str) -> Optional[str]:
        return self._meanings.get(word.strip().lower())

    def make_game(self, word: str, game_id: str) -> CrosswordGame:
        definition = self.definition(word)
        if definition is None:
            raise ValueError(f"no definition for {word!r}")
        return CrosswordGame(word=word, definition=definition, game_id=game_id)

    def sample_targets(self, n: int, mode: str, rng: random.Random) -> list[str]:
        """``n`` distinct seed words: half from the Wordle vocab, half from general words."""
        n_wordle = n // 2
        n_general = n - n_wordle
        if n_wordle > len(self.wordle_words) or n_general > len(self.general_words):
            raise ValueError(
                f"asked for {n_wordle}+{n_general} but pools have "
                f"{len(self.wordle_words)}+{len(self.general_words)}"
            )
        out = rng.sample(self.wordle_words, n_wordle) + rng.sample(self.general_words, n_general)
        rng.shuffle(out)
        return out

    def sample(self, mode: str) -> str:
        """One random seed word (either pool) — for terminal play."""
        return self._rng.choice(self.wordle_words + self.general_words)
