"""Pure Mistake-identification game logic (Word-skill game #8).

A **single-turn** environment: ``reset`` poses a Wordle board (past guesses + their per-letter
feedback) and a **proposed next guess**, and ``step(answer)`` scores whether the player correctly
identified the *repeated mistakes* in that proposed guess. It teaches the model to read Wordle
feedback and notice when a guess throws away information:

  - a **grey** mistake: the guess reuses a letter already proven absent (marked ``x`` before);
  - a **yellow** mistake: the guess re-places a letter in a slot already shown ``-`` (yellow)
    for it (the letter is in the word but known *not* to go there).

This is a *reasoning* game distilled from Claude (``<think>…</think><answer>…</answer>``) with
**rejection sampling**: the env computes the true error set from the feedback alone (no target
needed), so a trace is kept only if its reported errors exactly match. The single-turn ``status``
convention mirrors the other games:

    "in_progress"  -> board + proposed guess posed, no answer yet
    "correct"      -> reported mistakes (and the yes/no flag) match the truth   (the "good" status)
    "incorrect"    -> reported set wrong, flag wrong, or answer unparseable
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

GAME_NAME = "mistakeid"

Status = Literal["in_progress", "correct", "incorrect"]
_DIR = Path(__file__).parent
_CHALLENGES_FILE = _DIR / "challenges.jsonl"


def score_feedback(guess: str, target: str) -> str:
    """Standard two-pass Wordle scoring → a 5-char string of ``g``/``y``/``x`` (green/yellow/grey)."""
    g, t = guess.lower(), target.lower()
    n = len(g)
    res = ["x"] * n
    tc = Counter(t)
    for i in range(n):
        if g[i] == t[i]:
            res[i] = "g"
            tc[g[i]] -= 1
    for i in range(n):
        if res[i] == "g":
            continue
        if tc.get(g[i], 0) > 0:
            res[i] = "y"
            tc[g[i]] -= 1
    return "".join(res)


def encode_target(rounds: list[tuple[str, str]], attempt: str) -> str:
    """Encode a board + proposed guess as one target string: ``g1:f1|g2:f2;attempt``."""
    board = "|".join(f"{g.lower()}:{f}" for g, f in rounds)
    return f"{board};{attempt.lower()}"


def decode_target(target: str) -> tuple[list[tuple[str, str]], str]:
    """Split a ``"g1:f1|g2:f2;attempt"`` target into ``(rounds, attempt)``."""
    board, _, attempt = target.partition(";")
    rounds: list[tuple[str, str]] = []
    if board:
        for tok in board.split("|"):
            guess, _, fb = tok.partition(":")
            rounds.append((guess.strip().lower(), fb.strip().lower()))
    return rounds, attempt.strip().lower()


def constraints_from_rounds(rounds: list[tuple[str, str]]) -> tuple[set[str], set[tuple[str, int]]]:
    """From past feedback, derive ``(absent_letters, yellow_positions)``.

    ``absent`` = letters that were only ever grey (never green/yellow → truly not in the word).
    ``yellow_positions`` = ``(letter, index)`` pairs marked yellow (letter is in the word, but
    known not at that index).
    """
    present: set[str] = set()
    greyed: set[str] = set()
    yellow_pos: set[tuple[str, int]] = set()
    for guess, fb in rounds:
        for i, (c, f) in enumerate(zip(guess, fb)):
            if f == "g":
                present.add(c)
            elif f == "y":
                present.add(c)
                yellow_pos.add((c, i))
            elif f == "x":
                greyed.add(c)
    return greyed - present, yellow_pos


class ErrorItem(BaseModel):
    """One repeated mistake: a 1-based position, the letter, and its kind."""

    position: int
    letter: str          # UPPERCASE
    kind: Literal["grey", "yellow"]

    def key(self) -> tuple[int, str, str]:
        return (self.position, self.letter.upper(), self.kind)


def true_errors(rounds: list[tuple[str, str]], attempt: str) -> list[ErrorItem]:
    """The repeated mistakes in ``attempt`` given the board's feedback (sorted by position)."""
    absent, yellow_pos = constraints_from_rounds(rounds)
    out: list[ErrorItem] = []
    for i, ch in enumerate(attempt.lower()):
        if ch in absent:
            out.append(ErrorItem(position=i + 1, letter=ch.upper(), kind="grey"))
        elif (ch, i) in yellow_pos:
            out.append(ErrorItem(position=i + 1, letter=ch.upper(), kind="yellow"))
    return out


class Solution(BaseModel):
    """The revealed truth once the episode ends."""

    has_mistakes: bool
    errors: list[ErrorItem] = Field(default_factory=list)


class GameState(BaseModel):
    """Serializable snapshot. The solution is revealed only once the episode ends."""

    game_id: str
    rounds: list[tuple[str, str]]               # (guess, feedback) for each past round
    attempt: str                                # the proposed next guess to review
    status: Status = "in_progress"
    submitted: Optional[str] = None
    solution: Optional[Solution] = None


_FLAG = re.compile(r"mistakes?\s*[:=]?\s*(yes|no|none)", re.IGNORECASE)
_ERROR_LINE = re.compile(
    r"position\D*(\d+)\D+letter\s*[:=]?\s*([a-z])\b.*?(grey|gray|yellow)",
    re.IGNORECASE,
)


def parse_report(text: str) -> Optional[tuple[bool, set[tuple[int, str, str]]]]:
    """Parse a submitted report into ``(claims_mistakes, error_keys)``; ``None`` if no flag found.

    The flag comes from a ``mistakes: yes|no`` line; each error line is
    ``position N, letter X, grey|yellow``. Tolerant to spacing/case and ``gray`` spelling.
    """
    flag = _FLAG.search(text)
    if flag is None:
        return None
    claims = flag.group(1).lower() == "yes"
    errors: set[tuple[int, str, str]] = set()
    for pos, letter, kind in _ERROR_LINE.findall(text):
        k = "grey" if kind.lower() in ("grey", "gray") else "yellow"
        errors.add((int(pos), letter.upper(), k))
    return claims, errors


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class MistakeGame:
    """A single Mistake-identification episode. ``step`` scores once and ends the game."""

    def __init__(self, rounds: list[tuple[str, str]], attempt: str, game_id: str):
        self.rounds = [(g.lower(), f.lower()) for g, f in rounds]
        self.attempt = attempt.strip().lower()
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        """Score the report against the true error set; terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        self.status = "correct" if self._is_correct(answer) else "incorrect"
        return self.state()

    def _is_correct(self, answer: str) -> bool:
        parsed = parse_report(answer)
        if parsed is None:
            return False
        claims, reported = parsed
        truth = {e.key() for e in true_errors(self.rounds, self.attempt)}
        if not truth:
            return (not claims) and not reported          # must say "no", list nothing
        return claims and reported == truth                # must say "yes" and match exactly

    def state(self) -> GameState:
        errs = true_errors(self.rounds, self.attempt)
        return GameState(
            game_id=self.game_id,
            rounds=self.rounds,
            attempt=self.attempt,
            status=self.status,
            submitted=self.submitted,
            solution=Solution(has_mistakes=bool(errs), errors=errs) if self.status != "in_progress" else None,
        )


def load_challenges(path: Path = _CHALLENGES_FILE) -> list[dict]:
    """Read the committed challenge asset (one ``{"target","mistake"}`` JSON per line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Challenge asset missing ({path}). Generate it with: "
            "python -m games.mistakeid.build_challenges"
        )
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


class MistakeBank:
    """Loads the committed real-Wordle challenge set and balances mistake vs clean boards.

    Challenges are extracted from the original Wordle teacher trajectories (see
    ``build_challenges.py``): each is a board + the agent's actual next guess, labelled by whether
    that guess repeated a grey/yellow mistake. ``sample_targets`` returns a 50/50 mix.
    """

    def __init__(self, path: Optional[Path] = None):
        rows = load_challenges(path) if path is not None else load_challenges()
        self.mistakes: list[str] = [r["target"] for r in rows if r["mistake"]]
        self.cleans: list[str] = [r["target"] for r in rows if not r["mistake"]]
        if not self.mistakes or not self.cleans:
            raise ValueError("Need both mistake and clean challenges — rebuild challenges.jsonl")
        import random

        self._rng = random.Random()

    def sample_targets(self, n: int, mode: str, rng) -> list[str]:
        """``n`` encoded targets, 50/50 mistake/clean (capped by the available mistake supply)."""
        half = min(n // 2, len(self.mistakes), len(self.cleans))
        out = rng.sample(self.mistakes, half) + rng.sample(self.cleans, half)
        rng.shuffle(out)
        return out

    def sample(self, mode: str) -> str:
        """One random challenge (50/50 mistake/clean) — for terminal play."""
        pool = self.mistakes if self._rng.random() < 0.5 else self.cleans
        return self._rng.choice(pool)
