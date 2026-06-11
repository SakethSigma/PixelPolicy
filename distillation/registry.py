"""Per-game wiring for the batch player — the ONE place a new game is added.

The batch player (`batch_play.py`) is game-agnostic: it only ever calls an agent's
``build_messages`` / ``parse_action`` / ``system_prompt`` and an env's ``step`` / ``state``.
A :class:`GameSpec` supplies the three game-specific bits — how to make an agent, how to make
an env already reset to a target, and how to sample target instances — plus the round cap.

Add a new game = add one entry to ``GAMES``. Nothing in ``batch_play.py`` changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from agents.charcount.agent import CharCountAgent, CharCountEnv
from agents.wordle.agent import WordleAgent, WordleEnv
from games.charcount.client import LocalCharCountClient
from games.charcount.game import CharCountBank
from games.wordle.client import LocalWordleClient
from games.wordle.game import WordBank


@dataclass(frozen=True)
class GameSpec:
    """Everything the generic batch player needs to drive one game type.

    - ``make_agent()`` -> a stateless agent (``build_messages`` / ``parse_action`` / ``system_prompt``).
    - ``make_env(target)`` -> an env already reset to ``target`` (``step(action)`` / ``state()``);
      ``state().status == "in_progress"`` means the game is still going.
    - ``sample_targets(n, mode, rng)`` -> ``n`` distinct targets drawn with the caller's seeded ``rng``.
    - ``max_rounds`` -> hard cap on rounds per game.
    - ``good_status`` -> the terminal status that counts as a *solved* episode (the rejection
      gate / the unified ``valid`` flag). Wordle's is ``"won"``; single-turn games use ``"correct"``.
    """

    make_agent: Callable[[], Any]
    make_env: Callable[[str], Any]
    sample_targets: Callable[[int, str, random.Random], list[str]]
    max_rounds: int
    good_status: str = "won"


def _wordle_spec() -> GameSpec:
    """Build the Wordle spec, loading the train/val word lists once (shared across the run)."""
    bank = WordBank()

    def make_env(target: str):
        env = WordleEnv(LocalWordleClient(bank))
        env.reset(word=target)  # pin the secret word; mode is irrelevant once the word is fixed
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        pool = bank.train if mode == "train" else bank.val
        if n > len(pool):
            raise ValueError(f"asked for {n} distinct {mode} targets but the pool has {len(pool)}")
        return rng.sample(pool, n)  # distinct words, deterministic for a given seed

    return GameSpec(make_agent=WordleAgent, make_env=make_env, sample_targets=sample_targets, max_rounds=6)


def _charcount_spec() -> GameSpec:
    """Build the Character-counts spec (single-turn), loading the shared vocabulary once.

    Charcount is a programmatic game (see ``distillation/programmatic.py``); this spec lets the
    generic machinery — and any future live/batch path — drive it the same way as Wordle. The
    only structural difference is ``max_rounds == 1``: one ``step`` ends the episode.
    """
    bank = CharCountBank()

    def make_env(target: str):
        env = CharCountEnv(LocalCharCountClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        pool = bank.train if mode == "train" else bank.val
        if n > len(pool):
            raise ValueError(f"asked for {n} distinct {mode} targets but the pool has {len(pool)}")
        return rng.sample(pool, n)  # distinct words, deterministic for a given seed

    return GameSpec(make_agent=CharCountAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


# name -> zero-arg factory that builds the spec (loads shared resources lazily, once per run).
GAMES: dict[str, Callable[[], GameSpec]] = {
    "wordle": _wordle_spec,
    "charcount": _charcount_spec,
}

# Stable game number per game name — a unified-schema column (wordle is 0; the six word-skill
# games are 1-6 per games/DATA_SOURCING.md). Used by the SFT writers and push normalization.
GAME_NUMBERS: dict[str, int] = {
    "wordle": 0,
    "charcount": 1,
}
