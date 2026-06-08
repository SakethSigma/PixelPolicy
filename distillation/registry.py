"""Per-game wiring — the ONLY place a new game is plugged into distillation.

Keeps generate.py / dataset.py game-agnostic (the repo's hard rule: a new game must not
require changes to shared code). Each entry says how to build that game's agent, its
shared resources, and a freshly-reset env for one episode. Adding a game later = add one
GAMES entry; its agent already lives in agents/<game>/.

Reuses, unchanged:
  - WordleAgent, WordleEnv   from agents/wordle/agent.py
  - LocalWordleClient        from games/wordle/client.py   (verb: guess, mapped to step by WordleEnv)
  - WordBank                 from games/wordle/game.py      (.sample(mode), .is_valid)
"""

from __future__ import annotations

# TODO imports:
#   from dataclasses import dataclass
#   from typing import Any, Callable
#   from agents.wordle.agent import WordleAgent, WordleEnv
#   from games.wordle.client import LocalWordleClient
#   from games.wordle.game import WordBank


# TODO: @dataclass(frozen=True)
# class GameSpec:
#     make_agent: Callable[[], Any]                 # () -> a fresh GameAgent (stateless)
#     make_bank:  Callable[[], Any]                 # () -> shared resources (Wordle: a WordBank) loaded once
#     reset_env:  Callable[..., Any]                # (bank, *, mode, word) -> an Env already reset to a target
#     sample_target: Callable[..., Any] = None      # (bank, mode) -> a target word, for logging/grouping (optional)


# TODO: a small helper that builds + resets a Wordle env, e.g.
# def _wordle_reset(bank, *, mode="train", word=None):
#     env = WordleEnv(LocalWordleClient(bank))
#     env.reset(mode=mode, word=word)     # word pins the target (deterministic smoke tests); else bank.sample(mode)
#     return env


# TODO: the registry itself
# GAMES: dict[str, GameSpec] = {
#     "wordle": GameSpec(
#         make_agent=WordleAgent,
#         make_bank=WordBank,
#         reset_env=_wordle_reset,
#         sample_target=lambda bank, mode: bank.sample(mode),
#     ),
# }
