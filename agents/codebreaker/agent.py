"""The Codebreaker agent — the only game-aware code for game #10.

Multi-turn adapter mirroring ``agents/wordle/agent.py``: ``build_messages`` replays the episode as
a conversation (system → first ask → (guess, feedback)*), and :class:`CodebreakerEnv` maps the
generic ``step`` onto the client's ``guess`` verb. Programmatic game — the teacher emits a bare
``<guess>CODE</guess>`` (no ``<think>``); the student learns to turn feedback into the next guess.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.codebreaker.render import render_observation, render_round

_GUESS_TAG = re.compile(r"<guess>\s*([A-Za-z]+)\s*</guess>", re.IGNORECASE | re.DOTALL)


class CodebreakerAgent:
    """Multi-turn Codebreaker policy adapter. Stateless: a pure function of (state, history)."""

    system_prompt = (
        "You are cracking a secret code.\n"
        "\n"
        "The code has 4 slots, each one of 6 symbols (A B C D E F); symbols can repeat. "
        "After each guess every slot is scored:\n"
        "  ✓  right symbol in the right slot\n"
        "  -  right symbol but in the wrong slot\n"
        "  x  the symbol is not in the code (or all its copies are already accounted for)\n"
        "\n"
        "Use every clue: keep ✓ symbols in place, move - symbols to a different slot, and stop "
        "using x symbols. Briefly note what the clues tell you, then give your guess on its own line as:\n"
        "<guess>CODE</guess>\n"
        "where CODE is 4 symbols from A-F (e.g. <guess>ACEF</guess>). Output the <guess> tag "
        "exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]
        for turn in history:
            guess = f"<guess>{turn.action}</guess>" if turn.action else ""
            messages.append({"role": "assistant", "content": guess})
            messages.append({"role": "user", "content": self._feedback(turn.state)})
        return messages

    def parse_action(self, text: str) -> str:
        """Return the code in the last ``<guess>…</guess>`` (uppercased); ``""`` if absent."""
        tagged = _GUESS_TAG.findall(text)
        return tagged[-1].upper() if tagged else ""

    @staticmethod
    def _feedback(state: Any) -> str:
        line = render_round(state.rounds[-1])
        if state.status == "in_progress":
            remaining = state.max_rounds - state.current_round
            return f"{line}\nRounds left: {remaining}. Make your next guess."
        return line


class CodebreakerEnv:
    """Adapt a ``CodebreakerClient`` (verb: ``guess``) to the generic ``Env`` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.guess(action)

    def state(self):
        return self._client.state()
