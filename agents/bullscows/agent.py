"""The Bulls & Cows agent — the only game-aware code for game #11.

Multi-turn adapter mirroring ``agents/wordle/agent.py``: ``build_messages`` replays the episode as
a conversation, and :class:`BullsCowsEnv` maps the generic ``step`` onto the client's ``guess``
verb. Programmatic game — the teacher emits a short worded rationale (from the bull/cow clues so
far) then ``<guess>1234</guess>``, no ``<think>`` tags. On replay only the bare ``<guess>`` is
kept (the rationale, like Wordle's ``<think>``, is dropped).
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.bullscows.render import render_observation, render_round

_GUESS_TAG = re.compile(r"<guess>\s*([0-9]+)\s*</guess>", re.IGNORECASE | re.DOTALL)


class BullsCowsAgent:
    """Multi-turn Bulls & Cows policy adapter. Stateless: a pure function of (state, history)."""

    system_prompt = (
        "You are guessing a secret 4-digit number. All four digits are different (digits 0-9).\n"
        "\n"
        "After each guess you are told:\n"
        "  bulls = how many digits are correct AND in the right position\n"
        "  cows  = how many digits are in the number but in the wrong position\n"
        "\n"
        "Use the bulls and cows from every guess to narrow it down. Briefly explain your "
        "reasoning, then give your guess on its own line as:\n"
        "<guess>NNNN</guess>\n"
        "where NNNN is four different digits (e.g. <guess>1234</guess>). Output the <guess> tag "
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
        """Return the digits in the last ``<guess>…</guess>``; ``""`` if absent."""
        tagged = _GUESS_TAG.findall(text)
        return tagged[-1] if tagged else ""

    @staticmethod
    def _feedback(state: Any) -> str:
        line = render_round(state.rounds[-1])
        if state.status == "in_progress":
            remaining = state.max_rounds - state.current_round
            return f"{line}\nRounds left: {remaining}. Make your next guess."
        return line


class BullsCowsEnv:
    """Adapt a ``BullsCowsClient`` (verb: ``guess``) to the generic ``Env`` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.guess(action)

    def state(self):
        return self._client.state()
