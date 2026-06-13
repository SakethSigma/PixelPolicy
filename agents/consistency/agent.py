"""The Candidate-consistency agent — the only game-aware code for game #12.

Single-turn yes/no adapter: a system prompt, a pure ``build_messages`` that ignores ``history``,
and a ``parse_action`` that returns the verdict; plus :class:`ConsistencyEnv`. Programmatic.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.consistency.render import render_observation

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class ConsistencyAgent:
    """Single-turn Candidate-consistency policy adapter. Stateless."""

    system_prompt = (
        "You are checking whether a candidate word is still possible in a game of Wordle.\n"
        "\n"
        "You are shown the clues so far — past guesses, each scored per letter:\n"
        "  ✓  the letter is correct and in the right position\n"
        "  -  the letter is in the word but in the wrong position\n"
        "  x  the letter is not in the word\n"
        "\n"
        "A candidate word is POSSIBLE only if it is consistent with every clue: it keeps each ✓ "
        "letter in place, contains each - letter but in a different position, and respects the "
        "counts implied by the x marks. Decide whether the given candidate could still be the "
        "secret word.\n"
        "\n"
        "Briefly explain how you checked the clues, then answer inside a single <answer> tag: "
        "<answer>yes</answer> or <answer>no</answer>. Output the <answer> tag exactly once, as "
        "the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the verdict inside the last ``<answer>…</answer>`` (lowercased); ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip().lower() if tagged else ""


class ConsistencyEnv:
    """Adapt a ``ConsistencyClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
