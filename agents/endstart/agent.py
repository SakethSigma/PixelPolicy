"""The Ends-with → starts-with agent — the only game-aware code for word-skill game #4.

Single-turn MCQ adapter (mirrors charset): a system prompt, a pure ``build_messages`` that ignores
``history``, and a strict ``parse_action``; plus :class:`EndstartEnv`. Programmatic (no `<think>`).
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.endstart.render import render_observation

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class EndstartAgent:
    """Single-turn Ends-with → starts-with policy adapter. Stateless."""

    system_prompt = (
        "You are given a word and five candidate words.\n"
        "\n"
        "Choose the single candidate whose FIRST letter is the same as the LAST letter of the "
        "given word. Exactly one candidate matches.\n"
        "\n"
        "Give your final answer inside a single <answer> tag, e.g. <answer>oasis</answer>.\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the word inside the last ``<answer>…</answer>`` (lowercased); ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip().lower() if tagged else ""


class EndstartEnv:
    """Adapt an ``EndstartClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
