"""The Character-counts agent — the only game-aware code for word-skill game #1.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`CharCountEnv` that adapts the game's ``CharCountClient`` (verb: ``step``) onto the
generic :class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores
``history`` and emits one user turn — the *self-contained variant* the agent doc sanctions
(``agents/Readme.md`` → Conversation framing). Nothing about models or networks lives here.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.charcount.render import render_observation  # the challenge text a human also reads

# The model puts its final analysis inside <answer>…</answer>; parse_action reads strictly from
# the last such tag. No <think> is requested — this is a mechanical task with a short answer.
_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class CharCountAgent:
    """Single-turn Character-counts policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You analyze the characters of a word.\n"
        "\n"
        "You are given one English word. Report:\n"
        "  - its length (total number of letters),\n"
        "  - its vowels (the letters A, E, I, O, U), listed left-to-right with repeats, and\n"
        "  - its consonants (every other letter, including Y), listed left-to-right with repeats.\n"
        "\n"
        "Rules:\n"
        "- Count every letter; length must equal (number of vowels) + (number of consonants).\n"
        "- Keep repeated letters (e.g. 'banana' has vowels A A A).\n"
        "- Treat 'Y' as a consonant.\n"
        "- List the letters in UPPERCASE, separated by single spaces.\n"
        "\n"
        "Give your final answer in exactly this format, inside a single <answer> tag:\n"
        "<answer>\n"
        "length: N\n"
        "vowels (V): space-separated UPPERCASE letters\n"
        "consonants (C): space-separated UPPERCASE letters\n"
        "</answer>\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the challenge word. ``history`` is unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the body of the last ``<answer>…</answer>`` tag (the analysis block).

        No lenient fallback: if there is no ``<answer>`` tag we return ``""``, which the env
        scores ``incorrect`` — the same "malformed costs you the round" contract as Wordle.
        """
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip() if tagged else ""


class CharCountEnv:
    """Adapt a ``CharCountClient`` to the generic :class:`~agents.base.Env` protocol.

    The client's verb is already ``step``, so this just forwards ``reset`` / ``step`` /
    ``state`` — keeping ``agents/rollout.py`` game-agnostic and mirroring ``WordleEnv``.
    """

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
