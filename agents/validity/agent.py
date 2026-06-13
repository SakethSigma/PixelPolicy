"""The Validity + meaning agent — the only game-aware code for word-skill game #2.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`ValidityEnv` that adapts the game's ``ValidityClient`` (verb: ``step``) onto the generic
:class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores ``history``.

``parse_action`` reconstructs the *canonical* answer (``<answer>valid</answer>`` +
``<meaning>…</meaning>`` or ``<answer>invalid</answer>``) that :func:`games.validity.game.parse_answer`
scores — so the verdict and meaning the env sees are exactly what the model wrote.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.validity.render import render_observation  # the challenge text a human also reads

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_MEANING_TAG = re.compile(r"<meaning>\s*(.*?)\s*</meaning>", re.IGNORECASE | re.DOTALL)


class ValidityAgent:
    """Single-turn Validity policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You judge whether a string is a real English word, and if so, give its meaning.\n"
        "\n"
        "You are given one string. Decide:\n"
        "  - 'valid'   if it is a real English word, or\n"
        "  - 'invalid' if it is not a real word (e.g. a misspelling or a made-up string).\n"
        "\n"
        "Rules:\n"
        "- If the word is valid, also give a short, accurate definition of it.\n"
        "- If the word is invalid, do not give a meaning.\n"
        "\n"
        "Give your final answer in exactly this format:\n"
        "<answer>valid</answer>\n"
        "<meaning>a short definition here</meaning>\n"
        "or, when the word is not real:\n"
        "<answer>invalid</answer>\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the challenge word. ``history`` is unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Reconstruct the canonical verdict (+ meaning when valid); ``""`` if no ``<answer>`` tag."""
        bodies = _ANSWER_TAG.findall(text)
        if not bodies:
            return ""
        verdict = bodies[-1].lower()
        if "invalid" in verdict:
            return "<answer>invalid</answer>"
        if "valid" in verdict:
            meanings = _MEANING_TAG.findall(text)
            meaning = meanings[-1].strip() if meanings else ""
            return f"<answer>valid</answer>\n<meaning>{meaning}</meaning>"
        return ""


class ValidityEnv:
    """Adapt a ``ValidityClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
