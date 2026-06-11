"""The unified SFT row schema shared across every game and both producers.

One byte-shape for every sample — Wordle (Claude-distilled) and the new word-skill games
(programmatic or distilled). The columns the user asked to standardize on are ``game_name``,
``game_no``, ``round``, and ``valid``; the rest carry the prompt/completion and provenance.

Both writers (``distillation/batch_play.py`` for reasoning games, ``distillation/programmatic.py``
for programmatic ones) emit :func:`sft_row`. ``push.py`` upgrades any *legacy* row (the original
Wordle files, whose ``game`` field was actually the episode index) via :func:`normalize_legacy`,
so old and new files combine without re-running expensive rollouts.
"""

from __future__ import annotations

import re
from typing import Any

# A row's column order (push.py adds `source` on top of these for file provenance).
UNIFIED_COLUMNS = [
    "game_name", "game_no", "round", "valid", "target",
    "system", "messages", "completion", "completion_no_think", "has_think", "episode",
]

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop the ``<think>…</think>`` block, leaving just the final answer."""
    return _THINK.sub("", text)


def sft_row(
    *,
    game_name: str,
    game_no: int,
    round: int,
    target: str,
    system: str,
    messages: list[dict],
    completion: str,
    valid: bool = True,
    episode: int = 0,
) -> dict[str, Any]:
    """Build one unified SFT row. ``completion_no_think`` / ``has_think`` are derived."""
    return {
        "game_name": game_name,
        "game_no": game_no,
        "round": round,
        "valid": valid,
        "target": target,
        "system": system,
        "messages": messages,
        "completion": completion,
        "completion_no_think": strip_think(completion),
        "has_think": "<think>" in completion,
        "episode": episode,
    }


def normalize_legacy(row: dict, *, game_name: str, game_no: int) -> dict:
    """Upgrade a row to the unified schema, backfilling any missing common columns.

    Handles the original Wordle SFT rows whose ``game`` field was the *episode index*: it is
    renamed to ``episode`` and the real game identity (``game_name`` / ``game_no``) is supplied
    by the caller (from the file's source). Rows already in the unified schema pass through
    unchanged. ``valid`` defaults to ``True`` — every committed row is a kept (correct) sample.
    """
    out = dict(row)
    if "game_name" not in out:
        # Legacy: `game` held the episode index. Move it; set the true identity.
        if "game" in out and "episode" not in out:
            out["episode"] = out.pop("game")
        out["game_name"] = game_name
        out["game_no"] = game_no
    out.setdefault("episode", 0)
    out.setdefault("valid", True)
    out.setdefault("round", 1)
    if "completion" in out:
        out.setdefault("completion_no_think", strip_think(out["completion"]))
        out.setdefault("has_think", "<think>" in out["completion"])
    return out
