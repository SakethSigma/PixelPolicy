"""Generic, game-agnostic core for agents.

Nothing in this module knows about a specific game or a specific model provider. It
defines the small set of protocols and data records the rest of the layer is built on:

- :class:`GameAgent`  — the only game-aware seam (one tiny adapter per game).
- :class:`Env`        — the loop's view of a game: ``step`` + ``state`` with a ``status``.
- :class:`LLMBackend` — ``messages -> text``; the only model/network seam (inference).
- :class:`Completion` / :class:`Turn` / :class:`Trajectory` — the **text + game-state**
  record a rollout produces. Deliberately no ``token_ids`` / ``logprobs``: those belong
  to the generation engine (the RL library at train time), not to us.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ── Model I/O ────────────────────────────────────────────────────────────────────


class Completion(BaseModel):
    """One model reply. Text only — tokens/logprobs are the engine's concern."""

    text: str
    finish_reason: str | None = None
    raw: dict | None = None  # full provider response, kept for debugging


@runtime_checkable
class LLMBackend(Protocol):
    """Turns prompts into completions. Batch-first; a single call is batch-of-1.

    At inference this is :class:`agents.backend.OpenAICompatBackend`. In training the
    RL library supplies its own (its policy engine) — see ``training_integration.md``.
    """

    def generate(self, prompts: list[list[dict]], **sampling: Any) -> list[Completion]: ...


# ── Episode record ─────────────────────────────────────────────────────────────--


class Turn(BaseModel):
    """One (prompt → reply → action → resulting state) step of an episode.

    ``state`` is the game's post-action snapshot, kept loosely typed so this record
    stays game-agnostic; a game's agent reads whatever it needs from it (for Wordle,
    the latest round's feedback). ``response`` holds the model's full reply — this is
    where the model's reasoning is remembered (the env never sees it).
    """

    messages: list[dict]
    response: str
    action: str
    state: Any = None


class Trajectory(BaseModel):
    """A full episode: its turns plus the final game state (``status`` / ``target``)."""

    turns: list[Turn] = Field(default_factory=list)
    final: Any = None


# ── The game-facing seams ────────────────────────────────────────────────────────


@runtime_checkable
class EpisodeState(Protocol):
    """Minimal contract the rollout needs from a game state.

    Convention across games: ``status == "in_progress"`` means the episode is live;
    any other value is terminal. (Wordle uses ``"in_progress" | "won" | "lost"``.)
    """

    status: str


@runtime_checkable
class Env(Protocol):
    """The loop's view of one game episode: act, and observe.

    A game's own client may use a different verb (Wordle's is ``guess``); a thin
    per-game adapter (e.g. ``agents.wordle.agent.WordleEnv``) maps it onto ``step`` so
    the rollout stays game-agnostic.
    """

    def state(self) -> EpisodeState: ...
    def step(self, action: str) -> EpisodeState: ...


@runtime_checkable
class GameAgent(Protocol):
    """The only game-aware code you write: prompt construction + action parsing.

    Both methods are **pure** (no I/O, no network, no stored state) so a trainer can
    import and call them directly. ``history`` — the prior turns of *this* episode — is
    threaded in by the rollout; the agent never stores it.
    """

    system_prompt: str

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]: ...
    def parse_action(self, text: str) -> str: ...
