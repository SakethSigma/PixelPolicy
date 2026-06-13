"""Uniform Mistake-identification client — one interface, two transports.

A *client* is a handle to a single episode exposing ``reset`` / ``step`` / ``state``. ``reset``'s
``word`` argument is the encoded board+attempt target (``"g1:f1|g2:f2;attempt"``), so the same
client signature as the other games works and the registry/batch driver can pin a challenge.
:class:`LocalMistakeClient` and :class:`HTTPMistakeClient` both delegate to the same
:mod:`games.mistakeid.game` core.
"""

from __future__ import annotations

import uuid
from typing import Optional, Protocol, runtime_checkable

from games.mistakeid.game import GameState, MistakeBank, MistakeGame, decode_target


@runtime_checkable
class MistakeClient(Protocol):
    """Anything that can drive one Mistake-identification episode. Both transports satisfy this."""

    def reset(self, *, mode: str = "train", word: Optional[str] = None) -> GameState: ...
    def step(self, answer: str) -> GameState: ...
    def state(self) -> GameState: ...


class LocalMistakeClient:
    """In-process episode handle wrapping :class:`MistakeGame` directly."""

    def __init__(self, bank: Optional[MistakeBank] = None):
        self._bank = bank if bank is not None else MistakeBank()
        self._game: Optional[MistakeGame] = None

    def reset(self, *, mode: str = "train", word: Optional[str] = None) -> GameState:
        target = word if word is not None else self._bank.sample(mode)
        rounds, attempt = decode_target(target)
        self._game = MistakeGame(rounds=rounds, attempt=attempt, game_id=str(uuid.uuid4()))
        return self._game.state()

    def step(self, answer: str) -> GameState:
        return self._require_game().step(answer)

    def state(self) -> GameState:
        return self._require_game().state()

    def _require_game(self) -> MistakeGame:
        if self._game is None:
            raise RuntimeError("Call reset() before step()/state().")
        return self._game

    def __enter__(self) -> "LocalMistakeClient":
        return self

    def __exit__(self, *exc) -> None:
        return None


class HTTPMistakeClient:
    """Episode handle backed by the FastAPI server over HTTP (sync ``httpx``)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, client=None):
        import httpx  # local import: only HTTP users pay for httpx

        self._owns_client = client is None
        self._http = client if client is not None else httpx.Client(base_url=base_url)
        self._game_id: Optional[str] = None

    def reset(self, *, mode: str = "train", word: Optional[str] = None) -> GameState:
        body: dict = {"mode": mode}
        if word is not None:
            body["word"] = word
        resp = self._http.post("/reset", json=body)
        resp.raise_for_status()
        state = GameState.model_validate(resp.json())
        self._game_id = state.game_id
        return state

    def step(self, answer: str) -> GameState:
        resp = self._http.post("/step", json={"game_id": self._require_id(), "answer": answer})
        resp.raise_for_status()
        return GameState.model_validate(resp.json())

    def state(self) -> GameState:
        resp = self._http.get(f"/state/{self._require_id()}")
        resp.raise_for_status()
        return GameState.model_validate(resp.json())

    def _require_id(self) -> str:
        if self._game_id is None:
            raise RuntimeError("Call reset() before step()/state().")
        return self._game_id

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> "HTTPMistakeClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
