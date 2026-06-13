"""Uniform Bulls & Cows client — one interface, two transports (multi-turn, verb ``guess``)."""

from __future__ import annotations

import uuid
from typing import Optional, Protocol, runtime_checkable

from games.bullscows.game import BullsCowsBank, BullsCowsGame, GameState


@runtime_checkable
class BullsCowsClient(Protocol):
    def reset(self, *, mode: str = "train", word: Optional[str] = None) -> GameState: ...
    def guess(self, code: str) -> GameState: ...
    def state(self) -> GameState: ...


class LocalBullsCowsClient:
    """In-process episode handle wrapping :class:`BullsCowsGame` directly."""

    def __init__(self, bank: Optional[BullsCowsBank] = None):
        self._bank = bank if bank is not None else BullsCowsBank()
        self._game: Optional[BullsCowsGame] = None

    def reset(self, *, mode: str = "train", word: Optional[str] = None) -> GameState:
        secret = word.strip() if word is not None else self._bank.sample(mode)
        self._game = BullsCowsGame(secret=secret, game_id=str(uuid.uuid4()))
        return self._game.state()

    def guess(self, code: str) -> GameState:
        self._require_game().guess(code)
        return self._game.state()

    def state(self) -> GameState:
        return self._require_game().state()

    def _require_game(self) -> BullsCowsGame:
        if self._game is None:
            raise RuntimeError("Call reset() before guess()/state().")
        return self._game

    def __enter__(self) -> "LocalBullsCowsClient":
        return self

    def __exit__(self, *exc) -> None:
        return None


class HTTPBullsCowsClient:
    """Episode handle backed by the FastAPI server over HTTP (sync ``httpx``)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, client=None):
        import httpx

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

    def guess(self, code: str) -> GameState:
        resp = self._http.post("/guess", json={"game_id": self._require_id(), "guess": code})
        resp.raise_for_status()
        return GameState.model_validate(resp.json())

    def state(self) -> GameState:
        resp = self._http.get(f"/state/{self._require_id()}")
        resp.raise_for_status()
        return GameState.model_validate(resp.json())

    def _require_id(self) -> str:
        if self._game_id is None:
            raise RuntimeError("Call reset() before guess()/state().")
        return self._game_id

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> "HTTPBullsCowsClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
