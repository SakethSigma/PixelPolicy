"""Uniform Character-set client — one interface, two transports.

A *client* is a handle to a single episode exposing ``reset`` / ``step`` / ``state``, each
returning a :class:`GameState`. The challenge is a *list of words*; ``reset``'s ``word`` argument
is the encoded comma-joined target (so the same client signature as the other games works, and
the registry/batch driver can pin a challenge). :class:`LocalCharsetClient` and
:class:`HTTPCharsetClient` both delegate to the same :mod:`games.charset.game` core.
"""

from __future__ import annotations

import uuid
from typing import Optional, Protocol, runtime_checkable

from games.charset.game import CharsetBank, CharsetGame, GameState, decode_words
from games.wordvocab.split import Mode


@runtime_checkable
class CharsetClient(Protocol):
    """Anything that can drive one Character-set episode. Both transports satisfy this."""

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None) -> GameState: ...
    def step(self, answer: str) -> GameState: ...
    def state(self) -> GameState: ...


class LocalCharsetClient:
    """In-process episode handle wrapping :class:`CharsetGame` directly."""

    def __init__(self, bank: Optional[CharsetBank] = None):
        self._bank = bank if bank is not None else CharsetBank()
        self._game: Optional[CharsetGame] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None) -> GameState:
        words = decode_words(word) if word is not None else self._bank.sample_words(mode)
        self._game = CharsetGame(words=words, game_id=str(uuid.uuid4()))
        return self._game.state()

    def step(self, answer: str) -> GameState:
        return self._require_game().step(answer)

    def state(self) -> GameState:
        return self._require_game().state()

    def _require_game(self) -> CharsetGame:
        if self._game is None:
            raise RuntimeError("Call reset() before step()/state().")
        return self._game

    def __enter__(self) -> "LocalCharsetClient":
        return self

    def __exit__(self, *exc) -> None:
        return None


class HTTPCharsetClient:
    """Episode handle backed by the FastAPI server over HTTP (sync ``httpx``)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, client=None):
        import httpx  # local import: only HTTP users pay for httpx

        self._owns_client = client is None
        self._http = client if client is not None else httpx.Client(base_url=base_url)
        self._game_id: Optional[str] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None) -> GameState:
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

    def __enter__(self) -> "HTTPCharsetClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
