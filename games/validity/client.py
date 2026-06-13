"""Uniform Validity client — one interface, two transports.

A *client* is a handle to a single episode exposing ``reset`` / ``step`` / ``state``, each
returning a :class:`GameState`. :class:`LocalValidityClient` (in-process, for rollouts) and
:class:`HTTPValidityClient` (over the FastAPI server) both delegate to the same
:mod:`games.validity.game` core, so a step behaves identically on either transport.

Note ``reset`` may pin any ``word`` — a real word *or* a pseudo-word — since the validity oracle
is computed from the word itself; ``kind`` is a convenience that lets a caller ask the bank for a
fresh valid or invalid challenge.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional, Protocol, runtime_checkable

from games.validity.game import GameState, ValidityBank, ValidityGame
from games.wordvocab.split import Mode

Kind = Literal["valid", "invalid"]


@runtime_checkable
class ValidityClient(Protocol):
    """Anything that can drive one Validity episode. Both transports satisfy this."""

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None,
              kind: Kind = "valid") -> GameState: ...
    def step(self, answer: str) -> GameState: ...
    def state(self) -> GameState: ...


class LocalValidityClient:
    """In-process episode handle wrapping :class:`ValidityGame` directly."""

    def __init__(self, bank: Optional[ValidityBank] = None):
        self._bank = bank if bank is not None else ValidityBank()
        self._game: Optional[ValidityGame] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None,
              kind: Kind = "valid") -> GameState:
        if word is not None:
            target = word.strip().lower()
        elif kind == "invalid":
            import random

            target = self._bank.make_pseudo_word(random.Random())
        else:
            target = self._bank.sample(mode)
        self._game = ValidityGame(word=target, game_id=str(uuid.uuid4()))
        return self._game.state()

    def step(self, answer: str) -> GameState:
        return self._require_game().step(answer)

    def state(self) -> GameState:
        return self._require_game().state()

    def _require_game(self) -> ValidityGame:
        if self._game is None:
            raise RuntimeError("Call reset() before step()/state().")
        return self._game

    def __enter__(self) -> "LocalValidityClient":
        return self

    def __exit__(self, *exc) -> None:
        return None


class HTTPValidityClient:
    """Episode handle backed by the FastAPI server over HTTP (sync ``httpx``)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, client=None):
        import httpx  # local import: only HTTP users pay for httpx

        self._owns_client = client is None
        self._http = client if client is not None else httpx.Client(base_url=base_url)
        self._game_id: Optional[str] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None,
              kind: Kind = "valid") -> GameState:
        body: dict = {"mode": mode, "kind": kind}
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

    def __enter__(self) -> "HTTPValidityClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
