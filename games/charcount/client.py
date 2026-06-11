"""Uniform Character-counts client — one interface, two transports.

A *client* is a handle to a single episode exposing three verbs: ``reset`` / ``step`` /
``state``, each returning a :class:`GameState`. (Single-turn games use ``step`` as their verb;
Wordle's is ``guess``.) Two implementations sit behind the same :class:`CharCountClient`
protocol — :class:`LocalCharCountClient` (in-process, for rollouts) and
:class:`HTTPCharCountClient` (over the FastAPI server). Both delegate to the same
:mod:`games.charcount.game` core, so a step behaves identically on either transport — proven by
a field-for-field parity test.
"""

from __future__ import annotations

import uuid
from typing import Optional, Protocol, runtime_checkable

from games.charcount.game import CharCountBank, CharCountGame, GameState
from games.wordvocab.split import Mode


@runtime_checkable
class CharCountClient(Protocol):
    """Anything that can drive one Character-counts episode. Both transports satisfy this."""

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None) -> GameState: ...
    def step(self, answer: str) -> GameState: ...
    def state(self) -> GameState: ...


class LocalCharCountClient:
    """In-process episode handle wrapping :class:`CharCountGame` directly.

    Share one :class:`CharCountBank` across many clients (load the vocab once, then build N
    handles). One client == one episode; :meth:`reset` starts a fresh one.
    """

    def __init__(self, bank: Optional[CharCountBank] = None):
        self._bank = bank if bank is not None else CharCountBank()
        self._game: Optional[CharCountGame] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None) -> GameState:
        target = word.strip().lower() if word is not None else self._bank.sample(mode)
        self._game = CharCountGame(word=target, game_id=str(uuid.uuid4()))
        return self._game.state()

    def step(self, answer: str) -> GameState:
        game = self._require_game()
        return game.step(answer)

    def state(self) -> GameState:
        return self._require_game().state()

    def _require_game(self) -> CharCountGame:
        if self._game is None:
            raise RuntimeError("Call reset() before step()/state().")
        return self._game

    def __enter__(self) -> "LocalCharCountClient":
        return self

    def __exit__(self, *exc) -> None:
        return None


class HTTPCharCountClient:
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

    def __enter__(self) -> "HTTPCharCountClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
