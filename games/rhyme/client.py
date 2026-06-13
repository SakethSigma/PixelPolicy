"""Uniform Rhymes client — one interface, two transports.

A *client* is a handle to a single episode exposing ``reset`` / ``step`` / ``state``, each
returning a :class:`GameState`. :class:`LocalRhymeClient` (in-process, for rollouts) and
:class:`HTTPRhymeClient` (over the FastAPI server) both delegate to the same
:mod:`games.rhyme.game` core, so a step behaves identically on either transport.
"""

from __future__ import annotations

import uuid
from typing import Optional, Protocol, runtime_checkable

from games.rhyme.game import GameState, RhymeBank, RhymeGame, Variant
from games.wordvocab.split import Mode


@runtime_checkable
class RhymeClient(Protocol):
    """Anything that can drive one Rhymes episode. Both transports satisfy this."""

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None,
              variant: Variant = "free") -> GameState: ...
    def step(self, answer: str) -> GameState: ...
    def state(self) -> GameState: ...


class LocalRhymeClient:
    """In-process episode handle wrapping :class:`RhymeGame` directly.

    Share one :class:`RhymeBank` across many clients (load the vocab once, build N handles).
    """

    def __init__(self, bank: Optional[RhymeBank] = None):
        self._bank = bank if bank is not None else RhymeBank()
        self._game: Optional[RhymeGame] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None,
              variant: Variant = "free") -> GameState:
        import random

        rng = random.Random()
        target = word.strip().lower() if word is not None else self._bank.sample_seed(mode)
        options = self._bank.mcq_options(target, rng) if variant == "mcq" else None
        self._game = RhymeGame(word=target, game_id=str(uuid.uuid4()),
                               variant=variant, options=options)
        return self._game.state()

    def step(self, answer: str) -> GameState:
        return self._require_game().step(answer)

    def state(self) -> GameState:
        return self._require_game().state()

    def _require_game(self) -> RhymeGame:
        if self._game is None:
            raise RuntimeError("Call reset() before step()/state().")
        return self._game

    def __enter__(self) -> "LocalRhymeClient":
        return self

    def __exit__(self, *exc) -> None:
        return None


class HTTPRhymeClient:
    """Episode handle backed by the FastAPI server over HTTP (sync ``httpx``)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, client=None):
        import httpx  # local import: only HTTP users pay for httpx

        self._owns_client = client is None
        self._http = client if client is not None else httpx.Client(base_url=base_url)
        self._game_id: Optional[str] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None,
              variant: Variant = "free") -> GameState:
        body: dict = {"mode": mode, "variant": variant}
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

    def __enter__(self) -> "HTTPRhymeClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
