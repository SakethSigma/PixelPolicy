"""Uniform Wordle client — one interface, two transports.

A *client* is a handle to a single episode exposing exactly three verbs:
``reset`` / ``guess`` / ``state``, each returning a :class:`GameState`. Two
implementations sit behind the same :class:`WordleClient` protocol:

- :class:`LocalWordleClient` wraps the pure :mod:`games.wordle.game` in-process —
  used by training rollouts (thousands of envs, zero network).
- :class:`HTTPWordleClient` talks to :mod:`games.wordle.server` over HTTP — used by
  eval / inference / a remote terminal.

Because both go through the same env core, a guess behaves identically on either
transport (same feedback, same round-consumption for invalid guesses). The client
knows nothing about reward — that stays on the training side. The target is read
back through ``GameState.target``, which the env reveals only once the game ends.
"""

from __future__ import annotations

import uuid
from typing import Optional, Protocol, runtime_checkable

from games.wordle.game import GameState, Mode, WordBank, WordleGame


@runtime_checkable
class WordleClient(Protocol):
    """Anything that can drive one Wordle episode. Both transports satisfy this."""

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None) -> GameState: ...
    def guess(self, word: str) -> GameState: ...
    def state(self) -> GameState: ...


class LocalWordleClient:
    """In-process episode handle wrapping :class:`WordleGame` directly.

    Share one :class:`WordBank` across many clients (load the split once, then build
    N handles for a rollout batch). One client == one episode; call :meth:`reset` to
    start a fresh one (this abandons any game in progress).
    """

    def __init__(
        self, word_bank: Optional[WordBank] = None, *, max_rounds: int | None = None
    ):
        self._bank = word_bank if word_bank is not None else WordBank()
        self._max_rounds = max_rounds
        self._game: Optional[WordleGame] = None

    def reset(self, *, mode: Mode = "train", word: Optional[str] = None) -> GameState:
        target = word.strip().lower() if word is not None else self._bank.sample(mode)
        kwargs = {} if self._max_rounds is None else {"max_rounds": self._max_rounds}
        self._game = WordleGame(
            target=target,
            game_id=str(uuid.uuid4()),
            validate_word=self._bank.is_valid,
            **kwargs,
        )
        return self._game.state()

    def guess(self, word: str) -> GameState:
        game = self._require_game()
        game.guess(word)
        return game.state()

    def state(self) -> GameState:
        return self._require_game().state()

    def _require_game(self) -> WordleGame:
        if self._game is None:
            raise RuntimeError("Call reset() before guess()/state().")
        return self._game

    # No-op context manager so trainer code can `with client:` regardless of transport.
    def __enter__(self) -> "LocalWordleClient":
        return self

    def __exit__(self, *exc) -> None:
        return None


class HTTPWordleClient:
    """Episode handle backed by the FastAPI server over HTTP.

    Uses a synchronous ``httpx.Client`` (the protocol stays sync, matching
    :class:`LocalWordleClient`). Pass an existing ``client`` to reuse a connection
    pool or to drive the app through ``fastapi.testclient.TestClient`` in tests; if
    omitted, one is created from ``base_url`` and closed by the context manager.
    """

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

    def guess(self, word: str) -> GameState:
        resp = self._http.post(
            "/guess", json={"game_id": self._require_id(), "guess": word}
        )
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

    def __enter__(self) -> "HTTPWordleClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_local_group(
    n: int, *, word: str, word_bank: Optional[WordBank] = None
) -> list[LocalWordleClient]:
    """Build ``n`` local clients all reset to the same pinned ``word``.

    Convenience for GRPO-style grouping, where a group of rollouts shares one target
    so rewards can be normalized within the group. Note: a pinned word that is not in
    the allowed vocabulary can never be legally guessed — draw ``word`` from a pool
    (e.g. ``bank.sample("train")``) rather than inventing one.
    """
    bank = word_bank if word_bank is not None else WordBank()
    clients = [LocalWordleClient(bank) for _ in range(n)]
    for c in clients:
        c.reset(word=word)
    return clients
