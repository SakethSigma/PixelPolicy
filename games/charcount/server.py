"""FastAPI server wrapping the pure Character-counts environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-charcount uvicorn games.charcount.server:app --reload

Endpoints are ``async``; every handler is microsecond CPU work, so the event loop is never
bottlenecked and no locking is needed. The single-turn API is ``/reset`` then ``/step``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.charcount.game import CharCountBank, CharCountGame, GameOverError, GameState
from games.wordvocab.split import Mode

app = FastAPI(title="CharCount", description="A pure Character-counts environment over HTTP.")

# Loaded once at startup: the shared multi-length vocabulary + the charcount train/val split.
bank = CharCountBank()

# In-memory game store, keyed by game_id. Single-process only (see module docstring).
games: dict[str, CharCountGame] = {}


class ResetRequest(BaseModel):
    mode: Mode = "train"
    word: Optional[str] = None  # pin the challenge word (testing/debugging/eval)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its (unanswered) challenge state."""
    target = req.word.strip().lower() if req.word is not None else bank.sample(req.mode)
    game_id = str(uuid.uuid4())
    games[game_id] = CharCountGame(word=target, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the character analysis; returns the scored terminal state."""
    game = games.get(req.game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {req.game_id}")
    try:
        game.step(req.answer)
    except GameOverError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return game.state()


@app.get("/state/{game_id}", response_model=GameState)
async def state(game_id: str) -> GameState:
    """Observe the current episode state without acting."""
    game = games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {game_id}")
    return game.state()
