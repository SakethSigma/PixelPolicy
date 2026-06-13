"""FastAPI server wrapping the pure Crossword-fill environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-crossword uvicorn games.crossword.server:app --reload

The single-turn API is ``/reset`` then ``/step``. Pin a seed word with ``{"word": "crane"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.crossword.game import CrosswordBank, CrosswordGame, GameOverError, GameState

app = FastAPI(title="Crossword", description="A pure Crossword-fill environment over HTTP.")

# Loaded once at startup: the meanings asset + the Wordle/general word pools.
bank = CrosswordBank()

# In-memory game store, keyed by game_id. Single-process only.
games: dict[str, CrosswordGame] = {}


class ResetRequest(BaseModel):
    mode: str = "train"
    word: Optional[str] = None  # pin the seed word (the clue is derived from it)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its clue (definition + length + masked pattern)."""
    target = req.word.strip().lower() if req.word is not None else bank.sample(req.mode)
    game_id = str(uuid.uuid4())
    games[game_id] = bank.make_game(target, game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the solved word; returns the scored terminal state."""
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
