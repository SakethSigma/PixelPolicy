"""FastAPI server wrapping the pure Character-set environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-charset uvicorn games.charset.server:app --reload

The single-turn API is ``/reset`` then ``/step``. Pin a challenge with ``{"word": "cat,planet"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.charset.game import CharsetBank, CharsetGame, GameOverError, GameState, decode_words
from games.wordvocab.split import Mode

app = FastAPI(title="Charset", description="A pure used/unused character-set environment over HTTP.")

# Loaded once at startup: the shared vocabulary + the charset train/val split.
bank = CharsetBank()

# In-memory game store, keyed by game_id. Single-process only.
games: dict[str, CharsetGame] = {}


class ResetRequest(BaseModel):
    mode: Mode = "train"
    word: Optional[str] = None  # a comma-joined word list to pin (else a sampled challenge)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its (unanswered) challenge state."""
    words = decode_words(req.word) if req.word is not None else bank.sample_words(req.mode)
    game_id = str(uuid.uuid4())
    games[game_id] = CharsetGame(words=words, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the used/unused letter sets; returns the scored terminal state."""
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
