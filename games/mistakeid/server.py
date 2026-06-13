"""FastAPI server wrapping the pure Mistake-identification environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-mistakeid uvicorn games.mistakeid.server:app --reload

The single-turn API is ``/reset`` then ``/step``. Pin a challenge with the encoded target string
``{"word": "crane:xxxxx|plate:xy gx...;slate"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.mistakeid.game import GameOverError, GameState, MistakeBank, MistakeGame, decode_target

app = FastAPI(title="MistakeID", description="A pure Wordle mistake-identification environment over HTTP.")

# Loaded once at startup: the committed real-Wordle challenge set.
bank = MistakeBank()

# In-memory game store, keyed by game_id. Single-process only.
games: dict[str, MistakeGame] = {}


class ResetRequest(BaseModel):
    mode: str = "train"
    word: Optional[str] = None  # an encoded "board;attempt" target to pin (else a sampled one)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its (unanswered) board + proposed-guess state."""
    target = req.word if req.word is not None else bank.sample(req.mode)
    rounds, attempt = decode_target(target)
    game_id = str(uuid.uuid4())
    games[game_id] = MistakeGame(rounds=rounds, attempt=attempt, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the mistake report; returns the scored terminal state."""
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
