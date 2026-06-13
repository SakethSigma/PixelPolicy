"""FastAPI server wrapping the pure Validity environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-validity uvicorn games.validity.server:app --reload

The single-turn API is ``/reset`` then ``/step``.
"""

from __future__ import annotations

import random
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.validity.game import GameOverError, GameState, ValidityBank, ValidityGame
from games.wordvocab.split import Mode

app = FastAPI(title="Validity", description="A pure word-validity + meaning environment over HTTP.")

# Loaded once at startup: the Wordle vocabulary + the committed meanings asset.
bank = ValidityBank()

# In-memory game store, keyed by game_id. Single-process only.
games: dict[str, ValidityGame] = {}


class ResetRequest(BaseModel):
    mode: Mode = "train"
    word: Optional[str] = None              # pin the challenge word (real or pseudo)
    kind: Literal["valid", "invalid"] = "valid"


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its (unanswered) challenge state."""
    if req.word is not None:
        target = req.word.strip().lower()
    elif req.kind == "invalid":
        target = bank.make_pseudo_word(random.Random())
    else:
        target = bank.sample(req.mode)
    game_id = str(uuid.uuid4())
    games[game_id] = ValidityGame(word=target, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the verdict (+ meaning when valid); returns the scored terminal state."""
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
