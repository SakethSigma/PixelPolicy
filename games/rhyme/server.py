"""FastAPI server wrapping the pure Rhymes environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-rhyme uvicorn games.rhyme.server:app --reload

The single-turn API is ``/reset`` then ``/step``.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.rhyme.game import GameOverError, GameState, RhymeBank, RhymeGame, Variant
from games.wordvocab.split import Mode

app = FastAPI(title="Rhyme", description="A pure Rhymes environment over HTTP.")

# Loaded once at startup: the shared vocabulary + the rhyme train/val split.
bank = RhymeBank()

# In-memory game store, keyed by game_id. Single-process only.
games: dict[str, RhymeGame] = {}


class ResetRequest(BaseModel):
    mode: Mode = "train"
    word: Optional[str] = None       # pin the challenge word (testing/debugging/eval)
    variant: Variant = "free"


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its (unanswered) challenge state."""
    rng = random.Random()
    target = req.word.strip().lower() if req.word is not None else bank.sample_seed(req.mode)
    options = bank.mcq_options(target, rng) if req.variant == "mcq" else None
    game_id = str(uuid.uuid4())
    games[game_id] = RhymeGame(word=target, game_id=game_id, variant=req.variant, options=options)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit a rhyme; returns the scored terminal state."""
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
