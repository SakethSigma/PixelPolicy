"""FastAPI server wrapping the pure Anagrams environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-anagram uvicorn games.anagram.server:app --reload

The single-turn API is ``/reset`` then ``/step``. Pin a pair with ``{"word": "listen,silent"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.anagram.game import AnagramBank, AnagramGame, GameOverError, GameState, decode_pair
from games.wordvocab.split import Mode

app = FastAPI(title="Anagram", description="A pure Anagrams environment over HTTP.")

# Loaded once at startup: the shared vocabulary + the anagram train/val split + indexes.
bank = AnagramBank()

# In-memory game store, keyed by game_id. Single-process only.
games: dict[str, AnagramGame] = {}


class ResetRequest(BaseModel):
    mode: Mode = "train"
    word: Optional[str] = None  # an encoded "w1,w2" pair to pin (else a sampled pair)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its (unanswered) challenge state."""
    target = req.word if req.word is not None else bank.sample_pair(req.mode)
    w1, w2 = decode_pair(target)
    game_id = str(uuid.uuid4())
    games[game_id] = AnagramGame(word1=w1, word2=w2, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the yes/no verdict; returns the scored terminal state."""
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
