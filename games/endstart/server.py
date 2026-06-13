"""FastAPI server wrapping the pure Ends-with → starts-with environment.

    uv run --package game-endstart uvicorn games.endstart.server:app --reload

Pin a challenge with the encoded target ``{"word": "mango;river,oasis,tundra,cliff,marsh"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.endstart.game import EndstartBank, EndstartGame, GameOverError, GameState, decode_target
from games.wordvocab.split import Mode

app = FastAPI(title="Endstart", description="A pure ends-with → starts-with environment over HTTP.")

bank = EndstartBank()
games: dict[str, EndstartGame] = {}


class ResetRequest(BaseModel):
    mode: Mode = "train"
    word: Optional[str] = None  # encoded "word1;opt1,opt2,..." to pin (else sampled)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its (unanswered) challenge state."""
    if req.word is not None:
        word1, options = decode_target(req.word)
    else:
        word1, options = bank.sample(req.mode)
    game_id = str(uuid.uuid4())
    games[game_id] = EndstartGame(word1=word1, options=options, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the chosen candidate; returns the scored terminal state."""
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
