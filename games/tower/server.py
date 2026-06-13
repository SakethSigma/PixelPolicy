"""FastAPI server wrapping the pure Tower-deduction environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-tower uvicorn games.tower.server:app --reload

The single-turn API is ``/reset`` then ``/step``. Pin a challenge with the encoded target string
``{"word": "Alice,Bob,Carol;2L,1R,3L;01,10,00"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.tower.game import GameOverError, GameState, TowerBank, TowerGame, decode_target

app = FastAPI(title="Tower", description="A pure tower-deduction environment over HTTP.")

bank = TowerBank()
games: dict[str, TowerGame] = {}


class ResetRequest(BaseModel):
    mode: str = "train"
    word: Optional[str] = None  # an encoded "names;shown;feedback" target to pin (else sampled)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new episode and return its proposed-placement + feedback state."""
    target = req.word if req.word is not None else bank.sample(req.mode)
    names, sf, sr, fok, rok = decode_target(target)
    game_id = str(uuid.uuid4())
    games[game_id] = TowerGame(names, sf, sr, fok, rok, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
    """Submit the list of consistent placements; returns the scored terminal state."""
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
