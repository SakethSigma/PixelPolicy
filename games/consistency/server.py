"""FastAPI server wrapping the pure Candidate-consistency environment.

    uv run --package game-consistency uvicorn games.consistency.server:app --reload

Pin a challenge with the encoded target ``{"word": "CRANE:xxxxx|SLATE:xxgxx;PLANT"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.consistency.game import ConsistencyBank, ConsistencyGame, GameOverError, GameState, decode_target

app = FastAPI(title="Consistency", description="A pure Wordle candidate-consistency environment.")

bank = ConsistencyBank()
games: dict[str, ConsistencyGame] = {}


class ResetRequest(BaseModel):
    mode: str = "train"
    word: Optional[str] = None  # encoded "rows;candidate" to pin (else sampled)


class StepRequest(BaseModel):
    game_id: str
    answer: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    target = req.word if req.word is not None else bank.sample(req.mode)
    rows, candidate = decode_target(target)
    game_id = str(uuid.uuid4())
    games[game_id] = ConsistencyGame(rows=rows, candidate=candidate, game_id=game_id)
    return games[game_id].state()


@app.post("/step", response_model=GameState)
async def step(req: StepRequest) -> GameState:
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
    game = games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {game_id}")
    return game.state()
