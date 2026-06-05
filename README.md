# PixelPolicy

From pixels to policies — multimodal agents that learn to play games.

PixelPolicy is a modular research framework for training and evaluating LLMs and VLMs as game-playing agents. The goal is to make it easy to plug in new games, swap out models, and experiment with training strategies — all without tangling those concerns together.

---

## Design

The repo is organized around four clear responsibilities:

**`games/`** — Each game is an isolated FastAPI server. Games expose a standard REST interface (reset, step, render, valid actions) so any agent can play any game without game-specific logic leaking into agent or training code. Each game lives in its own subdirectory with its own `pyproject.toml`, keeping dependencies isolated.

**`agents/`** — Agents are thin wrappers over models (e.g. Claude, GPT-4o). They receive a game observation and return an action. Keeping agents lightweight means the interesting logic lives in training, not in agent scaffolding.

**`training/`** — Training pipelines (e.g. RL, GRPO, supervised fine-tuning) that drive an agent through a game environment and update model weights or prompts.

**`inference/`** — Inference pipelines for evaluating a trained or prompted agent against a game without any weight updates.

**`.env`** — API keys and configuration. Never committed. See `.env.example`.

This separation means you can run a game server independently, swap models in agents without touching games, and run inference without pulling in training dependencies.

---

## Repo Structure

```
PixelPolicy/
├── pyproject.toml          # uv workspace root
├── uv.lock
├── .env                    # your keys (gitignored)
├── .env.example
├── games/
│   ├── snake/
│   │   ├── pyproject.toml  # fastapi, uvicorn
│   │   └── server.py
│   └── <game>/             # add new games here
│       ├── pyproject.toml
│       └── server.py
├── agents/
│   ├── pyproject.toml      # anthropic, openai, pydantic
│   └── ...
├── training/
│   ├── pyproject.toml      # torch, transformers, datasets
│   └── ...
└── inference/
    ├── pyproject.toml      # httpx, pydantic
    └── ...
```

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — install once with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Install

```bash
git clone https://github.com/your-username/PixelPolicy.git
cd PixelPolicy
uv sync
```

`uv sync` reads the workspace `pyproject.toml`, resolves all member dependencies together, and writes a single `uv.lock`. No need to manage separate virtual environments per subproject.

### Configure API Keys

```bash
cp .env.example .env
# edit .env and fill in your keys
```

### Run a Game Server (example)

```bash
uv run --package game-snake uvicorn games.snake.server:app --reload
```

### Run an Agent

```bash
uv run --package agents python -m agents.run --game http://localhost:8000
```

---

## Adding a New Game

1. Create `games/<your-game>/` with a `pyproject.toml` and a `server.py` that implements the standard game interface.
2. Run `uv sync` to pick it up in the workspace.
3. The game is immediately playable by any existing agent.

---

## Contributing

Games, agents, and training recipes are all welcome. The only hard rule: a new game must not require changes to agent or training code, and a new agent must not require changes to any game.
