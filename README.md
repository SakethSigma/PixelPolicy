# PixelPolicy

From pixels to policies — multimodal agents that learn to play games.

PixelPolicy is a modular research framework for training and evaluating LLMs and VLMs as game-playing agents. The goal is to make it easy to plug in new games, swap out models, and experiment with training strategies — all without tangling those concerns together.

---

## Design

The repo is organized around five clear responsibilities:

**`games/`** — Each game is an isolated FastAPI server. Games expose a standard REST interface (reset, step, render, valid actions) so any agent can play any game without game-specific logic leaking into agent or training code. Each game lives in its own subdirectory with its own `pyproject.toml`, keeping dependencies isolated.

**`agents/`** — Agents are thin wrappers over models reached through an OpenAI-compatible API (a local model via vLLM, or a hosted OpenAI/Claude endpoint). They receive a game observation and return an action. Keeping agents lightweight means the interesting logic lives in training, not in agent scaffolding.

**`training/`** — Training pipelines (e.g. RL, GRPO, supervised fine-tuning) that drive an agent through a game environment and update model weights or prompts.

**`inference/`** — Inference pipelines for evaluating a trained or prompted agent against a game without any weight updates.

**`distillation/`** — Generate teacher trajectories with a strong model (Claude), turn them into supervised fine-tuning data, and push to the HuggingFace Hub. Reuses the *same* game loop as inference — a teacher is just another backend — so no game or agent changes. See **[distillation/README.md](distillation/README.md)**.

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
│   ├── wordle/             # the reference game
│   │   ├── pyproject.toml  # fastapi, uvicorn, (rich for the [tui])
│   │   ├── game.py         # pure core    server.py  client.py  render.py  play.py
│   │   └── ...
│   └── <game>/             # add new games here (same layout)
├── agents/
│   ├── pyproject.toml      # openai, pydantic, python-dotenv, game-wordle ([tui]=rich)
│   ├── base.py backend.py rollout.py run.py config.py
│   └── wordle/agent.py     # the only Wordle-aware agent code
├── training/
│   ├── pyproject.toml      # torch, transformers, datasets
│   └── ...
├── inference/
│   ├── pyproject.toml      # vllm
│   └── server.py           # thin launcher over `vllm serve`
└── distillation/
    ├── pyproject.toml      # anthropic, datasets, huggingface-hub
    ├── batch_play.py       # lockstep Batch-API teacher rollouts (game-agnostic)
    ├── registry.py         # GameSpec per game — the one place a new game is added
    ├── push.py             # combine SFT samples → datasets.Dataset → Hub
    └── cost_probe.py       # measure teacher cost at a given reasoning effort
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

### Watch an agent play Wordle

Two steps: host a model, then run the agent. (The agent steps the game in-process — no game
server needed for this.)

```bash
# 1. host a model locally (vLLM, OpenAI-compatible). First start compiles graphs (~minutes);
#    wait for "Application startup complete". See inference/README.md.
uv run --package inference python -m inference.server

# 2. in another shell, watch one game on a colored board
uv run --package agents python -m agents.run --demo --word crane
#    or a headless win-rate over many games:
uv run --package agents python -m agents.run --episodes 20
```

A Wordle-fine-tuned model plays far better than the base model — host it with
`--model saketh-chervu/qwen3-06b-wordle-sft-phase1-best`. See **[agents/Readme.md](agents/Readme.md)**,
**[agents/wordle/README.md](agents/wordle/README.md)** (exact prompts/parsing per round), and
**[inference/README.md](inference/README.md)**.

### Play Wordle yourself / run the game's HTTP server

```bash
uv run --package game-wordle python -m games.wordle.play        # play in the terminal
uv run --package game-wordle uvicorn games.wordle.server:app    # the game's REST API
```

### Generate teacher data (distillation)

Have a strong model (Claude) play many games, export per-move SFT samples, and push them to
the HuggingFace Hub — to later fine-tune a small open model on the teacher's play. Uses the
same game loop as inference (the teacher is just a Claude backend). Needs `ANTHROPIC_API_KEY`;
pushing also needs `HF_TOKEN` + `HF_HUB_REPO_ID` in `.env`.

```bash
# play N games via the Anthropic Batch API (~50% cheaper); writes raw + per-move SFT JSONL.
# --effort low|medium|high trades cost for reasoning depth (see distillation/blog_notes.md).
uv run --package distillation python -m distillation.batch_play --game wordle --episodes 100 --effort low

# combine all SFT samples into one dataset and push to the Hub (90/10 split)
uv run --package distillation python -m distillation.push --test-size 0.1
```

Example dataset: **[saketh-chervu/word-games-distillation](https://huggingface.co/datasets/saketh-chervu/word-games-distillation)**.
See **[distillation/README.md](distillation/README.md)** for the pipeline and **[distillation/blog_notes.md](distillation/blog_notes.md)** for the cost/effort story.

---

## Adding a New Game

1. Create `games/<your-game>/` with a `pyproject.toml` and a `server.py` that implements the standard game interface.
2. Run `uv sync` to pick it up in the workspace.
3. The game is immediately playable by any existing agent.

---

## Contributing

Games, agents, and training recipes are all welcome. The only hard rule: a new game must not require changes to agent or training code, and a new agent must not require changes to any game.
