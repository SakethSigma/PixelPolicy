# PixelPolicy

From pixels to policies — multimodal agents that learn to play games.

PixelPolicy is a modular research framework for training and evaluating LLMs and VLMs as game-playing agents. The goal is to make it easy to plug in new games, swap out models, and experiment with training strategies — all without tangling those concerns together.

---

## Design

The repo is organized around five clear responsibilities:

**`games/`** — Each game is an isolated FastAPI server. Games expose a standard REST interface (reset, step, render, valid actions) so any agent can play any game without game-specific logic leaking into agent or training code. Each game lives in its own subdirectory with its own `pyproject.toml`, keeping dependencies isolated. Today the repo ships **Wordle** (the multi-turn reference game) plus a family of *word/deduction* games that broaden the word model beyond Wordle: nine **single-turn** games — **`charcount`**, **`validity`**, **`anagram`**, **`endstart`**, **`rhyme`**, **`crossword`**, **`charset`**, **`mistakeid`**, and **`consistency`** — and two more **multi-turn** deduction games, **`codebreaker`** (Mastermind) and **`bullscows`**, that teach the core Wordle feedback loop on a non-vocabulary space (see [`games/DATA_SOURCING.md`](games/DATA_SOURCING.md); every game in the family is now built). A shared support package, **`games/wordvocab/`**, supplies the common multi-length vocabulary (and a word→meaning asset) the word games draw from.

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
│   ├── wordle/             # the reference game (multi-turn)
│   │   ├── pyproject.toml  # fastapi, uvicorn, (rich for the [tui])
│   │   ├── game.py         # pure core    server.py  client.py  render.py  play.py
│   │   └── ...
│   ├── charcount/          # game #1 (single-turn) — same layout as wordle
│   ├── validity/           # game #2 (single-turn) — valid/invalid + meaning
│   ├── anagram/            # game #3 (single-turn) — yes/no anagram, Claude-distilled
│   ├── endstart/           # game #4 (single-turn) — MCQ: candidate starting with word's last letter
│   ├── rhyme/              # game #5 (single-turn) — rhyme MCQ + free
│   ├── crossword/          # game #6 (single-turn) — clue→word, Claude-distilled (reasoning)
│   ├── charset/            # game #7 (single-turn) — used/unused letters across words
│   ├── mistakeid/          # game #8 (single-turn) — Wordle repeated-mistake ID, Claude-distilled (reasoning)
│   ├── tower/              # game #9 (single-turn) — deduce placements from ✓/x feedback, programmatic
│   ├── codebreaker/        # game #10 (MULTI-turn) — Mastermind, per-position ✓/-/x feedback, programmatic
│   ├── bullscows/          # game #11 (MULTI-turn) — bulls/cows count feedback, programmatic
│   ├── consistency/        # game #12 (single-turn) — is a candidate still possible? reuses Wordle scorer
│   ├── wordvocab/          # shared vocab.txt + meanings.jsonl + game-salted split (support pkg)
│   └── <game>/             # add new games here (same layout)
├── agents/
│   ├── pyproject.toml      # openai, pydantic, python-dotenv, game-* packages ([tui]=rich)
│   ├── base.py backend.py rollout.py run.py config.py
│   ├── wordle/agent.py     # the only Wordle-aware agent code
│   ├── charcount/agent.py  # the only charcount-aware agent code (single-turn)
│   ├── validity/agent.py   # validity agent (single-turn)
│   ├── anagram/agent.py    # anagram agent (single-turn, reasoning yes/no)
│   ├── endstart/agent.py   # endstart agent (single-turn, MCQ)
│   ├── rhyme/agent.py      # rhyme agent (single-turn)
│   ├── crossword/agent.py  # crossword agent (single-turn, reasoning)
│   ├── charset/agent.py    # charset agent (single-turn)
│   ├── mistakeid/agent.py  # mistakeid agent (single-turn, reasoning)
│   ├── tower/agent.py      # tower agent (single-turn, deduction; also defines TowerEnv)
│   ├── codebreaker/agent.py# codebreaker agent (multi-turn; also defines CodebreakerEnv)
│   ├── bullscows/agent.py  # bulls & cows agent (multi-turn; also defines BullsCowsEnv)
│   └── consistency/agent.py# consistency agent (single-turn, yes/no)
├── training/
│   ├── pyproject.toml      # torch, transformers, datasets
│   └── ...
├── inference/
│   ├── pyproject.toml      # vllm
│   └── server.py           # thin launcher over `vllm serve`
└── distillation/
    ├── pyproject.toml      # anthropic, datasets, huggingface-hub
    ├── schema.py           # the unified SFT row schema (every game shares one shape)
    ├── batch_play.py       # lockstep Batch-API teacher rollouts (reasoning games)
    ├── programmatic.py     # no-Claude "synthetic teacher" → SFT (e.g. charcount)
    ├── reexport.py         # re-shape a raw Claude dump → current schema (no re-run)
    ├── registry.py         # GameSpec + GAME_NUMBERS — the one place a new game is added
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

### Play a game yourself / run its HTTP server

```bash
uv run --package game-wordle python -m games.wordle.play        # play Wordle in the terminal
uv run --package game-wordle uvicorn games.wordle.server:app    # Wordle's REST API

uv run --package game-charcount python -m games.charcount.play  # play Character counts (single-turn)
uv run --package game-charcount uvicorn games.charcount.server:app

uv run --package game-validity   python -m games.validity.play   # Validity + meaning (--kind invalid)
uv run --package game-anagram    python -m games.anagram.play    # Anagrams (--pair listen,silent)
uv run --package game-endstart   python -m games.endstart.play   # Ends-with → starts-with (MCQ)
uv run --package game-rhyme      python -m games.rhyme.play      # Rhymes (--variant mcq)
uv run --package game-crossword  python -m games.crossword.play  # Crossword fill (--word crane)
uv run --package game-charset    python -m games.charset.play    # Character set (--words cat,planet)
uv run --package game-mistakeid  python -m games.mistakeid.play  # Wordle mistake identification
uv run --package game-tower      python -m games.tower.play      # Tower deduction (✓/x feedback)
uv run --package game-consistency python -m games.consistency.play  # Candidate consistency (yes/no)

# multi-turn deduction games (a guess verb + per-round feedback, like Wordle)
uv run --package game-codebreaker python -m games.codebreaker.play  # Codebreaker / Mastermind
uv run --package game-bullscows   python -m games.bullscows.play    # Bulls & Cows
```

Every game follows the same `play.py` / `server.py` / `client.py` layout, so the commands
only differ by package name. See **[games/wordle/README.md](games/wordle/README.md)**,
**[games/charcount/README.md](games/charcount/README.md)**,
**[games/validity/README.md](games/validity/README.md)**,
**[games/anagram/README.md](games/anagram/README.md)**,
**[games/endstart/README.md](games/endstart/README.md)**,
**[games/rhyme/README.md](games/rhyme/README.md)**,
**[games/crossword/README.md](games/crossword/README.md)**,
**[games/charset/README.md](games/charset/README.md)**,
**[games/mistakeid/README.md](games/mistakeid/README.md)**,
**[games/tower/README.md](games/tower/README.md)**,
**[games/codebreaker/README.md](games/codebreaker/README.md)**,
**[games/bullscows/README.md](games/bullscows/README.md)**, and
**[games/consistency/README.md](games/consistency/README.md)**.

### Generate teacher data (distillation)

Build SFT data for every game, combine it into one dataset under a **unified schema**, and
push to the HuggingFace Hub — to later fine-tune a small open model. There are two producers:
**Claude-distilled** (reasoning games like Wordle, via the Anthropic Batch API — needs
`ANTHROPIC_API_KEY`) and **programmatic** (no-reasoning games like `charcount`, where a
"synthetic teacher" writes the gold answer — zero API cost). Pushing needs `HF_TOKEN` +
`HF_HUB_REPO_ID` in `.env`.

```bash
# Claude-distilled games via the Anthropic Batch API (~50% cheaper); writes raw + per-move SFT JSONL.
# --effort low|medium|high trades cost for reasoning depth (see distillation/blog_notes.md).
uv run --package distillation python -m distillation.batch_play --game wordle    --episodes 100 --effort low
uv run --package distillation python -m distillation.batch_play --game anagram   --episodes 1000 \
  --model claude-sonnet-4-6 --effort high     # reasoning; require_think keeps only <think> traces
uv run --package distillation python -m distillation.batch_play --game crossword --episodes 1500 \
  --model claude-sonnet-4-6 --effort high     # reasoning; require_think keeps only <think> traces
uv run --package distillation python -m distillation.batch_play --game mistakeid --episodes 330 \
  --model claude-sonnet-4-6 --effort max      # reasoning; boards from the committed challenges.jsonl

# programmatic games: generate SFT rows with no Claude (self-checked). Pick the game with --game.
# (--game choices: charcount|validity|rhyme|charset|tower|endstart|codebreaker|bullscows|consistency)
uv run --package distillation python -m distillation.programmatic --game charcount  # default ~14k rows
uv run --package distillation python -m distillation.programmatic --game validity   # ~13.3k rows
uv run --package distillation python -m distillation.programmatic --game rhyme       # 10k rows (MCQ+free)
uv run --package distillation python -m distillation.programmatic --game charset     # 12k rows (2-4 words each)
uv run --package distillation python -m distillation.programmatic --game tower       # 5k rows (deduction puzzles)
uv run --package distillation python -m distillation.programmatic --game endstart    # 6k rows (MCQ)
uv run --package distillation python -m distillation.programmatic --game consistency # 10k rows (5k yes + 5k no)
# multi-turn programmatic games: an unbiased solver is replayed, one SFT row per turn; --max-rows
# caps the output at an episode boundary (whole episodes kept, so the round distribution is unbiased).
uv run --package distillation python -m distillation.programmatic --game codebreaker --episodes 5000 --max-rows 10000  # 10k rows
uv run --package distillation python -m distillation.programmatic --game bullscows   --max-rows 10000                  # 10k rows

# combine every game's SFT into one dataset and push to the Hub (90/10 split)
uv run --package distillation python -m distillation.push --test-size 0.1
```

Example dataset: **[saketh-chervu/word-games-distillation](https://huggingface.co/datasets/saketh-chervu/word-games-distillation)**
— **96,162 rows** (86,545 train / 9,617 test) across **13 games**, of which **95,520 are valid**:
3,078 Wordle (2,602 valid) + 14,000 charcount + 13,254 validity + 1,000 anagram (932 valid) +
10,000 rhyme + 1,500 crossword (1,415 valid) + 12,000 charset + 330 mistakeid (317 valid) +
5,000 tower + 6,000 endstart + 10,000 codebreaker + 10,000 bullscows + 10,000 consistency. The
**multi-turn share** (Wordle + codebreaker + bullscows) is **~24%** (up from ~5% with Wordle alone).
Wordle's `valid` flag is **format compliance** — whether the move carries a `<think>` block,
regardless of win/loss — while the other games gate on answer correctness.
See **[distillation/README.md](distillation/README.md)** for the pipeline and **[distillation/blog_notes.md](distillation/blog_notes.md)** for the cost/effort story.

---

## Adding a New Game

1. Create `games/<your-game>/` with a `pyproject.toml` and a `server.py` that implements the standard game interface.
2. Run `uv sync` to pick it up in the workspace.
3. The game is immediately playable by any existing agent.

To also produce training data for the new game, add one `GameSpec` entry in
`distillation/registry.py` (see **[distillation/README.md](distillation/README.md)**). For the
single-turn *word-skill* family specifically, **[`games/DATA_SOURCING.md`](games/DATA_SOURCING.md)**
and **[`games/CODE_IMPLEMENTATION.md`](games/CODE_IMPLEMENTATION.md)** are the design docs, and
`games/charcount/` + `games/wordvocab/` are the reference implementations to mirror.

---

## Contributing

Games, agents, and training recipes are all welcome. The only hard rule: a new game must not require changes to agent or training code, and a new agent must not require changes to any game.
