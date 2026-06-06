# PLAN — watch Qwen3.5-0.8B play Wordle in the terminal (temporary, delete when done)

Goal: host the model locally with vLLM, then run the agent demo so the model plays Wordle in
the terminal. Fix the Wordle agent so its prompt/parse logic matches Qwen3.5 thinking output
(`<think>…</think>` reasoning, final answer in `<guess>…</guess>`).

Environment (verified): `Qwen/Qwen3.5-0.8B` fully downloaded; vLLM 0.22.1; 12 GB RTX 4070 SUPER.

## Progress
- [x] `agents/wordle/agent.py` — system prompt no longer tells the model to emit `<think>`
      (the thinking template does that); parse falls back past `<think>` blocks; prior
      chain-of-thought stripped on multi-turn replay.
- [x] `agents/config.py` — `max_tokens` 1024 → 2048, `temperature` 0.7 → 0.6.
- [x] `inference/server.py` — adds `--max-model-len 8192`, `--gpu-memory-utilization 0.85`,
      `--limit-mm-per-prompt '{"image":0,"video":0}'` unless the user overrides them.
- [x] `.env` created (base url / model / `EMPTY` key).
- [x] `agents/Readme.md` — fixed stale `Turn.round`→`.state`, `GUESS:`→`<guess>` parsing,
      `render_round(turn.round)`→`turn.state.rounds[-1]`, dropped non-existent `--game` flag.
- [ ] Verify end-to-end (server up, demo plays, headless eval, unit tests).

## Run
```bash
# terminal A — server (defaults to Qwen/Qwen3.5-0.8B); fall back to --model Qwen/Qwen3-0.6B
uv run --package inference python -m inference.server
curl -s http://127.0.0.1:8000/v1/models

# terminal B — watch it play
uv run --package agents python -m agents.run --demo --word crane
uv run --package agents python -m agents.run --episodes 5     # headless win-rate
uv run --package agents pytest agents/tests/ -q               # tests
```
