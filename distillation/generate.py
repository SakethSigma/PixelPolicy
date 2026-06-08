"""Stage 1 — drive Claude through real episodes; record raw teacher Trajectories.

The whole point: a teacher is just an LLMBackend, so we reuse the EXACT inference game
loop (run_eval) with AnthropicBackend injected as `generate`. Nothing in agents/ or
games/ changes. Output is durable, format-neutral raw Trajectory JSONL — SFT shaping
happens later in dataset.py, so we never have to re-run (expensive) Claude rollouts to
change the training format.

Reuses:
  - AnthropicBackend                 from agents/backend.py  (teacher; emits <think>…</think><guess>…)
  - run_eval                         from agents/rollout.py  ([(agent, env)], generate, concurrency)
  - Trajectory                       from agents/base.py     (.model_dump() -> JSON-able dict)
  - GAMES registry                   from distillation/registry.py
"""

from __future__ import annotations

# TODO imports:
#   import json
#   from agents.backend import AnthropicBackend
#   from agents.rollout import run_eval
#   from distillation.config import DistillConfig
#   from distillation.registry import GAMES


# TODO: def generate_game(cfg, game, n, *, mode="train", word=None) -> Path:
#   1. spec = GAMES[game]; bank = spec.make_bank()  (load shared resources ONCE, share across episodes)
#   2. teacher = AnthropicBackend(cfg.teacher_model, max_tokens=cfg.max_tokens, effort=cfg.effort)
#   3. pairs = [(spec.make_agent(), spec.reset_env(bank, mode=mode, word=word)) for _ in range(n)]
#        - each env is its own reset episode; if `word` is None the registry samples a fresh target per game.
#   4. trajs = run_eval(pairs, teacher.generate, concurrency=cfg.concurrency)
#        - run_eval threads the episodes; the inference engine (here Claude) handles overlap.
#   5. write one Trajectory per line to cfg.raw_dir / f"{game}.jsonl":
#        - mkdir parents=True, exist_ok=True
#        - for t in trajs: f.write(json.dumps(t.model_dump(mode="json")) + "\n")
#        - mode="json" so GameState/UUIDs serialize cleanly (final/turn.state are typed `Any`).
#   6. return the output path; print a one-line summary (n episodes, win-rate via agents.rollout.win_rate).
#
# Notes:
#   - Batch API: the loop is multi-turn (guess N depends on feedback N-1), so an episode can't be a single
#     Batch request. Use run_eval concurrency (live calls). Batch only fits a future single-step dataset mode.
#   - Append vs overwrite: decide whether re-running accumulates more data (append "a") or replaces ("w").
