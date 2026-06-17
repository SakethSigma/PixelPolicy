# Handoff: GRPO RL on word-games (verl)

**For:** a coding agent implementing GRPO training. **Library:** [verl](https://github.com/volcengine/verl).
**Base model:** `Qwen/Qwen3.5-0.8B`. **You are writing new code under `training/grpo/`** (mirror the
layout/conventions of the existing `training/sft/`).

## 2. Experiment (what the training must support)
Hold the **RL objective fixed (wordle), vary only the init** — this isolates transfer:
- **Arm A:** init = `saketh-chervu/word-games-sft-wordle` → GRPO on wordle.
- **Arm B:** init = `saketh-chervu/word-games-sft-full-v2` → GRPO on wordle.  ← the key comparison vs A
- **Arm C (dilution check):** init = `full-v2` → GRPO on *all* games jointly.

So the trainer needs: (a) configurable **init checkpoint** (HF repo + `--revision epoch-N`), (b)
configurable **task set** (single game `wordle` or the full list), (c) GRPO via verl.

## 3. The hard part: multi-turn rollout
Our games are **multi-turn** (wordle = up to 6 guesses; the agent sees feedback and guesses again).
verl's default GRPO is single-turn (prompt → response → scalar reward). You must bridge one of two ways:
- **Preferred:** use verl's **multi-turn / agent-loop rollout** (verl supports multi-turn rollout with a
  custom interaction/tool or "agent loop"; check current verl docs for `multi_turn` / SGLang agent
  rollout). Each wordle guess = one turn; the env returns feedback as the next observation.
- **Fallback:** treat the whole episode as one trajectory you generate yourself by driving the existing
  env, then hand verl the (prompt, full multi-turn token sequence, reward) for the GRPO update.

Reuse the existing rollout machinery — **do not reimplement game logic**:
- `agents/rollout.py::run_episode(agent, env, generate)` — drives one full episode, loop is
  `while state.status == "in_progress"`; returns `Trajectory(turns=[Turn(response, action)], final)`.
- `distillation/registry.py::GAMES` — per-game `GameSpec` with `make_env(target)`, `make_agent()`,
  `sample_targets(n, mode, rng)`, `good_status` (default `"won"`; single-turn games use `"correct"`).
- For training rollouts draw targets with `mode="train"` (eval uses `mode="val"` — **keep them disjoint;
  never train on the val targets** used by `inference/evaluate.py`).

## 4. Reward
- **Sparse (correctness):** `1.0` if `trajectory.final.status == spec.good_status` (wordle: `"won"`),
  else `0.0`. This is the source of truth and what eval reports.
- **Sparse reward will starve GRPO** at a ~2–7% solve rate (most rollout groups all-fail → zero
  group-relative advantage). **You almost certainly need dense shaping.** Build it from wordle's own
  feedback: `games/wordle/game.py::compute_feedback(guess, target) -> [LetterFeedback]` (greens/yellows/
  grays). Reasonable shaped reward per episode = e.g. `win_bonus + Σ (greens*g + yellows*y)` across
  guesses, minus a small penalty for invalid guesses (`RoundResult.error` = LENGTH/VOCAB). Keep the
  terminal win term dominant so you optimize solving, not just color-farming. Make the shaping weights
  config flags so they're easy to ablate.
- For Arm C (all games), reward = each game's `good_status` check; consider per-game reward
  normalization so easy games don't drown wordle's signal.

## 5. Eval (don't build a new one)
Measure with the existing harness so numbers are comparable to the SFT runs:
- Serve the RL checkpoint with `inference/server.py`, then `inference/evaluate.py --label <name>
  --games wordle --n 300 --seed 0` → writes `<label>.json` with accuracy + Wilson CI.
- **Match sampling** to `inference/evaluate.py::EVAL_SAMPLING` = `temperature 0.6, top_p 0.95,
  enable_thinking=True`, `max_tokens 4096`. (Training rollout temp can differ, but report eval at these.)
- Pull/plot with `inference/analysis/viz_eval.py`. See `_RUNPOD_METRICS.md` for the eval flow.

## 6. Conventions / infra to mirror
- Follow `training/sft/`: arg-parsed entrypoint (`training/grpo/train.py`), HF push via
  `training/sft/upload.py::push_file`/`upload_folder`, optional wandb. Models are private HF repos
  (`HF_TOKEN` required); push checkpoints to a new repo e.g. `saketh-chervu/word-games-grpo-<arm>`.
- Workspace is uv (`package = false`, PYTHONPATH). verl + its deps likely need their own install step;
  document it. **Do NOT disturb the pinned inference env** (vllm 0.19 / instrumentator <8 in
  `inference/pyproject.toml` + `uv.lock`) — verl may want a different vllm; isolate it (separate
  venv/package) if so.
- GPU: A100 (single, 80GB). 0.8B → GRPO with vLLM rollout fits comfortably.

## 7. Scope for the first PR (don't boil the ocean)
- **MVP = Arm A only**: wordle, single-task, GRPO from the wordle-SFT init, with sparse + shaped
  reward, eval after. Get the loop training and wordle accuracy moving above the SFT ~7% floor.
- Then make init + task-set configurable to run Arms B and C with the same code.
- Non-goals for now: multi-game reward balancing tuning, curriculum, anything beyond getting a correct,
  measurable GRPO loop. Leave knobs as flags.

## 8. Open questions to resolve while implementing
- Does the installed verl version have first-class multi-turn rollout, or must we drive the env and feed
  verl pre-rolled trajectories? (Decides §3 path.)
- Shaping weights (green/yellow/invalid/win) — start simple, expose as flags, ablate later.
- `enable_thinking` during *training* rollouts: thinking tokens help but lengthen rollouts/cost — decide
  and make it a flag.
- KL/clip/group-size GRPO hyperparams — use verl defaults first; the bottleneck is reward signal, not HPs.

## Key files (read these first)
- `agents/rollout.py` — `run_episode`, `Trajectory`/`Turn`, the in_progress loop.
- `distillation/registry.py` — `GameSpec` / `GAMES` (env, agent, targets, good_status).
- `games/wordle/game.py` — `WordleGame`, `compute_feedback`, `LetterFeedback`, `Status`, `RoundResult`.
- `inference/evaluate.py` — eval harness + `EVAL_SAMPLING` (match it).
- `training/sft/train.py`, `training/sft/upload.py` — infra/conventions to mirror.
