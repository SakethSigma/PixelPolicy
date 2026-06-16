# GRPO RL — continue an SFT checkpoint on the word games (per-turn, DAPO-stabilized, on verl)

Group-Relative Policy Optimization on top of our SFT checkpoints, using [verl](https://github.com/volcengine/verl)
as the **unpatched optimizer core** and a thin custom **per-turn rollout layer** on top. We reuse the
existing game env / agent / eval machinery verbatim; the only new game-aware code is the per-turn
reward (`reward.py`) and the rollout driver (`rollout.py`).

> Develop locally; **run on a single RunPod A100 80 GB**. Runbook: [`_RUNPOD_GRPO.md`](../../_RUNPOD_GRPO.md).
> Canonical design + every decision: [`PLAN.md`](PLAN.md).

## The experiment (3 arms, one trainer) — starting with **Arm B**

| `--arm` | init checkpoint | task | question |
|--------|-----------------|------|----------|
| `a` | `word-games-sft-wordle` | wordle | RL from narrow SFT |
| **`b`** | `word-games-sft-full-v2` | wordle | **key + first run:** does broad SFT transfer better? |
| `c` | `word-games-sft-full-v2` | all 13 games | dilution check |

## Why per-turn (the core design)

Our pipeline (SFT, `WordleAgent`, eval) **strips prior `<think>`**, and Qwen3's template strips it too.
You can't strip prior reasoning *and* keep one masked trajectory sequence — so **each move is its own
GRPO sample**: prompt = `WordleAgent.build_messages` (stripped, byte-identical to inference), completion
= that turn's `<think><guess>`. This kills the train/inference skew and the sequence-length blowup of the
one-sample-per-game approach. Samples are grouped by **`uid=(target, round)`**: round-1 samples across a
target's `G` episodes share the *identical* prior (a perfect GRPO baseline); later rounds use the
same-round baseline. The gradient is unbiased regardless (the group mean is an action-independent
baseline). Loss is on the completion only — the prompt, **including feedback**, is masked.

## Reward — purely per-turn local (no terminal by default)

`R_t = format_t + invalid_t + novelty_t·decay**(round-1)`. Novelty = **new** green/yellow only
(re-confirmations pay nothing); round-decayed so early discovery (→ fast win) is worth more.
**No terminal/broadcast by default** (`win_bonus=0.0`) — novel-green already encodes winning, and a
purely-local reward keeps reward-variance high (anti-collapse). `win_bonus` is a flag (safety valve).

## Stability — DAPO/StarPO-S (verl `recipe/dapo`)

Clip-Higher (`clip_ratio_low=0.2`, `clip_ratio_high=0.28`), token-mean loss, dynamic sampling
(`filter_groups`), TIS (vLLM↔FSDP), KL as a flag (`--kl-loss-coef`, default `0.001`; `0` = DAPO removal).
StarPO-S variance filtering wired but **off** (`--filter-top-variance-frac 1.0`) — our dense reward
already keeps reward-std high. Watch `entropy`, `reward_std`, `clip_ratio/high` on wandb.

## Layout

```
training/grpo/
  PLAN.md            # canonical design (crash-recoverable)
  reward.py          # per_turn_rewards (local, round-decayed) + RewardWeights + compute_score  [tested]
  rollout.py         # per-turn driver: roll_batch → TurnSamples (uid=(target,round))           [tested]
  sample_packing.py  # TurnSamples → padded tensors / verl DataProto (loss masked to completion) [tested]
  data.py            # train-target parquet (disjoint from val)                                  [tested]
  train.py           # CLI + arm resolution + --smoke-local (full pipeline, no verl/GPU)
  trainer.py         # verl-backed fit loop; two VERL-VERSION seams to complete on the pod
  push.py            # push step-N checkpoints (reuses training/sft/upload)
  config/            # grpo_wordle.yaml (A) · grpo_wordle_full.yaml (B) · grpo_all.yaml (C)
  tests/             # reward / data / rollout / packing — 29 tests, run in the workspace venv
```

## Validate locally (no verl, no GPU)

```bash
# unit tests
PYTHONPATH=. uv run --no-sync --package agents python -m pytest training/grpo/tests/ -q
# full rollout→reward→packing pipeline on real targets with a fake tokenizer:
PYTHONPATH=. uv run --no-sync --package agents python -m training.grpo.train --arm b --smoke-local \
  --train-batch-size 3 --group-size 4
```

## Run on the pod (Arm B)

See [`_RUNPOD_GRPO.md`](../../_RUNPOD_GRPO.md). In short: isolated verl venv (never touches the inference
vllm 0.19 pin), complete the two `# VERL-VERSION` seams in `trainer.py` against the installed verl
(`_build_workers` + `verl_batch_generate` — copy from verl's `recipe/dapo/main_dapo.py`), then:

```bash
# PROBE first — verifies the verl seam + reads steps/hour, before committing the run:
PYTHONPATH=. /workspace/verl-venv/bin/python -m training.grpo.train --arm b \
  --work-dir /workspace/grpo --hub-model-id saketh-chervu/word-games-grpo-armb --max-steps 5
# full run: drop --max-steps.
```

## Eval (no new code — reuse the frozen harness)

```bash
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-grpo-armb --revisions step-25,step-50 \
  --games wordle --n 300 --seed 0 --out eval_results_grpo/
```
Frozen sampling (`temperature 0.6, top_p 0.95, enable_thinking`, max_tokens 4096) on the **val** pool,
which training never touches. Plot with `inference/analysis/viz_eval.py`. Headline: Arm A vs Arm B
wordle win-rate across `step-N`.

## Status

Reward, rollout, packing, data, CLI + local smoke are **done and tested** (29 passing). The verl
optimizer glue (`trainer.py`) has two clearly-marked `# VERL-VERSION` seams completed on the pod at the
probe — that was the one flagged integration risk (verl's worker API moves across releases). Fallbacks:
vendor RAGEN's rollout layer, or a compact self-contained GRPO update (see `PLAN.md`).
