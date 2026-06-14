# Session handoff (training → eval → learning-dynamics)

Pointer doc for picking up after this session compresses. Reference the linked files for detail;
this is just *what exists, what's running, what's next, and the gotchas we paid for.*

## Where things are

- **Dataset:** `saketh-chervu/word-games-distillation` (13 games, 96,162 rows / ~95.5k valid, 90/10
  split, train/test). Built by `distillation/`. Story in [distillation/blog_notes.md](distillation/blog_notes.md).
- **Models on HF** (`saketh-chervu/word-games-sft-<variant>`), per-epoch revisions `epoch-1..4`:
  - `…-wordle` — **trained (4 epochs), done.** Eval done locally but with NO raw stored and
    `max_tokens=2048` (truncated ~14% of thinking turns) — **re-run eval** (see below).
  - `…-full` — **dead run** (RunPod container-disk loss); has only epoch-1/2 weights. Don't reuse.
  - `…-full-v2` — the clean full re-run (in progress / to launch). New revisions: `epoch-1..4`
    (weights), `resume` (full checkpoint w/ optimizer), `probe` (holds `grad_probe.jsonl`).
- **Git HEAD = `bc0fa0d`** (push it; `git log origin/main..main` should be empty). On a pod always
  `git fetch origin && git reset --hard origin/main` (fetch first — reset alone uses a stale ref).

## Code map (all built this session)

- **Training** (`training/sft/`): `format.py` (row→prompt/completion, the only chat-template seam,
  `REASONING_GAMES` incl. wordle), `data_flat.py` (wordle/full loaders + `max_tokens` filter),
  `data_curriculum.py` (widening/sorted/weighted), `train.py` (TRL SFTTrainer, completion-only loss,
  `chunked_nll`, `--resume`/`--resume-from`, `--game-probe-steps`), `upload.py`
  (`EpochHubPushCallback`: epoch-N weights + `resume` full ckpt; `push_file`), `dynamics.py`
  (`GradUpdateNormCallback` + `PerGameGradProbe`). Docs: [training/README.md](training/README.md),
  [training/CURRICULUM_NOTES.md](training/CURRICULUM_NOTES.md),
  [training/LEARNING_DYNAMICS_NOTES.md](training/LEARNING_DYNAMICS_NOTES.md).
- **Eval** (`inference/`): `server.py` (vLLM launcher, `--enforce-eager` for WSL/repeat-launch),
  `metrics.py` (accuracy/win-rate, Wilson CI, aggregates), `evaluate.py` (plays all games via the
  `distillation/registry.GAMES` registry + `agents/rollout.run_eval`; stores raw incrementally),
  `run_checkpoints.py` (per-checkpoint serve→eval→teardown orchestrator), `recompute.py` (metrics
  from raw, no re-inference). Docs: [inference/README.md](inference/README.md).
- **Viz** (PEP-723 local scripts, no torch): `training/analysis/viz_dynamics.py`
  (`--component layer|attn|mlp|norm`), `inference/analysis/viz_eval.py` (heatmap, per-game, wordle
  headline, base→best delta, `results_table.md`). Dynamics deep-dive context:
  [training/analysis/HANDOFF_dynamics_viz.md](training/analysis/HANDOFF_dynamics_viz.md).
- **Ops cheatsheet:** [_RUNPOD_COMMANDS.md](_RUNPOD_COMMANDS.md) — setup, the 3 jobs, crash-recovery,
  artifact exfil. (Temp.)

## What's running / next actions

1. **`full-v2` training** (RunPod A100, `/workspace/runs/full-v2`, `--game-probe-steps 250`). Verify
   `epoch-1..4` + `resume` + `probe/grad_probe.jsonl` land on HF. Command in `_RUNPOD_COMMANDS.md` Job 2.
2. **Re-run wordle eval with raw + 4096** → fresh dir `eval_results_v2/` (old run had no raw, 2048).
   `inference.run_checkpoints --repo …-wordle --epochs 2,1,3,4 --games all --n 300 --enforce-eager`.
3. **curriculum run** (Job 3), then the three-way + base eval, then plots/`results_table.md`.
4. **GRPO/RL phase** on Wordle from the SFT inits (the compositional payoff — see blog_notes Part II).
5. **Deep-dives:** per-game `grad_probe.jsonl` (which game→which layer; simple-vs-reasoning), and the
   loss-vs-accuracy divergence (eval loss rose at epoch 4 while win-rate held — loss isn't the metric).
6. **TODO (deferred):** make the eval resilient to a vLLM server drop mid-run (it crashed with
   `Connection refused` — server died, orchestrator had no retry/relaunch).

## Gotchas we paid for (don't relearn these)

- **Storage:** pod **container disk** (`/`, where the repo clones) is ephemeral and was LOST on a
  failed migration. **Write checkpoints to the persistent volume `/workspace`.** A network volume
  survives pod death.
- **HF push is the only off-pod channel** (no git on the pod). We push: epoch weights, `resume` full
  checkpoint (crash-recoverable), and `grad_probe.jsonl` (`@probe`, auto each epoch) + wandb.
- **torch/CUDA:** RunPod driver is CUDA 12.8; default PyPI torch (CUDA 13) fails. After `uv sync`,
  `uv pip install torch --reinstall --index-url …/cu128`, then **every `uv run` uses `--no-sync`**
  (else it reverts torch). Don't pin the index in pyproject (it strips the nvidia libs → no libcudnn).
- **OOM:** ~248k vocab → use `chunked_nll` (default). Eval `--max-tokens 4096` (2048 truncated thinking).
- **Sessions:** run inside `tmux` (web terminal is ephemeral; a disconnect doesn't stop the job but
  loses the view). Losing internet does NOT stop the container.
- **Resume:** `--resume-from <download of `resume` branch>` (or `--resume` if `/workspace` survived).
