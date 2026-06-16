"""Per-turn GRPO trainer on top of verl's UNPATCHED optimizer core (RAGEN-style integration).

We reuse verl for what it's good at — FSDP actor/ref workers, the vLLM/SGLang rollout engine, the
GRPO advantage (`compute_grpo_outcome_advantage`, grouped by `uid`), the DAPO actor update
(clip-higher / token-mean / dynamic sampling / KL), and checkpoint/push — and replace ONLY the
rollout step with our per-turn driver (`rollout.roll_batch`) + packer (`sample_packing.to_dataproto`).

Structure of one training step (`_fit` loop):
  1. draw `B` train targets; expand each to `G` episodes (`rollout.make_groups`)
  2. `roll_batch(... batch_generate=<verl rollout engine> ...)` → per-turn samples (uid=(target,round))
  3. `to_dataproto(samples)` → verl `DataProto`
  4. verl: ref log-probs → `compute_grpo_outcome_advantage(uid grouping)` → `update_actor` (DAPO knobs)
  5. every `save_freq` steps: save + push `step-N` (reuse `training/sft/upload.push_checkpoint`)

NOTE (VERL-VERSION): the worker-group construction and the exact method names
(`RayPPOTrainer`, `ActorRolloutRefWorker`, `compute_grpo_outcome_advantage`, `DataProto.from_dict`)
move across verl releases. The two seams to confirm at the `--max-steps 5` probe are marked
`# VERL-VERSION`: (a) `_build_workers` (how the rollout/actor/ref worker groups are spun up), and
(b) `verl_batch_generate` (how to ask the rollout engine for completions given prompt token ids).
Everything between them — our rollout, reward, packing, uid grouping — is verl-agnostic and unit-tested.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from training.grpo.rollout import make_groups, roll_batch
from training.grpo.reward import RewardWeights


# ---------------------------------------------------------------------------
# Config assembly (OmegaConf) — pure, no verl needed to read/inspect.
# ---------------------------------------------------------------------------
def build_config(config_dir: str, config_name: str, *, init_model_dir: str, train_parquet: str,
                 run_name: str, wandb_project: str, train_batch_size: int, group_size: int,
                 clip_ratio_low: float, clip_ratio_high: float, kl_loss_coef: float, lr: float,
                 total_epochs: int, max_steps: int | None, n_gpus: int,
                 extra_overrides: list[str]) -> Any:
    """Load `config/<config_name>.yaml` and apply the run's overrides → an OmegaConf config."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(Path(config_dir) / f"{config_name}.yaml"))
    ov = {
        "actor_rollout_ref.model.path": init_model_dir,
        "actor_rollout_ref.ref.model.path": init_model_dir,
        "actor_rollout_ref.actor.optim.lr": lr,
        "actor_rollout_ref.actor.clip_ratio_low": clip_ratio_low,
        "actor_rollout_ref.actor.clip_ratio_high": clip_ratio_high,
        "actor_rollout_ref.actor.kl_loss_coef": kl_loss_coef,
        "actor_rollout_ref.rollout.n": group_size,
        "data.train_files": train_parquet,
        "data.train_batch_size": train_batch_size,
        "trainer.total_epochs": total_epochs,
        "trainer.n_gpus_per_node": n_gpus,
        "trainer.experiment_name": run_name,
        "trainer.project_name": wandb_project,
    }
    if max_steps is not None:
        ov["trainer.total_training_steps"] = max_steps
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in ov.items()]))
    if extra_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(extra_overrides)))
    return cfg


# ---------------------------------------------------------------------------
# verl seam (a): build the worker groups. # VERL-VERSION
# ---------------------------------------------------------------------------
def _build_workers(cfg: Any):
    """Spin up verl's actor/ref/rollout worker groups for `cfg`.

    Mirror verl's DAPO recipe `main_dapo.py` / `RayPPOTrainer.init_workers()`. Returns whatever the
    loop needs: (actor_rollout_wg, ref_wg, tokenizer, checkpoint_manager). Kept isolated so a verl
    bump is a localized edit.
    """
    raise NotImplementedError(
        "VERL-VERSION seam: wire verl's RayPPOTrainer/worker groups here, copying the setup from the "
        "installed verl's recipe/dapo/main_dapo.py (ResourcePoolManager, Role, ActorRolloutRefWorker). "
        "Run with --smoke-local to validate the rollout/reward/packing pipeline without this.")


# ---------------------------------------------------------------------------
# verl seam (b): rollout engine → token ids. # VERL-VERSION
# ---------------------------------------------------------------------------
def verl_batch_generate(actor_rollout_wg: Any, tokenizer: Any, sampling: dict):
    """Return a `batch_generate(prompts: list[list[int]]) -> list[list[int]]` bound to verl's engine.

    verl's rollout worker generates from a `DataProto` of prompts and returns responses; we adapt that
    to the simple token-ids-in/token-ids-out contract `roll_batch` expects. Confirm the worker method
    (`generate_sequences`) and the response field against the installed verl.
    """
    from training.grpo.sample_packing import pad_left

    def batch_generate(prompts: list[list[int]]) -> list[list[int]]:
        # VERL-VERSION: build a prompt-only DataProto, call actor_rollout_wg.generate_sequences,
        # slice off the prompt to get each response's new token ids.
        raise NotImplementedError("VERL-VERSION seam: bind to verl rollout engine generate_sequences.")

    return batch_generate


# ---------------------------------------------------------------------------
# The training loop — our logic; verl calls behind the two seams above.
# ---------------------------------------------------------------------------
def run(*, config_dir: str, config_name: str, init_model_dir: str, train_parquet: str,
        reward_weights_json: str, run_name: str, wandb_project: str, hub_model_id: str | None,
        train_batch_size: int, group_size: int, max_turns: int, enable_thinking: bool,
        clip_ratio_low: float, clip_ratio_high: float, kl_loss_coef: float, lr: float,
        filter_top_variance_frac: float, total_epochs: int, max_steps: int | None, n_gpus: int,
        extra_overrides: list[str]) -> None:
    import json

    from distillation.registry import GAMES

    cfg = build_config(config_dir, config_name, init_model_dir=init_model_dir,
                       train_parquet=train_parquet, run_name=run_name, wandb_project=wandb_project,
                       train_batch_size=train_batch_size, group_size=group_size,
                       clip_ratio_low=clip_ratio_low, clip_ratio_high=clip_ratio_high,
                       kl_loss_coef=kl_loss_coef, lr=lr, total_epochs=total_epochs,
                       max_steps=max_steps, n_gpus=n_gpus, extra_overrides=extra_overrides)

    with open(reward_weights_json) as f:
        weights = RewardWeights.from_overrides(**json.load(f).get("weights", {}))

    actor_rollout_wg, ref_wg, tokenizer, ckpt = _build_workers(cfg)   # VERL-VERSION seam (a)
    sampling = {"temperature": float(cfg.actor_rollout_ref.rollout.temperature),
                "top_p": float(cfg.actor_rollout_ref.rollout.top_p),
                "max_tokens": int(cfg.data.max_response_length)}
    batch_generate = verl_batch_generate(actor_rollout_wg, tokenizer, sampling)   # seam (b)

    rng = random.Random(int(cfg.get("data", {}).get("seed", 0)) if hasattr(cfg, "get") else 0)
    spec_w = GAMES["wordle"]()

    step = 0
    while max_steps is None or step < max_steps:
        # 1) targets → episode specs (G per target)
        targets = spec_w.sample_targets(train_batch_size, "train", rng)
        specs = make_groups(targets, group_size, game="wordle")
        # 2) per-turn rollout via verl's engine
        samples, outcomes = roll_batch(specs, tokenizer=tokenizer, batch_generate=batch_generate,
                                       weights=weights, enable_thinking=enable_thinking,
                                       max_turns=max_turns)
        # 3) pack → DataProto ; 4) verl advantage(uid)+update ; 5) save/push
        # VERL-VERSION: to_dataproto(samples) → ref logprobs → compute_grpo_outcome_advantage(index=uid)
        #   → update_actor(DAPO knobs) → if step % save_freq == 0: save + push_checkpoint(step-N).
        raise NotImplementedError(
            "VERL-VERSION: complete the optimize step (pack→advantage→update→save) using the installed "
            "verl's RayPPOTrainer methods — the rollout/reward/packing above are done and tested.")
