"""Per-turn GRPO trainer on verl 0.8.0 — subclass `RayPPOTrainer`, override ONLY `fit()`.

The integration is deliberately minimal (the lightweight path on the trainer side): we let verl's
`run_ppo` do ALL the heavy setup (ray init, worker groups, rollout engine / `async_rollout_manager`,
checkpoint manager, dataloader, `init_workers`) by monkeypatching our subclass in as `RayPPOTrainer`.
We override just `fit()` to:

  per step (one dataloader batch of B target words):
    1. expand each target to G episodes → drive them TURN BY TURN, generating each turn via verl's
       `async_rollout_manager.generate_sequences` (stripped-context messages in → prompt+response ids).
    2. emit ONE per-turn sample per move (uid=(target,round)), reward = per-turn local R_t.
    3. pack → DataProto; reuse verl's `_compute_old_log_prob` / `_compute_ref_log_prob` /
       `compute_advantage` (GRPO, groups by `uid`) / `_update_actor` / `_save_checkpoint`.

verl's optimizer core is untouched. The only verl-version-sensitive bits are the generate_sequences
input/output field names and the DataProto contract — verified against verl 0.8.0.
"""

from __future__ import annotations

import os
import random
import uuid
from typing import Any

import numpy as np

from training.grpo.reward import RewardWeights, reward_breakdown
from training.grpo.rollout import make_groups, roll_batch
from training.grpo.sample_packing import to_dataproto


def build_overrides(*, init_model_dir: str, train_parquet: str, run_name: str, wandb_project: str,
                    train_batch_size: int, group_size: int, clip_ratio_low: float,
                    clip_ratio_high: float, kl_loss_coef: float, lr: float, total_epochs: int,
                    max_steps: int | None, n_gpus: int, save_freq: int,
                    default_local_dir: str, extra: list[str]) -> list[str]:
    """Hydra dotlist overrides applied on top of verl's default `ppo_trainer` config."""
    ov = [
        f"actor_rollout_ref.model.path={init_model_dir}",
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
        f"actor_rollout_ref.actor.use_kl_loss=true",
        f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
        f"actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={group_size}",
        f"actor_rollout_ref.rollout.temperature=1.0",
        f"actor_rollout_ref.rollout.gpu_memory_utilization=0.55",
        f"algorithm.adv_estimator=grpo",
        f"algorithm.use_kl_in_reward=false",
        f"data.train_files={train_parquet}",
        f"data.val_files={train_parquet}",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length=1024",
        f"data.max_response_length=1024",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.nnodes=1",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq=0",
        f"trainer.val_before_train=false",
        f"trainer.project_name={wandb_project}",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={default_local_dir}",
        f"trainer.logger=[console,wandb]",
    ]
    if max_steps is not None:
        ov.append(f"trainer.total_training_steps={max_steps}")
    ov.extend(extra or [])
    return ov


def run(*, config_dir: str, config_name: str, init_model_dir: str, train_parquet: str,
        reward_weights_json: str, run_name: str, wandb_project: str, hub_model_id: str | None,
        train_batch_size: int, group_size: int, max_turns: int, enable_thinking: bool,
        clip_ratio_low: float, clip_ratio_high: float, kl_loss_coef: float, lr: float,
        filter_top_variance_frac: float, total_epochs: int, max_steps: int | None, n_gpus: int,
        extra_overrides: list[str]) -> None:
    """Compose verl's config + monkeypatch our trainer + launch verl's run_ppo."""
    import verl
    import verl.trainer.main_ppo as mp
    from hydra import compose, initialize_config_dir

    # stash run-scoped settings for fit() to read (verl constructs the trainer for us)
    _RUN_STATE.update(
        weights=RewardWeights.from_overrides(**_load_weights(reward_weights_json)),
        max_turns=max_turns, enable_thinking=enable_thinking, group_size=group_size,
        hub_model_id=hub_model_id, filter_top_variance_frac=filter_top_variance_frac,
    )

    verl_cfg_dir = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    overrides = build_overrides(
        init_model_dir=init_model_dir, train_parquet=train_parquet, run_name=run_name,
        wandb_project=wandb_project, train_batch_size=train_batch_size, group_size=group_size,
        clip_ratio_low=clip_ratio_low, clip_ratio_high=clip_ratio_high, kl_loss_coef=kl_loss_coef,
        lr=lr, total_epochs=total_epochs, max_steps=max_steps, n_gpus=n_gpus, save_freq=25,
        default_local_dir=os.path.join(os.path.dirname(train_parquet), "ckpt"), extra=extra_overrides)

    with initialize_config_dir(config_dir=verl_cfg_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)

    mp.RayPPOTrainer = PerTurnGRPOTrainer        # verl's TaskRunner will build OURS
    print("[grpo] launching verl run_ppo with PerTurnGRPOTrainer (per-turn, DAPO knobs)")
    mp.run_ppo(cfg)


def _load_weights(path: str) -> dict:
    import json

    if path and os.path.exists(path):
        return json.load(open(path)).get("weights", {})
    return {}


# Run-scoped state (set in run(), read in fit()); verl owns trainer construction so we can't pass args.
_RUN_STATE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# The trainer: only fit() is overridden.
# ---------------------------------------------------------------------------
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage  # noqa: E402
from verl.trainer.ppo.core_algos import AdvantageEstimator  # noqa: E402


class PerTurnGRPOTrainer(RayPPOTrainer):
    def fit(self):
        from omegaconf import OmegaConf
        from tqdm import tqdm
        from verl.protocol import DataProto
        from verl.utils.tracking import Tracking

        st = _RUN_STATE
        weights: RewardWeights = st["weights"]
        G = st["group_size"]
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self.global_steps += 1
        pbar = tqdm(total=self.total_training_steps, initial=0, desc="per-turn GRPO")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict[str, Any] = {}
                base = DataProto.from_single_dict(batch_dict)
                targets = self._targets_from_batch(base)

                # ---- per-turn rollout via verl's rollout engine ----
                specs = make_groups(targets, G, game="wordle")
                samples, outcomes = roll_batch(
                    specs, tokenizer=self.tokenizer, generate=self._make_generate(),
                    weights=weights, enable_thinking=st["enable_thinking"], max_turns=st["max_turns"])
                if not samples:
                    continue

                batch = to_dataproto(samples, pad_id=self.tokenizer.pad_token_id or 0,
                                     max_prompt_len=self.config.data.max_prompt_length,
                                     max_response_len=self.config.data.max_response_length)
                # GRPO groups by `index`; map our uid=(target,round) onto it.
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"]
                batch.non_tensor_batch["index"] = batch.non_tensor_batch["uid"]
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # ---- reuse verl: old/ref logprob → advantage(grpo, by uid) → update ----
                old = self._compute_old_log_prob(batch)
                batch = batch.union(old[0] if isinstance(old, tuple) else old)
                ref = self._compute_ref_log_prob(batch)
                batch = batch.union(ref)
                batch = compute_advantage(
                    batch, adv_estimator=AdvantageEstimator.GRPO,
                    gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam,
                    num_repeat=G,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=self.config.algorithm)
                actor_output = self._update_actor(batch)

                # ---- metrics ----
                metrics.update(self._reward_metrics(outcomes, samples, weights))
                try:
                    from verl.utils.metric import reduce_metrics
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                except Exception:
                    pass
                logger.log(data=metrics, step=self.global_steps)
                pbar.set_postfix({k: round(v, 3) for k, v in metrics.items()
                                  if isinstance(v, (int, float))} or {})

                # ---- save + push ----
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()
                    self._push_step()
                self.checkpoint_manager.update_weights(self.global_steps)

                pbar.update(1)
                self.global_steps += 1
                if self.global_steps > self.total_training_steps:
                    self._save_checkpoint(); self._push_step()
                    return

    # ---- helpers ----
    def _targets_from_batch(self, base) -> list[str]:
        nb = base.non_tensor_batch
        if "reward_model" in nb:
            return [rm["ground_truth"] for rm in nb["reward_model"]]
        if "extra_info" in nb:
            return [ei["target"] for ei in nb["extra_info"]]
        raise RuntimeError("no target in batch (need reward_model.ground_truth or extra_info.target)")

    def _make_generate(self):
        """messages -> [(prompt_ids, response_ids)] via verl's async_rollout_manager.generate_sequences."""
        from verl.protocol import DataProto

        def generate(messages_list: list[list[dict]]):
            n = len(messages_list)
            proto = DataProto.from_dict(
                non_tensors={"raw_prompt": np.array(messages_list + [None], dtype=object)[:n],
                             "agent_name": np.array(["single_turn_agent"] * n, dtype=object)},
                meta_info={"global_steps": self.global_steps})
            out = self.async_rollout_manager.generate_sequences(proto)
            prompts = out.batch["prompts"]            # [n, prompt_len] left-padded
            responses = out.batch["responses"]        # [n, resp_len] right-padded
            resp_mask = out.batch.get("response_mask")
            res = []
            for i in range(n):
                p = prompts[i].tolist()
                p = [t for t in p if t != (self.tokenizer.pad_token_id or 0)] or p  # strip left pad
                rmask = resp_mask[i].tolist() if resp_mask is not None else None
                r = responses[i].tolist()
                if rmask is not None:
                    r = [t for t, m in zip(r, rmask) if m]
                res.append((p, r))
            return res

        return generate

    def _reward_metrics(self, outcomes, samples, weights) -> dict[str, float]:
        n = len(outcomes)
        solved = sum(1 for o in outcomes if o["status"] == "won")
        agg = {"novel": 0.0, "format": 0.0, "invalid": 0.0, "win": 0.0}
        for o in outcomes:
            for k, v in reward_breakdown(o, weights).items():
                agg[k] += v
        import statistics
        rewards = [s.reward for s in samples]
        return {
            "rollout/win_rate": solved / max(1, n),
            "rollout/avg_turns": sum(len(o["turns"]) for o in outcomes) / max(1, n),
            "rollout/n_samples": float(len(samples)),
            "reward/mean": sum(rewards) / max(1, len(rewards)),
            "reward/std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "reward/novel": agg["novel"] / max(1, n),
            "reward/format": agg["format"] / max(1, n),
            "reward/invalid": agg["invalid"] / max(1, n),
            "reward/win": agg["win"] / max(1, n),
        }

    def _push_step(self):
        hub = _RUN_STATE.get("hub_model_id")
        if not hub:
            return
        try:
            from glob import glob
            from training.grpo.push import push_step
            ckpts = sorted(glob(os.path.join(self.config.trainer.default_local_dir, "global_step_*")))
            if ckpts:
                push_step(ckpts[-1], hub, self.global_steps)
        except Exception as e:  # never let a push failure kill training
            print(f"[grpo] push step-{self.global_steps} failed (non-fatal): {e}")
