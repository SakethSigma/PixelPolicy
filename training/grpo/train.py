"""GRPO (per-turn, DAPO-stabilized) trainer entrypoint for the word games.

Pipeline per step (custom rollout on top of verl's UNPATCHED optimizer core):
  draw B target words → expand each to G episodes → `rollout.roll_batch` (turn-by-turn, stripped
  context, batched generation via verl's rollout engine) → per-turn samples with per-turn local
  reward and `uid=(target,round)` → `sample_packing.to_dataproto` → verl's GRPO advantage (group by
  `uid`) + DAPO actor update (clip-higher, token-mean loss, dynamic sampling, KL flag) → save/push.

This file owns the deterministic prep + CLI and the rollout/packing integration. The verl worker
orchestration (actor/ref/rollout RayWorkerGroup, `compute_advantage`, `update_actor`) is verl's; the
one seam we provide is `verl_batch_generate` (verl rollout engine → token ids). Everything in
`--smoke-local` runs with NO verl/GPU, validating our rollout→reward→packing logic before the pod.

VERL-VERSION markers flag the few places to confirm against the installed verl at the `--max-steps 5`
probe (see `_RUNPOD_GRPO.md`).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from distillation.registry import GAMES
from training.grpo import data as data_mod
from training.grpo import reward as reward_mod
from training.grpo.rollout import make_groups, roll_batch

ARMS = {
    "a": {"init_repo": "saketh-chervu/word-games-sft-wordle",  "task": "wordle"},
    "b": {"init_repo": "saketh-chervu/word-games-sft-full-v2", "task": "wordle"},
    "c": {"init_repo": "saketh-chervu/word-games-sft-full-v2", "task": "all"},
}
DEFAULT_INIT_REVISION = "epoch-4"
CONFIG_DIR = Path(__file__).parent / "config"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GRPO (per-turn, DAPO) for the word games.")
    ap.add_argument("--arm", choices=["a", "b", "c"], help="preset init+task (a/b/c).")
    ap.add_argument("--init-repo", help="HF repo of the SFT init (overrides --arm).")
    ap.add_argument("--init-revision", default=DEFAULT_INIT_REVISION)
    ap.add_argument("--task", choices=["wordle", "all"], help="RL task (overrides --arm).")
    ap.add_argument("--config", help="verl config name in config/ (no .yaml); default by arm.")
    # data / rollout
    ap.add_argument("--n", type=int, default=2000, help="distinct train targets (per game for all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-batch-size", type=int, default=64, help="target words per step (B).")
    ap.add_argument("--group-size", type=int, default=8, help="episodes per target (G = rollout.n).")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--no-thinking", dest="enable_thinking", action="store_false", default=True)
    # reward weights (None → dataclass default)
    for f in ("green-new", "yellow-new", "novelty-decay", "format-ok", "format-bad",
              "invalid-guess", "win-bonus", "win-discount-k"):
        ap.add_argument(f"--{f}", type=float)
    # DAPO / GRPO knobs (forwarded as verl overrides)
    ap.add_argument("--clip-ratio-low", type=float, default=0.2)
    ap.add_argument("--clip-ratio-high", type=float, default=0.28)   # clip-higher
    ap.add_argument("--kl-loss-coef", type=float, default=0.001)     # set 0 to remove KL (DAPO)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--filter-top-variance-frac", type=float, default=1.0)  # StarPO-S off by default
    ap.add_argument("--total-epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, help="cap steps (use ~5 to PROBE the verl seam).")
    ap.add_argument("--n-gpus", type=int, default=1)
    # infra
    ap.add_argument("--work-dir", default="/workspace/grpo")
    ap.add_argument("--hub-model-id", help="push GRPO checkpoints here (step-N revisions).")
    ap.add_argument("--wandb-project", default="pixelpolicy-grpo")
    ap.add_argument("--run-name")
    ap.add_argument("--smoke-local", action="store_true",
                    help="validate rollout→reward→packing locally with a fake tokenizer (no verl/GPU).")
    ap.add_argument("overrides", nargs="*", help="extra verl Hydra overrides (key=value).")
    return ap.parse_args(argv)


def resolve(args: argparse.Namespace) -> dict:
    preset = ARMS.get(args.arm, {}) if args.arm else {}
    init_repo = args.init_repo or preset.get("init_repo")
    task = args.task or preset.get("task")
    if not (init_repo and task):
        sys.exit("Specify --arm, or both --init-repo and --task.")
    config = args.config or {"wordle": "grpo_wordle", "all": "grpo_all"}[task]
    if args.arm == "b" and not args.config:
        config = "grpo_wordle_full"
    return {"init_repo": init_repo, "task": task, "config": config,
            "run_name": args.run_name or (f"grpo-{args.arm}" if args.arm else f"grpo-{task}")}


def reward_weights(args: argparse.Namespace) -> reward_mod.RewardWeights:
    g = lambda name: getattr(args, name.replace("-", "_"))  # noqa: E731
    return reward_mod.RewardWeights.from_overrides(
        green_new=g("green-new"), yellow_new=g("yellow-new"), novelty_decay=g("novelty-decay"),
        format_ok=g("format-ok"), format_bad=g("format-bad"), invalid_guess=g("invalid-guess"),
        win_bonus=g("win-bonus"), win_discount_k=g("win-discount-k"), require_think=args.enable_thinking)


def good_status_map(task: str) -> dict[str, str]:
    games = ["wordle"] if task == "wordle" else list(GAMES.keys())
    return {gname: GAMES[gname]().good_status for gname in games}


# ---------------------------------------------------------------------------
# Local smoke: exercise the full rollout→reward→packing pipeline with NO verl/GPU.
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)

    def encode(self, text):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True, **kw):
        s = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return self.encode(s + ("<|assistant|>" if add_generation_prompt else ""))


def smoke_local(args: argparse.Namespace, r: dict) -> None:
    from training.grpo.sample_packing import pack_rows

    weights = reward_weights(args)
    rng = random.Random(args.seed)
    spec = GAMES["wordle"]()
    targets = spec.sample_targets(min(args.train_batch_size, 4), "train", rng)
    specs = make_groups(targets, args.group_size, game="wordle")
    tok = _FakeTokenizer()

    # a scripted, lockstep generator: round 1 a broad open, then guess the (known, smoke-only) target
    def batch_generate(prompts):
        # Round-robin a couple of canned replies; correctness of REWARDS isn't the point here,
        # only that the pipeline runs and shapes/uids are sane.
        return [tok.encode("<think>x</think>\n<guess>crane</guess>") for _ in prompts]

    samples, outcomes = roll_batch(specs, tokenizer=tok, batch_generate=batch_generate,
                                   weights=weights, enable_thinking=args.enable_thinking,
                                   max_turns=args.max_turns)
    rows = pack_rows(samples, pad_id=0, max_prompt_len=1024, max_response_len=1024)
    uids = sorted({s.uid for s in samples})
    print(f"[smoke] targets={len(targets)} group_size={args.group_size} episodes={len(specs)}")
    print(f"[smoke] per-turn samples={len(samples)} (variable per episode); distinct uids={len(uids)}")
    print(f"[smoke] example uids: {uids[:6]}")
    print(f"[smoke] reward range: [{min(s.reward for s in samples):.3f}, "
          f"{max(s.reward for s in samples):.3f}]")
    print(f"[smoke] packed prompt_ids[0] len={len(rows['prompt_ids'][0])}, "
          f"response_mask[0] sum={sum(rows['response_mask'][0])}")
    print("[smoke] OK — rollout→reward→packing pipeline runs end-to-end with no verl/GPU.")


# ---------------------------------------------------------------------------
# Real run: prep + launch the verl-backed per-turn GRPO trainer.
# ---------------------------------------------------------------------------
def run_verl(args: argparse.Namespace, r: dict) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    weights_json = str(work / "reward_weights.json")
    reward_mod.dump_config(weights_json, reward_weights(args), good_status_map(r["task"]))
    os.environ["GRPO_REWARD_WEIGHTS_JSON"] = weights_json

    init_dir = str(work / "init" / r["init_repo"].split("/")[-1])
    parquet = str(work / f"{r['task']}_train.parquet")

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=r["init_repo"], revision=args.init_revision,
                      local_dir=init_dir, token=os.getenv("HF_TOKEN"))
    data_mod.build_dataset(r["task"], args.n, args.seed, parquet)

    # VERL-VERSION: launch the per-turn GRPO trainer. We base it on verl's DAPO recipe
    # (recipe/dapo) and replace its rollout step with `rollout.roll_batch` (per-turn samples) +
    # `sample_packing.to_dataproto`. The trainer module (training/grpo/trainer.py) wires verl's
    # RayPPOTrainer (advantage by uid, actor update, ref, save) — confirm its API at the probe.
    from training.grpo import trainer

    trainer.run(
        config_dir=str(CONFIG_DIR), config_name=r["config"], init_model_dir=init_dir,
        train_parquet=parquet, reward_weights_json=weights_json,
        run_name=r["run_name"], wandb_project=args.wandb_project, hub_model_id=args.hub_model_id,
        train_batch_size=args.train_batch_size, group_size=args.group_size, max_turns=args.max_turns,
        enable_thinking=args.enable_thinking, clip_ratio_low=args.clip_ratio_low,
        clip_ratio_high=args.clip_ratio_high, kl_loss_coef=args.kl_loss_coef, lr=args.lr,
        filter_top_variance_frac=args.filter_top_variance_frac, total_epochs=args.total_epochs,
        max_steps=args.max_steps, n_gpus=args.n_gpus, extra_overrides=args.overrides,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    r = resolve(args)
    print(f"[grpo] arm={args.arm} init={r['init_repo']}@{args.init_revision} task={r['task']} "
          f"config={r['config']} run={r['run_name']} (B={args.train_batch_size}, G={args.group_size})")
    if args.smoke_local:
        smoke_local(args, r)
        return
    run_verl(args, r)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    main()
