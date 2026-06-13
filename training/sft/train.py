"""SFT entrypoint — fine-tune Qwen3.5-0.8B on the word-games dataset via TRL `SFTTrainer`.

Three variants (one shared trainer, selected by `--variant`):
  wordle      → wordle-only baseline                 (data_flat, games=["wordle"])
  full        → full set, no curriculum, shuffled    (data_flat, games=None)
  curriculum  → full set, curriculum order           (data_curriculum, --curriculum-strategy)

Every hyperparameter is a CLI flag (shell-overridable). Trains 4 epochs by default, saves a
checkpoint each epoch (keeps ALL of them), and — since the training machine has no git — pushes
each epoch checkpoint to its own Hub revision (`epoch-N`) so inference can later pull any one with
`from_pretrained(repo, revision="epoch-N")`.

Eval (optional, on by default; loss is a coarse sanity signal — the real metric is downstream task
accuracy you compute later per checkpoint). `eval_dataset` is a **dict** so the built-in Trainer→
wandb path logs BOTH an aggregated `eval_all_loss` and a per-game `eval_<game>_loss` each epoch, with
no custom metric code. Everything stays on the training box + wandb cloud.

    uv run --package training python -m training.sft.train --variant full --bf16 --push-to-hub \
        --hub-model-id you/word-games-sft-full --hub-per-epoch --report-to wandb
"""

from __future__ import annotations

import argparse
import os
from dataclasses import fields

from training.sft.data_flat import DEFAULT_MODEL, DEFAULT_REPO, load_flat
from training.sft.data_curriculum import ORDERED_STRATEGIES, load_curriculum
from training.sft.format import GAME_NO


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT trainer for the word-games dataset.")
    # what / where
    ap.add_argument("--variant", choices=["wordle", "full", "curriculum"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset-repo", default=DEFAULT_REPO)
    ap.add_argument("--output-dir", default=None, help="default ./runs/<variant>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-proc", type=int, default=4)
    # curriculum knobs
    ap.add_argument("--curriculum-strategy", choices=sorted(ORDERED_STRATEGIES | {"weighted"}),
                    default="widening")
    ap.add_argument("--replay-frac", type=float, default=0.03)
    ap.add_argument("--no-reasoning-throughout", action="store_true")
    # optimization (all shell-overridable)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="4096 covers all rows (longest train row = 5553 tokens, a single Wordle "
                         "outlier, only ~1.5k tokens truncated). See `python -m training.sft.stats`.")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lr-scheduler", default="cosine")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=-1, help="override for smoke tests.")
    # eval (coarse; loss is not the metric — downstream accuracy is)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--eval-samples-per-game", type=int, default=200,
                    help="0 disables the per-game eval_<game>_loss curves.")
    ap.add_argument("--eval-samples-all", type=int, default=1000,
                    help="0 disables the aggregated eval_all_loss.")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    # tracking
    ap.add_argument("--report-to", choices=["wandb", "trackio", "none"], default="wandb")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb-project", default="pixelpolicy-sft")
    ap.add_argument("--gradlog-steps", type=int, default=50,
                    help="log per-layer grad/update norms to wandb every N steps (0 disables). "
                         "See LEARNING_DYNAMICS_NOTES.md.")
    # hub
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-model-id", default=None)
    ap.add_argument("--hub-per-epoch", action="store_true",
                    help="push each epoch checkpoint to its own Hub revision epoch-N.")
    ap.add_argument("--hub-private", action="store_true", default=True)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------------

def build_train_dataset(args, tokenizer):
    if args.variant == "wordle":
        return load_flat(args.dataset_repo, split="train", games=["wordle"], tokenizer=tokenizer,
                         seed=args.seed, shuffle=True, num_proc=args.num_proc)
    if args.variant == "full":
        return load_flat(args.dataset_repo, split="train", games=None, tokenizer=tokenizer,
                         seed=args.seed, shuffle=True, num_proc=args.num_proc)
    return load_curriculum(args.dataset_repo, split="train", games=None, tokenizer=tokenizer,
                           strategy=args.curriculum_strategy,
                           reasoning_throughout=not args.no_reasoning_throughout,
                           replay_frac=args.replay_frac, seed=args.seed, num_proc=args.num_proc)


def build_eval_datasets(args, tokenizer):
    """A dict {"all": …, "<game>": …} on the eval split → aggregated + per-game eval loss in wandb."""
    games = ["wordle"] if args.variant == "wordle" else list(GAME_NO)
    evals: dict = {}

    def _subsample(g: list[str] | None, cap: int):
        if cap <= 0:
            return None
        ds = load_flat(args.dataset_repo, split=args.eval_split, games=g, tokenizer=tokenizer,
                       seed=args.seed, shuffle=True, num_proc=args.num_proc)
        return ds.select(range(min(cap, len(ds)))) if len(ds) else None

    agg = _subsample(games, args.eval_samples_all)
    if agg is not None:
        evals["all"] = agg
    if args.eval_samples_per_game > 0:
        for g in games:
            d = _subsample([g], args.eval_samples_per_game)
            if d is not None:
                evals[g] = d
    return evals or None


# --------------------------------------------------------------------------------------------
# trainer plumbing
# --------------------------------------------------------------------------------------------

def _make_sequential_trainer_cls(SFTTrainer):
    """SFTTrainer that iterates the train set in dataset order (honors curriculum ordering)."""
    from torch.utils.data import SequentialSampler

    class SequentialSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, *a, **kw):
            return SequentialSampler(self.train_dataset)

    return SequentialSFTTrainer


def make_sft_config(SFTConfig, **kwargs):
    """Construct an SFTConfig, tolerating field renames across TRL/transformers versions."""
    valid = {f.name for f in fields(SFTConfig)}
    aliases = {  # our name → possible config field names, newest first
        "max_seq_length": ["max_seq_length", "max_length"],
        "eval_strategy": ["eval_strategy", "evaluation_strategy"],
    }
    resolved: dict = {}
    for k, v in kwargs.items():
        target = next((c for c in aliases.get(k, [k]) if c in valid), None)
        if target is not None:
            resolved[target] = v
        else:
            print(f"[warn] SFTConfig has no field for {k!r}; ignoring.")
    return SFTConfig(**resolved)


def _make_trainer(cls, *, model, args_cfg, train_dataset, eval_dataset, tokenizer, callbacks):
    """Instantiate SFTTrainer across the processing_class/tokenizer arg rename."""
    common = dict(model=model, args=args_cfg, train_dataset=train_dataset,
                  eval_dataset=eval_dataset, callbacks=callbacks)
    try:
        return cls(processing_class=tokenizer, **common)
    except TypeError:
        return cls(tokenizer=tokenizer, **common)


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--push-to-hub requires --hub-model-id")

    from dotenv import load_dotenv
    load_dotenv()

    output_dir = args.output_dir or f"./runs/{args.variant}"
    if args.report_to == "wandb":
        # Force the project — a stray WANDB_PROJECT in the pod env (e.g. set to a HF repo id with a
        # "/") makes wandb.init reject it. Our --wandb-project always wins.
        os.environ["WANDB_PROJECT"] = args.wandb_project

    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen tokenizers ship `padding_side="left"` (the right choice for batched *generation*). For
    # causal-LM SFT we pad on the RIGHT so the completion tokens stay contiguous and next-token loss
    # aligns with the attention mask; pad positions are masked out of the loss anyway. (Inference/
    # vLLM uses left padding internally — separate concern.)
    tokenizer.padding_side = "right"

    print(f"[train] variant={args.variant} model={args.model} out={output_dir}")
    train_dataset = build_train_dataset(args, tokenizer)
    eval_dataset = build_eval_datasets(args, tokenizer)
    print(f"[train] train rows={len(train_dataset)} "
          f"eval datasets={list(eval_dataset) if eval_dataset else None}")

    model_init_kwargs = {"trust_remote_code": True}
    if args.bf16:
        model_init_kwargs["torch_dtype"] = "bfloat16"
    elif args.fp16:
        model_init_kwargs["torch_dtype"] = "float16"

    cfg = make_sft_config(
        SFTConfig,
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_len,
        packing=False,                       # required: keep prompt-completion masking intact
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="epoch",
        save_total_limit=None,               # keep ALL epoch checkpoints
        eval_strategy=("epoch" if eval_dataset else "no"),
        logging_steps=args.logging_steps,
        report_to=([] if args.report_to == "none" else [args.report_to]),
        run_name=args.run_name or f"{args.variant}",
        seed=args.seed,
        model_init_kwargs=model_init_kwargs,
        push_to_hub=False,                   # we push per-epoch revisions ourselves (see callback)
    )

    callbacks = []
    if args.push_to_hub and args.hub_per_epoch:
        from training.sft.upload import EpochHubPushCallback
        callbacks.append(EpochHubPushCallback(
            hub_model_id=args.hub_model_id, output_dir=output_dir, private=args.hub_private))
    if args.gradlog_steps > 0 and args.report_to == "wandb":
        from training.sft.dynamics import GradUpdateNormCallback
        callbacks.append(GradUpdateNormCallback(every=args.gradlog_steps))
        print(f"[train] logging per-layer grad/update norms to wandb every {args.gradlog_steps} steps")

    ordered = args.variant == "curriculum" and args.curriculum_strategy in ORDERED_STRATEGIES
    trainer_cls = _make_sequential_trainer_cls(SFTTrainer) if ordered else SFTTrainer
    if ordered:
        print(f"[train] curriculum '{args.curriculum_strategy}' → SequentialSampler (no reshuffle)")

    trainer = _make_trainer(trainer_cls, model=args.model, args_cfg=cfg,
                            train_dataset=train_dataset, eval_dataset=eval_dataset,
                            tokenizer=tokenizer, callbacks=callbacks)
    trainer.train()
    trainer.save_model(output_dir)

    # Push the final weights to the repo's main revision too (per-epoch branches handled by callback).
    if args.push_to_hub:
        from training.sft.upload import push_checkpoint
        push_checkpoint(output_dir, args.hub_model_id, revision="main", private=args.hub_private)
        print(f"[train] pushed final model → {args.hub_model_id} (main)")


if __name__ == "__main__":
    main()
