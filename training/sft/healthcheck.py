"""Pre-flight checks before a real run: model loads, data loads, training is stable, and a
batch-size sweep to cap GPU memory. Needs torch + a GPU (run on the training machine).

    # smoke: load model+data, run a few steps, assert loss is finite
    uv run --package training python -m training.sft.healthcheck --variant full --bf16 --steps 10

    # batch-size sweep: find the largest per-device batch that fits, with peak VRAM per size
    uv run --package training python -m training.sft.healthcheck --variant full --bf16 \
        --gradient-checkpointing --sweep 1,2,4,8,16 --max-seq-len 4096
"""

from __future__ import annotations

import argparse

from training.sft.data_flat import DEFAULT_MODEL, DEFAULT_REPO, load_flat
from training.sft.train import make_sft_config, _make_trainer
from training.sft.format import GAME_NO


def _small_dataset(args, tokenizer, n: int):
    games = ["wordle"] if args.variant == "wordle" else None
    ds = load_flat(args.dataset_repo, split="train", games=games, tokenizer=tokenizer,
                   seed=0, shuffle=False, num_proc=args.num_proc)
    # WORST CASE: take the LONGEST rows so the batch pads near max_seq_len — random short rows
    # under-report peak memory (the cross-entropy logits scale with batch×seq×vocab).
    ds = ds.map(lambda x: {"_len": len(x["prompt"]) + len(x["completion"])}, num_proc=args.num_proc)
    ds = ds.sort("_len", reverse=True).select(range(min(n, len(ds))))
    return ds.remove_columns("_len")


def _run_once(SFTConfig, SFTTrainer, *, model_id, tokenizer, dataset, batch_size, args):
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model_init_kwargs = {"trust_remote_code": True}
    if args.bf16:
        model_init_kwargs["torch_dtype"] = "bfloat16"
    elif args.fp16:
        model_init_kwargs["torch_dtype"] = "float16"

    cfg = make_sft_config(
        SFTConfig,
        output_dir=f"./runs/_healthcheck/bs{batch_size}",
        max_steps=args.steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        packing=False,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        report_to=[],
        seed=0,
        model_init_kwargs=model_init_kwargs,
        push_to_hub=False,
    )
    trainer = _make_trainer(SFTTrainer, model=model_id, args_cfg=cfg, train_dataset=dataset,
                            eval_dataset=None, tokenizer=tokenizer, callbacks=[])
    out = trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    loss = out.training_loss
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss, peak


def _main() -> None:
    ap = argparse.ArgumentParser(description="SFT pre-flight checks + batch-size sweep.")
    ap.add_argument("--variant", choices=["wordle", "full"], default="full")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset-repo", default=DEFAULT_REPO)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--rows", type=int, default=512, help="data rows used for the checks.")
    ap.add_argument("--sweep", default=None, help="comma list of batch sizes, e.g. 1,2,4,8.")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"[healthcheck] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[healthcheck] device={torch.cuda.get_device_name(0)} "
              f"total_vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # 1) tokenizer + data load
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    ds = _small_dataset(args, tok, args.rows)
    assert {"prompt", "completion"} <= set(ds.column_names), "loader must yield prompt/completion"
    print(f"[healthcheck] data OK: {len(ds)} rows; columns={ds.column_names}")
    print(f"[healthcheck] example prompt[:200]={ds[0]['prompt'][:200]!r}")

    sizes = [int(x) for x in args.sweep.split(",")] if args.sweep else [4]

    print(f"\n[healthcheck] running {args.steps} steps per batch size: {sizes}")
    print(f"  {'batch':>6} {'status':<10} {'peak_vram_GB':>13} {'final_loss':>12}")
    largest_ok = None
    for bs in sizes:
        try:
            loss, peak = _run_once(SFTConfig, SFTTrainer, model_id=args.model, tokenizer=tok,
                                   dataset=ds, batch_size=bs, args=args)
            finite = loss == loss and abs(loss) != float("inf")  # NaN/inf guard
            status = "ok" if finite else "NON-FINITE"
            if finite:
                largest_ok = bs
            print(f"  {bs:>6} {status:<10} {peak:>13.2f} {loss:>12.4f}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {bs:>6} {'OOM':<10} {'-':>13} {'-':>12}")
            break  # bigger sizes will also OOM
        except Exception as e:  # noqa: BLE001 - surface any other failure clearly
            print(f"  {bs:>6} {'ERROR':<10} {str(e)[:60]}")
            break

    if largest_ok is not None:
        print(f"\n[healthcheck] PASS — training stable; largest fitting per-device batch = {largest_ok}")
        print("  tip: set --per-device-batch-size to that and pick --grad-accum for your target "
              "effective batch (effective = per_device * grad_accum * n_gpus).")
    else:
        print("\n[healthcheck] no batch size fit / loss non-finite — see errors above.")


if __name__ == "__main__":
    _main()
