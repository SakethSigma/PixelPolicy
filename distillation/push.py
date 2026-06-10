"""Combine per-run SFT JSONL into one HuggingFace dataset and push to the Hub.

Defaults to the batch_play outputs (batch_low_sft.jsonl + batch_high_sft.jsonl). Each row is
one exploded move — {game, round, target, system, messages, completion, completion_no_think,
has_think} — and we add a `source` column (the file stem) for provenance. `huggingface_hub`
reads HF_TOKEN from the env; the repo id comes from --repo-id or HF_HUB_REPO_ID.

    # inspect without pushing (no token needed)
    uv run --package distillation python -m distillation.push --dry-run

    # push (needs HF_TOKEN + HF_HUB_REPO_ID in .env)
    uv run --package distillation python -m distillation.push
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_INPUTS = [
    "distillation/data/batch_low_sft.jsonl",
    "distillation/data/batch_high_sft.jsonl",
]


def load_rows(paths: list[str]) -> list[dict]:
    """Read every JSONL line from each path, tagging rows with their file stem as `source`."""
    rows: list[dict] = []
    for p in paths:
        stem = Path(p).stem
        for line in Path(p).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["source"] = stem
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Combine SFT JSONL and push to the HuggingFace Hub.")
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="SFT JSONL files to combine")
    ap.add_argument("--repo-id", default=os.environ.get("HF_HUB_REPO_ID"), help="Hub dataset repo id")
    ap.add_argument("--public", dest="private", action="store_false", default=True, help="push as a public dataset")
    ap.add_argument("--test-size", type=float, default=0.0, help="if >0, make a seeded train/test split")
    ap.add_argument("--dry-run", action="store_true", help="build + report stats, do NOT push (no token needed)")
    args = ap.parse_args(argv)

    from datasets import Dataset  # lazy: only pushing needs `datasets`

    rows = load_rows(args.inputs)
    ds = Dataset.from_list(rows)

    print(f"rows: {len(ds)}  |  columns: {ds.column_names}")
    print("by source :", dict(Counter(r["source"] for r in rows)))
    print("has_think :", dict(Counter(r["has_think"] for r in rows)))

    if args.test_size > 0:
        ds = ds.train_test_split(test_size=args.test_size, seed=0)
        print("split     :", {k: len(v) for k, v in ds.items()})

    if args.dry_run:
        print("dry-run: built the dataset but did NOT push.")
        return

    if not args.repo_id:
        raise SystemExit("No repo id — set HF_HUB_REPO_ID in .env or pass --repo-id.")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("No HF_TOKEN — set a write token in .env (huggingface.co/settings/tokens).")

    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"pushed -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
