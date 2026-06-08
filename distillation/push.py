"""Stage 3 — combine every game's SFT samples into one dataset and push to the Hub.

Reads all data/sft/*.jsonl, concatenates them (the `game` column keeps provenance),
optionally dedups and splits, then push_to_hub. huggingface_hub reads HF_TOKEN from the
env for auth; the repo id comes from cfg.hub_repo_id (HF_HUB_REPO_ID in .env).
"""

from __future__ import annotations

# TODO imports:
#   import json
#   from datasets import Dataset, DatasetDict
#   from distillation.config import DistillConfig


# TODO: def load_all(cfg) -> list[dict]:
#   - glob cfg.sft_dir / "*.jsonl"; read every line -> json.loads; return the merged list.
#   - each record already has {"game", "messages", "completion"}.


# TODO: def build_dataset(cfg) -> Dataset | DatasetDict:
#   1. rows = load_all(cfg)
#   2. ds = Dataset.from_list(rows)
#   3. optional: dedup identical (messages, completion) pairs (e.g. hash the json of each).
#   4. optional: ds.train_test_split(test_size=...) -> DatasetDict({"train":..., "test":...}).
#   5. return ds. Print len(ds) and ds.features for a dry-run sanity check BEFORE pushing.


# TODO: def push(cfg, *, private=True) -> None:
#   - ds = build_dataset(cfg)
#   - guard: require cfg.hub_repo_id is set (else raise with a helpful message).
#   - ds.push_to_hub(cfg.hub_repo_id, private=private)
#   - tip: print the resulting https://huggingface.co/datasets/<repo_id> URL.
#
# Verification: push to a throwaway repo first, then `load_dataset(repo_id)` and assert
# the row count / features round-trip.
