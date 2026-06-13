"""Combine per-game SFT JSONL into one HuggingFace dataset and push to the Hub.

Defaults to the Wordle batch outputs + the programmatic charcount file. Every row is upgraded
to the **unified schema** — common columns {game_name, game_no, round, valid, target, system,
messages, completion, completion_no_think, has_think, episode} — plus a `source` column (the
file stem) for provenance. Legacy Wordle rows (whose `game` field was the episode index) are
normalized on load, so old and new files combine without re-running rollouts.
`huggingface_hub` reads HF_TOKEN from the env; the repo id comes from --repo-id or HF_HUB_REPO_ID.

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

from distillation.schema import normalize_legacy

DEFAULT_INPUTS = [
    "distillation/data/batch_low_sft.jsonl",
    "distillation/data/batch_high_sft.jsonl",
    "distillation/data/charcount_sft.jsonl",
    "distillation/data/validity_sft.jsonl",
    "distillation/data/anagram_sft.jsonl",
    "distillation/data/rhyme_sft.jsonl",
    "distillation/data/crossword_sft.jsonl",
    "distillation/data/charset_sft.jsonl",
    "distillation/data/mistakeid_sft.jsonl",
    "distillation/data/tower_sft.jsonl",
    "distillation/data/endstart_sft.jsonl",
    "distillation/data/codebreaker_sft.jsonl",
    "distillation/data/bullscows_sft.jsonl",
    "distillation/data/consistency_sft.jsonl",
]

# Map a file stem to its game, so legacy rows (game = episode index) can be normalized. New
# unified-schema rows already carry game_name/game_no and pass through unchanged.
_STEM_GAME = {
    "batch_low_sft": ("wordle", 0),
    "batch_high_sft": ("wordle", 0),
    "batch_sft": ("wordle", 0),
    "charcount_sft": ("charcount", 1),
    "validity_sft": ("validity", 2),
    "anagram_sft": ("anagram", 3),
    "rhyme_sft": ("rhyme", 5),
    "crossword_sft": ("crossword", 6),
    "charset_sft": ("charset", 7),
    "mistakeid_sft": ("mistakeid", 8),
    "tower_sft": ("tower", 9),
    "endstart_sft": ("endstart", 4),
    "codebreaker_sft": ("codebreaker", 10),
    "bullscows_sft": ("bullscows", 11),
    "consistency_sft": ("consistency", 12),
}


def load_rows(paths: list[str]) -> list[dict]:
    """Read every JSONL line, normalize to the unified schema, and tag with `source` (file stem)."""
    rows: list[dict] = []
    for p in paths:
        if not Path(p).exists():
            print(f"  skipping missing input: {p}")
            continue
        stem = Path(p).stem
        game_name, game_no = _STEM_GAME.get(stem, ("unknown", -1))
        for line in Path(p).read_text().splitlines():
            if line.strip():
                row = normalize_legacy(json.loads(line), game_name=game_name, game_no=game_no)
                # Wordle quality gate is FORMAT compliance, not winning: a well-formed move (one
                # that reasoned in a <think> block) is a good SFT target even from a lost game, and
                # a move with no <think> is dropped regardless of the outcome. So re-derive Wordle's
                # `valid` from has_think rather than the won-based flag the rollouts wrote.
                if row.get("game_name") == "wordle":
                    row["valid"] = bool(row.get("has_think"))
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
    ap.add_argument("--overwrite", action="store_true",
                    help="wipe the repo's existing data + card before pushing (needed when the "
                         "schema changed — push_to_hub refuses to merge mismatched features)")
    ap.add_argument("--dry-run", action="store_true", help="build + report stats, do NOT push (no token needed)")
    args = ap.parse_args(argv)

    from datasets import Dataset  # lazy: only pushing needs `datasets`

    rows = load_rows(args.inputs)
    ds = Dataset.from_list(rows)

    print(f"rows: {len(ds)}  |  columns: {ds.column_names}")
    print("by game   :", dict(Counter(r["game_name"] for r in rows)))
    print("by source :", dict(Counter(r["source"] for r in rows)))
    print("valid     :", dict(Counter(r["valid"] for r in rows)))
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

    if args.overwrite:
        _wipe_existing(args.repo_id, private=args.private)

    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"pushed -> https://huggingface.co/datasets/{args.repo_id}")


def _wipe_existing(repo_id: str, *, private: bool) -> None:
    """Remove the repo's existing data shards + dataset card so a schema change can push clean.

    ``push_to_hub`` compares new features against the existing dataset card's ``dataset_info``
    and refuses to merge a different schema. When we deliberately change the schema, delete the
    old ``data/`` folder and ``README.md`` first (the next push regenerates both).
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError  # EntryNotFoundError (404) is a subclass

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    try:
        api.delete_folder(path_in_repo="data", repo_id=repo_id, repo_type="dataset",
                          commit_message="overwrite: drop old-schema data")
    except HfHubHTTPError:  # nothing to delete (no existing data/ folder)
        pass
    for path in ("README.md", "dataset_infos.json"):
        try:
            api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type="dataset",
                            commit_message="overwrite: drop old dataset card")
        except HfHubHTTPError:
            pass
    print(f"overwrite: cleared existing data/card on {repo_id}")


if __name__ == "__main__":
    main()
