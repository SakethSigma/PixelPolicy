"""Re-export SFT JSONL from a raw batch_play dump, in the unified schema.

Raw is the source of truth (``distillation/README.md``): the full ``Trajectory`` dumps in
``data/*_raw.json`` carry every turn plus each episode's final ``status``, so we can re-shape
the SFT rows — here, upgrade them to the unified schema with a correct per-row ``valid`` flag
(``status == good_status``) — **without re-running any Claude rollouts**. This is how the
original Wordle data (whose SFT files predate the unified columns) joins the combined dataset.

    # regenerate the committed Wordle SFT files in the new schema:
    uv run --package distillation python -m distillation.reexport \
        distillation/data/batch_low_raw.json distillation/data/batch_high_raw.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillation.registry import GAME_NUMBERS, GAMES
from distillation.schema import sft_row


def _good_status(game_name: str) -> str:
    """The terminal status that counts as solved, from the game's spec (default ``won``)."""
    factory = GAMES.get(game_name)
    return factory().good_status if factory else "won"


def reexport(raw_path: Path, out_path: Path) -> int:
    """Read one raw dump, write unified-schema SFT rows, return the row count."""
    doc = json.loads(raw_path.read_text())
    game_name = doc["game"]
    game_no = GAME_NUMBERS.get(game_name, -1)
    good = _good_status(game_name)
    system = doc.get("system", "")

    rows: list[dict] = []
    for g in doc["games"]:
        valid = g.get("status") == good
        episode = g.get("episode", 0)
        for t in g["turns"]:
            rows.append(sft_row(
                game_name=game_name, game_no=game_no, round=t["round"], target=g["target"],
                system=system, messages=t["prompt"], completion=t["output"],
                valid=valid, episode=episode,
            ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept = sum(r["valid"] for r in rows)
    print(f"{raw_path.name} -> {out_path}  ({len(rows)} rows, {kept} valid)")
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Re-export raw batch dumps to unified-schema SFT JSONL.")
    ap.add_argument("raw", nargs="+", help="raw json dumps (each *_raw.json -> *_sft.jsonl)")
    args = ap.parse_args(argv)
    for r in args.raw:
        raw_path = Path(r)
        out_path = raw_path.with_name(raw_path.stem.replace("_raw", "_sft") + ".jsonl")
        reexport(raw_path, out_path)


if __name__ == "__main__":
    main()
