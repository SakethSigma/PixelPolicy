"""Sequence-length stats for the SFT data — prompt + completion token lengths.

Tokenizes the rendered prompt/completion (the real training sequence) and reports percentiles, max,
a histogram, and how many rows would be **truncated** at a given `--max-seq-len`. CPU-only — no model,
no GPU, no training compute needed. Use this to pick `--max-seq-len` and reason about batch memory.

    uv run --package training python -m training.sft.stats                       # full train split
    uv run --package training python -m training.sft.stats --games wordle         # one game
    uv run --package training python -m training.sft.stats --sample 5000 --max-seq-len 2048
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from training.sft.format import GAME_NO, build_example
from training.sft.data_flat import DEFAULT_MODEL, DEFAULT_REPO, load_valid


def _pcts(values: list[int], ps=(50, 90, 95, 99)) -> dict:
    if not values:
        return {p: 0 for p in ps} | {"max": 0, "mean": 0, "n": 0}
    s = sorted(values)
    out = {p: s[min(len(s) - 1, int(p / 100 * len(s)))] for p in ps}
    out["max"] = s[-1]
    out["mean"] = round(sum(s) / len(s), 1)
    out["n"] = len(s)
    return out


def _row(label: str, st: dict) -> str:
    return (f"  {label:<12} n={st['n']:<7} mean={st['mean']:<7} "
            f"p50={st[50]:<6} p90={st[90]:<6} p95={st[95]:<6} p99={st[99]:<6} max={st['max']}")


def _histogram(totals: list[int], edges=(256, 512, 1024, 2048, 3072, 4096, 8192)) -> None:
    buckets = defaultdict(int)
    for t in totals:
        placed = False
        for e in edges:
            if t <= e:
                buckets[e] += 1
                placed = True
                break
        if not placed:
            buckets[f">{edges[-1]}"] += 1
    n = len(totals)
    print("\ntotal-length histogram (prompt+completion tokens):")
    prev = 0
    for e in edges:
        c = buckets.get(e, 0)
        bar = "#" * int(40 * c / n) if n else ""
        print(f"  {prev:>5}-{e:<5} {c:>7} ({100*c/n:4.1f}%) {bar}")
        prev = e
    over = buckets.get(f">{edges[-1]}", 0)
    if over:
        bar = "#" * int(40 * over / n)
        print(f"  >{edges[-1]:<10} {over:>7} ({100*over/n:4.1f}%) {bar}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="SFT sequence-length stats (CPU-only).")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--games", nargs="*", default=None, help="filter to these game(s).")
    ap.add_argument("--sample", type=int, default=0, help="cap rows (0 = full split).")
    ap.add_argument("--max-seq-len", type=int, default=4096, help="report truncation at this length.")
    ap.add_argument("--num-proc", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    raw = load_valid(args.repo_id, split=args.split, games=args.games)
    if args.sample and len(raw) > args.sample:
        raw = raw.shuffle(seed=0).select(range(args.sample))
    print(f"[stats] {args.repo_id} split={args.split} "
          f"games={args.games or 'ALL'} rows={len(raw)} model={args.model}\n")

    def measure(row):
        ex = build_example(row, tok)
        return {"_plen": len(tok(ex["prompt"], add_special_tokens=False)["input_ids"]),
                "_clen": len(tok(ex["completion"], add_special_tokens=False)["input_ids"])}

    ds = raw.map(measure, num_proc=args.num_proc, desc="tokenize lengths")
    plen, clen, names = ds["_plen"], ds["_clen"], ds["game_name"]
    total = [p + c for p, c in zip(plen, clen)]

    print("OVERALL token lengths:")
    print(_row("prompt", _pcts(plen)))
    print(_row("completion", _pcts(clen)))
    print(_row("total", _pcts(total)))

    over = sum(1 for t in total if t > args.max_seq_len)
    print(f"\ntruncation @ max-seq-len={args.max_seq_len}: "
          f"{over} rows ({100*over/len(total):.2f}%) exceed it "
          f"(longest = {max(total)} tokens)")

    _histogram(total)

    by_game_total = defaultdict(list)
    for n, t in zip(names, total):
        by_game_total[n].append(t)
    print("\nper-game TOTAL length:")
    for name in sorted(by_game_total, key=lambda g: GAME_NO.get(g, 99)):
        print(_row(name, _pcts(by_game_total[name])))


if __name__ == "__main__":
    _main()
