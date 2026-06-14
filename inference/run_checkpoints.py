"""Orchestrate evaluation across a variant's epoch checkpoints (local, one GPU, sequential).

For each checkpoint it: launches `inference.server` (vLLM) on the chosen revision, waits until the
server is ready, runs the full per-game evaluation against it, then tears the server down and moves
to the next. Resumable — a checkpoint whose `<out>/<label>.json` already exists is skipped.

    uv run --package inference python -m inference.run_checkpoints \
        --repo saketh-chervu/word-games-sft-wordle --epochs 1,2,3,4 --base \
        --games all --n 300 --seed 0 --out eval_results/
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def _variant(repo: str) -> str:
    """saketh-chervu/word-games-sft-wordle → 'wordle' (else the repo's last path component)."""
    name = repo.rstrip("/").split("/")[-1]
    return name.replace("word-games-sft-", "") or name


def _checkpoints(args) -> list[tuple[str, str, str | None]]:
    """[(label, model, revision)] — optional base first, then one per epoch."""
    variant = _variant(args.repo)
    ckpts: list[tuple[str, str, str | None]] = []
    if args.base:
        ckpts.append(("base", args.base_model, None))
    for e in [int(x) for x in args.epochs.split(",") if x.strip()]:
        ckpts.append((f"{variant}-e{e}", args.repo, f"epoch-{e}"))
    return ckpts


def _wait_ready(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    """Poll /v1/models until the server answers (first vLLM load is slow on WSL)."""
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode}) before becoming ready")
        try:
            if httpx.get(f"{base_url}/models", timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"server not ready after {timeout:.0f}s")


def _serve(model: str, revision: str | None, host: str, port: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "inference.server", "--model", model,
           "--host", host, "--port", str(port)]
    if revision:
        cmd += ["--revision", revision]
    print(f"[orchestrator] launching: {' '.join(cmd)}", file=sys.stderr)
    # New session so the whole vLLM process group can be signalled on teardown.
    return subprocess.Popen(cmd, start_new_session=True)


def _teardown(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a variant's epoch checkpoints sequentially.")
    ap.add_argument("--repo", required=True, help="Hub model repo, e.g. you/word-games-sft-wordle.")
    ap.add_argument("--epochs", default="1,2,3,4", help="comma list of epoch numbers.")
    ap.add_argument("--base", action="store_true", help="also evaluate the untrained base model.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--games", default="all")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--show", type=int, default=0,
                    help="print the first N raw episodes per game per checkpoint (format check).")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    from dotenv import load_dotenv
    from inference.evaluate import games_arg, run_and_save
    load_dotenv()

    games = games_arg(args.games)
    base_url = f"http://{args.host}:{args.port}/v1"
    ckpts = _checkpoints(args)
    print(f"[orchestrator] {len(ckpts)} checkpoint(s): {[c[0] for c in ckpts]}", file=sys.stderr)

    done = []
    for label, model, revision in ckpts:
        out_path = Path(args.out) / f"{label}.json"
        if out_path.exists():
            print(f"[orchestrator] skip {label} (exists: {out_path})", file=sys.stderr)
            done.append(label)
            continue
        proc = _serve(model, revision, args.host, args.port)
        try:
            _wait_ready(base_url, proc, args.ready_timeout)
            print(f"[orchestrator] {label} server ready → evaluating", file=sys.stderr)
            run_and_save(label=label, model=model, revision=revision, base_url=base_url,
                         games=games, n=args.n, seed=args.seed, concurrency=args.concurrency,
                         max_tokens=args.max_tokens, out=args.out, show=args.show)
            done.append(label)
        finally:
            _teardown(proc)

    print(f"[orchestrator] done: {done} → results in {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    _main()
