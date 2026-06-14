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


def _serve(model: str, revision: str | None, host: str, port: int,
           enforce_eager: bool = False, max_model_len: int | None = None,
           max_num_seqs: int | None = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "inference.server", "--model", model,
           "--host", host, "--port", str(port)]
    if revision:
        cmd += ["--revision", revision]
    if enforce_eager:
        # Skip CUDA-graph capture (very slow on WSL/12GB and re-run per server) → fast startup.
        cmd += ["--enforce-eager"]
    if max_model_len:
        # Smaller KV cache → fewer engine OOM crashes on a tight (12GB) card under concurrency.
        cmd += ["--max-model-len", str(max_model_len)]
    if max_num_seqs:
        # vLLM caps concurrently-running sequences at 256 by default; raise it to let a big-VRAM
        # box (A100) actually run a high client --concurrency instead of queueing the overflow.
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    print(f"[orchestrator] launching: {' '.join(cmd)}", file=sys.stderr)
    # New session so the whole vLLM process group can be signalled on teardown.
    return subprocess.Popen(cmd, start_new_session=True)


def _push_results(out_dir: str, repo: str, revision: str) -> None:
    """Upload the whole eval results dir (metrics + raw/) to a branch of `repo` — hands-off exfil."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.create_repo(repo, repo_type="model", exist_ok=True)
        if revision != "main":
            api.create_branch(repo, branch=revision, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=repo, repo_type="model", revision=revision,
                          commit_message=f"eval results {os.path.basename(out_dir.rstrip('/'))}")
        print(f"[orchestrator] pushed {out_dir} → {repo}@{revision}", file=sys.stderr)
    except Exception as e:                                  # noqa: BLE001 — push failure must not abort the run
        print(f"[orchestrator] results push skipped: {e}", file=sys.stderr)


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
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="max NEW tokens generated per turn (2048 truncated ~14% of thinking turns).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="forward to the vLLM server; smaller = smaller KV cache (fewer engine OOMs "
                         "on a 12GB card). Pair with a lower --concurrency.")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="forward to vLLM; raises its 256 default cap on concurrently-running "
                         "sequences so a high --concurrency actually runs in parallel (A100).")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="pass --enforce-eager to vLLM — skips slow CUDA-graph capture (recommended "
                         "on WSL/12GB; capture otherwise re-runs ~20-30 min per checkpoint).")
    ap.add_argument("--show", type=int, default=0,
                    help="print the first N raw episodes per game per checkpoint (format check).")
    ap.add_argument("--no-store-raw", action="store_true",
                    help="do NOT persist raw per-episode predictions (default: store to out/raw/<label>/).")
    ap.add_argument("--push-results-repo", default=None,
                    help="HF repo to auto-upload the whole eval dir (metrics + raw/) to after each "
                         "checkpoint — hands-off, fetch on local. e.g. saketh-chervu/word-games-sft-wordle")
    ap.add_argument("--push-results-revision", default="eval", help="branch for --push-results-repo.")
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
        proc = _serve(model, revision, args.host, args.port, enforce_eager=args.enforce_eager,
                      max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs)
        try:
            _wait_ready(base_url, proc, args.ready_timeout)
            print(f"[orchestrator] {label} server ready → evaluating", file=sys.stderr)
            run_and_save(label=label, model=model, revision=revision, base_url=base_url,
                         games=games, n=args.n, seed=args.seed, concurrency=args.concurrency,
                         max_tokens=args.max_tokens, out=args.out, show=args.show,
                         store_raw=not args.no_store_raw)
            done.append(label)
            if args.push_results_repo:                     # auto-exfil after each checkpoint
                _push_results(args.out, args.push_results_repo, args.push_results_revision)
        finally:
            _teardown(proc)

    print(f"[orchestrator] done: {done} → results in {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    _main()
