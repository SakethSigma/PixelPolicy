"""Push a GRPO checkpoint dir to the Hub — thin reuse of the SFT uploader.

GRPO checkpoints are pushed to ``saketh-chervu/word-games-grpo-<arm>`` under per-step revisions
(``step-N``) so the existing eval harness can serve any one with ``--revision step-N`` (identical to
how the SFT epochs are evaluated). Weights-only by default — all inference needs, far less bandwidth.
"""

from __future__ import annotations

import argparse

from training.sft.upload import push_checkpoint


def push_step(local_dir: str, hub_model_id: str, step: int, *, private: bool = True) -> str:
    """Upload one verl checkpoint dir as revision ``step-<N>``."""
    return push_checkpoint(local_dir, hub_model_id, revision=f"step-{step}",
                           private=private, weights_only=True)


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Push a GRPO checkpoint dir to the Hub.")
    ap.add_argument("local_dir")
    ap.add_argument("--hub-model-id", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    push_step(args.local_dir, args.hub_model_id, args.step, private=not args.public)


if __name__ == "__main__":
    _main()
