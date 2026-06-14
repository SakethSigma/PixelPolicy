"""Push checkpoints to the HuggingFace Hub — the only output channel on the training machine.

The training box has no git checkout and we keep every epoch checkpoint, so each is uploaded to its
own **Hub revision** (`epoch-1`, `epoch-2`, …). Later, inference loads any one with
`AutoModelForCausalLM.from_pretrained(repo, revision="epoch-N")` (vLLM: `--revision epoch-N`).

- `push_checkpoint(dir, repo, revision=…)` — one-shot push of a local checkpoint dir.
- `EpochHubPushCallback` — a `TrainerCallback` that pushes each epoch checkpoint as it is saved
  (so completed epochs survive even if the run later dies).

Reads `HF_TOKEN` from the environment / `.env` (same convention as `distillation/push.py`).
By default uploads are **weights-only** (optimizer/scheduler/RNG state skipped) — all inference needs,
and far less bandwidth than full resumable checkpoints.
"""

from __future__ import annotations

import os

# Big training-only artifacts inference never needs.
_OPTIMIZER_PATTERNS = ["optimizer.pt", "scheduler.pt", "rng_state*.pth", "scaler.pt"]


def push_checkpoint(local_dir: str, hub_model_id: str, *, revision: str | None = None,
                    private: bool = True, weights_only: bool = True,
                    token: str | None = None) -> str:
    """Upload one checkpoint directory to `hub_model_id` (optionally on branch `revision`)."""
    from huggingface_hub import HfApi

    token = token or os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(hub_model_id, repo_type="model", private=private, exist_ok=True)
    if revision and revision != "main":
        api.create_branch(hub_model_id, branch=revision, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=hub_model_id,
        repo_type="model",
        revision=revision,
        commit_message=f"upload {os.path.basename(local_dir.rstrip('/'))}"
                       + (f" @ {revision}" if revision else ""),
        ignore_patterns=_OPTIMIZER_PATTERNS if weights_only else None,
    )
    where = revision or "main"
    print(f"[upload] {local_dir} → {hub_model_id} (revision={where})")
    return f"{hub_model_id}@{where}"


def _epoch_callback_base():
    """Build the callback class lazily so importing this module doesn't require transformers."""
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    class EpochHubPushCallback(TrainerCallback):
        """On each checkpoint save, push to Hub:

        1. `epoch-N` — **weights-only** (clean, small; what inference pulls), and
        2. `resume`  — the **FULL** checkpoint incl. optimizer/scheduler/RNG/trainer_state,
           OVERWRITTEN each epoch. This is the crash-recovery copy: if the box dies, download
           `revision=resume` and `train(resume_from_checkpoint=…)` continues from the last epoch
           with the optimizer state intact. (Set `push_resume=False` to skip the heavier upload.)
        """

        def __init__(self, *, hub_model_id: str, output_dir: str, private: bool = True,
                     weights_only: bool = True, push_resume: bool = True):
            self.hub_model_id = hub_model_id
            self.output_dir = output_dir
            self.private = private
            self.weights_only = weights_only
            self.push_resume = push_resume

        def on_save(self, args, state, control, **kwargs):
            ckpt = os.path.join(self.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            epoch = round(state.epoch) if state.epoch is not None else state.global_step
            if os.path.isdir(ckpt):
                push_checkpoint(ckpt, self.hub_model_id, revision=f"epoch-{epoch}",
                                private=self.private, weights_only=self.weights_only)
                if self.push_resume:                       # full checkpoint → recoverable from HF
                    push_checkpoint(ckpt, self.hub_model_id, revision="resume",
                                    private=self.private, weights_only=False)
            return control

    return EpochHubPushCallback


def __getattr__(name: str):
    """Lazily build `EpochHubPushCallback` on first access (defers the transformers import).

    Keeps `push_checkpoint` and the manual CLI usable with only `huggingface_hub` installed.
    """
    if name == "EpochHubPushCallback":
        return _epoch_callback_base()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Push a checkpoint directory to the Hub.")
    ap.add_argument("local_dir")
    ap.add_argument("--hub-model-id", required=True)
    ap.add_argument("--revision", default=None, help="e.g. epoch-2 (branch); default main.")
    ap.add_argument("--public", action="store_true", help="create a public repo (default private).")
    ap.add_argument("--full", action="store_true", help="include optimizer/scheduler state.")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    push_checkpoint(args.local_dir, args.hub_model_id, revision=args.revision,
                    private=not args.public, weights_only=not args.full)


if __name__ == "__main__":
    _main()
