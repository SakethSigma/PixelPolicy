"""Distillation config — teacher model + Hub target + on-disk paths, loaded from .env.

Mirrors agents/config.py (a frozen-ish dataclass with a `from_env` classmethod). Keys
live in .env.example: ANTHROPIC_API_KEY (read by the SDK directly), TEACHER_MODEL,
HF_HUB_REPO_ID, and HF_TOKEN (read by huggingface_hub directly).
"""

from __future__ import annotations

# TODO imports: `os`, `from dataclasses import dataclass, field`, `from pathlib import Path`.
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


class DistillConfig:
    teacher_model: str = "claude-opus-4-8"
    effort: str = "high"
    max_tokens: int = 4096
    concurrency: int = 8
    hub_repo_id: str = ""
    data_data: Path = Path("distillation/data")
    max_guesses: int = 6

    @classmethod
    def from_env(cls) -> "DistillConfig":
        load_dotenv()

# TODO: @dataclass
# class DistillConfig:
#     Fields to define (with sensible defaults):
#       teacher_model: str   = "claude-opus-4-8"   # the Claude teacher
#       effort: str          = "high"              # adaptive-thinking depth: low|medium|high|xhigh|max
#       max_tokens: int      = 4096               # per teacher reply (room for <think> + <guess>)
#       concurrency: int     = 8                  # parallel episodes in run_eval
#       hub_repo_id: str     = ""                 # where push.py uploads (e.g. "user/pixelpolicy-distill")
#       data_dir: Path       = Path("distillation/data")
#       max_guesses: int     = 6                  # quality gate ceiling (won in <= N rounds); 6 = no extra gate
#
#     Convenience properties:
#       raw_dir  -> data_dir / "raw"   (Trajectory JSONL, one file per game)
#       sft_dir  -> data_dir / "sft"   (exploded per-move SFT JSONL, one file per game)
#
#     @classmethod
#     def from_env(cls) -> "DistillConfig":
#         - load_dotenv() (from python-dotenv), then read:
#             TEACHER_MODEL   -> teacher_model (default "claude-opus-4-8")
#             HF_HUB_REPO_ID  -> hub_repo_id
#           (ANTHROPIC_API_KEY / HF_TOKEN are read by their SDKs, not here.)
#         - return cls(...). CLI flags in run.py override these.
