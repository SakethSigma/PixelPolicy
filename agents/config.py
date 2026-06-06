"""Where the model endpoint is defined. Loaded from ``.env``; CLI flags override.

Keys (see ``.env.example``):
    OPENAI_BASE_URL   local vLLM server (``http://127.0.0.1:8000/v1``) or an OpenAI URL
    INFERENCE_MODEL   the model id the server is serving
    OPENAI_API_KEY    any non-empty value for local vLLM
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    # Sampling. Qwen3.5-0.8B is prone to thinking-loops; its card recommends top_p≈0.95 +
    # presence_penalty≈1.5 + top_k≈20 to keep generation terminating. temperature is kept low
    # (0.3) for steadier format-following — the card suggests 1.0, so raise it if you want more
    # exploratory play.
    temperature: float = 0.3
    top_p: float = 0.95
    presence_penalty: float = 1.5
    top_k: int = 20
    max_tokens: int = 2048  # room for <think>…</think> reasoning before the <guess>
    enable_thinking: bool = True  # Qwen3.5 has no /think switch — this API param turns it on

    @classmethod
    def from_env(cls) -> "AgentConfig":
        from dotenv import load_dotenv

        load_dotenv()
        return cls(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("INFERENCE_MODEL", "Qwen/Qwen3.5-0.8B"),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
