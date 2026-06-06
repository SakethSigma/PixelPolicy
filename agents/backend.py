"""OpenAI-compatible backend — the only model/network seam (inference).

vLLM and OpenAI both speak the Chat Completions API, so one adapter covers a local
Qwen3-VL server and hosted OpenAI; only ``base_url`` differs. ``openai`` is imported
lazily so importing this module (or the rest of the layer) doesn't require it until a
backend is actually constructed.

Training does NOT use this — the RL library supplies its own ``generate`` (its policy
engine). See ``training_integration.md``.
"""

from __future__ import annotations

from typing import Any

from agents.base import Completion


class OpenAICompatBackend:
    """``messages -> Completion`` over an OpenAI-compatible endpoint.

    Batch-first to satisfy :class:`~agents.base.LLMBackend`; each prompt is one Chat
    Completions call (episode-level concurrency is handled by ``run_eval``).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",  # vLLM accepts any non-empty key
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        enable_thinking: bool | None = True,
    ):
        from openai import OpenAI  # lazy: only constructing a backend needs openai

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        # Standard Chat Completions params (None → leave at server default).
        self._defaults: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if top_p is not None:
            self._defaults["top_p"] = top_p
        if presence_penalty is not None:
            self._defaults["presence_penalty"] = presence_penalty
        # vLLM-only knobs go in `extra_body`:
        #   top_k                — sampling param not in the OpenAI schema.
        #   chat_template_kwargs — forwarded to the model's chat template; Qwen3.5 reads
        #                          `enable_thinking` there. (Qwen3.5 has no /think soft switch,
        #                          so this API parameter is how you turn reasoning on.)
        extra: dict[str, Any] = {}
        if top_k is not None:
            extra["top_k"] = top_k
        if enable_thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        self._extra_body: dict[str, Any] | None = extra or None

    def generate(self, prompts: list[list[dict]], **sampling: Any) -> list[Completion]:
        params = {**self._defaults, **sampling}
        if self._extra_body is not None:
            params["extra_body"] = {**self._extra_body, **params.get("extra_body", {})}
        out: list[Completion] = []
        for messages in prompts:
            resp = self._client.chat.completions.create(
                model=self._model, messages=messages, **params
            )
            choice = resp.choices[0]
            out.append(
                Completion(
                    text=choice.message.content or "",
                    finish_reason=choice.finish_reason,
                    raw=resp.model_dump(),
                )
            )
        return out
