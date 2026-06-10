"""OpenAI-compatible backend — the only model/network seam (inference).

vLLM and OpenAI both speak the Chat Completions API, so one adapter covers a local
Qwen3-VL server and hosted OpenAI; only ``base_url`` differs. ``openai`` is imported
lazily so importing this module (or the rest of the layer) doesn't require it until a
backend is actually constructed.

Training does NOT use this — the RL library supplies its own ``generate`` (its policy
engine). See ``training_integration.md``.
"""

from __future__ import annotations

import time
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


class AnthropicBackend:
    """``messages -> Completion`` via the native Anthropic SDK — the **teacher** seam.

    Same :class:`~agents.base.LLMBackend` contract as :class:`OpenAICompatBackend`
    (``generate(prompts) -> [Completion]``), so it drops straight into
    ``run_eval`` / ``run_episode`` with no rollout changes — that is how distillation
    reuses the existing game loop to record teacher trajectories.

    Two adaptations to the native API:

    1. **System turn is hoisted.** The OpenAI shape carries ``{"role": "system"}`` inside
       ``messages``; Anthropic takes the system prompt as a separate ``system=`` argument.
       We split it out per request. (Agents build exactly one leading system turn — see
       ``WordleAgent.build_messages`` — and the remainder already starts with ``user`` and
       alternates, which is what the Messages API requires.)
    2. **Reasoning is re-wrapped into ``<think>…</think>``.** Claude returns thinking and
       text as separate content blocks; the student's format (and the agents' parser) is
       ``<think>reasoning</think>\\n<guess>word</guess>``. We concatenate the summarized
       thinking back into that single string so teacher text is byte-compatible with the
       student target and parses through ``WordleAgent.parse_action`` unchanged.

    Adaptive thinking only (Opus 4.8/4.7): no ``temperature``/``top_p``/``budget_tokens``
    — those 400 on these models. Depth is controlled by ``effort``. ``anthropic`` is
    imported lazily (optional ``[teacher]`` extra) so the agents layer doesn't require it.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        api_key: str | None = None,  # None → SDK reads ANTHROPIC_API_KEY from the env
        max_tokens: int = 4096,
        effort: str = "high",  # low | medium | high | xhigh | max
        thinking_display: str = "summarized",  # "summarized" surfaces reasoning; "omitted" hides it
        cache_system: bool = True,  # cache the system prompt (game prompts are long + repeated)
    ):
        from anthropic import Anthropic  # lazy: only the teacher backend needs anthropic

        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._thinking_display = thinking_display
        self._cache_system = cache_system

    def generate(self, prompts: list[list[dict]], **sampling: Any) -> list[Completion]:
        max_tokens = sampling.get("max_tokens", self._max_tokens)
        effort = sampling.get("effort", self._effort)
        out: list[Completion] = []
        for messages in prompts:
            resp = self._client.messages.create(
                **self._create_params(messages, max_tokens=max_tokens, effort=effort)
            )
            out.append(
                Completion(
                    text=self._to_text(resp),
                    finish_reason=resp.stop_reason,
                    raw=resp.model_dump(),
                )
            )
        return out

    def batch_generate(
        self,
        prompts: list[list[dict]],
        *,
        poll_interval: float = 5.0,
        on_created=None,
        resume_batch_id: str | None = None,
        **sampling: Any,
    ) -> list[Completion]:
        """Same contract as :meth:`generate`, but via the **Message Batches API**.

        Submits all prompts as one asynchronous batch (~50% cheaper than live calls),
        polls until it finishes, then returns completions **in input order** (the API
        returns results unordered, keyed by ``custom_id``). Use this only for *independent*
        prompts — a multi-turn episode (guess N depends on feedback N-1) can't be one batch.

        Batches process within 24h (usually minutes); ``poll_interval`` seconds between
        status checks. Failed requests come back with ``text=""`` and ``finish_reason`` set
        to the failure type (e.g. ``"errored"``), with the error payload in ``raw``.

        **Resilience.** A batch is durable on Anthropic's side (runs server-side, results
        kept ~29 days), so a dropped connection only kills the local poller, not the batch:

        - ``on_created(batch_id)`` fires the instant the batch is created — persist the id so
          a crash mid-poll is recoverable.
        - ``resume_batch_id`` skips creation and re-attaches to an existing batch (fetches
          its results when ready) — no re-submit, so no double cost. ``prompts`` is then only
          used for its length/order expectation and may be the same list you submitted.
        - The poll + results reads are idempotent GETs, so transient network errors are
          retried with backoff instead of raising (waits out the outage, then continues).
        """
        from anthropic import APIConnectionError  # APITimeoutError is a subclass

        max_tokens = sampling.get("max_tokens", self._max_tokens)
        effort = sampling.get("effort", self._effort)

        def _resilient(call):
            """Retry an idempotent read through transient network drops (batch survives)."""
            backoff = poll_interval
            while True:
                try:
                    return call()
                except APIConnectionError:
                    time.sleep(min(backoff, 60.0))
                    backoff = min(backoff * 1.5, 60.0)

        if resume_batch_id is None:
            requests = [
                {
                    "custom_id": f"req-{i}",
                    "params": self._create_params(messages, max_tokens=max_tokens, effort=effort),
                }
                for i, messages in enumerate(prompts)
            ]
            # NOT wrapped in _resilient: a retried create could spawn a duplicate (double cost).
            batch_id = self._client.messages.batches.create(requests=requests).id
        else:
            batch_id = resume_batch_id
        if on_created is not None:
            on_created(batch_id)

        while _resilient(lambda: self._client.messages.batches.retrieve(batch_id)).processing_status != "ended":
            time.sleep(poll_interval)

        entries = _resilient(lambda: list(self._client.messages.batches.results(batch_id)))
        by_id = {entry.custom_id: entry for entry in entries}
        out: list[Completion] = []
        for i in range(len(by_id)):  # results count, so this works for both fresh and resumed runs
            result = by_id[f"req-{i}"].result
            if result.type == "succeeded":
                msg = result.message
                out.append(
                    Completion(
                        text=self._to_text(msg),
                        finish_reason=msg.stop_reason,
                        raw=msg.model_dump(),
                    )
                )
            else:  # errored | canceled | expired
                out.append(
                    Completion(text="", finish_reason=result.type, raw=result.model_dump())
                )
        return out

    def _create_params(self, messages: list[dict], *, max_tokens: int, effort: str) -> dict:
        """Build the ``messages.create`` kwargs for one prompt (shared by live + batch)."""
        system, convo = self._split_system(messages)
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": convo,
            "thinking": {"type": "adaptive", "display": self._thinking_display},
            "output_config": {"effort": effort},
        }
        if system is not None:
            # Cache the (long, repeated-across-episodes) system prompt as a prefix. The
            # structured-block form is what carries `cache_control`; a plain string can't.
            if self._cache_system:
                params["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                params["system"] = system
        return params

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Pull a leading ``system`` turn out into Anthropic's separate ``system=`` field."""
        system: str | None = None
        convo: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                # Concatenate in the unlikely case of multiple system turns.
                system = m["content"] if system is None else f"{system}\n{m['content']}"
            else:
                convo.append(m)
        return system, convo

    def _to_text(self, resp: Any) -> str:
        """Flatten Claude's content blocks into ``<think>…</think>\\n<final text>``.

        Mirrors the student's reply format so the teacher completion is a drop-in SFT
        target and parses through the agents' ``parse_action`` with no special-casing.
        If adaptive thinking chose not to think (no thinking block, or display omitted),
        we return just the final text — which still carries the ``<guess>`` tag.
        """
        thinking = "".join(
            b.thinking for b in resp.content if getattr(b, "type", None) == "thinking"
        )
        final = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        if thinking.strip():
            return f"<think>{thinking}</think>\n{final}"
        return final
