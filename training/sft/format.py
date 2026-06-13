"""Row → (prompt, completion), byte-identical to inference. The ONLY chat-template seam.

A distillation row (`distillation/schema.py`) stores `messages` (the conversation **prefix**,
already including the leading system turn — it is `agent.build_messages(state)`) and `completion`
(the assistant reply, with `<think>…</think>` kept for the reasoning games). We render the prompt
with the model's chat template exactly as the vLLM inference server does
(`agents/backend.py`: `chat_template_kwargs={"enable_thinking": True}` → `add_generation_prompt`),
then hand TRL a `{"prompt": str, "completion": str}` pair. With both columns present, TRL's
`SFTTrainer` masks the prompt tokens and trains on the completion only.

Keeping this in one module guarantees the train-time prompt string matches inference — there is no
second copy of the templating to drift.
"""

from __future__ import annotations

from typing import Any

# game_name → game_no (stable IDs, mirrors distillation/registry.py::GAME_NUMBERS).
GAME_NO: dict[str, int] = {
    "wordle": 0,
    "charcount": 1,
    "validity": 2,
    "anagram": 3,
    "endstart": 4,
    "rhyme": 5,
    "crossword": 6,
    "charset": 7,
    "mistakeid": 8,
    "tower": 9,
    "codebreaker": 10,
    "bullscows": 11,
    "consistency": 12,
}

# Difficulty stage per game_no, for the curriculum loader (lower = easier).
#   stage 0 — trivial single-turn lookups / classification (no reasoning)
#   stage 1 — single-turn reasoning (the <think> games + tower deduction)
#   stage 2 — multi-turn deduction (parse several turns of feedback and refine)
STAGE: dict[int, int] = {
    1: 0, 2: 0, 4: 0, 5: 0, 7: 0, 12: 0,   # charcount, validity, endstart, rhyme, charset, consistency
    3: 1, 6: 1, 8: 1, 9: 1,                 # anagram, crossword, mistakeid, tower
    0: 2, 10: 2, 11: 2,                     # wordle, codebreaker, bullscows
}

# The games whose completions carry real chain-of-thought (<think>…</think>). All four are
# Claude-distilled: anagram/crossword/mistakeid via `require_think`, and **wordle** — whose `valid`
# flag IS `has_think`, so every valid Wordle row carries a <think> block (see distillation/push.py).
# (The multi-turn deduction games codebreaker/bullscows/consistency use a TEMPLATED rationale, not
# <think>, so they are NOT here.) The curriculum loader keeps these present throughout so this
# fragile capability is never starved.
REASONING_GAMES: frozenset[str] = frozenset({"wordle", "anagram", "crossword", "mistakeid"})


def build_example(row: dict[str, Any], tokenizer) -> dict[str, str]:
    """Render one unified-schema row into a TRL prompt-completion example.

    `row["messages"]` already includes the leading system turn, so it is passed to the chat
    template as-is (do NOT re-prepend `row["system"]`). The completion is used verbatim, keeping
    any `<think>…</think>` block so the reasoning games train on their chain-of-thought.
    """
    prompt = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return {"prompt": prompt, "completion": row["completion"]}
