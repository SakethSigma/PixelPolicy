# Distillation pipeline — how to organize the code

## Context

The goal is a **data-distillation pipeline**: use Claude as a *teacher* to generate
high-quality gameplay trajectories (starting with Wordle, then new games), convert them
into SFT training data, combine across games, and push to the HuggingFace Hub. The
student is a small open model (e.g. `Qwen/Qwen3.5-0.8B`) trained to imitate the teacher.

The repo already has the right seams for this — it just needs a new package that *drives*
the existing pieces. Key facts established during exploration:

- A teacher is just another **`LLMBackend.generate(prompts) -> list[Completion]`**
  ([agents/base.py:31-39](agents/base.py#L31-L39)). The generic rollout
  **`run_eval(pairs, generate, concurrency=)`** ([agents/rollout.py:115-136](agents/rollout.py#L115-L136))
  takes any `generate` and returns `list[Trajectory]`. So "generate teacher data" =
  inject a Claude backend into `run_eval`. Nothing in `agents/`, `games/`, or the rollout
  changes.
- **`Trajectory` / `Turn`** ([agents/base.py:45-65](agents/base.py#L45-L65)) are pure
  Pydantic records (text + game state, no tokens/logprobs) — the natural serialization
  unit. `Turn` already holds `messages`, `response`, `action`, post-action `state`.
- The dependency rule (from [agents/training_integration.md](agents/training_integration.md)):
  data/training code imports `agents/` + `games/`, **never** the reverse. `training/` is
  currently empty except for a `pyproject.toml` declaring `torch`/`transformers`/`datasets`.
- The student SFT target format is `<think>…</think>\n<guess>word</guess>`; the agent
  replays prior turns with `<think>` **stripped** ([agents/wordle/agent.py:50-81](agents/wordle/agent.py#L50-L81)).

### Decisions (from clarifying questions)

1. **New top-level `distillation/` package** — own `pyproject.toml` (no torch); keeps
   Claude-data-gen decoupled from the heavy training install, mirroring the repo's
   one-package-per-responsibility design.
2. **Teacher backend = native `AnthropicBackend` in `agents/backend.py`** (adaptive
   thinking), behind an optional `[teacher]` extra. Emits `<think>…</think>\n<guess>…</guess>`
   so teacher text is drop-in compatible with the existing agent replay/parse and the
   student's output format.
3. **SFT shape = store raw `Trajectory` JSONL, then explode to per-move, completion-masked
   samples** by replaying each move through `agent.build_messages` (rationale below).
4. **Quality gate = keep only solved episodes** (`final.status == "won"`, optionally
   `len(final.rounds) <= N`). Rejection-sampling distillation; filters on `GameState`
   directly, no reward module needed.

### Why explode-per-move rather than one packed multi-turn sequence

We train on **completions only**, on **every move**. The agent strips `<think>` from
*prior* turns on replay but the *current* move keeps it. In a single packed sequence each
assistant turn is both a loss target (needs `<think>`) and context for later turns (where
`<think>` is stripped) — impossible to satisfy at once, forcing a train/inference mismatch.
Exploding makes each sample exactly one inference-shaped call:
`prompt = build_messages(state, history[:i])` (byte-identical to inference) →
`completion = turn.response` (full `<think>…</think><guess>…</guess>`), loss masked to the
completion. Raw trajectories remain the stored source of truth, so a packed variant can be
re-derived later if think-stripping is ever dropped.

**Implementation simplification:** the per-move prompt is *already stored* — `run_episode`
saves `messages = build_messages(state, turns)` into `Turn.messages`
([agents/rollout.py:85-112](agents/rollout.py#L85-L112)). So the explode is simply, per
turn, `{"messages": turn.messages, "completion": turn.response}` — no agent re-instantiation
needed, and provably identical to inference. `dataset.py` operates directly on the raw
JSONL dicts.

---

## Execution scope (this session)

Per the user's request: **I scaffold; the user writes the implementation to learn.** Split:

- **Fully implemented by me** (these are wiring/boilerplate, not the learning target):
  - `agents/backend.py` → add the complete `AnthropicBackend` class.
  - `agents/pyproject.toml` → add the optional `[teacher]` extra (`anthropic`).
  - root `pyproject.toml` → add `distillation` workspace member + source.
  - `distillation/pyproject.toml` → full dependency/setup file (mirrors `agents/pyproject.toml`).
  - `distillation/PLAN.md` → a copy of this plan, placed in the package as requested.
  - `.env.example` → add `ANTHROPIC_API_KEY`, `TEACHER_MODEL`, `HF_HUB_REPO_ID`.
- **Scaffold only — filename + module docstring + step-by-step `# TODO` comments, NO code**
  (the user fills these in): `distillation/{__init__.py, config.py, registry.py,
  generate.py, dataset.py, push.py, run.py, README.md}`. The "Component plan" section below
  is the spec those comment blocks summarize.

Note: this is the only change in intent from the approved plan — the same files/design, but
the `distillation/*.py` bodies are left as commented stubs instead of being written out.

---

## Target layout

```
PixelPolicy/
├── agents/
│   ├── backend.py          # ADD AnthropicBackend (native SDK, adaptive thinking)
│   └── pyproject.toml       # ADD optional [teacher] extra: anthropic
└── distillation/            # NEW workspace member
    ├── pyproject.toml        # [me] anthropic, datasets, huggingface_hub, + workspace: agents, game-wordle
    ├── PLAN.md               # [me] copy of this plan, placed in the package
    ├── __init__.py           # [stub] empty package marker
    ├── README.md             # [stub] pipeline walkthrough (mirror style of agents/Readme.md)
    ├── config.py             # [stub] DistillConfig.from_env(): teacher model/effort, hub repo id, paths
    ├── registry.py           # [stub] GAMES: name -> GameSpec(make_agent, make_bank, reset_env)
    ├── generate.py           # [stub] drive teacher rollouts via run_eval -> raw Trajectory JSONL per game
    ├── dataset.py            # [stub] filter solved + explode Trajectory -> per-move SFT samples
    ├── push.py               # [stub] load per-game samples -> datasets.Dataset -> push_to_hub
    └── run.py                # [stub] CLI: `generate` / `build` / `push` subcommands
```

`[me]` = fully written this session; `[stub]` = filename + docstring + `# TODO` comments only.

`pyproject.toml` at the root ([pyproject.toml](pyproject.toml)) `[tool.uv.workspace].members`
gets `"distillation"`, and `[tool.uv.sources]` gets `distillation = { workspace = true }`.
`.env.example` gains `ANTHROPIC_API_KEY`, `TEACHER_MODEL`, `HF_HUB_REPO_ID`.

---

## Component plan

### 1. `agents/backend.py` — `AnthropicBackend` (reuse the existing seam)

Add a class with the **same `generate(self, prompts, **sampling) -> list[Completion]`**
signature as `OpenAICompatBackend` ([agents/backend.py:61-78](agents/backend.py#L61-L78)),
so it slots into `run_eval` unchanged.

- Lazy `from anthropic import Anthropic` (mirror the lazy `openai` import).
- Per prompt: split the messages into `system=` (the system turn) + `messages=` (rest),
  call `client.messages.create(model=..., thinking={"type": "adaptive"}, output_config={"effort": ...})`.
- Assemble `Completion.text` as `f"<think>{summarized_thinking}</think>\n{final_text}"` so
  the teacher reply matches the student's expected format and the existing
  `WordleAgent.parse_action` ([agents/wordle/agent.py:70-81](agents/wordle/agent.py#L70-L81))
  parses it with no change. Keep the full Claude response in `Completion.raw`.
- Model id default `claude-opus-4-8`; adaptive thinking with `display:"summarized"` to
  surface reasoning content. Add `anthropic` as an optional `[teacher]` extra in
  `agents/pyproject.toml` so normal agent runs don't pull it in.

### 2. `distillation/registry.py` — per-game wiring (the only place a new game is added)

A small registry so the core stays game-agnostic, preserving the repo's hard rule
("a new game must not require changes to … code"):

```python
GameSpec = namedtuple("GameSpec", "make_agent make_env sample_target")
GAMES = {
    "wordle": GameSpec(
        make_agent = lambda: WordleAgent(),                       # agents/wordle/agent.py
        make_env   = lambda bank, target: _reset(WordleEnv(LocalWordleClient(bank)), target),
        sample_target = lambda bank, mode: bank.sample(mode),     # games/wordle/game.py WordBank
    ),
}
```

Adding a game later = add one `GameSpec` entry (its agent already lives in `agents/<game>/`).

### 3. `distillation/generate.py` — teacher rollouts → raw Trajectory JSONL

- Build `pairs = [(spec.make_agent(), spec.make_env(bank, spec.sample_target(bank, mode)))
  for _ in range(n)]`, exactly as the eval example in
  [agents/training_integration.md:188-193](agents/training_integration.md#L188-L193).
- `trajs = run_eval(pairs, generate=AnthropicBackend(...).generate, concurrency=...)`
  ([agents/rollout.py:115](agents/rollout.py#L115)).
- Serialize each `Trajectory` with Pydantic `.model_dump()` to
  `data/raw/<game>.jsonl` (one line per episode). This is the durable, game-agnostic
  source of truth; nothing about SFT format is baked in yet.
- Note on Batch API: the teacher loop is multi-turn (each guess depends on prior
  feedback), so episodes can't be one Batch request; use `run_eval` concurrency (live
  calls). The Anthropic Batch API only fits if you later add a single-step dataset mode.

### 4. `distillation/dataset.py` — filter + explode into per-move SFT samples

- **Filter:** keep `t` where `t.final.status == "won"` (and optional
  `len(t.final.rounds) <= max_guesses`). Pure function of `GameState`.
- **Explode:** for each kept trajectory and each turn `i`, re-derive the exact inference
  prompt by calling `agent.build_messages(turn_state_before_i, history=t.turns[:i])` and
  pair it with `completion = t.turns[i].response`. Reusing the agent's own
  `build_messages` guarantees train == inference (including `<think>`-stripping of prior
  turns). Emit `{"messages": prompt, "completion": completion}` (or chat-template form
  with an assistant-mask) to `data/sft/<game>.jsonl`.
- Keep this converter generic across games (it only calls the agent's pure functions),
  so new games need no converter changes.

### 5. `distillation/push.py` — combine games + push to Hub

- Load all `data/sft/*.jsonl`, build a `datasets.Dataset` (add a `game` column),
  optional dedup + train/val split, then `dataset.push_to_hub(repo_id, ...)`.
- Reads `HF_HUB_REPO_ID` and the HF token from env (`huggingface_hub` picks up
  `HF_TOKEN`).

### 6. `distillation/run.py` — CLI

Subcommands mirroring [agents/run.py](agents/run.py): `generate --game wordle --n 500
--mode train`, `build --game wordle`, `push`. Config via `DistillConfig.from_env()`
(mirror [agents/config.py:26-40](agents/config.py#L26-L40)).

---

## Critical files

- **New:** `distillation/{pyproject.toml, config.py, registry.py, generate.py, dataset.py, push.py, run.py, README.md}`
- **Edit:** [agents/backend.py](agents/backend.py) (add `AnthropicBackend`),
  [agents/pyproject.toml](agents/pyproject.toml) (`[teacher]` extra),
  [pyproject.toml](pyproject.toml) (workspace member + source),
  `.env.example` (`ANTHROPIC_API_KEY`, `TEACHER_MODEL`, `HF_HUB_REPO_ID`).
- **Reused unchanged:** `run_eval`/`run_episode` ([agents/rollout.py](agents/rollout.py)),
  `Trajectory`/`Turn`/`Completion` ([agents/base.py](agents/base.py)),
  `WordleAgent`/`WordleEnv` ([agents/wordle/agent.py](agents/wordle/agent.py)),
  `WordBank`/`LocalWordleClient` ([games/wordle/game.py](games/wordle/game.py), client.py).

---

## Verification

1. `uv sync` resolves the new `distillation` member (and the `[teacher]` extra).
2. **Backend parity:** a tiny script feeds one hand-built Wordle prompt to
   `AnthropicBackend.generate` and asserts the returned `Completion.text` parses via
   `WordleAgent().parse_action(...)` to a 5-letter guess — proving teacher output is
   format-compatible.
3. **End-to-end smoke (small):** `python -m distillation.run generate --game wordle --n 3
   --word crane` (pin a target for determinism) → inspect `data/raw/wordle.jsonl` has 3
   trajectories with populated `turns`/`final`.
4. **Explode correctness:** `python -m distillation.run build --game wordle` → assert each
   SFT sample's `messages` equals `agent.build_messages(...)` for that move (train ==
   inference) and that only solved episodes survived.
5. **Push (dry run first):** build the `datasets.Dataset` and print
   `len(ds)`/`ds.features`/per-game counts before `push_to_hub`; then push to a throwaway
   repo and reload to confirm.
6. Optional: host the pushed dataset's teacher format against the student by running an
   existing eval (`agents.run --episodes`) to sanity-check the guesses look reasonable.
