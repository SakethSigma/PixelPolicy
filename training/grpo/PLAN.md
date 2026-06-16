# Plan: GRPO RL training of Qwen3.5-0.8B on Wordle (and all games) via verl

## Context

We have three SFT checkpoints on the Hub (`word-games-sft-wordle`, `word-games-sft-full-v2`, and a
curriculum variant). The next research step (`_GRPO_HANDOFF.md`) is to take the **two key SFT inits**
and continue training them with **GRPO reinforcement learning** on the Wordle game, to push win-rate
above the ~2–7% SFT floor and isolate *transfer*: does broad multi-game SFT (full-v2) give a better
RL starting point than narrow wordle-only SFT?

The repo already has everything except the RL loop: a pure Wordle env, a stateless `WordleAgent`, an
episode driver (`run_episode`), per-game specs (`GAMES`), a deterministic train/val split, and a
frozen eval harness. We will **reuse all of it** and add only `training/grpo/`. The hard part is that
Wordle is **multi-turn** (up to 6 guesses with feedback) while vanilla GRPO is single-turn. Reward is
**dense shaping** (information-gain on letter discovery + format + discounted win bonus) so GRPO has
signal even at a low solve rate.

**Decisions (confirmed with user):**
- **Engine = a thin custom layer on verl** (not ROLL/RAGEN). verl is the de-facto base both of those
  build on; staying on it directly means **no framework lock-in above verl**, and model *and* algorithm
  swaps stay trivial. RAGEN is consulted only as a **reference pattern** for the per-turn rollout. We do
  **not** patch verl's core (advantage / FSDP update / KL / checkpoint); we add a rollout+data layer.
- **Model-agnostic by construction:** the only Qwen-specific thing is that Qwen3's chat template strips
  prior `<think>` — which we *embrace* (per-turn stripped samples). Nothing hardcodes Qwen; swapping to
  Gemma/Llama is a model-path + chat-template change (`build_messages` is already model-agnostic).
- Build all **three arms** (A, B, C); target hardware is a **single A100 80GB on RunPod**.
- Rollout uses **per-turn samples with stripped context** (see "Rollout representation"), each carrying a
  **purely per-turn LOCAL reward**: novel green/yellow discovery (decayed by round) + format + invalid.
  **No terminal/trajectory reward by default** — the dense discovery signal makes reward non-sparse, and
  novel-green *is* the win signal (5 greens = solved), so a separate terminal is redundant and would only
  re-introduce trajectory coupling + collapse-prone low-variance signal. `win_bonus` is kept as a **flag
  defaulting to 0.0** (safety valve if eval win-rate stalls while discovery climbs).
- Rollouts are **independent `G` episodes per target**, grouped by **`uid=(target, round)`** for the
  GRPO baseline (round-1 = an identical-prior group, the perfect baseline; round-t = same-round
  best-available baseline; drop singleton/`std=0` groups via dynamic sampling). Not a tree/beam rollout.
- **Why per-move is correct even though priors diverge past round 1:** the group mean is an
  *action-independent baseline*, so subtracting it leaves the policy gradient **unbiased** (`E_a[∇log π·b]=0`);
  differing priors only cost variance, which `(target,round)` grouping minimizes. Pure per-turn LOCAL
  reward makes each sample a contextual-bandit `(state, action, immediate reward)` → per-`(target,round)`
  GRPO is clean G-sample GRPO at each round. Loss is masked to the **completion only**; the prompt —
  **including the feedback** — never receives gradient (same as SFT's prompt/completion split).
- **Stability (anti-collapse):** GRPO **+ DAPO/StarPO-S gradient shaping** — clip-higher, token-level
  loss, dynamic sampling, KL as a flag (see "Stability"). Uncertainty/variance trajectory filtering is
  **default-off** (our dense reward keeps reward-std high — the very thing StarPO-S filtering protects);
  no replay buffer (unneeded with dense reward). Both are wired as knobs to enable only if collapse shows.

### Why per-turn-stripped (the key correctness decision)
Our whole pipeline (SFT data, `WordleAgent`, eval) **strips prior `<think>`** on replay — and Qwen3's
chat template *itself* strips reasoning from earlier assistant turns ("rolling checkpoint"). You
**cannot** strip prior think *and* keep one masked trajectory sequence: the think tokens generated at
turn *i* vanish from later context, so the only consistent place to train them is turn *i*'s own
stripped context. verl's own docs confirm delta-tokenization of a single sequence is *inaccurate* for
Qwen3 for exactly this reason. So a training sample must look **exactly like one inference step**:
`(stripped conversation so far) → (one generation)`. Hence **each turn = one GRPO sample**, built with
the same `WordleAgent.build_messages` used at inference. This is the each-move-is-a-sample design from
the turn-level agentic-RL line (MT-GRPO / RAGEN-StarPO).

## Experiment arms (one configurable trainer)

| Arm | Init checkpoint | RL task | Purpose |
|-----|-----------------|---------|---------|
| **A** | `saketh-chervu/word-games-sft-wordle` | wordle | baseline RL from narrow SFT |
| **B** | `saketh-chervu/word-games-sft-full-v2` | wordle | **key comparison** — transfer from broad SFT |
| **C** | `saketh-chervu/word-games-sft-full-v2` | all 13 games | dilution check (per-game reward norm) |

`train.py` takes `--init-repo` + `--init-revision` and `--task {wordle,all}`; the three arms are three
invocations + config overlays, not three code paths.

## Rollout representation (per-turn samples, stripped, per-turn local reward + terminal-on-final-turn)

We do **not** use verl's single-sequence Agent Loop (it can't represent stripped multi-turn without
the Qwen3 delta-tokenization bug). Instead, a **custom rollout driver** produces per-turn samples and
hands verl's GRPO update a flat batch of single-turn samples:

**Per training step:**
1. Draw `B` train target words; for each, make a **group of `G` episodes** all pinned to that word
   (`reset(word=...)` — the GRPO group shares a target so advantages are comparable). `B*G` episodes.
2. Drive every episode **turn-by-turn** with the existing `agents/rollout.py::run_episode`-style loop:
   `WordleAgent.build_messages(state, history)` (already strips `<think>`, byte-identical to inference)
   → generate via verl's rollout engine (vLLM/SGLang) → `parse_action` → `WordleEnv.step`.
3. Compute a reward **per turn-sample** (see "Credit assignment" below), not one episode scalar:
   `R_t = format_t + invalid_t + novelty_t(decayed by round)` — **purely local to turn *t*, no terminal
   by default** (`win_bonus=0.0` flag; if enabled it adds only to the winning turn). No broadcast, so an
   early turn is never blamed for a later mistake; reward-std stays high even at ~0% win rate.
4. **Explode** each episode into per-turn samples: `sample_t = (prompt_ids_t, response_ids_t, R_t)` where
   `prompt_ids_t` = the stripped context at turn *t* and `response_ids_t` = that turn's full
   `<think>…</think><guess>…</guess>` generation. Tag every sample with **`uid = target word`**.
5. Pool all per-turn samples → **standard single-turn GRPO update**: verl groups by `uid` and normalizes
   `R_t` to an advantage. Loss only on the completion (`<think>+<guess>`) tokens; KL to the SFT ref.

This makes each sample short (one turn — no 6-turn context blowup), **strip-consistent with inference**
(same `build_messages`), and in-distribution with the SFT model. Reward, env, and message-building are
reused verbatim; only the per-turn flattening + grouping is new.

### Credit assignment — purely per-turn local reward (no terminal, no broadcast)
Every turn-sample carries **only its own** reward: `format_t + invalid_t + novelty_t`, with `novelty_t`
(novel green/yellow) **decayed by round** (a discovery in round 1 is worth more than the same discovery
in round 5 — the "make progress early / win fast" pressure, applied locally). **No broadcast, no
terminal by default.** Rationale: (1) novel-**green** already encodes winning (a guess showing all 5
greens *is* the solve, and the round-discount rewards reaching it fast); (2) the dense per-turn signal
keeps **reward-std high even at ~0% win rate**, which is precisely the collapse-predictor StarPO-S works
to preserve — a near-never-firing terminal would only add a low-variance, trajectory-coupled term;
(3) round 1 is never blamed for round 5. The one pathological gap (greens revealed across turns but never
consolidated into one winning guess) is covered by the `win_bonus` flag (default **0.0**), re-enabled
only if eval win-rate stalls while discovery climbs. Grouping for the GRPO baseline is `uid=target` over
**independent `G` rollouts** (≥ `G` samples/group → std defined); the per-round baseline is approximate
(mixes states) — GiGPO `(target, round)` grouping is a noted later upgrade, **not** first-cut (tree
rollouts would narrow training to a greedy off-policy path).

### Stability — DAPO / StarPO-S gradient shaping (anti-collapse), as config flags
GRPO collapses in multi-turn via entropy collapse + declining reward-std. We keep GRPO's group-relative
advantage and turn on the DAPO stabilizers (verl config, not new algorithms): **Clip-Higher**
(`clip_ratio_low=0.2`, `clip_ratio_high=0.28` — preserves exploratory low-prob tokens, the #1 collapse
guard), **token-level loss** (`loss_agg_mode=token-mean` — removes long-`<think>` length bias),
**dynamic sampling** (drop degenerate `std=0` groups — rare for us thanks to dense reward), **truncated-
completion masking**, and **vLLM↔FSDP importance-sampling correction (TIS)** for the generation/training
logprob mismatch. **KL is a flag** (`kl_loss_coef` default light `0.001`, `use_kl_in_reward=false`):
kept as a light anchor to the SFT init (the experiment is about the *init*), with **KL-removal** (`β=0`,
DAPO/StarPO-S) as the first lever if the policy stagnates — but Clip-Higher, not KL, is the primary
collapse guard. **Uncertainty filtering** (StarPO-S top-`p%`-reward-variance prompts) is wired as
`filter_top_variance_frac` (default **1.0 = keep all**) — *not* adopted by default because our dense
reward already keeps variance healthy and 25%-filtering would discard useful data; enable only if the
**reward-std trend on wandb** (the StarPO-S early-warning signal) starts declining. **No replay buffer**
(off-policy, unneeded with dense reward); self-imitation on banked wins is a documented fallback if wins
stay near-zero. Dashboard the collapse signals: `entropy`, `reward_std`, `frac_reward_zero_std`,
`clip_ratio/high`.

### verl integration & how much we patch (answers the feasibility questions)
- **verl's optimization core is reused UNPATCHED**: GRPO advantage = group-by-`uid` normalization over a
  `DataProto`, FSDP actor update, KL-to-ref, checkpoint/push. verl already groups by a `uid`/`index`
  field and normalizes per group — exactly our per-turn-samples-share-a-target-uid case.
- **Variable turns per episode → NO shape mismatch.** Samples are **row-oriented**: a 2-turn game adds 2
  rows, a 5-turn game adds 5. Each `(prompt, response)` row is padded to the batch's max length
  independently (verl's normal ragged→padded handling for variable-length responses); a "group" is just
  the rows sharing a `uid`, of whatever size. Group std is well-defined because each target's group has
  Σ(turns over its `G` episodes) ≥ `G` rows.
- **What we ADD (not a fork):** a custom **rollout/data layer** — drive episodes turn-by-turn through
  verl's generation engine, build per-turn samples, compute `R_t`, pack a `DataProto` with `uid` +
  rewards — then hand it to verl's existing advantage+update. This is precisely RAGEN's design (verl as a
  **submodule**, decomposed into Env-State-Manager / Context-Manager / Agent-Proxy; "each turn becomes a
  training sample"). Concretely it's a subclass of verl's PPO/Ray trainer overriding the
  generate→reward→make-batch portion of `fit()`, reusing `compute_advantage` / `update_actor` / ref / save.
- **Effort & risk (honest):** ~a few hundred lines (our rollout + `R_t` already exist; the new work is the
  `DataProto` packing + trainer-loop override). The **main risk** is the exact injection seam against the
  *pinned* verl version (which `fit()` method to override; whether an assertion rejects variable group
  sizes — RAGEN itself upstreamed small fixes like a KL patch). We verify this at the `--max-steps 5`
  probe. **Fallback:** vendor RAGEN's rollout layer; last resort, a compact self-contained GRPO update.
- **Scope note:** this is the cost of the per-turn-strip choice. The cheaper-but-mismatched alternative
  (keep-think, one masked trajectory per game) is near-stock verl config; we are deliberately paying the
  custom-rollout cost to get exact train/inference parity.

## File layout — new package `training/grpo/` (mirrors `training/sft/`)

```
training/grpo/
  README.md             # isolated verl install, the 3 arms, eval flow, knobs
  train.py              # argparse entrypoint → resolve init ckpt, patch verl config, launch GRPO
  data.py               # sample_targets → verl parquet (one row per pinned target word)
  reward.py             # compute_reward (dense shaping) + verl compute_score wrapper + RewardWeights
  rollout.py            # NEW: per-turn rollout driver — drive episodes (run_episode + WordleAgent),
                        #      explode into per-turn samples, tag uid=target + episode reward
  game_loop.py          # (Arm C) generic per-turn driver over GAMES factories
  push.py               # reuse training/sft/upload.push_checkpoint for per-step HF revisions
  config/
    grpo_wordle.yaml       # Arm A base config
    grpo_wordle_full.yaml  # Arm B overlay (only init path changes)
    grpo_all.yaml          # Arm C overlay (multi-game data + per-game reward norm)
```

### What changes from the code already written (this session)
- **Replace** `wordle_agent_loop.py` (single-sequence `AgentLoopBase` + `roll_episode_tokens`) with
  `rollout.py`: a **per-turn** driver. Drop the keep-think delta-tokenization helper and its smoke test
  (it had the Qwen3 stripping bug). The new pure helper returns a **list of per-turn samples**
  (one `(prompt_ids, response_ids, reward, uid)` per turn), tested with the same fake-tokenizer style.
- **`reward.py` is refactored** from one episode scalar to **`per_turn_rewards(outcome) -> list[R_t]`**:
  purely per-turn local shaping (novelty **decayed by round** + format + invalid), **no terminal by
  default** (`win_bonus=0.0`; if >0 it adds only to the winning turn). The novelty/format/invalid math +
  `RewardWeights` are kept; what changes is the aggregation (per-turn list) and that **novelty is now
  round-decayed** and the **win term is off by default**. `compute_score` (verl signature) is retained
  for the reward-fn slot / offline sanity; the live path uses `per_turn_rewards`.
- **`config/*.yaml` also gain the DAPO knobs** (clip-higher, token-mean loss, dynamic sampling, TIS) and
  the `win_bonus`/`filter_top_variance_frac` flags (defaults 0.0 / 1.0).
- **`data.py` stays** (one row per pinned target; the rollout expands a target into `G` episodes).
- **`game_loop.py`** generalizes the per-turn driver over `GAMES` (Arm C).
- **`config/*.yaml`**: drop the single-sequence agent-loop keys. Stripping is handled by **our**
  `WordleAgent.build_messages` inside the custom rollout (not a verl `multi_turn` flag), and GRPO groups
  by `uid=target` (`adv_estimator=grpo`). Keep `use_kl_in_reward=false` + `use_kl_loss=true`.

## Reward design (`reward.py`) — dense, information-gain shaping

The user's spec, made anti-exploit: reward **novel discovery** (information gain), not raw color
count, so the model can't farm reward by repeating known-good letters.

- **Novel green** (`green_new≈0.30`): a `✓` at a position **never before** confirmed green. Re-greens
  (correctly *keeping* a pinned letter) pay **nothing** — that is the "explore, don't re-confirm" signal.
- **Novel yellow** (`yellow_new≈0.10`, < greens): a `-` letter not **previously known** to be in the
  word (`known_letters` is seeded by both greens and yellows). Re-yellows pay nothing.
- **Round decay on novelty:** the green/yellow values are multiplied by a per-round decay (e.g.
  `decay**(round-1)`, `decay≈0.9`), so a discovery in round 1 is worth more than the same discovery
  later — the "make progress / win fast" pressure, applied **locally** (no terminal needed for it).
- **Format per turn** (`format_ok≈+0.05` / `format_bad≈-0.05`): regex check that the reply has a
  `</think>` then a `<guess>` with exactly 5 letters (the Qwen3.5 template opens `<think>` in the
  prompt, so we match the closing tag + a 5-char guess, matching `agents/wordle/agent.py` semantics).
- **Invalid-guess penalty** (`invalid_guess≈-0.20`): when `RoundResult.error` is set (LENGTH/VOCAB).
- **Win bonus — OFF by default** (`win_bonus=0.0`): the safety-valve flag. If set >0 it adds
  `win_bonus·(1 − k·(round−1)/max_rounds)` to the **winning turn only** (round-discounted). Default 0
  because novel-green already encodes the win and a near-never-firing terminal hurts reward variance.

All weights live in a `RewardWeights` dataclass and are CLI-overridable for ablation.
`reward_breakdown` returns `{novel, format, invalid, win}` so wandb shows which term drives learning.
**Source of truth** stays `status=="won"`, identical to eval's solved definition (the training reward
need not contain it — eval measures the true objective).

Signatures (purely per-turn local reward; terminal off by default):
```python
def per_turn_rewards(outcome: EpisodeOutcome, weights: RewardWeights) -> list[float]
    # R_t = format_t + invalid_t + novelty_t(decayed by round).  PURELY LOCAL, no terminal by default.
    # If weights.win_bonus > 0 (default 0.0), the WINNING turn also += win_term (round-discounted).
    # novelty tracked across the episode so re-greens/re-yellows pay nothing; NO broadcast.
def reward_breakdown(outcome, weights) -> dict[str,float]   # {novel, format, invalid, win} for wandb
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float  # verl reward-fn slot
```
`EpisodeOutcome` = `{target, status, rounds_used, max_rounds, turns:[{guess, feedback, error, format_valid}]}`,
assembled by the **per-turn rollout driver** (`rollout.py`); `per_turn_rewards` returns one `R_t` per turn
(novelty tracked across the episode so re-greens/re-yellows pay nothing). The driver attaches each `R_t`
to that turn's sample. For Arm C the driver dispatches on the game: wordle → dense per-turn shaping; other
games (single-turn, one sample) → `1.0 if status==good_status else 0.0` (sparse). (`compute_score` is kept
for verl's reward-fn slot / offline sanity; the live path computes `R_t` in the rollout.)

## Per-turn rollout driver (`rollout.py`)

For each episode (pinned to `target`), drive turn-by-turn exactly like inference and **emit one sample
per turn**:

```
for turn t while state.status == "in_progress":
    messages_t = WordleAgent.build_messages(state, history)   # STRIPPED context (= inference)
    prompt_ids_t  = tokenizer.apply_chat_template(messages_t, add_generation_prompt=True, tokenize=True)
    response_ids_t = <generate via verl's rollout engine>      # this turn's <think>+<guess>
    action = WordleAgent.parse_action(decode(response_ids_t)); state = WordleEnv.step(action)
    samples.append(TurnSample(prompt_ids_t, response_ids_t))
R = reward.per_turn_rewards(episode_outcome)                  # one R_t PER turn (NOT one scalar)
for t, s in enumerate(samples): s.uid = target; s.reward = R[t]   # purely per-turn local (no terminal by default)
```

Each turn-sample is a clean single-turn `(prompt, completion)`: loss is on the completion tokens only
(verl masks the prompt structurally — same as SFT), so there is **no masked-feedback bookkeeping and no
delta-tokenization** (the bug source is gone). `prompt_ids_t` is rebuilt fresh each turn from the
stripped `build_messages`, so it is **byte-identical to what the model sees at inference turn `t`**.
Reused verbatim: `WordleAgent` (stripping replay), `WordleEnv`, `compute_feedback`/`RoundResult`.

**Guards / smoke test (CPU, fake tokenizer):** assert each turn-sample's prompt decodes to the stripped
conversation through turn `t` and its completion decodes to that turn's reply; assert all turns of an
episode share one `uid=target` but carry their **own** local `R_t`; assert an early-round discovery
scores higher than the same discovery later; assert with `win_bonus=0` (default) no turn carries a
terminal, and with `win_bonus>0` only the winning turn does. (Replaces the old single-sequence mask test.)

`game_loop.py` (Arm C) is the same driver generic over `GAMES[name]()` (`make_agent`, `make_env`,
`good_status`), selected by `data_source`; non-wordle games get the sparse solved/not-solved reward.

## Data prep (`data.py`)

`build_dataset(task, n, seed, out_path)` → parquet with verl's columns: `prompt` (the genuine
first-turn messages from `agent.system_prompt` + "Make your first guess."), `data_source` (game name),
`reward_model={"style":"rule","ground_truth":target}`, `extra_info={"target":..,"game":..}`. One row
per pinned target. Targets drawn with `spec.sample_targets(n,"train",rng)` — **only the train pool**
(~10,384 words), disjoint from the val pool (~2,588) the eval harness uses, by the sha256 `assign_pool`
split. `n≈2,000–4,000` distinct train targets is plenty; diversity per step comes from `group_size`
rollouts. Arm C emits per-game rows tagged with `data_source`.

## Environment isolation (do NOT disturb the inference vllm 0.19 pin)

verl needs its own vLLM/SGLang. Install it in a **standalone venv outside uv**, sharing only the source
tree via `PYTHONPATH=.` — the workspace `.venv` and `inference/`'s `vllm==0.19.0` are never touched.
Documented in `training/grpo/README.md`:
```bash
python -m venv /workspace/verl-venv && source /workspace/verl-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match cu128 driver first
pip install "verl[sglang]"        # verl + its own sglang/vllm/flashinfer (fallback: verl[vllm])
pip install pandas pyarrow huggingface-hub wandb python-dotenv pydantic transformers
cd /workspace/PixelPolicy
PYTHONPATH=. /workspace/verl-venv/bin/python -m training.grpo.train --config training/grpo/config/grpo_wordle.yaml ...
```
Eval still runs from the workspace `.venv` (`uv run --no-sync --package inference …`). Two venvs, one tree.

## GRPO config defaults (single A100 80GB, 0.8B) — `config/grpo_wordle.yaml`

`algorithm.adv_estimator=grpo` **+ DAPO/StarPO-S stabilizers** (see "Stability" below); KL as a *flag*
(default light `kl_loss_coef≈0.001`, `use_kl_in_reward=false`); `data.train_batch_size=64` (target
words/step); `actor.optim.lr=1e-6` (RL LR ≪ SFT 2e-5), `ppo_mini_batch_size=32`. Because samples are
**per-turn**, `max_response_length`
only needs to hold **one turn** (≈768–1024, think+guess), not the whole game — a big win over the
trajectory approach. `rollout.name=sglang`, `temperature=1.0` (train-time exploration; eval is 0.6
out-of-band), `n=8` (group size = `G` episodes per target), `gpu_memory_utilization≈0.55`, `max_turns=6`
(our custom rollout's episode cap; stripping is done by `WordleAgent.build_messages`, not a verl flag).
Rewards are the per-turn `R_t` computed in `rollout.py` and carried on the samples (verl groups by
`uid=target`). **DAPO stabilizers — verl ships a first-class `recipe/dapo/`; we base the config on it:**
`actor.clip_ratio_low=0.2`, `actor.clip_ratio_high=0.28` (clip-higher), `actor.loss_agg_mode=token-mean`,
`algorithm.filter_groups.enable=true` (drop `std=0` groups), `actor.use_kl_loss=true` + `kl_loss_coef=0.001`
(flag; set 0 to remove), TIS on. StarPO-S variance filtering left **off** (`filter_top_variance_frac=1.0`).
`trainer.total_epochs=2`,
`save_freq=25`, `logger=[console,wandb]`,
`project_name=pixelpolicy-grpo`, `n_gpus_per_node=1`. Arm B = same with init path →
`word-games-sft-full-v2`; Arm C = `train_files=all_train.parquet` + per-game sparse reward.
`train.py` resolves `--init-repo@--init-revision` via `snapshot_download` and patches `model.path` /
`ref.model.path`. Checkpoints pushed to `saketh-chervu/word-games-grpo-<arm>@step-N` via
`training/sft/upload.py::push_checkpoint` (reused as-is, weights-only).

## Compute estimate (single A100 80GB, Qwen3.5-0.8B, thinking on)

Assumptions: 0.8B decode ≈ 4k–8k tok/s on one A100 with continuous batching; avg ~4 turns/episode,
~400 generated tok/turn (thinking is verbose) → ~1,600 tok/episode.
- Per step (`batch=64 × n=8` = 512 episodes): ~0.82M gen tokens → rollout ~100–205 s; actor update
  ~30–60 s; ref pass ~15–30 s → **~2.5–5 min/step** (rollout-dominated).
- Per run (`2 epochs` over ~2,000 targets @ batch 64 ≈ 60 steps): **~3–7 h/arm** (with warmup/push/straggler headroom).
- **Three arms (A+B+C): ~12–25 h** of GPU time. Eval per checkpoint (`--n 300`) ≈ 10–20 min on the
  vllm-0.19 server; ~3 checkpoints/arm adds ~2–3 h. Levers if steps run hot: lower `max_response_length`,
  `n=4`, or (last resort) `enable_thinking=false` — each roughly halves rollout cost. VRAM is comfortable
  at 0.8B; the binding constraint is rollout latency (thinking-token count), not memory.

## Eval integration (no new code)

Reuse the frozen harness unchanged — it already accepts repo + revision:
```bash
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --model saketh-chervu/word-games-grpo-armA --revisions step-25,step-50 \
  --games wordle --n 300 --seed 0 --out eval_results_grpo/
```
`evaluate.py` uses `EVAL_SAMPLING={temperature:0.6, top_p:0.95, enable_thinking:True}`, max_tokens 4096,
on the **val** pool (disjoint from training). Then plot with `inference/analysis/viz_eval.py`.

## Critical files to create / reuse

- **Create:** all of `training/grpo/` (above).
- **Reuse verbatim:** [agents/wordle/agent.py](agents/wordle/agent.py) (`WordleAgent`, `WordleEnv`),
  [games/wordle/game.py](games/wordle/game.py) (`compute_feedback`, `RoundResult`, `LetterFeedback`,
  `Status`), [distillation/registry.py](distillation/registry.py) (`GAMES`, `GameSpec`),
  [training/sft/upload.py](training/sft/upload.py) (`push_checkpoint`),
  [inference/run_checkpoints.py](inference/run_checkpoints.py) + [inference/evaluate.py](inference/evaluate.py) (eval).
- **Mirror conventions:** [training/sft/train.py](training/sft/train.py) (argparse/wandb/hub style).

## Verification

1. **Reward unit tests** (`training/grpo/tests/`): hand-built `EpisodeOutcome`s assert novel-green pays
   once and re-green pays zero; yellow < green; invalid penalized; discounted win monotonic in round;
   a known winning game scores > a known losing game.
2. **Per-turn rollout smoke test (CPU, fake tokenizer):** drive an episode with a stub generate
   returning canned `<think>…</think><guess>…</guess>` replies; assert one sample **per turn**, each
   sample's prompt decodes to the **stripped** conversation through that turn (byte-identical to
   `WordleAgent.build_messages` at inference), its completion decodes to that turn's reply, all turns
   share one `uid=target` and one reward, and a winning episode's reward > a losing one's.
3. **Data prep:** `build_dataset("wordle", 2000, 0, …)` → assert 2000 distinct rows, every
   `ground_truth ∈ bank.train`, **zero overlap with `bank.val`**.
4. **End-to-end (RunPod A100):** install isolated verl venv; run Arm A for a handful of steps
   (`total_epochs` tiny / `max_steps` small); confirm reward breakdown logs to wandb and reward/win-rate
   trend upward; push one `step-N` revision.
5. **Eval parity:** serve a GRPO `step-N` revision and run `inference.run_checkpoints --games wordle
   --n 300 --seed 0`; confirm a `<label>.json` with accuracy + Wilson CI, comparable to SFT numbers.
6. Repeat 4–5 for Arms B and C; compare A vs B win-rate (the transfer question).
