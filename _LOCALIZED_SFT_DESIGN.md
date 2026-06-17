# Design: depth-localized SFT for forced skill reuse (composition probe)

## 1. Context & hypothesis
SFT on these word-games shows a **compositionality failure**: the model learns each skill in
isolation (standalone `validity` 85–95%, `consistency`/`codebreaker` 70–90%) but does **not invoke
them** on the hard multi-turn task — in-wordle `out_of_vocab` (~10–17%) shows *no* benefit from the
strong standalone validity skill, and `clue_violation` (~20%) persists. Flat multitask SFT also shows
no positive transfer and mild interference at 0.8B. The primitives are **present but not reused**.

**Hypothesis (H1):** in flat SFT the model must learn primitives *and* composition simultaneously,
entangled across all layers, under capacity competition. If we **force the primitive skills into the
early blocks** (they can only be supervised there) and route the **hard, multi-step tasks through the
full depth**, then the deep layers are left to learn *only the composition* — and they must read the
early, primitive-rich representation as their input → **forced reuse**. The demonstrations already
contain composition, so the deep-layer gradient gets a much easier target.

**This is the thing we are testing.** Positive result → composition is *representation-inducible* under
imitation (you don't need RL). Negative → composition is *objective-bound* (motivates the RL arm).
Either outcome is a clean, publishable claim. We do **not** assume it works.

## 2. Core mechanism — per-sample, game-keyed routing to depth
Routing is **per training sample, keyed on the sample's game** (NOT per-token, NOT learned routing).
- **Easy (shallow) games → early head `H_early` at layer `L_early`.** The sample's loss is computed at
  `H_early`; gradients therefore reach only layers `0..L_early`. (The sample need not forward deeper —
  "early exit" — and even on a full forward its loss does not depend on deep layers, so deep layers get
  zero gradient from it. autograd handles the routing; no manual surgery.)
- **Hard (multi-step) games + wordle → full head `H_full` at the final layer `L`.** Loss backprops the
  full stack `0..L`.

Consequence — exactly the intended structure:
- Layers `0..L_early` receive gradient from **both** easy (directly, via `H_early`) and hard (through the
  full path). Easy games can be learned **only** here → primitives are *localized* early.
- Layers `L_early..L` receive gradient **only** from hard games → they specialize in composition, and
  their input is the primitive-rich `h_{L_early}`. Wordle's deep computation is *forced* to build on the
  localized primitives → reuse.

## 3. Heads / architecture
Base: `Qwen/Qwen3.5-0.8B` (decoder, `L` layers). Add one auxiliary readout:
- `H_early` = `RMSNorm_early` (new, tiny) + projection to vocab. **Default: tie the vocab projection to
  the model's `lm_head`/embeddings** (shared with `H_full`), so early layers are pushed into the same
  decodable space — the only new params are `RMSNorm_early`. *Ablation:* a separate (untied) early
  `lm_head`.
- `H_full` = the model's existing final norm + `lm_head` (unchanged).
- `L_early` is a hyperparameter (default ≈ ⅓ depth; **swept** — see ablations).
Implementation = a thin wrapper over the HF model: run the decoder, capture `hidden_states[L_early]`
and the final hidden, apply the routed head per sample. Reuse `training/sft/` infra (data, optimizer,
HF push); new package `training/localized_sft/`.

## 4. Objective
Standard next-token cross-entropy on the SFT trajectory (loss masked to response tokens, as in
`training/sft`). Per sample, exactly one head is active by routing:

```
L_total = Σ_{s∈easy} CE( H_early(h_{L_early}(s)),  y_s )
        + λ · Σ_{s∈hard} CE( H_full(h_L(s)),       y_s )
```

`λ` balances the two streams (default 1.0; tune so neither dominates). Easy and hard samples are mixed
per batch in the joint regime (§6). No auxiliary/contrastive terms in v1 — keep the test clean.

## 5. Game → depth assignment
Split by reasoning depth, **not** by single/multi-turn alone. Initial assignment (a design knob, to be
validated empirically per §5.1):
- **Early / shallow (lexical, ~1 step):** `validity`, `anagram`, `charcount`, `charset`, `rhyme`,
  `endstart`.
- **Full depth / multi-step (with wordle):** `wordle`, `codebreaker`, `bullscows`, `consistency`,
  `crossword`, `tower`, `mistakeid`.

### 5.1 Principled assignment (not hand-waved)
Run a **per-game early-exit probe** first: train each game *through `H_early` alone* and measure its
accuracy. Games that reach near their full-depth accuracy via the early head are genuinely shallow →
assign **early**. Games that need the full stack → **deep**. This makes the split a measurement, and the
probe itself is a useful figure ("skill depth spectrum").

## 6. Two training regimes (both run)
**(A) Joint.** One run; batches mix easy + hard samples, each routed to its head; both loss streams
optimized concurrently. Early layers stay primitive-shaped *while* deep layers learn composition. This
is the cleanest test of H1 and the headline config.

**(B) Two-step (staged).** Phase 1: train **only easy games via `H_early`** → optimize `0..L_early`
+`H_early` to localize primitives. Phase 2: **freeze** `0..L_early` (+`H_early`) and train the hard
games (incl. wordle) full-depth → `L_early..L`+`H_full` are *forced* to compose **fixed** primitives.
*Ablation:* phase-2 with early layers unfrozen at low LR (soft localization) vs frozen (hard).

Staged tests a stronger claim ("compose *frozen* primitives"); joint tests the co-adaptation version.

## 7. Baselines & ablation grid
Baselines (matched base, compute, data exposure, eval set/sampling):
- **flat multitask SFT** = `full-v2` (no localization) — the key control.
- **wordle-only SFT** — single-task reference.

Ablations (each isolates one factor):
| run | what it isolates |
|---|---|
| flat multitask | baseline (no localization) |
| localized **joint** | the proposed mechanism |
| localized **two-step (frozen)** | composing fixed primitives |
| aux head **full-depth backprop** (no early-exit/truncation) | localization vs. *just* extra supervision |
| `L_early` sweep (≈ L/4, L/3, L/2) | where primitives must live for reuse |
| game→depth swap (move 1 borderline game) | sensitivity of the assignment |

**Controls:** identical wordle exposure across runs (don't let localized see more/less wordle than the
baseline — the confound that otherwise explains everything), identical total tokens/steps, same seed,
same held-out `val` eval set.

## 8. Validation — how we know if it worked
Win-rate alone is too coarse; the **mechanistic readout is the headline evidence.**

1. **Behavioral (primary):** wordle win-rate + the other deep games via `inference/evaluate.py`
   (`temperature 0.6, top_p 0.95, enable_thinking, n=300, seed 0`). Compare to flat multitask at matched
   exposure.
2. **Composition signature (the real test) — `inference/analysis/wordle_errors.py`:** induced reuse has
   a *specific, falsifiable* signature:
   - in-wordle `out_of_vocab` **collapses toward standalone `validity`** (the validity primitive is now
     invoked), and
   - `clue_violation` / `grey_reuse` **drop** (constraint primitives invoked).
   If win-rate moves but these don't → it isn't composition. If these drop → direct evidence the deep
   layers reuse the localized primitives.
3. **Representational probes:** linear-probe `h_{L_early}` for primitive features (is-valid-word,
   letter-presence) — present in localized but not in flat baseline? And `H_early` accuracy on the easy
   games (did primitives actually localize early?).
4. **Causal probe:** ablate / mean-patch the `h_{L_early}` primitive directions and measure wordle
   degradation — if wordle *uses* the primitives, performance should drop sharply (flat baseline should
   be insensitive).

### Success criteria
- **Positive (H1 supported):** localized (joint and/or staged) beats flat multitask on wordle **and**
  shows the composition signature (OOV↓ toward validity, clue_violation↓), at matched exposure.
- **Negative (H1 rejected):** no behavioral/mechanistic gain → composition is objective-bound →
  proceed to the RL arm (`_GRPO_HANDOFF.md`). Still a clean result.

## 9. Risks / open questions
- **Assignment validity:** the early/deep split must be earned by the §5.1 probe, not asserted.
- **`H_early` weight tying:** tied vs untied changes whether early layers are pushed into vocab space;
  ablate.
- **Forgetting in staged Phase 2:** track easy-game accuracy after Phase 2 (frozen early should prevent
  it; report it).
- **Batching efficiency:** easy samples needn't forward deep layers — split sub-batches by exit depth
  for compute, or accept one full forward for simplicity in v1.
- **Capacity:** if even localized fails, the 0.8B-capacity confound remains → the model-size point is the
  tiebreaker.

## 10. File layout (mirror `training/sft/`)
- `training/localized_sft/model.py` — HF wrapper: capture `h_{L_early}`, `H_early` head, per-sample routing.
- `training/localized_sft/train.py` — arg-parsed entrypoint; `--regime joint|staged`, `--l-early`,
  `--early-games ...`, `--lambda`, `--freeze-early`; reuse `training/sft/data_flat.py`, `upload.py`.
- `training/localized_sft/config/*.yaml` — the runs in §7.
- Reuse `inference/evaluate.py` + `inference/analysis/wordle_errors.py` for §8. Push checkpoints to
  `saketh-chervu/word-games-localized-<regime>`; eval via the `_RUNPOD_METRICS.md` flow.

## Read first
- `_GRPO_HANDOFF.md` — the sibling RL arm (the objective-level alternative to this representation-level test).
- `inference/analysis/wordle_errors.py` — the composition readout (§8.2).
- `distillation/registry.py` (`GameSpec`/`GAMES`), `agents/rollout.py`, `training/sft/` — data, episodes, infra.
