# Handoff — learning-dynamics visualization (per-layer grad/update norms)

Context for a fresh session to **brainstorm what the per-layer learning-dynamics plots mean** and
**why the tooling is built the way it is**. Goal of the work: build intuition for *where* in the
network learning happens, how it moves across epochs, and how it differs across the three SFT recipes
(`wordle` / `full` / `curriculum`). This is exploratory intuition-building, not a rigorous study.

Model under study: **Qwen3.5-0.8B**, **24 transformer blocks** (indices 0–23), **full fine-tune**
(not LoRA), bf16, AdamW, on the word-games SFT dataset. See `training/README.md`,
`training/CURRICULUM_NOTES.md`, `training/LEARNING_DYNAMICS_NOTES.md`.

---

## 1. What's built (code map)

- **`training/sft/dynamics.py`** — two callbacks (lazy-built; module-level `_buckets` shared):
  - **`GradUpdateNormCallback`** (`--gradlog-steps`, default 50, needs `--report-to wandb`) — logs
    the *training batch's* per-bucket **gradient** (`gradnorm/*`) and **update** (`updnorm/*`) norms.
  - **`PerGameGradProbe`** (`--game-probe-steps`, default 0 = off) — *which game drives which layer.*
    Every N steps, runs a fixed held-out probe batch **per game** through fwd+bwd in isolation and
    writes that game's per-bucket grad norm to `<output_dir>/grad_probe.jsonl` (one line per
    `(step, game)`). See §3b. (`build_game_probes` builds the per-game batches.)
- **`training/analysis/viz_dynamics.py`** — standalone local viz (PEP 723 inline deps:
  wandb/pandas/matplotlib/numpy). Pulls the wandb norms and renders heatmaps + a cross-run per-layer
  line plot. `--component layer|attn|mlp|norm`. (It plots the *wandb* batch-level norms; the per-game
  `grad_probe.jsonl` is analyzed separately — see §3b.)
- Wired in `training/sft/train.py`: `--gradlog-steps`, `--game-probe-steps`, `--game-probe-k`.

---

## 2. What the numbers mean

**Two metrics per bucket, logged every N optimizer steps:**
- **`gradnorm`** = L2 norm of the bucket's **gradient** (read in `on_pre_optimizer_step`, i.e. after
  backward + grad clipping, *before* the optimizer step). Raw learning signal.
- **`updnorm`** = L2 norm of the **actual parameter change** ‖θ_after − θ_before‖ for that bucket
  (snapshot in `on_pre_optimizer_step`, diff in `on_optimizer_step`). This is what the optimizer
  *actually did* — it folds in learning rate, AdamW's per-parameter scaling, and weight decay.
  **`updnorm` is usually the more meaningful "how much did this part of the net move" signal.**

**Buckets** (a param can land in several — see `_buckets` in `dynamics.py`):
- `layer_NN` — the **whole** transformer block NN (attention + MLP + its layernorms). Kept
  byte-identical to the wordle run's logging, so `layer_NN` stays comparable across all runs.
- `attn_NN` — that block's **self-attention** (`self_attn.{q,k,v,o}_proj`, + `q_norm/k_norm`).
- `mlp_NN` — that block's **feed-forward** (`mlp.{gate,up,down}_proj`).
- `norm_NN` — that block's **layernorms** (`input_layernorm` / `post_attention_layernorm`).
  (`layer_NN` == `attn_NN` + `mlp_NN` + `norm_NN` summed in quadrature.)
- `embed`, `lm_head`, `final_norm`, `other` — non-block params.
- `_total` — whole-model norm.

**Layer index:** `layer_00` = **first** block (closest to embeddings / input); `layer_23` = **last**
block (closest to `final_norm` / `lm_head` / output). In the heatmaps, y-origin is at the bottom, so
**bottom = input side, top = output side**.

**Cadence:** logged every `--gradlog-steps` (default 50) optimizer steps. An optimizer step consumes
`per_device_batch × grad_accum × n_gpus` samples. So the number of points = total_steps / 50.

---

## 3. How to run the viz

```bash
# (local; needs WANDB_API_KEY / `wandb login`)
uv run --no-project training/analysis/viz_dynamics.py \
  --entity saketh-chervu-personal --project pixelpolicy-sft \
  --runs full curriculum-widening \
  --metric updnorm --component layer --out ./dynamics_plots
```
- `--metric updnorm|gradnorm`
- `--component layer|attn|mlp|norm` (attn/mlp/norm only exist for runs trained with the per-block
  split, i.e. `full-v2`/`curriculum` onward; the original `wordle` run has only `layer`)
- `--normalize` — per-step column-normalize the heatmap (shows the *relative* distribution across
  layers at each step, removing the overall-magnitude trend)
- Outputs: `<run>_<metric>_<component>[_norm]_heatmap.png` (layer×step) and
  `compare_<metric>_<component>_perlayer.png` (one line per run = mean over steps per layer).

---

## 3b. Per-game gradient probe — *which game drives which layer*

The headline new analysis. The **training-batch gradient mixes games** (a batch of 32 spans many
games), and a summed/mean gradient **cannot be decomposed per-game**. So to ask "do simple games
drive early layers and reasoning games drive late layers," we measure each game's gradient
**separately**: `PerGameGradProbe` periodically (every `--game-probe-steps`) runs a fixed held-out
probe batch **per game** through forward+backward (completion-only loss, same as training, in
isolation), records that game's per-bucket grad norm, then discards the grads (never touches the
real optimizer step).

**Output:** `<output_dir>/grad_probe.jsonl` (on the persistent `/workspace` volume), one line per
`(step, game)`:
```json
{"step": 1500, "epoch": 1.0, "game": "charcount",
 "norms": {"layer_00": .., "attn_00": .., "mlp_00": .., "norm_00": .., ..., "embed": .., "lm_head": ..}}
```

**Analysis (the deep-dive):** for each game, look at the per-layer profile (and attn/mlp/norm split)
and how it shifts over training. Hypothesis to test: simple lookup games (charcount, validity,
rhyme) concentrate gradient in **early** layers; reasoning games (anagram, crossword, mistakeid,
wordle) in **later** layers. Compare game profiles at matched steps; watch movement across epochs.
(There's no plotting script for this yet — load the JSONL with pandas; it's the raw data for the
deep-dive, deliberately stored rather than pre-plotted.)

**What the probe number IS (and isn't):**
- It's each game's **gradient in isolation** — "if the model right now saw only game G, where across
  layers would its loss gradient be large." A valid, standard per-task gradient signature.
- It is **not** the Adam **update** (gradient, not ‖Δθ‖ — Adam rescales), and it **ignores
  cross-game interference** (batched together, per-layer grads add/cancel; the probe sees each game
  alone). The probe batch is held-out representative rows, not the literal training rows.
- Enabled with `--game-probe-steps 250` (full/curriculum); off by default. One game at a time,
  batch `k=8` < training batch → no extra peak GPU memory, runs at `on_step_end`.

---

## 4. What we've seen so far (the one real data point)

`compare_updnorm_layer_perlayer.png` for the **`wordle`** run (mean update-norm per layer):
- Layers **0–1** moderate (~0.0205), then a **dip at layer 2–3** (layer 3 lowest, ~0.0172).
- **Rises through the middle**, peaking around **layers 12–14** (~0.026).
- Stays high but oscillates through **15–22** (~0.024–0.025), then **drops at the last layer 23**
  (~0.0216).

Shape: **early layers move least, middle/upper-middle layers move most, the very last block tapers.**

> ⚠️ **This wordle curve is from only ~2 logged points** (wordle is tiny: ~37 steps/epoch, logged
> every 50). Treat it as a *hint of a shape*, NOT a finding. Re-run wordle with `--gradlog-steps 10`,
> and lean on the `full` / `curriculum` runs (hundreds of points) for anything real.

---

## 5. Why it might look like this — hypotheses to brainstorm

(For the new session to argue/test, ideally on the dense `full`/`curriculum` runs.)

1. **Lower layers are more "done" from pretraining.** Early blocks encode general lexical/syntactic
   features that transfer; SFT changes them little → small update norms. Classic fine-tuning finding
   that later layers adapt more. The dip at layers 2–3 may be this (or noise — see caveat).
2. **Middle/upper-middle layers do the task-specific semantic work.** The Wordle skill (track
   ✓/-/x feedback, constrain candidates) likely lives in mid-stack composition → largest movement.
3. **The last block tapers because `lm_head` absorbs output adaptation.** The unembedding/`lm_head`
   (logged separately) may take the output-distribution shift, leaving block 23 less changed. **Check
   `embed`, `lm_head`, `final_norm` series directly** — the viz only plots blocks today.
4. **`updnorm` ≠ `gradnorm` shape.** AdamW normalizes per-parameter, so a layer with small raw grads
   can still take a sizable step. Compare both `--metric` views — where they disagree is interesting.
5. **Cross-variant predictions (the real payoff):**
   - `wordle` (1 task) → maybe a sharper, more concentrated peak.
   - `full` (13 games) → broader/flatter, or different layers carry shared vs game-specific skills.
   - `curriculum` → **order/forgetting signatures**: does a layer's `updnorm` stay high *late* in the
     curriculum (re-learning / interference) vs settling in `full`? Ties to the forgetting
     hypotheses H1–H5 in `CURRICULUM_NOTES.md`.
6. **attn vs mlp:** does Wordle adapt attention (routing/where-to-look over the feedback tokens) more
   than MLP (stored associations), or vice versa, and does that flip across layers?

---

## 6. Caveats & limitations (read before drawing conclusions)

- **Sparsity:** wordle has ~2 points. Any wordle "structure" is likely noise. Use dense runs.
- **`updnorm` is absolute, not relative.** It's ‖Δθ‖, not ‖Δθ‖/‖θ‖. A layer with larger weights can
  show larger ‖Δθ‖ without changing more *fractionally*. A relative-to-weight view would be a useful
  addition (we don't have it yet).
- **attn vs mlp param counts differ.** MLP (gate+up+down) has more parameters than attention
  (q+k+v+o, and GQA shrinks k/v), so raw `attn_NN` vs `mlp_NN` norms aren't directly comparable —
  normalize by param count (or per-param RMS) before concluding "MLP moves more."
- **Whole-block `layer_NN` is comparable across blocks** (uniform transformer blocks, equal param
  counts) — that comparison is fair.
- **Single process/GPU view.** Norms are per local process; multi-GPU would need reduction.
- **grad is post-clip.** `gradnorm` reflects clipped gradients (what drives the step), not raw
  pre-clip gradients.
- **attn/mlp split is new** (commit `9987c57`) — only runs trained after it have `attn_*`/`mlp_*`.
- **Multiple dead `wordle` runs** exist in wandb (debugging crashes) — filter by run ID or delete
  them, or the viz churns through empty ones.

---

## 7. Design rationale (why the tooling is the way it is)

- **Why a callback, not post-hoc?** Grad/update norms need the optimizer state mid-step; cheap to log
  live, awkward to reconstruct later. Heavier representational analyses (CKA, logit-lens, Δweight
  heatmaps) are deliberately left to post-hoc over the saved Hub checkpoints — see
  `LEARNING_DYNAMICS_NOTES.md` for that menu (only method #1, this callback, is implemented).
- **Why `on_pre_optimizer_step` + snapshot?** Grads are live there (before the step); the param
  snapshot taken there, diffed in `on_optimizer_step`, gives the true update norm. Snapshot only on
  logging steps → ~one model's worth of memory transiently, only every N steps.
- **Why log to wandb (not files)?** Keeps everything in the cross-machine, no-git story; the viz pulls
  it back. The viz lives in `training/analysis/` with PEP 723 inline deps so it runs locally with one
  `uv run` and never drags in torch.
- **Why `--gradlog-steps 50` default?** Cost/density tradeoff — fine for the big `full`/`curriculum`
  runs; too sparse for tiny `wordle` (use 10).

---

## 8. State of runs / next actions

> ⚠️ The original **`wordle`** run used the OLD callback → it has **only `layer_NN`** (no
> `attn/mlp/norm`, no `grad_probe.jsonl`). `--component attn|mlp|norm` returns "no data" for it;
> analyze it with `--component layer`. The per-block split + per-game probe land with **`full-v2`**
> and `curriculum` (run with the current code, commit `2f04ab0`+).

- `wordle` run: norm data but only ~2 points (+ several dead `wordle` runs from crashes), `layer_NN`
  only. (The dead `full` run is gone — RunPod container-disk loss; `full-v2` is the clean re-run.)
- **`full-v2`** (re-run with `--game-probe-steps 250`): will have wandb `attn/mlp/norm/layer` norms
  **and** `/workspace/runs/full-v2/grad_probe.jsonl` (per-game per-layer signatures) — the real
  dataset for the §3b deep-dive.
- **To get good data:** push `main` → on pod `git reset --hard origin/main` → run `full-v2` /
  `curriculum` (with `--game-probe-steps 250`); optionally re-run `wordle` with `--gradlog-steps 10`.
  Then viz the wandb norms with `--component {layer,attn,mlp,norm}`, and load `grad_probe.jsonl` for
  the per-game analysis.
- **Quick wins to add when brainstorming:** a plotting script for `grad_probe.jsonl` (per-game
  layer profiles, simple-vs-reasoning overlay); plot `embed`/`lm_head`/`final_norm` series; a
  relative-to-weight (‖Δθ‖/‖θ‖) option; param-count normalization for attn-vs-mlp-vs-norm.

---

## 9. Related docs
- `training/LEARNING_DYNAMICS_NOTES.md` — the full menu of where-learning-happens methods + the
  during-vs-post-hoc split (this callback is method #1, ✅ implemented).
- `training/CURRICULUM_NOTES.md` — curriculum/forgetting hypotheses (H1–H5) the cross-variant
  dynamics should illuminate.
- `training/README.md` — training harness, the three variants, how checkpoints/logging work.
