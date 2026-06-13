# Curriculum learning for the word-games SFT — research notes & brainstorm

A living doc to design the curriculum variant and reason about **catastrophic forgetting / loss of
reasoning**. The concrete strategies described here are implemented in
[`sft/data_curriculum.py`](sft/data_curriculum.py). This is meant to be argued with and edited.

## The setup we're reasoning about

- Model: `Qwen/Qwen3.5-0.8B` (small — capacity matters).
- Data: `saketh-chervu/word-games-distillation`, ~95.5k **valid** rows, 13 games, 4 epochs.
- **4 games carry real chain-of-thought** (`<think>…</think>`): `wordle`, `anagram`, `crossword`,
  `mistakeid` — together ≈ **5,266 valid rows (~5.5%)** of the valid set (wordle 2,602 + anagram 932
  + crossword 1,415 + mistakeid 317). All are Claude-distilled; **Wordle's `valid` flag literally IS
  `has_think`**, so every valid Wordle row carries a `<think>` block (regardless of win/loss). The
  multi-turn deduction games (codebreaker, bullscows, consistency) use a short **templated**
  rationale, not `<think>`, so they are *not* counted as reasoning data.
- Three difficulty stages (see `STAGE` in [`sft/format.py`](sft/format.py)):
  - **stage 0** — trivial single-turn lookups/classification: charcount, validity, endstart, rhyme,
    charset, consistency.
  - **stage 1** — single-turn reasoning: anagram, crossword, mistakeid (CoT) + tower (deduction).
  - **stage 2** — multi-turn deduction: wordle (CoT, multi-turn), codebreaker, bullscows.
  - Note the cross-cut: stage = *difficulty* (used for widening's introduction schedule), while the
    **CoT/reasoning set** (`REASONING_GAMES`) = *which carry `<think>`* (used to keep reasoning
    present throughout). Wordle is in both stage 2 **and** the reasoning set.

## The question

Will a **strict easy→hard** curriculum (all the trivial direct-answer games first) help — or will it
**erode the fragile reasoning behavior**, especially across 4 epochs where, with a fixed order, every
epoch *restarts* on the trivial data right after the model last saw reasoning?

**Short answer from the literature: the strict version is the risky one. Random shuffle is a strong
baseline. To beat shuffle without hurting reasoning, the curriculum must be *soft* (widening +
reasoning kept throughout + a little replay), not a hard global sort.**

## What the research says (cited)

**1. Curriculum in LLM SFT is weak/inconsistent; shuffle is hard to beat.**
- Wu, Dyer, Neyshabur, *When Do Curricula Work?*, ICLR 2021 — "randomly ordered samples perform as
  well or better than curricula"; benefits appear only under limited budget / noisy data.
  <https://arxiv.org/abs/2012.03107>
- Mordig et al., *On the Limits of Curriculum Learning for Post-Training LLMs*, 2025 — clean
  reasoning-difficulty setup: "no single best curriculum strategy," random is "very competitive,"
  "no significant impact" for SFT/RL. <https://openreview.net/forum?id=sHn5rq6L0O>
- Positive but small: Kim & Lee, *Strategic Data Ordering*, arXiv:2405.07490 (2024)
  <https://arxiv.org/abs/2405.07490>; CAMPUS, Findings of EMNLP 2025
  <https://aclanthology.org/2025.findings-emnlp.629.pdf> (note: naive curricula there *underperform*
  random shuffle). Foundational: Bengio et al., *Curriculum Learning*, ICML 2009.

**2. Blocked/sequential ordering increases catastrophic forgetting; reasoning is fragile.**
- Flesch et al., *Comparing continual task learning in minds and machines*, PNAS 2018 — nets suffer
  catastrophic forgetting under *blocked* training; *interleaving* fixes it.
  <https://www.pnas.org/doi/10.1073/pnas.1800755115>
- Luo et al., *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning*,
  arXiv:2308.08747 (2023) — forgetting hits **reasoning**; general instruction data alleviates it.
- Dong et al., *How Abilities in LLMs are Affected by SFT Data Composition*, ACL 2024 — "sequential
  learning of multiple abilities is prone to catastrophic forgetting"; their fix (**DMT**) =
  specialized/reasoning-first, then general **with a replay slice** of the specialized data.
  <https://openreview.net/forum?id=6M5G5hNiAU>
- Scialom et al. (**CT0**), EMNLP 2022 — **0.25–1% replay ≈ near-perfect retention**.
  <https://arxiv.org/abs/2205.12393>

**3. Small (<1B) models: multi-task interference + reasoning fragility are worst.**
- Radford et al. (Whisper), 2022 — "for small models there is negative transfer between tasks…
  models benefit more from scale." <https://arxiv.org/pdf/2212.04356>
- Fu et al., *Specializing Smaller LMs towards Multi-Step Reasoning*, ICML 2023 — over-specializing a
  small model "loses… CoT abilities." <https://arxiv.org/abs/2301.12726>

## Hypotheses to test (downstream task accuracy per checkpoint is the metric)

- **H1** — strict `sorted` easy→hard shows the largest end-of-training drop in the 4 reasoning
  games vs `full` (shuffle), because trivial data re-floods each epoch start.
- **H2** — `full` shuffle matches or beats `sorted` on aggregate accuracy.
- **H3** — `wordle`-only tops Wordle alone but loses on transfer; multi-task Wordle is capacity-bound.
- **H4** — non-reasoning-dominated final updates hurt `<think>` quality more than aggregate accuracy
  shows → measure the reasoning games separately (and watch the `<think>` format rate).
- **H5** — `widening` (reasoning kept throughout + small replay) beats both `sorted` and `full` on
  reasoning retention at equal aggregate accuracy.

## Strategies implemented (`--curriculum-strategy`)

- **`widening`** (default) — *competence-widening*: stage 0 from the start, stage 1 eligible after
  ~30%, stage 2 after ~55% (`INTRO` in `data_curriculum.py`), but once eligible a game stays in the
  shuffled mix. The 4 CoT games (incl. **wordle**) are **un-gated (present throughout)** so reasoning
  is never starved — note this pulls some wordle (stage 2, multi-turn) in early; see open question 6.
  A `replay_frac` (~3%) slice of easy+reasoning rows is spliced into the tail. Trained with a
  **SequentialSampler** (the order is the curriculum). Across 4 epochs the order repeats — so easy
  data *is* revisited each epoch, but reasoning is always interspersed (unlike `sorted`).
- **`sorted`** — strict easy→hard sort `(stage, round, completion_len)`. The cautionary baseline arm
  (H1). SequentialSampler.
- **`weighted`** — no hard order; mildly oversample harder rows by stage, then shuffle (normal
  sampler). Coarse: we have no model-loss signal at load time, so it can't "target the frontier"
  (the ideal per HS-STaR <https://arxiv.org/pdf/2505.19866>); revisit if we add an online difficulty
  signal.

Inspect any ordering without a GPU: `… data_curriculum --strategy widening --dry-run` prints the
per-bucket stage/reasoning composition.

## Recommendation (current)

Run all three planned arms (`wordle`, `full`, `curriculum=widening`). Treat **`full` shuffle as the
benchmark to beat** and **`sorted` as a cautionary arm**, not the bet. Protect reasoning: keep CoT
games interspersed (widening does), never block them to the end. Cheap extra arms worth a run:
`curriculum=sorted` (to confirm/deny H1) and a *warmup-on-easy-then-shuffle* variant if we add one.

## Open questions for us to brainstorm

1. **Enforce a reasoning *floor* by oversampling the 4 CoT games?** Today they're un-gated but still
   only ~5.5% of the stream. Oversampling 2–5× risks overfitting ~5.3k rows. Worth an arm?
2. **INTRO points (0.30 / 0.55)** and **replay_frac (0.03)** — pulled from priors, not tuned. Sweep?
3. Is `tower` (stage 1, deduction, but no `<think>`) correctly placed, or should it sit with stage 2?
4. Should multi-turn games be ordered by `round` *within* the curriculum (early turns are easier)?
   The secondary sort key already does this for `sorted`; `widening` only uses stage + randomness.
5. Do we want a **2-phase** arm (brief easy warmup → full shuffle) as the "safe curriculum"?
6. **Wordle is CoT *and* the hardest (stage 2, multi-turn).** Un-gating it (reasoning-throughout)
   pulls ~2.6k wordle rows in from position 0, which softens the difficulty-widening. Keep it
   un-gated (protect the flagship reasoning skill) or gate wordle to stage 2 and only keep the
   single-turn CoT games throughout? Toggle today with `--no-reasoning-throughout` (gates ALL of
   them) — we may want a wordle-specific switch.
