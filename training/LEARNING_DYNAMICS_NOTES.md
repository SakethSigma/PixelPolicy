# Brainstorm thread — visualizing *where* learning happens in the network

Separate research thread (not wired into the trainer yet). Goal: build intuition for **where in the
network learning concentrates**, how that **moves across the 4 epochs**, and how it **differs across
the three variants** (`wordle` / `full` / `curriculum`). This doc is a menu of ideas + a recommended
starter — let's argue and prune it together.

We're in a good spot for this: the trainer saves **every epoch checkpoint to the Hub** (`epoch-N`
revisions) + the base model, so we have the full `{base, e1, e2, e3, e4}` trajectory for each
variant to analyze offline. Qwen3.5-0.8B is small (~28 layers) → all of this is cheap.

## During training vs. post-hoc — the answer is *both*

- **During training (cheap scalars → wandb):** things that need the optimizer/gradients and are
  awkward to reconstruct later. Log per-layer **gradient norm** and **update norm** (the actual
  step the optimizer took) every N steps. This is the live "where is signal flowing right now" view
  and is nearly free.
- **Post-hoc (rich, needs multiple checkpoints):** anything comparing checkpoints or needing
  activations — Δweight heatmaps, CKA, logit-lens, probing, PCA trajectories. This is where the
  cross-variant / cross-epoch contrasts live, and it's clean because we have all checkpoints.

So: a tiny during-training callback for grad/update norms + a post-hoc analysis script over the Hub
checkpoints. Don't over-instrument the training loop.

## The menu (roughly cheapest → heaviest)

### A. Weight-space "where did parameters move" — cheapest, most direct
- **Relative Δweight norm per layer**: `‖W_ckpt − W_base‖ / ‖W_base‖` for each parameter tensor
  (attn q/k/v/o, MLP gate/up/down, norms, embeddings). Post-hoc, just two state_dicts. Plot a
  **heatmap: layer (y) × {e1..e4} (x)** per variant, and a **variant-diff** heatmap. Immediately
  shows whether learning is bottom/middle/top-heavy and how the front moves across epochs.
- **Per-module-type breakdown**: are MLPs or attention moving more? Embeddings/LM-head vs blocks?
- **During-training analog**: per-parameter-group **update norm** and **grad norm** to wandb
  (histogram + scalar). Same story, live.

### B. Representation-space "where did computation change" — needs activations (free via HF)
- **CKA (Centered Kernel Alignment)** between base and each checkpoint's hidden states, per layer,
  on a fixed probe batch. Low CKA = that layer's representation changed a lot. `output_hidden_states=
  True` gives every layer's activations in one forward pass — no hooks needed. Heatmap layer × epoch,
  per variant. (CKA is the standard "did this layer's function change" metric.)
- **Logit lens / tuned lens**: project each layer's hidden state through the (final norm +) LM head
  to see *at which layer the answer forms* (e.g., the `<guess>`/`<answer>` token, or yes/no). Track
  how that **depth shifts** across epochs and variants — a great intuition builder for "where the
  skill lives."
- **Linear probes per layer**: train a cheap linear probe at each layer to predict something
  meaningful (game id, answer correctness, "is this a `<think>` region") at each checkpoint → shows
  which layers carry task information and whether it migrates with training.

### C. Forgetting / interference localization — ties to the curriculum hypotheses
- **Fisher information / gradient-importance per parameter per game**: estimate which params matter
  for each game (diagonal Fisher from a small batch). **Overlap** between games = an interference
  map — directly visualizes the capacity-competition story for a 0.8B model, and could explain a
  curriculum vs shuffle difference. Heavier but very on-theme with `CURRICULUM_NOTES.md`.
- **Per-game activation drift**: CKA(base, ckpt) computed on each game's own probe batch → which
  games reshaped the network most, and whether `sorted`/`widening`/`full` localize differently.

### D. Parameter trajectory across the run — the "movie"
- Flatten each checkpoint's Δfrom-base, run **PCA** over `{e1..e4}` (and across variants) → a 2-D
  trajectory plot showing how far/which-direction each variant travels and whether variants diverge.
- **Mode connectivity / interpolation** between two variants' final weights (loss along the line) —
  are `full` and `curriculum` in the same basin?

### E. Attention/circuit-level (heaviest, save for targeted questions)
- Attention-head entropy and "attention to the feedback tokens" in Wordle across epochs.
- Activation patching on a specific behavior (the `<think>` open, the `<guess>` tag). Use `nnsight`
  or raw hooks; TransformerLens if it supports Qwen3.5.

## Recommended starter (3 artifacts, high intuition / low effort)
1. **During training: ✅ IMPLEMENTED** — `sft/dynamics.py::GradUpdateNormCallback` logs **per-layer
   update-norm and grad-norm** to wandb every N steps (`--gradlog-steps`, default 50; wandb shows a
   line per layer → read as a layer×step heatmap). Wired into `train.py`; active with `--report-to
   wandb`. Nearly free.
2. **Post-hoc #1 — Δweight heatmap:** load base + `epoch-N` for each variant from the Hub, compute
   relative Δweight per layer/module, render layer×epoch heatmaps + variant diffs. One script, no
   activations.
3. **Post-hoc #2 — CKA + logit-lens depth:** on a fixed mixed-game probe batch, compute per-layer
   CKA(base, ckpt) and the layer at which the answer token forms; plot vs epoch, per variant.

(1)+(2) alone already answer "where, and how it moves across epochs and variants." (3) adds the
functional view. Everything heavier (Fisher interference map, PCA movie, patching) is a follow-up
once we see what (1)–(3) reveal.

## Tooling notes
- Activations: HF `output_hidden_states=True` is enough for CKA/logit-lens on a 0.8B — no extra dep.
- Weights: plain `state_dict()` diffs; `safetensors` to stream a checkpoint without full load.
- Plots: log heatmaps/images to **wandb** (keeps the cross-machine, no-git story) or save PNGs.
- Probe data: reuse `data_flat.load_flat(split="test", games=[g])` for fixed, per-game probe batches.

## Open questions for us
1. Which **probe behaviors** matter most — the `<guess>`/`<answer>` token, the `<think>` open, or
   per-game correctness? That choice drives logit-lens/probing.
2. Do we care more about **weight movement** (where params change) or **functional change** (where
   computation changes)? They can disagree (big weight move, small functional change in late layers).
3. Is the **interference map** (Fisher overlap across games) worth the cost as the headline artifact
   for the capacity-competition story, or a later deep-dive?
4. Live heatmaps in wandb vs. a post-hoc notebook over Hub checkpoints — preference?
