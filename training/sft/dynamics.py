"""Learning-dynamics instrumentation (method #1 from LEARNING_DYNAMICS_NOTES.md).

Two callbacks:

- **`GradUpdateNormCallback`** — every N steps, log the per-layer/component **gradient** and
  **update** (‖Δparams‖) norms of the *training* batch to wandb. Buckets: whole block `layer_NN`
  (kept identical to the wordle run for comparability) + `attn_NN` / `mlp_NN` / `norm_NN`, plus
  `embed` / `lm_head` / `final_norm` / `other`. The "where is learning happening right now" view.

- **`PerGameGradProbe`** — *which game drives which layer.* The training batch MIXES games, so its
  gradient can't be attributed to a game type. This instead periodically runs a FIXED held-out probe
  batch **per game** through forward+backward (completion-only loss, same as training) and records
  that game's per-layer/component grad norm — WITHOUT touching the real optimizer step. Writes
  `<output_dir>/grad_probe.jsonl` (one line per (step, game)) for offline deep-dives (e.g. do simple
  games hit early layers and reasoning games late?).

Both are nearly free relative to training; probe cost = (#games × one fwd+bwd on a small batch)
every `--game-probe-steps`.
"""

from __future__ import annotations

import re

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _buckets(name: str) -> list[str]:
    """Buckets a param contributes to. Per transformer block: the WHOLE block (`layer_NN`) AND each
    component — attention (`attn_NN`), MLP (`mlp_NN`), layernorms (`norm_NN`). `layer_NN` ==
    attn+mlp+norm summed, kept identical to the wordle run so the two stay comparable."""
    m = _LAYER_RE.search(name)
    if m:
        nn = f"{int(m.group(1)):02d}"
        out = [f"layer_{nn}"]                      # whole block (comparable to wordle)
        if "self_attn" in name:
            out.append(f"attn_{nn}")               # q/k/v/o (+ q_norm/k_norm)
        elif "mlp" in name:
            out.append(f"mlp_{nn}")                # gate/up/down
        else:
            out.append(f"norm_{nn}")               # input_layernorm / post_attention_layernorm
        return out
    if "embed" in name:
        return ["embed"]
    if "lm_head" in name:
        return ["lm_head"]
    if "norm" in name:
        return ["final_norm"]
    return ["other"]


def _grad_norms(model) -> dict:
    """Per-bucket L2 grad norm over the model's current `.grad` (call after a backward)."""
    grad_sq: dict = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        sq = p.grad.detach().float().pow(2).sum().item()
        for g in _buckets(n):
            grad_sq[g] = grad_sq.get(g, 0.0) + sq
    return {g: v ** 0.5 for g, v in grad_sq.items()}


def build_game_probes(repo: str, tokenizer, games: list[str], *, k: int = 8, max_len: int = 2048,
                      split: str = "test", seed: int = 0) -> dict:
    """Build a fixed completion-only probe batch per game (held-out rows), for PerGameGradProbe.

    Reuses the SFT loader's `{prompt, completion}` and masks the prompt tokens (labels=-100) so the
    probe loss matches the training objective. Returns {game: {input_ids, attention_mask, labels}}
    of CPU LongTensors (moved to the model's device at probe time).
    """
    import torch

    from training.sft.data_flat import load_flat

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    probes: dict = {}
    for g in games:
        ds = load_flat(repo, split=split, games=[g], tokenizer=tokenizer, seed=seed,
                       shuffle=True, num_proc=1)
        rows = ds.select(range(min(k, len(ds))))
        seqs = []
        for ex in rows:
            p = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            c = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                c = c + [eos_id]
            ids = (p + c)[:max_len]
            labels = ([-100] * len(p) + c)[:max_len]
            seqs.append((ids, labels))
        if not seqs:
            continue
        width = max(len(ids) for ids, _ in seqs)
        input_ids, attn, lbls = [], [], []
        for ids, labels in seqs:
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            lbls.append(labels + [-100] * pad)
        probes[g] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(lbls, dtype=torch.long),
        }
    return probes


def _build_callback():
    from transformers import TrainerCallback

    class GradUpdateNormCallback(TrainerCallback):
        def __init__(self, every: int = 50):
            self.every = max(1, every)
            self._model = None
            self._snap: dict | None = None
            self._grad_norm: dict = {}

        def on_train_begin(self, args, state, control, model=None, **kw):
            self._model = model
            return control

        def _logging_step(self, state) -> bool:
            return state.global_step > 0 and state.global_step % self.every == 0

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            m = model or self._model
            if m is None or not self._logging_step(state):
                self._snap = None
                return control
            self._grad_norm = _grad_norms(m)
            self._snap = {n: p.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
            return control

        def on_optimizer_step(self, args, state, control, model=None, **kw):
            m = model or self._model
            if m is None or self._snap is None:
                return control
            upd_sq: dict = {}
            for n, p in m.named_parameters():
                prev = self._snap.get(n)
                if prev is None:
                    continue
                sq = (p.detach() - prev).float().pow(2).sum().item()
                for g in _buckets(n):
                    upd_sq[g] = upd_sq.get(g, 0.0) + sq
            self._snap = None
            self._log(state, self._grad_norm, {g: v ** 0.5 for g, v in upd_sq.items()})
            return control

        @staticmethod
        def _log(state, gnorm: dict, unorm: dict) -> None:
            try:
                import wandb
            except ImportError:
                return
            if wandb.run is None:
                return
            d = {f"gradnorm/{g}": v for g, v in gnorm.items()}
            d.update({f"updnorm/{g}": v for g, v in unorm.items()})
            if gnorm:
                d["gradnorm/_total"] = sum(v * v for v in gnorm.values()) ** 0.5
            if unorm:
                d["updnorm/_total"] = sum(v * v for v in unorm.values()) ** 0.5
            wandb.log(d, step=state.global_step)

    return GradUpdateNormCallback


def _build_probe_callback():
    import contextlib
    import json
    import os
    import sys

    import torch
    from transformers import TrainerCallback

    class PerGameGradProbe(TrainerCallback):
        """Per-game gradient probe → `<output_dir>/grad_probe.jsonl` (does not affect training).

        Auto-exfils the JSONL so no manual step is needed: each epoch it's pushed to HF
        (`<hub_model_id>@probe/grad_probe.jsonl`) and synced to wandb. The volume keeps the file too.
        """

        def __init__(self, probes: dict, *, every: int = 500, output_dir: str = ".", bf16: bool = True,
                     hub_model_id: str | None = None, private: bool = True):
            self.probes = probes
            self.every = max(1, every)
            self.path = os.path.join(output_dir, "grad_probe.jsonl")
            self.bf16 = bf16
            self.hub_model_id = hub_model_id
            self.private = private
            self._model = None
            self._wandb_saved = False
            os.makedirs(output_dir, exist_ok=True)

        def on_train_begin(self, args, state, control, model=None, **kw):
            self._model = model
            return control

        def on_step_end(self, args, state, control, model=None, **kw):
            m = model or self._model
            if m is None or state.global_step == 0 or state.global_step % self.every != 0:
                return control
            dev = next(m.parameters()).device
            autocast = (torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
                        if self.bf16 and dev.type == "cuda" else contextlib.nullcontext())
            records = []
            for game, batch in self.probes.items():
                m.zero_grad(set_to_none=True)
                b = {k: v.to(dev) for k, v in batch.items()}
                with autocast:
                    out = m(**b)
                out.loss.backward()
                records.append({"step": state.global_step, "epoch": state.epoch,
                                "game": game, "norms": _grad_norms(m)})
            m.zero_grad(set_to_none=True)        # discard probe grads — never pollute the real step
            with open(self.path, "a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            self._wandb_sync()
            print(f"[grad-probe] step {state.global_step}: {len(records)} games → {self.path}",
                  file=sys.stderr)
            return control

        def on_save(self, args, state, control, **kw):
            self._exfil_hub()                    # push the latest JSONL to HF each epoch
            return control

        def on_train_end(self, args, state, control, **kw):
            self._exfil_hub()
            self._wandb_sync()
            return control

        def _wandb_sync(self) -> None:
            if self._wandb_saved:
                return
            try:
                import wandb
                if wandb.run is not None and os.path.exists(self.path):
                    wandb.save(self.path, policy="live")     # syncs the file (and its updates) to the run
                    self._wandb_saved = True
            except Exception as e:                            # noqa: BLE001 — never break training on telemetry
                print(f"[grad-probe] wandb sync skipped: {e}", file=sys.stderr)

        def _exfil_hub(self) -> None:
            if not self.hub_model_id or not os.path.exists(self.path):
                return
            try:
                from training.sft.upload import push_file
                push_file(self.path, self.hub_model_id, "grad_probe.jsonl",
                          revision="probe", private=self.private)
            except Exception as e:                            # noqa: BLE001 — push failure must not kill training
                print(f"[grad-probe] HF push skipped: {e}", file=sys.stderr)

    return PerGameGradProbe


def __getattr__(name: str):
    """Lazily build the callbacks on first access (defers the transformers import)."""
    if name == "GradUpdateNormCallback":
        return _build_callback()
    if name == "PerGameGradProbe":
        return _build_probe_callback()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
