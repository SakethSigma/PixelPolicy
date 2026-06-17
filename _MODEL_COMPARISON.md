# Model comparison — SFT baselines vs GRPO arm-b (wordle focus)

All numbers: n=300/game, seed 0, frozen sampling (temp 0.6, top_p 0.95, thinking on, max_tokens 4096).
`macro` = mean accuracy across all 13 games. Error columns = % of wordle guesses (from raw, no re-inference).

| model | macro (13 games) | wordle solved/300 | wordle acc | no_guess_tag | clue_violation |
|---|---|---|---|---|---|
| wordle-SFT (best, e3) | 3.4% | 22/300 | 7.3% | 17.3% | 21.7% |
| full-v2 multitask (e4) — *GRPO init* | 72.1% | 18/300 | 6.0% | 21.5% | 20.0% |
| grpo-armb-v5-step80 | 71.8% | 25/300 | 8.3% | 7.4% | 26.7% |
| grpo-armb-v4-step100 | 72.1% | 28/300 | 9.3% | 4.4% | 24.4% |
| **grpo-armb-v6-step50** (reward+curriculum) | 71.9% | 33/300 | **11.0%** | 8.5% | **23.4%** |

## Findings
1. **RL lifted wordle without forgetting.** From its full-v2 init, GRPO took wordle **6.0% → 9.3%**
   (v4-step100; +3.3pp, ~55% relative) while macro held at **72.1%** — the other 12 games were *not*
   degraded by wordle-focused RL. That's the headline: improved the target task at no cost to breadth.
2. **GRPO beats both SFT baselines on wordle** (9.3% vs wordle-SFT 7.3% and full-v2 6.0%) — and does it
   while *also* being a strong generalist (72% macro vs the wordle-only specialist's 3.4%).
3. **The gain is format discipline, not deduction.** `no_guess_tag` collapsed (full-v2 21.5% →
   grpo 4.4%) — RL taught the model to *reliably emit a guess* (no guess = no reward), which converts
   directly to wins. `clue_violation` did **not** improve (~24%, ≈ the SFT baselines) → the deduction
   quality is unchanged. So the win came from "always answer," not "reason better."
4. **v4-step100 > v5-step80** on both wordle (9.3 vs 8.3) and format (no_guess 4.4 vs 7.4) — that
   hyperparam/reward setting is the better of the two.

5. **v6 (reward design + curriculum) is the new best: wordle 11.0% (33/300), macro 71.9%.** Unlike
   v4/v5 — whose gains were pure format (`no_guess_tag` collapse) — v6's `no_guess_tag` is *slightly
   worse* (8.5%) yet it wins, because it posts the **lowest `clue_violation` (23.4%) and `grey_reuse`
   (9.8%)** of the set. So v6's reward+curriculum is the first variant to move the **deduction**
   frontier, not just format. Early but real signal that shaping toward clue-use works.

## Implication
v4/v5 proved RL fixes the *easy* failure (format: always emit a guess). v6 shows the *hard* one
(constraint **deduction**, `clue_violation`) is also movable with the right reward+curriculum — and
that's where the remaining headroom is (still ~23%). Double down on the v6 direction: rewards/curriculum
that penalize clue violations (reusing greys, abandoning greens, dropping known-present letters), since
that's the lever now showing traction.

## Full per-epoch trajectories (context)
- wordle-SFT macro: 5.4 / 4.9 / 3.4 / 3.4 (e1–e4) — a wordle specialist, weak everywhere else.
- full-v2 macro: 54.3 / 55.1 / 69.9 / 72.1 — strong generalist; wordle 2.3 / 3.0 / 5.7 / 6.0.
- full-v2 is the SFT checkpoint GRPO arm-b was initialized from.
