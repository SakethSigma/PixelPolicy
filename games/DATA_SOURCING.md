# Word-skill games — data sourcing & ideation

How we generate training data for the **word/deduction tasks** that broaden the
PixelPolicy word model beyond Wordle. The original family is six **single-turn** tasks (#1–#6);
three more single-turn tasks (#7 Character set, #8 Wordle mistake identification, #9 Tower
deduction) extend it, and three **multi-turn** games (#10 Codebreaker, #11 Bulls & Cows, #12
Candidate consistency — the last is single-turn) round it out. Each task teaches a foundational
sub-skill the model needs to be a strong word player:

| # | Game | Skill it teaches |
|---|------|------------------|
| 1 | Character counts | word → character/token mapping; length & vowel/consonant awareness |
| 2 | Validity + meaning | vocabulary membership; meaning recall |
| 3 | Anagrams (reasoning) | letter-multiset reasoning |
| 4 | Ends-with → starts-with | first/last character attention |
| 5 | Rhymes | phonetic / sound mapping |
| 6 | Crossword fill (reasoning) | meaning + partial-pattern → word retrieval |
| 7 | Character set | aggregate letter coverage across words (which a–z letters are in play) |
| 8 | Wordle mistake identification (reasoning) | reading Wordle feedback; spotting repeated grey/yellow mistakes |
| 9 | Tower deduction | reasoning from ✓/x feedback to the full set of consistent placements |
| 10 | Codebreaker (Mastermind, **multi-turn**) | parse multiple turns of **per-position** feedback and refine |
| 11 | Bulls & Cows (**multi-turn**) | parse multiple turns of **count** feedback and refine |
| 12 | Candidate consistency | is a candidate word still possible given a Wordle board? (clue filtering) |

This doc is about **what data we make and where the ground truth comes from**. For how it's
wired into the repo (game packages, agents, the distillation pipeline), see
[CODE_IMPLEMENTATION.md](CODE_IMPLEMENTATION.md).

> Status: **complete.** All twelve games ship as the `charcount` / `validity` / `anagram` /
> `endstart` / `rhyme` / `crossword` / `charset` / `mistakeid` / `tower` / `codebreaker` /
> `bullscows` / `consistency` packages, the **shared vocabulary + meanings assets** ship as
> `games/wordvocab/`, and the **programmatic producer** (now with a **multi-turn** generator),
> **batch distillation path**, and **unified SFT schema** are implemented in `distillation/`.
> Their data is in the combined Hub dataset alongside Wordle. **Game #4 (`endstart`) was the last
> of the original six to be built — none of the original family remains a spec.** Games #10–#12
> extend the family with two multi-turn deduction games and one clue-filtering game.

---

## Design principles

- **Offline & deterministic ground truth.** Every label is computable locally and
  reproducibly — no network in the data pipeline. Definitions/validity come from **WordNet
  (NLTK)**; rhymes come from the **`pronouncing`** library (CMU Pronouncing Dictionary).
  (Online APIs were researched and rejected; see [Sources researched](#sources-researched).)
- **The env owns the answer.** Mirroring Wordle, each game's *pure core* computes the
  correct answer and scores a submission. This is what lets the same rejection-sampling
  filter work for every word-skill game on **correctness** (see
  [Two data-production modes](#two-data-production-modes)). **Wordle itself is the exception**:
  its SFT `valid` flag is **format compliance** — whether the move carries a `<think>` block,
  re-derived in `distillation/push.py` — not whether the game was won. A well-formed reasoned
  move is a useful target even from a lost game; a move with no `<think>` is dropped regardless of
  outcome.
- **One SFT shape for everything.** Whether a sample is built programmatically or distilled
  from Claude, it ends up in the unified schema (`distillation/schema.py`) — keyed by
  `game_name`/`game_no` and carrying `messages` + `completion` that are byte-identical to what
  the student model sees at inference. The legacy Wordle distillation rows are normalized into
  this same shape, so every game shares one row format.
- **Single-turn.** Unlike Wordle, every task is one prompt → one reply. This makes the
  Anthropic **Batch API** a natural fit for the reasoning games (no lockstep needed).

---

## Two data-production modes

Both emit the same unified-schema rows (`distillation/schema.py`); they differ only in *who
writes the completion*.

**A. Programmatic (no Claude) — games 1, 2, 4, 5, 7, 9, 10, 11, 12.**
The label is cheap and exact, and no Claude chain-of-thought is wanted. A tiny "synthetic teacher"
formats the env's gold answer straight into the `<answer>`/`<meaning>`/`<guess>` completion. Zero
API cost, fully reproducible. (Game 9 Tower is programmatic even though the *student* learns a
deductive skill: the consistent placement set is exactly enumerable, so the teacher just lists it
— no reasoning trace is distilled.) **Games 10 (Codebreaker) and 11 (Bulls & Cows) are
programmatic *and* multi-turn**: an **unbiased solver** (random opening + a uniformly random code
consistent with all feedback so far) is replayed through the same game loop Wordle uses, emitting
one SFT row per turn — see the multi-turn note in
[CODE_IMPLEMENTATION.md](CODE_IMPLEMENTATION.md) and `distillation/README.md`.

> The deduction games (10 Codebreaker, 11 Bulls & Cows, 12 Consistency) prefix their `<guess>` /
> `<answer>` with a short, **programmatically-generated worded rationale** ("programmed reasoning")
> — true by construction (derived from the same feedback computation as the label and self-checked
> every row), so the dataset never teaches false reasoning. It is **templated, not Claude
> chain-of-thought**, and **not** wrapped in `<think>`, so `has_think` stays `False`. For the
> multi-turn pair, `build_messages` replays only the bare `<guess>` of prior turns, so the
> rationale is a training target without growing the context.

**B. Claude-distilled + rejection sampling — games 3, 6, 8.**
All are *reasoning* games, distilled at high adaptive-thinking effort (game 8 at `max` effort):
Claude produces `<think>…</think><answer>…</answer>`. We then **keep only the samples whose parsed
answer matches the env's ground truth**. This is the same quality gate Wordle distillation uses
(`final.status == "won"`), just with the env scoring a one-shot answer instead of a 6-round
game. Each spec additionally requires the trace to actually carry a `<think>` block
(`require_think`) — a solved trace with no reasoning is unusable as an SFT target for a reasoning
game.

> Rejection sampling matters even when the truth is trivially checkable: we are distilling
> *correct reasoning traces*, and we discard any trace that reasons its way to the wrong
> answer so the student never imitates a confident-but-wrong explanation.

---

## Shared vocabulary asset

All games draw seed words from one **multi-length** word list with **per-game** deterministic
train/val splits. The deliberate design: **a word that is val for one game is train for
another, so the model becomes familiar with every word.** This asset is **built** as
`games/wordvocab/` (committed `vocab.txt` + `assign_pool`); see its
[README](wordvocab/README.md).

- **One vocabulary.** The full pool = the Wordle vocabulary (its `train_words.txt` +
  `val_words.txt`, all 12,972 words) **unioned** with a multi-length augmentation from
  WordNet lemmas (`wn.all_lemma_names()`, filtered to lowercase-alpha single tokens of length
  **3–20**). WordNet gives length variety *and* guarantees every word has a definition (needed
  by games 2 and 6). The result is committed as `vocab.txt`, so downstream packages read it
  with **no `nltk`** at runtime (`nltk` is only the wordvocab `[build]` extra).
- **Each game splits independently — this is the whole point.** A word's pool is assigned
  *per game* by a **game-salted** hash:
  `assign_pool(game, word) = sha256(f"{game}:{word}") % 1000 < 200 → val else train`.
  So a word that is *val* for one game is almost always *train* for another. In particular,
  **Wordle's held-out val words are deliberately used in the *training* data of the
  dictionary / rhyme / crossword games.** That is the explicit goal: the model must learn the
  spelling, meaning, letters, and sound of *every* word in the Wordle vocab — including the
  ones it is never trained to *play Wordle* on.
- **Coverage guarantee.** Across ≥3 salted games the chance a word lands in val *everywhere*
  is ≈ 0.2³ ≈ 0.8% and shrinks fast, so nearly every word already appears in some game's
  train. We also **verify** the union of all games' train sets covers the full vocab and
  force any straggler into one game's train — so the guarantee is exact: **every word is
  trained in at least one game.**
- **Why this isn't leakage.** Wordle eval stays clean because we never train the
  *Wordle-playing* skill on Wordle val words. Teaching the model what those words *mean* and
  how they're *spelled / sound* is a different skill and an intended outcome — not a leak.
- **Wordle keeps its own split.** Wordle continues to use its committed unsalted artifacts
  for backward-compatible Wordle eval; only the new auxiliary games use salted per-game
  splits. Regenerating any split is a deliberate, run-on-purpose step.

---

## The games

For each: the input **X**, the output **Y** (with exact tags), the offline ground-truth
source, and the production mode. Games #1–#6 are the original family; #7–#9 extend it with more
single-turn skills, and #10–#12 add two multi-turn deduction games and a clue-filtering game.

### 1. Character counts / vowels / consonants — *programmatic, no reasoning* — ✅ **built (`charcount`)**

Teaches the model to map a word to its characters (useful for the word↔token boundary).
Shipped as the `games/charcount/` package; see its [README](charcount/README.md).

- **X:** a word (any length).
- **Y:** an `<answer>` block giving overall length, the vowels (list + count), and the
  consonants (list + count). No `<think>`. As built, letters are **space-separated and
  UPPERCASE**; `y` is a consonant.
- **Ground truth:** pure Python — iterate the characters, classify against `aeiou`.
- **Example**

  ```
  X:  Word: planet
  Y:  <answer>
      length: 6
      vowels (2): A E
      consonants (4): P L N T
      </answer>
  ```

### 2. Validity + dictionary meaning — *programmatic, no reasoning* — ✅ **built (`validity`)**

Teaches vocabulary membership and meaning recall. Shipped as the `games/validity/` package; see
its [README](validity/README.md).

- **X:** a word — either a real word (positive) or a generated pseudo-word outside the
  vocabulary (negative).
- **Y:** `<answer>valid</answer>` (+ a `<meaning>…</meaning>` block) or `<answer>invalid</answer>`.
- **Ground truth:** **WordNet membership** is the validity oracle, served at runtime from the
  committed **meanings asset** (`games/wordvocab/meanings.jsonl`, built once from WordNet) so
  there is **no `nltk` at runtime**: a word is valid iff it carries a definition, and that
  definition is the gold meaning. The meaning is checked **loosely** (non-empty).
- **Word universe:** unlike the other games, validity draws from the **Wordle vocabulary**
  (train + val union, 12,972 words) — deliberately, so the model learns the meaning/spelling of
  every Wordle word. Only the **6,627** Wordle words with a WordNet definition can be valid
  challenges. The salted split decouples from Wordle's own split, so Wordle val words enter
  validity training.
- **Negatives:** pseudo-words built by perturbing real words (swap/insert/delete a letter),
  then **confirmed absent from WordNet** (and from the Wordle vocab) so the "invalid" label is
  trustworthy. The built default is a balanced **50/50** valid/invalid mix (**13,254 rows** =
  6,627 + 6,627).
- **Example (valid / invalid)**

  ```
  X:  Word: kindle
  Y:  <answer>valid</answer>
      <meaning>a fire that has been kindled or is burning; to catch fire or cause to catch fire</meaning>

  X:  Word: brimth
  Y:  <answer>invalid</answer>
  ```

### 3. Anagrams (yes/no) — *Claude-distilled + rejection* — ✅ **built (`anagram`)**

Teaches letter-multiset reasoning. Shipped as the `games/anagram/` package; see its
[README](anagram/README.md).

- **X:** two words.
- **Y:** `<think>…</think><answer>yes</answer>` or `<think>…</think><answer>no</answer>` — a
  reasoned verdict (see below).
- **Ground truth:** `sorted(w1.lower()) == sorted(w2.lower())`.
- **Word source:** the **full multi-length vocab** (85,909 words — general words, not
  Wordle-only), salted `anagram` split.
- **Pair construction:** a **40/60 positive/negative** mix.
  - *Positives:* group the vocab by sorted-letter signature; emit pairs from groups of size
    ≥ 2 (e.g. `listen` / `silent`).
  - *Negatives:* mostly **hard** same-length near-misses (highest letter overlap that isn't an
    anagram) with some easy different-length pairs mixed in, so the model can't shortcut on
    length or letter set alone.
- **Reasoning / `require_think`:** distilled at **high adaptive-thinking effort**, the agent
  prompt asks the model to *think it through* but does **not** tell it to sort — it must work out
  the multiset comparison itself. The rejection gate keeps a trace only when it is **correct AND
  carries a `<think>` block** (a solved trace with no reasoning is unusable as an SFT target).
  This is the `GameSpec.require_think=True` flag, enforced in `distillation/batch_play.py` (the
  same flag crossword uses).
- **Built result:** a 1,000-episode batch (`--model claude-sonnet-4-6 --effort high`, seed 0)
  scored all 1,000 correct; after `require_think` (drop the 68 with no `<think>`), **932 valid
  rows** remain. Cost ≈ $1.78 (Batch API, 50% off).
- **Example**

  ```
  X:  Are 'listen' and 'silent' anagrams of each other?
  Y:  <think>listen = e,i,l,n,s,t. silent = e,i,l,n,s,t. Same multiset.</think>
      <answer>yes</answer>
  ```

### 4. Ends-with → which candidate starts with that char — *programmatic, no reasoning (MCQ)* — ✅ **built (`endstart`)**

Teaches first/last-character attention. The **last** of the original six to be built. Shipped as
the `games/endstart/` package; see its [README](endstart/README.md).

- **X:** `word1` and a list of 5 candidate words.
- **Y:** `<answer>` = the candidate whose **first** letter equals `word1`'s **last** letter.
- **Construction:** pick `word1`; choose exactly one candidate starting with its last
  letter; fill the other four with distractors that start with *different* letters (so there
  is a unique answer). Shuffle candidate order (so the correct position is unbiased).
- **Ground truth:** programmatic — `word1[-1]` vs each `cand[0]`.
- **Word source:** the shared multi-length vocab via the salted `endstart` split.
- **Built result:** the programmatic default emits **6,000 rows**, all valid by construction.
- **Example**

  ```
  X:  word1 = "mango". Which of these starts with mango's last letter?
      [river, oasis, tundra, cliff, marsh]
  Y:  <answer>oasis</answer>
  ```

### 5. Rhymes — *programmatic; both MCQ and free-generation* — ✅ **built (`rhyme`)**

Teaches sound/phonetic mapping. Shipped as the `games/rhyme/` package; see its
[README](rhyme/README.md).

- **Ground truth:** `pronouncing.rhymes(word)` (CMU dict, bundled/offline — a runtime dep of
  `game-rhyme`). A word "rhymes" iff it's in that set; options/gold answers are restricted to
  plain alphabetic words.
- **Word source:** the **full multi-length vocab** (85,909 words — general words, not
  Wordle-only), salted `rhyme` split. The built default emits **10,000 rows** = 5,000 MCQ +
  5,000 free.
- **MCQ variant**
  - **X:** a word + 5 options, exactly one of which rhymes; distractors are sampled
    non-rhyming words.
  - **Y:** `<answer>option</answer>`.
- **Free-generation variant**
  - **X:** "name a word that rhymes with `<word>`".
  - **Y:** `<answer>word</answer>` — scored by **set membership** in `rhymes(word)`, so
    multiple answers are accepted (the synthetic teacher just picks one valid rhyme to write
    the SFT completion).
- **Example (MCQ / free)**

  ```
  X:  Which word rhymes with "bright"?  [table, flight, garden, purple, ocean]
  Y:  <answer>flight</answer>

  X:  Name a word that rhymes with "bright".
  Y:  <answer>night</answer>
  ```

### 6. Crossword fill — *Claude-distilled + rejection; reasoning* — ✅ **built (`crossword`)**

Teaches meaning + partial-pattern → word retrieval (the core crossword skill). Shipped as the
`games/crossword/` package; see its [README](crossword/README.md).

- **X:** a definition (WordNet) + the word length + a **masked pattern** where about half the
  characters are revealed and the rest hidden as `_` (e.g. `c _ a _ e`). The revealed positions
  are chosen by a **word-seeded RNG**, so the mask is deterministic in the seed word — pinning a
  target reconstructs the whole clue.
- **Y:** `<think>…</think><answer>word</answer>`.
- **Ground truth:** the **seed word**. `step` scores `correct` iff `<answer>` exactly equals the
  seed word *and* is consistent with the revealed letters.
- **Word source:** seeds are words carrying a WordNet definition in the committed
  `meanings.jsonl`; `sample_targets(n)` draws **half from the Wordle vocabulary (train + val
  union) and half from general multi-length words** (the rest of the shared vocab, lengths 3–20).
  Not Wordle-only.
- **Reasoning / `require_think`:** distilled at **high adaptive-thinking effort**. The rejection
  gate keeps a distilled trace only when `<answer>` equals the target **and** the trace carries a
  `<think>` block (`GameSpec.require_think=True`).
- **Built result:** a 1,500-episode batch (`--model claude-sonnet-4-6 --effort high`) scored
  1,415 correct (1,498/1,500 had `<think>`); after `require_think`, **1,415 valid rows** remain,
  split 750 Wordle-seed + 750 general-seed. Cost ≈ $4.96 (Batch API, 50% off).
- **Example**

  ```
  X:  Definition: "a large bird with a long neck and long legs"
      Length: 5. Pattern: c _ a _ e
  Y:  <think>5 letters, c _ a _ e, a long-necked wading bird → "crane".</think>
      <answer>crane</answer>
  ```

### 7. Character set — *programmatic, no reasoning* — ✅ **built (`charset`)**

Teaches the model to aggregate letter coverage across several words — which letters of a–z are in
play (directly useful for Wordle). Shipped as the `games/charset/` package; see its
[README](charset/README.md).

- **X:** a small list of words (2–4 words).
- **Y:** an `<answer>` block giving the **used** letters (the union across the words) and the
  **unused** letters (the rest of a–z), each with a count. No `<think>`. Letters are
  **space-separated and UPPERCASE**.
- **Ground truth:** pure Python — `used` = union of letters across the words; `unused` = the
  26-letter alphabet minus `used`. **Both** sets must match to score `correct`.
- **Word source:** the shared multi-length vocab via the salted `charset` split. Each challenge
  **mixes lengths**: **one five-letter Wordle word + 1–3 non-five-letter words** (2–4 words total).
- **Built result:** the programmatic default emits **12,000 rows**, all valid by construction,
  with an even spread of 2-, 3-, and 4-word challenges.
- **Example**

  ```
  X:  Words: cat, dog
  Y:  <answer>
      used (5): A C D G O T
      unused (21): B E F H I J K L M N P Q R S U V W X Y Z
      </answer>
  ```

### 8. Wordle mistake identification — *Claude-distilled + rejection; reasoning* — ✅ **built (`mistakeid`)**

Teaches the model to read Wordle feedback and spot when a proposed guess repeats a mistake.
Shipped as the `games/mistakeid/` package; see its [README](mistakeid/README.md).

- **X:** a Wordle board (past guesses + per-letter feedback, rendered `✓`/`-`/`x`) and a
  **proposed next guess** to review.
- **Y:** `<think>…</think><answer>…</answer>` — a `mistakes: yes|no` flag and, if yes, one line
  per error: `position N, letter X, grey|yellow` (1-based positions, UPPERCASE letters).
- **Repeated mistakes:** a **grey** mistake reuses a letter already proven absent (only ever
  grey); a **yellow** mistake re-places a letter in a slot already shown yellow for it. Only these
  two count.
- **Ground truth:** computed from the board feedback **alone — no target word needed**. The
  reported error set + flag must exactly match the truth.
- **Challenge source:** extracted from the **original Wordle teacher trajectories** into the
  committed `games/mistakeid/challenges.jsonl` (built by `games/mistakeid/build_challenges.py`):
  165 mistake boards + 1,498 clean boards. `sample_targets` returns a 50/50 mix, so a balanced set
  maxes at 165 + 165 = 330.
- **Reasoning / `require_think`:** distilled at **`max`** adaptive-thinking effort (note: `xhigh`
  is **not** a supported level for `claude-sonnet-4-6` — valid levels are low/medium/high/max).
  The gate keeps a trace only when it is **correct AND carries a `<think>` block**
  (`GameSpec.require_think=True`).
- **Built result:** a 330-episode batch (165 mistake + 165 clean, `--model claude-sonnet-4-6
  --effort max`) yielded **317 valid rows** (157 mistake + 160 clean) after `require_think` dropped
  the 13 wrong traces. Cost ≈ $2.78 (Batch API, 50% off).
- **Example**

  ```
  X:  Board: CRANE  x x ✓ x x
      Proposed guess: TRACE
  Y:  <think>N was grey (absent); TRACE doesn't reuse it. A is green in slot 3...</think>
      <answer>
      mistakes: no
      </answer>
  ```

### 9. Tower deduction — *programmatic, no reasoning* — ✅ **built (`tower`)**

Teaches the model to **reason from Wordle-style ✓/x feedback** to the complete set of consistent
states — the same deductive skill Wordle rewards, isolated into one self-contained challenge.
Shipped as the `games/tower/` package; see its [README](tower/README.md).

- **X:** a tower of **3 floors** (1 = bottom, 3 = top), each with two rooms (Left / Right). Three
  named people each occupy a different room, **one per floor** (a bijection). A *proposed*
  placement is shown, and per person two ✓/x flags — is their **floor** correct, is their **room**
  correct.
- **Y:** an `<answer>` block listing **every** consistent placement as numbered `solution N:`
  blocks, one person per line `Name: floor N, Left/Right`. No `<think>`.
- **Ground truth:** pure Python — enumerate the 6 floor permutations, keep those whose ✓/x pattern
  matches, and flip each mismatched room. The set of listed placements must **exactly** equal the
  consistent set to score `correct`.
- **Provable answer size — always 1 or 2.** Rooms never branch (each is one of two; a wrong-room
  flag means the other room). Floors fix the bijection up to derangements: if **any** floor flag
  is ✓ there is exactly **1** solution; if **all three** floors are wrong there are exactly **2**
  (the two derangements of 3). Never more than 2.
- **Variety from names, not logic.** The whole distinct logic space is only **1,920** structures,
  so surface variety comes from a pool of 60 random first names rather than harder puzzles.
- **Built result:** the programmatic default emits **5,000 rows**, all valid by construction:
  ~3,343 single-solution + ~1,657 two-solution (the ~1/3 all-floors-wrong derangement rate).
- **Example**

  ```
  X:  Alice — guess: floor 2, Left   -> floor x, room ✓
      Bob   — guess: floor 1, Right  -> floor ✓, room x
      Carol — guess: floor 3, Left   -> floor x, room x
  Y:  <answer>
      solution 1:
      Alice: floor 3, Left
      Bob: floor 1, Left
      Carol: floor 2, Right
      </answer>
  ```

### 10. Codebreaker (Mastermind) — *programmatic, **multi-turn**, no reasoning* — ✅ **built (`codebreaker`)**

Teaches the **core Wordle loop** — parse several turns of per-position feedback and adjust — on a
non-vocabulary symbol space, so the deduction skill is decoupled from word knowledge. Shipped as
the `games/codebreaker/` package; see its [README](codebreaker/README.md).

- **X (per turn):** the conversation so far — prior guesses and their ✓/-/x feedback.
- **Y (per turn):** a short, true **worded rationale** recapping the deductions, then
  `<guess>CODE</guess>` (4 symbols A–F). **Not** chain-of-thought — the rationale is
  programmatically templated, **not** wrapped in `<think>`, so `has_think` stays `False`.
- **Rules:** secret = 4 slots over 6 symbols (`A`–`F`), **repeats allowed**; up to 12 rounds.
  Per-slot feedback ✓/-/x uses **Wordle's exact two-pass duplicate rule** (`compute_feedback`).
- **Ground truth / teacher:** the env scores each guess; the SFT teacher is `CodebreakerSolver`,
  a deliberately **unbiased** solver — random opening, then a **uniformly random** code among
  those still consistent with all feedback (no fixed/ordered opening, no symbol-order bias). A
  deterministic solver would teach a biased policy. The teacher derives the worded rationale
  (`_reason`) from the same feedback/candidate-set it draws the guess from, so it is **always
  true**; every row is self-checked (the episode must reach `won`). On replay, `build_messages`
  keeps only the bare `<guess>` of prior turns (the rationale is dropped, like Wordle strips
  `<think>`), so prompt + completion stays ~400 tokens.
- **Built result:** capped at **10,000 rows** (≈2,726 episodes, ~3.7 turns/episode) via
  `distillation.programmatic --game codebreaker --episodes 5000 --max-rows 10000`.
- **Example (one turn)**

  ```
  X:  AAEF -> ✓-xx      (so far)
  Y:  Clues so far — fixed: slot 1=A; in the code but misplaced: none; not in the code: B, D. 64 codes still fit; AAEF is one of them, so I'll try it.
      <guess>AAEF</guess>
  ```

### 11. Bulls & Cows — *programmatic, **multi-turn**, no reasoning* — ✅ **built (`bullscows`)**

Same core loop as Codebreaker, but the feedback is a different **representation**: aggregate
**counts** rather than per-position tiles, so the deduction skill is decoupled from positional
cues. Shipped as the `games/bullscows/` package; see its [README](bullscows/README.md).

- **X (per turn):** the conversation so far — prior guesses and their bulls/cows counts.
- **Y (per turn):** a short, true **worded rationale** recapping the bull/cow clues, then
  `<guess>NNNN</guess>` (4 distinct digits). **Not** chain-of-thought — the rationale is
  programmatically templated, **not** wrapped in `<think>`, so `has_think` stays `False`.
- **Rules:** secret = **4 distinct digits** (0–9); up to 10 rounds. Feedback = `bulls` (right
  digit, right place) + `cows` (right digit, wrong place).
- **Ground truth / teacher:** `BullsCowsSolver`, the same **unbiased** random-consistent solver
  (random opening + a uniformly random code consistent with all bulls/cows so far). The teacher
  derives the worded rationale (`_reason`) from the same counts/candidate-set it draws the guess
  from, so it is **always true**; every row is self-checked (the episode must reach `won`). On
  replay, `build_messages` keeps only the bare `<guess>` of prior turns (the rationale is dropped,
  like Wordle strips `<think>`), so prompt + completion stays ~400 tokens.
- **Built result:** capped at **10,000 rows** (≈1,823 episodes, ~5.5 turns/episode) via
  `distillation.programmatic --game bullscows --max-rows 10000`.
- **Example (one turn)**

  ```
  X:  0932 -> bulls: 0, cows: 1      (so far)
  Y:  From the clues so far (0932 → 0 bulls, 1 cow), 1440 numbers still fit every count; 2158 is one of them, so I'll try it.
      <guess>2158</guess>
  ```

### 12. Candidate consistency — *programmatic, no reasoning (yes/no)* — ✅ **built (`consistency`)**

Teaches the **positive** side of feedback reasoning: given a Wordle board, decide whether a
candidate word is **still possible**. It is the positive-selection complement to `mistakeid`
(which *locates* a guess's errors). Shipped as the `games/consistency/` package; see its
[README](consistency/README.md).

- **X:** a Wordle board (1–3 past guesses + ✓/-/x feedback) and a **candidate** word.
- **Y:** a short, true **worded rationale** of the per-clue check, then `<answer>yes</answer>` /
  `<answer>no</answer>` — is the candidate consistent with every clue? **Not** chain-of-thought —
  the rationale is programmatically templated, **not** wrapped in `<think>`, so `has_think` stays
  `False`.
- **Ground truth:** **reuses Wordle's scorer** (dep on `game-wordle`) — consistent iff
  `compute_feedback(guess, candidate) == feedback` for every row. The rationale
  (`consistency/render.py::render_reasoning`) recomputes each clue from the same `compute_feedback`
  and, on the first failing clue, pinpoints the conflict, so it is **always true**; every row is
  self-checked (its parsed verdict must score `correct`).
- **Balance / size:** balanced **50/50** yes/no; each challenge kept **under 4k tokens** (max
  ~250 tokens).
- **Built result:** the programmatic default emits **10,000 rows** (5,000 yes + 5,000 no), all
  valid by construction.
- **Example**

  ```
  X:  Board: CRANE  x x ✓ ✓ x
      Candidate: DOGGY
  Y:  If the word were DOGGY, guessing CRANE would score x x x x x, but the clue shows x x ✓ ✓ x: position 3 must be A, but DOGGY has G there. So DOGGY is ruled out.
      <answer>no</answer>
  ```

---

## Summary table

| # | Game | Format | Reasoning? | Offline source | Producer |
|---|------|--------|:----------:|----------------|----------|
| 1 | Char counts ✅ | free `<answer>` | no | Python (char classify) | programmatic |
| 2 | Validity + meaning ✅ | `<answer>` (+`<meaning>`) | no | WordNet (committed `meanings.jsonl`) | programmatic |
| 3 | Anagrams ✅ | `<think>`+`<answer>` (`require_think`) | yes | `sorted()` check | Claude + rejection |
| 4 | Ends→starts ✅ | MCQ `<answer>` | no | Python (char match) | programmatic |
| 5 | Rhymes ✅ | MCQ + free | no | `pronouncing` (CMU) | programmatic |
| 6 | Crossword ✅ | `<think>`+`<answer>` (`require_think`) | yes | WordNet defs + seed word | Claude + rejection |
| 7 | Character set ✅ | free `<answer>` (used/unused) | no | Python (letter union) | programmatic |
| 8 | Mistake identification ✅ | `<think>`+`<answer>` (`require_think`) | yes | Python (board feedback) | Claude + rejection |
| 9 | Tower deduction ✅ | free `<answer>` (numbered solutions) | no | Python (permutation enumeration) | programmatic |
| 10 | Codebreaker ✅ | worded rationale + `<guess>` per turn (**multi-turn**) | no (templated, no `<think>`) | Python (`compute_feedback`) + unbiased solver | programmatic (multi-turn) |
| 11 | Bulls & Cows ✅ | worded rationale + `<guess>` per turn (**multi-turn**) | no (templated, no `<think>`) | Python (bulls/cows counts) + unbiased solver | programmatic (multi-turn) |
| 12 | Candidate consistency ✅ | worded rationale + `<answer>` yes/no | no (templated, no `<think>`) | Python (reuses Wordle scorer) | programmatic |

---

## Sources researched

We evaluated both offline libraries and free online APIs and chose **offline** everywhere
for determinism, zero rate limits, and a network-free pipeline.

**Chosen (offline):**
- **WordNet via NLTK** — lexical database, ~155k words, ships as a downloadable corpus
  (`nltk.download('wordnet')`, `nltk.download('omw-1.4')`), then fully offline.
  `wn.synsets(w)` → senses; `.definition()` → gloss. Used for games 2 and 6, and to source
  the multi-length vocabulary. <https://www.nltk.org/howto/wordnet.html>
- **`pronouncing`** — thin Python wrapper over the CMU Pronouncing Dictionary, no external
  deps, offline after install (`pip install pronouncing`). `pronouncing.rhymes(word)` →
  rhyme set; `pronouncing.search(regex)` → pattern match. Used for game 5.
  <https://pronouncing.readthedocs.io/> · <https://pypi.org/project/pronouncing/>

**Considered and rejected (online), kept here as fallbacks:**
- **Free Dictionary API** (`dictionaryapi.dev`) — no key, JSON definitions/phonetics/
  synonyms at `/api/v2/entries/en/{word}`. Rejected: network-dependent, informal rate
  limits, non-deterministic availability. <https://dictionaryapi.dev/>
- **Datamuse API** — free up to 100k req/day, no key (until 2027). Useful params:
  `rel_rhy=` (rhymes), `sp=` (spelling wildcards `?`/`*`), `md=d|p|f` (definitions from
  Wiktionary/WordNet, parts of speech, frequency), `max=` (≤1000). A strong single-source
  option for rhymes + definitions + pattern search, but online. Kept as an enrichment
  fallback for words the offline sources miss. <https://www.datamuse.com/api/>

Sources:
- [NLTK WordNet howto](https://www.nltk.org/howto/wordnet.html)
- [pronouncing (PyPI)](https://pypi.org/project/pronouncing/) · [docs](https://pronouncing.readthedocs.io/en/latest/)
- [Free Dictionary API](https://dictionaryapi.dev/)
- [Datamuse API](https://www.datamuse.com/api/)
