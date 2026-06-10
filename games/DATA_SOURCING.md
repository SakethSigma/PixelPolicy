# Word-skill games — data sourcing & ideation

How we generate training data for six new **single-turn word tasks** that broaden the
PixelPolicy word model beyond Wordle. Each task teaches a foundational sub-skill the model
needs to be a strong word player:

| # | Game | Skill it teaches |
|---|------|------------------|
| 1 | Character counts | word → character/token mapping; length & vowel/consonant awareness |
| 2 | Validity + meaning | vocabulary membership; meaning recall |
| 3 | Anagrams (reasoning) | letter-multiset reasoning |
| 4 | Ends-with → starts-with | first/last character attention |
| 5 | Rhymes | phonetic / sound mapping |
| 6 | Crossword fill (reasoning) | meaning + partial-pattern → word retrieval |

This doc is about **what data we make and where the ground truth comes from**. For how it's
wired into the repo (game packages, agents, the distillation pipeline), see
[CODE_IMPLEMENTATION.md](CODE_IMPLEMENTATION.md).

---

## Design principles

- **Offline & deterministic ground truth.** Every label is computable locally and
  reproducibly — no network in the data pipeline. Definitions/validity come from **WordNet
  (NLTK)**; rhymes come from the **`pronouncing`** library (CMU Pronouncing Dictionary).
  (Online APIs were researched and rejected; see [Sources researched](#sources-researched).)
- **The env owns the answer.** Mirroring Wordle, each game's *pure core* computes the
  correct answer and scores a submission. This is what lets the same rejection-sampling
  filter work for every game (see [Two data-production modes](#two-data-production-modes)).
- **One SFT shape for everything.** Whether a sample is built programmatically or distilled
  from Claude, it ends up as `{game, messages, completion}` — byte-identical to what the
  student model sees at inference. Identical to the existing Wordle distillation output.
- **Single-turn.** Unlike Wordle, every task is one prompt → one reply. This makes the
  Anthropic **Batch API** a natural fit for the reasoning games (no lockstep needed).

---

## Two data-production modes

Both emit the same `{game, messages, completion}` rows; they differ only in *who writes the
completion*.

**A. Programmatic (no Claude) — games 1, 2, 4, 5.**
The label is cheap and exact, and no chain-of-thought is wanted. A tiny "synthetic teacher"
formats the env's gold answer straight into the `<answer>`/`<meaning>` completion. Zero API
cost, fully reproducible.

**B. Claude-distilled + rejection sampling — games 3, 6.**
Reasoning is wanted, so Claude produces `<think>…</think><answer>…</answer>`. We then
**keep only the samples whose parsed answer matches the env's programmatic ground truth**.
This is the same quality gate Wordle distillation uses (`final.status == "won"`), just with
the env scoring a one-shot answer instead of a 6-round game.

> Rejection sampling matters even when the truth is trivially checkable: we are distilling
> *correct reasoning traces*, and we discard any trace that reasons its way to the wrong
> answer so the student never imitates a confident-but-wrong explanation.

---

## Shared vocabulary asset

All games draw seed words from one **multi-length** word list with **per-game** deterministic
train/val splits. The deliberate design: **a word that is val for one game is train for
another, so the model becomes familiar with every word.**

- **One vocabulary.** The full pool = the Wordle vocabulary (its `train_words.txt` +
  `val_words.txt`, all 12,972 words) **unioned** with a multi-length augmentation from
  WordNet lemmas (`wn.all_lemma_names()`, filtered to lowercase-alpha single tokens in a
  sensible length range). WordNet gives length variety *and* guarantees every word has a
  definition (needed by games 2 and 6).
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

## The six games

For each: the input **X**, the output **Y** (with exact tags), the offline ground-truth
source, and the production mode.

### 1. Character counts / vowels / consonants — *programmatic, no reasoning*

Teaches the model to map a word to its characters (useful for the word↔token boundary).

- **X:** a word (any length).
- **Y:** an `<answer>` block giving overall length, the vowels (list + count), and the
  consonants (list + count). No `<think>`.
- **Ground truth:** pure Python — iterate the characters, classify against `aeiou`.
- **Example**

  ```
  X:  Word: planet
  Y:  <answer>
      length: 6
      vowels (2): a, e
      consonants (4): p, l, n, t
      </answer>
  ```

### 2. Validity + dictionary meaning — *programmatic, no reasoning*

Teaches vocabulary membership and meaning recall.

- **X:** a word — either a real word from the vocab (positive) or a generated pseudo-word
  outside the vocabulary (negative).
- **Y:** `<answer>valid</answer>` or `<answer>invalid</answer>`; if valid, also a
  `<meaning>…</meaning>` block.
- **Ground truth:** **WordNet membership** is the validity oracle (`bool(wn.synsets(w))`);
  the meaning is `wn.synsets(w)[0].definition()` (optionally include the part of speech and
  a second sense).
- **Negatives:** pseudo-words built by perturbing real words (swap/insert/delete a letter)
  or by sampling letter strings, then **confirmed absent from WordNet** (and from the Wordle
  vocab) so the "invalid" label is trustworthy.
- **Example (valid / invalid)**

  ```
  X:  Word: kindle
  Y:  <answer>valid</answer>
      <meaning>a fire that has been kindled or is burning; to catch fire or cause to catch fire</meaning>

  X:  Word: brimth
  Y:  <answer>invalid</answer>
  ```

### 3. Anagrams (yes/no) — *Claude-distilled + rejection, reasoning required*

Teaches letter-multiset reasoning.

- **X:** two words.
- **Y:** `<think>…</think><answer>yes</answer>` or `<answer>no</answer>`.
- **Ground truth:** `sorted(w1.lower()) == sorted(w2.lower())`.
- **Pair construction:**
  - *Positives:* group the vocab by sorted-letter signature; emit pairs from groups of size
    ≥ 2 (e.g. `listen` / `silent`).
  - *Hard negatives:* near-misses — same letters but one swapped/added/removed, or same
    length and high letter overlap — so the model can't shortcut on length or letter set
    alone. Mix in some easy negatives (different length) for balance.
- **Rejection:** keep a distilled trace only when its `<answer>` equals the sort-check
  result.
- **Example**

  ```
  X:  Are 'listen' and 'silent' anagrams?
  Y:  <think>Sort listen -> eilnst. Sort silent -> eilnst. Same multiset.</think>
      <answer>yes</answer>
  ```

### 4. Ends-with → which candidate starts with that char — *programmatic, no reasoning (MCQ)*

Teaches first/last-character attention.

- **X:** `word1` and a list of 5 candidate words.
- **Y:** `<answer>` = the candidate whose **first** letter equals `word1`'s **last** letter.
- **Construction:** pick `word1`; choose exactly one candidate starting with its last
  letter; fill the other four with distractors that start with *different* letters (so there
  is a unique answer). Shuffle candidate order.
- **Ground truth:** programmatic — `word1[-1]` vs each `cand[0]`.
- **Example**

  ```
  X:  word1 = "mango". Which of these starts with mango's last letter?
      [river, oasis, tundra, cliff, marsh]
  Y:  <answer>oasis</answer>
  ```

### 5. Rhymes — *programmatic; both MCQ and free-generation*

Teaches sound/phonetic mapping.

- **Ground truth:** `pronouncing.rhymes(word)` (CMU dict). A word "rhymes" iff it's in that
  set.
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

### 6. Crossword fill — *Claude-distilled + rejection; both MCQ and free-gen; reasoning preferred*

Teaches meaning + partial-pattern → word retrieval (the core crossword skill).

- **X:** a definition (WordNet) + the word length + a **masked pattern** where 20–100% of
  characters are hidden (e.g. `c _ a _ e`). Higher mask % = harder.
- **Y:** `<think>…</think><answer>word</answer>` (free-gen) — or a single choice from
  options (MCQ variant: distractors are real words of the same length that also fit some of
  the revealed letters).
- **Ground truth:** the seed word. The revealed letters must also be respected.
- **Rejection:** keep a distilled trace only when `<answer>` **equals the target** *and* is
  consistent with the revealed letters.
- **Ambiguity caveat:** a definition + loose pattern can admit synonyms or other
  pattern-matching words. We keep SFT clean by (a) revealing enough letters that the pattern
  is fairly constraining, and (b) requiring **exact match against the seed word**. The
  free-gen scorer may optionally also accept any real word that matches the pattern *and*
  shares the WordNet sense, but exact-match is the default for SFT cleanliness.
- **Example**

  ```
  X:  Definition: "a hot drink made by infusing dried leaves in boiling water"
      Length: 5. Pattern: _ _ a _ t   (mask 60%)
  Y:  <think>5 letters, ends in 'a t' ... third letter 'a'. A hot infused drink: "chait"?
      No. The pattern _ _ a _ t with that meaning is "chait"… reconsider: it's "cha" based.
      Word is "chait"? The intended answer fitting _ _ a _ t is "chait".</think>
      <answer>chait</answer>
  ```
  *(Illustrative only — the real generator picks seed words whose pattern + definition yield
  a unique target.)*

---

## Summary table

| # | Game | Format | Reasoning? | Offline source | Producer |
|---|------|--------|:----------:|----------------|----------|
| 1 | Char counts | free `<answer>` | no | Python (char classify) | programmatic |
| 2 | Validity + meaning | `<answer>` (+`<meaning>`) | no | WordNet (NLTK) | programmatic |
| 3 | Anagrams | `<think>`+`<answer>` | yes | `sorted()` check | Claude + rejection |
| 4 | Ends→starts | MCQ `<answer>` | no | Python (char match) | programmatic |
| 5 | Rhymes | MCQ + free | no | `pronouncing` (CMU) | programmatic |
| 6 | Crossword | MCQ + free, `<think>`+`<answer>` | yes (preferred) | WordNet defs + seed word | Claude + rejection |

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
