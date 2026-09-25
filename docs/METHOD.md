# How we got here — one-pass span extraction from the Jev idea

This is the method record for `span-bakeoff/`: where the architecture came from, what we built and why,
every fix that moved a number, and how to reproduce each result. The result tables live in
[`README.md`](../README.md). This file explains how they were reached.

## 1. Starting point: Jev / "System One" → kev

TypeSafe's **Jev** ("System One" models) answers a schema of typed questions (yes/no, choice, score)
**in one forward pass**, with no free-text generation. Its internals are unpublished. The open
reproduction **kev-0.5b** fills them in, and we took these pieces from it:

- **A packed sequence.** The input comes first, then one short *branch* per question.
- **A block-causal branch mask.** Every branch sees the input and itself, never another branch, so
  answering N questions costs one pass plus N short branches.
- **A pointer head.** The hidden state at a branch's final `<decide>` token is dot-producted against
  candidate positions (in kev, the option markers), and a softmax over them is the answer.
- **The backbone.** A small decoder (Qwen2.5-0.5B), frozen, with LoRA r16.
- **Calibration.** One post-hoc temperature fitted on held-out data.

**The adaptation:** a span question has no option list. So the pointer aims at the **document's own tokens**,
and the answer is a substring of the input. It can't be malformed JSON or an invented value. That one
change turns kev's classifier into an extractor.

## 2. kev-span: one span or null per field (`kev/span.py`)

```
[ document tokens ............ ] [\n<|fim_prefix|>date: … <|fim_suffix|>] [\n<|fim_prefix|>number: … <|fim_suffix|>] …
  positions 0..d-1                 positions d..d+n1-1                      positions d..d+n2-1   ← restart per branch
```

- **Markers.** Qwen's reserved `<|fim_prefix|>` and `<|fim_suffix|>` tokens are reused as the branch
  opener and `<decide>`, as kev does. User text is scrubbed of `<|` so it can't forge them.
- **Branch text.** Each branch is `name: description`, straight from the user's schema. Adding a field
  means adding a branch, with no retraining of structure.
- **The mask.** It is a 4D additive attention mask passed to Qwen via SDPA, and is causal within the
  document. Each branch sees the whole document and itself causally, and nothing else.
- **Positions.** Every branch's position ids restart at `d`, so no branch looks "further away" from the
  document than another.
- **Isolation check.** It is asserted at start-up. The check reverses field 0's branch tokens and
  requires field 1's logits to move by less than 1e-2. The first version edited a *later* branch, which
  plain causal attention would also have hidden, so it tested nothing. It now edits branch 0.
- **Readout.** Two pointers (start and end) run from each `<decide>` state over document tokens, plus a
  learned **null slot** meaning "not printed". The best span maximises `log p(start) + log p(end)` within
  a `--max-span` band. It wins if `log p(span) + bias > log p(null)`.
- **Typed value.** A normaliser chosen by the field's declared **type** (not its name) builds the
  value: `date` goes through `dateparser` with the declared languages, `number` through digits or Roman
  numerals with a Cyrillic homoglyph fold, and `text` stays as printed.
- **Training.** Cross-entropy on start and end, with `ignore_index` for unknowable targets. AdamW, heads
  at 5× the LoRA learning rate (2e-4), OneCycle schedule, 12 epochs.

**Data.** `scripts/pull.py 21` freezes a production training run from talpa (schema, current gold, the run's
held-out collections as eval, and the released generative model's eval dump as baseline). The run is
`header_metadata`, with 9 fields: dates, numbers, ordinals, a place and a free-text subject. Two whole
training collections are carved off as dev, so nothing is tuned on eval. The held-out-collections split
is the only generalisation measure.

### What moved kev-span from 0.835 to 0.891 (no architecture change)

| fix | symptom | cause |
|---|---|---|
| `--max-span` 96 → 384 | long subjects always wrong | 14/58 eval subjects exceed 96 Qwen tokens: **unreachable** |
| span-vs-null **bias** picked on dev | null over-predicted | argmax of joint start×end vs a single null slot is not like-for-like |
| partial rows → `IGNORE` | fields trained toward null | correction rows carry only the corrected fields (name-keyed cells), and the rest were being taught "not printed" |

Also along the way:
- The pulled baseline `pred` holds only the *wrong* fields, so the rest take their value from gold.
- Row ids come in two formats (`sha#sN` / `sha:N`), so `scripts/pull.py` splits on the 64-hex prefix.

**Gold review.** Every eval mismatch (kev's and the baseline's) was read against the text. 13 gold
problems are recorded in [`adjudications/run-21.json`](../adjudications/run-21.json), local only and never written back. The review also
surfaced a **scorer bug**: `SequenceMatcher`'s default `autojunk=True` scores long strings near 0. It is
reported, not fixed here.

**Outcome.**
- kev-1.5b scores 0.891 (0.903 adjudicated), against 0.846 for the distilled generative model.
- It runs at 52 ms per unit on an H100.
- It cannot emit invalid JSON or invent a value, and both are baseline failure modes.

## 3. kev-ner: many spans per type (`kev/ner.py`)

NER breaks one assumption: a sentence can hold *several* PERs. So the readout changes from "one span or
null" to "a set of spans". Everything else is kept: packing, mask, restarted positions, the isolation
assert, LoRA r16, and one forward pass for all types.

**Benchmark.** `universalner/uner_llm_inst_english`. `kev/uner.py` strips the LLM instruction wrapper back
to the sentence plus character-offset entities, and asserts `text[Start:End] == Text` for every entity.
5 entities with a null `Text` (the dataset's own conversion bug) are dropped. The branch descriptions
are the dataset's own instruction wording (e.g. `ORG: organization; organizations can represent other
groups of people; nationalities are not organizations`).
- Train on EWT train (12.5k sentences) and tune on EWT dev.
- Test on **EWT test** (in-source) and **PUD** (news and Wikipedia, a different source: the cross-source bar).

**Span head.** `SpanHead` enumerates every span `(i, w)` with `w < 16` tokens and scores it with each
branch's `<decide>` state as the query:

```
rep(i, w)   = MLP( W_s·h_i  +  W_e·h_{i+w}  +  W_n·h_{i+w+1}  +  width_emb[w] )
score(t,i,w) = q_t · rep(i, w) / √256  +  bias_t          (q_t, bias_t from <decide> of type t)
```

- **Why `h_{i+w+1}`.** In a causal document a token can't see what follows it. The token *after* the span
  gives one token of lookahead, which is what the model needs to tell "Bank" from "Bank of America".
- **Loss.** Binary cross-entropy over all valid spans, **summed per sentence** (not averaged over the
  ~1000 mostly-negative spans, which would drown the few positives).
- **Bias prior.** The bias starts at **−6**, the prior that nearly no span is an entity. Without it,
  epoch 1 spent its loss (683 per sentence at init) unlearning p=0.5 on every span.
- **Decode.** Keep spans with logit above a threshold picked on dev (grid −3…+3), best first, and drop
  any that overlaps a kept span (the data is flat), across all types at once.
- **Offsets.** Qwen's byte-level BPE folds the leading space into a token, so predicted character spans
  are left-trimmed (`Packed.char_span`) before scoring.
- **Padding fix.** In a padded batch the "next token" after a shorter sentence was that sentence's
  first branch token, while at batch 1 it is zero. Document states past each sentence's end are now
  zeroed, so train and inference see the same thing.

**Scoring and sanity.**
- Entity-level micro P/R/F1, requiring the exact character span *and* type (`uner.score`). Gold is
  de-duplicated: EWT test has 17 duplicate entries.
- A **gold round-trip oracle** fed gold through kev's decoder and through the BIO decoder. Both score
  F1 ≈ 0.99 on both test sets, so the decoding and scoring are sound and the two approaches share one ceiling.
- About 1% of gold is unreachable for kev because its edge falls inside a Qwen token (14/1076 EWT, 7/1071 PUD).
- **Reference model:** `baselines/encoder_ner.py`, a RoBERTa-large BIO token classifier (5 epochs, lr 2e-5), scored
  by the same scorer. **Published reference:** XLM-R-large fine-tuned on en_ewt, UNER paper (Mayhew et al.,
  NAACL 2024, Fig. 4): EWT 0.858, PUD 0.805.

### The step that won: scale plus a bidirectional document

**`--bidir-doc`** makes the document block's attention full instead of causal (LLM2Vec-style).
- Branches are unchanged: they still see the whole document and themselves only.
- Nothing else changes. LoRA adapts the backbone, which was pre-trained causally, to reading both ways.

| backbone | causal EWT / PUD | bidirectional EWT / PUD |
|---|---|---|
| Qwen2.5-0.5B | 0.838 / 0.798 | 0.839 / 0.813 |
| Qwen2.5-1.5B | 0.844 / 0.810 | 0.842 / 0.811 |
| **Qwen3-4B-Base** | 0.838 / 0.819 | **0.854 / 0.833** |
| RoBERTa-large (reference) | 0.835 / 0.809 | |

Up to 1.5B, every kev variant ties RoBERTa: in the paired bootstrap, every 95% interval includes 0. At
**4B with a bidirectional document** the gain is significant:
- vs RoBERTa: EWT +0.019 [+0.001, +0.039], PUD **+0.024 [+0.009, +0.039]**
- vs its causal twin: EWT +0.016, PUD +0.014, with both intervals above 0

It also beats XLM-R-large's published PUD score (0.833 vs 0.805).

**Reading.** At 4B, reading the document both ways matters: boundaries and types depend on what comes
*after* the token, and the one-token lookahead in the head only partly covers that at small scale.
Scale alone (causal 4B) buys nothing over 1.5B on EWT.

**Qwen3-4B specifics.**
- `Qwen/Qwen3-4B-Base` (the base model, not instruct) keeps the same `<|fim_*|>` tokens, and
  transformers 4.51.3 already ships Qwen3, so no code changed except:
- **`--grad-ckpt`**, because a batch of 32 of 4B activations (36 layers, 9.7k MLP width, sentences up to
  256 tokens) OOMs an 80 GB H100. It is checkpointing only, so results are unaffected. Training takes 14.5 min.

## 4. Cost and speed (H100, batch 64, forward plus decode)

| model | sentences/s | ≈ $ per 1M sentences at ~$4/H100-h |
|---|---|---|
| RoBERTa-large | 2 967 | ~0.4 |
| kev Qwen2.5-0.5B bidir | 814 | ~1.4 |
| kev Qwen2.5-1.5B | 448 | ~2.5 |
| kev Qwen3-4B bidir | 214 | ~5 |

Every kev variant is still one forward pass with no decoding. A generative LLM emitting the same JSON
pays per output token and can produce malformed or invented values. On run 21 the estimate was about
3–5× throughput and 15–30× lower latency than the vLLM generative student. The encoder remains the
cheapest by far. kev's case is F1 at 4B plus the interface: types are schema text read at inference.
In principle one model serves any declared type list, but that has **not been tested** on types
unseen in training.

## 5. Caveats

- **One training seed per model.** The bootstrap covers test-set sampling, not seed variance. Two more
  seeds of the 4B bidir run (about 15 min of H100 each) would settle it.
- Run 21 is small (59 eval units). The UNER result is the statistically meaningful one.
- CPU latencies were measured on shared Modal cores at batch 1 and are noisy.
- The `autojunk` scorer bug is unfixed in `distiller/…/generative/scoring.py`.

## 6. What the model reads and emits (raw)

kev-ner writes **no text and no JSON**. It reads one token sequence and returns one tensor of scores;
a few lines of Python turn that into character spans. `scripts/show_io.py` dumps each stage for any sentence;
[`io-example.txt`](io-example.txt) is the full dump below, from the trained Qwen3-4B bidir model on a PUD test
sentence.

```bash
python -m scripts.show_io results/ner-kev-q3-4b-bidir --base Qwen/Qwen3-4B-Base --bidir-doc --threshold -2.0 \
  --text "According to Parker, Russian Secret Service agents are active in large numbers in Great Britain."
# needs the run's adapter/ + head.pt: modal volume get kev-ner ner-kev-q3-4b-bidir/ results/
```

**Input: one packed token sequence.** It holds the sentence and then one branch per type from the
schema. There is no instruction prompt and no few-shot example.

```
  0 pos= 0  'According'                  doc
  2 pos= 2  'ĠParker'                    doc
 16 pos=16  '.'                          doc      … 17 sentence tokens
 17 pos=17  'Ċ'                          branch PER
 18 pos=18  '<|fim_prefix|>'             branch PER
 19 pos=19  'PER'  ':'  'Ġperson'
 22 pos=22  '<|fim_suffix|>'             branch PER   ← <decide>: the head reads this state
 23 pos=17  'Ċ'                          branch ORG   ← positions restart after the sentence
     …  'ORG: organization; organizations can represent other groups of people; nationalities are not organizations'
 42 pos=36  '<|fim_suffix|>'             branch ORG   ← <decide>
 43 pos=17  'Ċ'                          branch LOC   ← restart again
 54 pos=28  '<|fim_suffix|>'             branch LOC   ← <decide>
```

**Attention mask** (row attends to column, `#` = allowed):

```
      DDDDDDDDDDDDDDDDDPPPPPPOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLL
D   5 #################......................................   sentence ↔ sentence (bidirectional)
P  22 #######################................................   PER: sentence + own branch
O  42 #################......####################............   ORG: sentence + ORG only, never PER
L  54 #################..........................############   LOC: sentence + LOC only
```

This is one forward pass through the backbone (`AutoModel`, hidden states only). The language-model
head is never used.

**Raw output: a single tensor of span logits,** shape `(3, 17, 16)` = [type, start token, width − 1].
Each cell answers "is the span of w+1 tokens starting at token i an entity of type t?". 456 of the 816
cells are real spans; the rest run past the sentence end and are masked. The top cells:

```
logit  +8.79  p=1.000  LOC  tokens 14..15  chars 82..95  'Great Britain'
logit  +6.18  p=0.998  PER  tokens  2..2   chars 13..19  'Parker'
logit  +6.08  p=0.998  ORG  tokens  4..6   chars 21..43  'Russian Secret Service'
logit  -8.75  p=0.000  ORG  tokens  2..2               'Parker'                ← right span, wrong type
logit -10.26  p=0.000  ORG  tokens  2..6               'Parker, Russian Secret Service'
logit -12.50  p=0.000  ORG  tokens  5..6               'Secret Service'        ← too short
```

**Decode: plain Python (`kev.ner.decode`), no model.** It keeps logits above the dev-tuned threshold
(−2.0), best first, and drops overlaps:

```
[[82, 95, 'LOC'], [13, 19, 'PER'], [21, 43, 'ORG']]          ← [char_start, char_end, type]
  LOC: 'Great Britain'   PER: 'Parker'   ORG: 'Russian Secret Service'
```

| | generative LLM | kev |
|---|---|---|
| input | instruction + example + sentence | sentence + `name: description` per type |
| output | text tokens `"Results": [{"TypeName": "LOC", "Text": …` | a (types × start × width) float tensor |
| forward passes | one per output token (~70 here) | **1**, whatever the entity count |
| can fail by | broken JSON, invented or paraphrased text, wrong offsets | only a wrong or missing span; output is always a substring of the input |
| confidence | none directly | a probability for every candidate span |

Any JSON a consumer sees is built afterwards from these offsets. The run-21 extractor (`kev/span.py`) is
the same idea with one start and one end pointer per field plus a null slot, instead of the span grid.

## 7. Follow-ups: layer cutting, few-shot descriptions, long documents, zero-shot, composed conversion

### Cutting the top layers (`kev.ner --keep-layers N`)

Only the backbone's first N decoder layers run; the top ones mostly serve next-token prediction.
Qwen3-4B bidir on UNER, one seed each; the bootstrap is against the full 36 layers.

| layers | EWT F1 | PUD F1 | sent/s b64 | CPU ms/sent | vs 36 (EWT / PUD, 95% CI) |
|---|---|---|---|---|---|
| 36 | 0.854 | 0.833 | 214 | 924 | — |
| 30 | 0.853 | 0.828 | 258 | — | — |
| **24** | **0.862** | **0.836** | **315** | **430** | +0.008 [−0.005, +0.022] / +0.004 [−0.009, +0.015] |
| 18 | 0.847 | 0.823 | 398 | 337 | −0.007 [−0.021, +0.008] / −0.010 [−0.022, +0.002] |
| 12 | 0.830 | 0.810 | 553 | 221 | — |

24 layers is no worse and 1.5× faster on GPU, 2× on CPU. It also beats the full 1.5B model
(0.844 / 0.810) at a similar speed: a cut big model beats a whole small one.

### Few-shot descriptions (`--examples K`)

Each type's description gets its K most frequent training mentions (`PER: person; e.g. Bush, …`).
Full data: 0.856 / 0.834, no change, at 145 vs 214 sent/s, because the packed length grows 2.3×. With 500
training sentences (`--train-limit 500 --batch 8 --epochs 10`): 0.723 / 0.765 against 0.758 / 0.773
without examples; EWT −0.035 [−0.057, −0.013]. Dropped.

### Long documents (`kev/long_docs.py`, `scripts/long_eval.py`, `kev.ner --long-train`)

UNER test sentences are joined into documents of about N Qwen tokens, carrying gold offsets along. Every
length holds the same entities. kev reads a document in one pass. RoBERTa reads it in 512-token windows
that overlap by 128 tokens, and each token takes its label from the window where it sits furthest from an
edge. A gold-label oracle through those windows scores 0.992–0.993, so the stitching is sound.

| length | kev keep24, trained on sentences | kev keep24, `--long-train 4096` | RoBERTa-large, windows |
|---|---|---|---|
| sentence | 0.862 / 0.836 | 0.862 / 0.833 | 0.835 / 0.809 |
| 512 | 0.776 / 0.774 | 0.859 / 0.824 | 0.722 / 0.738 |
| 1 024 | 0.731 / 0.715 | 0.857 / 0.824 | 0.713 / 0.741 |
| 2 048 | 0.630 / 0.572 | 0.862 / 0.817 | 0.713 / 0.738 |
| 4 096 | 0.495 / 0.413 | 0.853 / 0.814 | 0.708 / 0.738 |

Each cell is EWT / PUD F1.

- **Trained on sentences, kev collapses with length.** At 4k tokens recall is 0.38 and precision 0.71:
  it never saw a long document or branch positions that far out.
- **`--long-train` fixes it.** Every epoch adds the sentences joined afresh into documents of
  log-uniform length in [256, 4096]; batches are packed by tokens; the threshold is tuned on dev sentences
  plus dev documents. One 4k-token pass then stays within about 1–2 points of sentence level, and
  sentence F1 is unchanged.
- **Throughput is higher on long documents:** 800–1 300 sentences/s against 315 per sentence, because the
  branches are paid once per document. Peak is about 13 GB.
- **RoBERTa is still faster** (3 500–9 400/s, 5.6 GB), but about 12 points lower. It too was trained on
  sentences only; windows holding many sentences are new to it, so part of its drop is the same effect.

### Zero- and few-shot on unseen types (`scripts/zs_data.py`, `kev/zero_shot.py`)

One epoch on **Pile-NER** (45k passages, about 13k LLM-labelled open types; 31 min on an H100), then
datasets whose types it never trained on. The type **name** is the only schema. The threshold was picked
on held-out Pile passages, never on a test set. F1 in %:

| CrossNER domain | kev zero-shot | **kev few-shot** (Pile model + 50 sentences) | 50 sentences, no Pile | UniNER-7B zero-shot | GLiNER-L zero-shot |
|---|---|---|---|---|---|
| AI | 51.6 | **70.0** | 0.0 | 53.6 | 57.2 |
| literature | 45.4 | **75.0** | 10.3 | 59.3 | 64.4 |
| music | 59.4 | **81.6** | 9.1 | 67.0 | 69.6 |
| politics | 53.0 | **77.9** | 1.9 | 60.9 | 72.6 |
| **average** | **52.4** | **76.1** | 5.3 | 60.2 | 66.0 |

The published numbers are from the GLiNER paper, Table 1.

- **Zero-shot works, but trails the specialists.** kev reads types it has never seen, but it is 8–14
  points behind UniNER-7B and GLiNER-L. It got one epoch, label names only, and no tuning; they are
  trained for this.
- **Few-shot is where it shines.** 50 labelled sentences from the Pile model reach 70–82 F1. The
  same 50 sentences without Pile pretraining barely learn anything (0–10).

  That no-Pile arm uses a fixed threshold of 0 with no dev set, so it is somewhat pessimistic. The gap
  is still the story.
- **Incomplete.** CrossNER science and MIT movie/restaurant didn't finish: the GPU provider stopped the
  run partway. These numbers were recovered from the run log ([`results/zs-q3b-keep24/report.json`](../results/zs-q3b-keep24/report.json)).

### Composed conversion: the value built from user-declared parts (`archive/compose.py`)

The fixed normaliser only reads what its rules know: `dateparser`, digits and Roman numerals. It cannot read
«шістдесят другої сесії» (62). And because gold is aligned to the text by the same rules, **75
training values were dropped as unfindable**: 31 `session_no`, 27 `parent_date`, 12 `convocation_no`,
5 `date`.

`archive/rules/run-21.json` declares, per field, the **parts** (each a set of options), a **gold** pattern that
splits a gold value into parts, and an **output** template:

```json
"parent_date": {"parts": {"day": "1-31", "month": "1-12", "year": "2000-2030"},
                "gold": "(?P<year>\\d{4})(?:-(?P<month>\\d{2})(?:-(?P<day>\\d{2}))?)?",
                "output": "{year:04d}-{month:02d}-{day:02d}"}
```

Each part is one more branch in the same pass. It lists its options, each followed by a marker
(`… 11<|fim_middle|> 12<|fim_middle|><|fim_suffix|>`). The `<decide>` state points at a marker or a
null slot; this is kev's own choice readout. The field is null when the parts' summed null log-prob
beats their best options plus a bias tuned on dev. The span branches still train, so one model
(Qwen2.5-1.5B, 12 epochs) gives both readouts. Correct values out of units:

| field | dev fixed | dev composed | dev gated | eval fixed | eval composed | eval gated |
|---|---|---|---|---|---|---|
| date | 20/20 | 20/20 | 20/20 | 56/59 | 56/59 | 56/59 |
| parent_date | 20/20 | 20/20 | 20/20 | 52/59 | 51/59 | 50/59 |
| session_no | 12/20 | **19/20** | 18/20 | 59/59 | 57/59 | 59/59 |
| convocation_no | 11/20 | **20/20** | 17/20 | 57/59 | 55/59 | 57/59 |

*gated*: the parts give the value, but only when the span readout is non-null (computed offline from
[`results/compose-1.5b/parts.json`](../results/compose-1.5b/parts.json)).

- **Dev** is two whole training collections the model never trained on. Only the compose bias was tuned
  on it. The fixed path fails there:
  - «шістдесят другої сесії» → `62227`: the span grabbed stray digits.
  - «107 позачергова сесія» → `7`: Qwen splits digits, and the span started on the last one.
  - «VIIІ скликання» (the last І is Cyrillic) → `7`: the span stopped before it.

  Composition reads all of them right.
- **Eval** has no word-written numbers at all, so it can only show regressions. Composition alone
  **invents** values:
  - `session_no` 105 from «№105/105»;
  - `convocation_no` 8 where none is printed, because 8 is the only convocation in training.

  Parts are bounded choices, not text pointers, so "cannot invent" no longer holds. Gating on the span
  readout removes those. What remains is the prior: «14 лютого» → month 12, because 173 training
  units are nearly all December. It also fixed one OCR case: «180.12.2025» → 2025-12-18.
- **Cost:** the part branches add about 1,900 tokens (700 for 150 session options). The packed p50 is
  3,215 and GPU time is 98 ms/unit against 52 ms.

**Verdict:** worth keeping as an opt-in, gated readout for fields printed as words. The gain is real
where the rules are blind, and gating keeps eval flat. It needs a proper span-null gate instead of the
offline approximation, and more varied months in training. Shorter option lists would also help: digit
parts (tens and ones) instead of 150 whole numbers.

## 8. Reproduce

Run from the repository root (every entry point is a module: `python -m kev.ner`, `python -m scripts.show_io`, …).

```bash
# run 21 (production fields, held-out collections)
python -m scripts.pull 21 --ground <store-url>                    # needs talpa access
modal run modal_app.py::main --only kev-                           # kev-0.5b, kev-1.5b

# Universal NER English
mkdir -p data/uner
for f in uner-en_ewt-train uner-en_ewt-dev uner-en_ewt-test uner-en_pud-test; do
  curl -sL -o data/uner/$f.jsonl \
    https://huggingface.co/datasets/universalner/uner_llm_inst_english/resolve/main/$f.jsonl
done
python -m kev.uner data/uner fixtures/uner-en.json
modal run modal_app.py::main --only ner-                           # all kev-ner variants + RoBERTa
modal run modal_app.py::main --only ner-kev-q3-4b-bidir            # just the winner
modal volume get kev-ner ner-kev-q3-4b-bidir/ results/   # adapter, head.pt, predictions, report
```

The winning configuration, in full:

```bash
python -m kev.ner fixtures/uner-en.json --out results/ner-kev-q3-4b-bidir \
  --base Qwen/Qwen3-4B-Base --bidir-doc --grad-ckpt \
  --rank 16 --epochs 4 --batch 32 --lr 2e-4 --max-width 16 --max-doc 256 --seed 0
```

| file | role |
|---|---|
| `kev/span.py` | single-span-or-null extractor (run 21) |
| `kev/ner.py` | multi-span NER variant (`--keep-layers`, `--examples`, `--train-limit`) |
| `archive/compose.py`, `archive/rules/run-21.json` | composed conversion: typed values built from user-declared parts (§7) |
| `baselines/encoder_ner.py` | RoBERTa BIO reference |
| `baselines/gliner_span.py` | GLiNER reference (run 21; 0.567, dropped) |
| `kev/uner.py` | UNER fixture builder + entity-level scorer |
| `scripts/show_io.py`, `docs/io-example.txt` | raw input / mask / output-tensor / decode dump for one sentence (§6) |
| `scripts/pull.py`, `kev/field_task.py`, `scripts/review.py`, `adjudications/` | run-21 fixture, scoring, gold review |
| `modal_app.py` | every run, parallel on H100s; outputs on the `kev-ner` volume |
