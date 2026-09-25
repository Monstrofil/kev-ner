# kev-span: all fields in one forward pass, answers pointed at the text

An experiment in making a decoder LLM **read** instead of **write**: every field or entity type is
answered in a single forward pass, and each answer is a pointer to a span of the input document, not
generated text. It can't produce malformed JSON, and it can't invent a value that isn't printed.

It started from TypeSafe's **Jev / "System One"** (typed questions answered in one pass by pointing at
option markers) and its open reproduction **kev-0.5b**. The change here is small: point at the
**document's own tokens** instead of option markers, and a classifier becomes an extractor.

```
[ document tokens ............ ] [\n<|fim_prefix|>PER: person<|fim_suffix|>] [\n<|fim_prefix|>ORG: …<|fim_suffix|>] …
  positions 0..d-1                 positions d..                               positions d..   ← restart per branch
```

- **One branch per field or type**, written as schema text (`name: description`). Branches see the
  document and themselves, never each other: a block attention mask, with positions restarting after the
  document.
- **Readout.** The hidden state at each branch's last token (`<|fim_suffix|>`, Qwen's reserved FIM
  token reused as a "decide" marker) scores document spans.
  - **Fields (`kev_span.py`):** start and end pointers plus a null slot for "not printed".
  - **NER (`kev_ner.py`):** every span up to 16 tokens gets a score, and spans above a threshold are
    kept.
- **Backbone:** a frozen Qwen decoder plus LoRA r16. `--bidir-doc` makes the document part attend in both
  directions.
- **Types** go through fixed per-type normalisers: dates via `dateparser`, numbers via digits or Roman
  numerals.

Raw input, the attention mask and the output tensor for one sentence are in
[`io-example.txt`](io-example.txt). The full walkthrough is in [`METHOD.md`](METHOD.md). The results
page with interactive heatmaps is [`results.html`](results.html); download it and open it locally.

## Results at a glance

All runs are on H100s via Modal, one seed each unless noted.

### 1. Production field extraction (9 fields, 4 held-out collections, 59 units)

Documents are Ukrainian local-council decisions: long scanned or native PDFs rendered to text. The
fields are dates, numbers, ordinals, a place and a free-text subject.

| model | mean field acc. | after gold review | GPU ms/unit |
|---|---|---|---|
| **kev-span, Qwen2.5-1.5B + LoRA** | **0.891** | **0.903** | 52 |
| kev-span, Qwen2.5-0.5B + LoRA | 0.855 | 0.867 | 35 |
| distilled generative Qwen (the production baseline) | 0.846 | 0.858 | — |
| GLiNER multi-v2.1, fine-tuned | 0.567 | 0.580 | 90 |

The baseline makes errors kev can't make: invalid JSON (6 units), invented dates, and subjects
translated to English. The gold review (`adjudications/`) found 13 gold problems.

### 2. Universal NER English: exact span + type, micro F1

| model | EWT test | PUD test (other source) | sent/s, batch 64 |
|---|---|---|---|
| **kev, Qwen3-4B-Base, 24 of 36 layers, bidir** | **0.862** | **0.836** | 315 |
| kev, Qwen3-4B-Base, 36 layers, bidir | 0.854 | 0.833 | 214 |
| kev, Qwen2.5-1.5B | 0.844 | 0.810 | 448 |
| kev, Qwen2.5-0.5B, bidir | 0.839 | 0.813 | 814 |
| RoBERTa-large BIO tagger | 0.835 | 0.809 | 2 967 |
| XLM-R-large (UNER paper) | 0.858 | 0.805 | — |

The 4B model beats RoBERTa-large significantly on both test sets (paired bootstrap).

### 3. Cheap tricks

- **Cutting the top layers works.** With 24 of 36 layers, F1 is no worse (+0.008 / +0.004, 95% CIs
  include 0), GPU throughput is 1.5× and CPU is 2× faster. 18 layers costs about 1 point; 12 layers
  about 2.5.
- **Few-shot examples in the type descriptions don't help.** With full data the change is ±0, and
  the input gets 2.3× longer. With 500 training sentences, EWT drops 3.5 points, which is significant.

### 4. Long documents: one pass over up to 4k tokens

UNER test sentences were joined into documents of about N tokens, with the same entities at every
length.

| document length | kev, trained on sentences only | **kev, trained on sentences + joined docs** | RoBERTa-large, 512-token windows |
|---|---|---|---|
| sentence | 0.862 / 0.836 | 0.862 / 0.833 | 0.835 / 0.809 |
| 512 | 0.776 / 0.774 | **0.859 / 0.824** | 0.722 / 0.738 |
| 1 024 | 0.731 / 0.715 | **0.857 / 0.824** | 0.713 / 0.741 |
| 2 048 | 0.630 / 0.572 | **0.862 / 0.817** | 0.713 / 0.738 |
| 4 096 | 0.495 / 0.413 | **0.853 / 0.814** | 0.708 / 0.738 |

Each cell is EWT / PUD F1.

- **A model trained only on sentences falls apart on long input.** Once documents are mixed into
  training, one 4k-token pass stays within about 1–2 points of sentence-level F1.
- **Speed:** kev reads long documents at 800–1 300 sentences/s (peak about 13 GB), faster than
  per-sentence, because the type branches are paid once per document.
- **RoBERTa** was trained on sentences only, so part of its drop is the same effect.

### 5. Zero- and few-shot on unseen types

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
  run partway. These numbers were recovered from the run log (`out/zs-q3b-keep24/report.json`).

### 6. Tried and dropped: composing typed values from declared parts

`kev_compose.py` builds typed values from parts the user declares, instead of a fixed normaliser: day,
month and year as choices, and a template for the output. Where numbers are printed as words it wins
(`шістдесят другої сесії` → 62). But a choice can assert a value that isn't printed, and 173 training
units carry a strong December prior. Kept only as a record; see `METHOD.md` §7.

## How it relates to prior work

Every ingredient exists; the combination seems not to. The search was quick, so treat that claim
lightly.

- **Intra-Prompt Parallel Decoding** (Glavas et al., 2026) uses the same packing, block mask and
  restarted positions, but **generates** its answers.
- **GLiNER** (Zaratiana et al., 2024) puts type prompts and span scoring in one pass, but on an
  encoder where the types attend to each other.
- **MRC-NER** (Li et al., 2020) has a query per type and start/end pointers, but runs one pass per type.
- **LLM2Vec** and **LS-unLLaMA** make decoders bidirectional.

## Reproduce

```bash
# data (third-party datasets are not redistributed here)
mkdir -p data/uner && for f in uner-en_ewt-train uner-en_ewt-dev uner-en_ewt-test uner-en_pud-test; do
  curl -sL -o data/uner/$f.jsonl https://huggingface.co/datasets/universalner/uner_llm_inst_english/resolve/main/$f.jsonl; done
python uner.py data/uner fixtures/uner-en.json
python long_docs.py fixtures/uner-en.json fixtures/uner-long.json
# zero-shot data: see the header of zs_data.py (Pile-NER from HF, CrossNER from GitHub, MIT from CSAIL)
python zs_data.py data/zs fixtures/zs.json

modal run modal_app.py::main --only ner-q3b-keep24       # any run name, or a prefix
```

Every run is listed in `RUNS` in [`modal_app.py`](modal_app.py). `out/<run>/report.json` and
`predictions.json` hold the numbers behind every table above. Weights are not included.

| file | what it is |
|---|---|
| `kev_span.py` | one span or null per field (the production task) |
| `kev_ner.py` | many spans per type; `--bidir-doc`, `--keep-layers`, `--examples`, `--long-train` |
| `kev_zs.py`, `zs_data.py` | Pile-NER pretraining, then zero- and few-shot on unseen types |
| `long_docs.py`, `long_eval.py` | long-document fixture and one-pass evaluation |
| `encoder_ner.py` | RoBERTa-large BIO baseline (with 512-token windows for long documents) |
| `gliner_span.py` | GLiNER baseline |
| `kev_compose.py`, `rules/` | the dropped composed-conversion experiment |
| `show_io.py`, `show_span_io.py`, `build_page.py` | raw I/O dumps and the results page |
| `common.py`, `scoring.py`, `uner.py` | splits, normalisers and scorers |
| `fixtures/run-21.json`, `adjudications/` | the production field task and its gold review |
| `pull.py`, `review.py` | how that fixture was pulled and reviewed (needs the private store) |

Built in an afternoon-ish of back-and-forth with Claude. Numbers are single-seed unless a bootstrap is
quoted.
