"""Contender A — GLiNER: a bidirectional encoder scoring every candidate span against every field label.

Every schema field is a label; one encoder pass per text window scores all spans x all labels at once.
Per field the best span over all windows wins if it clears a threshold picked on the dev collections
(else null), then the field TYPE's normaliser turns it into the typed value.

    python -m baselines.gliner_span fixtures/run-21.json --out results/gliner --languages uk --dev tsrada.gov.ua uzmr.gov.ua
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from gliner import GLiNER
from gliner.data_processing import WordsSplitter

from kev.field_task import align, accuracy, ece, field_ok, normalise, plain, report, split

SPLITTER = WordsSplitter("whitespace")


def label(field: dict) -> str:
    return field["name"].replace("_", " ")


def windows(text: str, size: int, stride: int) -> list[tuple[int, list[tuple[str, int, int]]]]:
    """``(char_offset, words)`` windows of ``size`` words every ``stride`` words; offsets are window-local."""
    words = list(SPLITTER(text))
    out = []
    for lo in range(0, max(1, len(words)), stride):
        chunk = words[lo:lo + size]
        if not chunk:
            break
        base = chunk[0][1]
        out.append((base, [(w, s - base, e - base) for w, s, e in chunk]))
        if lo + size >= len(words):
            break
    return out


def examples(unit: dict, fields: list[dict], size: int, stride: int, languages: list[str],
             stats: Counter, rng: random.Random) -> list[dict]:
    text = plain(unit["text"])
    spans = {}
    for f in fields:
        value = unit["gold"][f["name"]]
        if value is not None and f["name"] in unit["labelled"]:
            spans[f["name"]] = align(f, value, text, languages)
            stats[f"{f['name']}:{'aligned' if spans[f['name']] else 'unprintable'}"] += 1
    by_name = {f["name"]: f for f in fields}
    positives, negatives = [], []
    for base, words in windows(text, size, stride):
        ner = []
        for name, span in spans.items():
            if span is None:
                continue
            cs, ce = span[0] - base, span[1] - base
            idx = [i for i, (_, s, e) in enumerate(words) if s < ce and e > cs]
            if idx and words[idx[0]][1] <= cs + 1 and words[idx[-1]][2] >= ce - 1:
                ner.append([idx[0], idx[-1], label(by_name[name])])
        present = {n for _, _, n in ner}
        # Only a LABELLED field may be a negative: a correction row's unlabelled fields are unknown.
        example = {"tokenized_text": [w for w, _, _ in words], "ner": ner,
                   "ner_negatives": [label(f) for f in fields
                                     if label(f) not in present and f["name"] in unit["labelled"]]}
        (positives if ner else negatives).append(example)
    # Every window with a span, plus as many span-free windows: the model must learn "nothing here".
    return positives + rng.sample(negatives, min(len(negatives), max(1, len(positives))))


def predict(model: GLiNER, unit: dict, fields: list[dict], size: int, stride: int) -> dict[str, tuple[str, float]]:
    """Best (span_text, score) per field over all windows of the unit."""
    text = plain(unit["text"])
    texts = [text[base:base + words[-1][2]] for base, words in windows(text, size, stride)]
    by_label = {label(f): f["name"] for f in fields}
    best: dict[str, tuple[str, float]] = {}
    for ents in model.inference(texts, list(by_label), threshold=0.05, flat_ner=True, batch_size=8):
        for e in ents:
            name = by_label[e["label"]]
            if name not in best or e["score"] > best[name][1]:
                best[name] = (e["text"], e["score"])
    return best


def decide(best: dict, fields: list[dict], tau: float, languages: list[str]) -> tuple[dict, dict]:
    """(typed prediction, confidence) per field; confidence in the null answer is 1 - best span score."""
    pred, conf = {}, {}
    for f in fields:
        text, score = best.get(f["name"], (None, 0.0))
        pred[f["name"]] = normalise(f, text, languages) if score >= tau else None
        conf[f["name"]] = score if score >= tau else 1 - score
    return pred, conf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="urchade/gliner_multi-v2.1")
    parser.add_argument("--languages", nargs="+", required=True, help="date-parser languages, e.g. uk")
    parser.add_argument("--dev", nargs="+", required=True, help="training collections held out to pick the threshold")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--max-width", type=int, default=48)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cpu-latency", type=int, default=0, help="time N eval units on CPU after eval")
    parser.add_argument("--review", type=Path, help="adjudicated-gold file to also score against")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    fields = fixture["task"]["fields"]
    train, dev, ev = split(fixture, args.dev)
    print(f"train {len(train)} / dev {len(dev)} / eval {len(ev)} units", flush=True)

    model = GLiNER.from_pretrained(args.base)
    model.config.max_width = args.max_width
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_sec = 0.0
    if not args.zero_shot:
        stats: Counter = Counter()
        data = [ex for u in train for ex in examples(u, fields, args.window, args.stride, args.languages, stats, rng)]
        rng.shuffle(data)
        print(f"alignment {dict(sorted(stats.items()))}\n{len(data)} training windows", flush=True)
        started = time.time()
        model.train_model(
            train_dataset=data, eval_dataset=None, output_dir=str(args.out / "ckpt"),
            learning_rate=1e-5, others_lr=5e-5, weight_decay=0.01, others_weight_decay=0.01,
            per_device_train_batch_size=args.batch, max_steps=args.steps, warmup_ratio=0.1,
            focal_loss_alpha=0.75, focal_loss_gamma=2, save_strategy="no",
            logging_steps=50, bf16=device == "cuda", dataloader_num_workers=0,
        )
        train_sec = time.time() - started
    model.to(device).eval()

    def run(units: list[dict]) -> tuple[list[dict], float]:
        started = time.time()
        with torch.no_grad():
            bests = [predict(model, u, fields, args.window, args.stride) for u in units]
        return bests, (time.time() - started) / len(units)

    dev_best, _ = run(dev)
    taus = [round(0.05 * i, 2) for i in range(1, 19)]
    dev_scores = {t: accuracy(dev, [decide(b, fields, t, args.languages)[0] for b in dev_best], fields)["mean_field"]
                  for t in taus}
    tau = max(dev_scores, key=dev_scores.get)

    ev_best, sec_per_unit = run(ev)
    decided = [decide(b, fields, tau, args.languages) for b in ev_best]
    preds = [p for p, _ in decided]
    confs = [c[f["name"]] for _, c in decided for f in fields]
    right = [field_ok(f["type"], u["gold"][f["name"]], p[f["name"]]) for u, (p, _) in zip(ev, decided) for f in fields]

    cpu_ms = None
    if args.cpu_latency:
        model.to("cpu")
        _, cpu_sec = run(ev[:args.cpu_latency])
        cpu_ms = cpu_sec * 1000

    extra = {"contender": "gliner", "base": args.base, "zero_shot": args.zero_shot, "threshold": tau,
             "dev_mean_field": dev_scores[tau], "dev_collections": args.dev, "train_sec": train_sec,
             "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
             "gpu_ms_per_unit": sec_per_unit * 1000, "cpu_ms_per_unit": cpu_ms, "ece": ece(confs, right)}
    report(args.out, fixture, ev, preds, fields, args.languages, extra,
           [{n: list(v) for n, v in b.items()} for b in ev_best], args.review)


if __name__ == "__main__":
    main()
