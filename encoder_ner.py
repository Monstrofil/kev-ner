"""Reference: the standard fine-tuned encoder tagger (BIO token classification), same fixture and scorer.

``--long`` also scores the long-document fixture (``long_docs.py``) the way a 512-token encoder must read
it: overlapping windows, each token labelled by the window where it sits furthest from an edge.

    python encoder_ner.py fixtures/uner-en.json --out out/roberta-large --long fixtures/uner-long.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from uner import line, score


def labels_of(text: str, entities: list[list], offsets: list[tuple[int, int]], tags: dict) -> list[int]:
    """B- on the token that opens an entity, I- inside it, O elsewhere; special tokens are ignored."""
    out = []
    for a, b in offsets:
        if a == b:
            out.append(-100)
            continue
        tag = "O"
        for start, end, kind in entities:
            if a < end and b > start:
                tag = ("B-" if a <= start else "I-") + kind
        out.append(tags[tag])
    return out


def spans_of(offsets: list[tuple[int, int]], pred: list[int], names: list[str]) -> list[list]:
    """BIO → character spans; an I- that does not continue the same type opens a new entity."""
    out, open_ = [], False
    for (a, b), label in zip(offsets, pred):
        if a == b:
            continue
        tag = names[label]
        if tag == "O":
            open_ = False
        elif tag.startswith("I-") and open_ and out[-1][2] == tag[2:]:
            out[-1][1] = b
        else:
            out.append([a, b, tag[2:]])
            open_ = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="FacebookAI/roberta-large")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cpu-latency", type=int, default=0)
    parser.add_argument("--long", type=Path, help="long-document fixture to also score, in windows")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128, help="tokens shared by neighbouring windows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    names = ["O"] + [f"{p}-{t['name']}" for t in fixture["types"] for p in "BI"]
    tags = {n: i for i, n in enumerate(names)}
    tok = AutoTokenizer.from_pretrained(args.base, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.base, num_labels=len(names)).to(device)

    def encode(units: list[dict], labelled: bool) -> dict:
        enc = tok([u["text"] for u in units], return_offsets_mapping=True, truncation=True, max_length=256,
                  padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        if labelled:
            enc["labels"] = torch.tensor([labels_of(u["text"], u["entities"], o, tags)
                                          for u, o in zip(units, offsets)])
        return enc, offsets

    train = list(fixture["splits"]["train"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        total = 0.0
        for i in range(0, len(train), args.batch):
            enc, _ = encode(train[i:i + args.batch], True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / math.ceil(len(train) / args.batch):.4f}", flush=True)
    train_sec = time.time() - started
    model.eval()

    def predict(units: list[dict], batch: int) -> list[list]:
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(units), batch):
                enc, offsets = encode(units[i:i + batch], False)
                pred = model(**enc.to(device)).logits.argmax(-1).tolist()
                out.extend(spans_of(o, p, names) for o, p in zip(offsets, pred))
        return out

    results, predictions = {}, {}
    for name in ("dev", "ewt_test", "pud_test"):
        units = fixture["splits"][name]
        predictions[name] = predict(units, 64)
        results[name] = score(units, predictions[name])
        print(line(name, results[name]), flush=True)

    test = fixture["splits"]["ewt_test"] + fixture["splits"]["pud_test"]
    speed = {}
    for batch in (1, 64):
        torch.cuda.synchronize()
        started = time.time()
        predict(test, batch)
        torch.cuda.synchronize()
        speed[f"gpu_sent_per_sec_b{batch}"] = len(test) / (time.time() - started)
    if args.cpu_latency:
        model.to("cpu").float()
        with torch.no_grad():
            started = time.time()
            for u in test[:args.cpu_latency]:
                model(**tok([u["text"]], return_tensors="pt"))
        speed["cpu_ms_per_sent_b1"] = (time.time() - started) / args.cpu_latency * 1000
    print(json.dumps(speed), flush=True)

    def predict_long(units: list[dict], batch: int) -> list[list]:
        """Windows of ``--window`` tokens overlapping by ``--stride``; per token, the label from the window
        whose nearer edge is furthest away."""
        enc = tok([u["text"] for u in units], return_offsets_mapping=True, truncation=True, max_length=args.window,
                  stride=args.stride, return_overflowing_tokens=True, padding=True, return_tensors="pt")
        owner, offsets = enc.pop("overflow_to_sample_mapping").tolist(), enc.pop("offset_mapping").tolist()
        best = [{} for _ in units]   # char span -> (distance to the window edge, label)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(owner), batch):
                pred = model(**{k: v[i:i + batch].to(device) for k, v in enc.items()}).logits.argmax(-1).tolist()
                for w, labels in zip(range(i, i + batch), pred):
                    real = [k for k, (a, b) in enumerate(offsets[w]) if a != b]
                    for rank, k in enumerate(real):
                        key, edge = tuple(offsets[w][k]), min(rank, len(real) - 1 - rank)
                        if key not in best[owner[w]] or edge > best[owner[w]][key][0]:
                            best[owner[w]][key] = (edge, labels[k])
        return [spans_of(sorted(b), [b[k][1] for k in sorted(b)], names) for b in best]

    long_results = {}
    if args.long:
        long_fixture = json.loads(args.long.read_text(encoding="utf-8"))
        for name, units in long_fixture["splits"].items():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.time()
            preds = predict_long(units, 64)
            torch.cuda.synchronize()
            sec = time.time() - started
            long_results[name] = {**score(units, preds), "docs": len(units),
                                  "sent_per_sec": sum(u["sentences"] for u in units) / sec,
                                  "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            print(line(name, long_results[name]) + f" | {long_results[name]['sent_per_sec']:.0f} sent/s, "
                  f"peak {long_results[name]['peak_gb']:.1f} GB", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "model"))
    tok.save_pretrained(str(args.out / "model"))
    (args.out / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    report = {"contender": "encoder-bio", "base": args.base, "epochs": args.epochs, "train_sec": train_sec,
              "gpu": torch.cuda.get_device_name(0), **speed, "results": results, "long_results": long_results}
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
