"""Zero- and few-shot kev-ner: train once on Pile-NER's open types, then read datasets whose types it never
saw (``scripts/zs_data.py``), with the type names as the only schema.

- **zero-shot**: the Pile-trained model on each test set; its threshold is picked on held-out Pile passages,
  never on a test set.
- **few-shot**: from the Pile-trained weights, fine-tuned on the dataset's 50-sentence sample, then tested.
- **few-shot, no pretraining**: the same 50 sentences on a fresh LoRA and head, threshold 0 (p = 0.5),
  so what Pile-NER buys is visible.

A unit's branches are its own types; a batch pads to its widest type list and masks the padding.

    python -m kev.zero_shot fixtures/zs.json --out results/zs --base Qwen/Qwen3-4B-Base --keep-layers 24 --bidir-doc
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from kev.ner import KevNer, Packed, by_tokens, collate, decode, to


def micro(units: list[dict], preds: list[list]) -> dict:
    """Entity-level micro P/R/F1: exact character span and type."""
    tp = n_pred = n_gold = 0
    for u, p in zip(units, preds, strict=True):
        gold, guess = {tuple(e) for e in u["entities"]}, {tuple(e) for e in p}
        tp, n_pred, n_gold = tp + len(gold & guess), n_pred + len(guess), n_gold + len(gold)
    precision, recall = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"p": precision, "r": recall, "f1": f1, "gold": n_gold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--keep-layers", type=int, default=0)
    parser.add_argument("--bidir-doc", action="store_true")
    parser.add_argument("--max-doc", type=int, default=512)
    parser.add_argument("--max-width", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--pile-limit", type=int, default=0, help="train on the first N Pile passages (0 = all)")
    parser.add_argument("--few-epochs", type=int, default=10)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    schema = lambda names: [{"name": t, "desc": t} for t in names]
    pack = lambda units: [Packed(u, schema(u["types"]), tok, args.max_doc, args.max_width) for u in units]

    def col(ps: list[Packed]) -> dict:
        batch = collate(ps, pad_id, torch.bfloat16, max(len(p.branches) for p in ps), args.max_width, args.bidir_doc)
        batch["type_mask"] = torch.tensor([[t < len(p.branches) for t in range(batch["decide_at"].shape[1])]
                                           for p in ps])
        return batch

    model = KevNer(args.base, args.rank, args.max_width, args.grad_ckpt, args.keep_layers).to(device)
    model.head.float()
    fresh = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k or "head" in k})

    def train(ps: list[Packed], epochs: int, lr: float, batch_n: int) -> float:
        """AdamW + OneCycle over ``epochs`` passes; batches by padded tokens, length-sorted within blocks."""
        size = lambda p: len(p.doc_ids) + sum(map(len, p.branches))
        plan = []
        for _ in range(epochs):
            order = list(ps)
            random.shuffle(order)
            order = [p for i in range(0, len(order), 256) for p in sorted(order[i:i + 256], key=size)]
            batches = by_tokens(order, batch_n, args.batch_tokens)
            random.shuffle(batches)
            plan.append(batches)
        params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr},
                  {"params": model.head.parameters(), "lr": lr * 5}]
        opt = torch.optim.AdamW(params, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 5], total_steps=sum(map(len, plan)),
                                                    pct_start=0.1)
        started = time.time()
        model.train()
        for epoch, batches in enumerate(plan):
            total = 0.0
            for i, chunk in enumerate(batches):
                batch = to(col(chunk), device)
                logits = model(batch)
                valid = (batch["valid"][:, None] & batch["type_mask"][..., None, None]).expand_as(logits)
                loss = F.binary_cross_entropy_with_logits(logits[valid], batch["target"][valid], reduction="sum")
                (loss / logits.shape[0]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item() / logits.shape[0]
                if len(batches) > 500 and (i + 1) % 500 == 0:
                    print(f"  step {i + 1}/{len(batches)} loss {total / (i + 1):.4f} ({time.time() - started:.0f}s)",
                          flush=True)
            print(f"epoch {epoch + 1}/{epochs} loss {total / len(batches):.4f}", flush=True)
        model.eval()
        return time.time() - started

    def predict(ps: list[Packed], threshold: float) -> tuple[list[list], float]:
        preds, started = [], time.time()
        with torch.no_grad():
            for chunk in by_tokens(ps, 64, args.batch_tokens):
                logits = model(to(col(chunk), device)).cpu()
                preds += [decode(p, logits[j, :len(p.branches), :len(p.doc_ids)], threshold, schema(p.unit["types"]))
                          for j, p in enumerate(chunk)]
        return preds, time.time() - started

    # 1. Pile-NER.
    pile_train = fixture["pile"]["train"][:args.pile_limit] if args.pile_limit else fixture["pile"]["train"]
    train_p, dev_p = pack(pile_train), pack(fixture["pile"]["dev"])
    print(f"pile: {len(train_p)} passages, {sum(p.unreachable for p in train_p)} unreachable of "
          f"{sum(len(p.unit['entities']) for p in train_p)}, packed p50 "
          f"{sorted(len(p.doc_ids) + sum(map(len, p.branches)) for p in train_p)[len(train_p) // 2]}", flush=True)
    pile_sec = train(train_p, 1, args.lr, 64)
    dev_units = [p.unit for p in dev_p]
    grid = [round(-3 + 0.25 * i, 2) for i in range(25)]
    with torch.no_grad():
        dev_logits = []
        for chunk in by_tokens(dev_p, 64, args.batch_tokens):
            logits = model(to(col(chunk), device)).cpu()
            dev_logits += [logits[j, :len(p.branches), :len(p.doc_ids)] for j, p in enumerate(chunk)]
    dev_f1 = {th: micro(dev_units, [decode(p, l, th, schema(p.unit["types"])) for p, l in zip(dev_p, dev_logits)])["f1"]
              for th in grid}
    threshold = max(dev_f1, key=dev_f1.get)
    print(f"pile trained in {pile_sec:.0f}s; threshold {threshold} (pile dev F1 {dev_f1[threshold]:.4f})", flush=True)
    pile_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k or "head" in k})
    args.out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(args.out / "adapter"))
    torch.save(model.head.state_dict(), args.out / "head.pt")

    # 2. Each unseen dataset: zero-shot, few-shot from Pile, few-shot from nothing.
    results, examples = {}, {}
    for name, ds in fixture["eval"].items():
        test_p, few_p = pack(ds["test"]), pack(ds["few"])
        units = [p.unit for p in test_p]
        row = {"types": ds["types"], "test_sentences": len(units), "few_sentences": len(few_p)}
        model.load_state_dict(pile_state, strict=False)
        preds, sec = predict(test_p, threshold)
        row["zero_shot"] = {**micro(units, preds), "sent_per_sec": len(units) / sec}
        examples[name] = [{"text": u["text"], "gold": u["entities"], "pred": p} for u, p in zip(units[:5], preds[:5])]
        train(few_p, args.few_epochs, args.lr / 2, 8)
        row["few_shot"] = micro(units, predict(test_p, threshold)[0])
        model.load_state_dict(fresh, strict=False)
        train(few_p, args.few_epochs, args.lr, 8)
        row["few_shot_no_pretrain"] = micro(units, predict(test_p, 0.0)[0])
        results[name] = row
        print(f"{name:15} zero-shot F1 {row['zero_shot']['f1']:.3f} (P {row['zero_shot']['p']:.3f} R "
              f"{row['zero_shot']['r']:.3f}) | few-shot {row['few_shot']['f1']:.3f} | "
              f"few-shot no pretrain {row['few_shot_no_pretrain']['f1']:.3f}", flush=True)
    avg = {k: sum(r[k]["f1"] for r in results.values()) / len(results)
           for k in ("zero_shot", "few_shot", "few_shot_no_pretrain")}
    print("average F1: " + ", ".join(f"{k} {v:.3f}" for k, v in avg.items()), flush=True)
    report = {"contender": "kev-zs", "base": args.base, "keep_layers": args.keep_layers, "bidir_doc": args.bidir_doc,
              "pile_passages": len(train_p), "pile_train_sec": pile_sec, "threshold": threshold,
              "pile_dev_f1": dev_f1[threshold], "gpu": torch.cuda.get_device_name(0),
              "average_f1": avg, "results": results}
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
