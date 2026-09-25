"""kev-span for flat multi-entity NER: every type answered in ONE forward pass, any number of spans each.

Same packing as ``kev_span.py``: ``[document] [branch t1] [branch t2] ...`` with a branch per entity type
(``<field> name: description <decide>``), a block mask so each branch sees the document and only
itself, and branch positions restarting after the document. The readout changes from "one span or
null" to "a set of spans": each ``<decide>`` state scores EVERY document span of up to ``--max-width``
tokens (a span is represented by its first token, last token, the token after it — the one-token
lookahead a causal document otherwise lacks — and its width). Spans above a threshold picked on dev
are kept greedily by score, overlaps dropped (the data is flat), across all types at once.

``--bidir-doc`` lets document tokens attend to the whole document instead of causally (LLM2Vec-style;
LoRA adapts the backbone to it). Branches are unchanged either way.

``--keep-layers N`` runs only the backbone's first N decoder layers (the top ones mostly serve next-token
prediction, which a span readout does not need). ``--examples K`` appends each type's K most frequent
training mentions to its description; ``--train-limit`` trains on a seeded subset, to see where that helps.

``--long-train MAX`` also trains on documents: each epoch adds the training sentences joined afresh into
documents of log-uniform length in [256, MAX] tokens, batched by ``--batch-tokens``; the threshold is then
picked on dev sentences and dev documents together.

    python kev_ner.py fixtures/uner-en.json --out out/kev-ner
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

from long_docs import join, sizes_of
from uner import line, score

FIELD_OPEN, DECIDE = "<|fim_prefix|>", "<|fim_suffix|>"


class Packed:
    """One sentence as document token ids + offsets, one branch per type, and its gold token spans."""

    def __init__(self, unit: dict, types: list[dict], tok, max_doc: int, max_width: int):
        self.unit = unit
        self.text = unit["text"].replace("<|", "< |")
        enc = tok(self.text, return_offsets_mapping=True, add_special_tokens=False)
        self.doc_ids = enc["input_ids"][:max_doc]
        self.offsets = enc["offset_mapping"][:max_doc]
        self.branches = [tok(f"\n{FIELD_OPEN}{t['name']}: {t['desc']}", add_special_tokens=False)["input_ids"]
                         + [tok.convert_tokens_to_ids(DECIDE)] for t in types]
        index = {t["name"]: i for i, t in enumerate(types)}
        # (type, first token, width - 1); an entity whose edges fall inside a token, past the truncation,
        # or wider than max_width is unreachable and simply absent from the targets.
        self.gold, self.unreachable = [], 0
        for start, end, kind in unit["entities"]:
            toks = [i for i, (a, b) in enumerate(self.offsets) if a < end and b > start]
            if toks and self.char_span(toks[0], toks[-1]) == (start, end) and toks[-1] - toks[0] < max_width:
                self.gold.append((index[kind], toks[0], toks[-1] - toks[0]))
            else:
                self.unreachable += 1

    def char_span(self, s: int, e: int) -> tuple[int, int]:
        """Character span of tokens s..e with the byte-level BPE's leading whitespace trimmed."""
        a, b = self.offsets[s][0], self.offsets[e][1]
        return a + len(self.text[a:b]) - len(self.text[a:b].lstrip()), b


def collate(batch: list[Packed], pad_id: int, dtype: torch.dtype, n_types: int, max_width: int,
            bidir_doc: bool) -> dict:
    max_doc = max(len(p.doc_ids) for p in batch)
    L = max(len(p.doc_ids) + sum(map(len, p.branches)) for p in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    allow = torch.zeros((B, L, L), dtype=torch.bool)
    decide_at = torch.zeros((B, n_types), dtype=torch.long)
    doc_mask = torch.zeros((B, max_doc), dtype=torch.bool)
    valid = torch.zeros((B, max_doc, max_width), dtype=torch.bool)   # span (i, w) lies inside the document
    target = torch.zeros((B, n_types, max_doc, max_width))
    for b, p in enumerate(batch):
        d = len(p.doc_ids)
        ids[b, :d] = torch.tensor(p.doc_ids)
        pos[b, :d] = torch.arange(d)
        doc_mask[b, :d] = True
        allow[b, :d, :d] = True if bidir_doc else torch.tril(torch.ones(d, d, dtype=torch.bool))
        at = d
        for t, branch in enumerate(p.branches):
            n = len(branch)
            ids[b, at:at + n] = torch.tensor(branch)
            pos[b, at:at + n] = torch.arange(d, d + n)
            allow[b, at:at + n, :d] = True
            allow[b, at:at + n, at:at + n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
            decide_at[b, t] = at + n - 1
            at += n
        for i in range(at, L):  # padding rows attend to themselves only, so no row is fully masked
            allow[b, i, i] = True
        i = torch.arange(max_doc)[:, None]
        valid[b] = (i + torch.arange(max_width)[None]) < d
        for t, s, w in p.gold:
            target[b, t, s, w] = 1.0
    mask = torch.zeros(allow.shape, dtype=dtype).masked_fill(~allow, torch.finfo(dtype).min)[:, None]
    return {"input_ids": ids, "position_ids": pos, "attention_mask": mask, "decide_at": decide_at,
            "doc_mask": doc_mask, "valid": valid, "target": target}


class SpanHead(nn.Module):
    """score(type t, span i..i+w) = q_t · MLP(start_i + end_{i+w} + next_{i+w+1} + width_w) + bias_t."""

    def __init__(self, hidden: int, max_width: int, dim: int = 256):
        super().__init__()
        self.start, self.end, self.next = (nn.Linear(hidden, dim) for _ in range(3))
        self.width = nn.Embedding(max_width, dim)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(dim, dim))
        self.q = nn.Linear(hidden, dim)
        self.bias = nn.Linear(hidden, 1)
        nn.init.constant_(self.bias.bias, -6.0)   # prior: almost no span is an entity
        self.max_width, self.dim = max_width, dim

    def forward(self, h: torch.Tensor, batch: dict) -> torch.Tensor:
        h = h.float()
        T = batch["valid"].shape[1]
        doc = h[:, :T] * batch["doc_mask"][..., None]   # a shorter sentence's tail reads zeros, as at batch 1
        pad = lambda x, k: F.pad(x, (0, 0, 0, k))  # zero rows past the end, for end/next lookups
        start, end = self.start(doc), pad(self.end(doc), self.max_width)
        nxt = pad(self.next(doc), self.max_width + 1)
        idx = torch.arange(T, device=h.device)[:, None] + torch.arange(self.max_width, device=h.device)[None]
        rep = start[:, :, None] + end[:, idx] + nxt[:, idx + 1] + self.width.weight[None, None]   # B T W D
        rep = self.mlp(rep)
        dec = torch.gather(h, 1, batch["decide_at"][..., None].expand(-1, -1, h.shape[-1]))      # B F H
        logits = torch.einsum("bfd,btwd->bftw", self.q(dec), rep) / math.sqrt(self.dim)
        return logits + self.bias(dec)[..., None]


class KevNer(nn.Module):
    def __init__(self, base: str, rank: int, max_width: int, grad_ckpt: bool, keep_layers: int = 0):
        super().__init__()
        backbone = AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if keep_layers:
            assert keep_layers <= backbone.config.num_hidden_layers
            backbone.layers = backbone.layers[:keep_layers]
            backbone.config.num_hidden_layers = keep_layers
        if grad_ckpt:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.head = SpanHead(backbone.config.hidden_size, max_width)

    def forward(self, batch: dict) -> torch.Tensor:
        h = self.backbone(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                          attention_mask=batch["attention_mask"]).last_hidden_state
        return self.head(h, batch)


def to(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def decode(p: Packed, logits: torch.Tensor, threshold: float, types: list[dict]) -> list[list]:
    """Spans above ``threshold`` (a logit), best first, dropping any that overlaps one already kept."""
    n = len(p.doc_ids)
    hits = (logits > threshold).nonzero().tolist()
    hits = sorted(((logits[t, s, w].item(), t, s, w) for t, s, w in hits if s + w < n), reverse=True)
    taken, out = set(), []
    for _, t, s, w in hits:
        toks = set(range(s, s + w + 1))
        if toks & taken:
            continue
        taken |= toks
        out.append([*p.char_span(s, s + w), types[t]["name"]])
    return out


def by_tokens(ps: list[Packed], max_n: int, max_tokens: int) -> list[list[Packed]]:
    """Consecutive batches of at most ``max_n`` units and ``max_tokens`` padded tokens (at least one unit)."""
    out, cur, longest = [], [], 0
    for p in ps:
        n = len(p.doc_ids) + sum(map(len, p.branches))
        if cur and (len(cur) + 1 > max_n or (len(cur) + 1) * max(longest, n) > max_tokens):
            out.append(cur)
            cur, longest = [], 0
        cur.append(p)
        longest = max(longest, n)
    return out + [cur]


def log_uniform(lo: int, hi: int):
    while True:
        yield round(math.exp(random.uniform(math.log(lo), math.log(hi))))


def with_examples(types: list[dict], train: list[dict], k: int) -> list[dict]:
    """Each type's description plus its ``k`` most frequent training mentions."""
    counts = {t["name"]: Counter() for t in types}
    for unit in train:
        for start, end, kind in unit["entities"]:
            counts[kind][unit["text"][start:end]] += 1
    return [{**t, "desc": f"{t['desc']}; e.g. " + ", ".join(m for m, _ in counts[t["name"]].most_common(k))}
            for t in types]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--bidir-doc", action="store_true")
    parser.add_argument("--max-doc", type=int, default=256)
    parser.add_argument("--max-width", type=int, default=16, help="longest entity, in tokens")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--keep-layers", type=int, default=0, help="first N decoder layers only (0 = all)")
    parser.add_argument("--examples", type=int, default=0, help="training mentions appended per description")
    parser.add_argument("--train-limit", type=int, default=0, help="train on a seeded subset of N sentences")
    parser.add_argument("--long-train", type=int, default=0, help="also train on joined documents up to N tokens")
    parser.add_argument("--batch-tokens", type=int, default=16384, help="padded tokens per batch")
    parser.add_argument("--cpu-latency", type=int, default=0, help="time N test sentences on CPU, batch 1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if args.train_limit:
        fixture["splits"]["train"] = random.sample(fixture["splits"]["train"], args.train_limit)
    types = fixture["types"]
    if args.examples:
        types = with_examples(types, fixture["splits"]["train"], args.examples)
    for t in types:
        print(f"{t['name']}: {t['desc']}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    max_doc = max(args.max_doc, args.long_train)
    pack = lambda units: [Packed(u, types, tok, max_doc, args.max_width) for u in units]
    packed = {name: pack(units) for name, units in fixture["splits"].items()}
    sizes = {name: sizes_of(fixture["splits"][name], tok) for name in ("train", "dev")} if args.long_train else {}
    if args.long_train:
        packed["dev_docs"] = pack(join(fixture["splits"]["dev"], sizes["dev"], log_uniform(256, args.long_train)))
    for name, ps in packed.items():
        gold = sum(len(p.unit["entities"]) for p in ps)
        print(f"{name}: {len(ps)} sentences, {gold} entities, {sum(p.unreachable for p in ps)} unreachable "
              f"(token-boundary / width), packed len p50 "
              f"{sorted(len(p.doc_ids) + sum(map(len, p.branches)) for p in ps)[len(ps) // 2]}", flush=True)
    col = lambda ps, dtype=torch.bfloat16: collate(ps, pad_id, dtype, len(types), args.max_width, args.bidir_doc)

    model = KevNer(args.base, args.rank, args.max_width, args.grad_ckpt, args.keep_layers).to(device)
    model.head.float()
    model.backbone.print_trainable_parameters()

    # Editing type 0's branch must not move type 1's span scores — the block mask's whole promise.
    probe = next(p for p in packed["dev"] if p.gold)
    other = Packed.__new__(Packed)
    other.__dict__ = {**probe.__dict__, "branches": [probe.branches[0][-2::-1] + probe.branches[0][-1:],
                                                     *probe.branches[1:]]}
    model.eval()
    with torch.no_grad():
        drift = (model(to(col([probe]), device))[0, 1] - model(to(col([other]), device))[0, 1]).abs().max().item()
    assert drift < 1e-2, f"type 1 scores moved by {drift} when only type 0's branch changed"
    print(f"branch isolation ok (max drift {drift:.2e})", flush=True)

    def epoch_batches() -> list[list[Packed]]:
        """This epoch's batches. Plain: the shuffled sentences, ``--batch`` at a time. With --long-train, the
        sentences plus them joined afresh into documents, length-sorted within shuffled blocks of 256 so a
        batch pads little, and cut by ``--batch-tokens``."""
        units = list(packed["train"])
        random.shuffle(units)
        if not args.long_train:
            return [units[i:i + args.batch] for i in range(0, len(units), args.batch)]
        units += pack(join(fixture["splits"]["train"], sizes["train"], log_uniform(256, args.long_train)))
        random.shuffle(units)
        size = lambda p: len(p.doc_ids) + sum(map(len, p.branches))
        units = [p for i in range(0, len(units), 256) for p in sorted(units[i:i + 256], key=size)]
        batches = by_tokens(units, args.batch, args.batch_tokens)
        random.shuffle(batches)
        return batches

    plan = [epoch_batches() for _ in range(args.epochs)]
    print(f"{sum(map(len, plan))} steps over {args.epochs} epochs", flush=True)
    params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
              {"params": model.head.parameters(), "lr": args.lr * 5}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = sum(map(len, plan))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 5], total_steps=steps, pct_start=0.1)
    started = time.time()
    model.train()
    for epoch, batches in enumerate(plan):
        total = 0.0
        for chunk in batches:
            batch = to(col(chunk), device)
            logits = model(batch)
            valid = batch["valid"][:, None].expand_as(logits)
            # Summed per sentence, not averaged over the ~T*W*F mostly-negative spans.
            loss = F.binary_cross_entropy_with_logits(logits[valid], batch["target"][valid], reduction="sum")
            loss = loss / logits.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(batches):.4f}", flush=True)
    train_sec = time.time() - started
    model.eval()

    def logits_of(ps: list[Packed], batch: int) -> list[torch.Tensor]:
        out = []
        with torch.no_grad():
            for chunk in by_tokens(ps, batch, args.batch_tokens):
                logits = model(to(col(chunk), device)).cpu()
                out.extend(logits[j, :, :len(p.doc_ids)] for j, p in enumerate(chunk))
        return out

    tune = packed["dev"] + (packed["dev_docs"] if args.long_train else [])
    dev_logits = logits_of(tune, 64)
    grid = [round(-3 + 0.25 * i, 2) for i in range(25)]
    dev_f1 = {th: score([p.unit for p in tune],
                        [decode(p, l, th, types) for p, l in zip(tune, dev_logits)])["micro"]["f1"]
              for th in grid}
    threshold = max(dev_f1, key=dev_f1.get)
    print(f"threshold {threshold} (dev F1 {dev_f1[threshold]:.4f})", flush=True)

    results, predictions = {}, {}
    for name in ("dev", *(["dev_docs"] if args.long_train else []), "ewt_test", "pud_test"):
        ps = packed[name]
        preds = [decode(p, l, threshold, types) for p, l in zip(ps, logits_of(ps, 64))]
        results[name] = score([p.unit for p in ps], preds)
        predictions[name] = preds
        print(line(name, results[name]), flush=True)

    # Throughput: batched (how a corpus is run) and batch-1 latency, both forward pass + decode.
    test = packed["ewt_test"] + packed["pud_test"]
    speed = {}
    for batch in (1, 64):
        torch.cuda.synchronize()
        started = time.time()
        for p, l in zip(test, logits_of(test, batch)):
            decode(p, l, threshold, types)
        torch.cuda.synchronize()
        speed[f"gpu_sent_per_sec_b{batch}"] = len(test) / (time.time() - started)
    if args.cpu_latency:
        model.to("cpu").float()
        with torch.no_grad():
            started = time.time()
            for p in test[:args.cpu_latency]:
                model(col([p], torch.float32))
        speed["cpu_ms_per_sent_b1"] = (time.time() - started) / args.cpu_latency * 1000
    print(json.dumps(speed), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(args.out / "adapter"))
    torch.save(model.head.state_dict(), args.out / "head.pt")
    (args.out / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    ceiling = {name: 1 - sum(p.unreachable for p in packed[name]) / sum(len(p.unit["entities"]) for p in packed[name])
               for name in ("ewt_test", "pud_test")}
    report = {"contender": "kev-ner", "base": args.base, "bidir_doc": args.bidir_doc, "rank": args.rank,
              "keep_layers": args.keep_layers, "examples": args.examples, "train_limit": args.train_limit,
              "long_train": args.long_train,
              "types": types,
              "epochs": args.epochs, "max_width": args.max_width, "threshold": threshold, "train_sec": train_sec,
              "gpu": torch.cuda.get_device_name(0), "recall_ceiling": ceiling, **speed, "results": results}
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
