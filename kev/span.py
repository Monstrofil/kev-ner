"""Contender B — the kev/Jev decision-model architecture with the pointer aimed at the TEXT, not at options.

One packed sequence per unit: ``[document] [branch f1] [branch f2] ...``, a branch being
``<field> name: description <decide>`` straight from the user's schema. A block mask lets the document
attend causally to itself and each branch attend to the whole document plus only its own branch, and
every branch restarts its position ids right after the document — so all fields are answered in ONE
forward pass and adding a field costs only its branch tokens.

kev reads a ``choice`` by pointing the ``<decide>`` state at option markers. Here each ``<decide>`` state
points twice over the DOCUMENT tokens — a start pointer and an end pointer — plus one learned "null" slot
per pointer for "this field is not printed". The answer is always a real substring of the input (or
null): it cannot be malformed and cannot be invented. A per-TYPE normaliser then makes the typed value.

Backbone: a frozen decoder with LoRA (kev: Qwen2.5-0.5B, r16). Calibration: one temperature fitted on
the dev collections (kev's recipe), reported as ECE on eval.

    python -m kev.span fixtures/run-21.json --out results/kev --languages uk --dev tsrada.gov.ua uzmr.gov.ua
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

from kev.field_task import accuracy, align, ece, field_ok, normalise, plain, report, split

# Reserved Qwen tokens reused as structure markers (kev does the same); user text is scrubbed of them.
FIELD_OPEN, DECIDE = "<|fim_prefix|>", "<|fim_suffix|>"
IGNORE = -100


class Packed:
    """One unit as a packed sequence: document token ids + offsets, then one branch per field."""

    def __init__(self, unit: dict, fields: list[dict], tok, max_doc: int, languages: list[str], train: bool):
        self.text = plain(unit["text"]).replace("<|", "< |")
        enc = tok(self.text, return_offsets_mapping=True, add_special_tokens=False)
        self.doc_ids = enc["input_ids"][:max_doc]
        self.offsets = enc["offset_mapping"][:max_doc]
        self.fields = fields
        self.branches = [tok(f"\n{FIELD_OPEN}{f['name']}: {f['desc']}", add_special_tokens=False)["input_ids"]
                         + [tok.convert_tokens_to_ids(DECIDE)] for f in fields]
        # Targets: token index of the span's first/last token, len(doc) for null, IGNORE when the field is
        # unlabelled (a correction row) or gold names a value that is not printed / lies past the
        # truncation — unknowable, so it is not trained on.
        self.starts, self.ends = [], []
        for f in fields:
            value = unit["gold"][f["name"]] if train else None
            s = e = len(self.doc_ids)
            if train and f["name"] not in unit["labelled"]:
                s = e = IGNORE
            elif value is not None:
                span = align(f, value, self.text, languages)
                kept = span is not None and span[1] <= self.offsets[-1][1]
                toks = [i for i, (a, b) in enumerate(self.offsets) if kept and a < span[1] and b > span[0]]
                s, e = (toks[0], toks[-1]) if toks else (IGNORE, IGNORE)
            self.starts.append(s)
            self.ends.append(e)

    def span_text(self, s: int, e: int) -> str:
        return self.text[self.offsets[s][0]:self.offsets[e][1]]


def collate(batch: list[Packed], pad_id: int, dtype: torch.dtype) -> dict:
    """Pad to the longest packed sequence; build block masks, restarted positions, pointer targets."""
    n_fields = len(batch[0].fields)
    max_doc = max(len(p.doc_ids) for p in batch)
    lengths = [len(p.doc_ids) + sum(map(len, p.branches)) for p in batch]
    L = max(lengths)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    pos = torch.zeros((len(batch), L), dtype=torch.long)
    allow = torch.zeros((len(batch), L, L), dtype=torch.bool)
    doc_mask = torch.zeros((len(batch), max_doc), dtype=torch.bool)
    decide_at = torch.zeros((len(batch), n_fields), dtype=torch.long)
    starts = torch.full((len(batch), n_fields), IGNORE, dtype=torch.long)
    ends = torch.full((len(batch), n_fields), IGNORE, dtype=torch.long)
    for b, p in enumerate(batch):
        d = len(p.doc_ids)
        ids[b, :d] = torch.tensor(p.doc_ids)
        pos[b, :d] = torch.arange(d)
        allow[b, :d, :d] = torch.tril(torch.ones(d, d, dtype=torch.bool))
        doc_mask[b, :d] = True
        at = d
        for f, branch in enumerate(p.branches):
            n = len(branch)
            ids[b, at:at + n] = torch.tensor(branch)
            pos[b, at:at + n] = torch.arange(d, d + n)
            allow[b, at:at + n, :d] = True
            allow[b, at:at + n, at:at + n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
            decide_at[b, f] = at + n - 1
            at += n
        for i in range(at, L):  # padding rows attend to themselves only, so no row is fully masked
            allow[b, i, i] = True
        # Null is slot ``max_doc`` in the padded pointer logits; remap each unit's own null index to it.
        starts[b] = torch.tensor([max_doc if s == d else s for s in p.starts])
        ends[b] = torch.tensor([max_doc if e == d else e for e in p.ends])
    mask = torch.zeros(allow.shape, dtype=dtype).masked_fill(~allow, torch.finfo(dtype).min)[:, None]
    return {"input_ids": ids, "position_ids": pos, "attention_mask": mask, "doc_mask": doc_mask,
            "decide_at": decide_at, "starts": starts, "ends": ends}


class PointerHeads(nn.Module):
    """Start/end pointers from each ``<decide>`` state over the document states, plus a null slot."""

    def __init__(self, hidden: int, dim: int = 256):
        super().__init__()
        self.q = nn.Linear(hidden, 2 * dim)
        self.k = nn.Linear(hidden, 2 * dim)
        self.null = nn.Linear(hidden, 2)
        self.dim = dim

    def forward(self, h: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h = h.float()
        max_doc = batch["doc_mask"].shape[1]
        dec = torch.gather(h, 1, batch["decide_at"][..., None].expand(-1, -1, h.shape[-1]))  # B F H
        qs, qe = self.q(dec).split(self.dim, -1)
        ks, ke = self.k(h[:, :max_doc]).split(self.dim, -1)
        null_s, null_e = self.null(dec).unbind(-1)
        scale = math.sqrt(self.dim)
        neg = torch.finfo(torch.float32).min

        def logits(q, k, null):
            doc = torch.einsum("bfd,btd->bft", q, k) / scale
            doc = doc.masked_fill(~batch["doc_mask"][:, None], neg)
            return torch.cat([doc, null[..., None]], -1)  # B F (T+1); slot T is null

        return logits(qs, ks, null_s), logits(qe, ke, null_e)


class KevSpan(nn.Module):
    def __init__(self, base: str, rank: int, grad_ckpt: bool):
        super().__init__()
        backbone = AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if grad_ckpt:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.heads = PointerHeads(backbone.config.hidden_size)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                          attention_mask=batch["attention_mask"]).last_hidden_state
        return self.heads(h, batch)


def to(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def check_branch_isolation(model: KevSpan, packed: Packed, pad_id: int, device: str) -> None:
    """Editing field 0's branch must not move field 1's pointer logits — the block mask's whole promise.
    (Field 1 sits AFTER field 0, so plain causal attention would let it see the edit.)"""
    other = Packed.__new__(Packed)
    other.__dict__ = {**packed.__dict__, "branches": [packed.branches[0][-2::-1] + packed.branches[0][-1:],
                                                      *packed.branches[1:]]}
    with torch.no_grad():
        a = model(to(collate([packed], pad_id, torch.bfloat16), device))[0][0, 1]
        b = model(to(collate([other], pad_id, torch.bfloat16), device))[0][0, 1]
    drift = (a - b).abs().max().item()
    assert drift < 1e-2, f"field 1 logits moved by {drift} when only field 0's branch changed"
    print(f"branch isolation ok (max drift {drift:.2e})", flush=True)


def best_span(start: torch.Tensor, end: torch.Tensor, n_doc: int, max_span: int) -> tuple[int, int, float, float]:
    """(s, e, log p(span), log p(null)) for one field: the most probable span with e in [s, s+max_span)
    against the null answer, each as a joint start*end log-probability."""
    ls, le = start.log_softmax(-1), end.log_softmax(-1)
    null = (ls[-1] + le[-1]).item()
    s_idx = torch.arange(n_doc, device=start.device)
    band = (s_idx[None] >= s_idx[:, None]) & (s_idx[None] < s_idx[:, None] + max_span)
    joint = (ls[:n_doc, None] + le[None, :n_doc]).masked_fill(~band, float("-inf"))
    flat = joint.argmax().item()
    return flat // n_doc, flat % n_doc, joint.max().item(), null


def decide(packed: "Packed", cands: list[tuple], bias: float, languages: list[str]) -> tuple[dict, dict, dict]:
    """(typed prediction, span record, confidence) per field: the span wins when its log-probability plus
    ``bias`` beats null's. The bias is the one decision knob, picked on dev."""
    pred, span, conf = {}, {}, {}
    for field, (s, e, lp_span, lp_null) in zip(packed.fields, cands):
        if lp_span + bias > lp_null:
            text = packed.span_text(s, e)
            pred[field["name"]], conf[field["name"]] = normalise(field, text, languages), math.exp(lp_span)
        else:
            text = None
            pred[field["name"]], conf[field["name"]] = None, math.exp(lp_null)
        span[field["name"]] = [text, conf[field["name"]]]
    return pred, span, conf


def fit_temperature(logits: list[tuple[torch.Tensor, int]]) -> float:
    """One temperature minimising pointer NLL on the dev collections (kev's post-hoc recipe)."""
    temps = [0.5 + 0.1 * i for i in range(26)]
    nll = {t: sum(F.cross_entropy(l[None] / t, torch.tensor([y])).item() for l, y in logits) for t in temps}
    return min(nll, key=nll.get)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--languages", nargs="+", required=True, help="date-parser languages, e.g. uk")
    parser.add_argument("--dev", nargs="+", required=True, help="training collections held out for calibration")
    parser.add_argument("--max-doc", type=int, default=1536, help="document tokens kept (the rest is truncated)")
    parser.add_argument("--max-span", type=int, default=384, help="longest answer span, in tokens")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--cpu-latency", type=int, default=0, help="time N eval units on CPU after eval")
    parser.add_argument("--review", type=Path, help="adjudicated-gold file to also score against")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    fields = fixture["task"]["fields"]
    train, dev, ev = split(fixture, args.dev)
    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    pack = lambda units, train_: [Packed(u, fields, tok, args.max_doc, args.languages, train_) for u in units]
    train_p, dev_p, ev_p = pack(train, True), pack(dev, True), pack(ev, False)
    trainable = sum(s != IGNORE for p in train_p for s in p.starts)
    print(f"train {len(train)} / dev {len(dev)} / eval {len(ev)} units; {trainable} trainable field targets; "
          f"packed len p50 {sorted(len(p.doc_ids) + sum(map(len, p.branches)) for p in train_p)[len(train_p) // 2]}",
          flush=True)

    model = KevSpan(args.base, args.rank, args.grad_ckpt).to(device)
    model.heads.float()
    model.eval()
    check_branch_isolation(model, ev_p[0], pad_id, device)
    model.backbone.print_trainable_parameters()

    params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
              {"params": model.heads.parameters(), "lr": args.lr * 5}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train_p) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 5], total_steps=steps, pct_start=0.1)
    started, step = time.time(), 0
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train_p)
        total = 0.0
        for i in range(0, len(train_p), args.batch):
            batch = to(collate(train_p[i:i + args.batch], pad_id, torch.bfloat16), device)
            s_log, e_log = model(batch)
            loss = (F.cross_entropy(s_log.flatten(0, 1), batch["starts"].flatten(), ignore_index=IGNORE)
                    + F.cross_entropy(e_log.flatten(0, 1), batch["ends"].flatten(), ignore_index=IGNORE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            step += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / math.ceil(len(train_p) / args.batch):.4f}", flush=True)
    train_sec = time.time() - started
    model.eval()

    def logits_of(packed: list[Packed]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float]:
        started = time.time()
        out = []
        with torch.no_grad():
            for p in packed:
                s_log, e_log = model(to(collate([p], pad_id, torch.bfloat16), device))
                out.append((s_log[0].cpu(), e_log[0].cpu()))
        return out, (time.time() - started) / len(packed)

    # Calibrate: one temperature over the dev start/end pointer targets.
    dev_logits, _ = logits_of(dev_p)
    # Single-unit batches carry no padding, so a unit's logits are [doc tokens][null] and its own null
    # index len(doc) is the right target as-is.
    temp = fit_temperature([(logit[f], target)
                            for p, (s_log, e_log) in zip(dev_p, dev_logits) for f in range(len(fields))
                            for logit, target in ((s_log, p.starts[f]), (e_log, p.ends[f])) if target != IGNORE])

    def candidates(packed: list[Packed], logits: list[tuple[torch.Tensor, torch.Tensor]]) -> list[list[tuple]]:
        return [[best_span(s_log[f] / temp, e_log[f] / temp, len(p.doc_ids), args.max_span)
                 for f in range(len(fields))] for p, (s_log, e_log) in zip(packed, logits)]

    # Decide: the span-vs-null bias that maximises dev mean-field accuracy.
    dev_cands = candidates(dev_p, dev_logits)
    biases = [round(0.5 * i, 1) for i in range(-4, 21)]
    dev_acc = {b: accuracy(dev, [decide(p, c, b, args.languages)[0] for p, c in zip(dev_p, dev_cands)],
                           fields)["mean_field"] for b in biases}
    bias = max(dev_acc, key=dev_acc.get)
    print(f"temperature {temp:.2f}, span-vs-null bias {bias} (dev mean-field {dev_acc[bias]:.3f})", flush=True)

    ev_logits, sec_per_unit = logits_of(ev_p)
    decided = [decide(p, c, bias, args.languages) for p, c in zip(ev_p, candidates(ev_p, ev_logits))]
    preds, spans = [d[0] for d in decided], [d[1] for d in decided]
    confs = [d[2][f["name"]] for d in decided for f in fields]
    right = [field_ok(f["type"], u["gold"][f["name"]], p[f["name"]]) for u, p in zip(ev, preds) for f in fields]

    args.out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(args.out / "adapter"))
    torch.save(model.heads.state_dict(), args.out / "heads.pt")

    cpu_ms = None
    if args.cpu_latency:
        model.to("cpu").float()
        device = "cpu"
        started = time.time()
        with torch.no_grad():
            for p in ev_p[:args.cpu_latency]:
                model(collate([p], pad_id, torch.float32))
        cpu_ms = (time.time() - started) / args.cpu_latency * 1000

    extra = {"contender": "kev-span", "base": args.base, "rank": args.rank, "epochs": args.epochs,
             "max_doc": args.max_doc, "max_span": args.max_span, "temperature": temp, "null_bias": bias,
             "dev_mean_field": dev_acc[bias], "dev_collections": args.dev, "train_sec": train_sec,
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             "gpu_ms_per_unit": sec_per_unit * 1000, "cpu_ms_per_unit": cpu_ms, "ece": ece(confs, right)}
    report(args.out, fixture, ev, preds, fields, args.languages, extra, spans, args.review)


if __name__ == "__main__":
    main()
