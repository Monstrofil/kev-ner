"""kev-span plus user-declared conversion: a typed field's value is ASSEMBLED from parts the rules declare,
each a choice the model makes, instead of being parsed out of the printed span by a fixed normaliser.

A rules file (``--rules``) maps field name → parts, a gold pattern and an output template::

    {"parent_date": {"parts": {"day": "1-31", "month": "1-12", "year": "2000-2030"},
                     "gold": "(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})",
                     "output": "{year:04d}-{month:02d}-{day:02d}"}}

Each part is one more branch in the same packed pass, listing its options with a marker AFTER each
(``… 11<opt> 12<opt><decide>``, so under causal attention a marker has read its option): the ``<decide>``
state points at the markers or a null slot — kev's own choice readout. The value is the output template
over the chosen options. Gold parts come from the gold value through the ``gold`` pattern; a part the
pattern leaves out trains nothing. The span branches still train as in ``kev_span.py``, so one model reports
both readouts: the fixed normaliser over the span, and the composed parts.

    python kev_compose.py fixtures/run-21.json rules/run-21.json --out out/compose --languages uk \
        --dev tsrada.gov.ua uzmr.gov.ua
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

from common import accuracy, align, field_ok, plain, report, split
from kev_span import DECIDE, FIELD_OPEN, IGNORE, Packed, PointerHeads, best_span, decide, fit_temperature, to

OPTION = "<|fim_middle|>"


def options_of(spec: str | list) -> list[int]:
    """``"1-31"`` → 1..31; a list is taken as is."""
    if isinstance(spec, list):
        return spec
    lo, hi = map(int, spec.split("-"))
    return list(range(lo, hi + 1))


class Part:
    """One declared part of one field: its option values and the tokens of its branch."""

    def __init__(self, field: dict, name: str, spec: str | list, tok):
        self.field, self.name, self.options = field, name, options_of(spec)
        ids = tok(f"\n{FIELD_OPEN}{field['name']}.{name}: {field['desc']} Options:", add_special_tokens=False)["input_ids"]
        self.markers = []
        for value in self.options:
            ids += tok(f" {value}", add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids(OPTION)]
            self.markers.append(len(ids) - 1)
        self.ids = ids + [tok.convert_tokens_to_ids(DECIDE)]


def gold_parts(rule: dict, value) -> dict[str, int] | None:
    """The gold value split into parts by the rule's pattern; None when it does not match at all."""
    if value is None:
        return None
    m = re.fullmatch(rule["gold"], str(value))
    return {k: int(v) for k, v in m.groupdict().items() if v is not None} if m else {}


class ComposePacked(Packed):
    """kev-span's packed unit plus one branch per declared part, and each part's target option."""

    def __init__(self, unit: dict, fields: list[dict], parts: list[Part], rules: dict, tok, max_doc: int,
                 languages: list[str], train: bool):
        super().__init__(unit, fields, tok, max_doc, languages, train)
        self.parts = parts
        self.part_targets = []   # option index, len(options) for null, IGNORE when unknowable
        for part in parts:
            name = part.field["name"]
            if not train or name not in unit["labelled"]:
                self.part_targets.append(IGNORE)
                continue
            got = gold_parts(rules[name], unit["gold"][name])
            if got is None:
                self.part_targets.append(len(part.options))
            elif part.name in got and got[part.name] in part.options:
                self.part_targets.append(part.options.index(got[part.name]))
            else:
                self.part_targets.append(IGNORE)


def collate(batch: list[ComposePacked], pad_id: int, dtype: torch.dtype) -> dict:
    """As ``kev_span.collate``, with the part branches appended after the field branches."""
    n_fields, n_parts = len(batch[0].fields), len(batch[0].parts)
    n_opts = max(len(p.options) for p in batch[0].parts)
    max_doc = max(len(p.doc_ids) for p in batch)
    L = max(len(p.doc_ids) + sum(map(len, p.branches)) + sum(len(q.ids) for q in p.parts) for p in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    allow = torch.zeros((B, L, L), dtype=torch.bool)
    doc_mask = torch.zeros((B, max_doc), dtype=torch.bool)
    decide_at = torch.zeros((B, n_fields), dtype=torch.long)
    part_at = torch.zeros((B, n_parts), dtype=torch.long)
    opt_at = torch.zeros((B, n_parts, n_opts), dtype=torch.long)
    opt_mask = torch.zeros((B, n_parts, n_opts), dtype=torch.bool)
    starts = torch.full((B, n_fields), IGNORE, dtype=torch.long)
    ends = torch.full((B, n_fields), IGNORE, dtype=torch.long)
    part_y = torch.full((B, n_parts), IGNORE, dtype=torch.long)
    for b, p in enumerate(batch):
        d = len(p.doc_ids)
        ids[b, :d] = torch.tensor(p.doc_ids)
        pos[b, :d] = torch.arange(d)
        allow[b, :d, :d] = torch.tril(torch.ones(d, d, dtype=torch.bool))
        doc_mask[b, :d] = True
        at = d
        for k, branch in enumerate(p.branches + [q.ids for q in p.parts]):
            n = len(branch)
            ids[b, at:at + n] = torch.tensor(branch)
            pos[b, at:at + n] = torch.arange(d, d + n)
            allow[b, at:at + n, :d] = True
            allow[b, at:at + n, at:at + n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
            if k < n_fields:
                decide_at[b, k] = at + n - 1
            else:
                j = k - n_fields
                part_at[b, j] = at + n - 1
                opt_at[b, j, :len(p.parts[j].markers)] = at + torch.tensor(p.parts[j].markers)
                opt_mask[b, j, :len(p.parts[j].markers)] = True
            at += n
        for i in range(at, L):  # padding rows attend to themselves only, so no row is fully masked
            allow[b, i, i] = True
        starts[b] = torch.tensor([max_doc if s == d else s for s in p.starts])
        ends[b] = torch.tensor([max_doc if e == d else e for e in p.ends])
        # A part's null is slot n_opts in the padded choice logits; remap from its own option count.
        part_y[b] = torch.tensor([n_opts if y == len(q.options) else y for y, q in zip(p.part_targets, p.parts)])
    mask = torch.zeros(allow.shape, dtype=dtype).masked_fill(~allow, torch.finfo(dtype).min)[:, None]
    return {"input_ids": ids, "position_ids": pos, "attention_mask": mask, "doc_mask": doc_mask,
            "decide_at": decide_at, "starts": starts, "ends": ends,
            "part_at": part_at, "opt_at": opt_at, "opt_mask": opt_mask, "part_y": part_y}


class ChoiceHead(nn.Module):
    """Each part's ``<decide>`` state against its own option-marker states, plus a null slot."""

    def __init__(self, hidden: int, dim: int = 256):
        super().__init__()
        self.q, self.k, self.null = nn.Linear(hidden, dim), nn.Linear(hidden, dim), nn.Linear(hidden, 1)
        self.dim = dim

    def forward(self, h: torch.Tensor, batch: dict) -> torch.Tensor:
        h = h.float()
        B, P, K = batch["opt_at"].shape
        dec = torch.gather(h, 1, batch["part_at"][..., None].expand(-1, -1, h.shape[-1]))            # B P H
        opt = torch.gather(h, 1, batch["opt_at"].view(B, P * K)[..., None].expand(-1, -1, h.shape[-1]))
        logits = torch.einsum("bpd,bpkd->bpk", self.q(dec), self.k(opt).view(B, P, K, -1)) / math.sqrt(self.dim)
        logits = logits.masked_fill(~batch["opt_mask"], torch.finfo(torch.float32).min)
        return torch.cat([logits, self.null(dec)], -1)   # B P (K+1); slot K is null


class KevCompose(nn.Module):
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
        self.choice = ChoiceHead(backbone.config.hidden_size)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                          attention_mask=batch["attention_mask"]).last_hidden_state
        return (*self.heads(h, batch), self.choice(h, batch))


def compose(parts: list[Part], rules: dict, part_logits: torch.Tensor, bias: float) -> tuple[dict, dict]:
    """(value per ruled field, per-part record). A field is null when its parts' summed null
    log-probability beats their summed best-option log-probability plus ``bias`` (picked on dev)."""
    by_field: dict[str, list] = {}
    for part, logits in zip(parts, part_logits):
        # This part's own options, then the null slot (always last; the slots between are padding).
        lp = torch.cat([logits[:len(part.options)], logits[-1:]]).float().log_softmax(-1)
        best = int(lp[:-1].argmax())
        by_field.setdefault(part.field["name"], []).append(
            {"part": part.name, "option": part.options[best], "logp": lp[best].item(), "logp_null": lp[-1].item(),
             "top": [[part.options[i], math.exp(lp[i].item())] for i in lp[:-1].topk(3).indices.tolist()]})
    values = {}
    for name, got in by_field.items():
        if sum(g["logp_null"] for g in got) > sum(g["logp"] for g in got) + bias:
            values[name] = None
        else:
            values[name] = rules[name]["output"].format(**{g["part"]: g["option"] for g in got})
    return values, by_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("rules", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--max-doc", type=int, default=1536)
    parser.add_argument("--max-span", type=int, default=384)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    fields = fixture["task"]["fields"]
    train, dev, ev = split(fixture, args.dev)
    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    by_name = {f["name"]: f for f in fields}
    parts = [Part(by_name[name], part, spec, tok) for name, rule in rules.items() for part, spec in rule["parts"].items()]
    for q in parts:
        print(f"part {q.field['name']}.{q.name}: {len(q.options)} options, {len(q.ids)} branch tokens", flush=True)

    pack = lambda units, tr: [ComposePacked(u, fields, parts, rules, tok, args.max_doc, args.languages, tr) for u in units]
    train_p, dev_p, ev_p = pack(train, True), pack(dev, True), pack(ev, False)
    for q, j in zip(parts, range(len(parts))):
        ys = [p.part_targets[j] for p in train_p]
        print(f"  {q.field['name']}.{q.name}: {sum(y not in (IGNORE, len(q.options)) for y in ys)} option targets, "
              f"{sum(y == len(q.options) for y in ys)} null, {sum(y == IGNORE for y in ys)} ignored", flush=True)
    rule_only = {name: sum(u["gold"][name] is not None and name in u["labelled"]
                           and align(by_name[name], u["gold"][name], plain(u["text"]), args.languages) is None
                           for u in train) for name in rules}
    print(f"train values the fixed normaliser cannot find in the text (span target ignored): {rule_only}", flush=True)
    col = lambda ps, dtype=torch.bfloat16: collate(ps, pad_id, dtype)

    model = KevCompose(args.base, args.rank, args.grad_ckpt).to(device)
    model.heads.float()
    model.choice.float()
    model.eval()
    # Editing field 0's branch must move no part's choice logits.
    probe = ev_p[0]
    other = ComposePacked.__new__(ComposePacked)
    other.__dict__ = {**probe.__dict__, "branches": [probe.branches[0][-2::-1] + probe.branches[0][-1:], *probe.branches[1:]]}
    with torch.no_grad():
        a, b = model(to(col([probe]), device))[2], model(to(col([other]), device))[2]
    drift = (a - b).abs().max().item()
    assert drift < 1e-2, f"part logits moved by {drift} when only field 0's branch changed"
    print(f"branch isolation ok (max drift {drift:.2e})", flush=True)

    params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
              {"params": [*model.heads.parameters(), *model.choice.parameters()], "lr": args.lr * 5}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train_p) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 5], total_steps=steps, pct_start=0.1)
    started = time.time()
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train_p)
        total = 0.0
        for i in range(0, len(train_p), args.batch):
            batch = to(col(train_p[i:i + args.batch]), device)
            s_log, e_log, c_log = model(batch)
            loss = (F.cross_entropy(s_log.flatten(0, 1), batch["starts"].flatten(), ignore_index=IGNORE)
                    + F.cross_entropy(e_log.flatten(0, 1), batch["ends"].flatten(), ignore_index=IGNORE))
            if (batch["part_y"] != IGNORE).any():
                loss = loss + F.cross_entropy(c_log.flatten(0, 1), batch["part_y"].flatten(), ignore_index=IGNORE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / math.ceil(len(train_p) / args.batch):.4f}", flush=True)
    train_sec = time.time() - started
    model.eval()

    def run(packed: list[ComposePacked]) -> tuple[list, float]:
        out, started = [], time.time()
        with torch.no_grad():
            for p in packed:
                s_log, e_log, c_log = model(to(col([p]), device))
                out.append((s_log[0].cpu(), e_log[0].cpu(), c_log[0].cpu()))
        return out, (time.time() - started) / len(packed)

    dev_out, _ = run(dev_p)
    temp = fit_temperature([(logit[f], target) for p, (s, e, _) in zip(dev_p, dev_out) for f in range(len(fields))
                            for logit, target in ((s, p.starts[f]), (e, p.ends[f])) if target != IGNORE])
    cands = lambda ps, outs: [[best_span(s[f] / temp, e[f] / temp, len(p.doc_ids), args.max_span)
                               for f in range(len(fields))] for p, (s, e, _) in zip(ps, outs)]
    dev_cands = cands(dev_p, dev_out)
    grid = [round(0.5 * i, 1) for i in range(-4, 21)]
    span_acc = {b: accuracy(dev, [decide(p, c, b, args.languages)[0] for p, c in zip(dev_p, dev_cands)], fields)["mean_field"]
                for b in grid}
    span_bias = max(span_acc, key=span_acc.get)
    ruled = [f for f in fields if f["name"] in rules]
    comp_acc = {b: accuracy(dev, [compose(parts, rules, c, b)[0] for _, _, c in dev_out], ruled)["mean_field"] for b in grid}
    comp_bias = max(comp_acc, key=comp_acc.get)
    print(f"temperature {temp:.2f}, span bias {span_bias}, compose bias {comp_bias}", flush=True)

    def readouts(ps: list[ComposePacked], outs: list) -> tuple[list[dict], list[dict], list[dict]]:
        """(fixed-normaliser predictions, composed predictions, per-unit part records)."""
        span_preds = [decide(p, c, span_bias, args.languages)[0] for p, c in zip(ps, cands(ps, outs))]
        composed, records = [], []
        for pred, (_, _, c) in zip(span_preds, outs):
            values, rec = compose(parts, rules, c, comp_bias)
            composed.append({**pred, **values})
            records.append(rec)
        return span_preds, composed, records

    ev_out, sec = run(ev_p)
    comparison, dumps = {}, {}
    for split_name, units, ps, outs in (("dev", dev, dev_p, dev_out), ("eval", ev, ev_p, ev_out)):
        span_preds, composed, records = readouts(ps, outs)
        rows = {}
        for f in ruled:
            n = f["name"]
            hard = [i for i, u in enumerate(units) if u["gold"][n] is not None
                    and align(f, u["gold"][n], plain(u["text"]), args.languages) is None]
            ok = lambda preds, idx: sum(field_ok(f["type"], units[i]["gold"][n], preds[i][n]) for i in idx)
            everyone = range(len(units))
            rows[n] = {"n": len(units), "fixed": ok(span_preds, everyone), "composed": ok(composed, everyone),
                       "unfindable_n": len(hard), "unfindable_fixed": ok(span_preds, hard),
                       "unfindable_composed": ok(composed, hard)}
            print(f"{split_name:4} {n:15} fixed {rows[n]['fixed']:3}/{len(units)}  composed {rows[n]['composed']:3}/{len(units)}"
                  f"  | rule-unfindable {len(hard)}: fixed {rows[n]['unfindable_fixed']} composed {rows[n]['unfindable_composed']}",
                  flush=True)
        comparison[split_name] = rows
        dumps[split_name] = [{"id": u["id"][:12], "collection": u["collection"],
                              "gold": {f["name"]: u["gold"][f["name"]] for f in ruled},
                              "fixed": {f["name"]: s[f["name"]] for f in ruled},
                              "composed": {f["name"]: c[f["name"]] for f in ruled}, "parts": r}
                             for u, s, c, r in zip(units, span_preds, composed, records)]
        if split_name == "eval":
            ev_span, ev_comp = span_preds, composed

    args.out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(args.out / "adapter"))
    torch.save({"heads": model.heads.state_dict(), "choice": model.choice.state_dict()}, args.out / "heads.pt")
    (args.out / "parts.json").write_text(json.dumps(dumps, ensure_ascii=False, indent=1), encoding="utf-8")
    extra = {"contender": "kev-compose", "base": args.base, "rules": rules, "temperature": temp, "span_bias": span_bias,
             "compose_bias": comp_bias, "train_sec": train_sec, "gpu_ms_per_unit": sec * 1000,
             "fixed_readout_eval": accuracy(ev, ev_span, fields), "comparison": comparison,
             "packed_len_p50": sorted(len(p.doc_ids) + sum(map(len, p.branches)) + sum(len(q.ids) for q in parts)
                                      for p in ev_p)[len(ev_p) // 2]}
    report(args.out, fixture, ev, ev_comp, fields, args.languages, extra,
           [{f["name"]: [c[f["name"]], None] for f in fields} for c in ev_comp], args.review)


if __name__ == "__main__":
    main()
