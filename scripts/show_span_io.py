"""Dump what a trained kev-span model reads and emits for chosen units, as JSON (for the results page).

    python -m scripts.show_span_io fixtures/run-21.json results/kev-1.5b --base Qwen/Qwen2.5-1.5B \
        --temperature 2.5 --bias -1.0 --units 03d0cef0f946 0908aaa1501d:2 04cef6491219:1 > results/span-io.json

A unit is picked by id prefix, optionally ``:annex_no`` when one document has several annexes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from kev.field_task import field_ok
from kev.span import KevSpan, Packed, PointerHeads, best_span, collate, decide, to


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--bias", type=float, required=True)
    parser.add_argument("--units", nargs="+", required=True)
    parser.add_argument("--languages", nargs="+", default=["uk"])
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    fields = fixture["task"]["fields"]
    tok = AutoTokenizer.from_pretrained(args.base)
    model = KevSpan.__new__(KevSpan)
    torch.nn.Module.__init__(model)
    backbone = AutoModel.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.backbone = PeftModel.from_pretrained(backbone, str(args.run / "adapter"))
    model.heads = PointerHeads(backbone.config.hidden_size)
    model.heads.load_state_dict(torch.load(args.run / "heads.pt", weights_only=True))
    model = model.cuda().eval()
    model.heads.float()

    out = []
    for spec in args.units:
        prefix, _, annex = spec.partition(":")
        unit = next(u for u in fixture["units"] if u["id"].startswith(prefix)
                    and (not annex or u["gold"]["annex_no"] == annex))
        p = Packed(unit, fields, tok, 1536, args.languages, train=False)
        batch = collate([p], tok.pad_token_id, torch.bfloat16)
        with torch.no_grad():
            s_log, e_log = model(to(batch, "cuda"))
        s_log, e_log = s_log[0].float().cpu() / args.temperature, e_log[0].float().cpu() / args.temperature
        d = len(p.doc_ids)
        cands = [best_span(s_log[f], e_log[f], d, 384) for f in range(len(fields))]
        pred, _, conf = decide(p, cands, args.bias, args.languages)
        rows = []
        for f, field in enumerate(fields):
            ls, le = s_log[f].log_softmax(-1), e_log[f].log_softmax(-1)
            top_start = [{"token": i, "text": tok.decode(p.doc_ids[i]), "p": math.exp(ls[i].item())}
                         for i in ls[:d].topk(3).indices.tolist()]
            s, e, lp_span, lp_null = cands[f]
            rows.append({
                "field": field["name"], "type": field["type"], "branch": f"{field['name']}: {field['desc']}",
                "branch_tokens": len(p.branches[f]),
                "top_start": top_start, "p_start_null": math.exp(ls[d].item()), "p_end_null": math.exp(le[d].item()),
                "best_span": {"start_token": s, "end_token": e, "text": p.span_text(s, e), "logp": lp_span},
                "logp_null": lp_null, "span_wins": lp_span + args.bias > lp_null,
                "value": pred[field["name"]], "confidence": conf[field["name"]],
                "gold": unit["gold"][field["name"]],
                "ok": field_ok(field["type"], unit["gold"][field["name"]], pred[field["name"]]),
            })
        out.append({"id": unit["id"][:12], "collection": unit["collection"], "chars": len(p.text),
                    "doc_tokens": d, "packed_tokens": d + sum(map(len, p.branches)),
                    "logit_shape": [len(fields), d + 1], "text": p.text[:1400], "fields": rows})
    print(json.dumps({"temperature": args.temperature, "bias": args.bias, "units": out}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
