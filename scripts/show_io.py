"""Print what a trained kev-ner model actually consumes and emits for one sentence — no JSON anywhere.

    python -m scripts.show_io results/ner-kev-q3-4b-bidir --base Qwen/Qwen3-4B-Base --bidir-doc --threshold -2.0 \
        --text "Angela Merkel met Microsoft executives in Berlin."
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from kev.ner import KevNer, Packed, SpanHead, collate, decode, to


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--bidir-doc", action="store_true")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--json", action="store_true", help="emit every text's dump as one JSON document")
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/uner-en.json"))
    args = parser.parse_args()

    types = json.loads(args.fixture.read_text(encoding="utf-8"))["types"]
    tok = AutoTokenizer.from_pretrained(args.base)
    model = KevNer.__new__(KevNer)
    torch.nn.Module.__init__(model)
    backbone = AutoModel.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.backbone = PeftModel.from_pretrained(backbone, str(args.run / "adapter"))
    model.head = SpanHead(backbone.config.hidden_size, 16)
    model.head.load_state_dict(torch.load(args.run / "head.pt", weights_only=True))
    model = model.cuda().eval()
    model.head.float()
    dumps = [dump(text, types, tok, model, args) for text in args.text]
    if args.json:
        print(json.dumps(dumps, ensure_ascii=False, indent=1))


def dump(text: str, types: list[dict], tok, model: KevNer, args: argparse.Namespace) -> dict:
    """Print (unless --json) and return one sentence's packed input, mask, raw span logits and decode."""
    show = (lambda *a: None) if args.json else print
    p = Packed({"text": text, "entities": []}, types, tok, 256, 16)
    batch = collate([p], tok.pad_token_id, torch.bfloat16, len(types), 16, args.bidir_doc)
    d = len(p.doc_ids)
    parts = ["doc"] * d + [f"branch {t['name']}" for t, br in zip(types, p.branches) for _ in br]
    ids = p.doc_ids + [t for br in p.branches for t in br]
    tokens = [{"k": k, "pos": batch["position_ids"][0, k].item(), "id": t, "piece": tok.convert_ids_to_tokens(t),
               "part": part} for k, (t, part) in enumerate(zip(ids, parts))]

    show("=== 1. INPUT: one packed token sequence ===")
    for r in tokens:
        show(f"{r['k']:3} pos={r['pos']:3}  id={r['id']:6}  {r['piece']!r:28} {r['part']}")
    show(f"\n<decide> positions (read by the head): {batch['decide_at'][0].tolist()}")

    allow = batch["attention_mask"][0, 0] == 0
    marks = "".join("D" if part == "doc" else part.split()[1][0] for part in parts)
    mask = ["".join("#" if allow[k, j] else "." for j in range(len(ids))) for k in range(len(ids))]
    show("\n=== 2. ATTENTION MASK (row attends to column; # = allowed) ===")
    show("      " + marks)
    for k, row in enumerate(mask):
        show(f"{marks[k]} {k:3} {row}")

    with torch.no_grad():
        logits = model(to(batch, "cuda"))[0, :, :d].float().cpu()
    # Spans running past the sentence end are never trained or decoded; blank them for display.
    logits = logits.masked_fill(~batch["valid"][0, :d][None], float("-inf"))
    top = []
    flat = logits.flatten().topk(12)
    for v, ix in zip(flat.values, flat.indices):
        t, s, w = (ix // (d * 16)).item(), (ix // 16 % d).item(), (ix % 16).item()
        a, b = p.char_span(s, s + w)
        top.append({"logit": v.item(), "p": torch.sigmoid(v).item(), "type": types[t]["name"],
                    "tokens": [s, s + w], "chars": [a, b], "text": text[a:b]})
    show(f"\n=== 3. RAW OUTPUT: one float tensor, shape {tuple(logits.shape)} = [types, span start token, span width-1] ===")
    show(f"{int(batch['valid'][0].sum()) * len(types)} real span scores (logits). Top 12 of them:")
    for r in top:
        show(f"  logit {r['logit']:+7.2f}  p={r['p']:.3f}  type={r['type']}  tokens {r['tokens'][0]}..{r['tokens'][1]}  "
             f"chars {r['chars'][0]}..{r['chars'][1]}  {r['text']!r}")

    out = decode(p, logits, args.threshold, types)
    show(f"\n=== 4. DECODE (plain Python): keep logit > {args.threshold}, best first, drop overlaps ===")
    show(out)
    show("\nas [char_start, char_end, type] → text:")
    for a, b, t in out:
        show(f"  {t}: {text[a:b]!r}")
    return {"text": text, "tokens": tokens, "decide_at": batch["decide_at"][0].tolist(), "marks": marks,
            "mask": mask, "shape": list(logits.shape), "real_spans": int(batch["valid"][0].sum()) * len(types),
            "top": top, "threshold": args.threshold, "decoded": out,
            "logits": [[[round(v, 2) if math.isfinite(v) else None for v in row] for row in plane]
                       for plane in logits.tolist()]}


if __name__ == "__main__":
    main()
