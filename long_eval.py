"""Run a trained kev-ner model over the long-document fixture, one document per pass, and score it.

The model, its threshold and its settings come from the run's own ``report.json``; nothing is retrained.
Throughput is original sentences per second, so it compares directly with the per-sentence numbers.

    python long_eval.py fixtures/uner-long.json out/ner-q3b-keep24 --out out/long-keep24
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from kev_ner import KevNer, Packed, SpanHead, collate, decode, to
from uner import line, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    args = parser.parse_args()

    trained = json.loads((args.run / "report.json").read_text())
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    types = fixture["types"]
    tok = AutoTokenizer.from_pretrained(trained["base"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    backbone = AutoModel.from_pretrained(trained["base"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if trained["keep_layers"]:
        backbone.layers = backbone.layers[:trained["keep_layers"]]
        backbone.config.num_hidden_layers = trained["keep_layers"]
    model = KevNer.__new__(KevNer)
    torch.nn.Module.__init__(model)
    model.backbone = PeftModel.from_pretrained(backbone, str(args.run / "adapter"))
    model.head = SpanHead(backbone.config.hidden_size, trained["max_width"])
    model.head.load_state_dict(torch.load(args.run / "head.pt", weights_only=True))
    model = model.cuda().eval()
    model.head.float()

    results = {}
    for name, units in fixture["splits"].items():
        ps = [Packed(u, types, tok, 1 << 20, trained["max_width"]) for u in units]
        longest = max(len(p.doc_ids) + sum(map(len, p.branches)) for p in ps)
        batch = max(1, args.tokens_per_batch // longest)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.time()
        preds = []
        with torch.no_grad():
            for i in range(0, len(ps), batch):
                chunk = ps[i:i + batch]
                logits = model(to(collate(chunk, pad_id, torch.bfloat16, len(types), trained["max_width"],
                                          trained["bidir_doc"]), "cuda")).cpu()
                preds += [decode(p, logits[j, :, :len(p.doc_ids)], trained["threshold"], types)
                          for j, p in enumerate(chunk)]
        torch.cuda.synchronize()
        sec = time.time() - started
        results[name] = {**score(units, preds), "docs": len(units), "doc_tokens_max": longest, "batch": batch,
                         "unreachable": sum(p.unreachable for p in ps),
                         "sent_per_sec": sum(u["sentences"] for u in units) / sec,
                         "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        print(line(name, results[name]) + f" | {results[name]['sent_per_sec']:.0f} sent/s, batch {batch}, "
              f"peak {results[name]['peak_gb']:.1f} GB", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"contender": "kev-ner-long", "run": str(args.run), "base": trained["base"],
              "keep_layers": trained["keep_layers"], "sentence_results": trained["results"], "results": results}
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
