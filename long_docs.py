"""Long-document variant of the UNER test sets: consecutive sentences joined into documents of about N
tokens, gold offsets shifted with them. Every length holds the same sentences and entities, so F1 is
comparable across lengths; only how much text a model reads at once changes.

    python long_docs.py fixtures/uner-en.json fixtures/uner-long.json --lengths 512 1024 2048 4096
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterator
from pathlib import Path

from transformers import AutoTokenizer

SEP = "\n"


def join(units: list[dict], sizes: list[int], budgets: Iterator[int]) -> list[dict]:
    """Greedy: keep adding the next sentence (``sizes``: its tokens) while the document stays within its
    budget; each new document takes the next budget."""
    docs, text, entities, used, n = [], "", [], 0, 0
    budget = next(budgets)
    for u, size in zip(units, sizes, strict=True):
        if text and used + size > budget:
            docs.append({"text": text, "entities": entities, "sentences": n})
            text, entities, used, n = "", [], 0, 0
            budget = next(budgets)
        at = len(text) + (len(SEP) if text else 0)
        text = f"{text}{SEP}{u['text']}" if text else u["text"]
        entities += [[s + at, e + at, kind] for s, e, kind in u["entities"]]
        used, n = used + size, n + 1
    docs.append({"text": text, "entities": entities, "sentences": n})
    return docs


def sizes_of(units: list[dict], tok) -> list[int]:
    return [len(tok(SEP + u["text"], add_special_tokens=False)["input_ids"]) for u in units]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base", help="the budget is counted in its tokens")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    splits = {}
    for split in ("ewt_test", "pud_test"):
        sizes = sizes_of(fixture["splits"][split], tok)
        for n in args.lengths:
            docs = join(fixture["splits"][split], sizes, itertools.repeat(n))
            for d in docs:
                assert all(d["text"][s:e].strip() for s, e, _ in d["entities"])
            splits[f"{split}@{n}"] = docs
            print(f"{split}@{n}: {len(docs)} docs, {sum(d['sentences'] for d in docs)} sentences, "
                  f"{sum(len(d['entities']) for d in docs)} entities", flush=True)
    args.out.write_text(json.dumps({"types": fixture["types"], "splits": splits}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
