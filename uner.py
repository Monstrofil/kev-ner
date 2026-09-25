"""Universal NER English (``universalner/uner_llm_inst_english``) as a flat multi-span NER fixture + scorer.

The HF rows are LLM instructions; the sentence is the text after the prompt's last question and the
target lists ``{TypeName, Text, Start, End}`` in character offsets. Splits: EWT train / dev / test
(web text) and PUD test (news + Wikipedia, a different source — the cross-source number).

    python uner.py data/uner fixtures/uner-en.json      # jsonl files → one compact fixture
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# The schema is the dataset's own instruction wording, one entry per type.
TYPES = [
    {"name": "PER", "desc": "person"},
    {"name": "ORG", "desc": "organization; organizations can represent other groups of people; "
                            "nationalities are not organizations"},
    {"name": "LOC", "desc": "location; nationalities are not locations"},
]
FILES = {"train": "uner-en_ewt-train.jsonl", "dev": "uner-en_ewt-dev.jsonl",
         "ewt_test": "uner-en_ewt-test.jsonl", "pud_test": "uner-en_pud-test.jsonl"}
QUESTION = "what is the output result?\n\n"


def convert(path: Path) -> tuple[list[dict], int]:
    """(sentences, entities dropped): an entity whose Text is null was lost by the dataset's own
    conversion (tokens like ``U$``), so it has no offsets to score against."""
    units, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        text = row["inputs"].split(QUESTION, 1)[1].removesuffix("\n")
        entities = []
        for e in json.loads("{" + row["targets"] + "}")["Results"]:
            if e["Text"] is None:
                dropped += 1
                continue
            assert text[e["Start"]:e["End"]] == e["Text"], (text, e)
            entities.append([e["Start"], e["End"], e["TypeName"]])
        units.append({"text": text, "entities": entities})
    return units, dropped


def score(units: list[dict], preds: list[list[list]]) -> dict:
    """Entity-level micro P/R/F1: a prediction counts only with the exact character span AND type."""
    tp, n_pred, n_gold = Counter(), Counter(), Counter()
    for unit, pred in zip(units, preds, strict=True):
        gold = {tuple(e) for e in unit["entities"]}
        guess = {tuple(e) for e in pred}
        for kind, bag in (("pred", guess), ("gold", gold)):
            for e in bag:
                (n_pred if kind == "pred" else n_gold)[e[2]] += 1
        for e in gold & guess:
            tp[e[2]] += 1

    def prf(t: int, p: int, g: int) -> dict:
        precision, recall = t / p if p else 0.0, t / g if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"p": precision, "r": recall, "f1": f1, "gold": g}

    out = {"micro": prf(sum(tp.values()), sum(n_pred.values()), sum(n_gold.values()))}
    out.update({t["name"]: prf(tp[t["name"]], n_pred[t["name"]], n_gold[t["name"]]) for t in TYPES})
    return out


def line(name: str, s: dict) -> str:
    return (f"{name:9} F1 {s['micro']['f1']:.4f} (P {s['micro']['p']:.4f} R {s['micro']['r']:.4f}, "
            f"{s['micro']['gold']} gold) | " + " ".join(f"{t['name']} {s[t['name']]['f1']:.3f}" for t in TYPES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    fixture = {"types": TYPES, "splits": {}}
    for split, name in FILES.items():
        units, dropped = convert(args.src / name)
        fixture["splits"][split] = units
        print(f"{split}: {len(units)} sentences, {sum(len(u['entities']) for u in units)} entities, "
              f"{dropped} dropped (null text)")
    args.out.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
