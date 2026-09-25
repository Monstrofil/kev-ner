"""Zero-/few-shot NER fixture: train on Pile-NER (open types, LLM-labelled), test on datasets whose types it
never trained on — the CrossNER domains and MIT movie / restaurant, the benchmark GLiNER and UniversalNER
report on.

Every unit carries its own ``types`` (the branches it is asked). Pile-NER answers are strings, so each
becomes every whole-word occurrence in its passage; each passage also asks a few types it does not
contain, drawn from the other passages, so "nothing of this type" is trained too.

    python zs_data.py data/zs fixtures/zs.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

QUESTION = re.compile(r"^What describes (.+) in the text\?$")
NEGATIVES, MAX_TYPES, DEV, FEW = 5, 25, 500, 50
# Label spellings of the test sets → the words a type prompt uses.
READABLE = {
    "programlang": "programming language", "literarygenre": "literary genre", "musicalartist": "musical artist",
    "musicalinstrument": "musical instrument", "musicgenre": "music genre", "politicalparty": "political party",
    "academicjournal": "academic journal", "astronomicalobject": "astronomical object",
    "chemicalcompound": "chemical compound", "chemicalelement": "chemical element", "misc": "miscellaneous",
    "organisation": "organization", "ratings_average": "average rating", "restaurant_name": "restaurant name",
}
EVAL = {  # name: (test file, train file, column of the tag)
    **{f"crossner_{d}": (f"crossner-{d}-test.txt", f"crossner-{d}-train.txt", 1)
       for d in ("ai", "literature", "music", "politics", "science")},
    "mit_movie": ("mit-engtest.bio", "mit-engtrain.bio", 0),
    "mit_restaurant": ("mit-restauranttest.bio", "mit-restauranttrain.bio", 0),
}


def readable(label: str) -> str:
    label = label.lower()
    return READABLE.get(label, label.replace("_", " "))


def pile(path: Path, rng: random.Random) -> list[dict]:
    units = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        turns = row["conversations"]
        text = turns[0]["value"].removeprefix("Text: ")
        asked = {}
        for q, a in zip(turns[2::2], turns[3::2]):
            kind = QUESTION.match(q["value"])
            try:
                mentions = json.loads(a["value"])
            except json.JSONDecodeError:
                continue
            if kind and isinstance(mentions, list):
                asked[kind.group(1).strip().lower()] = [m for m in mentions if isinstance(m, str) and m.strip()]
        entities = set()
        for kind, mentions in asked.items():
            for m in mentions:
                for hit in re.finditer(rf"(?<!\w){re.escape(m)}(?!\w)", text):
                    entities.add((hit.start(), hit.end(), kind))
        units.append({"text": text, "entities": sorted(entities, key=lambda e: e[:2]), "types": list(asked)})
    pool = Counter(t for u in units for t in u["types"])
    names, weights = list(pool), list(pool.values())
    for u in units:
        negatives = {t for t in rng.choices(names, weights, k=NEGATIVES * 2) if t not in u["types"]}
        u["types"] = (u["types"] + sorted(negatives)[:NEGATIVES])[:MAX_TYPES]
        u["entities"] = [list(e) for e in u["entities"] if e[2] in u["types"]]
    return units


def bio(path: Path, column: int) -> list[dict]:
    """CoNLL-style BIO, one ``token<TAB>tag`` (or ``tag<TAB>token``) per line, blank line between sentences."""
    units, words, tags = [], [], []
    for raw in path.read_text(encoding="utf-8").splitlines() + [""]:
        cols = raw.split("\t")
        if len(cols) == 2:
            words.append(cols[1 - column])
            tags.append(cols[column])
            continue
        if not words:
            continue
        text, entities, at = "", [], []
        for w in words:
            at.append(len(text) + (1 if text else 0))
            text = f"{text} {w}" if text else w
        for i, tag in enumerate(tags):
            if tag.startswith("B-") or (tag.startswith("I-") and (i == 0 or tags[i - 1][2:] != tag[2:])):
                entities.append([at[i], at[i] + len(words[i]), readable(tag[2:])])
            elif tag.startswith("I-"):
                entities[-1][1] = at[i] + len(words[i])
        units.append({"text": text, "entities": entities})
        words, tags = [], []
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rng = random.Random(0)
    train = pile(args.src / "pile-ner.json", rng)
    rng.shuffle(train)
    fixture = {"pile": {"train": train[DEV:], "dev": train[:DEV]}, "eval": {}}
    print(f"pile-ner: {len(train)} passages, {len({t for u in train for t in u['types']})} types, "
          f"{sum(len(u['entities']) for u in train)} entity occurrences", flush=True)
    for name, (test_file, train_file, column) in EVAL.items():
        test, pool = bio(args.src / test_file, column), bio(args.src / train_file, column)
        types = sorted({e[2] for u in test + pool for e in u["entities"]})
        few = random.Random(0).sample(pool, FEW)
        for u in test + few:
            u["types"] = types
        fixture["eval"][name] = {"types": types, "test": test, "few": few}
        print(f"{name}: {len(test)} test sentences, {sum(len(u['entities']) for u in test)} entities, "
              f"{len(types)} types: {', '.join(types)}", flush=True)
    args.out.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
