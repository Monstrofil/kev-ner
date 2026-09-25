"""What every span bake-off contender shares: the fixture split, gold->text alignment, per-TYPE
normalisers, the substrate's field scorer, calibration and the report.

Nothing here knows what a field means — dispatch is on the declared field TYPE, and the one
language-dependent piece (date parsing) takes its languages as an argument.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import dateparser
import roman
from rapidfuzz import fuzz

from scoring import field_ok

MARKUP = re.compile(r"<[^>]+>|\*+|#+|\|")
NUMERIC_DATE = re.compile(r"\b\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b")
WORD_DATE = re.compile(r"\b\d{1,2}\s+\w+\s+\d{4}\b")
INTEGER = re.compile(r"\b\d+\b")
ROMAN = re.compile(r"\b[IVXLCDMІХМС]+\b")
# OCR renders roman numerals with look-alike letters from other scripts; fold them before parsing.
ROMAN_FOLD = str.maketrans({"І": "I", "Х": "X", "М": "M", "С": "C"})
FUZZY_MIN = 90


def plain(text: str) -> str:
    """Markup-free text: HTML tags and markdown emphasis/heading/table marks become spaces."""
    return " ".join(MARKUP.sub(" ", text).split())


def split(fixture: dict, dev_collections: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """(train, dev, eval): eval is the run's held-out collections; dev is whole TRAINING collections.
    Training keeps partially labelled rows (trainers supervise only ``labelled``); dev and eval score
    every field, so they keep fully labelled rows only."""
    n_fields = len(fixture["task"]["fields"])
    full = lambda u: len(u["labelled"]) == n_fields
    units = fixture["units"]
    dev = [u for u in units if u["split"] == "train" and u["collection"] in dev_collections and full(u)]
    train = [u for u in units if u["split"] == "train" and u["collection"] not in dev_collections]
    assert dev, f"no training units in dev collections {dev_collections}"
    return train, dev, [u for u in units if u["split"] == "eval" and full(u)]


def adjudicated(ev: list[dict], path: Path) -> tuple[list[dict], list[set[str]]]:
    """(eval units with reviewed gold substituted, per-unit fields excluded from scoring)."""
    review = json.loads(path.read_text(encoding="utf-8"))
    key = lambda e: (e["id_prefix"], e["suffix"])
    dropped = {key(e) for e in review["exclude_units"]}
    fixes = {}
    for e in review["fields"]:
        fixes.setdefault(key(e), []).append(e)
    units, excluded = [], []
    for u in ev:
        k = (u["id"][:12], u["id"][64:])
        if k in dropped:
            continue
        gold, skip = dict(u["gold"]), set()
        for e in fixes.get(k, []):
            if e.get("exclude"):
                skip.add(e["field"])
            else:
                gold[e["field"]] = e["adjudicated"]
        units.append({**u, "gold": gold})
        excluded.append(skip)
    return units, excluded


# ---- per-TYPE normalisers: printed span -> typed value ----------------------------------------------

def as_date(span: str, languages: list[str]) -> str | None:
    parsed = dateparser.parse(span, languages=languages, settings={"DATE_ORDER": "DMY", "STRICT_PARSING": True})
    return parsed.date().isoformat() if parsed else None


def as_number(span: str) -> int | None:
    span = re.sub(r"[^\w]", "", span)
    if span.isdigit():
        return int(span)
    try:
        return roman.fromRoman(span.translate(ROMAN_FOLD).upper())
    except roman.InvalidRomanNumeralError:
        return None


def normalise(field: dict, span: str, languages: list[str]):
    if field["type"] == "date":
        return as_date(span, languages)
    if field["type"] == "number":
        return as_number(span)
    return " ".join(span.split())


def align(field: dict, value, text: str, languages: list[str]) -> tuple[int, int] | None:
    """First char span of ``text`` that normalises to ``value``; None if the value is not printed."""
    if field["type"] == "date":
        for pattern in (NUMERIC_DATE, WORD_DATE):
            for m in pattern.finditer(text):
                if as_date(m.group(), languages) == value:
                    return m.span()
        return None
    if field["type"] == "number":
        for pattern in (INTEGER, ROMAN):
            for m in pattern.finditer(text):
                if as_number(m.group()) == value:
                    return m.span()
        return None
    needle = plain(str(value)).casefold()
    hay = text.casefold()
    at = hay.find(needle)
    if at >= 0:
        return at, at + len(needle)
    hit = fuzz.partial_ratio_alignment(needle, hay)
    return (hit.dest_start, hit.dest_end) if hit.score >= FUZZY_MIN else None


# ---- scoring ---------------------------------------------------------------------------------------

def accuracy(units: list[dict], preds: list[dict], fields: list[dict], excluded: list[set[str]] | None = None) -> dict:
    """Per-field accuracy, its mean, and the share of units with every scored field right."""
    excluded = excluded or [set()] * len(units)
    ok = [{f["name"]: field_ok(f["type"], u["gold"][f["name"]], p[f["name"]]) for f in fields if f["name"] not in skip}
          for u, p, skip in zip(units, preds, excluded)]
    per = {f["name"]: sum(row[f["name"]] for row in ok if f["name"] in row) / sum(f["name"] in row for row in ok)
           for f in fields}
    return {"fields": per, "mean_field": sum(per.values()) / len(per),
            "unit_exact": sum(all(row.values()) for row in ok) / len(ok)}


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy| averaged over equal-width confidence bins."""
    total, err = len(confidences), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if idx:
            err += len(idx) / total * abs(sum(confidences[i] for i in idx) / len(idx)
                                          - sum(correct[i] for i in idx) / len(idx))
    return err


def ceiling(units: list[dict], fields: list[dict], languages: list[str]) -> dict:
    """The tagger ceiling: a field whose gold is not printed in the text is unreachable by extraction."""
    preds = [{f["name"]: u["gold"][f["name"]] if u["gold"][f["name"]] is None
              or align(f, u["gold"][f["name"]], plain(u["text"]), languages) else "<unprintable>"
              for f in fields} for u in units]
    return accuracy(units, preds, fields)


def report(out: Path, fixture: dict, ev: list[dict], preds: list[dict], fields: list[dict],
           languages: list[str], extra: dict, spans: list[dict], review: Path | None = None) -> None:
    """Write ``report.json`` and print the per-field table vs the generative baseline and the ceiling,
    on the Studio gold and — given a review file — on the adjudicated gold."""
    ours = accuracy(ev, preds, fields)
    base = accuracy(ev, [u["baseline"] for u in ev], fields)
    ceil = ceiling(ev, fields, languages)
    adj = None
    if review:
        by_id = dict(zip([u["id"] for u in ev], preds))
        adj_ev, skip = adjudicated(ev, review)
        adj = {"n_units": len(adj_ev),
               "span_model": accuracy(adj_ev, [by_id[u["id"]] for u in adj_ev], fields, skip),
               "baseline": accuracy(adj_ev, [u["baseline"] for u in adj_ev], fields, skip)}
    per_collection = defaultdict(list)
    for u, p in zip(ev, preds):
        per_collection[u["collection"]].append((u, p))
    body = {
        **extra,
        "span_model": ours, "baseline": {"model": fixture["baseline"]["model"], **base}, "tagger_ceiling": ceil,
        "adjudicated": adj,
        "per_collection_mean_field": {c: accuracy([u for u, _ in r], [p for _, p in r], fields)["mean_field"]
                                      for c, r in per_collection.items()},
        "errors": [{"id": u["id"], "field": f["name"], "gold": u["gold"][f["name"]], "pred": p[f["name"]],
                    "span": s.get(f["name"])}
                   for u, p, s in zip(ev, preds, spans) for f in fields
                   if not field_ok(f["type"], u["gold"][f["name"]], p[f["name"]])],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = [("span", ours), ("base", base), ("ceil", ceil)]
    if adj:
        cols += [("adj-span", adj["span_model"]), ("adj-base", adj["baseline"])]
    print("\n" + f"{'field':16}" + "".join(f"{name:>9}" for name, _ in cols))
    for n in [f["name"] for f in fields]:
        print(f"{n:16}" + "".join(f"{c['fields'][n]:9.3f}" for _, c in cols))
    print(f"{'MEAN FIELD':16}" + "".join(f"{c['mean_field']:9.3f}" for _, c in cols))
    print(f"{'UNIT EXACT':16}" + "".join(f"{c['unit_exact']:9.3f}" for _, c in cols))
    print(json.dumps({k: v for k, v in extra.items()}, ensure_ascii=False))
    print("per collection:", body["per_collection_mean_field"], flush=True)
