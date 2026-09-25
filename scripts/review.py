"""Print each not-yet-adjudicated mismatch of a report: gold, prediction, baseline, and the source around them.

    python -m scripts.review results/kev-1.5b/report.json adjudications/run-21.json [field ...]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from kev.field_task import plain

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("fields", nargs="*", help="only these fields (default: every field)")
    parser.add_argument("--fixture", type=Path, default=ROOT / "fixtures" / "run-21.json")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    units = {u["id"]: u for u in json.loads(args.fixture.read_text(encoding="utf-8"))["units"]}
    done = {(e["id_prefix"], e["suffix"], e["field"]) for e in review["fields"]}
    errors = [e for e in report["errors"] if (e["id"][:12], e["id"][64:], e["field"]) not in done
              and (not args.fields or e["field"] in args.fields)]
    print(Counter(e["field"] for e in errors))
    for e in errors:
        u = units[e["id"]]
        text = plain(u["text"])
        span, conf = e["span"] or [None, None]
        print(f"\n## {e['field']}  {u['collection']}  {e['id'][:12]}{e['id'][64:]}")
        print(f"  gold: {e['gold']!r}\n  pred: {e['pred']!r}  (span {span!r}, p={conf:.2f})")
        print(f"  base: {u['baseline'][e['field']]!r}")
        print(f"  head: {text[:380]!r}")
        at = text.find(span) if span else -1
        if at > 380:
            print(f"  @pred: {text[max(0, at - 150):at + len(span) + 80]!r}")


if __name__ == "__main__":
    main()
