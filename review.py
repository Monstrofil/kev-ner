"""Print each not-yet-adjudicated mismatch of a report: gold, prediction, baseline, and the source around them.

    python review.py out/run2/kev-1.5b.json adjudications/run-21.json [field ...]
"""

import json
import sys
from collections import Counter

from common import plain

report, review, only = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3:]
units = {u["id"]: u for u in json.load(open("fixtures/run-21.json"))["units"]}
done = {(e["id_prefix"], e["suffix"], e["field"]) for e in review["fields"]}
errors = [e for e in report["errors"] if (e["id"][:12], e["id"][64:], e["field"]) not in done
          and (not only or e["field"] in only)]
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
