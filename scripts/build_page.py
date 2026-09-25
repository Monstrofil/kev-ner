"""Render the shareable results + raw-I/O page from the run outputs (no hand-copied numbers).

    python -m scripts.build_page > docs/results.html

Reads results/ner-*/report.json + predictions.json, results/{kev-0.5b,kev-1.5b,gliner-ft}/report.json,
results/ner-io.json (show_io.py --json), results/span-io.json (show_span_io.py),
results/llm-format-example.json, fixtures/uner-en.json, fixtures/run-21.json and docs/page.css.
"""

from __future__ import annotations

import json
import random
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))

NER_RUNS = [  # (run, label, highlight)
    ("ner-kev-q3-4b-bidir", "kev · Qwen3-4B-Base · bidirectional document", True),
    ("ner-kev-q3-4b", "kev · Qwen3-4B-Base · causal document", False),
    ("ner-kev-1.5b", "kev · Qwen2.5-1.5B · causal", False),
    ("ner-kev-1.5b-bidir", "kev · Qwen2.5-1.5B · bidirectional", False),
    ("ner-kev-0.5b-bidir", "kev · Qwen2.5-0.5B · bidirectional", False),
    ("ner-kev-0.5b", "kev · Qwen2.5-0.5B · causal", False),
    ("ner-roberta-large", "RoBERTa-large BIO tagger (fine-tuned reference)", False),
]
H100_PER_HOUR = 3.95
TYPES = ["PER", "ORG", "LOC"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def piece(p: str) -> str:
    return p.replace("Ġ", "␣").replace("Ċ", "⏎")


def span_text(tokens: list[dict], s: int, e: int) -> str:
    return "".join(t["piece"] for t in tokens[s:e + 1]).replace("Ġ", " ").replace("Ċ", "\n").strip()


def marked(text: str, spans: list[list], cls: str = "") -> str:
    """Sentence with [start, end, type] spans wrapped in <mark>."""
    out, at = [], 0
    for a, b, t in sorted(spans):
        out.append(escape(text[at:a]))
        out.append(f'<mark class="{cls} t-{t}">{escape(text[a:b])}<sub>{t}</sub></mark>')
        at = b
    out.append(escape(text[at:]))
    return "".join(out)


fx = load(ROOT / "fixtures" / "uner-en.json")
reports = {r: load(OUT / r / "report.json") for r, _, _ in NER_RUNS}
preds = {r: load(OUT / r / "predictions.json") for r in ("ner-kev-q3-4b-bidir", "ner-roberta-large")}
io = load(OUT / "ner-io.json")
span_io = load(OUT / "span-io.json")
llm = load(OUT / "llm-format-example.json")
r21 = {n: load(OUT / n / "report.json") for n in ("kev-1.5b", "kev-0.5b", "gliner-ft")}
win = reports["ner-kev-q3-4b-bidir"]
rob = reports["ner-roberta-large"]


def bootstrap(a: str, b: str, split: str, n: int = 1000) -> tuple[float, float, float]:
    from kev.uner import score
    units = fx["splits"][split]
    pa, pb = load(OUT / a / "predictions.json")[split], load(OUT / b / "predictions.json")[split]
    rng = random.Random(0)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(len(units)) for _ in units]
        u = [units[i] for i in idx]
        diffs.append(score(u, [pa[i] for i in idx])["micro"]["f1"] - score(u, [pb[i] for i in idx])["micro"]["f1"])
    diffs.sort()
    return sum(diffs) / n, diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def categorise(split: str) -> tuple[dict, dict]:
    counts = {k: 0 for k in ("exact", "wrong_type", "boundary", "missed", "spurious")}
    examples = {k: [] for k in counts}
    for u, p in zip(fx["splits"][split], preds["ner-kev-q3-4b-bidir"][split]):
        gold, guess, t = {tuple(e) for e in u["entities"]}, {tuple(e) for e in p}, u["text"]
        for e in sorted(gold):
            if e in guess:
                kind = "exact"
            elif any(x[:2] == e[:2] for x in guess):
                kind = "wrong_type"
            elif any(x[0] < e[1] and x[1] > e[0] for x in guess):
                kind = "boundary"
            else:
                kind = "missed"
            counts[kind] += 1
            examples[kind].append((u, p))
        for x in guess:
            if x not in gold and not any(e[0] < x[1] and e[1] > x[0] for e in gold):
                counts["spurious"] += 1
                examples["spurious"].append((u, p))
    return counts, examples


def ner_table() -> str:
    rows = []
    for run, label, hi in NER_RUNS:
        r = reports[run]
        e, p = r["results"]["ewt_test"]["micro"], r["results"]["pud_test"]["micro"]
        sps = r["gpu_sent_per_sec_b64"]
        cost = 1e6 / sps / 3600 * H100_PER_HOUR
        rows.append(f'<tr class="{"win" if hi else ""}"><td>{escape(label)}</td>'
                    f"<td>{f3(e['p'])}</td><td>{f3(e['r'])}</td><td><b>{f3(e['f1'])}</b></td>"
                    f"<td>{f3(p['p'])}</td><td>{f3(p['r'])}</td><td><b>{f3(p['f1'])}</b></td>"
                    f"<td>{sps:,.0f}</td><td>{r['gpu_sent_per_sec_b1']:.0f}</td><td>${cost:.2f}</td>"
                    f"<td>{r['train_sec'] / 60:.1f} min</td></tr>")
    rows.append('<tr><td>XLM-R-large, UNER paper (Mayhew et al. 2024, Fig. 4)</td><td>–</td><td>–</td><td><b>0.858</b></td>'
                '<td>–</td><td>–</td><td><b>0.805</b></td><td colspan="4" class="ceil">published, not rerun</td></tr>')
    c = win["recall_ceiling"]
    rows.append(f'<tr><td class="ceil">recall ceiling (gold reachable on Qwen token boundaries)</td><td></td>'
                f'<td class="ceil">{f3(c["ewt_test"])}</td><td></td><td></td><td class="ceil">{f3(c["pud_test"])}</td>'
                f'<td></td><td colspan="4"></td></tr>')
    return f"""<div class="tablewrap"><table class="wide">
<thead><tr><th rowspan="2">model</th><th colspan="3">EWT test (in-source)</th>
<th colspan="3">PUD (cross-source)</th><th colspan="2">sent. / s, H100</th><th rowspan="2">$ / 1M sent.</th><th rowspan="2">train</th></tr>
<tr><th>P</th><th>R</th><th>F1</th><th>P</th><th>R</th><th>F1</th><th>batch 64</th><th>batch 1</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def per_type_table() -> str:
    rows = []
    for run, label, hi in NER_RUNS:
        r = reports[run]["results"]
        cells = "".join(f"<td>{f3(r[s][t]['f1'])}</td>" for s in ("ewt_test", "pud_test") for t in TYPES)
        rows.append(f'<tr class="{"win" if hi else ""}"><td>{escape(label)}</td>{cells}</tr>')
    g = {s: {t: win["results"][s][t]["gold"] for t in TYPES} for s in ("ewt_test", "pud_test")}
    head = "".join(f"<th>{t}<br><span class='muted'>{g[s][t]}</span></th>" for s in ("ewt_test", "pud_test") for t in TYPES)
    return f"""<div class="tablewrap"><table>
<thead><tr><th rowspan="2">model</th><th colspan="3">EWT test F1</th><th colspan="3">PUD F1</th></tr>
<tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def bootstrap_table() -> str:
    pairs = [("ner-kev-q3-4b-bidir", "ner-roberta-large", "4B bidir − RoBERTa-large"),
             ("ner-kev-q3-4b-bidir", "ner-kev-q3-4b", "4B bidir − 4B causal"),
             ("ner-kev-q3-4b-bidir", "ner-kev-1.5b", "4B bidir − 1.5B causal"),
             ("ner-kev-1.5b", "ner-roberta-large", "1.5B causal − RoBERTa-large"),
             ("ner-kev-0.5b-bidir", "ner-roberta-large", "0.5B bidir − RoBERTa-large")]
    rows = []
    for a, b, label in pairs:
        cells = []
        for split in ("ewt_test", "pud_test"):
            m, lo, hi = bootstrap(a, b, split)
            sig = lo > 0
            cells.append(f'<td><b>{m:+.3f}</b></td><td>[{lo:+.3f}, {hi:+.3f}]</td>'
                         f'<td><span class="tag {"good" if sig else "est"}">{"above 0" if sig else "includes 0"}</span></td>')
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return f"""<div class="tablewrap"><table>
<thead><tr><th rowspan="2">paired difference in F1</th><th colspan="3">EWT test</th><th colspan="3">PUD</th></tr>
<tr><th>mean</th><th>95% CI</th><th></th><th>mean</th><th>95% CI</th><th></th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def error_bars(cats: dict) -> str:
    """Stacked horizontal bars of gold outcomes + spurious count, drawn to one scale."""
    colors = {"exact": "var(--good)", "wrong_type": "var(--hl-strong)", "boundary": "var(--accent)",
              "missed": "var(--bad)"}
    labels = {"exact": "exact", "wrong_type": "right span, wrong type", "boundary": "wrong boundary",
              "missed": "missed"}
    W, x0, bar_h = 600, 90, 26
    parts = []
    for i, split in enumerate(("ewt_test", "pud_test")):
        c = cats[split]
        total = sum(c[k] for k in colors)
        y = 20 + i * 58
        parts.append(f'<text x="0" y="{y + 17}" class="axis">{"EWT test" if i == 0 else "PUD"}</text>')
        x = x0
        for k in colors:
            w = (W - x0 - 10) * c[k] / total
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{colors[k]}"><title>{labels[k]}: {c[k]}</title></rect>')
            if w > 28:
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + 17}" text-anchor="middle" class="barlabel">{c[k]}</text>')
            x += w
        parts.append(f'<text x="{x0}" y="{y + bar_h + 15}" class="axis small">{total} gold: {c["exact"]} exact · {c["wrong_type"]} wrong type · '
                     f'{c["boundary"]} boundary · {c["missed"]} missed · plus {c["spurious"]} spurious</text>')
    legend = "".join(f'<span><i style="background:{colors[k]}"></i>{labels[k]}</span>' for k in colors)
    return (f'<div class="figure"><svg viewBox="0 0 {W} 136" role="img" aria-label="Outcome of every gold entity for the 4B bidirectional model">'
            f'{"".join(parts)}</svg><div class="legend">{legend}</div></div>')


def gallery(cats_examples: dict) -> str:
    """Real test sentences per error kind: gold vs kev 4B-bidir, short ones, deterministic pick."""
    titles = {"wrong_type": ("Right span, wrong type", "Mostly UNER's convention that a country acting politically is ORG."),
              "boundary": ("Wrong boundary", "Mostly whether a leading “the” or a trailing word belongs to the name."),
              "missed": ("Missed", "Lower-case, one-letter, or ambiguous mentions."),
              "spurious": ("Spurious", "A plausible name the annotators did not tag.")}
    blocks = []
    for kind, (title, note) in titles.items():
        seen, picks = set(), []
        rng = random.Random(7)
        pool = [x for x in cats_examples[kind] if len(x[0]["text"]) < 150]
        rng.shuffle(pool)
        for u, p in pool:
            if u["text"] in seen:
                continue
            seen.add(u["text"])
            picks.append((u, p))
            if len(picks) == 3:
                break
        items = "".join(f'<div class="pair"><div><span class="who">gold</span>{marked(u["text"], u["entities"])}</div>'
                        f'<div><span class="who">kev</span>{marked(u["text"], p)}</div></div>' for u, p in picks)
        blocks.append(f'<div class="card"><h3>{title}</h3><p class="small muted">{note}</p>{items}</div>')
    return f'<div class="gallery">{"".join(blocks)}</div>'


def versus() -> str:
    """Sentences where exactly one of kev-4B-bidir / RoBERTa matches gold, on PUD."""
    rows = {"kev": [], "rob": []}
    for i, u in enumerate(fx["splits"]["pud_test"]):
        g = {tuple(e) for e in u["entities"]}
        k = {tuple(e) for e in preds["ner-kev-q3-4b-bidir"]["pud_test"][i]} == g
        r = {tuple(e) for e in preds["ner-roberta-large"]["pud_test"][i]} == g
        if k != r and g and len(u["text"]) < 125:
            rows["kev" if k else "rob"].append(i)
    out = []
    for side, title in (("kev", "kev right, RoBERTa wrong"), ("rob", "RoBERTa right, kev wrong")):
        items = []
        for i in rows[side][:4]:
            u = fx["splits"]["pud_test"][i]
            items.append(f'<div class="pair"><div><span class="who">gold</span>{marked(u["text"], u["entities"])}</div>'
                         f'<div><span class="who">kev</span>{marked(u["text"], preds["ner-kev-q3-4b-bidir"]["pud_test"][i])}</div>'
                         f'<div><span class="who">RoBERTa</span>{marked(u["text"], preds["ner-roberta-large"]["pud_test"][i])}</div></div>')
        out.append(f'<div class="card"><h3>{title} <span class="muted small">({len(rows[side])} PUD sentences)</span></h3>{"".join(items)}</div>')
    return f'<div class="gallery two">{"".join(out)}</div>'


def token_table(ex: dict) -> str:
    rows = []
    for t in ex["tokens"]:
        part = t["part"]
        cls = "doc" if part == "doc" else "br-" + part.split()[1]
        restart = part != "doc" and t["piece"] == "Ċ"
        decide = t["k"] in ex["decide_at"]
        rows.append(f'<tr class="{cls}{" restart" if restart else ""}{" decide" if decide else ""}">'
                    f'<td>{t["k"]}</td><td>{t["pos"]}</td><td>{t["id"]}</td><td class="pc">{escape(piece(t["piece"]))}</td>'
                    f'<td>{escape(part)}{" ← position restarts" if restart else ""}{" ← &lt;decide&gt;: read by the head" if decide else ""}</td></tr>')
    return f"""<div class="tablewrap tall"><table class="tok">
<thead><tr><th>index</th><th>position id</th><th>token id</th><th>token</th><th>part</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def mask_grid(ex: dict) -> str:
    n = len(ex["mask"])
    marks = ex["marks"]
    cells = []
    for i, row in enumerate(ex["mask"]):
        for j, c in enumerate(row):
            cls = "on " + ("m-D" if marks[i] == "D" else f"m-{marks[i]}") if c == "#" else "off"
            cells.append(f'<i class="{cls}" title="row {i} ({marks[i]}) → col {j} ({marks[j]}): {"allowed" if c == "#" else "blocked"}"></i>')
    col_head = "".join(f'<b class="m-{m}">{m}</b>' for m in marks)
    return (f'<div class="maskwrap"><div class="maskhead" style="--n:{n}">{col_head}</div>'
            f'<div class="mask" style="--n:{n}">{"".join(cells)}</div></div>')


def heatmaps(ex: dict) -> str:
    """The whole output tensor: one [start token × width] panel per type, shaded by logit."""
    lo, hi = -16.0, 9.0
    panels = []
    d = len(ex["logits"][0])
    for t, plane in enumerate(ex["logits"]):
        cells = []
        for s in range(d):
            cells.append(f'<span class="rowlab">{escape(piece(ex["tokens"][s]["piece"]))}</span>')
            for w in range(16):
                v = plane[s][w]
                if v is None:
                    cells.append('<i class="na" title="runs past the sentence end — masked"></i>')
                    continue
                z = max(0.0, min(1.0, (v - lo) / (hi - lo)))
                above = v > ex["threshold"]
                txt = span_text(ex["tokens"], s, s + w)
                cells.append(f'<i class="{"hit" if above else ""}" style="--z:{z:.3f}" '
                             f'title="{TYPES[t]} · tokens {s}..{s + w} · logit {v:+.2f} · {escape(txt)}"></i>')
        wl = "".join(f"<b>{w + 1}</b>" for w in range(16))
        panels.append(f'<div class="hm"><div class="hmtitle">{TYPES[t]}</div><div class="hmgrid">'
                      f'<span class="rowlab"></span>{wl}{"".join(cells)}</div></div>')
    return f'<div class="hms">{"".join(panels)}</div>'


def decode_trace(ex: dict) -> str:
    """Candidates in score order with the decoder's verdict: kept / below threshold / overlaps a kept span."""
    kept_tok = []
    rows = []
    for c in ex["top"]:
        s, e = c["tokens"]
        if c["logit"] <= ex["threshold"]:
            verdict = '<span class="tag est">below threshold</span>'
        elif any(s <= b and a <= e for a, b in kept_tok):
            verdict = '<span class="tag bad">dropped · overlaps a kept span</span>'
        else:
            kept_tok.append((s, e))
            verdict = '<span class="tag good">kept</span>'
        rows.append(f'<tr><td>{c["logit"]:+.2f}</td><td>{c["p"]:.3f}</td><td>{c["type"]}</td><td>{s}..{e}</td>'
                    f'<td>{c["chars"][0]}..{c["chars"][1]}</td><td class="lt">{escape(c["text"])}</td><td>{verdict}</td></tr>')
    return f"""<div class="tablewrap"><table class="trace">
<thead><tr><th>logit</th><th>p</th><th>type</th><th>tokens</th><th>chars</th><th class="l">span</th><th>decoder</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def example_card(ex: dict, note: str) -> str:
    u = next(u for s in ("pud_test", "ewt_test") for u in fx["splits"][s] if u["text"] == ex["text"])
    g = {tuple(e) for e in u["entities"]}
    k = {tuple(e) for e in ex["decoded"]}
    verdict = "matches gold exactly" if g == k else f"{len(g & k)}/{len(g)} gold entities exact"
    return f"""<div class="card ex">
<div class="pair"><div><span class="who">gold</span>{marked(ex['text'], u['entities'])}</div>
<div><span class="who">kev</span>{marked(ex['text'], ex['decoded'])}</div></div>
<p class="small muted">Output tensor {ex['shape'][0]} × {ex['shape'][1]} × {ex['shape'][2]} ({ex['real_spans']} real spans) · {verdict}. {note}</p>
{decode_trace(ex)}
<div class="panel-label">decoded output</div><pre>{escape(json.dumps(ex['decoded']))}</pre>
</div>"""


def run21_tables() -> str:
    k15, k05, gl = r21["kev-1.5b"], r21["kev-0.5b"], r21["gliner-ft"]
    base = k15["baseline"]
    adj = k15["adjudicated"]
    rows = [
        ("win", "kev-span · Qwen2.5-1.5B + LoRA r16", k15["span_model"], adj["span_model"], k15),
        ("", "kev-span · Qwen2.5-0.5B + LoRA r16", k05["span_model"], k05["adjudicated"]["span_model"], k05),
        ("", "generative LLM student (released <code>header_metadata/model-1</code>)", base, adj["baseline"], None),
        ("", "GLiNER multi-v2.1, fine-tuned", gl["span_model"], gl["adjudicated"]["span_model"], gl),
    ]
    body = []
    for cls, label, m, a, r in rows:
        speed = (f"<td>{r['gpu_ms_per_unit']:.0f} ms</td><td>{r['cpu_ms_per_unit'] / 1000:.1f} s</td>"
                 f"<td>{r['train_sec'] / 60:.1f} min</td>") if r else "<td>–</td><td>–</td><td>–</td>"
        body.append(f'<tr class="{cls}"><td>{label}</td><td><b>{f3(m["mean_field"])}</b></td><td>{f3(a["mean_field"])}</td>'
                    f'<td>{f3(m["unit_exact"])}</td><td>{f3(a["unit_exact"])}</td>{speed}</tr>')
    c = k15["tagger_ceiling"]
    body.append(f'<tr><td class="ceil">ceiling: gold value printed in the text</td><td class="ceil">{f3(c["mean_field"])}</td>'
                f'<td></td><td class="ceil">{f3(c["unit_exact"])}</td><td colspan="4"></td></tr>')
    main = f"""<div class="tablewrap"><table>
<thead><tr><th>model</th><th>mean field</th><th>adjudicated</th><th>unit exact</th><th>adjudicated</th>
<th>GPU / unit</th><th>CPU / unit</th><th>train</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"""
    frows = []
    for f, v in k15["span_model"]["fields"].items():
        b = base["fields"][f]
        diff = v - b
        frows.append(f'<tr><td><code>{f}</code></td><td>{f3(v)}</td><td>{f3(b)}</td>'
                     f'<td class="{"up" if diff > 0 else "down" if diff < 0 else ""}">{diff:+.3f}</td><td class="ceil">{f3(c["fields"][f])}</td></tr>')
    per = f"""<div class="tablewrap"><table>
<thead><tr><th>field</th><th>kev-span 1.5B</th><th>LLM student</th><th>Δ</th><th>printed verbatim</th></tr></thead>
<tbody>{''.join(frows)}</tbody></table></div>
<p class="small muted">“Printed verbatim” = share of gold values that appear character-for-character in the text. Dates can beat it
because the type normaliser turns «08» грудня 2025 into 2025-12-08.</p>"""
    return main, per


def span_unit(u: dict, title: str, note: str) -> str:
    text = u["text"]
    spans = []
    for r in u["fields"]:
        if r["span_wins"]:
            t = r["best_span"]["text"].strip()
            at = text.find(t)
            if at >= 0 and len(t) < 200:
                spans.append([at, at + len(t), r["field"]])
    # keep non-overlapping marks only (two fields can point at the same text)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    shown, end = [], -1
    for s in spans:
        if s[0] >= end:
            shown.append(s)
            end = s[1]
    cut = max(600, max((s[1] for s in shown), default=0) + 80)
    excerpt = marked(text[:cut], [s for s in shown if s[1] <= cut], "fld") + "…"
    rows = []
    for r in u["fields"]:
        ts = " · ".join(f'{escape(c["text"].strip() or c["text"])} <span class="muted">{c["p"]:.2f}</span>' for c in r["top_start"])
        best = r["best_span"]
        bt = best["text"].strip()
        bt = escape(bt[:48]) + ("…" if len(bt) > 48 else "")
        best_cell = bt if r["span_wins"] else f'<span class="muted">{bt}<br>(rejected)</span>'
        verdict = f'<span class="tag {"good" if r["ok"] else "bad"}">{"✓" if r["ok"] else "✗"}</span>'
        gold = "" if r["ok"] else f'<br><span class="muted small">gold {escape(json.dumps(r["gold"], ensure_ascii=False))}</span>'
        rows.append(
            f'<tr class="{"" if r["ok"] else "miss"}"><td><code>{r["field"]}</code><br><span class="muted small">{r["type"]}</span></td>'
            f'<td class="lt small">{ts}<br><span class="muted">null {r["p_start_null"]:.2f}</span></td>'
            f'<td class="lt small">{best_cell}</td>'
            f'<td>{best["logp"]:.2f}</td><td>{r["logp_null"]:.2f}</td>'
            f'<td class="lt">{verdict} <b>{escape(json.dumps(r["value"], ensure_ascii=False))}</b>{gold}</td></tr>')
    out_json = {r["field"]: r["value"] for r in u["fields"]}
    return f"""<div class="card ex">
<h3>{title}</h3><p class="small muted">{note} Source <code>{u['collection']}</code> (held out: never seen in training) ·
{u['chars']} characters → {u['doc_tokens']} document tokens + 9 branches = {u['packed_tokens']} packed tokens ·
raw output: start and end logits, each of shape {u['logit_shape'][0]} × {u['logit_shape'][1]} (9 fields × {u['doc_tokens']} tokens + 1 null slot).</p>
<div class="doc small">{excerpt}</div>
<div class="tablewrap"><table class="trace fields">
<thead><tr><th>field</th><th class="l">top-3 start tokens (p)</th><th class="l">best span</th><th>log p(span)</th><th>log p(null)</th><th class="l">typed value · vs gold</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div class="panel-label">what a consumer receives (built from the table above)</div>
<pre>{escape(json.dumps(out_json, ensure_ascii=False, indent=1))}</pre>
</div>"""


cats, cat_examples = {}, {"wrong_type": [], "boundary": [], "missed": [], "spurious": [], "exact": []}
for split in ("ewt_test", "pud_test"):
    c, ex = categorise(split)
    cats[split] = c
    for k in cat_examples:
        cat_examples[k] += ex[k]
ex1, ex2, ex3, ex4 = io
r21_main, r21_fields = run21_tables()
branches = "".join(f'<li><code>{escape(t["name"])}: {escape(t["desc"])}</code></li>' for t in fx["types"])
run21_fields = load(ROOT / "fixtures" / "run-21.json")["task"]["fields"]
field_branches = "".join(f'<tr><td><code>{f["name"]}</code></td><td>{f["type"]}</td><td class="lt small">{escape(f["desc"])}</td></tr>'
                         for f in run21_fields)
w_ewt, w_pud = win["results"]["ewt_test"]["micro"]["f1"], win["results"]["pud_test"]["micro"]["f1"]
r_ewt, r_pud = rob["results"]["ewt_test"]["micro"]["f1"], rob["results"]["pud_test"]["micro"]["f1"]
k15 = r21["kev-1.5b"]

CSS = (ROOT / "docs" / "page.css").read_text(encoding="utf-8")
print(f"""<title>kev Results &amp; Raw I/O</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap">
<style>{CSS}</style>
<div class="page">

<header class="top">
  <div class="eyebrow">Span extraction · results and raw I/O · Sep 2026</div>
  <h1>kev: every entity type answered in one forward pass, no text generated</h1>
  <p class="lede">A decoder reads a document plus one short “question” per type, and returns a tensor of span scores.
  Answers are always substrings of the input. Below: the results on an open NER benchmark and on production
  extraction fields, then exactly what goes in and what comes out.</p>
  <div class="meta"><span>best model: Qwen3-4B-Base + LoRA r16</span><span>trained on H100 · 14.5 min</span><span>entity-level exact-match F1</span></div>
</header>

<section aria-label="Headline numbers">
  <div class="stats four">
    <div class="stat"><div class="v">{w_pud:.3f}</div><div class="l">F1 on PUD (cross-source)<br>RoBERTa-large {r_pud:.3f} · published XLM-R-large 0.805</div></div>
    <div class="stat"><div class="v">{w_ewt:.3f}</div><div class="l">F1 on EWT test (in-source)<br>RoBERTa-large {r_ewt:.3f} · published XLM-R-large 0.858</div></div>
    <div class="stat"><div class="v">{k15['span_model']['mean_field']:.3f}</div><div class="l">production fields, held-out sources<br>generative LLM student {k15['baseline']['mean_field']:.3f}</div></div>
    <div class="stat"><div class="v">1 <small>pass</small></div><div class="l">per document, whatever the number of types or entities · 0 tokens generated</div></div>
  </div>
</section>

<nav class="toc small"><b>On this page</b>
<a href="#ner">NER results</a><a href="#fields">Production-field results</a><a href="#input">Input</a><a href="#mask">Mask</a>
<a href="#output">Raw output</a><a href="#decode">Decoding</a><a href="#examples">NER examples</a><a href="#errors">Errors</a>
<a href="#fieldio">Field extraction I/O</a><a href="#llm">vs an LLM</a><a href="#caveats">Caveats</a></nav>

<section id="ner">
  <div class="eyebrow">Result 1 · open benchmark</div>
  <h2>Universal NER English: kev at 4B beats a fine-tuned RoBERTa-large, most clearly on unseen sources</h2>
  <p>Dataset <code>universalner/uner_llm_inst_english</code>: PER / ORG / LOC, any number per sentence. Trained on EWT train
  (12,542 web-text sentences), threshold tuned on EWT dev, tested on <b>EWT test</b> (same source) and <b>PUD</b>
  (news and Wikipedia, a different source). A prediction counts only with the exact character span <i>and</i> type.
  Throughput includes decoding; cost assumes ${H100_PER_HOUR}/H100-hour. One training seed per model.</p>
  {ner_table()}
  <h3>Per type</h3>
  <p class="small muted">Gold counts under each type. ORG is the hard class for every model; the 4B bidirectional model gains most there.</p>
  {per_type_table()}
  <h3>Is the gap real? Paired bootstrap over test sentences (1,000 resamples)</h3>
  {bootstrap_table()}
  <p class="small muted">Up to 1.5B, kev ties RoBERTa (every interval includes 0). At 4B with bidirectional document attention the
  gain over RoBERTa is significant on both test sets. Against its own causal twin the bidirectional 4B is clearly ahead on PUD;
  on EWT the interval just touches 0. The bootstrap covers test sampling, not training-seed variance.</p>
</section>

<section id="fields">
  <div class="eyebrow">Result 2 · production extraction</div>
  <h2>Nine typed fields on documents from sources never seen in training</h2>
  <p>Same architecture with one start/end pointer plus a “not printed” slot per field instead of a span grid. 173 training
  units from 11 sources, evaluated on 59 units from 4 held-out sources. Scored with the pipeline's own field scorer;
  “adjudicated” rescored after a manual gold review found 13 gold errors.</p>
  {r21_main}
  <div class="stack">
    <div>{r21_fields}</div>
    <div class="card"><h3>What the numbers mean</h3><ul class="small">
      <li>kev-span 1.5B beats the generative student on 6 of 9 fields and ties 1; it loses on <code>parent_number</code> and <code>subject</code>.</li>
      <li>“Unit exact” (all 9 right) is similar: 0.390 vs 0.407. kev wins more fields, the LLM more whole units.</li>
      <li>The LLM failed in ways kev cannot: 6 units of invalid JSON, invented dates, subjects translated to English.</li>
      <li>kev's misses are boundary or null decisions: truncating <code>105/105</code> to <code>105</code>, or picking null when the value is printed.</li>
      <li>Per held-out source: {", ".join(f"{v:.2f}" for v in k15['per_collection_mean_field'].values())}.</li>
    </ul></div>
  </div>
</section>

<section id="input">
  <div class="eyebrow">Input</div>
  <h2>What the model reads: one token sequence, no prompt</h2>
  <p>The sentence tokens come first. Then one branch per type, written from the schema as
  <code>⏎&lt;|fim_prefix|&gt;NAME: description&lt;|fim_suffix|&gt;</code>. Qwen's reserved fill-in-the-middle tokens are
  reused as markers, and the last token of each branch is the <b>&lt;decide&gt;</b> token whose hidden state asks the question.
  Each branch's position ids restart right after the sentence, so no type sits “further away” than another.</p>
  <p class="small">Branches used for NER (the dataset's own instruction wording):</p>
  <ul class="small">{branches}</ul>
  <p>Real input for the PUD sentence <i>“{escape(ex1['text'])}”</i>, {len(ex1['tokens'])} tokens:</p>
  {token_table(ex1)}
</section>

<section id="mask">
  <div class="eyebrow">Input · attention</div>
  <h2>The attention mask keeps the questions independent</h2>
  <p>Row = the token doing the attending, column = the token it may read. <span class="sw m-D"></span> sentence ↔ sentence
  (bidirectional in the winning model), <span class="sw m-P"></span> PER, <span class="sw m-O"></span> ORG, <span class="sw m-L"></span> LOC.
  Each branch reads the whole sentence and itself (causally), never another branch, so all types are answered in one pass. At start-up
  the code edits branch 0 and asserts branch 1's scores do not move (measured drift 0.00).</p>
  {mask_grid(ex1)}
</section>

<section id="output">
  <div class="eyebrow">Raw output</div>
  <h2>What comes out: one tensor of span scores, not text</h2>
  <p>Shape <b>{ex1['shape'][0]} × {ex1['shape'][1]} × {ex1['shape'][2]}</b> = [type, span start token, span width 1–16].
  Each cell is a logit answering “is the span of <i>w</i> tokens starting at token <i>s</i> an entity of this type?”.
  {ex1['real_spans']} cells are real spans; grey cells run past the sentence end and are masked. Stronger colour = higher logit;
  outlined cells are above the decode threshold ({ex1['threshold']}). Hover a cell for its span and score.</p>
  {heatmaps(ex1)}
  <p class="small muted">Three cells light up: <b>Parker</b> (PER, row “␣Parker”, width 1), <b>Russian Secret Service</b> (ORG, row
  “␣Russian”, width 3) and <b>Great Britain</b> (LOC, row “␣Great”, width 2). Every other cell is far below the threshold.</p>
</section>

<section id="decode">
  <div class="eyebrow">Decoding</div>
  <h2>From tensor to entities: three lines of plain Python</h2>
  <ol class="steps">
    <li>Take every cell whose logit is above the threshold picked on dev ({ex1['threshold']}).</li>
    <li>Sort them by score, best first.</li>
    <li>Keep a span unless it overlaps a span already kept (UNER is flat). Map its tokens to character offsets.</li>
  </ol>
  <p>The top-scoring cells for the same sentence, with the decoder's verdict:</p>
  {decode_trace(ex1)}
  <div class="panel-label">decoded output: [char_start, char_end, type]</div>
  <pre>{escape(json.dumps(ex1['decoded']))}
{chr(10).join(f"  {t}: {ex1['text'][a:b]!r}" for a, b, t in ex1['decoded'])}</pre>
</section>

<section id="examples">
  <div class="eyebrow">More real examples · PUD test</div>
  <h2>Three more sentences through the same pipeline</h2>
  {example_card(ex2, "“H Street” is kept at logit −1.56, just above the −2.0 threshold: the threshold decides borderline mentions.")}
  {example_card(ex3, "Two locations, one nested inside a longer candidate span; the longer span scores far lower.")}
  {example_card(ex4, "A wrong-type case: UNER tags countries acting as states (“China … fighting … with Great Britain”) as ORG; kev says LOC. The long span “Great Britain after the Governor-General of Hunan and Hubei” clears the threshold but is dropped for overlapping.")}
</section>

<section id="errors">
  <div class="eyebrow">Errors · 4B bidirectional model</div>
  <h2>Where the remaining errors are</h2>
  <p>Every gold entity in both test sets, classified by what kev did with it.</p>
  {error_bars(cats)}
  {gallery(cat_examples)}
  <h3>Head to head with RoBERTa-large on PUD</h3>
  {versus()}
</section>

<section id="fieldio">
  <div class="eyebrow">Field extraction I/O · production model</div>
  <h2>The same idea on long documents: one span or “not printed” per field</h2>
  <p>For typed fields the head is a <b>start pointer</b> and an <b>end pointer</b> from each &lt;decide&gt; token over the document tokens,
  plus a learned <b>null slot</b> for “not printed here”. The best span competes with null; a normaliser picked by the field's declared type
  (<code>date</code>, <code>number</code>, <code>text</code>) turns the span into the value. The 9 branches come from the task schema:</p>
  <div class="tablewrap"><table class="fields"><thead><tr><th>field</th><th>type</th><th>branch description (verbatim from the schema)</th></tr></thead>
  <tbody>{field_branches}</tbody></table></div>
  <p>Three real held-out documents through kev-span 1.5B (temperature {span_io['temperature']}, span-vs-null bias {span_io['bias']}).
  Highlights mark where each field's winning span sits.</p>
  {span_unit(span_io['units'][0], "An act: all 9 fields right", "Six printed values found; the three parent/annex fields correctly null.")}
  {span_unit(span_io['units'][1], "An annex: parent reference read, own number correctly refused", "The model's best span for <code>number</code> is “61”, but that is the <i>parent</i> act's number, and null wins (−0.07 vs −9.04).")}
  {span_unit(span_io['units'][2], "A miss: the right value found, then out-voted by null", "For <code>parent_number</code> the best span is the correct “9”, but null scores higher (−0.89 vs −4.79). The value was read correctly; the null decision lost.")}
</section>

<section id="llm">
  <div class="eyebrow">Comparison</div>
  <h2>The same sentence as a generative LLM sees it</h2>
  <div class="twocol">
    <div><div class="panel-label">LLM input (the dataset's own prompt, verbatim)</div><pre class="wrap">{escape(llm['inputs'])}</pre></div>
    <div><div class="panel-label">LLM output: generated one token at a time</div><pre class="wrap">{escape(llm['targets'])}</pre>
    <div class="panel-label" style="margin-top:14px">kev input</div><pre class="wrap">{len(ex1['tokens'])} tokens: the sentence + PER / ORG / LOC branches (table above)</pre>
    <div class="panel-label" style="margin-top:14px">kev output</div><pre class="wrap">float32[{ex1['shape'][0]}, {ex1['shape'][1]}, {ex1['shape'][2]}] → {json.dumps(ex1['decoded'])}</pre></div>
  </div>
  <div class="tablewrap"><table class="cmp">
    <thead><tr><th></th><th class="l">generative LLM</th><th class="l">kev</th></tr></thead>
    <tbody>
    <tr><td>input</td><td class="lt">instruction + worked example + sentence</td><td class="lt">sentence + <code>name: description</code> per type</td></tr>
    <tr><td>output</td><td class="lt">JSON text, token by token</td><td class="lt">one tensor of span scores (NER) or start/end/null pointers (fields)</td></tr>
    <tr><td>forward passes</td><td class="lt">one per output token (~70 for this answer)</td><td class="lt"><b>1</b>, however many entities</td></tr>
    <tr><td>can fail by</td><td class="lt">invalid JSON, invented or paraphrased values, wrong offsets</td><td class="lt">only a wrong or missing span: output is always a substring of the input</td></tr>
    <tr><td>confidence</td><td class="lt">not directly available</td><td class="lt">a probability for every candidate span</td></tr>
    <tr><td>new type</td><td class="lt">edit the prompt</td><td class="lt">add a branch (text); one model can in principle serve any declared type list (not yet tested on unseen types)</td></tr>
    </tbody></table></div>
</section>

<section id="caveats">
  <div class="eyebrow">Caveats</div>
  <h2>What these results do not show yet</h2>
  <ul>
    <li>One training seed per model. The bootstrap intervals cover test sampling, not seed-to-seed variance; two more 4B seeds (~15 min of H100 each) would settle it.</li>
    <li>Cost: the 4B model runs at {reports['ner-kev-q3-4b-bidir']['gpu_sent_per_sec_b64']:.0f} sentences/s vs RoBERTa's {rob['gpu_sent_per_sec_b64']:,.0f}, about 14× the GPU cost per sentence. Both are far below a generative LLM.</li>
    <li>The production-field evaluation is small (59 units); its differences are indicative, the NER benchmark is the statistically meaningful one.</li>
    <li>About 1% of NER gold is unreachable because its boundary falls inside a Qwen token.</li>
    <li>Types never seen in training have not been tested.</li>
  </ul>
</section>

<footer>
  <div>Code: <code>kev/ner.py</code>, <code>kev/span.py</code>, <code>baselines/encoder_ner.py</code>, <code>scripts/show_io.py</code>, <code>scripts/show_span_io.py</code>. Method notes: <code>docs/METHOD.md</code>.</div>
  <div>Every number and example on this page is generated from the run outputs by <code>scripts/build_page.py</code>.</div>
</footer>
</div>""")
