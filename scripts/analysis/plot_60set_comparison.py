"""Chart: spark-s1 vs Jev on the 60-row tool-call diagnostic set (accuracy, by slice, latency). Writes docs/img/toolcall-60-comparison.{svg,json}.
Jev = recorded hosted run committed in themsquared/jev-benchmark (data/external/ext-toolcall-risk). spark-s1 = runs/osdg/final60.json (+ rows)."""
import json
from collections import defaultdict
from html import escape

import numpy as np

from open_spark_jev.data.corpus import read_jsonl
from open_spark_jev.eval.external import _jev_probs

f = json.load(open("runs/osdg/final60.json"))
recs = read_jsonl("data/benchmarks/external/ext-toolcall-risk.jsonl")
slice_of = {r.id: r.domain.split("/")[-1] for r in recs}
jev_sl = defaultdict(list)
for r in recs:
    jp = _jev_probs(r, r.question_obj().labels)
    jev_sl[slice_of[r.id]].append(int(np.argmax(jp) == r.label_index()))
rows = [json.loads(l) for l in open("runs/osdg/final60.rows.jsonl")]
def sl(model):
    d = defaultdict(list)
    for r in rows:
        if r["model"] == model:
            d[r["slice"]].append(int(r["pred"] == r["gold"]))
    return d
M = [("Jev (hosted)", f["jev"]["acc"], f["jev"]["p50_ms"], "ink", {k: np.mean(v) for k, v in jev_sl.items()}),
     ("spark-s1-4b-v3", f["v3-4b"]["acc"], f["v3-4b"]["p50_ms"], "orange", {k: np.mean(v) for k, v in sl("v3-4b").items()}),
     ("spark-s1-1.7b-v3", f["v3-1.7b"]["acc"], f["v3-1.7b"]["p50_ms"], "ring", {k: np.mean(v) for k, v in sl("v3-1.7b").items()}),
     ("Qwen3-4B, untrained", f["v0-qwen3-4b"]["acc"], f["v0-qwen3-4b"]["p50_ms"], "grey", {k: np.mean(v) for k, v in sl("v0-qwen3-4b").items()})]
INK, INK2, MUTE, GRID, SURF, ORANGE, GREY = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb", "#eb6834", "#a9a8a2"
col = {"ink": INK, "orange": ORANGE, "ring": ORANGE, "grey": GREY}
W, H = 1200, 780
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">', f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
     f'<text x="48" y="54" font-size="26" font-weight="600" fill="{INK}">spark-s1 vs Jev on the 60-case tool-call diagnostic set</text>',
     f'<text x="48" y="82" font-size="15" fill="{INK2}">Same 60 hand-labelled agent tool calls, four risk postures. Jev is its recorded hosted run; spark-s1 ran locally on one DGX Spark (batch 1).</text>']
# panel 1 accuracy
o.append(f'<text x="48" y="128" font-size="16" font-weight="600" fill="{INK}">Accuracy</text>')
x0, x1, y = 250, 720, 156
sx = lambda v: x0 + v * (x1 - x0)
for g in (0, .25, .5, .75, 1):
    o.append(f'<line x1="{sx(g):.1f}" y1="146" x2="{sx(g):.1f}" y2="{y + len(M) * 44 - 8}" stroke="{GRID}"/><text x="{sx(g):.1f}" y="{y + len(M) * 44 + 8}" font-size="12" fill="{MUTE}" text-anchor="middle">{g:.2f}</text>')
for i, (nm, acc, ms, c, _) in enumerate(M):
    yy = y + i * 44
    b = 'font-weight="600"' if nm.startswith("spark") else ""
    o.append(f'<text x="{x0 - 14}" y="{yy + 17}" font-size="14" fill="{INK}" text-anchor="end" {b}>{escape(nm)}</text>')
    fill = SURF if c == "ring" else col[c]
    stroke = f' stroke="{ORANGE}" stroke-width="2.5"' if c == "ring" else ""
    o.append(f'<rect x="{x0}" y="{yy}" width="{sx(acc) - x0:.1f}" height="24" rx="3" fill="{fill}"{stroke}><title>{escape(nm)} accuracy {acc:.3f}</title></rect>')
    o.append(f'<text x="{x1 + 50}" y="{yy + 17}" font-size="14" fill="{INK}" text-anchor="end">{acc:.3f}</text>')
# panel 2 latency
lx0, lx1 = 850, 1060
mx = 450.0
o.append(f'<text x="850" y="128" font-size="16" font-weight="600" fill="{INK}">Median latency, ms (lower is better)</text>')
for g in (0, 150, 300, 450):
    xx = lx0 + g / mx * (lx1 - lx0)
    o.append(f'<line x1="{xx:.1f}" y1="146" x2="{xx:.1f}" y2="{y + len(M) * 44 - 8}" stroke="{GRID}"/><text x="{xx:.1f}" y="{y + len(M) * 44 + 8}" font-size="12" fill="{MUTE}" text-anchor="middle">{g}</text>')
for i, (nm, acc, ms, c, _) in enumerate(M):
    yy = y + i * 44
    fill = SURF if c == "ring" else col[c]
    stroke = f' stroke="{ORANGE}" stroke-width="2.5"' if c == "ring" else ""
    wbar = ms / mx * (lx1 - lx0)
    o.append(f'<rect x="{lx0}" y="{yy}" width="{wbar:.1f}" height="24" rx="3" fill="{fill}"{stroke}><title>{escape(nm)} p50 {ms:.1f} ms</title></rect>')
    o.append(f'<text x="{min(lx0 + wbar + 8, 1080):.1f}" y="{yy + 17}" font-size="13" fill="{INK if lx0 + wbar + 8 < 1000 else SURF}">{ms:.0f}</text>' if False else f'<text x="{lx1 + 90}" y="{yy + 17}" font-size="14" fill="{INK}" text-anchor="end">{ms:.1f}</text>')
# panel 3 slices
py = y + len(M) * 44 + 60
o.append(f'<text x="48" y="{py}" font-size="16" font-weight="600" fill="{INK}">Accuracy by slice of the set</text>')
lg = 48
for nm, c, _, _, _ in [(m[0], m[3], 0, 0, 0) for m in M]:
    cx, cy = lg + 7, py + 26
    if c == "ink":
        o.append(f'<polygon points="{cx},{cy - 7} {cx + 7},{cy} {cx},{cy + 7} {cx - 7},{cy}" fill="{INK}"/>')
    elif c == "ring":
        o.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{SURF}" stroke="{ORANGE}" stroke-width="2.5"/>')
    else:
        o.append(f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="{col[c]}"/>')
    o.append(f'<text x="{cx + 14}" y="{cy + 5}" font-size="13" fill="{INK2}">{escape(nm)}</text>')
    lg += 44 + len(nm) * 8
labels = {"clear": "Clear (34 calls)", "ambiguous": "Ambiguous (14)", "adversarial": "Adversarial, benign-looking framing (12)"}
ax0, ax1, ry0 = 340, 1080, py + 64
sx2 = lambda v: ax0 + v * (ax1 - ax0)
for g in (0, .25, .5, .75, 1):
    o.append(f'<line x1="{sx2(g):.1f}" y1="{ry0 - 16}" x2="{sx2(g):.1f}" y2="{ry0 + 3 * 46 - 16}" stroke="{GRID}"/><text x="{sx2(g):.1f}" y="{ry0 + 3 * 46 + 2}" font-size="12" fill="{MUTE}" text-anchor="middle">{g:.2f}</text>')
for i, k in enumerate(("clear", "ambiguous", "adversarial")):
    cy = ry0 + i * 46
    o.append(f'<text x="{ax0 - 14}" y="{cy + 5}" font-size="14" fill="{INK}" text-anchor="end">{labels[k]}</text><line x1="{ax0}" y1="{cy}" x2="{ax1}" y2="{cy}" stroke="{GRID}" stroke-dasharray="2 4"/>')
    for li, (nm, acc, ms, c, s) in enumerate(M):
        v = s[k]; cx = sx2(v); cy0 = cy; cy = cy0 + (li - 1.5) * 9
        t = f"<title>{escape(nm)} {k}: {v:.3f}</title>"
        if c == "ink":
            o.append(f'<polygon points="{cx:.1f},{cy - 7} {cx + 7:.1f},{cy} {cx:.1f},{cy + 7} {cx - 7:.1f},{cy}" fill="{INK}" stroke="{SURF}" stroke-width="1.5">{t}</polygon>')
        elif c == "ring":
            o.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="5.5" fill="{SURF}" stroke="{ORANGE}" stroke-width="2.2">{t}</circle>')
        else:
            o.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="6" fill="{col[c]}" stroke="{SURF}" stroke-width="1.5">{t}</circle>')
        cy = cy0
fy = ry0 + 3 * 46 + 34
for j, n in enumerate(["An early version: spark-s1 was fine-tuned on 967 code-labelled rows and is being improved with more data. It is not at Jev parity (0.850 vs 0.917) and",
                       "still misses security-weakening config changes and credential exfiltration. Rows in each slice line: Jev, 4B, 1.7B, untrained (top to bottom).",
                       "The 60-case set is a diagnostic benchmark inspected during development, not a locked holdout; n=60 is about +/-5 points. Jev's latency is a recorded hosted run that",
                       "includes the network, so the latency gap is not a pure model-to-model comparison. Sources: themsquared/jev-benchmark (Apache-2.0), docs/BENCHMARKS.md B29, docs/img/toolcall-60-comparison.json."]):
    o.append(f'<text x="48" y="{fy + j * 19}" font-size="12" fill="{INK2}">{escape(n)}</text>')
o.append("</svg>")
open("docs/img/toolcall-60-comparison.svg", "w").write("\n".join(o))
json.dump({m[0]: {"accuracy": round(m[1], 4), "p50_ms": m[2], "slices": {k: round(float(v), 4) for k, v in m[4].items()}} for m in M}, open("docs/img/toolcall-60-comparison.json", "w"), indent=1)
print("ok")
