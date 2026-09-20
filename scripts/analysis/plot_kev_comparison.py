"""Chart: spark-s1 vs Kev vs Jev on Kev's out-of-domain transfer-v4 suite (same items). Writes docs/img/kev-comparison.svg + .json.
Kev/Jev numbers: data/external/ext-kev-transfer-v4-dev/kev_published.json (Kev's own). spark-s1 numbers: runs/external/*."""
import json
from html import escape

pub = json.load(open("data/external/ext-kev-transfer-v4-dev/kev_published.json"))
def ours(name, src):
    r = json.load(open(f"runs/external/{name}/{src}.json"))
    return r["accuracy"], {k.split("/")[-1]: v["accuracy"] for k, v in r["slices"].items()}
d4, d4s = ours("kevdev-v3-4b", "ext-kev-transfer-v4-dev"); d17, d17s = ours("kevdev-v3-1.7b", "ext-kev-transfer-v4-dev")
t4, _ = ours("spark-s1-v3-4b-osdg-trained", "ext-kev-transfer-v4"); t17, _ = ours("spark-s1-v3-1.7b-osdg-trained", "ext-kev-transfer-v4")
INK, INK2, MUTE, GRID, SURF = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
rows = [("Jev (hosted)", pub["overall_dev"]["Jev"], None, INK), ("Kev-8B", pub["overall_dev"]["Kev-8B"], pub["overall_locked_test"]["Kev-8B"], BLUE),
        ("Kev-4B", pub["overall_dev"]["Kev-4B"], pub["overall_locked_test"]["Kev-4B"], BLUE), ("spark-s1-4b-v3", d4, t4, ORANGE),
        ("Kev-0.6B", pub["overall_dev"]["Kev-0.6B"], pub["overall_locked_test"]["Kev-0.6B"], BLUE), ("spark-s1-1.7b-v3", d17, t17, ORANGE)]
rows.sort(key=lambda r: -r[1])
W, H = 1200, 990
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
     f'<text x="48" y="54" font-size="26" font-weight="600" fill="{INK}">spark-s1 vs Kev vs Jev on Kev\'s out-of-domain suite</text>',
     f'<text x="48" y="82" font-size="15" fill="{INK2}">Kev transfer-v4: 764 items per partition; sources Kev and spark-s1 never trained on (Jev unknown). Kev and Jev numbers are Kev\'s published ones; spark-s1 was measured by us.</text>',
     f'<text x="48" y="130" font-size="16" font-weight="600" fill="{INK}">Overall accuracy</text>',
     f'<text x="48" y="150" font-size="13" fill="{INK2}">Bar = development partition (Jev measured there only). Ring = locked test, where published or measured.</text>']
x0, x1, bar_h, y = 250, 940, 22, 190
sx = lambda v: x0 + (v - 0.0) / 1.0 * (x1 - x0)
for g in (0, 0.25, 0.5, 0.75, 1.0):
    o.append(f'<line x1="{sx(g):.1f}" y1="178" x2="{sx(g):.1f}" y2="{y + len(rows) * 40 - 6}" stroke="{GRID}" stroke-width="1"/>')
    o.append(f'<text x="{sx(g):.1f}" y="{y + len(rows) * 40 + 12}" font-size="12" fill="{MUTE}" text-anchor="middle">{g:.2f}</text>')
for i, (name, dev, test, col) in enumerate(rows):
    yy = y + i * 40
    bold = 'font-weight="600"' if name.startswith("spark") else ""
    o.append(f'<text x="{x0 - 14}" y="{yy + 16}" font-size="14" fill="{INK}" text-anchor="end" {bold}>{escape(name)}</text>')
    o.append(f'<rect x="{x0}" y="{yy}" width="{sx(dev) - x0:.1f}" height="{bar_h}" rx="3" fill="{col}"><title>{escape(name)} development accuracy {dev:.3f}</title></rect>')
    o.append(f'<text x="990" y="{yy + 16}" font-size="13" fill="{INK}" text-anchor="end">{dev:.3f}</text>')
    o.append(f'<text x="1100" y="{yy + 16}" font-size="13" fill="{INK2}" text-anchor="end">{"-" if test is None else f"{test:.3f}"}</text>')
    if test is not None:
        o.append(f'<circle cx="{sx(test):.1f}" cy="{yy + bar_h / 2}" r="6" fill="{SURF}" stroke="{INK}" stroke-width="2"><title>{escape(name)} locked test accuracy {test:.3f}</title></circle>')
o.append(f'<text x="990" y="{y - 14}" font-size="12" font-weight="600" fill="{INK2}" text-anchor="end">dev</text>')
o.append(f'<text x="1100" y="{y - 14}" font-size="12" font-weight="600" fill="{INK2}" text-anchor="end">locked test</text>')
# panel B
py = y + len(rows) * 40 + 70
o.append(f'<text x="48" y="{py}" font-size="16" font-weight="600" fill="{INK}">Accuracy by source, development partition</text>')
lx = 48
for lab, col, shape in (("Jev", INK, "d"), ("Kev-4B", BLUE, "c"), ("spark-s1-4b-v3", ORANGE, "c"), ("spark-s1-1.7b-v3", ORANGE, "r")):
    cx, cy = lx + 7, py + 26
    if shape == "d":
        o.append(f'<polygon points="{cx},{cy - 7} {cx + 7},{cy} {cx},{cy + 7} {cx - 7},{cy}" fill="{col}"/>')
    elif shape == "c":
        o.append(f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="{col}"/>')
    else:
        o.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{SURF}" stroke="{col}" stroke-width="2.5"/>')
    o.append(f'<text x="{cx + 14}" y="{cy + 5}" font-size="13" fill="{INK2}">{lab}</text>')
    lx += 30 + len(lab) * 8 + 26
srcs = [("qnli", "QNLI (entailment)"), ("sciq", "SciQ (science QA)"), ("tweet_offensive", "TweetEval offensive"), ("paws", "PAWS (paraphrase)"), ("emotion", "Emotion (noisy labels)"), ("mmlu", "MMLU (knowledge)"), ("contrastive_deadline", "Deadline (date arithmetic)")]
ax0, ax1, ry0 = 250, 1080, py + 62
sx2 = lambda v: ax0 + (v - 0.2) / 0.8 * (ax1 - ax0)
for g in (0.2, 0.4, 0.6, 0.8, 1.0):
    o.append(f'<line x1="{sx2(g):.1f}" y1="{ry0 - 14}" x2="{sx2(g):.1f}" y2="{ry0 + len(srcs) * 44 - 16}" stroke="{GRID}"/>')
    o.append(f'<text x="{sx2(g):.1f}" y="{ry0 + len(srcs) * 44 + 4}" font-size="12" fill="{MUTE}" text-anchor="middle">{g:.1f}</text>')
for i, (k, lab) in enumerate(srcs):
    cy = ry0 + i * 44 + 4
    o.append(f'<text x="{ax0 - 14}" y="{cy + 5}" font-size="14" fill="{INK}" text-anchor="end">{lab}</text>')
    o.append(f'<line x1="{ax0}" y1="{cy}" x2="{ax1}" y2="{cy}" stroke="{GRID}" stroke-dasharray="2 4"/>')
    vals = [("Jev", pub["per_source_dev"][k]["Jev"], INK, "d"), ("Kev-4B", pub["per_source_dev"][k]["Kev-4B"], BLUE, "c"),
            ("spark-s1-4b-v3", d4s[k], ORANGE, "c"), ("spark-s1-1.7b-v3", d17s[k], ORANGE, "r")]
    for nm, v, col, shp in vals:
        cx = sx2(v)
        t = f"<title>{escape(nm)} {escape(lab)}: {v:.3f}</title>"
        if shp == "d":
            o.append(f'<polygon points="{cx:.1f},{cy - 8} {cx + 8:.1f},{cy} {cx:.1f},{cy + 8} {cx - 8:.1f},{cy}" fill="{col}" stroke="{SURF}" stroke-width="2">{t}</polygon>')
        elif shp == "c":
            o.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="7" fill="{col}" stroke="{SURF}" stroke-width="2">{t}</circle>')
        else:
            o.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="6.5" fill="{SURF}" stroke="{col}" stroke-width="2.5">{t}</circle>')
    o.append(f'<text x="{sx2(d4s[k]):.1f}" y="{cy - 12}" font-size="11" fill="{INK2}" text-anchor="middle">{d4s[k]:.2f}</text>')
fy = ry0 + len(srcs) * 44 + 38
notes = ["spark-s1 v3 is a public Qwen3 backbone fine-tuned on 967 code-labelled decision rows; Kev is trained on 10-13 public datasets plus programmatic policy pairs; Jev's training data is unknown.",
         "This is Kev's suite (their rendering and option order), so it favours Kev. Jev was scored only on the development partition, where spark-s1 was also read once for this chart.",
         "Kev-0.6B deadline accuracy is not published. Numbers: docs/img/kev-comparison.json. Sources: github.com/jaredpalmer/kev (2026-09-20), docs/BENCHMARKS.md B32."]
for j, n in enumerate(notes):
    o.append(f'<text x="48" y="{fy + j * 19}" font-size="12" fill="{INK2}">{escape(n)}</text>')
o.append("</svg>")
open("docs/img/kev-comparison.svg", "w").write("\n".join(o))
json.dump({"overall": {r[0]: {"dev": round(r[1], 4), "locked_test": None if r[2] is None else round(r[2], 4)} for r in rows},
           "per_source_dev": {k: {"Jev": pub["per_source_dev"][k]["Jev"], "Kev-4B": pub["per_source_dev"][k]["Kev-4B"], "spark-s1-4b-v3": round(d4s[k], 4), "spark-s1-1.7b-v3": round(d17s[k], 4)} for k, _ in srcs}},
          open("docs/img/kev-comparison.json", "w"), indent=1)
print("ok", H, fy + 60)
