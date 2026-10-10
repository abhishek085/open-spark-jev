"""Accuracy / confidence / latency of the decision model as the state grows (tool-call-risk holdout, call placed at the end of a long benign transcript).

  python scripts/eval_longctx.py --gateway http://localhost:8401 --key K --tokens 0,4000,16000 --n 100 --out runs/longctx/x.jsonl
Per-row predictions are written to --out (one line per item x length); a summary table is printed.
"""
import argparse, asyncio, glob, json, random, time

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--gateway", default="http://localhost:8400")
ap.add_argument("--key", default="")
ap.add_argument("--tokens", default="0,4000,16000")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data/benchmarks/toolcall_risk_holdout.jsonl")
ap.add_argument("--subset-n", type=int, default=0, help="for lengths > 0: re-test only this many items = the least-confident/wrong at length 0 (half) + random others (half)")
ap.add_argument("--hard-first", action="store_true")
ap.add_argument("--max-soc-c", type=float, default=88.0)
a = ap.parse_args()

CMDS = ["pytest -q tests/unit", "ls -la src/", "git status --short", "cat README.md", "npm run build", "cargo check", "grep -rn TODO src | head", "python -m mypy pkg", "docker ps", "make lint"]
OUT = ["collected 214 items\n214 passed in 11.2s", "total 48\ndrwxr-xr-x 6 dev dev 4096 .\n-rw-r--r-- 1 dev dev 1822 setup.cfg", "M  src/app.py\n M tests/test_app.py", "Compiling pkg v0.3.1\nFinished dev profile in 4.31s", "no issues found in 62 source files", "CONTAINER ID   IMAGE   STATUS\n3f2a1c   api:dev   Up 2 hours"]


def filler(chars: int, rng: random.Random) -> str:
    out, n = [], 0
    while n < chars:
        s = f"[{rng.randint(8,18):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}] assistant ran: {rng.choice(CMDS)}\n{rng.choice(OUT)}\n"
        out.append(s); n += len(s)
    return "".join(out)


def soc_c() -> float:
    v = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try: v.append(int(open(z).read()) / 1000)
        except Exception: pass
    return max(v) if v else 0.0


async def main():
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if r["question"]["type"] == "choice" and r["target"].get("label")]
    random.Random(7).shuffle(rows)
    if a.hard_first:  # hardest items first: easy ones saturate and hide calibration drift
        rows.sort(key=lambda r: r["meta"].get("difficulty") == "easy")
    rows = rows[: a.n]
    lens = [int(x) for x in a.tokens.split(",")]
    hdr = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    base: dict[str, dict] = {}
    fo = open(a.out, "w")
    async with httpx.AsyncClient(timeout=600, headers=hdr) as c:
        for L in lens:
            items = rows
            if L and a.subset_n and base:  # half least-confident / wrong at length 0, half random
                ranked = sorted(rows, key=lambda r: (base[r["id"]]["pred"] == base[r["id"]]["gold"], base[r["id"]]["conf"]))
                hard, rest = ranked[: a.subset_n // 2], ranked[a.subset_n // 2 :]
                items = hard + random.Random(3).sample(rest, min(len(rest), a.subset_n - len(hard)))
            hit = conf = 0.0; lat = []; toks = []
            for i, r in enumerate(items):
                while soc_c() >= a.max_soc_c:
                    print(f"  SoC {soc_c():.0f}C >= {a.max_soc_c}: pausing 20s", flush=True); await asyncio.sleep(20)
                rng = random.Random(i * 1009 + L)
                state = (filler(int(L * 1.9), rng) + "\n" if L else "") + (r["state"]["content"] if isinstance(r["state"]["content"], str) else json.dumps(r["state"]["content"], indent=1))
                q = r["question"]
                body = {"state": state, "questions": {"risk": {"type": "choice", "instructions": q["prompt"], "criteria": {o: "" for o in q["options"]}}}}
                t = time.perf_counter()
                resp = await c.post(a.gateway + "/v1/evaluate", json=body)
                ms = (time.perf_counter() - t) * 1000
                if resp.status_code != 200:
                    print("ERR", resp.status_code, resp.text[:200]); continue
                j = resp.json(); ans = j["answers"]["risk"]
                gold = r["target"]["label"]
                hit += ans["choice"] == gold; conf += ans["confidence"]; lat.append(ms); toks.append(j["usage"]["input_tokens"] or 0)
                rec = {"len_target": L, "id": r["id"], "gold": gold, "pred": ans["choice"], "conf": ans["confidence"], "probs": ans["probabilities"], "input_tokens": j["usage"]["input_tokens"], "ms": round(ms, 1)}
                if L == 0: base[r["id"]] = rec
                fo.write(json.dumps(rec) + "\n"); fo.flush()
            n = len(lat); lat.sort()
            print({"target_tokens": L, "actual_tokens_p50": sorted(toks)[n // 2], "n": n, "acc": round(hit / n, 3), "mean_conf": round(conf / n, 3), "ms_p50": round(lat[n // 2]), "ms_p90": round(lat[int(n * .9)])}, flush=True)


asyncio.run(main())
