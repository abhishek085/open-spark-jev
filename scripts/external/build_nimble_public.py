"""Build Bespoke Nimble's 13 human-labelled public benchmark subsets (Apache-2.0 tooling; each dataset keeps its own license) as separate
external sources for spark-s1: data/benchmarks/external/ext-nimble-<subset>.jsonl.

Nimble's repository is cloned to data/raw/nimble_public/nimble_src (not committed). Raw upstream files are downloaded to
data/raw/nimble_public/raw/ (not committed). Each subset is rebuilt with Nimble's own converter, selecting exactly the ids in the manifest that
Nimble commits, then converted to our Record format. Nothing is pooled: one file per subset.

  python scripts/external/build_nimble_public.py [subset ...]
"""
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

from open_spark_jev.data.corpus import Record, write_jsonl

ROOT = "data/raw/nimble_public"
SRC = f"{ROOT}/nimble_src"
RAW = f"{ROOT}/raw"
OUT_DIR = "data/benchmarks/external"
MAN = f"{SRC}/docs/assets/public-benchmarks/subsets"
HF = {  # name -> (repo, file in repo, local jsonl)
    "boolq": ("google/boolq", "data/validation-00000-of-00001.parquet", "boolq/validation.jsonl"),
    "squad_v2": ("rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet", "squad_v2/validation.jsonl"),
    "paws": ("google-research-datasets/paws", "labeled_final/test-00000-of-00001.parquet", "paws/test.jsonl"),
    "civil_comments": ("google/civil_comments", "data/test-00000-of-00001.parquet", "civil_comments/test.jsonl"),
    "summeval": ("mteb/summeval", "data/test-00000-of-00001-35901af5f6649399.parquet", "summeval/test.jsonl"),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", "pubmedqa/train.jsonl"),
}
# subset -> (converter dataset, raw file relative to RAW, extra args)
SUBSETS = {
    "vitaminc-dev": ("vitaminc", "vitaminc/dev.jsonl", []),
    "massive-en-US": ("massive", "massive/amazon-massive-dataset-1.1.tar.gz", ["--subset", "en-US"]),
    "boolq": ("boolq", "boolq/validation.jsonl", []),
    "squad2": ("squad2", "squad_v2/validation.jsonl", []),
    "paws": ("paws", "paws/test.jsonl", []),
    "multinli": ("multinli", "multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl", []),
    "civil_comments": ("civil_comments", "civil_comments/test.jsonl", []),
    "aegis2": ("aegis2", "aegis2/test.jsonl", []),
    "helpsteer2": ("helpsteer2", "helpsteer2/validation.jsonl", []),
    "summeval-relevance": ("summeval", "summeval/test.jsonl", ["--subset", "relevance"]),
    "summeval-consistency": ("summeval", "summeval/test.jsonl", ["--subset", "consistency"]),
    "pubmedqa": ("pubmedqa", "pubmedqa/train.jsonl", []),
}


def fetch(url, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("download", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "open-spark-jev"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def download():
    if not os.path.isdir(SRC):
        subprocess.run(["git", "clone", "-q", "https://github.com/bespokelabsai/nimble", SRC], check=True)
    import pandas as pd
    from huggingface_hub import hf_hub_download

    for name, (repo, fn, local) in HF.items():
        dest = f"{RAW}/{local}"
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        p = hf_hub_download(repo, fn, repo_type="dataset")
        pd.read_parquet(p).to_json(dest, orient="records", lines=True, force_ascii=False)
    ag = f"{RAW}/aegis2/test.jsonl"
    fetch("https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/resolve/main/test.json", ag)
    txt = open(ag, encoding="utf-8").read()
    if txt.lstrip().startswith("["):
        with open(ag, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in json.loads(txt)))
    hs = f"{RAW}/helpsteer2/validation.jsonl"
    if not os.path.exists(hs):
        fetch("https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/validation.jsonl.gz", hs + ".gz")
        with gzip.open(hs + ".gz") as g, open(hs, "wb") as f:
            shutil.copyfileobj(g, f)
    fetch("https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz", f"{RAW}/massive/amazon-massive-dataset-1.1.tar.gz")
    vz = f"{RAW}/vitaminc/vitaminc.zip"
    fetch("https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip", vz)
    if not os.path.exists(f"{RAW}/vitaminc/dev.jsonl"):
        with zipfile.ZipFile(vz) as z:
            member = next(n for n in z.namelist() if n.endswith("dev.jsonl") and "__MACOSX" not in n)
            with z.open(member) as s, open(f"{RAW}/vitaminc/dev.jsonl", "wb") as f:
                shutil.copyfileobj(s, f)
    mz = f"{RAW}/multinli/multinli_1.0.zip"
    fetch("https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip", mz)
    if not os.path.exists(f"{RAW}/multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl"):
        with zipfile.ZipFile(mz) as z:
            z.extractall(f"{RAW}/multinli")


def convert_all(only):
    env = dict(os.environ, PYTHONPATH=SRC)
    for sub, (ds, raw, extra) in SUBSETS.items():
        if only and sub not in only:
            continue
        out = f"{ROOT}/built/{sub}"
        if os.path.exists(f"{out}/all.jsonl"):
            continue
        manifest = f"{MAN}/{sub}-manifest.json"
        cmd = [sys.executable, "-m", "nimble.datasets.public_benchmarks", "--dataset", ds, "--source", f"{RAW}/{raw}", "--output-dir", out, "--ids-from", manifest] + extra
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0 and "partially covered" in (r.stdout + r.stderr):
            shutil.rmtree(out, ignore_errors=True)
            full = out + "_full"
            shutil.rmtree(full, ignore_errors=True)
            cmd2 = [sys.executable, "-m", "nimble.datasets.public_benchmarks", "--dataset", ds, "--source", f"{RAW}/{raw}", "--output-dir", full, "--limit", "1000000"] + extra
            r2 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
            keep = set(json.load(open(manifest))["ids"])
            os.makedirs(out, exist_ok=True)
            n = 0
            with open(f"{full}/all.jsonl", encoding="utf-8", newline="\n") as fi, open(f"{out}/all.jsonl", "w", encoding="utf-8", newline="\n") as fo:
                for line in fi:
                    if line.strip() and json.loads(line)["id"] in keep:
                        fo.write(line)
                        n += 1
            print(sub, f"ok (filtered to the manifest ids: {n} of {len(keep)})", flush=True)
            continue
        print(sub, "ok" if r.returncode == 0 else "FAIL", (r.stdout + r.stderr).strip().splitlines()[-1][:150], flush=True)
    if not only or "massive-de-DE" in only:
        out = f"{ROOT}/built/massive-de-DE"
        if not os.path.exists(f"{out}/all.jsonl"):
            cmd = [sys.executable, "-m", "nimble.datasets.public_benchmarks", "--dataset", "massive", "--source", f"{RAW}/massive/amazon-massive-dataset-1.1.tar.gz",
                   "--output-dir", out, "--subset", "de-DE", "--ids-from", f"{ROOT}/built/massive-en-US/manifest.json"]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print("massive-de-DE", "ok" if r.returncode == 0 else "FAIL", (r.stdout + r.stderr).strip().splitlines()[-1][:150], flush=True)


def to_records(sub):
    built = f"{ROOT}/built/{sub}/all.jsonl"
    recs, skipped = [], 0
    for line in open(built, encoding="utf-8", newline="\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r["input"]
        (qn, q), = inp["questions"].items()
        crit = q.get("criteria")
        ref = r["reference"]
        target = ref.get("label", ref.get("target"))
        typ = q["type"]
        if typ == "noul":
            question = {"type": "noul", "prompt": q["instructions"]}
            label = "yes" if str(target).lower() in ("true", "1", "yes") else "no"
        elif typ == "choice":
            keys = list(crit)
            body = "\n".join(f"- {k}: {v}" for k, v in crit.items())
            if len(keys) > 26:
                skipped += 1
                continue
            question = {"type": "choice", "prompt": q["instructions"] + "\nOption definitions:\n" + body, "options": keys}
            label = str(target)
        else:
            levels = crit if isinstance(crit, list) else [crit[k] for k in sorted(crit, key=lambda x: int(x))]
            question = {"type": "score", "prompt": q["instructions"], "levels": [str(i) for i in range(len(levels))], "rubric": "\n".join(f"{i}: {d}" for i, d in enumerate(levels))}
            label = str(int(target))
        meta = {"family": r.get("family"), "upstream": sub}
        if ref.get("distribution"):
            meta["human_distribution"] = ref["distribution"]
        recs.append(Record(id=f"ext-nimble-{sub}/{r['id']}", domain=f"ext-nimble-{sub}", source=f"ext-nimble-{sub}", state={"content": inp["state"], "domain": f"ext-nimble-{sub}"},
                           question={**question, "allow_abstain": False}, target={"label": label}, meta=meta))
    os.makedirs(OUT_DIR, exist_ok=True)
    write_jsonl(f"{OUT_DIR}/ext-nimble-{sub}.jsonl", recs)
    print(f"ext-nimble-{sub}: {len(recs)} records (skipped {skipped})", flush=True)


if __name__ == "__main__":
    only = sys.argv[1:]
    download()
    convert_all(only)
    for sub in list(SUBSETS) + ["massive-de-DE"]:
        if (not only or sub in only) and os.path.exists(f"{ROOT}/built/{sub}/all.jsonl"):
            to_records(sub)
