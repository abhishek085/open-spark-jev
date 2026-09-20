"""Option-order augmentation of data/synthetic/osdg_train.jsonl: each choice record is emitted in its original order plus
K random permutations of its options (label text unchanged, so the target stays valid). Score/Noul rows are unchanged."""
import copy, json, random, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
out = []
for line in open("data/synthetic/osdg_train.jsonl"):
    r = json.loads(line)
    out.append(r)
    if r["question"]["type"] != "choice":
        continue
    for k in range(1, K + 1):
        c = copy.deepcopy(r)
        opts = list(c["question"]["options"])
        random.Random(f"aug-{k}-{r['id']}").shuffle(opts)
        if opts == r["question"]["options"]:
            continue
        c["question"]["options"] = opts
        c["id"] = f"{r['id']}-p{k}"
        out.append(c)
open("data/synthetic/osdg_train_aug.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in out))
print(len(out), "rows")
