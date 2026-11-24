"""E5 split — per task × hand use (one hand, two hands) 80 % training / 20 % test by recording (seed 42). Input =
wuji2_e5_clips.tsv (the version with the edited keep and axis columns). Recording (group) = the clip name without the
segment (_seg..), retake (_retake), object piece (_t<i>, D36) and OakInk2 sub-task (_s000) suffixes — the clips of one
recording go to one side only. A recording spanning several tasks becomes test in all of them when it is drawn for test
(only if no task exceeds its 20 % test share)."""

import collections
import csv
import hashlib
import pathlib
import re

HERE = pathlib.Path(__file__).parent
SEED, TEST_FRAC = 42, 0.20


def group_of(ds, clip):
    g = clip
    while True:  # suffixes may stack in any order — strip until nothing is left
        h = re.sub(r"(_seg\d+|_retake|_t\d+)$", "", g)
        if h == g:
            break
        g = h
    if ds == "oakink2":
        g = re.sub(r"_s\d{3}$", "", g)
        g = g.split("__", 1)[1] if "__" in g else g  # recording = scene hash (motion category prefix removed)
    return f"{ds}/{g}"


with open(HERE / "wuji2_e5_clips.tsv") as fh:
    rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["keep"].strip() == "1" and r["axis"].strip()]
for r in rows:
    r["group"] = group_of(r["dataset"], r["clip"])


def hand(r):
    return "uni" if r["hand_use"].startswith("uni") else "bi"


tasks_of = collections.defaultdict(lambda: collections.Counter())  # recording → clip count per (task, hand use) cell
for r in rows:
    tasks_of[r["group"]][(r["axis"], hand(r))] += 1
total = collections.Counter((r["axis"], hand(r)) for r in rows)
test_n, test_groups = collections.Counter(), set()
order = sorted(tasks_of, key=lambda g: hashlib.sha1(f"{SEED}:{g}".encode()).hexdigest())
for g in order:
    if all(test_n[a] + k <= TEST_FRAC * total[a] for a, k in tasks_of[g].items()):
        test_groups.add(g)
        test_n.update(tasks_of[g])
out = HERE / "e5_split.tsv"
cols = ["split", "axis", "dataset", "hand_use", "clip", "group"]
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["axis"], r["dataset"], r["clip"])):
        w.writerow({**r, "split": "test" if r["group"] in test_groups else "train"})
sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
print(f"[E5_SPLIT] seed={SEED} clips={len(rows)} groups={len(tasks_of)} sha={sha}")
leak = {g for g in tasks_of if g in test_groups} & {r["group"] for r in rows if r["group"] not in test_groups}
assert not leak, f"recordings on both the training and the test side: {sorted(leak)[:5]}"
for a, h in sorted(total):
    print(f"  {a} {h}: test {test_n[(a, h)]}/{total[(a, h)]} ({100 * test_n[(a, h)] / total[(a, h)]:.1f} %)")
