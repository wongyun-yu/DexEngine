"""E123_HELDOUT_v1 — 64 TACO clips never trained on, for the zero-shot transfer probe of the RL ablation.
Pre-registered: no result looked at; the rule mirrors E123_CLIPS_v1 (select_B0_pool.py):
  pool   = the same 520 clips (canon TACO · bundle kind both · whole sequence · in the CHORD HF release) minus the 64
  trained
  seen   = 47 of the 54 trained (verb,tool,target) triplets that still have an untrained clip — random.Random(42)
  shuffles
           the sorted triplets, the first 47 are taken, one clip each = rng.choice(untrained clips of it sorted by name)
  unseen = all 17 triplets with no trained clip — one clip each, same rng, sorted triplets
  47 + 17 = 64 = the training judge's shape (NE 1024 · 64 rollouts per clip · 4 batches)
Per clip it records: stratum · whether the tool / target object id (tree task_info obj_id) appears in any trained clip
· the V1 single-clip RL completion bin (fail < 5 % · mid < 90 % · high) from rl_clip_owner.tsv — for the reader, not
the rule. The output's header lines are part of the frozen record and are written as they were.
usage: python -m experiments.scoring.clips.select_heldout_v1 <chord_taco_dirs.txt> <rl_clip_owner.tsv> <tree root> <out
tsv>
       (tree root = the SharpA bimanual TACO trees, <data>/taco/trees/rby1_sharpa_wave/bimanual)
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path

from experiments.scoring.clips import E123_CLIPS_V1


def base(c):
    return re.sub(r"_t\d+$", "", c)


def trip(r):
    return r["clip"].split("__")[0]


def bins(r):
    return "fail" if r["rate"] < 5 else ("mid" if r["rate"] < 90 else "high")


def _parse_args(argv):
    """Command line: the CHORD TACO folders, the SharpA outcome table, the tree root and the output tsv."""
    ap = argparse.ArgumentParser(description="Select the 64-clip held-out TACO set of the transfer probe.")
    ap.add_argument("chord_dirs", help="the CHORD release's TACO folders (chord_hf_taco_dirs.txt)")
    ap.add_argument("owner_tsv", help="the SharpA per-clip outcome table (rl_clip_owner.tsv)")
    ap.add_argument("tree_root", help="folder of the <clip>_pl_rt trees (task_info.json object ids)")
    ap.add_argument("out", help="output tsv")
    a = ap.parse_args(argv)
    return a


def _load_rows(CH, OWN):
    """The CHORD folder per clip and the canonical TACO rows of the outcome table."""
    ch = {}
    with open(CH) as fh:
        for ln in fh:
            m = re.match(r"taco/(taco_(.+?)__(.+?)__(.+)_(\d{8}_\d+))$", ln.strip())
            full, v, t, o, d = m.groups()
            ch[f"{v.replace('_', '-')}_{t}_{o}__{d}"] = full
    with open(OWN) as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["state"] == "canon" and r["dataset"] == "taco"]
    return ch, rows


def _pool_and_train(rows, ch):
    """The 520-clip pool (in CHORD, unsegmented, both hands) and the 64 trained clips of E123_CLIPS_v1."""
    pool = [
        r
        for r in rows
        if base(r["clip"]) in ch and not re.search(r"_t\d+$", r["clip"]) and r["bundle"].split(".")[1] == "both"
    ]
    assert len(pool) == 520 and len({r["clip"] for r in pool}) == 520, len(pool)
    tr = [
        line.split("\t") for line in Path(E123_CLIPS_V1).read_text().splitlines() if line and not line.startswith("#")
    ][1:]
    train = {x[1] for x in tr}
    assert len(train) == 64 and all(c in {r["clip"] for r in pool} for c in train)
    return pool, tr, train


def _pick_heldout(tr, pool, train):
    """47 seen-triplet and 17 unseen-triplet clips outside the trained set, one per triplet (seed 42)."""
    ttrip = {x[3] for x in tr}
    rest = {}
    for r in pool:
        if r["clip"] not in train:
            rest.setdefault(trip(r), []).append(r)
    rng = random.Random(42)
    seen_t = sorted(t for t in ttrip if t in rest)
    rng.shuffle(seen_t)
    assert len(seen_t) == 54, len(seen_t)
    unseen_t = sorted({trip(r) for r in pool} - ttrip)
    assert len(unseen_t) == 17, len(unseen_t)
    pick = [("seen_triplet", rng.choice(sorted(rest[t], key=lambda r: r["clip"]))) for t in seen_t[:47]]
    pick.extend(("unseen_triplet", rng.choice(sorted(rest[t], key=lambda r: r["clip"]))) for t in unseen_t)
    assert len(pick) == 64 and len({r["clip"] for _, r in pick}) == 64 and not ({r["clip"] for _, r in pick} & train)
    return pick


def _write_heldout(OUT, sha, pick, objs, ch, tobj):
    """The held-out tsv (header lines, names sha256, one row per clip), written atomically."""
    with open(OUT + ".tmp", "w") as f:
        f.write(
            "# E123_HELDOUT_v1 — 64 TACO clips never trained on (zero-shot transfer probe · RL ablation). "
            "No re-selection.\n"
        )
        f.write(
            "# rule = clips/select_heldout_v1.py docstring (seed 42 · 47 seen-triplet + 17 "
            "unseen-triplet · no result looked at)\n"
        )
        f.write(f"# names_sha256 = {sha}\n")
        f.write(
            "idx\tclip\tchord_dir\ttriplet\tstratum\tright_obj\tleft_obj\tright_obj_trained\tleft_obj_trained\tv1_bin\n"
        )
        for i, (s, r) in enumerate(pick):
            o = objs(r["clip"])
            f.write(
                f"{i}\t{r['clip']}\t{ch[base(r['clip'])]}\t{trip(r)}\t{s}\t{o.get('right', '')}\t{o.get('left', '')}\t"
                f"{int(o.get('right', '') in tobj)}\t{int(o.get('left', '') in tobj)}\t{bins(r)}\n"
            )
    os.replace(OUT + ".tmp", OUT)


def _say_heldout(pick, objs, tobj, sha):
    """One HELDOUT line: strata, bins, clips with an untrained object, sha prefix."""
    print(
        "[HELDOUT] 64 ·",
        dict(collections.Counter(s for s, _ in pick)),
        "· bins",
        dict(collections.Counter((s, bins(r)) for s, r in pick)),
        "· clips with an untrained object",
        sum(1 for _, r in pick if not set(objs(r["clip"]).values()) <= tobj),
        "· sha",
        sha[:16],
    )


def main(argv=None):
    a = _parse_args(argv)
    CH, OWN, TREE, OUT = a.chord_dirs, a.owner_tsv, a.tree_root, a.out
    ch, rows = _load_rows(CH, OWN)

    pool, tr, train = _pool_and_train(rows, ch)

    pick = _pick_heldout(tr, pool, train)

    def objs(clip):
        ti = json.loads(Path(f"{TREE}/{clip}_pl_rt/task_info.json").read_text())
        return {o["side"]: str(o["obj_id"]) for o in ti.get("objects", [])}

    tobj = set().union(*(set(objs(c).values()) for c in train))

    for _, r in pick:
        r["rate"] = int(r["complete"]) / max(1, int(r["total"])) * 100
    names = [r["clip"] for _, r in pick]
    sha = hashlib.sha256("".join(n + "\n" for n in names).encode()).hexdigest()
    _write_heldout(OUT, sha, pick, objs, ch, tobj)
    _say_heldout(pick, objs, tobj, sha)


if __name__ == "__main__":
    main()
