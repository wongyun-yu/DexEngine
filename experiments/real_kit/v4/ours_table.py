#!/usr/bin/env python3
"""The 'ours' row table of the v4 set manifest — the v4 variant rows of the v2 ours training table (owner of text,
labels and split), with video_frames set to the v4 look-D render length.

Why: vis_manifest.py --d-in-pieces needs the look-D episode length to equal the ours row's video_frames. v4 episodes are
longer (the approach comes first), so the v2 table would die at that gate. The manifest code stays unchanged; this
table moves the length owner to the v4 look-D render (pieces' meta/episodes.jsonl). Text, labels and split stay as in
the v2 table. Checks: exactly one v2 row and one look-D episode per variant, look-D length > v2 length (the approach
makes it longer) — else death.
  python -m experiments.real_kit.v4.ours_table --v2 <v2 ours_train.tsv> --pieces <extracted piece root> --variants
  a,b,…
      --out <v4 ours.tsv>   (.roots copied from v2)
Last line [V4_OURS] OK|FAIL
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys

TAG = "[V4_OURS]"


def die(m: str):
    print(f"{TAG} FAIL {m}", flush=True)
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--v2", "--pieces", "--variants", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    want = [v for v in a.variants.split(",") if v]
    if not want or len(set(want)) != len(want):
        die(f"bad variant list {a.variants!r}")
    with open(a.v2) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    cols = list(rows[0].keys()) if rows else []
    by = {}
    for r in rows:
        if r["clip"] in want:
            if r["clip"] in by:
                die(f"{r['clip']} twice in the v2 table")
            by[r["clip"]] = r
    miss = [v for v in want if v not in by]
    if miss:
        die(f"variants not in the v2 table {miss[:5]} (all {len(miss)})")
    d = {}
    for f in glob.glob(
        os.path.join(a.pieces, "*", "out_*", "root", "data", "*", "collect", "*", "meta", "episodes.jsonl")
    ):
        with open(f) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                e = json.loads(ln)
                if e["clip"] in want and (e.get("look") or {}).get("name", e.get("look")) == "D":
                    if e["clip"] in d:
                        die(f"look D episode {e['clip']} twice")
                    d[e["clip"]] = int(e["video_frames"])
    miss = [v for v in want if v not in d]
    if miss:
        die(f"look D render missing {miss[:5]} (all {len(miss)})")
    short = [(v, by[v]["video_frames"], d[v]) for v in want if d[v] <= int(by[v]["video_frames"])]
    if short:
        die(f"v4 look D length ≤ v2 (no approach?) {short[:3]}")
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for v in want:
            w.writerow(dict(by[v], video_frames=str(d[v])))
    shutil.copyfile(a.v2 + ".roots", a.out + ".roots")
    add = [d[v] - int(by[v]["video_frames"]) for v in want]
    print(
        f"{TAG} OK variants {len(want)} · added frames min/med/max "
        f"{min(add)}/{sorted(add)[len(add) // 2]}/{max(add)} · {a.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
