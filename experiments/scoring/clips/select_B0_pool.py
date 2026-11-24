"""The random candidate set the frozen Tab. 2 clip set was drawn from (the record of how E123_CLIPS_v1 began).

Inputs: the TACO folders of the CHORD release (``chord_hf_taco_dirs.txt`` here, one
``taco/taco_<verb>__<tool>__<obj>_<date>`` per line) and our SharpA per-clip table of the canonical TACO clips
(``rl_clip_owner.tsv``: clip · bundle · state · dataset · complete · total · verb). The pool = canonical TACO clips that
CHORD also released, not segments (no ``_t<k>`` suffix), from both-hand bundles. The candidate set:
  B0 random  — 64 verb-tool-object triplets drawn at random (seed 42), one clip each; no success rate is used to draw
Prints the pool statistics and writes <out-dir>/cand_B0_random.tsv (its complete · total · rate_pct · bin columns are
recorded for the reader, not used by the draw). The frozen list then replaced some B0 clips (``provenance/``).
usage: python -m experiments.scoring.clips.select_B0_pool --owner-tsv <rl_clip_owner.tsv> --out-dir <dir>
       [--chord-dirs <list>]   (default: chord_hf_taco_dirs.txt next to this file)
"""

import argparse
import collections
import csv
import os
import random
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def seg(c):
    return bool(re.search(r"_t\d+$", c))


def base(c):
    return re.sub(r"_t\d+$", "", c)


def cls(b):
    return b.split(".")[1]


def summ(name, L):
    rs = [r["rate"] for r in L]
    if not rs:
        print(name, 0)
        return
    q = st.quantiles(rs, n=4) if len(rs) > 1 else [rs[0]] * 3
    print(
        f"{name}: n={len(L)} SR5={100 * sum(x >= 5 for x in rs) / len(rs):.1f}% "
        f"median={st.median(rs):.1f} q1={q[0]:.1f} q3={q[2]:.1f} mean={st.mean(rs):.1f} "
        f"frac>=99={100 * sum(x >= 99 for x in rs) / len(rs):.1f}% trips={len({r['trip'] for r in L})} "
        f"verbs={len({r['verb'] for r in L})}"
    )


def bins(r):
    return "fail" if r["rate"] < 5 else ("mid" if r["rate"] < 90 else "high")


def _parse_args(argv):
    """Command line: the CHORD TACO folders, the SharpA per-clip outcome table and the output folder."""
    ap = argparse.ArgumentParser(description="Draw the random 64-clip candidate set of the Tab. 2 clip set.")
    ap.add_argument(
        "--chord-dirs", default=str(HERE / "chord_hf_taco_dirs.txt"), help="the CHORD release's TACO folders"
    )
    ap.add_argument("--owner-tsv", required=True, help="the SharpA per-clip outcome table (rl_clip_owner.tsv)")
    ap.add_argument("--out-dir", required=True, help="where cand_B0_random.tsv is written")
    a = ap.parse_args(argv)
    return a


def _load_rows(a):
    """The CHORD folder per clip and the canonical TACO rows with their rate, class, CHORD membership and triplet."""
    ch = {}
    with open(a.chord_dirs) as fh:
        for line in fh:
            m = re.match(r"taco/(taco_(.+?)__(.+?)__(.+)_(\d{8}_\d+))$", line.strip())
            full, v, t, o, d = m.groups()
            ch[f"{v.replace('_', '-')}_{t}_{o}__{d}"] = full
    with open(a.owner_tsv) as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["state"] == "canon" and r["dataset"] == "taco"]

    for r in rows:
        r["rate"] = int(r["complete"]) / max(1, int(r["total"])) * 100
        r["cls"] = cls(r["bundle"])
        r["inch"] = base(r["clip"]) in ch
        r["seg"] = seg(r["clip"])
        r["trip"] = r["clip"].split("__")[0]
        r["verb"] = r["trip"].split("_")[0]
    return ch, rows


def _print_rate_summaries(rows):
    """Rate summaries of all rows, the CHORD rows and the CHORD unsegmented rows per class."""
    summ("all canon taco", rows)
    summ("inCHORD", [r for r in rows if r["inch"]])
    summ("inCHORD noseg", [r for r in rows if r["inch"] and not r["seg"]])
    for c in sorted({r["cls"] for r in rows}):
        summ(" inCHORD noseg cls=" + c, [r for r in rows if r["inch"] and not r["seg"] and r["cls"] == c])
    print(collections.Counter(r["verb"] for r in rows if r["inch"] and not r["seg"]).most_common())
    print("total per-clip K:", collections.Counter(r["total"] for r in rows if r["inch"]).most_common(5))


def _draw_pool(pool):
    """Print the pool bins; draw B0 random (64 triplets, one clip per triplet)."""
    print(
        "pool bins",
        collections.Counter(bins(r) for r in pool),
        "triplets per bin",
        {b: len({r["trip"] for r in pool if bins(r) == b}) for b in ["fail", "mid", "high"]},
    )
    rng = random.Random(42)
    trips = sorted({r["trip"] for r in pool})
    rng.shuffle(trips)
    B0 = []
    for t in trips[:64]:
        c = sorted([r for r in pool if r["trip"] == t], key=lambda r: r["clip"])
        B0.append(rng.choice(c))
    return B0


def _print_set(B0):
    """Rate summary, CHORD count, verbs and source bundles of the B0 random set."""
    summ("B0 random", B0)
    print(
        "   inCHORD",
        sum(r["inch"] for r in B0),
        "verbs",
        dict(collections.Counter(r["verb"] for r in B0).most_common(6)),
        "src bundles",
        len({r["bundle"] for r in B0}),
    )


def _write_candidates(B0, a, ch):
    """cand_B0_random.tsv in the output folder."""
    with open(os.path.join(a.out_dir, "cand_B0_random.tsv"), "w") as f:
        f.write("clip\tchord_dir\tour_bundle\tcomplete\ttotal\trate_pct\tbin\n")
        for r in sorted(B0, key=lambda r: r["clip"]):
            f.write(
                f"{r['clip']}\t{ch[base(r['clip'])]}\t{r['bundle']}\t{r['complete']}\t{r['total']}\t"
                f"{r['rate']:.1f}\t{bins(r)}\n"
            )


def main(argv=None):
    a = _parse_args(argv)
    ch, rows = _load_rows(a)

    _print_rate_summaries(rows)
    pool = [r for r in rows if r["inch"] and not r["seg"] and r["cls"] == "both"]

    B0 = _draw_pool(pool)
    _print_set(B0)
    _write_candidates(B0, a, ch)


if __name__ == "__main__":
    main()
