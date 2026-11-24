#!/usr/bin/env python3
"""Exclude list of the GR00T mid-training corpus (reads only): the excluded source recordings → every clip of the same
original recording (the split table's group, i.e. the other segments too) → every episode of those clips in both
roots, in the manifest --exclude form (dataset · root · clip · look). A recording that matches no single dataset
dies.

  python -m experiments.policies.mid.exclude --roots orig=<dir>,aug=<dir> <e5_split.tsv> <excl_src.tsv> <out.tsv>
  <summary.json>

Last line `[MIDEXCL] OK …`.
"""

import argparse
import collections
import csv
import json
import re
import sys

from experiments.textio import read_lines

DSS = ["arctic", "dexcanvas", "grab", "h2o", "hocap", "hot3d", "hrdexdb", "oakink2", "taco"]
#: Original recording of a clip outside the split table (the table holds only the E5 task clips), for datasets with a
#: rule: hot3d clip '<sequence>__s<start>_e<end>[…]' → <sequence> (every time window of the sequence, the same meaning
#: as the table's 'other segments too'). An off-table recording of a dataset without a rule dies (no silent pass).
UNSPLIT_RULE = {"hot3d": lambda c: c.split("__", 1)[0]}


def base(c):
    return re.sub(r"_pv\d+$", "", c.removesuffix("_pl_rt"))


def look(e):
    return str(e["look"].get("name") if isinstance(e.get("look"), dict) else (e.get("look") or ""))


def _read_episodes(a):
    """Read every episode of both roots as (dataset, root tag, clip, look, base clip); index base clip → datasets."""
    roots = dict(r.split("=", 1) for r in a.roots.split(","))
    eps = []
    for tag, R in roots.items():
        for ds in DSS:
            with open(f"{R}/data/{ds}/collect/rby1_wuji2/meta/episodes.jsonl") as fh:
                for ln in fh:
                    if ln.strip():
                        e = json.loads(ln)
                        eps.append((ds, tag, e["clip"], look(e), base(e["clip"])))
    bases = collections.defaultdict(set)
    for ds, _, _, _, b in eps:
        bases[b].add(ds)
    return eps, bases


def _read_split(a):
    """Read the split table; index (dataset, clip) → row and (dataset, group) → clips."""
    with open(a.split) as fh:
        split = list(csv.DictReader(fh, delimiter="\t"))
    sp = {(r["dataset"], r["clip"]): r for r in split}
    grp = collections.defaultdict(set)
    for r in split:
        grp[(r["dataset"], r["group"])].add(r["clip"])
    return sp, grp


def _source_recordings(a, bases):
    """Read the exclude list and resolve each recording to its one dataset; returns (dataset, base clip) → sources."""
    src = collections.defaultdict(set)
    for r in csv.DictReader(read_lines(a.excl), delimiter="\t"):
        d = bases.get(r["base_clip"], set())
        if r["dataset"]:
            d = d & {r["dataset"]}
        if len(d) != 1:
            sys.exit(
                f"[MIDEXCL] FAIL recording {r['base_clip']} ({r['source']}) datasets "
                f"{sorted(bases.get(r['base_clip'], set()))}"
            )
        src[(next(iter(d)), r["base_clip"])].add(r["source"].split(":")[0])
    return src


def _expand_groups(eps, src, sp, grp):
    """Expand every excluded recording to all clips of its split group (or of its rule group when off the table)."""
    corpus_bases = collections.defaultdict(set)
    for ds, _, _, _, b in eps:
        corpus_bases[ds].add(b)
    ex, unsplit = {}, []
    for (ds, b), s in src.items():
        if (ds, b) in sp:
            g = sp[(ds, b)]["group"]
            members = grp[(ds, g)]
        elif ds in UNSPLIT_RULE:
            key = UNSPLIT_RULE[ds](b)
            g = f"{ds}/{key} (outside the split table · rule)"  # data: stored in the exclude list's group column
            members = {c for c in corpus_bases[ds] if UNSPLIT_RULE[ds](c) == key}
            unsplit.append((ds, b, g, len(members)))
        else:
            sys.exit(f"[MIDEXCL] FAIL {ds}/{b} is not in the split table and the dataset has no group rule")
        for c in members:
            ex.setdefault((ds, c), {"group": g, "via": b if c != b else "", "src": set()})["src"] |= s
    for u in unsplit:
        print(f"[MIDEXCL] off-table {u[0]}/{u[1]} → group {u[2]} · clips {u[3]}")
    return ex, unsplit


def _write_exclude(eps, ex, a):
    """Write the per-episode exclude table; returns its rows."""
    rows = [(ds, tag, clip, lk) for ds, tag, clip, lk, b in eps if (ds, b) in ex]
    with open(a.out, "w") as fo:
        fo.write("dataset\troot\tclip\tlook\tbase_clip\tgroup\tsource\n")
        for ds, tag, clip, lk in sorted(rows):
            b = base(clip)
            fo.write(
                f"{ds}\t{tag}\t{clip}\t{lk}\t{b}\t{ex[(ds, b)]['group']}\t{','.join(sorted(ex[(ds, b)]['src']))}\n"
            )
    return rows


def _write_summary(rows, eps, ex, src, unsplit, a):
    """Write the summary JSON (counts per dataset and root, groups added by expansion)."""
    per = collections.Counter((ds, tag) for ds, tag, _, _ in rows)
    allc = collections.Counter((ds, tag) for ds, tag, *_ in eps)
    added = sorted(
        f"{ds}/{c} (group {v['group']} · source {v['via']})" for (ds, c), v in ex.items() if v["via"]
    )  # data text
    payload = {
        "recordings_src": len(src),
        "unsplit_groups": [list(u) for u in unsplit],
        "group_clips": len(ex),
        "group_added": added,
        "episodes_excluded": len(rows),
        "per": [
            {
                "dataset": d,
                "root": t,
                "before": allc[(d, t)],
                "excluded": per[(d, t)],
                "after": allc[(d, t)] - per[(d, t)],
            }
            for d, t in sorted(allc)
        ],
    }
    with open(a.summary, "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", required=True, help="orig=<collect root>,aug=<collect root>")
    ap.add_argument("split")
    ap.add_argument("excl")
    ap.add_argument("out")
    ap.add_argument("summary")
    a = ap.parse_args(argv)
    eps, bases = _read_episodes(a)
    sp, grp = _read_split(a)
    src = _source_recordings(a, bases)
    ex, unsplit = _expand_groups(eps, src, sp, grp)
    rows = _write_exclude(eps, ex, a)
    _write_summary(rows, eps, ex, src, unsplit, a)
    print(f"[MIDEXCL] OK recordings {len(src)} → group clips {len(ex)} · episodes to exclude {len(rows)} → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
