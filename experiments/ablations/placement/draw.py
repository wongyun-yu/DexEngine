#!/usr/bin/env python3
"""Placement-search ablation (paper Sec. E.5): draw the clip set BEFORE the kinematic filter.

Universe = the folded status ledgers of one hand (dexengine.data.rules.ledger_files — the per-clip records the
funnel's "after kinematic-feasibility filter" row is counted from). Every judged row of those ledgers is a
scene-consistent clip of that hand (the funnel's 12,940 is the mean over the chain hands; Wuji 2's own count is
printed).

Stratified by dataset in the funnel's proportions (largest remainder), uniform inside a dataset, seed 42.
Sub-sets (all drawn from the main set, seed 42):
  canary  = production ADOPTED clips of the main set (topped up from the same ledgers when < N_CANARY)
  dtime   = (d) timing rerun, stratified by production status (ADOPTED / not) in the main set's proportion
  ecells  = (e) all-cells clips, half ADOPTED, half not
For every clip it records the production verdict, chunk tag, production judge wall seconds (p*/judged.tsv) and
the production placement the bake used (plans.jsonl line of the bake_ledger anchor).

Usage: python -m experiments.ablations.placement.draw --hand wuji2 --root <main checkout> --n 128 --out <dir>
Writes <out>/clips_main.tsv · clips_canary.tsv · clips_dtime.tsv · clips_ecells.tsv · draw_summary.json.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.placement import final_ledger as FL

SEED = 42
CARRY = "hand_carry("  # note prefix of a carried ledger row: carried rows are not judged rows
#: funnel row "After scene-consistency filter" — the stratification weights.
FUNNEL_SCENE = {
    "hot3d": 3790,
    "oakink2": 3425,
    "taco": 2325,
    "grab": 1403,
    "arctic": 729,
    "hrdexdb": 433,
    "h2o": 402,
    "hocap": 318,
    "dexcanvas": 115,
}
COLS = (
    "set",
    "dataset",
    "axis",
    "emb",
    "clip",
    "prod_status",
    "tag",
    "prod_wall_s",
    "prod_anchor",
    "prod_dx",
    "prod_dy",
    "prod_yaw",
    "prod_zoff",
    "rules",
)


def die(msg):
    raise SystemExit(f"[PLACE_ABL_DRAW] ✗ {msg}")


def alloc(n: int, w: dict) -> dict:
    tot = sum(w.values())
    raw = {k: n * v / tot for k, v in w.items()}
    out = {k: int(raw[k]) for k in w}
    rest = n - sum(out.values())
    for k in sorted(w, key=lambda k: (-(raw[k] - out[k]), k))[:rest]:
        out[k] += 1
    return out


def prod_record(root: Path, hand: str, tag: str, clip: str) -> dict:
    ch = root / "overfit/hands" / hand / "chunks" / tag
    if not ch.is_dir():
        die(f"chunk folder missing: {ch}")
    wall = None
    for f in sorted(glob.glob(str(ch / "**/judged.tsv"), recursive=True)):
        for ln in Path(f).read_text(encoding="utf-8").splitlines():
            p = ln.split("\t")
            if len(p) >= 2 and P.clip_name(p[0]) == clip:
                wall = float(p[1])
    anchor = None
    bl = ch / "bake_ledger.tsv"
    if bl.is_file():
        for ln in bl.read_text(encoding="utf-8").splitlines():
            p = ln.split("\t")
            if len(p) >= 3 and P.clip_name(p[0]) == clip and p[1] == "OK":
                anchor = int(p[2])
    plc = None
    if anchor is not None:
        for f in [ch / "plans.jsonl"] + [Path(x) for x in sorted(glob.glob(str(ch / "t*/plans.jsonl")))]:
            if not f.is_file():
                continue
            for ln in f.read_text(encoding="utf-8").splitlines():
                d = json.loads(ln)
                if P.clip_name(d["clip"]) == clip and int(d["anchor"]) == anchor:
                    plc = d
            if plc:
                break
    return {"wall": wall, "anchor": anchor, "plc": plc}


def _parse_args():
    """Parse the command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--root", required=True, help="data checkout that holds overfit/hands/<hand>/{ledger,chunks}")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-canary", type=int, default=64)
    ap.add_argument("--n-dtime", type=int, default=16)
    ap.add_argument("--n-ecells", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return a


def _read_ledgers(a, root):
    """Folded ledger rows of the hand (carry-over rows skipped), keyed by (dataset, clip), and the clips per dataset."""
    files = RR.ledger_files("*", a.hand, root=str(root))
    if not files:
        die(f"no folded ledger for {a.hand} under {root}")
    rows = {}  # clip -> row (dataset, axis, emb, status, tag, rules)
    by_ds = collections.defaultdict(list)
    for f in files:
        ax = RR.ledger_axis(f)
        ds, emb, _var = RR.ledger_axis_parts(ax)
        for r in FL.read_final_rows(root / f if not Path(f).is_absolute() else f):
            if str(r.get("note") or "").startswith(CARRY):
                continue
            c = P.clip_name(r["clip"])
            if (ds, c) in rows:
                die(f"clip {c} appears twice in {ds} ledgers")
            ji = r.get("judge_input") or ""
            rules = next((t for t in ji.split("|") if t.startswith("rules=")), "")
            rows[(ds, c)] = {
                "dataset": ds,
                "axis": ax,
                "emb": emb,
                "clip": c,
                "prod_status": r["status"],
                "tag": r.get("source_tag") or "",
                "rules": rules,
            }
            by_ds[ds].append(c)
    if set(by_ds) != set(FUNNEL_SCENE):
        die(f"ledger datasets {sorted(by_ds)} != funnel datasets {sorted(FUNNEL_SCENE)}")
    return files, rows, by_ds


def _draw_sets(a, by_ds, rows):
    """Draw the main set (per-dataset allocation) and its canary, dtime and ecells subsets from one seeded stream."""
    rng = random.Random(SEED)
    want = alloc(a.n, FUNNEL_SCENE)
    main_set = []
    for ds in sorted(by_ds):
        pool = sorted(by_ds[ds])
        main_set += [(ds, c) for c in rng.sample(pool, want[ds])]
    # canary: ADOPTED clips of the main set, topped up (same seed stream) from the rest of the ledgers
    adopted_main = [k for k in main_set if rows[k]["prod_status"] == "ADOPTED"]
    canary = list(adopted_main[: a.n_canary])
    if len(canary) < a.n_canary:
        rest = sorted(k for k in rows if rows[k]["prod_status"] == "ADOPTED" and k not in set(main_set))
        canary += rng.sample(rest, a.n_canary - len(canary))
    # dtime: stratified by production status in the main set's proportion
    acc = [k for k in main_set if rows[k]["prod_status"] == "ADOPTED"]
    rej = [k for k in main_set if rows[k]["prod_status"] != "ADOPTED"]
    n_acc = round(a.n_dtime * len(acc) / len(main_set))
    dtime = rng.sample(acc, n_acc) + rng.sample(rej, a.n_dtime - n_acc)
    ecells = rng.sample(acc, a.n_ecells // 2) + rng.sample(rej, a.n_ecells - a.n_ecells // 2)
    return want, main_set, adopted_main, canary, n_acc, dtime, ecells


def _summary(rows, main_set, a, universe, by_ds, want, adopted_main, canary, n_acc, miss, files, root):
    """The draw summary record."""
    st = collections.Counter(rows[k]["prod_status"] for k in main_set)
    summ = {
        "hand": a.hand,
        "seed": SEED,
        "universe_rows": universe,
        "universe_per_dataset": {d: len(v) for d, v in sorted(by_ds.items())},
        "funnel_scene_weights": FUNNEL_SCENE,
        "n": a.n,
        "per_dataset": want,
        "main_prod_status": dict(st),
        "main_prod_adopted_pct": round(100 * st["ADOPTED"] / len(main_set), 1),
        "canary_from_main": len(adopted_main[: a.n_canary]),
        "canary_topped_up": len(canary) - len(adopted_main[: a.n_canary]),
        "dtime_adopted": n_acc,
        "ecells_adopted": a.n_ecells // 2,
        "adopted_without_prod_placement": miss,
        "ledger_files": [str(f) for f in files],
        "ledger_sha256": {
            str(f): hashlib.sha256(Path(root / f if not Path(f).is_absolute() else f).read_bytes()).hexdigest()
            for f in files
        },
    }
    return st, summ


def main() -> int:
    a = _parse_args()
    root = Path(a.root).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files, rows, by_ds = _read_ledgers(a, root)
    universe = sum(len(v) for v in by_ds.values())
    want, main_set, adopted_main, canary, n_acc, dtime, ecells = _draw_sets(a, by_ds, rows)

    def write(name, keys, setname):
        lines = ["\t".join(COLS)]
        miss = 0
        for k in keys:
            r = rows[k]
            pr = prod_record(root, a.hand, r["tag"], r["clip"])
            plc = pr["plc"] or {}
            if r["prod_status"] == "ADOPTED" and not plc:
                miss += 1
            vals = [
                setname,
                r["dataset"],
                r["axis"],
                r["emb"],
                r["clip"],
                r["prod_status"],
                r["tag"],
                "" if pr["wall"] is None else f"{pr['wall']:.1f}",
                "" if pr["anchor"] is None else str(pr["anchor"]),
                str(plc.get("dx", "")),
                str(plc.get("dy", "")),
                str(plc.get("yaw", "")),
                str(plc.get("robot_z_off", "")),
                r["rules"],
            ]
            lines.append("\t".join(vals))
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return miss

    miss = {
        n: write(f"clips_{n}.tsv", ks, n)
        for n, ks in (("main", main_set), ("canary", canary), ("dtime", dtime), ("ecells", ecells))
    }
    st, summ = _summary(rows, main_set, a, universe, by_ds, want, adopted_main, canary, n_acc, miss, files, root)
    (out / "draw_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[PLACE_ABL_DRAW] hand={a.hand} seed={SEED} universe={universe} n={a.n} per_dataset={want} "
        f"prod_status={dict(st)} canary={len(canary)} (topped up {summ['canary_topped_up']}) dtime={len(dtime)} "
        f"ecells={len(ecells)} adopted_without_prod_placement={miss}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
