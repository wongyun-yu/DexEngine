#!/usr/bin/env python3
"""Placement-search ablation (paper Sec. E.5): the extra rows (rnd · rndt · xall · xfull), one JSON + one per-clip
TSV per row.

`aggregate.py` owns the paper table of rows a–e (table.json — its shape is read by the paper generator and is not
touched here). This file reads the same fetched items through the same functions (`aggregate.item_records` ·
`aggregate.summarize` — one rule for which attempt counts, what "accepted" means, where stretch and wall come from) and
adds what the new rows need:

  * the time-to-accept record: `[PLACE_STRATEGY_HIT] <clip> … k=<n> t=<s> checked=…` lines of judge_marks.txt, so the
    kept share can be read at any search time (e.g. at ours' per-clip time);
  * the split by the production verdict of the same clip (ADOPTED / UNPROVEN / EXHAUSTED / GUARD_FAIL) — for the rows
    that run on the clips ours dropped, "how many does a wider search rescue";
  * how the search ended per clip (accepted enough · list end · cell cap · time).

  python -m experiments.ablations.placement.aggregate_rows --fetched <plan>/fetched --row rnd --lists <lists dir>
  --commit <sha> \
      --out-json <f.json> --out-tsv <f.tsv> [--only <clip names file>] [--at 20.1,60,300,1200]

Definitions written into the JSON (`defs`):
  kept            final ledger status ADOPTED (after bake → retiming → final check), the same rule as rows a–e
  kept_by_search_t[T]  kept clips whose FIRST accepted cell came at search time ≤ T s (HIT t = seconds since the
                  strategy's search began — without process boot and clip preparation)
  kept_by_wall_t[T]    the same with the clip's preparation added: t + (judge wall − search wall) ≤ T — the scale of
                  the `GPU wall s/clip` column of rows a–e (p0/judged.tsv)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from experiments.ablations.placement.aggregate import item_records, p90, rows, summarize
from experiments.ablations.placement.rows import fetched_items

HIT_RE = re.compile(r"\[PLACE_STRATEGY_HIT\] (\S+) strategy=(\S+) k=(\d+) t=([\d.]+)s checked=(\d+) pruned=(\d+)$")
HEAD_RE = re.compile(r"\[PLACE_STRATEGY\] strategy=(\S+) cap=(\d+) budget_s=([\d.]+) ")


def hits(fetched: Path, row: str):
    """clip -> [(k, t, checked)] and the set of (strategy, cap, budget) header lines seen in the row's items."""
    out, heads = {}, set()
    for item in fetched_items(fetched):
        if item.parent.name != row:
            continue
        mk = item / "judge_marks.txt"
        if not mk.is_file():
            continue
        for ln in mk.read_text(encoding="utf-8").splitlines():
            m = HIT_RE.match(ln)
            if m:
                out.setdefault(m.group(1), []).append((int(m.group(3)), float(m.group(4)), int(m.group(5))))
                continue
            h = HEAD_RE.match(ln)
            if h:
                heads.add((h.group(1), int(h.group(2)), float(h.group(3))))
    return out, heads


def ended(mark, n_stop: int) -> str:
    if mark is None:
        return "no-marker"
    if mark["adopted"] >= n_stop:
        return "accepted-enough"
    if mark["judged"] >= mark["list"]:
        return "list-end" if mark["checked"] + mark["pruned"] >= mark["judged"] else "time"
    return "cell-cap" if mark["checked"] + mark["pruned"] >= mark["judged"] else "time"


def _parse_args():
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetched", required=True)
    ap.add_argument("--row", required=True)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--only", default="", help="file of clip names — the row's clip set (default: the whole main list)")
    ap.add_argument("--at", default="5,10,20.1,60,120,300,600,1200,3600,7200")
    ap.add_argument("--n-stop", type=int, default=3, help="accepted cells that end a search (judge --exhaust-cands)")
    a = ap.parse_args()
    return a


def _clip_set(a):
    """Production status per clip and the row clip set (the main list, or --only within it)."""
    main_rows = rows(Path(a.lists) / "clips_main.tsv")
    prod = {r["clip"]: r["prod_status"] for r in main_rows}
    want = [r["clip"] for r in main_rows]
    if a.only:
        only = [ln for ln in Path(a.only).read_text(encoding="utf-8").splitlines() if ln]
        if set(only) - set(want):
            raise SystemExit(
                f"[AGG_ROWS] --only names clips outside clips_main.tsv: {sorted(set(only) - set(want))[:3]}"
            )
        want = [c for c in want if c in set(only)]
    return prod, want


def _per_clip(want, rec, hit, a, prod):
    """Per-clip records and the counts by production status and by search end."""
    per, by_prod, by_end = [], {}, {}
    for c in want:
        r = rec.get(c)
        m = r["mark"] if r else None
        h = sorted(hit.get(c, []))
        kept = bool(r) and r["status"] == "ADOPTED"
        prep = (r["wall"] - m["wall"]) if (r and m and r["wall"] is not None) else None
        e = ended(m, a.n_stop) if r else "no-record"
        per.append(
            {
                "clip": c,
                "prod_status": prod[c],
                "status": r["status"] if r else None,
                "kept": kept,
                "ended": e,
                "list": m["list"] if m else None,
                "judged": m["judged"] if m else None,
                "checked": m["checked"] if m else None,
                "pruned": m["pruned"] if m else None,
                "search_accepted": m["adopted"] if m else None,
                "search_wall_s": m["wall"] if m else None,
                "judge_wall_s": r["wall"] if r else None,
                "prep_s": prep,
                "first_hit_t": h[0][1] if h else None,
                "first_hit_checked": h[0][2] if h else None,
                "hit_ts": [x[1] for x in h],
                "stretch": r["stretch"] if r else None,
                "tag": r["tag"] if r else None,
            }
        )
        b = by_prod.setdefault(prod[c], {"n": 0, "kept": 0, "search_accepted": 0})
        b["n"] += 1
        b["kept"] += kept
        b["search_accepted"] += bool(m and m["adopted"] > 0)
        by_end[e] = by_end.get(e, 0) + 1
    return by_end, by_prod, per


def _row_summary(per, a, heads, want, s, by_end, by_prod, ts):
    """The row record (summary, GPU walls, kept-by-time curves, definitions, per-clip records)."""
    sw = [p["search_wall_s"] for p in per if p["search_wall_s"] is not None]
    jw = [p["judge_wall_s"] for p in per if p["judge_wall_s"] is not None]
    out = {
        "row": a.row,
        "commit": a.commit,
        "strategy_headers": sorted(heads),
        "clip_set": a.only or "clips_main.tsv (128)",
        "n": len(want),
        "have": s["have"],
        "missing": s["missing"],
        "kept": s["accepted"],
        "kept_pct": s["acc_pct"],
        "search_accepted": sum(bool(p["search_accepted"]) for p in per),
        "ik_checks_median": s["ik_med"],
        "ik_checks_p90": s["ik_p90"],
        "ratio_mean": None if s["n_st"] == 0 else s["st_mean"],
        "ratio_p90": s["st_p90"],
        "ratio_n": s["n_st"],
        "gpu_s_per_clip": {
            "judge_wall_median": s["wall_med"],
            "judge_wall_p90": s["wall_p90"],
            "judge_wall_mean": statistics.fmean(jw) if jw else None,
            "judge_wall_sum": sum(jw),
            "judge_wall_median_without_item_first": s["wall_med_warm"],
            "n_without_item_first": s["n_warm"],
            "search_wall_median": statistics.median(sw) if sw else None,
            "search_wall_p90": p90(sw),
            "search_wall_mean": statistics.fmean(sw) if sw else None,
            "search_wall_sum": sum(sw),
        },
        "ended": by_end,
        "by_production_status": by_prod,
        "kept_by_search_t": {str(t): sum(p["kept"] and p["first_hit_t"] <= t for p in per) for t in ts},
        "kept_by_wall_t": {
            str(t): sum(p["kept"] and p["prep_s"] is not None and p["first_hit_t"] + p["prep_s"] <= t for p in per)
            for t in ts
        },
        "defs": {
            "kept": "final ledger status ADOPTED (bake → retiming → final check passed) — the rule of rows a–e",
            "search_accepted": "clips with ≥ 1 cell accepted by the search (before bake → retiming → final check)",
            "ik_checks": "cells that entered the per-cell IK feasibility check (judge_exhaust n_ex) per clip",
            "ratio": "retiming stretch of the kept clips (gate tsv) — the column of rows a–e",
            "judge_wall": "per-clip judge wall from p0/judged.tsv on one RTX 3090 — the GPU wall column of rows a–e",
            "search_wall": "wall of the strategy's search alone ([PLACE_STRATEGY] marker)",
            "kept_by_search_t": "kept clips whose first accepted cell came at search time ≤ T s",
            "kept_by_wall_t": "the same with the clip's preparation added (judge wall − search wall)",
            "ended": f"accepted-enough = {a.n_stop} cells accepted · list-end = every cell of the list judged · "
            "cell-cap = the cell budget ended the search · time = the time budget ended it",
        },
        "clips": per,
    }
    return out


def _write_tsv(a, per):
    """Write the per-clip TSV."""
    cols = [
        "clip",
        "prod_status",
        "status",
        "kept",
        "ended",
        "list",
        "judged",
        "checked",
        "pruned",
        "search_accepted",
        "search_wall_s",
        "judge_wall_s",
        "first_hit_t",
        "first_hit_checked",
        "stretch",
        "tag",
    ]
    with open(a.out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for p in per:
            f.write("\t".join("" if p[k] is None else str(p[k]) for k in cols) + "\n")


def main() -> int:
    a = _parse_args()
    prod, want = _clip_set(a)
    rec = item_records(Path(a.fetched)).get(a.row, {})
    if set(rec) - set(want):
        raise SystemExit(f"[AGG_ROWS] row {a.row} has clips outside its set: {sorted(set(rec) - set(want))[:3]}")
    s = summarize(a.row, want, rec)
    hit, heads = hits(Path(a.fetched), a.row)
    ts = [float(x) for x in a.at.split(",")]
    by_end, by_prod, per = _per_clip(want, rec, hit, a, prod)
    nohit = [p["clip"] for p in per if p["kept"] and p["first_hit_t"] is None]
    if nohit:
        raise SystemExit(f"[AGG_ROWS] kept clip without a HIT line: {nohit[:3]}")
    out = _row_summary(per, a, heads, want, s, by_end, by_prod, ts)
    Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    _write_tsv(a, per)
    print(
        f"[AGG_ROWS] row={a.row} have={s['have']}/{len(want)} kept={s['accepted']} ({s['acc_pct']:.1f} %) "
        f"search_accepted={out['search_accepted']} ik_med={s['ik_med']} ratio_mean={out['ratio_mean']} "
        f"judge_wall_med={s['wall_med']:.1f}s ended={by_end} "
        f"by_prod={ {k: (v['kept'], v['n']) for k, v in by_prod.items()} }",
        flush=True,
    )
    return 0 if not s["missing"] else 3


if __name__ == "__main__":
    sys.exit(main())
