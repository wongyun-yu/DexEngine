#!/usr/bin/env python3
"""Placement-search ablation (paper Sec. E.5): the table of rows a–e from the per-clip records (each item's
fetched run folder).

Per row and clip:  accepted  = the item's status_ledger.tsv row is ADOPTED (judge → bake → retime → gate all passed)
                   stretch   = max over sides of the gate table's stretch column (retimed length / original) on PASS
                   rows
                   wall      = p0/judged.tsv seconds (the judge's per-clip wall clock on that GPU)
Row (d) 'ours' on the main set = the production records (folded ledger status · production gate table · production
judged.tsv wall on shared GPUs); its like-for-like time column = the dtime rerun on RTX 3090 GPUs.
Row (e) also prints the extrapolation: wall per judged cell × the desk-valid grid of that clip ([PLACE_STRATEGY]
grid=). Every listed clip must have a record in its row: a missing one is printed and the row is marked incomplete.

Usage: python -m experiments.ablations.placement.aggregate --fetched <plan>/fetched --lists <list dir> --root <sim data
checkout> [--out table.md]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path

from experiments.ablations.placement.rows import E_BUDGET, E_CAP, fetched_items  # which fetched attempt counts

ROWNAME = {
    "a": "(a) fixed default cell",
    "b": "(b) 12 random cells",
    "c": "(c) map screening only",
    "d": "(d) map + IK (ours)",
    "e": "(e) IK on every cell, no map",
}
SETS = {"a": "main", "b": "main", "c": "main", "d": "main", "dtime": "dtime", "e": "ecells", "canary": "canary"}


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))] if xs else float("nan")


def rows(path, delim="\t"):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter=delim))


#: gate table columns (record keys written by the gate, Korean): clip · verdict · stretch (retimed length / original)
GATE_CLIP, GATE_VERDICT, GATE_STRETCH = "clip", "verdict", "stretch_ratio"


def gate_stretch(files):
    """clip → the largest stretch over the PASS rows of the given gate tables."""
    out = {}
    for f in files:
        for r in rows(f):
            if r.get(GATE_VERDICT) != "PASS":
                continue
            try:
                v = float(r[GATE_STRETCH])
            except (KeyError, ValueError):
                continue
            c = r[GATE_CLIP]
            out[c] = max(out.get(c, 0.0), v)
    return out


def item_records(fetched: Path):
    """row -> clip -> dict(status, stretch, wall, mark)"""
    rec = {}
    for item in fetched_items(fetched):  # each item's last DONE attempt only (ITEMS.tsv of the fetched folder)
        led = item / "status_ledger.tsv"
        row, tag = item.parent.name, item.name
        host = item.parents[2]
        gates = glob.glob(
            str(host / "sb" / row / tag / "precompute" / "*" / "robots_v5*" / "*" / "meta" / f"gate_{tag}.tsv")
        )
        st = gate_stretch(gates)
        wall = {}
        jt = item / "p0" / "judged.tsv"
        if jt.is_file():
            for ln in jt.read_text(encoding="utf-8").splitlines():
                p = ln.split("\t")
                if len(p) >= 2:
                    wall[p[0]] = float(p[1])
        # the item's first judged clip also pays the process's one-off costs (torso ruler torch.compile, graph
        # captures); with 2–4 clips per item that is a large share of the clips → a warm-only median is printed
        # beside the full one
        first = next(iter(wall), None)
        mark = {}
        mk = item / "judge_marks.txt"
        if mk.is_file():
            for ln in mk.read_text(encoding="utf-8").splitlines():
                m = re.match(
                    r"\[PLACE_STRATEGY\] (\S+) strategy=(\S+) list=(\d+) judged=(\d+) checked=(\d+) "
                    r"pruned=(\d+) adopted=(\d+) wall=([\d.]+)s grid=(.*)$",
                    ln,
                )
                if m:
                    g = re.search(r"desk-valid (\d+)", m.group(9))
                    mark[m.group(1)] = {
                        "list": int(m.group(3)),
                        "judged": int(m.group(4)),
                        "checked": int(m.group(5)),
                        "pruned": int(m.group(6)),
                        "adopted": int(m.group(7)),
                        "wall": float(m.group(8)),
                        "grid": int(g.group(1)) if g else None,
                    }
        for r in rows(led):
            c = r["clip"]
            rec.setdefault(row, {})[c] = {
                "status": r["status"],
                "stretch": st.get(c),
                "wall": wall.get(c),
                "mark": mark.get(c),
                "tag": tag,
                "first": (c == first),
            }
    return rec


def prod_records(lists: Path, root: Path):
    out = {}
    for r in rows(lists / "clips_main.tsv"):
        ds, ax, tag = r["dataset"], r["axis"], r["tag"]
        var = "_artic" if ax.endswith("-artic") else ""
        gates = glob.glob(str(root / "precompute" / ds / f"robots_v5{var}" / "*" / "meta" / f"gate_{tag}.tsv"))
        st = gate_stretch(gates)
        out[r["clip"]] = {
            "status": r["prod_status"],
            "stretch": st.get(r["clip"]),
            # 0 s = the production judge reused an earlier attempt's verdict — no time
            "wall": float(r["prod_wall_s"]) if r["prod_wall_s"] and float(r["prod_wall_s"]) > 0 else None,
            "mark": None,
            "tag": tag,
        }
    return out


#: end states that are not a verdict (ledger_states: BAKE_ERROR = code/env failure, SOURCE_DEFECT = input gap) — such a
#: clip has no result for its row yet: it counts as missing (row INCOMPLETE), never as a reject
TECHNICAL = ("BAKE_ERROR", "SOURCE_DEFECT")


def summarize(name, want, rec):
    have = [c for c in want if c in rec and rec[c]["status"] not in TECHNICAL]
    miss = [c for c in want if c not in have]
    acc = [c for c in have if rec[c]["status"] == "ADOPTED"]
    st = [rec[c]["stretch"] for c in acc if rec[c]["stretch"] is not None]
    wl = [rec[c]["wall"] for c in have if rec[c]["wall"] is not None]
    ww = [rec[c]["wall"] for c in have if rec[c]["wall"] is not None and rec[c].get("first") is False]
    ww90 = p90(ww)
    ik = [rec[c]["mark"]["checked"] for c in have if rec[c].get("mark")]
    return {
        "row": name,
        "n": len(want),
        "have": len(have),
        "missing": miss,
        "accepted": len(acc),
        "ik_med": statistics.median(ik) if ik else None,
        "ik_p90": p90(ik) if ik else None,
        "n_ik": len(ik),
        "wall_p90_warm": ww90,
        "acc_pct": 100.0 * len(acc) / len(want) if want else float("nan"),
        "st_mean": statistics.fmean(st) if st else float("nan"),
        "st_p90": p90(st),
        "n_st": len(st),
        "wall_med": statistics.median(wl) if wl else float("nan"),
        "wall_p90": p90(wl),
        "n_wall": len(wl),
        "wall_med_warm": statistics.median(ww) if ww else float("nan"),
        "n_warm": len(ww),
    }


def _num(x, nd=3):
    """JSON number or null (NaN / None → null; JSON has no NaN)."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    return round(float(x), nd)


IK_DEF = (
    "cells that entered the per-cell IK feasibility check of the exhaustion loop (judge_exhaust n_ex — the "
    "[PLACE_STRATEGY] checked= field); cells skipped by branch-and-bound are not counted"
)


def _row_common(s):
    """The fields every row has (counts, acceptance, stretch, IK checks)."""
    r = {
        "name": s["row"],
        "n": s["n"],
        "have": s["have"],
        "complete": not s["missing"],
        "accepted": s["accepted"],
        "accepted_pct": _num(s["acc_pct"], 1),
        "stretch_mean": _num(s["st_mean"]),
        "stretch_p90": _num(s["st_p90"]),
        "stretch_n": s["n_st"],
        "ik_median": _num(s["ik_med"], 1),
        "ik_p90": _num(s["ik_p90"], 1),
        "ik_n": s["n_ik"],
    }
    return r


def _add_production_fields(res, r, s):
    """Row d: production records and the 3090 re-run on the dtime set (no IK count recorded)."""
    dt = res["dtime"]
    r.update(
        ik_median=None,
        ik_p90=None,
        ik_n=0,
        ik_note=(
            "not recorded: the canonical judge (anchors, then map-sorted exhaustion) writes a check "
            "count only for clips cut by a budget ([JUDGE_BUDGET]) and at each adoption (plans.jsonl "
            "checks), not for every clip — no per-clip total exists to take a median of"
        ),
        accept_source="production records (clips_main.tsv prod_status · production gate tables)",
        gpu_production={
            "gpu": "production shared GPUs (clips_main.tsv prod_wall_s)",
            "first_clip_excluded": False,
            "median": _num(s["wall_med"], 1),
            "p90": _num(s["wall_p90"], 1),
            "n": s["n_wall"],
        },
        gpu_3090_rerun={
            "gpu": "RTX 3090 (two workstations), row dtime = canon judge on the dtime set",
            "clips": dt["n"],
            "have": dt["have"],
            "first_clip_excluded": {
                "median": _num(dt["wall_med_warm"], 1),
                "p90": _num(dt["wall_p90_warm"], 1),
                "n": dt["n_warm"],
            },
            "all_clips": {"median": _num(dt["wall_med"], 1), "p90": _num(dt["wall_p90"], 1), "n": dt["n_wall"]},
        },
    )


def _add_single_clip_fields(want, rec, r, s):
    """Row e: one clip per item (first-clip cost kept) and the full-grid extrapolation."""
    ex = []
    for c in want["ecells"]:
        m = (rec.get("e", {}).get(c) or {}).get("mark")
        if m and m["judged"] and m["grid"]:
            ex.append(m["wall"] / max(1, m["checked"] + m["pruned"]) * m["grid"])
    r.update(
        gpu="RTX 3090 (two workstations), this ablation's runs",
        gpu_first_clip_excluded=False,
        gpu_note="each (e) item holds one clip, so every clip is its item's first; the one-off boot cost stays in",
        gpu_s_median=_num(s["wall_med"], 1),
        gpu_s_p90=_num(s["wall_p90"], 1),
        gpu_n=s["n_wall"],
        extrapolated_full_grid_s_median=_num(statistics.median(ex), 0) if ex else None,
        extrapolated_full_grid_n=len(ex),
        extrapolated_def="wall / (checked + pruned cells) × desk-valid grid cells, per clip; median "
        f"(cap {E_CAP} cells, budget {E_BUDGET} s per clip)",
    )


def paper_json(res, rec, want, fetched: Path) -> dict:
    """The paper table as data: one record per row a–e, every number read from the records above (no typed value)."""
    items = {
        h.name: sum(1 for ln in (h / "ITEMS.tsv").read_text(encoding="utf-8").splitlines() if ln)
        for h in sorted(fetched.iterdir())
        if (h / "ITEMS.tsv").is_file()
    }
    out = {
        "generator": "experiments/ablations/placement/aggregate.py --json",
        "fetched_done_items": items,
        "stretch_def": "retimed length / original, max over sides, gate PASS rows of the accepted clips",
        "ik_def": IK_DEF,
        "rows": {},
    }
    for k in ("a", "b", "c", "d", "e"):
        s = res[k]
        r = _row_common(s)
        if k in ("a", "b", "c"):
            r.update(
                gpu="RTX 3090 (two workstations), this ablation's runs",
                gpu_first_clip_excluded=True,
                gpu_s_median=_num(s["wall_med_warm"], 1),
                gpu_s_p90=_num(s["wall_p90_warm"], 1),
                gpu_n=s["n_warm"],
                gpu_s_median_all=_num(s["wall_med"], 1),
                gpu_s_p90_all=_num(s["wall_p90"], 1),
                gpu_n_all=s["n_wall"],
            )
        elif k == "d":
            _add_production_fields(res, r, s)
        else:  # e: one clip per item — no warm clip exists, so the first-clip cost stays in
            _add_single_clip_fields(want, rec, r, s)
        out["rows"][k] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetched", required=True)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--json", default="", help="write the paper table as JSON (table.json — what the paper generator reads)"
    )
    a = ap.parse_args()
    lists = Path(a.lists)
    want = {s: [r["clip"] for r in rows(lists / f"clips_{s}.tsv")] for s in ("main", "dtime", "ecells", "canary")}
    rec = item_records(Path(a.fetched))
    prod = prod_records(lists, Path(a.root))
    lines = [
        "| Strategy | clips | accepted % | stretch mean | stretch p90 | GPU wall s/clip median | p90 | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    res = {}
    for k in ("a", "b", "c"):
        res[k] = summarize(ROWNAME[k], want["main"], rec.get(k, {}))
    res["d"] = summarize(ROWNAME["d"], want["main"], prod)
    res["dtime"] = summarize("(d) timing rerun (3090)", want["dtime"], rec.get("dtime", {}))
    res["e"] = summarize(ROWNAME["e"], want["ecells"], rec.get("e", {}))
    for k in ("a", "b", "c", "d", "e"):
        s = res[k]
        note = (
            ""
            if not s["missing"]
            else f"INCOMPLETE: {len(s['missing'])} clips without a verdict (no record, or a technical end state)"
        )
        if k == "d":
            dt = res["dtime"]
            note = (
                f"accept/stretch = production; wall = production shared GPUs · 3090 rerun on {dt['have']}/{dt['n']}: "
                f"median {dt['wall_med']:.1f} s · p90 {dt['wall_p90']:.1f} s"
            ) + (f" · {note}" if note else "")
        if k == "e":
            ex = []
            for c in want["ecells"]:
                m = (rec.get("e", {}).get(c) or {}).get("mark")
                if m and m["judged"] and m["grid"]:
                    ex.append(m["wall"] / max(1, m["checked"] + m["pruned"]) * m["grid"])
            if ex:
                note += (
                    f" · extrapolated full-grid wall per clip: median {statistics.median(ex) / 3600:.1f} h "
                    f"(n={len(ex)})"
                )
        note += f" · stretch n={s['n_st']} · wall n={s['n_wall']}"
        if s["n_warm"]:
            note += f" · wall median without each item's first clip {s['wall_med_warm']:.1f} s (n={s['n_warm']})"
        lines.append(
            f"| {s['row']} | {s['have']}/{s['n']} | {s['acc_pct']:.1f} ({s['accepted']}) | {s['st_mean']:.2f} | "
            f"{s['st_p90']:.2f} | {s['wall_med']:.1f} | {s['wall_p90']:.1f} | {note} |"
        )
    txt = "\n".join(lines)
    if a.json:
        Path(a.json).write_text(
            json.dumps(paper_json(res, rec, want, Path(a.fetched)), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(txt)
    if a.out:
        Path(a.out).write_text(
            txt + "\n\n```json\n" + json.dumps(res, indent=1, default=str) + "\n```\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
