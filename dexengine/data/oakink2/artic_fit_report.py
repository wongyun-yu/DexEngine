"""Measurement report of the OakInk2 part pairing over this hand's base trees (one row per tree: fit kind,
clusters, residuals). It is the CLI of dexengine/data/oakink2/artic_fit.py:
    python3 dexengine/data/oakink2/artic_fit.py --emb <emb> [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.data.oakink2 import parts as OP
from dexengine.data.oakink2.artic_fit import PART_MERGE_FAIL, POS_TOL_M, ROT_TOL_DEG, PartMergeError, part_clusters


def main() -> int:
    a = _parse_args()
    root = Path(P.layer_path("base", "oakink2", hand=RR.hand_dir(), emb=a.emb))
    rows, cnt, n_tree = _fit_rows(root, a)
    out = P.clips_record(
        RR.audit_out(f"oakink2_artic_fit_{datetime.date.today():%Y%m%d}", RR.active_hand()), bool(a.limit)
    )
    out.mkdir(exist_ok=True)  # a --limit run does not overwrite that day's full fit folder (paths.clips_record)
    _write_tsv(out, rows)
    md = _summary_md(rows, cnt, n_tree)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(f"→ {out}", file=sys.stderr)
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--emb", required=True, help="embodiment (no default)")
    a = ap.parse_args()
    return a


def _fit_rows(root, a) -> tuple[list, collections.Counter, int]:
    """One row per base tree with a multi-part instance (fit kind, clusters, residuals), the count per kind and the
    number of trees read."""
    rows = []
    cnt = collections.Counter()
    n_tree = 0
    for f in sorted(root.glob("*/task_info.json")):
        n_tree += 1
        if a.limit and len(rows) >= a.limit:
            break
        ti = json.loads(f.read_text())
        q = np.load(f.parent / "0/trajectory_kinematic.npz")["qpos"]
        try:
            pc = part_clusters(ti.get("objects") or [], q)
        except PartMergeError as e:
            pc = {"status": f"fail:{e.code}"}
        if pc["status"] == "no_parts":
            continue
        cl = pc.get("clusters") or []
        r = pc.get("fit") or {"kind": {"one_cluster": "rigid"}.get(pc["status"], pc["status"])}
        cnt[r["kind"]] += 1
        rid0 = OP.rid_of(cl[-1]["anchor"]["obj_id"]) if cl else ""
        names = " / ".join("+".join(OP.name_of(OP.rid_of(o["obj_id"])) for o in [c["anchor"], *c["fixed"]]) for c in cl)
        rows.append(
            [
                f.parent.name,
                r["kind"],
                OP.name_of(rid0) if rid0 else "",
                OP.name_of(pc["instance"]) if cl else "",
                names,
                round(r.get("rot_resid_deg", -1), 2),
                round(r.get("pos_resid_m", -1), 4),
                round(r.get("angle_range_deg", 0), 1),
                round(r.get("travel_m", 0), 4),
                round(r.get("pitch_m_per_rad", 0), 4),
                len(q),
            ]
        )
    return rows, cnt, n_tree


def _write_tsv(out, rows) -> None:
    """fit.tsv: one row per tree."""
    with open(out / "fit.tsv", "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "clip",
                "kind",
                "part_name",
                "instance",
                "clusters",
                "rot_resid_deg",
                "pos_resid_m",
                "angle_range_deg",
                "travel_m",
                "pitch_m_per_rad",
                "n_frames",
            ]
        )
        w.writerows(rows)


def _summary_md(rows, cnt, n_tree) -> list[str]:
    """summary.md lines: kinds with their meaning and the top instances."""
    by_inst = collections.Counter((r[3], r[1]) for r in rows)
    md = [
        f"# OakInk2 part clusters · joint fit (D1-g, {datetime.datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Of {n_tree} base trees, **{len(rows)} have an instance with two or more parts** (official part tree "
        "· clusters = "
        f"part_clusters). Thresholds = rotation {ROT_TOL_DEG}° · translation {POS_TOL_M * 100:.0f} cm.",
        "",
        "| kind | trees | meaning |",
        "|---|---|---|",
    ]
    why = {
        "revolute": "one-axis rotation (hinge): articulation candidate",
        "screw": "rotation + axial travel: articulation candidate",
        "prismatic": "linear travel: articulation candidate",
        "rigid": "all parts one body (no joint needed)",
        "free": "not one joint (assembly, picking): stays two rigid bodies as in D1-b",
        "multi": "two or more instances",
        **{f"fail:{c}": "the merge rule cannot decide; the converter fails" for c in PART_MERGE_FAIL},
    }
    for k, v in cnt.most_common():
        md.append(f"| {k} | {v} | {why.get(k, 'error')} |")
    md += ["", "## By instance (top 15)", "", "| object | kind | trees |", "|---|---|---|"]
    for (inst, kind), v in by_inst.most_common(15):
        md.append(f"| {inst} | {kind} | {v} |")
    return md
