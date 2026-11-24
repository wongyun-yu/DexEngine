#!/usr/bin/env python3
"""E5 kinematic baseline — IK reach table (dataset × hand, per frame); reads prep.py outputs only.

Input: --run <name>=<prep.py result jsonl glob>=<shadow root> (several; the same clip set is compared side by side). Per
  clip the active hands' (jsonl `active`) IK cache precompute/<ds>/robots/<combo>/arm_ik_poses_<side>_cont25/<tree>.npz,
  wrist_pos_error_all (m) and wrist_rot_error_deg_all (°) — every frame, gathered by dataset × hand (idle hands
  excluded).
Columns: clips, frames, position error median / p90 (mm), orientation error median / p90 (°), % within 1 cm, % within
  10°, % both. Output: --out-tsv (machine) · stdout = markdown. Marker [KIN_REACH]. Count identity: jsonl ok rows =
  the table's clip sum (else death).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

from experiments.policies.kin_data import reach as KR  # owner of the thresholds
from experiments.textio import read_lines

MARK = "[KIN_REACH]"
POS_TOL_M, ROT_TOL_DEG = KR.POS_TOL_M, KR.ROT_TOL_DEG


def run_rows(spec: str):
    name, pat, root = spec.split("=", 2)
    recs = [json.loads(ln) for f in sorted(glob.glob(pat)) for ln in read_lines(f) if ln.strip()]
    ok = [r for r in recs if r.get("ok")]
    cells: dict = {}
    for r in ok:
        for s in r["active"]:
            f = Path(root) / f"precompute/{r['ds']}/robots/{r['combo']}/arm_ik_poses_{s}_cont25/{r['tree']}.npz"
            z = np.load(f)
            if "wrist_pos_error_def" not in z.files or str(z["wrist_pos_error_def"]) != "euclid_palm_origin_m":
                raise SystemExit(
                    f"{MARK} {f}: position error definition is not Euclidean (old version — solver log "
                    "residual) · rebuild it"
                )
            c = cells.setdefault((r["ds"], s), {"clips": set(), "pe": [], "re": []})
            c["clips"].add(r["clip"])
            c["pe"].append(np.asarray(z["wrist_pos_error_all"], float))
            c["re"].append(np.asarray(z["wrist_rot_error_deg_all"], float))
    n_clip = len({r["clip"] for r in ok})
    if n_clip != len({c for v in cells.values() for c in v["clips"]}):
        raise SystemExit(f"{MARK} {name}: jsonl ok clips {n_clip} ≠ table clips — count identity broken")
    out = []
    for (ds, s), c in sorted(cells.items()):
        pe, re_ = np.concatenate(c["pe"]), np.concatenate(c["re"])
        out.append(
            {
                "run": name,
                "ds": ds,
                "hand": s,
                "clips": len(c["clips"]),
                "frames": len(pe),
                "pos_mm_med": round(float(np.median(pe)) * 1e3, 1),
                "pos_mm_p90": round(float(np.percentile(pe, 90)) * 1e3, 1),
                "rot_deg_med": round(float(np.median(re_)), 1),
                "rot_deg_p90": round(float(np.percentile(re_, 90)), 1),
                "in_1cm_pct": round(100 * float(np.mean(pe <= POS_TOL_M)), 1),
                "in_10deg_pct": round(100 * float(np.mean(re_ <= ROT_TOL_DEG)), 1),
                "in_both_pct": round(100 * float(np.mean((pe <= POS_TOL_M) & (re_ <= ROT_TOL_DEG))), 1),
            }
        )
    print(
        f"{MARK} {name}: jsonl rows {len(recs)} · ok {len(ok)} · clips {n_clip} · cells {len(out)}",
        file=sys.stderr,
        flush=True,
    )
    return out, len(recs) - len(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out-tsv", required=True)
    a = ap.parse_args()
    rows, fails = [], {}
    for spec in a.run:
        r, nf = run_rows(spec)
        rows += r
        fails[spec.split("=", 1)[0]] = nf
    cols = list(rows[0])
    with open(a.out_tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(
        "| placement | dataset | hand | clips | frames | position median mm | position p90 mm | rotation "
        "median ° | rotation p90 ° | ≤1 cm % | ≤10° % | both % |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in cols) + " |")
    print(f"{MARK} DONE cells {len(rows)} · failed rows {fails} → {a.out_tsv}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
