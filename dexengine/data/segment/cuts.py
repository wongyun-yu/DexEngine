#!/usr/bin/env python3
"""Cut point proposals for long clips (> TARGET_S), from data geometry.

Priority:
  1) placed intervals (support_audit rest_intervals: object at rest + hands away), natural episode boundaries; cut at
     the interval centre.
  2) pieces still longer than TARGET_S are cut at the frame of lowest bimanual wrist speed (0.4 s smoothing window), so
     the robot starts and ends near rest.
Constraint: a rest cut keeps MIN_S from both clip ends and a speed cut MIN_S on both sides; two close rest intervals
can still leave a shorter piece. The result is a proposal (cut frames); base_adapter splits into `<clip>_segNN`.
Output: <out>/segment_audit.{json,md} · with --write-intake data/<ds>/INTAKE/segment_audit.json
The cut_source value is copied into trees and stays as written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dexengine.core import paths as P  # noqa

dl = P.data_layout
TARGET_S, MIN_S, DT, SMOOTH = 15.0, 5.0, 0.02, 20  # target 5–15 s (a little longer is fine)


def wrist_speed(kp):
    v = []
    for side in ("right", "left"):
        w = kp.get(f"qpos_wrist_{side}")
        if w is None or np.abs(w[:, :3]).sum() == 0:
            continue
        v.append(np.linalg.norm(np.gradient(w[:, :3], DT, axis=0), axis=1))
    if not v:
        return None
    s = np.max(np.stack(v), axis=0)  # the faster of the two hands
    return np.convolve(s, np.ones(SMOOTH) / SMOOTH, mode="same")


def propose_cuts(T, rests, speed):
    """Cut frames (ascending). rests = [(start,end)] placed intervals."""
    target = int(TARGET_S / DT)
    mn = int(MIN_S / DT)
    cuts = []
    # 1) centres of placed intervals, excluding those near the clip ends
    for a, b in sorted(rests):
        m = (a + b) // 2
        if m >= mn and T - m >= mn:
            cuts.append(m)
    cuts = sorted(set(cuts))

    # 2) remaining long pieces at the speed minimum
    def split_long(a, b):
        out = []
        if b - a <= target or speed is None:
            return out
        # lowest-speed frame within [a+mn, b-mn] of the piece
        lo, hi = a + mn, b - mn
        if hi <= lo:
            return out
        k = int(lo + np.argmin(speed[lo:hi]))
        return split_long(a, k) + [k] + split_long(k, b)

    bounds = [0] + cuts + [T]
    extra = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        extra += split_long(a, b)
    cuts = sorted(set(cuts + extra))
    segs = [(a, b) for a, b in zip([0] + cuts, cuts + [T], strict=True)]
    return cuts, segs


def _clip_record(c: Path, emb: str, sup: dict) -> dict:
    """One clip's proposal: a short clip passes whole; a long one gets cuts, piece lengths, idle pieces, cut source."""
    kp = np.load(c / "0/trajectory_keypoints.npz")
    T = int(kp["timestamps"].shape[0]) if "timestamps" in kp.files else int(kp["qpos_wrist_right"].shape[0])
    if T * DT <= TARGET_S:
        return {"clip": c.name, "emb": emb, "T": T, "long": False, "cuts": [], "n_seg": 1}
    rests = [(r["start"], r["end"]) for r in (sup.get(c.name, {}).get("rest_intervals") or [])]
    sp = wrist_speed(kp)
    cuts, segs = propose_cuts(T, rests, sp)
    # a piece where the hands barely move (max wrist speed < 5 cm/s) is an 'idle' piece without
    # manipulation; candidate for the adapter to drop
    idle = [bool(sp is not None and float(sp[a:b].max()) < 0.05) for a, b in segs]
    return {
        "clip": c.name,
        "emb": emb,
        "T": T,
        "long": True,
        "cuts": cuts,
        "n_seg": len(segs),
        "seg_s": [round((b - a) * DT, 1) for a, b in segs],
        "idle": idle,
        "n_idle": int(sum(idle)),
        "cut_source": ("rest" if rests else "speed_min") if cuts else "none(min length constraint)",
        "still_long": int(sum(1 for a, b in segs if (b - a) * DT > TARGET_S)),
    }


def _dataset_records(ds: str) -> list:
    """Proposals for every MANO clip of a dataset (all embodiments), with the support audit's rest intervals."""
    try:
        sup = json.loads((P.layer_path("intake", ds) / "support_audit.json").read_text())["clips"]
    except (OSError, KeyError, ValueError):  # no support audit yet (missing, partial or unreadable): no rest cuts
        sup = {}
    recs = []
    for emb in dl.EMBODIMENTS:
        try:
            root = P.layer_path("mano", ds, emb=emb)
        except (KeyError, ValueError):
            continue
        if not root.is_dir():
            continue
        recs.extend(
            _clip_record(c, emb, sup)
            for c in sorted(p for p in root.iterdir() if (p / "0/trajectory_keypoints.npz").exists())
        )
    return recs


def _summary(recs: list) -> dict:
    """Per-dataset counts over the proposals (the clips list is kept in the row)."""
    L = [r for r in recs if r["long"]]
    return {
        "n": len(recs),
        "long": len(L),
        "segments_from_long": sum(r["n_seg"] for r in L),
        "still_long_segments": sum(r.get("still_long", 0) for r in L),
        "cut_by_rest": sum(1 for r in L if r.get("cut_source") == "rest"),
        "cut_by_speed": sum(1 for r in L if r.get("cut_source") == "speed_min"),
        "total_clips_after": len(recs) - len(L) + sum(r["n_seg"] for r in L),
        "idle_segments": sum(r.get("n_idle", 0) for r in L),
        "clips": recs,
    }


def _write_intake(ds: str, r: dict, recs: list) -> None:
    """data/<ds>/INTAKE/segment_audit.json (the base adapter reads the cuts from it)."""
    idir = P.layer_path("intake", ds)
    idir.mkdir(parents=True, exist_ok=True)
    with open(idir / "segment_audit.json", "w") as fh:
        json.dump(
            {
                "tool": "dexengine/data/segment/cuts.py",
                "TARGET_S": TARGET_S,
                "MIN_S": MIN_S,
                "summary": {k: v for k, v in r.items() if k != "clips"},
                "clips": {x["clip"]: x for x in recs},
            },
            fh,
            ensure_ascii=False,
            indent=1,
        )


def _write_reports(out: Path, rows: dict) -> None:
    """<out>/segment_audit.json (all rows) and segment_audit.md (one table line per dataset)."""
    with open(out / "segment_audit.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    with open(out / "segment_audit.md", "w") as f:
        f.write(
            "| dataset | clips | >15 s | pieces | by rest | by speed minimum | still >15 s | idle pieces | "
            "total clips after split |\n|---|---|---|---|---|---|---|---|---|\n"
        )
        for ds, r in rows.items():
            f.write(
                f"| {ds} | {r['n']} | {r['long']} | {r['segments_from_long']} | {r['cut_by_rest']} | "
                f"{r['cut_by_speed']} | {r['still_long_segments']} | {r['idle_segments']} | "
                f"{r['total_clips_after']} |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--write-intake", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dss = [d for d in a.datasets.split(",") if d] or sorted(dl.DATASETS)
    rows = {}
    for ds in dss:
        recs = _dataset_records(ds)
        if not recs:
            continue
        rows[ds] = _summary(recs)
        r = rows[ds]
        print(
            f"[{ds}] clips {r['n']} · >15 s {r['long']} → pieces {r['segments_from_long']} (by rest "
            f"{r['cut_by_rest']} · speed minimum {r['cut_by_speed']}) · pieces still >15 s "
            f"{r['still_long_segments']} · idle pieces {r['idle_segments']} · total after split "
            f"{r['total_clips_after']}",
            flush=True,
        )
        if a.write_intake:
            _write_intake(ds, r, recs)
    _write_reports(out, rows)
    print("→", out / "segment_audit.md")


if __name__ == "__main__":
    main()
