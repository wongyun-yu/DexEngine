#!/usr/bin/env python3
"""Retargeting baseline quality audit — measures a hand's retargeting results over the datasets with one definition.

The definition matches the reference script; this file is its only owner:
  LIMIT_TOL 1° · SATURATED = a finger joint within 1° of a limit in more than 20% of frames · STUCK = SATURATED and
  that joint's std < 1°.
Per clip, saturation and STUCK are counted from the diagnostics (fingertip and palm error, joint violation, valid
frames) and the qpos ↔ scene.xml joint ranges. Extra metrics: the thumb tip separately (npz `fingertip_error` columns
0 and 5) · the saturated frame fraction (per frame, not per joint) · the hyperextension frame count (final joint
violation > 1e-6). Sitting on a limit (saturation) and passing it (hyperextension) are different metrics — with the
clip on, the first grows and the second is 0.
Output: <out>/retarget_quality.{json,md}. No verdict (exclusion) is made — only the table.
"""

from __future__ import annotations

import argparse
import collections
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexengine.core import paths as P  # noqa

dl = P.data_layout
LIMIT_TOL = np.radians(1.0)
SAT_FRAC = 0.20
STUCK_STD = np.radians(1.0)


def finger_joint_ranges(scene: Path):
    """(name, qpos_index, lo, hi) in worldbody joint order — finger joints only.

    The selection rule is structural: hinges whose `range` is narrower than a full turn. The floating hand's 6 DoFs
    are 3 slides (`*_pos_*`) + 3 full-turn hinges (`*_rot_*`, range ±6.28) and objects are free/ball joints, so they
    all drop out. A name rule (`"finger" in name`) differs per hand — names without "finger" such as `right_joint_1_1`
    would match nothing and give 0 saturation, 0 STUCK and 0 saturated frames = a silent pass. Selecting nothing is a
    failure, not a pass.
    """
    wb = ET.parse(scene).getroot().find("worldbody")
    out = []
    q = 0
    full_turn = 2 * np.pi - 1e-9
    for j in wb.iter("joint"):
        t = j.get("type", "hinge")
        n = j.get("name", "")
        r = j.get("range")
        if t == "free":
            q += 7
            continue
        if t == "ball":
            q += 4
            continue
        if r and t == "hinge":
            lo, hi = map(float, r.split())
            if (hi - lo) < full_turn:
                out.append((n, q, lo, hi))
        q += 1
    for _ in wb.iter("freejoint"):
        q += 7
    if not out:
        raise ValueError(f"no finger joint selected: {scene} — the rule does not fit this rig (0 is not a pass)")
    return out


def audit_clip(d: Path):
    with open(d / "0/retarget_diagnostics.json") as fh:
        g = json.load(fh)
    q = np.load(d / "0/trajectory_kinematic.npz")["qpos"]
    rng = finger_joint_ranges(d / "scene.xml")
    sat = stuck = 0
    sat_names = []
    for n, i, lo, hi in rng:
        if i >= q.shape[1]:
            continue
        col = q[:, i]
        near = ((col - lo) < LIMIT_TOL) | ((hi - col) < LIMIT_TOL)
        if near.mean() > SAT_FRAC:
            sat += 1
            sat_names.append(n)  # without the joint names "11 saturated" is not actionable
            if col.std() < STUCK_STD:
                stuck += 1
    T = int(q.shape[0])
    # Two per-frame metrics and the thumb separately — per-joint and per-clip metrics cannot answer "how many frames
    # sit on a limit" or "the thumb alone". This function is their only owner.
    #   sat_frame_frac : fraction of frames in which at least one finger joint is within 1° of a limit
    #   overext_frames : frames with a remaining final joint violation = hyperextension past a limit. 0 with the
    #                    clip on; frames sitting on a limit are counted as saturation above, not as violations.
    #   tip_thumb_mm   : thumb tip error (both hands). The fingertip_error columns of diagnostics.npz follow the hand
    #                    tracker's [right,left]×[thumb,index,middle,ring,pinky], so the thumbs are columns 0 and 5.
    sat_frame = np.zeros(T, dtype=bool)
    for _n, i, lo, hi in rng:
        if i >= q.shape[1]:
            continue
        col = q[:, i]
        sat_frame |= ((col - lo) < LIMIT_TOL) | ((hi - col) < LIMIT_TOL)
    tip_thumb_mm = float("nan")
    overext = None
    npz = d / "0/retarget_diagnostics.npz"
    if npz.exists():
        z = np.load(npz)
        fe = z["fingertip_error"]
        if fe.ndim == 2 and fe.shape[1] >= 6:
            th = fe[:, [0, 5]]
            if np.isfinite(th).any():
                tip_thumb_mm = float(np.nanmean(th)) * 1000
        overext = int((np.asarray(z["joint_violation_final"]) > 1e-6).sum())
    return {
        "T": T,
        "n_joints": len(rng),
        "saturated": sat,
        "stuck": stuck,
        "sat_joints": sat_names,
        "sat_frame_frac": (round(float(sat_frame.mean()), 4) if T else float("nan")),
        "overext_frames": overext,
        "tip_thumb_mm": tip_thumb_mm,
        "hx_joints": len(g.get("hyperext_limited_joints") or []),
        "fingertip_mm": g.get("fingertip_error_mean_m", float("nan")) * 1000,
        "palm_r_mm": (g.get("palm_error_right_mean_m") or float("nan")) * 1000,
        "palm_l_mm": (g.get("palm_error_left_mean_m") or float("nan")) * 1000,
        "viol": g.get("joint_violation_final_max_rad", float("nan")),
        "valid": g.get("num_valid_frames"),
        "multistart": g.get("ik_multistart"),
        "finite": bool(np.isfinite(q).all()),
    }


def _diag_npz(root: Path, ti: dict) -> Path:
    return Path(root) / ti["provenance"]["retarget_dir"] / "0/retarget_diagnostics.npz"


def window_fingertip_mm(root: Path, ti: dict) -> float:
    """Mean fingertip error (mm) over a tree's window — the nanmean of the retargeting-layer diagnostic
    `fingertip_error` (MANO time axis) cut to the tree window [segment.start + frame_map.offset, + segment.T). The same
    definition as the take summary (`fingertip_error_mean_m`), applied to the window only — a piece is measured on its
    own window (never a copy of the take value). Missing diagnostics or a short window are an error."""
    sg = ti["segment"]
    a, n = int(sg["start"]) + int((sg.get("frame_map") or {}).get("offset", 0)), int(sg["T"])
    with np.load(_diag_npz(root, ti)) as z:
        w = np.asarray(z["fingertip_error"], float)[a : a + n]
    if len(w) != n or not np.isfinite(w).any():
        raise ValueError(
            f"{ti.get('clip_id')}: fingertip error window [{a}, {a + n}) is not in the diagnostics "
            f"({_diag_npz(root, ti)})"
        )
    return float(np.nanmean(w)) * 1000


def diag_ident(root: Path, ti: dict) -> list | None:
    """Identity (size, mtime) of the input file of the measurement above — read by the 4b input fingerprint. None when
    missing (the measurement then fails)."""
    try:
        st = _diag_npz(root, ti).stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--hand", required=True, help="hand spec directory (a paths.hand_spec_dir value, e.g. dg5f_s · no default)"
    )
    ap.add_argument("--datasets", default="")
    ap.add_argument(
        "--clips",
        default="",
        help="measure only these clips — a file with one per line or a comma list. Finding none is an error",
    )
    # Entry for measuring the outputs of an isolated run (stage1 --out-root/--variant) with the same definition — the
    # same measurement on another input, not a second path. Empty = the canonical layer.
    ap.add_argument(
        "--retarget-root",
        default=None,
        help="give the retargeting layer root directly (for isolated A/B). "
        "E.g. overfit/stage1_x/taco_retarget/processed/taco/<hand>/bimanual",
    )
    return ap


def _keep_clips(a):
    """The --clips selection as a set of clip names (None = every clip). A subset is a selection, not another
    measurement — a canary subset and the full set use the same definition."""
    keep = None
    if a.clips:
        src = Path(a.clips)
        names = [x.strip() for x in (src.read_text().splitlines() if src.is_file() else a.clips.split(","))]
        keep = set(P.read_clip_list([n for n in names if n], label="clips", quiet=True))
        if not keep:
            raise SystemExit(f"--clips is empty: {a.clips}")
    return keep


def _dataset_recs(a, ds, keep):
    """One audit record per retargeted clip of ``ds`` (every emb of the canonical layer, or the isolated root)."""
    recs = []
    for emb in dl.EMBODIMENTS:
        if a.retarget_root:
            if emb != "bimanual":
                continue  # an isolated root points at one emb
            root = Path(a.retarget_root)
        else:
            try:
                root = P.layer_path("retarget", ds, hand=a.hand, emb=emb)
            except (KeyError, ValueError):
                continue
        if not root.is_dir():
            continue
        cand = sorted(p for p in root.iterdir() if (p / "0/trajectory_kinematic.npz").exists())
        if keep is not None:
            cand = [p for p in cand if p.name in keep]
        for d in cand:
            try:
                r = audit_clip(d)
                r["clip"] = d.name
                r["emb"] = emb
                recs.append(r)
            except Exception as e:  # noqa: PERF203, BLE001 — an unreadable clip becomes an error row; the audit goes on
                recs.append({"clip": d.name, "emb": emb, "error": f"{type(e).__name__}: {str(e)[:80]}"})
    return recs


def _dataset_row(recs, ok):
    """The summary row of one dataset (medians, p95s and counts over the clips that audited cleanly)."""

    def med(k):
        v = [r[k] for r in ok if r.get(k) is not None and np.isfinite(r[k])]
        return round(float(np.median(v)), 2) if v else None

    def p95(k):
        v = [r[k] for r in ok if r.get(k) is not None and np.isfinite(r[k])]
        return round(float(np.percentile(v, 95)), 2) if v else None

    return {
        "n": len(recs),
        "errors": len(recs) - len(ok),
        "multistart_all4": all(r.get("multistart") == 4 for r in ok),
        "finite_all": all(r["finite"] for r in ok),
        "fingertip_mm_p50": med("fingertip_mm"),
        "fingertip_mm_p95": p95("fingertip_mm"),
        "palm_r_mm_p50": med("palm_r_mm"),
        "palm_l_mm_p50": med("palm_l_mm"),
        "palm_r_mm_p95": p95("palm_r_mm"),
        "palm_l_mm_p95": p95("palm_l_mm"),
        "viol_max": round(max((r["viol"] for r in ok if np.isfinite(r["viol"])), default=float("nan")), 4),
        "clips_with_saturated": sum(1 for r in ok if r["saturated"] > 0),
        "clips_with_stuck": sum(1 for r in ok if r["stuck"] > 0),
        "fingertip_gt_10mm": sum(1 for r in ok if r["fingertip_mm"] > 10),
        "palm_gt_30mm": sum(1 for r in ok if (r["palm_r_mm"] > 30) or (r["palm_l_mm"] > 30)),
        "thumb_mm_p50": med("tip_thumb_mm"),
        "thumb_mm_p95": p95("tip_thumb_mm"),
        "thumb_gt_10mm": sum(
            1 for r in ok if np.isfinite(r.get("tip_thumb_mm", float("nan"))) and r["tip_thumb_mm"] > 10
        ),
        "sat_frame_frac_p50": med("sat_frame_frac"),
        "sat_frame_frac_p95": p95("sat_frame_frac"),
        "clips_with_overext": sum(1 for r in ok if (r.get("overext_frames") or 0) > 0),
        "overext_frames_total": sum((r.get("overext_frames") or 0) for r in ok),
        "overext_unmeasured": sum(1 for r in ok if r.get("overext_frames") is None),
        "sat_joint_top": collections.Counter(j for r in ok for j in r.get("sat_joints", [])).most_common(8),
        "clips": recs,
    }


def _print_row(ds, r):
    """One console line per dataset."""
    print(
        f"[{ds}] n={r['n']} err={r['errors']} multistart4={r['multistart_all4']} finite={r['finite_all']} "
        f"tip p50/p95 {r['fingertip_mm_p50']}/{r['fingertip_mm_p95']} mm · thumb p50/p95 "
        f"{r['thumb_mm_p50']}/{r['thumb_mm_p95']} mm · palm R/L p50 {r['palm_r_mm_p50']}/{r['palm_l_mm_p50']} "
        f"· p95 {r['palm_r_mm_p95']}/{r['palm_l_mm_p95']} · violation max {r['viol_max']} "
        f"· saturated clips {r['clips_with_saturated']} · saturated frames p50 {r['sat_frame_frac_p50']} "
        f"· STUCK clips {r['clips_with_stuck']} · tip>10mm {r['fingertip_gt_10mm']} · thumb>10mm "
        f"{r['thumb_gt_10mm']} · palm>30mm {r['palm_gt_30mm']} · hyperextension frames {r['overext_frames_total']} "
        f"(clips {r['clips_with_overext']}, unmeasured {r['overext_unmeasured']})",
        flush=True,
    )


def _write_md(out, a, rows):
    """The markdown table next to retarget_quality.json."""
    with open(out / "retarget_quality.md", "w") as f:
        f.write(
            f"# Retargeting baseline quality (hand {a.hand}) — definition: limit 1°, saturation "
            f"20%, STUCK std<1°\n\n| dataset | clips | tip mm p50/p95 | thumb mm p50/p95 | palm "
            f"R/L mm p50 | palm R/L mm p95 | joint violation max | saturated clips | saturated "
            f"frames p50/p95 | STUCK clips | tip>10mm | thumb>10mm | palm>30mm | hyperextension "
            f"frames (clips) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        for ds, r in rows.items():
            f.write(
                f"| {ds} | {r['n']} | {r['fingertip_mm_p50']}/{r['fingertip_mm_p95']} "
                f"| {r['thumb_mm_p50']}/{r['thumb_mm_p95']} "
                f"| {r['palm_r_mm_p50']}/{r['palm_l_mm_p50']} | {r['palm_r_mm_p95']}/{r['palm_l_mm_p95']} "
                f"| {r['viol_max']} | {r['clips_with_saturated']} "
                f"| {r['sat_frame_frac_p50']}/{r['sat_frame_frac_p95']} "
                f"| {r['clips_with_stuck']} | {r['fingertip_gt_10mm']} | {r['thumb_gt_10mm']} "
                f"| {r['palm_gt_30mm']} | {r['overext_frames_total']}({r['clips_with_overext']}) "
                f"|\n"
            )


def main():
    ap = _parser()
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.hand not in set(P.HAND_SPEC_DIR.values()):  # callers pass the spec directory
        raise SystemExit(
            f"[rq] --hand {a.hand!r} is not a registered hand spec directory: {sorted(set(P.HAND_SPEC_DIR.values()))}"
        )
    dss = [d for d in a.datasets.split(",") if d] or sorted(dl.DATASETS)
    rows = {}
    keep = _keep_clips(a)
    for ds in dss:
        recs = _dataset_recs(a, ds, keep)
        ok = [r for r in recs if "error" not in r]
        if not recs:
            if keep is not None:
                raise SystemExit(f"[{ds}] none of the {len(keep)} --clips found — check the hand and layer")
            continue
        if keep is not None and len(recs) != len(keep):
            print(
                f"[{ds}] only {len(recs)} of {len(keep)} --clips found: {sorted(keep - {r['clip'] for r in recs})[:5]}",
                flush=True,
            )
        rows[ds] = _dataset_row(recs, ok)
        r = rows[ds]
        _print_row(ds, r)
    with open(out / "retarget_quality.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1, default=str)
    _write_md(out, a, rows)
    print("→", out / "retarget_quality.md")


if __name__ == "__main__":
    main()
