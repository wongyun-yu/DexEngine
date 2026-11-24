#!/usr/bin/env python3
"""Per-clip support surface (desk) verdict: measures from the data whether there is a desk, where it is, and whether the
hand goes below it.

Many sequences start with the object already held in the air (no desk, or a desk further down — arctic ...). Contact
flags differ in reliability per dataset (grab: 1 on every frame), so the verdict is geometric only:
  rest interval    smoothed (0.3 s) object speed < V_REST (3 cm/s) for ≥ REST_MIN_S (0.5 s), with the object bottom z
                   range < Z_STAB (1 cm) meanwhile (angular speed is not used: quaternion noise of the 30→50 Hz
                   interpolation breaks the rest verdict)
  placed interval  samples (every 5 frames) of a rest interval whose min distance between the 5×2 fingertips and the
                   object surface (sampled vertices) is > TIP_FREE (2.5 cm), for at least REST_MIN_S; separates it from
                   holding still (approach/release at the interval edges is filtered per sample)
  support z        median of the object mesh's lowest point over placed intervals. None → no support (aerial start /
      held throughout)
  footprint        object xy range over placed intervals ± FOOT_MARGIN (0.10 m)
  conflict frame   hand keypoint z < support z − HAND_TOL (0.02) inside the footprint
Classes: table (support, no conflict) · table_conflict (support, conflict) · none (no support).
     Also: a nonzero table_z (only when the placed interval is not at the first frame) is an estimate of the actual desk
         height.
The desk height (rest-estimate branch) is decided by desk_rule alone; the marker `[DESK_RULE] <dataset> …` prints counts
per branch.
Output: <out>/support_audit.{json,md}; base_adapter copies the per-clip records into task_info as `support`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dexengine.core import paths as P

dl = P.data_layout
# The rest-interval thresholds (V_REST · Z_STAB · REST_MIN_S · TIP_FREE) belong to the dependency-free support.rest;
# coords (D20 · D26′) reads the same Z_STAB. The rest are used by this audit only.
from dexengine.data.support.rest import TIP_FREE, V_REST  # noqa: E402,I001
from dexengine.data.support import desk_rule as DR  # noqa: E402 — owner of the desk height rule (thresholds · branches · marker)
from dexengine.data.support.plane_audit import FOOT_MARGIN, HAND_TOL, audit_clip


# Policy.
# Desk presence is decided from the **literature** (dataset profiles): TACO (per-session
# measured
# desk z) · HO-Cap ("cameras above a table") · ARCTIC ("props such as the table") · GRAB ("objects static on a table") ·
# H2O (tabletop, 3 environments) · OakInk2 (kitchen/study/lab/bathroom table) · HRDexDB · DexCanvas (grasp trials: the
# object is placed at the start; the audit finds every clip placed at frame 0). **Only SHOW3D** is in the wild
# (outdoors,
# corridors; held in hand). Clips are judged by geometry (placed intervals); only the reading without a placed interval
# follows the dataset (values = the desk_mode vocabulary below):
#   conflict (hand below the support inside the footprint): a desk is wide, so this is read as **annotation
#   penetration / tracking defect**, not a pedestal, and warned; grounding allows or cuts per frame.
#: desk_mode vocabulary: every value written to a tree's support. These names are the one owner of the values:
#: producers (apply_policy · desk recoverers hot3d/oakink2/hrdexdb_desk_recover) write them and consumers
#: (dexengine.data.candidates D4 · D20 · cases · visualisation · summaries) read them. A value outside the vocabulary
#: makes the first-round candidate fail (dexengine.data.candidates).
#:   table            placed interval present → desk top = that height (desk_z, data z; may be below 0 — h2o)
#:                    · official/by-construction desk surface · depth plane
#:   table_recovered  a desk recoverer retrieved the height from the source (hot3d · oakink2)
#:   table_unknown    dataset with a desk but no placed interval (grasp start) → height unknown,
#:                    placed at the hand/object lowest point − LOW_MARGIN (any desk is further down)
#:   none             no desk (SHOW3D in the wild · D10-g arctic); numbers as for table_unknown
#:   unknown          support classification failed (a class apply_policy does not know; support_status fail)
DESK_TABLE, DESK_RECOVERED, DESK_HEIGHT_UNKNOWN = "table", "table_recovered", "table_unknown"
DESK_NONE, DESK_UNCLASSIFIED = "none", "unknown"
DESK_MODES = (DESK_TABLE, DESK_RECOVERED, DESK_HEIGHT_UNKNOWN, DESK_NONE, DESK_UNCLASSIFIED)
TABLETOP = {
    "taco",
    "hrdexdb",
    "dexcanvas",
    "hocap",
    "grab",
    "h2o",
    "oakink2",
    "hot3d",
}  # hot3d: tabletop (object rest z≈−0.72 world, z-up), rest-estimate (D10) path
# D10-g: arctic has no desk. The source has no desk mesh (marker sets only) and objects are mostly handled in the air
# (lowest hand point p50 -9 cm). The delivered z0 is the object bottom at frame 0, so the rest estimate reads "table",
# but that is only the frame definition. It goes to the deskless scene variant (aerial start): desk_mode none, not a
# first-round candidate (D4). The NO_DESK reason text is stored in the support audit (kept verbatim).
NO_DESK = {"arctic": "no desk (mid-air manipulation, no raw desk geometry) — D10-g; same scene transform as show3d"}
# Datasets whose mano-frame z 0 is **the desk surface itself** (TACO: re-based to the desk frame from the per-session
# measurement in taco_frames.py). Elsewhere z 0 is "the object's lowest point at frame 0", so the desk height is unknown
# for clips starting in a grasp; here the desk = 0 even without a placed interval.
PLANE_IS_DESK = {"taco"}
LOW_MARGIN = 0.03
# D10-a · D10-d: datasets whose delivered-frame z 0 is **the official/by-construction desk surface**: constant 0 instead
# of the per-clip rest estimate. The rest estimate is an oracle (warn when it differs by more than 1 cm).
#   hocap     official code constant: world = tag_1 (AprilTag on the desk), point cloud crop z ≥ −0.01 (HO-Cap
#   sequence_loader.py) → desk surface = z 0
#   dexcanvas delivered z0 = object bottom at frame 0 (149/149 clips, |z|<2 mm 100 %), hand lowest point p5 +3 mm
#   (always above) → 0 by construction
#   The OFFICIAL_ZERO reason texts are stored in the support audit (kept verbatim).
OFFICIAL_ZERO = {
    "hocap": "official world tag_1, desk plane = z 0 (D10-a)",
    "dexcanvas": "delivered z0 = object bottom (all clips), 0 by construction (D10-d)",
}
OFFICIAL_ZERO_TOL = 0.01


def _tilted_plane_policy(rec: dict, hz) -> dict:
    """h2o with a depth-plane tilt: desk = plane height; rest estimate and object lowest point are oracles."""
    # D10-e'': the capture frame has been rotated so the source depth desk plane normal is +z, so desk = plane
    # height (pivot.z). Rest estimate and object lowest point are oracles.
    pz = float(rec["h2o_tilt"]["desk_z_plane"])
    tz = rec.get("table_z")
    oz = rec.get("obj_zmin")
    why = []
    if tz is not None and abs(float(tz) - pz) > 0.02:
        why.append(f"rest estimate {tz:.3f} differs from the plane {pz:.3f} by more than 2 cm")
    if oz is not None and float(oz) < pz - 0.01:
        why.append(
            f"object lowest point {oz:.3f} is {(pz - oz) * 100:.1f} cm below the plane (suspected tracking defect)"
        )
    if hz is not None and float(hz) < pz - HAND_TOL:
        why.append(f"hand lowest point {hz:.3f} below the plane (D14 candidate)")
    return {
        "desk_mode": DESK_TABLE,
        "desk_z": round(pz, 4),
        "desk_z_est": tz,
        "desk_z_confidence": "depth_plane",
        "support_status": "warn" if why else "ok",
        "why": f"D10-e'': raw depth plane (tilt {rec['h2o_tilt']['tilt_deg']}°, pivot "
        f"{rec['h2o_tilt']['pivot_source']}); after rotation desk = plane height"
        + ("; " + " · ".join(why) if why else ""),
    }


def _no_desk_policy(ds: str, rec: dict, hz) -> dict:
    """Dataset without a desk: lowest point − LOW_MARGIN (for visualisation)."""
    vals = [v for v in (hz, rec.get("obj_zmin")) if v is not None]
    return {
        "desk_mode": DESK_NONE,
        "desk_z": round(min(vals) - LOW_MARGIN, 3) if vals else -LOW_MARGIN,
        "desk_z_est": rec.get("table_z"),
        "desk_z_confidence": "none",
        "support_status": "ok",
        "why": NO_DESK[ds] + " · value = lowest point − 3 cm (for visualisation)",
    }


def _official_zero_policy(ds: str, rec: dict) -> dict:
    """Dataset whose desk surface is z 0 by construction; the rest estimate is a check."""
    tz = rec.get("table_z")
    ok = tz is None or abs(float(tz)) <= OFFICIAL_ZERO_TOL
    return {
        "desk_mode": DESK_TABLE,
        "desk_z": 0.0,
        "support_status": "ok" if ok else "warn",
        "desk_z_est": tz,
        "desk_z_confidence": "official",
        "why": OFFICIAL_ZERO[ds]
        + (
            ""
            if ok
            else f"; rest estimate table_z {tz} differs from the official 0 by more than "
            f"{OFFICIAL_ZERO_TOL * 100:.0f} cm"
        ),
    }


def _conflict_policy(ds: str, rec: dict, T: int) -> dict:
    """Desk with the hand below it inside the footprint: desk_rule height, warned."""
    d = rec.get("conflict_depth_p50", 0.0)
    out = {
        "desk_mode": DESK_TABLE,
        "desk_z": rec["table_z"],
        "support_status": "warn",
        "why": f"hand below the support inside the footprint, depth {d} · {rec['conflict_frames']}/{T} frames — "
        "annotation penetration/tracking defect, allowed/clipped in grounding",
    }
    f = DR.desk_fields(
        rec["table_z"], rec["obj_zmin_all"], ds in PLANE_IS_DESK
    )  # the same function for the conflict class too
    if "why" in f:
        f["why"] = f["why"] + " · " + out["why"]
    out.update(f)
    return out


def _no_support_policy(ds: str, rec: dict, hz) -> dict:
    """No placed interval (grasp start): desk plane, unknown height or no desk, per dataset."""
    vals = [v for v in (hz, rec.get("obj_zmin")) if v is not None]
    low = round(min(vals) - LOW_MARGIN, 3) if vals else -LOW_MARGIN
    if ds in PLANE_IS_DESK:
        st = "ok" if (hz is None or hz >= -0.02) else "warn"
        return {
            "desk_mode": DESK_TABLE,
            "desk_z": 0.0,
            "support_status": st,
            "why": "frame z 0 = desk plane (session measurement); grasp start, so only the resting pose is missing"
            + ("" if st == "ok" else f" · hand lowest {hz} — suspected tracking defect"),
        }
    if ds in TABLETOP:
        est = rec.get("desk_z_est_handrest")
        return {
            "desk_mode": DESK_HEIGHT_UNKNOWN,
            "desk_z": low,
            "desk_z_est": est,
            "desk_z_confidence": "low" if est is not None else "none",
            "support_status": "ok",
            "why": "dataset with a desk, grasp start → height unknown: lowest point − 3 cm (est = hand-rest "
            "cue; replaced when the raw ground_offset_z arrives)",
        }
    return {
        "desk_mode": DESK_NONE,
        "desk_z": low,
        "support_status": "ok",
        "why": "no desk (in-the-wild): lowest point − 3 cm",
    }


def apply_policy(ds: str, rec: dict) -> dict:
    """Desk verdict of one audited clip: desk_mode, desk_z, support_status and why (per dataset, then per class)."""
    cls = rec.get("cls")
    T = max(1, rec.get("T", 1))
    hz = rec.get("hand_zmin")
    if ds == "h2o" and rec.get("h2o_tilt"):
        return _tilted_plane_policy(rec, hz)
    if ds in NO_DESK:
        return _no_desk_policy(ds, rec, hz)
    if ds in OFFICIAL_ZERO:
        return _official_zero_policy(ds, rec)
    if cls == "table":
        # a missing input obj_zmin_all fails loudly (the sampled obj_zmin is not a substitute)
        out = {"desk_mode": DESK_TABLE, "desk_z": rec["table_z"], "support_status": "ok"}
        out.update(DR.desk_fields(rec["table_z"], rec["obj_zmin_all"], ds in PLANE_IS_DESK))
        return out
    if cls == "table_conflict":
        return _conflict_policy(ds, rec, T)
    if cls == "none":
        return _no_support_policy(ds, rec, hz)
    return {"desk_mode": DESK_UNCLASSIFIED, "desk_z": None, "support_status": "fail", "why": str(cls)}


def _parse_args() -> argparse.Namespace:
    """Command line: --out, --datasets, --max-clips, --write-intake, --h2o-untilted."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument(
        "--write-intake",
        action="store_true",
        help="also write per-dataset results to the canonical INTAKE/support_audit.json",
    )
    ap.add_argument(
        "--h2o-untilted",
        action="store_true",
        help="h2o only: the first pass without the desk-plane rotation (rest intervals for the plane pivot), written "
        "to INTAKE/" + UNTILTED_FILE,
    )
    a = ap.parse_args()
    if a.h2o_untilted and a.datasets != "h2o":
        ap.error("--h2o-untilted needs --datasets h2o")
    return a


def _clip_row(c: Path, obj_root: Path, ds: str, emb: str, h2o_untilted: bool = False) -> dict:
    """Support audit row of one clip with its desk policy; a failing clip becomes an error row."""
    try:
        r = audit_clip(c, json.loads((c / "task_info.json").read_text()), obj_root, ds=ds, h2o_untilted=h2o_untilted)
        r["clip"] = c.name
        r["emb"] = emb
        r["policy"] = apply_policy(ds, r)
    except Exception as e:  # noqa: BLE001  per-clip batch audit: the error is the clip's row, the run goes on
        return {"clip": c.name, "emb": emb, "cls": "error", "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return r


def _audit_dataset(ds: str, max_clips: int, h2o_untilted: bool = False) -> list[dict]:
    """Audit rows of every clip of one dataset over its embodiments (max_clips > 0 caps each embodiment)."""
    recs = []
    for emb in dl.EMBODIMENTS:
        try:
            root = P.layer_path("mano", ds, emb=emb)
        except (KeyError, ValueError):
            continue
        if not root.is_dir():
            continue
        obj_root = P.layer_path("assets_objects", ds)
        clips = sorted(
            p for p in root.iterdir() if (p / "0/trajectory_keypoints.npz").exists() and (p / "task_info.json").exists()
        )
        if max_clips:
            clips = clips[:max_clips]
        recs.extend(_clip_row(c, obj_root, ds, emb, h2o_untilted) for c in clips)
    return recs


def _summarize(rows: dict, ds: str, recs: list[dict]) -> tuple[dict, dict]:
    """Store the dataset row (class counts, table_z percentiles, conflict medians, clips) in rows[ds]; returns
    the desk_mode and support_status counts."""
    cnt = {
        k: sum(1 for r in recs if r.get("cls") == k) for k in ("table", "table_conflict", "none", "no_hand", "error")
    }
    tz = [r["table_z"] for r in recs if r.get("table_z") is not None]
    rows[ds] = {
        "n": len(recs),
        **cnt,
        "rest_at_start": sum(1 for r in recs if r.get("rest_at_start")),
        "table_z_p50": round(float(np.median(tz)), 3) if tz else None,
        "table_z_p95": round(float(np.percentile(tz, 95)), 3) if tz else None,
        "conflict_frames_p50": round(
            float(np.median([r["conflict_frames"] for r in recs if r.get("cls") == "table_conflict"])), 0
        )
        if cnt["table_conflict"]
        else None,
        "conflict_depth_p50": round(
            float(np.median([r["conflict_depth_p50"] for r in recs if r.get("cls") == "table_conflict"])), 3
        )
        if cnt["table_conflict"]
        else None,
        "clips": recs,
    }
    modes = {}
    for r_ in recs:
        k = (r_.get("policy") or {}).get("desk_mode", "?")
        modes[k] = modes.get(k, 0) + 1
    st = {}
    for r_ in recs:
        k = (r_.get("policy") or {}).get("support_status", "?")
        st[k] = st.get(k, 0) + 1
    rows[ds]["desk_modes"] = modes
    rows[ds]["support_status"] = st
    return modes, st


#: INTAKE file of the support audit, and of the H2O untilted first pass (read only by the desk-plane fit,
#: data/support/h2o.py — never by the base layer, so a chain that stops after it leaves no valid-looking audit)
AUDIT_FILE = "support_audit.json"
UNTILTED_FILE = "support_audit_untilted.json"


def _write_intake(rows: dict, ds: str, recs: list[dict], name: str = AUDIT_FILE) -> None:
    """INTAKE/<name> of one dataset: the policy constants, the summary and the per-clip records."""
    try:
        idir = P.layer_path("intake", ds)
        idir.mkdir(parents=True, exist_ok=True)
        with open(idir / name, "w") as fh:
            json.dump(
                {
                    "tool": "dexengine/data/support/plane.py",
                    "policy": {
                        "TABLETOP": sorted(TABLETOP),
                        "PLANE_IS_DESK": sorted(PLANE_IS_DESK),
                        "OFFICIAL_ZERO": OFFICIAL_ZERO,
                        "NO_DESK": NO_DESK,
                        "LOW_MARGIN": LOW_MARGIN,
                        "DESK_MID_START_M": DR.DESK_MID_START_M,
                        "DESK_MID_STOP_M": DR.DESK_MID_STOP_M,
                        "DESK_MID_FRACTION": DR.DESK_MID_FRACTION,
                        "V_REST": V_REST,
                        "TIP_FREE": TIP_FREE,
                        "FOOT_MARGIN": FOOT_MARGIN,
                        "HAND_TOL": HAND_TOL,
                    },
                    "summary": {k: v for k, v in rows[ds].items() if k != "clips"},
                    "clips": {
                        r_["clip"]: {
                            "emb": r_["emb"],
                            "cls": r_.get("cls"),
                            "table_z": r_.get("table_z"),
                            "footprint": r_.get("footprint"),
                            "conflict_frames": r_.get("conflict_frames"),
                            "conflict_depth_p50": r_.get("conflict_depth_p50"),
                            "hand_zmin": r_.get("hand_zmin"),
                            "obj_zmin": r_.get("obj_zmin"),
                            "obj_zmin_all": r_.get("obj_zmin_all"),
                            "rest_intervals": r_.get("rest_intervals"),
                            **(r_.get("policy") or {}),
                        }
                        for r_ in recs
                    },
                },
                fh,
                ensure_ascii=False,
                indent=1,
                default=str,
            )
    except (KeyError, ValueError) as e:
        print(f"  [{ds}] INTAKE write failed: {e}")


def _write_reports(out: Path, rows: dict) -> None:
    """<out>/support_audit.json and the per-dataset table <out>/support_audit.md."""
    with open(out / "support_audit.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1, default=str)
    with open(out / "support_audit.md", "w") as f:
        f.write(
            "| dataset | clips | desk (no conflict) | desk (hand below) | no support (aerial/held "
            "throughout) | placed from frame 0 | desk z p50/p95 | policy desk_mode | status "
            "|\n|---|---|---|---|---|---|---|---|---|\n"
        )
        for ds, r in rows.items():
            f.write(
                f"| {ds} | {r['n']} | {r['table']} | {r['table_conflict']} | {r['none']} | "
                f"{r['rest_at_start']} | {r['table_z_p50']}/{r['table_z_p95']} | {r['desk_modes']} | "
                f"{r['support_status']} |\n"
            )
    print("→", out / "support_audit.md")


def main():
    a = _parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # the description fields enter the tree content fingerprint; rewriting datasets with unchanged values would split
    # clips
    if a.write_intake and not a.datasets:
        raise SystemExit("[SUPPORT_AUDIT] ✗ --write-intake requires --datasets (no rewrite of every dataset)")
    dss = [d for d in a.datasets.split(",") if d] or sorted(dl.DATASETS)
    rows = {}
    for ds in dss:
        recs = _audit_dataset(ds, a.max_clips, a.h2o_untilted)
        if not recs:
            continue
        if a.h2o_untilted:
            print(f"[H2O_TILT] support_audit: untilted first pass, {len(recs)} clips (pivot input)", flush=True)
        elif ds == "h2o":
            nt = sum(1 for r in recs if r.get("h2o_tilt"))
            ne = sum(1 for r in recs if r.get("cls") == "error")
            print(
                f"[H2O_TILT] support_audit: plane rotation {nt}/{len(recs)} clips · error {ne} (no plane / "
                "below quality = excluded)",
                flush=True,
            )
        modes, st = _summarize(rows, ds, recs)
        print(DR.desk_rule_mark(ds, recs, ds in PLANE_IS_DESK), flush=True)
        if a.write_intake:
            _write_intake(rows, ds, recs, UNTILTED_FILE if a.h2o_untilted else AUDIT_FILE)
        r = rows[ds]
        print(f"[{ds}] desk_modes={modes} status={st}")
        print(
            f"[{ds}] n={r['n']} table={r['table']} table_conflict={r['table_conflict']} none={r['none']} err="
            f"{r['error']} "
            f"· placed from frame 0 {r['rest_at_start']} · table_z p50/p95 {r['table_z_p50']}/{r['table_z_p95']} "
            f"· conflict frames p50 {r['conflict_frames_p50']} · conflict depth p50 {r['conflict_depth_p50']}",
            flush=True,
        )
    _write_reports(out, rows)


if __name__ == "__main__":
    main()
