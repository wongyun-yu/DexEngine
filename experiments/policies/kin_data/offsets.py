#!/usr/bin/env python3
"""E5 kinematic baseline — the **single owner** of the placement rules and values.

Rule (placement 'clipside'):
  · start wrist centre s_clip = mean of the clip's **active hands** (collision.idle_arm.active_idle_for_tree) frame-0
    wrist positions (two hands = midpoint) — positions[0] of the raw wrist cache wrist_poses_<side>/<clip>.npz in the
    scene frame: p_scene = Rz(−90°)·p_capture + (0, 0, DESK_HEIGHT).
  · target centre: bimanual clips = c_robot (component-wise median of the same point over all hands and clips of the
  TACO
    pool) · one-hand clips = that hand's work centre c_L | c_R (the same median over the TACO pool wrists where that
    hand is active). The scene frame is defined by the TACO capture.
  · move o_clip = target centre − s_clip (z included). Desk, objects and hands get the same rigid transform (the
  source
    height relations kept).
  · rotation = per-dataset ψ_ds (canon_yaw — the dataset median of the per-clip placement yaw our judge chose) about
    pivot s_clip (tree plane), so s_clip stays on the target centre after rotating.
  · no choice by success (judge or RL), no physics, judge or retiming.
  · applied by prep.py: tree-plane move (dx, dy)_tree = Rz(+90°)·(o_x, o_y) — bake rigid_pl (yaw ψ_ds, pivot
  s_clip).
    z to the robot height: base_z_off = z_def + scene_z_offset(tree) − (DESK_HEIGHT + o_z), z_def = the robot default
    pose's root z (dexengine.placement.search_policy.TACO_CANON × CELL_M).
Output (--out JSON): rule, c_robot, c_side, ψ_ds per dataset (source fingerprint), pool sha256. The per-clip values
(clip_placement) are written by prep.py into the tree's task_info.e5_kin and adopted, marker [KIN_PLACE].
  python3 experiments/policies/kin_data/offsets.py --pool <kinematic_pool.tsv> --data-root <data root> --out <json>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

MARK = "[KIN_OFFSETS]"
HAND_DIR = "wuji2"
CANON_DS = "taco"
RULE = (
    "placement 'clipside': s_clip = mean over the clip's active hands of the frame-0 wrist "
    "position in the scene frame (Rz(-90deg)*capture + (0,0,DESK_HEIGHT)); target = c_robot for a two-hand clip "
    "(per-component median of the same point over all hands of all taco pool clips) and c_<side> for a one-hand "
    "clip (the same median over that side's active hands of the taco pool clips); o_clip = target - s_clip, z "
    "included (robot root z = dz_policy.TACO_CANON*CELL_M + scene_z_offset - (DESK_HEIGHT + o_z)); yaw = "
    "per-dataset yaw_canon_deg = median over that dataset's pool clips of our judging's per-clip adopted placement "
    "yaw (precompute/<ds>/robots/<combo>/placement/<clip>_pl_rt.adopted.json, valid; clips without a record are "
    "counted, not used) about s_clip (tree plane); desk, objects and hands move together; computed once, never "
    "tuned on success; no physics, judge or retime"
)
RZ_NEG90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def desk_height() -> float:
    from dexengine.rl.tasks.cfg import constants as m  # noqa: PLC0415

    return float(m.DESK_HEIGHT)


def robot_default_z() -> float:
    """Root z of the robot default pose — the canonical TACO layer (dexengine.placement.search_policy.TACO_CANON, cell
    size CELL_M) is the only owner (the value is not copied).
    """
    from dexengine.placement import search_policy as m  # noqa: PLC0415

    return float(m.TACO_CANON) * float(m.CELL_M)


def _rule_mods():
    from dexengine.core import qpos_layout as QL  # noqa: PLC0415
    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415

    return IA, QL


RULE_PIPELINE = (
    "placement 'pipeline' (a baseline that keeps all data and attaches the arm in sim by any means): "
    "per clip, our pipeline's placement degrees of freedom - tree-plane shift (dx, dy), yaw about pivot (rigid_pl) and "
    "robot root z (robot_z_off) - are copied verbatim from our stage-1 placement search record "
    "precompute/<ds>/robots/<combo>/placement/<clip>_pl_rt.adopted.json of the inputs root (valid; "
    "placement_src=adopted); a clip without a record gets the same four values from "
    "experiments/policies/kin_data/opt.py "
    "(kinematic-only search over the same space maximizing the fraction of all frames the arm reaches, min over active "
    "hands - not tied to frame 0; placement_src=kin_opt); no reach gate; no physics, judge or retime; IK naive - "
    "unreachable frames keep the nearest solution and only the error is recorded"
)
# clipside = per-hand work centre; pipeline replaced it as the default (clipside is tied to frame 0).
# older rules are gone (history = git)
PLACEMENTS = ("clipside", "pipeline")
RULES = {"clipside": RULE, "pipeline": RULE_PIPELINE}
PLACEMENT_SRCS = ("adopted", "kin_opt")  # per-clip source of pipeline (meta placement_src) — other values die


def adopted_record(inputs: Path, ds: str, clip: str) -> tuple[Path, dict] | None:
    """The clip record of our stage-1 placement search (canonical adopted.json, valid) — None when missing or invalid
    (the caller counts).
    """
    combo = f"rby1_wuji2{ds_flags(ds)['variant']}"
    f = inputs / f"precompute/{ds}/robots/{combo}/placement/{clip}_pl_rt.adopted.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_bytes())
    if not d.get("valid", True) or any(d.get(k) is None for k in ("dx", "dy", "yaw", "pivot", "robot_z_off")):
        return None
    return f, d


def kinopt_path(shadow: Path, ds: str, clip: str) -> Path:
    """Location of the opt.py record (inside the shadow root, our output) — written by opt.py, read by
    clip_placement.
    """
    combo = f"rby1_wuji2{ds_flags(ds)['variant']}"
    return shadow / f"precompute/{ds}/robots/{combo}/placement_kinopt/{clip}_pl_rt.kinopt.json"


def pipeline_placement(inputs: Path, shadow: Path | None, ds: str, clip: str) -> dict:
    """One clip's value for placement 'pipeline' — adopted as is when present, else the opt.py record, else death."""
    rec = adopted_record(inputs, ds, clip)
    if rec is not None:
        f, d = rec
        src, dd = "adopted", d
    else:
        f = kinopt_path(shadow, ds, clip) if shadow is not None else None
        if f is None or not f.is_file():
            raise ValueError(
                f"{MARK} {ds}/{clip}: no placement record (adopted · kin_opt {f}) — run "
                "experiments/policies/kin_data/opt.py first"
            )
        dd = json.loads(f.read_bytes())
        src = dd.get("placement_src")
    if src not in PLACEMENT_SRCS:
        raise ValueError(f"{MARK} {ds}/{clip}: placement_src {src!r} — known: {PLACEMENT_SRCS}")
    return {
        "placement": "pipeline",
        "placement_src": src,
        "record": str(f),
        "record_sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16],
        "tree_shift_xy": [float(dd["dx"]), float(dd["dy"])],
        "yaw": float(dd["yaw"]),
        "pivot": [float(dd["pivot"][0]), float(dd["pivot"][1]), 0.0],
        "robot_z_off": float(dd["robot_z_off"]),
        "offset_scene": None,
        "start_center_scene": None,
        "target_center_scene": None,
    }


def ds_flags(ds: str) -> dict:
    """Dataset launch axes (--variant, --scene of the rules table, datasets.launch_flags) — read only (prep.py uses it
    too).
    """
    from dexengine.data import rules as RR  # noqa: PLC0415

    toks = RR.launch_flags(ds) if hasattr(RR, "launch_flags") else []
    flags = dict(zip(toks[::2], toks[1::2], strict=False)) if toks else {}
    return {"variant": flags.get("--variant", ""), "scene": flags.get("--scene", "table")}


def canon_yaw(root: Path, ds: str, clips: list[str]) -> dict:
    """ψ_ds — the **only owner**: the median of the per-clip placement yaw our judge chose (canonical adopted.json
    `yaw`, valid) over the dataset's pool clips. Source records =
    precompute/<ds>/robots/<combo>/placement/<clip>_pl_rt.adopted.json;
    pool clips without a judge record (clips we dropped) leave the count and their number is written. Source fingerprint
    = sha256 of the sorted 'name\tsha256(file)' lines — it changes when the judge records change (the frozen json
    carries
    it)."""
    combo = f"rby1_wuji2{ds_flags(ds)['variant']}"
    pdir = root / f"precompute/{ds}/robots/{combo}/placement"
    ys, lines, n_none, n_invalid = [], [], 0, 0
    for c in sorted(set(clips)):
        f = pdir / f"{c}_pl_rt.adopted.json"
        if not f.is_file():
            n_none += 1
            continue
        b = f.read_bytes()
        d = json.loads(b)
        if not d.get("valid", True) or d.get("yaw") is None:
            n_invalid += 1
            continue
        ys.append(float(d["yaw"]))
        lines.append(f"{f.name}\t{hashlib.sha256(b).hexdigest()}")
    if not ys:
        raise SystemExit(f"{MARK} {ds}: no judged placement record for the pool clips ({pdir}) — cannot measure ψ_ds")
    return {
        "yaw_canon_deg": round(float(np.median(ys)), 3),
        "yaw_n": len(ys),
        "yaw_pool_no_record": n_none,
        "yaw_pool_invalid": n_invalid,
        "yaw_iqr": [round(float(x), 2) for x in np.percentile(ys, [25, 75])],
        "yaw_combo": combo,
        "yaw_source_dir": str(pdir.relative_to(root)),
        "yaw_source_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16],
    }


def wrist0_scene(root: Path, ds: str, clip: str, side: str, dh: float) -> np.ndarray:
    """One hand's frame-0 wrist position (scene frame) — the one point definition c_robot and s_clip share."""
    p = np.load(root / f"precompute/{ds}/robots/{HAND_DIR}/wrist_poses_{side}/{clip}.npz")["positions"][0]
    return RZ_NEG90 @ np.asarray(p, float) + np.array([0.0, 0.0, dh])


def clip_placement(offs: dict, root: Path, ds: str, clip: str, active: list[str], shadow: Path | None = None) -> dict:
    """One clip's value for placement 'clipside' — the only calculator (called by prep.py): s_clip, o_clip, tree move,
    ψ_ds, pivot (tree plane)."""
    if not active:
        raise SystemExit(f"{MARK} {ds}/{clip}: no active hand — cannot set s_clip")
    pl = offs["placement"]
    if pl not in PLACEMENTS:
        raise SystemExit(f"{MARK} offsets table placement {pl!r} — known: {PLACEMENTS} (old version)")
    if pl == "pipeline":
        return dict(pipeline_placement(root, shadow, ds, clip), active=list(active))
    dh = float(offs["desk_height"])
    s = np.mean([wrist0_scene(root, ds, clip, sd, dh) for sd in active], axis=0)
    c = offs["c_side"][active[0]] if len(active) == 1 else offs["c_robot"]
    o = np.asarray(c, float) - s
    piv = RZ_NEG90.T @ np.array([s[0], s[1], 0.0])
    dso = offs["datasets"][ds]
    return {
        "start_center_scene": [round(float(x), 5) for x in s],
        "offset_scene": [round(float(x), 5) for x in o],
        "tree_shift_xy": [round(x, 6) for x in tree_shift(list(o))],
        "yaw": float(dso["yaw_canon_deg"]),
        "pivot": [round(float(piv[0]), 6), round(float(piv[1]), 6), 0.0],
        "active": list(active),
        "target_center_scene": [round(float(x), 5) for x in c],
        "placement": pl,
    }


def tree_shift(o: list[float]) -> tuple[float, float]:
    """Scene-plane move (o_x, o_y) → tree (capture) plane move = Rz(+90°)·o (scene = Rz(−90°)·tree)."""
    v = RZ_NEG90.T @ np.array([o[0], o[1], 0.0])
    return float(v[0]), float(v[1])


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("rule") != RULES.get(d.get("placement")):
        raise SystemExit(f"{MARK} {path}: rule text differs from this module")
    if abs(float(d.get("robot_default_z", 1e9)) - robot_default_z()) > 1e-6:
        raise SystemExit(f"{MARK} {path}: robot_default_z differs from dexengine.placement.search_policy")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--placement", default="clipside", choices=PLACEMENTS)
    a = ap.parse_args()
    root = Path(a.data_root)
    dh = desk_height()
    with open(a.pool) as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["in_pool"] == "1"]
    IA, QL = _rule_mods()
    pts: dict[str, list] = {}
    side_pts: dict[str, list] = {}
    skipped: dict[str, list] = {}
    for r in rows:
        bdir = root / r["base_dir"]
        ti = json.loads((bdir / "task_info.json").read_text())
        # active hands = canonical D16 + 2a rule (collision.idle_arm.active_idle_for_tree — the function prep.py uses)
        try:
            act, _idle, _rule = IA.active_idle_for_tree(ti.get("case"), QL.read(str(bdir), nq=None).hands)
        except ValueError as e:
            skipped.setdefault(r["ds"], []).append(f"{r['clip']}: {e}"[:200])
            continue
        for s in act:
            pts.setdefault(r["ds"], []).append(wrist0_scene(root, r["ds"], r["clip"], s, dh))
            if r["ds"] == CANON_DS:
                side_pts.setdefault(s, []).append(pts[r["ds"]][-1])
    c = np.median(np.asarray(pts[CANON_DS]), axis=0)
    out = {
        "rule": RULES[a.placement],
        "placement": a.placement,
        "desk_height": dh,
        "robot_default_z": round(robot_default_z(), 6),
        "canon_dataset": CANON_DS,
        "c_robot": [round(float(x), 4) for x in c],
        "c_robot_n_hands": len(pts[CANON_DS]),
        "pool_sha256": hashlib.sha256(Path(a.pool).read_bytes()).hexdigest()[:12],
        "datasets": {},
    }
    if CANON_DS in pts:  # work centre of one-hand clips (per hand, TACO)
        out["c_side"] = {
            sd: [round(float(x), 4) for x in np.median(np.asarray(v), axis=0)] for sd, v in side_pts.items()
        }
        out["c_side_n_hands"] = {sd: len(v) for sd, v in side_pts.items()}
    for ds in sorted(pts):
        out["datasets"][ds] = dict(
            n_clips=sum(1 for r in rows if r["ds"] == ds),
            n_hands=len(pts[ds]),
            **canon_yaw(root, ds, [r["clip"] for r in rows if r["ds"] == ds]),
            n_skipped_no_active=len(skipped.get(ds, [])),
            skipped_examples=skipped.get(ds, [])[:3],
        )
        print(f"{MARK} {ds}: hands {len(pts[ds])} · ψ_ds {out['datasets'][ds]['yaw_canon_deg']}", flush=True)
    tmp = a.out + f".tmp{os.getpid()}"
    Path(tmp).write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.out)
    print(f"{MARK} DONE → {a.out} · c_robot {out['c_robot']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
