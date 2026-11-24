#!/usr/bin/env python3
"""Arm-IK limits of an arm-joint cache — how many clips of a list pass, limit by limit (a report, no verdict of
record).

Why: the RL ablation row "sequential IK" (paper Sec. E.6 · `arm_ik_seq_frames.py` in this folder) replaces the
canonical arm path by one warm-started run. The table needs the number of clips whose run would have been accepted by
the arm-IK limits of the placement judge. This script measures those limits on ANY arm-IK cache dir pair with the
owners' functions and thresholds (nothing is re-typed here):

  track     wrist position error, 99th percentile weighted by the retiming stretch   stretch.weighted_p99
            ≤ the judge's --dp-err-mm (judge._build_parser default)
  jump      largest step of the 7 arm joints between frames (L2 rad)                  arm_flip.judge_jump_max
  flip      a joint travelling more than arm_flip.TRAVEL_MAX_RAD inside WINDOW_S      arm_flip.triggered
  swivel    elbow swivel change inside 0.5 s > judge.SWIVEL_MAX_DEG                   judge.Judge._swivel_win
  pen       desk · support · torso · arm-arm penetration of the whole run            final_guard.check / values
            (the deciding mesh ruler — arm_proxy.mesh_ruler)
  speed     joint speed / limit per frame: p99 ≤ gate.GATE_SPEED_P99_MAX and max ≤ gate.GATE_SPEED_MAX_MAX
            (reported next to the others: the clips are already retimed for the canonical run)

`wrist_pos_error_all` of a cache is the residual to the target its solver was given. For a cache whose targets are
the FK of another cache (the sequential builder), pass that cache as --ref-right/--ref-left: the error counted is
then residual + the reference's own residual (an upper bound of the distance to the original wrist path).

  python -m experiments.ablations.rl_arm.arm_ik_limits_report --arm rby1 --hand sharpa_wave --ik-right <dir> \
      --ik-left <dir> --trees <bimanual tree root> --out <json> [--ref-right <dir> --ref-left <dir>] [--jobs 16]

The four limit reports of Sec. E.7 were produced by this tool (cold · sequential, each with and
without the reference cache).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SIDES = ("right", "left")
LIMITS = ("track", "jump", "flip", "swivel", "pen")  # the arm-IK limits; speed is reported beside them
_W: dict = {}


def _init(arm: str, hand: str) -> None:
    import pinocchio as pin  # noqa: PLC0415 — heavy imports only in the pool workers

    from dexengine.placement import arm_flip as AF  # noqa: PLC0415
    from dexengine.placement import arm_ik as AIP  # noqa: PLC0415
    from dexengine.placement import final_guard as FG  # noqa: PLC0415
    from dexengine.placement.collision import arm_proxy as APX  # noqa: PLC0415
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.gate import core as GT  # noqa: PLC0415
    from dexengine.placement.judge import core as JD  # noqa: PLC0415
    from dexengine.placement.retime import stretch as VST  # noqa: PLC0415

    budget = {}
    for side in SIDES:
        urdf, _, joints, _, _ = AIP.resolve_combo(arm, hand, side)
        model = pin.buildModelFromUrdf(str(urdf))
        idx = [model.joints[model.getJointId(n)].idx_v for n in joints]
        budget[side] = np.asarray(model.velocityLimit)[idx].astype(np.float64) * VST.FRAME_DT
    dflt = JD._build_parser()
    # The judge fills its heavy modules at boot (judge._import_heavy: torch · the GPU optimizer). The swivel measure
    # needs only the collision module's link transforms, so hand it the same module instead of booting the judge.
    if JD.SCL is None:
        JD.SCL = SCL
    with APX.mesh_ruler():
        ctx = {side: SCL.build_arm_ctx(arm, hand, side) for side in SIDES}
    _W.update(
        arm=arm,
        hand=hand,
        AF=AF,
        APX=APX,
        FG=FG,
        JD=JD,
        VST=VST,
        budget=budget,
        ctx=ctx,
        thr={
            "track_mm": float(dflt.get_default("dp_err_mm")),
            "jump_rad": float(AF.judge_jump_max()),
            "flip_rad": float(AF.TRAVEL_MAX_RAD),
            "flip_window_s": float(AF.WINDOW_S),
            "swivel_deg": float(JD.SWIVEL_MAX_DEG),
            "speed_p99": float(GT.GATE_SPEED_P99_MAX),
            "speed_max": float(GT.GATE_SPEED_MAX_MAX),
        },
    )


def _clip(job) -> dict:
    name, ik, ref, tree = job
    AF, FG, JD, VST, thr = _W["AF"], _W["FG"], _W["JD"], _W["VST"], _W["thr"]
    q, row, zoff = {}, {"clip": name}, {}
    for side in SIDES:
        z = np.load(Path(ik[side]) / f"{name}.npz")
        q[side] = np.asarray(z["arm_joint_positions_all"], dtype=np.float64)
        err = np.asarray(z["wrist_pos_error_all"], dtype=np.float64)
        zoff[side] = float(z["base_z_off"])
        if ref:
            zr = np.load(Path(ref[side]) / f"{name}.npz")
            if int(zr["num_frames"]) != int(z["num_frames"]):
                raise RuntimeError(f"[ARM_IK_LIMITS] {name}/{side}: frames differ from the reference cache")
            err = err + np.asarray(zr["wrist_pos_error_all"], dtype=np.float64)
        b = _W["budget"][side]
        ratio = VST.frame_ratio(q[side], b) if len(q[side]) > 1 else np.zeros(1)
        step = np.linalg.norm(np.diff(q[side], axis=0), axis=1) if len(q[side]) > 1 else np.zeros(1)
        m = {
            "frames": int(len(q[side])),
            "track_p99_mm": VST.weighted_p99(err, q[side], b) * 1e3,
            "track_max_mm": float(err.max() * 1e3),
            "track_frames_over": int((err * 1e3 > thr["track_mm"]).sum()),
            "jump_max_rad": float(step.max()),
            "flip_travel_rad": float(AF.window_travel(q[side])[0]),
            "swivel_win_deg": float(JD.Judge._swivel_win(SimpleNamespace(PEN_CTX=_W["ctx"]), side, q[side])),
            "speed_p99": float(np.quantile(ratio, 0.99)),
            "speed_max": float(ratio.max()),
            "stretch": VST.total_stretch(q[side], b),
        }
        m["ok"] = {
            "track": m["track_p99_mm"] <= thr["track_mm"],
            "jump": m["jump_max_rad"] <= thr["jump_rad"],
            "flip": not AF.triggered(q[side]),
            "swivel": m["swivel_win_deg"] <= thr["swivel_deg"],
            "speed": m["speed_p99"] <= thr["speed_p99"] and m["speed_max"] <= thr["speed_max"],
        }
        row[side] = m
    if zoff["right"] != zoff["left"]:
        raise RuntimeError(f"[ARM_IK_LIMITS] {name}: base_z_off differs between the sides {zoff}")
    with _W["APX"].mesh_ruler():
        ok, why = FG.check(_W["arm"], _W["hand"], str(tree), q, zoff["right"])
        vals = FG.values(_W["arm"], _W["hand"], str(tree), q, zoff["right"])
    # None = a guard term with no pair in this scene
    row["pen"] = {
        "ok": bool(ok),
        "why": str(why),
        "max_mm": {k: None if v is None else float(v) for k, v in vals.items()},
    }
    okd = {k: bool(row["right"]["ok"][k] and row["left"]["ok"][k]) for k in row["right"]["ok"]}
    okd["pen"] = bool(ok)
    row["ok"] = okd
    row["pass_limits"] = all(okd[k] for k in LIMITS)
    row["pass_limits_and_speed"] = row["pass_limits"] and okd["speed"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--ik-right", required=True)
    ap.add_argument("--ik-left", required=True)
    ap.add_argument("--ref-right", default="")
    ap.add_argument("--ref-left", default="")
    ap.add_argument("--trees", required=True, help="tree root whose children are the clip trees")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if bool(a.ref_right) != bool(a.ref_left):
        raise SystemExit("[ARM_IK_LIMITS] --ref-right and --ref-left go together")
    ik = {"right": a.ik_right, "left": a.ik_left}
    ref = {"right": a.ref_right, "left": a.ref_left} if a.ref_right else None
    names = {s: sorted(p.stem for p in Path(ik[s]).glob("*.npz")) for s in SIDES}
    if not names["right"] or names["right"] != names["left"]:
        raise SystemExit(
            f"[ARM_IK_LIMITS] clip sets differ or empty: right {len(names['right'])} left {len(names['left'])}"
        )
    jobs = []
    for n in names["right"]:
        tree = Path(a.trees) / n
        if not (tree / "0" / "trajectory_kinematic.npz").is_file():
            raise SystemExit(f"[ARM_IK_LIMITS] no tree for {n} under {a.trees}")
        jobs.append((n, ik, ref, tree))
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.arm, a.hand)) as pool:
        rows = pool.map(_clip, jobs, chunksize=1)
        thr = pool.apply(_thr)
    if sorted(r["clip"] for r in rows) != names["right"]:
        raise SystemExit("[ARM_IK_LIMITS] measured set != clip set")
    n = len(rows)
    fails = {k: sum(not r["ok"][k] for r in rows) for k in (*LIMITS, "speed")}
    summ = {
        "arm": a.arm,
        "hand": a.hand,
        "ik": ik,
        "ref": ref,
        "trees": a.trees,
        "thresholds": thr,
        "clips": n,
        "pass_limits": sum(r["pass_limits"] for r in rows),
        "pass_limits_and_speed": sum(r["pass_limits_and_speed"] for r in rows),
        "fail_by_limit": fails,
        "limits": list(LIMITS),
        "rows": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=1))
    print(
        f"[ARM_IK_LIMITS] {Path(a.ik_right).name} · clips {n} · pass (track·jump·flip·swivel·pen) "
        f"{summ['pass_limits']}/{n} · with speed {summ['pass_limits_and_speed']}/{n} · fails "
        + " ".join(f"{k}={v}" for k, v in fails.items())
        + f" → {out}",
        flush=True,
    )
    return 0


def _thr() -> dict:
    return _W["thr"]


if __name__ == "__main__":
    sys.exit(main())
