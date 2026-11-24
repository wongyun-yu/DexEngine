"""Arm of the SPIDER + arm rows: the arm chain context, the arm targets of a variant, the A6 rollout and commanded-
target IK, the arm cache writer and our joint limits."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.policies.spider_arm.spider_load import FREQ_REF, MARK

# ── A.6 constants (the only home)
# ─────────────────────────────────────────────────────────
A6 = {
    "table_n": 4096,  # healthy-table size per arm
    "table_seed": 42,
    "table_center": "robot default arm pose = all 7 arm joints 0 "
    "(combo_articulation InitialStateCfg catch-all '.*': 0)",
    # each joint uniform in [0 − 2.5, 0 + 2.5] ∩ URDF limits (all 7 joints) — the highest-acceptance
    "table_spread_rad": 2.5,
    # value of the sweep (30 variants, 10/task, seed 42: 2.5 → 25/30; 1.5 … 3.2)
    "proximal": ("arm_0", "arm_1", "arm_2", "arm_3"),  # shoulder ×3 + elbow (joint-name suffixes)
    "prox_tol_rad": 0.5,  # proximal joints within 0.5 rad (L∞ over proximal) of the nearest table entry, every frame
    "nearest_rot_w": 0.1,  # nearest table entry to waypoint 0: ‖Δp‖ [m] + 0.1·angle(ΔR) [rad]
    "iters_first": 400,
    "iters_next": 50,
    "iters_cmd": 50,  # commanded-wrist IK, warm-started from the same frame's achieved solution
    "pos_tol_m": 0.03,
    "rot_tol_rad": 0.15,
}


# ── (c)
# A.6
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────
_ARM: dict = {}


def arm_ctx(side: str, hand: str, arm: str = "rby1") -> dict:
    """Robot model + healthy table for one arm (cached). Model/FK = the kin IK code's (dexengine.placement.arm_ik)."""
    key = (arm, hand, side)
    if key in _ARM:
        return _ARM[key]
    import pinocchio as pin  # noqa: PLC0415

    from dexengine.placement import arm_ik as AIK  # noqa: PLC0415

    urdf, ee_frame, joints, base_xyz, _ = AIK.resolve_combo(arm, hand, side)
    from dexengine.robots import urdf_load as UL  # noqa: PLC0415

    model = UL.build_model(str(urdf))
    data = model.createData()
    ee = model.getFrameId(ee_frame)
    qidx = tuple(model.joints[model.getJointId(n)].idx_q for n in joints)
    lo = np.array([model.lowerPositionLimit[i] for i in qidx])
    hi = np.array([model.upperPositionLimit[i] for i in qidx])
    rng = np.random.default_rng(A6["table_seed"])
    s = A6["table_spread_rad"]
    lo_s, hi_s = np.maximum(lo, -s), np.minimum(hi, s)
    if np.any(lo_s > hi_s):
        raise SystemExit(f"{MARK} health table: joint limits do not overlap 0 ± {s} {lo} {hi}")
    tab = rng.uniform(lo_s, hi_s, size=(A6["table_n"], len(joints)))
    prox = [k for k, n in enumerate(joints) if any(n.endswith(p) for p in A6["proximal"])]
    if len(prox) != len(A6["proximal"]):
        raise SystemExit(f"{MARK} not every proximal joint {A6['proximal']} found in {joints}")
    qn = pin.neutral(model)
    tp, tr = np.zeros((len(tab), 3)), np.zeros((len(tab), 3, 3))
    for i, v in enumerate(tab):
        qq = qn.copy()
        for k, qi in enumerate(qidx):
            qq[qi] = v[k]
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        tp[i], tr[i] = data.oMf[ee].translation, data.oMf[ee].rotation
    c = {
        "model": model,
        "data": data,
        "ee": ee,
        "qidx": qidx,
        "joints": list(joints),
        "base_xyz": np.asarray(base_xyz, float),
        "table": tab,
        "table_p": tp,
        "table_R": tr,
        "prox": prox,
        "palm_off": AIK.palm_frame_offset(hand),
        "AIK": AIK,
        "pin": pin,
        "vlim": np.array([float(model.velocityLimit[model.joints[model.getJointId(n)].idx_v]) for n in joints]),
    }
    _ARM[key] = c
    return c


def targets(c: dict, P: np.ndarray, Qwxyz: np.ndarray, zoff: float, base_z_off: float):
    """Palm poses (tree frame) → robot-base targets — the same rule as kin_prep.kin_ik / stage_dp."""
    R = c["AIK"].Rotation  # dexengine.placement.arm_ik's own Rotation (one quaternion at a time — as kin_ik)
    rz = R.from_euler("z", -90, degrees=True)
    base = c["base_xyz"] + np.array([0.0, 0.0, base_z_off])
    TP, TR = np.zeros((len(P), 3)), np.zeros((len(P), 3, 3))
    for t in range(len(P)):
        r = R.from_quat([Qwxyz[t][1], Qwxyz[t][2], Qwxyz[t][3], Qwxyz[t][0]])
        p = P[t] + r.apply(c["palm_off"])
        TP[t] = rz.apply(p) + np.array([0.0, 0.0, zoff]) - base
        TR[t] = (rz * r).as_matrix()
    return TP, TR


def _full(c, v):
    q = c["pin"].neutral(c["model"])
    for k, qi in enumerate(c["qidx"]):
        q[qi] = v[k]
    return q


def _err(c, q, tp, tr):
    pin = c["pin"]
    pin.forwardKinematics(c["model"], c["data"], q)
    pin.updateFramePlacements(c["model"], c["data"])
    mf = c["data"].oMf[c["ee"]]
    return float(np.linalg.norm(mf.translation - tp)), float(np.linalg.norm(pin.log3(mf.rotation.T @ tr)))


def a6_rollout(c: dict, TP: np.ndarray, TR: np.ndarray) -> dict:
    AIK = c["AIK"]
    T = len(TP)
    d = np.linalg.norm(c["table_p"] - TP[0], axis=1) + A6["nearest_rot_w"] * np.array(
        [np.linalg.norm(c["pin"].log3(Rt.T @ TR[0])) for Rt in c["table_R"]]
    )
    i0 = int(np.argmin(d))
    qa = np.zeros((T, len(c["qidx"])))
    pe, re_ = np.zeros(T), np.zeros(T)
    q = _full(c, c["table"][i0])
    for t in range(T):
        q = AIK.solve_ik_pose(
            c["model"],
            c["data"],
            TP[t],
            TR[t],
            c["ee"],
            c["qidx"],
            max_iters=A6["iters_first"] if t == 0 else A6["iters_next"],
            q0=q,
        )[0]
        qa[t] = [q[i] for i in c["qidx"]]
        pe[t], re_[t] = _err(c, q, TP[t], TR[t])
    P = c["prox"]
    dev = np.abs(qa[:, None, P] - c["table"][None, :, P]).max(axis=2).min(axis=1)  # (T,) L∞ to nearest table entry
    acc = bool(pe.max() < A6["pos_tol_m"] and re_.max() < A6["rot_tol_rad"] and dev.max() <= A6["prox_tol_rad"])
    return {"q": qa, "pe": pe, "re": re_, "dev": dev, "i0": i0, "accepted": acc}


def cmd_ik(c: dict, TP: np.ndarray, TR: np.ndarray, q_state: np.ndarray) -> dict:
    AIK = c["AIK"]
    T = len(TP)
    qc = np.zeros_like(q_state)
    pe, re_ = np.zeros(T), np.zeros(T)
    for t in range(T):
        q = AIK.solve_ik_pose(
            c["model"], c["data"], TP[t], TR[t], c["ee"], c["qidx"], max_iters=A6["iters_cmd"], q0=_full(c, q_state[t])
        )[0]
        qc[t] = [q[i] for i in c["qidx"]]
        pe[t], re_[t] = _err(c, q, TP[t], TR[t])
    return {"q": qc, "pe": pe, "re": re_}


def write_arm_cache(out: Path, c: dict, r: dict, base_z_off: float, zoff: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}.npz")
    q = r["q"].astype(np.float32)
    T = len(q)
    np.savez(
        tmp,
        base_xyz=c["base_xyz"].astype(np.float64),
        arm_joint_names=np.array(c["joints"]),
        arm_joint_positions=q[0],
        arm_joint_positions_all=q,
        wrist_pos_error=np.float32(r["pe"][0]),
        wrist_pos_error_all=r["pe"].astype(np.float32),
        wrist_pos_error_def=np.array("euclid_palm_origin_m", dtype="<U24"),
        wrist_rot_error_deg=np.float32(np.degrees(r["re"][0])),
        wrist_rot_error_deg_all=np.degrees(r["re"]).astype(np.float32),
        base_z_off=np.float32(base_z_off),
        num_frames=np.int32(T),
        scene_z_offset=np.float32(zoff),
        solver=np.array("spider_a6_dls", dtype="<U24"),
        arm_vel_limit=c["vlim"],
        frequency=np.float64(FREQ_REF),
        a6_accepted=np.bool_(r["accepted"]),
        a6_prox_dev_all=r["dev"].astype(np.float32),
        a6_table_index=np.int32(r["i0"]),
        a6_constants=np.array(json.dumps(A6, sort_keys=True)),
    )
    os.replace(tmp, out)


# ── our arm limits (paper supp S2 lines 114–124) — the owners'
# functions ──────────────────────────────────────────────
def our_limits(arm: str, hand: str, tree: Path, q: dict, pe_mm: dict, z_off: float, sides: list) -> dict:
    """tracking = stretch-weighted p99 wrist error ≤ judge --dp-err-mm (stretch.weighted_p99, judge
    budget/safety/finger
    floor) · jump = ‖Δq‖₂ > judge --jump-max · swivel = judge._swivel_win > judge.SWIVEL_MAX_DEG ·
    penetration = final_guard.check (table · fixed objects · torso · the other arm) — all thresholds read from the
    owners."""
    from dexengine.placement import final_guard as FG  # noqa: PLC0415
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import hand_surface as HSF  # noqa: PLC0415
    from dexengine.placement.judge import core as J  # noqa: PLC0415
    from dexengine.placement.retime import stretch as VST  # noqa: PLC0415

    ap = J._build_parser()
    err_thr, jump_max, safety = ap.get_default("dp_err_mm"), ap.get_default("jump_max"), ap.get_default("dp_safety")
    J.SCL = SCL
    qpos = np.load(Path(tree) / "0/trajectory_kinematic.npz")["qpos"]
    floor = HSF.finger_floor_for(None, hand, safety, step=1, fq=HSF.tree_finger_q(tree, hand, qpos=qpos))
    out = {}
    for s in sides:
        c = arm_ctx(s, hand, arm)
        bud = c["vlim"] * VST.FRAME_DT
        ep99 = VST.weighted_p99(pe_mm[s], q[s], bud, safety, floor=floor) if len(q[s]) > 1 else float(pe_mm[s][0])
        dq = np.linalg.norm(np.diff(q[s], axis=0), axis=1) if len(q[s]) > 1 else np.zeros(1)
        shim = SimpleNamespace(PEN_CTX={s: SCL.build_arm_ctx(arm, hand, s)})
        sw = J.Judge._swivel_win(shim, s, np.asarray(q[s], float))
        out[s] = {
            "track_p99_mm": round(float(ep99), 2),
            "track_ok": bool(ep99 <= err_thr),
            "jump_max_rad": round(float(dq.max()), 3),
            "jump_ok": bool(dq.max() <= jump_max),
            "swivel_deg_05s": round(float(sw), 1),
            "swivel_ok": bool(sw <= J.SWIVEL_MAX_DEG),
        }
    ok, why = FG.check(arm, hand, tree, q, z_off)  # first violation (its reason carries the measured mm) — values()
    # would re-measure every term (×2 time) · not needed for the flag
    pen = {"pen_ok": bool(ok), "pen_reason": str(why)}
    passes = pen["pen_ok"] and all(v["track_ok"] and v["jump_ok"] and v["swivel_ok"] for v in out.values())
    return dict(
        per_side=out,
        **pen,
        passes=bool(passes),
        thresholds={"track_p99_mm": err_thr, "jump_rad": jump_max, "swivel_deg": J.SWIVEL_MAX_DEG, "safety": safety},
    )
