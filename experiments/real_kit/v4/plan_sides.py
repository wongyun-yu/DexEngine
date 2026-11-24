"""Per-side pieces of plan.py: the side record of a chosen approach option, the common length K over the sides
(timing.choose_frames) and the side arrays of the schedule (arm and finger rows, check values)."""

from __future__ import annotations

import numpy as np

from experiments.real_kit.v4 import rules as V4
from experiments.real_kit.v4 import timing as TM


def side_record(o: dict, base_name: str) -> dict:
    """One chosen candidate (an active_opts item) → that hand's schedule inputs and record values (route length,
    rotation, clip wrist p90).
    """
    from scipy.spatial.transform import Rotation

    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.retime import stretch as VST

    ik, torch, DT, DEV = o["ik"]
    qc, Tc, T0, qb, rt, q_ent, err = o["qc"], o["Tc"], o["T0"], o["qb"], o["rt"], o["q_ent"], o["err"]
    pq = np.vstack([rt["q"][:-1], q_ent])
    # rotation record (measures for 'not unnatural'): per-joint max rotation |q − q_base|, wrist joints (4–6) max,
    # palm pose rotation angle (fixed pose → frame 0)
    Tb = ik.fk(torch.tensor(qb[None], dtype=DT, device=DEV)).detach().cpu().numpy()[0].astype(np.float64)
    rot = float(np.degrees(Rotation.from_matrix(T0[:3, :3] @ Tb[:3, :3].T).magnitude()))
    jrot = np.abs(pq - qb[None]).max(0)
    Pp = ik.fk(torch.tensor(pq, dtype=DT, device=DEV)).detach().cpu().numpy()[:, :3, 3]
    # clip wrist (palm frame) speed p90 — FK of the retimed IK (IK frame, m/s)
    vc = np.linalg.norm(np.diff(Tc[:, :3, 3], axis=0), axis=1) / VST.FRAME_DT
    p90 = float(np.percentile(vc, 90))
    v_ref0 = float(
        vc[: V4.JOIN_REF_FRAMES].mean()
    )  # reference frame-0 wrist speed (m/s) — target speed at the end of the join
    bud_f = HS.finger_budget(V4.COMBO[1])
    limit = o["limit"]
    return {
        "joints": o["joints"],
        "qc": qc,
        "pq": pq,
        "n_route": len(rt["q"]) - 1,
        "lo": o["lo"],
        "hi": o["hi"],
        "bud": o["bud"],
        "p90": p90,
        "v_ref0": v_ref0,
        "f_relax": o["f_relax"],
        "f0": o["f0"],
        "bud_f": bud_f,
        "rule": o["rule"],
        "L": float(np.linalg.norm(np.diff(Pp, axis=0), axis=1).sum()),
        "rot_deg": rot,
        "base_name": base_name,
        "q_base": qb,
        "entry_err_mm": float(err.max() * 1000),
        "backoff_tried": o["tried"],
        "route": {
            "n_cands": o["n_cands"],
            "desk_mm": round(1000 * rt["desk_m"], 1),
            "torso_mm": round(1000 * rt["torso_m"], 1),
            "limit_mm": [round(1000 * x, 1) for x in limit],
            "c1": [round(float(x), 3) for x in rt["c1"]],
        },
        "joint_rot_rad": [round(float(x), 3) for x in jrot],
        "wrist_rot_rad": round(float(jrot[4:].max()), 3),
        "branch_dq_rad": [round(float(x), 3) for x in np.abs(qc[0] - qb)],
    }


def time_sides(ctx: dict, side_data: dict) -> int:
    """Per hand: wrist route, speed target, schedule parameter σ, length K (timing.choose_frames) → common window
    K_all
    (the longer).
    """
    from dexengine.placement.collision.geom import arm_link_transforms
    from dexengine.placement.retime import stretch as VST

    for side, sd in side_data.items():
        sd["pos"] = (arm_link_transforms(ctx[side], sd["pq"])[:, -1] @ ctx[side]["ee_local"])[:, :3, 3]
        sd["v_tgt"] = max(V4.SPEED_RATIO * sd["p90"], V4.V_TARGET_MIN)
        sd["sig"] = TM.path_time(sd["pq"], sd["bud"], sd["pos"], sd["v_tgt"], VST.FRAME_DT)
        sd["e"] = TM.join_speed(sd["sig"], sd["pos"], sd["v_ref0"], VST.FRAME_DT, VST.SAFETY)
    K_all = 0
    for sd in side_data.values():

        def extra(K, sd=sd):
            return V4.finger_schedule(K, sd["f_relax"], sd["f0"]), sd["bud_f"]

        K, tinfo = TM.choose_frames(
            sd["sig"], sd["pq"], sd["pos"], sd["v_tgt"], VST.FRAME_DT, sd["bud"], VST.SAFETY, extra=extra, e=sd["e"]
        )
        sd["K"], sd["tinfo"] = K, tinfo
        K_all = max(K_all, K)
    return K_all


def side_arrays(ctx: dict, side_data: dict, K_all: int, rec: dict, floor: float | None = None) -> dict:
    """Bimanual clips use one window for both hands (the longer) — one approach length per episode. Per hand the
    schedule
    arrays and check values (rec["sides"][hand])."""
    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.collision.geom import arm_link_transforms
    from dexengine.placement.retime import stretch as VST

    arrays = {}
    for side, sd in side_data.items():
        K = K_all
        sd["e"], jf = TM.join_floor(K_all, sd["sig"], sd["pq"], sd["e"], VST.FRAME_DT, VST.SAFETY, floor)
        if jf["on"]:
            print(
                f"{V4.JOIN_FLOOR_TAG} {side} end joint speed {jf['end0']}→{jf['end1']} rad/s (threshold × "
                f"{V4.JOIN_FLOOR} = "
                f"{jf['tgt']}) · end σ speed {jf['e0']}→{jf['e1']}",
                flush=True,
            )
        (qk,) = TM.resample(K_all, sd["sig"], sd["pq"], e=sd["e"])
        _s, prof = TM.speed_profile(K_all, float(sd["sig"][-1]), sd["e"])
        fk = V4.finger_schedule(K_all, sd["f_relax"], sd["f0"])  # fingers = the whole window (as the session)
        r = VST.frame_ratio(qk, sd["bud"]) * VST.SAFETY
        rf = VST.frame_ratio(fk, sd["bud_f"]) * VST.SAFETY
        T = arm_link_transforms(ctx[side], qk)[:, -1] @ ctx[side]["ee_local"]
        v = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) / VST.FRAME_DT
        Tq = arm_link_transforms(ctx[side], sd["qc"][:3])[:, -1] @ ctx[side]["ee_local"]
        jv = np.r_[v, np.linalg.norm(np.diff(Tq[:, :3, 3], axis=0), axis=1) / VST.FRAME_DT]
        margin = float(np.minimum(qk - sd["lo"], sd["hi"] - qk).min())
        rec["sides"][side] = {
            "K": K,
            "rule": sd["rule"],
            "L_cm": round(100 * sd["L"], 1),
            "rot_deg": round(sd["rot_deg"], 1),
            "why": sd["tinfo"]["why"],
            "v_peak": round(float(v.max()), 3),
            "clip_p90": round(sd["p90"], 3),
            "peak_over_p90": round(float(v.max()) / sd["p90"], 2),
            "retime_p99": round(float(np.percentile(np.r_[r, rf], 99)), 3),
            "retime_max": round(float(max(r.max(), rf.max())), 3),
            "joint_margin_rad": round(margin, 3),
            "max_dq_frame": round(float(np.abs(np.diff(qk, axis=0)).max()), 4),
            "max_dq_path": round(float(np.abs(np.diff(sd["pq"], axis=0)).max()), 4),
            "join_slow_frames": int((jv[max(0, K - 15) : K + 3] < 0.01).sum()),
            "base_name": sd["base_name"],
            "join_ref_v": round(sd["v_ref0"], 3),
            "join_v": round(float(v[-1]), 3),
            "join_e": round(prof["e_used"], 3),
            "join_floor": jf,
            "cruise": round(prof["c"], 3),
            "route": sd["route"],
            "joint_rot_rad": sd["joint_rot_rad"],
            "wrist_rot_rad": sd["wrist_rot_rad"],
            "branch_dq_rad": sd["branch_dq_rad"],
            "entry_err_mm": round(sd["entry_err_mm"], 2),
        }
        arrays[f"q_{side}"] = qk
        arrays[f"f_{side}"] = fk
        arrays[f"fnames_{side}"] = np.array(HS.finger_names(V4.COMBO[1], side))
        arrays[f"joints_{side}"] = np.array(sd["joints"])
        arrays[f"wrist_{side}"] = T
    return arrays
