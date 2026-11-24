"""v4 approach schedule (after the bake, CPU): per variant, the joint and finger schedule from the fixed pose to frame 0
and its check values.

Input (the shadow root — canonical bake, the same IK branch as v2): the clip IK cache (after retiming,
arm_ik_poses_<side>_cont25) and the tree (fingers at frame 0, supports). The rule values live in rules.py (backoff
point, entry, joint-space route, min-jerk, fingers, speed target).
(1) active arm approach = fixed pose → joint-space cubic Bézier (rules.active_routes, filtered by desk, hand and torso
    margins) → backoff point (rules.backoff_cands) → straight entry (sequential DLS backwards from the frame-0 v2
    solution — ends exactly at IK[0], v2 branch)
(2) length K = timing.choose_frames (start at wrist peak speed = clip wrist p90 × SPEED_RATIO, stretched until the arm
    and finger speed ratio (stretch.frame_ratio × SAFETY) ≤ 1) (3) fingers relax → frame 0 (rules.finger_schedule)
(4) checks: retiming ratio p99/max, joint limit margins, max per-frame |Δq|, judge guard (collision.pen_cross_guard —
    desk, supports, torso, arm crossing; fingers = schedule values), wrist speed (peak / clip p90).
The idle arm = park (rules.park_q — the value the env's v2 RSI uses), fixed over the whole window: schedule
q_<idle side> = park repeated; the guard (arm crossing included) measures the park arm as a static body; the session
checks it against the RSI reset target (rules.park_check). Scene clearance (scene.scene_check) bodies = supports, main
objects, static bodies (extras.decide rows) and the desk.
Output <out>/<variant>.npz + <out>/plan.json (variant → npz, check values). Marker [AUG_V4_PLAN], last line DONE.
  python -m experiments.real_kit.v4.plan --shadow S --dataset ds --emb emb --names n1,n2 --out D
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from experiments.real_kit.v4 import rules as V4
from experiments.real_kit.v4 import scene as SCN
from experiments.real_kit.v4.plan_sides import side_arrays, side_record, time_sides

MARK = "[AUG_V4_PLAN]"


def die(msg: str):
    raise SystemExit(f"{MARK} {msg}")


def urdf_limits(side: str, joints: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import xml.etree.ElementTree as ET

    from dexengine.placement import arm_ik as AIP

    urdf = AIP.resolve_combo(*V4.COMBO, side)[0]
    lim = {j.get("name"): j.find("limit") for j in ET.parse(str(urdf)).getroot().findall("joint")}
    lo = np.array([float(lim[n].get("lower")) for n in joints])
    hi = np.array([float(lim[n].get("upper")) for n in joints])
    vel = np.array([float(lim[n].get("velocity")) for n in joints])
    return lo, hi, vel


def arm_iks() -> dict:
    """Per side (ArmIK, torch, DT, DEV, joint names) — built like the bake (stage_dp: optimize_place_gpu.ArmIK,
    urdf_load).
    """
    import torch

    from dexengine.placement import arm_ik as AIP
    from dexengine.placement.reach_map import batch_ik as OPG
    from dexengine.robots import urdf_load

    out = {}
    for side in ("right", "left"):
        urdf, ee, joints, _b, _ = AIP.resolve_combo(*V4.COMBO, side)
        out[side] = (
            OPG.ArmIK(side, urdf_load.build_model(str(urdf)), urdf, ee, joints),
            torch,
            OPG.DT,
            OPG.DEV,
            list(joints),
        )
    return out


def _ik_cache(rob, side, name, iks):
    """The clip IK cache of one side (after retiming) and the IK solver; dies when their joints differ."""
    ikz = np.load(rob / f"arm_ik_poses_{side}_cont25/{name}.npz")
    joints = [str(x) for x in ikz["arm_joint_names"]]
    qc = ikz["arm_joint_positions_all"].astype(np.float64)
    ik, torch, DT, DEV, ik_joints = iks[side]
    if ik_joints != joints:
        die(f"{name}/{side}: IK joints {ik_joints} ≠ cache {joints}")
    return joints, qc, DEV, DT, ik, torch


def _clip_frames(ik, torch, qc, DT, DEV, HS, side, fq):
    """Clip palm frames (IK fk), frame 0, relaxed and frame-0 fingers and the finger rows of the route samples."""
    Tc = ik.fk(torch.tensor(qc, dtype=DT, device=DEV)).detach().cpu().numpy().astype(np.float64)
    T0 = Tc[0]
    f_relax = V4.relaxed_fingers(HS.finger_names(V4.COMBO[1], side))
    f0 = fq[side][0]
    fz = [np.repeat(f_relax[None], V4.ACTIVE_SAMPLES, 0), np.repeat(np.asarray(f0, float)[None], V4.ACTIVE_SAMPLES, 0)]
    return Tc, T0, f_relax, f0, fz


def _entry_clearance(q_rev, qc, f_relax, f0, SCL, ctx, side, z_off, dxy):
    """Straight-entry joints (ends exactly at the clip frame 0) and the desk / torso clearance limit of its routes."""
    q_ent = q_rev[::-1].copy()
    q_ent[-1] = qc[0]
    fe = [np.repeat(f_relax[None], len(q_ent), 0), np.repeat(np.asarray(f0, float)[None], len(q_ent), 0)]
    we = SCL.desk_penetration(ctx[side], q_ent, z_off=z_off, desk_xy=dxy, want_components=True)
    ent_desk = max(we[3], SCN.hand_desk_m(ctx[side], side, q_ent, fe, z_off, dxy))
    ent_tor = we[4] if we[4] is not None else -1e9
    limit = (max(V4.ACTIVE_CLEAR[0], ent_desk + 0.002), max(V4.ACTIVE_CLEAR[1], ent_tor + 0.002))
    return q_ent, limit


def _note_why_none(n_ik, why_none, side, tried):
    """Record why one hand has no (further) candidate."""
    if not n_ik:
        why_none[side] = f"{side}: straight entry IK error > {V4.ENTRY_MAX_ERR_M * 1000:.0f} mm — candidates {tried}"
    why_none.setdefault(side, f"{side}: (applied rule × path) candidates exhausted")


def active_opts(side, name, rob, ctx, iks, fq, z_off, dxy, base_name, why_none: dict, entry_bad: dict):
    """Candidates (backoff rule × route) of one active hand in order — for every backoff candidate
    (rules.backoff_cands) whose entry IK is within threshold, the first SCENE_ROUTES_PER_BACKOFF route candidates
    (rules.active_routes, best first). The first candidate equals the plain choice (best route of the first backoff
    point passing IK); later ones are used only when scene clearance blocks. why_none and entry_bad are the caller's
    (one) dicts — entry_bad (scene clearance verdict) is read again after each yield."""
    from dexengine.placement.collision import core as SCL
    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.retime import stretch as VST

    joints, qc, DEV, DT, ik, torch = _ik_cache(rob, side, name, iks)
    lo, hi, vel = urdf_limits(side, joints)
    bud = vel * VST.FRAME_DT
    qb = V4.base_q(side, base_name)
    # clip = the v2 IK branch as is (cache after retiming) · palm frame = IK frame (ik.fk)
    Tc, T0, f_relax, f0, fz = _clip_frames(ik, torch, qc, DT, DEV, HS, side, fq)

    def comp(q, side=side, fz=fz):
        w_ = SCL.desk_penetration(ctx[side], q, z_off=z_off, desk_xy=dxy, want_components=True)
        return max(w_[3], SCN.hand_desk_m(ctx[side], side, q, [f[: len(q)] for f in fz], z_off, dxy)), w_[4]

    def plen(q, ik=ik, torch=torch, DT=DT, DEV=DEV):
        P_ = ik.fk(torch.tensor(q, dtype=DT, device=DEV)).detach().cpu().numpy()[:, :3, 3]
        return float(np.linalg.norm(np.diff(P_, axis=0), axis=1).sum())

    # straight entry w → frame 0: sequential DLS backwards from the frame-0 v2 solution (v2 branch, ends exactly at
    # IK[0])
    tried, n_ik = [], 0
    for w, _e, rule in V4.backoff_cands(T0, Tc[:, :3, 3], VST.FRAME_DT):
        Te = V4.entry_poses(T0, w)
        q_rev, err = V4.seq_ik(ik, Te[::-1].copy(), qc[0], torch, DT, DEV)
        tried.append(f"{rule} {err.max() * 1000:.1f} mm")
        if float(err.max()) > V4.ENTRY_MAX_ERR_M:
            continue
        n_ik += 1
        q_ent, limit = _entry_clearance(q_rev, qc, f_relax, f0, SCL, ctx, side, z_off, dxy)
        cands = V4.active_routes(qb, q_ent[0], q_ent[1] - q_ent[0], comp, bud, lo, hi, limit, plen=plen)
        if not cands:
            why_none[side] = (
                f"{side}: 0 joint-space path candidates for the active arm (desk ≤ {limit[0] * 1000:.0f} mm · torso ≤ "
                f"{limit[1] * 1000:.0f} mm · applied rule {rule})"
            )
            continue
        entry_bad.pop(side, None)
        for rt in cands[: V4.SCENE_ROUTES_PER_BACKOFF]:
            yield {
                "joints": joints,
                "qc": qc,
                "Tc": Tc,
                "T0": T0,
                "lo": lo,
                "hi": hi,
                "bud": bud,
                "qb": qb,
                "f_relax": f_relax,
                "f0": f0,
                "rule": rule,
                "err": err,
                "tried": list(tried),
                "q_ent": q_ent,
                "limit": limit,
                "rt": rt,
                "n_cands": len(cands),
                "ik": (ik, torch, DT, DEV),
            }
            if entry_bad.pop(
                side,
                False,
                # blocked at a place that does not depend on the route — the straight entry (scene_pick) or the backoff
                # point
            ):
                # (scene.bite_blocked); both are set by the backoff point, not the route → next backoff rule
                break
    _note_why_none(n_ik, why_none, side, tried)


def park_guard(ctx, side_data, arrays, park, K_all, rec, z_off, dxy, sup) -> tuple:
    """Judge guard (canonical function — first violation, value per term) · idle arm = park, static (fingers = rest
    shape) → (ok, why, term values).
    """
    from dexengine.placement.collision import core as SCL
    from dexengine.placement.collision import hand_surface as HS

    qg = {s: arrays[f"q_{s}"] for s in side_data}
    fg = {s: arrays[f"f_{s}"] for s in side_data}
    for s, (q_p, src) in park.items():
        arrays[f"q_{s}"] = np.repeat(q_p[None], K_all + 1, 0)
        arrays[f"joints_{s}"] = np.array([f"{s}_arm_{i}" for i in range(7)])
        arrays[f"park_src_{s}"] = np.array(src)
        qg[s] = arrays[f"q_{s}"]
        fg[s] = np.repeat(HS.rest_finger_q(V4.COMBO[1])[None], K_all + 1, 0)
        rec["idle_park"] = {"side": s, "q": [round(float(x), 6) for x in q_p], "source": src}
    with contextlib.redirect_stdout(io.StringIO()):
        ok, why = SCL.pen_cross_guard(ctx, qg, z_off=z_off, desk_xy=dxy, fingers=fg, supports=sup)
        tv = SCL.term_values(ctx, qg, z_off=z_off, desk_xy=dxy, fingers=fg, supports=sup)
    return ok, why, tv


def scene_pick(name, pick, side_data, bad, sc, entry_bad, scene_hist) -> None:
    """Scene clearance result of one candidate → per hand whether the straight entry is blocked (entry_bad, read by
    active_opts), record, marker.
    """
    for s_ in side_data:
        mine = [b for b in bad if b.startswith(f"{s_} ")]
        entry_bad[s_] = any(f" {SCN.ENTRY_WORD} " in b for b in mine)
    scene_hist.append(
        {
            "pick": pick,
            "rules": {s: sd["rule"] for s, sd in side_data.items()},
            "c1": {s: sd["route"]["c1"] for s, sd in side_data.items()},
            "bad": bad[:4],
            "f0": {s: sc[s]["f0"] for s in side_data},
        }
    )
    print(
        f"{V4.SCENE_TAG} {name} candidate {pick + 1} {'pass' if not bad else 'blocked'} · "
        + " · ".join(
            f"{s} rule {sd['rule']} min clearance pre "
            f"{min(sc[s].get('pre', {'-': float('nan')}).values()):.1f} · entry "
            f"{min(sc[s]['ent'].values()):.1f} mm · frame 0 {sc[s]['f0']}"
            for s, sd in side_data.items()
        )
        + (f" · {bad[:3]}" if bad else ""),
        flush=True,
    )


def _idle_park(emb, rob, name, idle):
    """Park pose of the idle arm (rules.park_q; at most one idle arm per episode)."""
    park = {s: V4.park_q(s, emb, rob / f"arm_ik_poses_{s}_cont25/{name}.npz") for s in idle}
    if len(park) > 1:
        die(f"{name}: idle arms {sorted(park)} — an episode has at most one idle arm")
    for s, (q_p, src) in park.items():
        print(f"{V4.PARK_TAG} plan {name} side={s} q={[round(float(x), 4) for x in q_p]} · {src}", flush=True)
    return park


def _next_candidates(active, gens, base_name):
    """The next candidate of every active hand; stop = the first hand that has none left."""
    side_data, stop = {}, None
    for side in active:
        o = next(gens[side], None)
        if o is None:
            stop = side
            break
        side_data[side] = side_record(o, base_name)
    return side_data, stop


def _skip_blocked_routes(side_data, entry_bad, ctx, bad, z_off, dxy, bodies, name, pick):
    """Mark hands blocked at the backoff point (route-independent): their remaining routes of that rule are skipped."""
    for (
        s_,
        sd_,
    ) in (
        side_data.items()
    ):  # a block at a route-independent place (backoff point) → skip that backoff rule's other routes
        blk = {} if entry_bad.get(s_) else SCN.bite_blocked(ctx, s_, sd_, bad, z_off, dxy, bodies)
        if blk:
            entry_bad[s_] = True
            print(
                f"{V4.SCENE_TAG} {name} candidate {pick + 1} {s_} blocked at the backoff point {blk} "
                "mm (best over the finger "
                f"schedule range · threshold {(V4.SCENE_CLEAR_M - V4.SCENE_TOL_M) * 1000:.0f} mm) → "
                f"rule {sd_['rule']}: "
                "remaining routes skipped",
                flush=True,
            )


def _finish(rec, pick, sc, scene_hist, K_all, ok, why, tv, out, floor_off, name, active, idle, side_data, arrays):
    """Record the scene pick, guard and K, and write the schedule npz."""
    rec["scene"] = {
        "ok": True,
        "picks": pick + 1,
        "sides": sc,
        "hist": scene_hist,
        "clear_mm": V4.SCENE_CLEAR_M * 1000,
        "tol_mm": V4.SCENE_TOL_M * 1000,
    }
    rec.update(
        K=K_all,
        guard_ok=bool(ok),
        guard_why=why,
        guard_terms={k: (None if v is None else round(v, 1)) for k, v in tv.items()},
    )
    npz = out / (
        f"{name}.f0.npz" if floor_off else f"{name}.npz"
    )  # a re-plan gets another name (the earlier round's record is kept)
    np.savez(
        npz,
        K=np.int64(K_all),
        active=np.array(active),
        idle=np.array(idle),
        base_name=np.array(next(iter(side_data.values()))["base_name"]),
        **arrays,
    )
    rec["npz"] = str(npz)
    return rec


def one(shadow: Path, ds: str, emb: str, name: str, out: Path, ctx: dict, iks: dict, floor_off: bool = False) -> dict:
    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.collision import support_surface as SSF
    from dexengine.placement.coords import core as VC

    combo = "_".join(V4.COMBO)
    rob = shadow / P.data_layout.render("precompute", ds, combo=combo)
    tree = shadow / P.data_layout.render("trees", ds, combo=combo, emb=emb, tree=name)
    ti = json.loads((tree / "task_info.json").read_text())
    active = list(ti["active_sides"])
    idle = list(ti.get("idle_sides") or [])
    qpos = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    fq = HS.tree_finger_q(tree, V4.COMBO[1], qpos=qpos)
    rec = {"name": name, "active": active, "idle": idle, "sides": {}}
    z_off = float(np.load(rob / f"arm_ik_poses_{active[0]}_cont25/{name}.npz")["base_z_off"])
    dxy = VC.desk_bounds_scene_for(str(tree), margin=0.0)
    sup = SSF.scene_supports(str(tree))
    base_name = V4.base_name()
    bodies = SCN.scene_bodies(tree, sup)
    why_none: dict = {}
    # hand → was the previous candidate blocked on the straight entry (written by scene_pick, read by active_opts)
    entry_bad: dict = {}
    # idle arm = park (rules.park_q — fixed over the window; a static body in the guard)
    park = _idle_park(emb, rob, name, idle)
    gens = {
        side: active_opts(side, name, rob, ctx, iks, fq, z_off, dxy, base_name, why_none, entry_bad) for side in active
    }
    scene_hist = []
    for pick in range(V4.SCENE_MAX_OPTS):
        side_data, stop = _next_candidates(active, gens, base_name)
        if stop is not None:
            rec["fail"] = (
                f"{why_none.get(stop)}"
                if not pick
                else f"{stop}: no approach candidate passed the scene clearance (measured candidates {pick}) — "
                f"{scene_hist[-3:]}"
            )
            rec["scene"] = {"ok": False, "picks": pick, "hist": scene_hist}
            return rec
        K_all = time_sides(ctx, side_data)
        arrays = side_arrays(ctx, side_data, K_all, rec, 0.0 if floor_off else None)
        ok, why, tv = park_guard(ctx, side_data, arrays, park, K_all, rec, z_off, dxy, sup)
        # scene clearance (rules.SCENE_*): chosen approach ↔ desk, objects, supports
        bad, sc = SCN.scene_check(ctx, side_data, arrays, K_all, z_off, dxy, bodies)
        scene_pick(name, pick, side_data, bad, sc, entry_bad, scene_hist)
        if not bad:
            break
        _skip_blocked_routes(side_data, entry_bad, ctx, bad, z_off, dxy, bodies, name, pick)
    else:
        rec["fail"] = (
            f"no approach candidate passed the scene clearance (measured candidates {V4.SCENE_MAX_OPTS} — cap) — "
            f"{[h['bad'][:2] for h in scene_hist[-3:]]}"
        )
        rec["scene"] = {"ok": False, "picks": V4.SCENE_MAX_OPTS, "hist": scene_hist}
        return rec
    return _finish(rec, pick, sc, scene_hist, K_all, ok, why, tv, out, floor_off, name, active, idle, side_data, arrays)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shadow", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--names", required=True, help="comma list (tree names, with _pl_rt)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--join-floor-off",
        action="store_true",
        help="re-plan: join floor off, join at the reference speed (experiments.real_kit.v4.rules.REPLAN_FLOOR_OFF)",
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from dexengine.placement.collision import core as SCL

    with contextlib.redirect_stdout(io.StringIO()):
        ctx = {s: SCL.build_arm_ctx(*V4.COMBO, s) for s in ("right", "left")}
    iks = arm_iks()
    print(
        f"{V4.JOIN_FLOOR_TAG} join floor = {V4.JOIN_FLOOR} (0 = off) · threshold = approach peak × {V4.JOIN_LOW_FRAC}",
        flush=True,
    )
    pj = a.out / "plan.json"
    plan = json.loads(pj.read_text()) if pj.is_file() else {}
    names = [n for n in a.names.split(",") if n]
    for n in names:
        rec = one(a.shadow, a.dataset, a.emb, n, a.out, ctx, iks, a.join_floor_off)
        if a.join_floor_off:
            rec["join_floor_off"] = True
            print(f"{V4.REPLAN_TAG} {n} join floor off — join at the reference speed", flush=True)
        plan[n] = rec
        if "K" not in rec:
            print(f"{MARK} {n} failed {rec.get('fail')}", flush=True)
            continue
        s = next(iter(rec["sides"].values()))
        ip = rec.get("idle_park")
        print(
            f"{MARK} {n} K={rec['K']} ({rec['K'] * 0.02:.2f} s) "
            + (f"idle arm {ip['side']} park {ip['q']} · " if ip else "")
            + f"rule {s['rule']} · L {s['L_cm']} cm · peak/p90 "
            f"{s['peak_over_p90']} · join speed {s['join_v']} m/s (reference {s['join_ref_v']} · "
            f"floor {s['join_floor']['on']}) · "
            f"retiming p99 {s['retime_p99']} · margin {s['joint_margin_rad']} rad · "
            f"max|Δq| {s['max_dq_frame']} · "
            f"guard {'pass' if rec['guard_ok'] else rec['guard_why']}"
            + (f" · failed {rec['fail']}" if rec.get("fail") else ""),
            flush=True,
        )
    pj.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    print(f"{MARK} DONE {len(names)} · {pj}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
