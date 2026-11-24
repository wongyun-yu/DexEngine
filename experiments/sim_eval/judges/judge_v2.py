#!/usr/bin/env python3
"""E5 judge v2: the Sweep contact-travel fix and the v2 rules, wrapped around the frozen v1 judge (judge_v1).

The frozen v1 judge is used unchanged; the rules file selects the behaviour by name.
sweep.travel selects how Sweep measures "travel while in contact":
  nearest (default, key absent) = the frozen v1 judge as is (its function is not touched → byte-identical).
            Per row, the one tool point (of 1500) nearest to the target; the travel of that point between two rows in
            contact is summed. Flaw: when the nearest point jumps to another part of the tool, "travel" grows without
            real motion.
  tracked = between two rows t, t+1 in contact, the mean travel of the *same* tool points (same index = material
            point of the tool; target-body coordinates x = tp @ Rr[t].T + pr[t]) that are within dc of the target in
            both rows; 0 when no point is shared. The reference (kinematic) L_ref uses the same rule, so the threshold
            L = min(L_cap, L_frac · L_ref) is on the same scale.
Unknown values raise. Marker: [E5_SWEEPFIX] travel=<value>.

v2 rules (keys absent = frozen v1):
  lift.v2 {gt: data, rise_frac_gt, liftoff_m, force_floor_N, force_k_gt} — Lift ground truth = our data episode
  (physics). Added on top of a frozen-v1 Lift success only (an old failure stays a failure):
    rise   rise_max ≥ rise_frac_gt × data rise max
    height during the success window [first lift (rise ≥ liftoff_m), grasp end + common.post_success_s] the rise
           stays ≥ min(0, data rise min) − common.off_desk_below_m (data min < 0 = picked high and put down lower;
           no time alignment)
    force  median hand force over the grasp window ≤ max(force_floor_N, force_k_gt × median data hand force while the
           data object is lifted (data rise ≥ h))
    Not caught: tipping over (no direction signal), dropping after the success window.
  sweep.hold_tool {both|right} — tracked travel counts only between two rows where the hand holds the tool
  (hand-link force > common.grasp_N; both = sum of both hands, right = right hand only).
  sweep.ref: data — the reference length L_ref is measured on the data-episode trajectory with the same tracked rule.

Usage:
  python -m experiments.sim_eval.judges.judge_v2 --run <session out> --mode <policy-…> --rules <yaml>
      --data-root <root> --home-map <build HOME>=<local copy> --out <json>
Library: load_frozen() → the v1 module · judge_run(M, run, mode, rules, home_map) / judge_run_v2(...) → verdict dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

MARK = "[E5_SWEEPFIX]"
TRAVEL = ("nearest", "tracked")


def load_frozen():
    """The frozen v1 judge module; remembers its original contact_travel once (use_travel restores it)."""
    from experiments.sim_eval.judges import judge_v1 as M  # noqa: PLC0415

    M._SWEEPFIX_ORIG_CT = getattr(M, "_SWEEPFIX_ORIG_CT", M.contact_travel)
    return M


def point_dists(M, sc, tool: str, tgt: str, ds: str) -> tuple[np.ndarray, np.ndarray]:
    """(T, N) distance of each tool point to the target point cloud · (T, N, 3) tool points in the target body frame —
    the same points and the same formula as the frozen contact_points."""
    from scipy.spatial import cKDTree

    tp, gp = M.mesh_pts(ds, sc.o[tool]["meta"], 1500), M.mesh_pts(ds, sc.o[tgt]["meta"], 1500)
    kd = cKDTree(gp)
    pr, Rr = M.rel_pose(sc, tool, tgt)
    X = np.einsum("nj,tij->tni", tp, Rr) + pr[:, None]  # per row tp @ Rr[t].T + pr[t]
    D = kd.query(X.reshape(-1, 3))[0].reshape(sc.T, len(tp))
    return D, X


def tracked_steps(D: np.ndarray, X: np.ndarray, dc: float) -> tuple[np.ndarray, np.ndarray]:
    """Steps between rows (T−1,) and per-row min distance (T,). Step = mean travel of the same tool points that are
    within dc in both rows (0 if none)."""
    dmin = D.min(axis=1)
    c = dmin <= dc
    step = np.zeros(max(len(D) - 1, 0))
    for t in np.flatnonzero(c[1:] & c[:-1]):
        m = (D[t] <= dc) & (D[t + 1] <= dc)
        if m.any():
            step[t] = float(np.linalg.norm(X[t + 1, m] - X[t, m], axis=1).mean())
    return step, dmin


def make_tracked_ct(M):
    """A contact_travel with the frozen signature and return value ((sum, per-row min distance, per-row cumulative));
    only the travel formula is tracked."""

    def contact_travel_tracked(sc, tool, tgt, ds, dc, pts=None):
        D, X = point_dists(M, sc, tool, tgt, ds)
        step, dmin = tracked_steps(D, X, dc)
        cum = np.concatenate([[0.0], np.cumsum(step)])
        return float(cum[-1]), dmin, cum

    return contact_travel_tracked


def travel_of(rules: dict) -> str:
    """The rules' sweep.travel (default nearest); unknown values raise."""
    v = (rules.get("sweep") or {}).get("travel", "nearest")
    if v not in TRAVEL:
        raise SystemExit(f"{MARK} sweep.travel={v!r} — only {TRAVEL}")
    return v


def use_travel(M, travel: str) -> None:
    """Select the module-global contact_travel; the frozen sweep() looks the name up at call time.
    nearest = the original function."""
    M.contact_travel = M._SWEEPFIX_ORIG_CT if travel == "nearest" else make_tracked_ct(M)


def judge_run(M, run: Path, mode: str, rules: dict, home_map: str) -> dict:
    """Frozen v1 judge_run with the rules' Sweep travel."""
    tr = travel_of(rules)
    use_travel(M, tr)
    try:
        res = M.judge_run(Path(run), mode, rules, home_map)
    finally:
        use_travel(M, "nearest")
    res["sweep_travel"] = tr
    return res


def main() -> None:
    """CLI: judge one run with v1 or v2 rules (chosen by the rules file) and write --out (outside the run folder)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--home-map", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = load_frozen()
    M.set_data_root(a.data_root)
    with open(a.rules) as fh:
        rules = yaml.safe_load(fh)
    print(f"{MARK} travel={travel_of(rules)} · v2 {v2_of(rules)} · frozen {Path(M.__file__).resolve()}", flush=True)
    lv, ht = v2_of(rules)
    v2 = (
        lv is not None or ht is not None or (rules.get("sweep") or {}).get("ref") is not None
    )  # any v2 key selects the v2 path
    res = (
        judge_run_v2(M, Path(a.run), a.mode, rules, a.home_map)
        if v2
        else judge_run(M, Path(a.run), a.mode, rules, a.home_map)
    )
    out = Path(a.out)
    if out.resolve().is_relative_to(Path(a.run).resolve()):
        raise SystemExit(f"{MARK} --out {out} is inside the run folder — records are not overwritten")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._dump))
    print(
        f"{MARK} {res['clip']} {a.mode} {res['axis']} travel={res['sweep_travel']} · judgeable "
        f"{res['judgeable']} · invalid "
        f"{res['invalid']} · success {res['success']} → {out}",
        flush=True,
    )


HOLD = ("both", "right")


def v2_of(rules: dict) -> tuple[dict | None, str | None]:
    """(lift.v2 or None, sweep.hold_tool or None), validated."""
    lv = (rules.get("lift") or {}).get("v2")
    if lv is not None:
        need = {"gt", "rise_frac_gt", "liftoff_m", "force_floor_N", "force_k_gt"}
        if set(lv) != need:
            raise SystemExit(f"{MARK} lift.v2 keys {sorted(lv)} != {sorted(need)}")
        if lv["gt"] != "data":
            raise SystemExit(f"{MARK} lift.v2.gt={lv['gt']!r} — only 'data' (our data episode physics)")
        if "off_desk_below_m" not in rules["common"]:
            raise SystemExit(f"{MARK} lift.v2 needs common.off_desk_below_m")
    ht = (rules.get("sweep") or {}).get("hold_tool")
    if ht is not None:
        if ht not in HOLD:
            raise SystemExit(f"{MARK} sweep.hold_tool={ht!r} — only {HOLD}")
        if travel_of(rules) != "tracked":
            raise SystemExit(f"{MARK} sweep.hold_tool needs sweep.travel: tracked")
    sr = (rules.get("sweep") or {}).get("ref")
    if sr is not None and (sr != "data" or travel_of(rules) != "tracked"):
        raise SystemExit(f"{MARK} sweep.ref={sr!r} — only 'data', with sweep.travel: tracked")
    return lv, ht


def load_ctx(M, run: Path, mode: str, rules: dict, home_map: str) -> dict:
    """The read steps of the frozen judge_run (same functions) plus what v2 needs: per-object hand-link forces
    (both hands, right hand)."""
    run = Path(run)
    job = json.loads((run / "job_run.json").read_text())
    tc = M.tree_clip_of(job, home_map)
    ti, objs = M.task_objects(tc)
    kin, hz = M.kin_scene(tc, objs)
    if abs(hz - M.HZ) > 1e-6:
        raise SystemExit(f"{MARK} kinematic rate {hz} != {M.HZ}")
    envo, chk = M.env_ref(run, objs, kin)
    ref_env = M.Scene(envo, None, "reference_env")
    emap = M.entity_map(tc, ti)
    sc, info = M.roll_scene(run / mode / "episode.npz", objs, emap, kin, envo, rules["common"])
    release, rel_ev = M.release_of(job, rules, job["axis"], emap, ref_env)
    z = np.load(run / mode / "episode.npz")
    meta = json.loads(str(z["meta"]))
    held, links = meta.get("held", []), meta.get("link_names", [])
    right = np.array([str(x).startswith("right") for x in links], dtype=bool)
    hand = {}
    for n, v in sc.o.items():
        e = v.get("entity")
        if e in held and len(links):
            L = z["hand_obj_force"][:, held.index(e), :]
            hand[n] = {"both": L.sum(axis=1), "right": L[:, right].sum(axis=1)}
    return {
        "run": run,
        "mode": mode,
        "job": job,
        "tc": tc,
        "kin": kin,
        "ref_env": ref_env,
        "chk": chk,
        "sc": sc,
        "info": info,
        "release": release,
        "rel_ev": rel_ev,
        "hand": hand,
        "gt_force": None,
    }


def _gt_data(M, ctx: dict) -> dict:
    """Ground truth = our data episode (physics): object poses (job_run.json episode_arrays
    'obj::observation.object_pose.<entity>', judge-rate rows) and hand–object forces (frozen data_object_force, force
    rate hz). Missing data or a rate mismatch raises."""
    if ctx["gt_force"] is None:
        pose = _gt_pose(M, ctx)
        ser, fhz, _f = M.data_object_force(ctx["job"].get("episode") or {})
        ctx["gt_force"] = {"pose": pose, "force": ser, "fhz": float(fhz)}
    return ctx["gt_force"]


def _gt_pose(M, ctx: dict) -> dict:
    """Data-episode object poses {entity: (T, 7) position + quaternion} from job_run.json episode_arrays (judge-rate
    rows). Missing arrays raise."""
    if ctx.get("_gt_pose") is None:
        job = ctx["job"]
        ep = job.get("episode") or {}
        fps = float(ep.get("fps", 0))
        if abs(fps - M.HZ) > 1e-6:
            raise SystemExit(f"{MARK} v2 data ground truth — data episode rate {fps} != judge {M.HZ}")
        a = job.get("episode_arrays")
        if not a or not Path(a).is_file():
            raise SystemExit(f"{MARK} v2 data ground truth — data episode arrays missing {a}")
        z = np.load(a)
        ctx["_gt_pose"] = {
            k.split(".")[-1]: z[k].astype(np.float64) for k in z.files if k.startswith("obj::observation.object_pose.")
        }
    return ctx["_gt_pose"]


def _data_scene(M, ctx: dict):
    """Sweep ground-truth scene = the data episode (physics): rollout objects whose entity has a data pose take the data
    trajectory. Frame: a row-0 translation (rollout row 0 − data row 0; equal within 1 cm across entities); axes must
    agree (row-0 rotation gap ≤ 10°). Objects without data poses (static, reference pose) are cut or padded to the
    data
    length. Forces = the data hand–tool forces."""
    if ctx.get("_dsc") is not None:
        return ctx["_dsc"]
    sc = ctx["sc"]
    G = _gt_pose(M, ctx)
    ents = {n: v.get("entity") for n, v in sc.o.items() if v.get("entity") in G}
    if not ents:
        raise SystemExit(f"{MARK} sweep.ref=data — no data pose entity {sorted(G)} matches a scene object")
    Td = min(len(G[e]) for e in ents.values())
    offs, o = [], {}
    for n, e in ents.items():
        q = G[e]
        R = M.quat_R(q[:, 3:7])
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R[0].T @ sc.o[n]["R"][0]) - 1) / 2, -1, 1))))
        if ang > 10.0:
            raise SystemExit(
                f"{MARK} sweep.ref=data — {n}({e}) row-0 rotation gap {ang:.1f}° > 10° (quaternion order / "
                "frame mismatch)"
            )
        offs.append(sc.o[n]["p"][0] - q[0, :3])
    offs = np.array(offs)
    if np.ptp(offs, axis=0).max() > 0.01:
        raise SystemExit(f"{MARK} sweep.ref=data — row-0 translation differs across entities {offs.round(4).tolist()}")
    off = offs.mean(axis=0)

    def fit(a):
        return a[:Td] if len(a) >= Td else np.concatenate([a, np.repeat(a[-1:], Td - len(a), 0)])

    for n, v in sc.o.items():
        if n in ents:
            q = G[ents[n]][:Td]
            o[n] = dict(v, p=q[:, :3] + off, R=M.quat_R(q[:, 3:7]))
        else:
            o[n] = dict(v, p=fit(v["p"]), R=fit(v["R"]))
    # Hand–object force = frozen data_object_force (per entity, sum of link |F|; force_object.<entity>.force_n has 40
    # links
    # = right 20 + left 20 → both hands, the same quantity as the rollout hand['both']). Force → pose rows: row t ↦
    # force sample round(t / HZ × fhz) (as Lift v2). A missing force file or tool entity, or a length (s) more than
    # 0.1 s off the poses, raises.
    ser, fhz, ff = M.data_object_force(ctx["job"].get("episode") or {})
    force = {}
    for n, e in ents.items():
        if e not in ser:
            continue
        fs = np.asarray(ser[e], np.float64)
        if abs(len(fs) / fhz - Td / M.HZ) > 0.1:
            raise SystemExit(
                f"{MARK} sweep.ref=data — data force {e} {len(fs)} samples/{fhz} Hz and poses {Td} rows/{M.HZ} Hz "
                "differ by more than 0.1 s"
            )
        force[n] = fs[np.minimum((np.arange(Td) / M.HZ * fhz).round().astype(int), len(fs) - 1)]
    ctx["_dsc"] = M.Scene(o, force, "data_episode")
    ctx["_dsc_info"] = {
        "rows": int(Td),
        "offset_m": off.round(4).tolist(),
        "entities": ents,
        "force_file": str(ff),
        "force_hz": float(fhz),
        "force_entities": sorted(force),
        "force_rule": "row t -> round(t/HZ*fhz) · |F| link sum (both hands)",
    }
    return ctx["_dsc"]


def _install_lift_v2(lv, orig_lift, sc_roll, M, ctx):
    """Swap the frozen judge's lift for the v2 lift rule while this run is judged."""
    if lv is not None:

        def lift_v2(ref, s, P, C_, hand_use, assume):
            # Call the frozen Lift; only when the rollout (sc_roll) succeeded, add the three data-ground-truth
            # conditions (an old failure stays a failure). The reference / control scenes get no addition — the data
            # ground truth only means something for the rollout.
            r = orig_lift(ref, s, P, C_, hand_use, assume)
            if s is not sc_roll or not r.get("ok"):
                return r
            G = _gt_data(M, ctx)
            row = int(r["row"])
            out, ok = {}, True
            for n, d in r["objs"].items():
                ent = s.o[n].get("entity")
                if ent not in G["pose"] or ent not in G["force"]:
                    raise SystemExit(
                        f"{MARK} lift.v2 — the data episode has no pose/force for {n}({ent}) · poses "
                        f"{sorted(G['pose'])} · "
                        f"forces {sorted(G['force'])}"
                    )
                rise = s.o[n]["p"][:, 2] - s.o[n]["p"][0, 2]
                T = len(rise)
                dz = G["pose"][ent][:, 2]
                drise = dz - dz[0]
                # (1) rise ≥ rise_frac_gt × data rise max
                c_rise = float(rise.max()) >= lv["rise_frac_gt"] * float(drise.max())
                # (2) during the success window [first lift (rise ≥ liftoff_m), grasp end + post_success_s] the rise
                # stays
                #     ≥ min(0, data rise min) − off_desk_below_m (data min < 0 = picked high, put down lower; no time
                #     alignment — the policy is slower than the data)
                lo = int(np.argmax(rise >= lv["liftoff_m"])) if (rise >= lv["liftoff_m"]).any() else 0
                hi = min(T - 1, row + int(round(C_["post_success_s"] * M.HZ)))
                zfloor = min(0.0, float(drise.min())) - C_["off_desk_below_m"]
                zmin = float(rise[lo : hi + 1].min())
                c_z = zmin >= zfloor
                # (3) median hand force over the grasp window (hold_s ending at the success row) ≤ max(force_floor_N,
                #     force_k_gt × median data hand force while the data object is lifted (data rise ≥ h))
                f = (s.force or {}).get(n)
                if f is None:
                    raise SystemExit(f"{MARK} lift.v2 — no rollout hand force for {n}")
                Hn = max(1, int(round(d["hold_s"] * M.HZ)))
                hold = float(np.median(f[max(0, row - Hn + 1) : row + 1]))
                fs = G["force"][ent]
                di = np.flatnonzero(drise >= d["h_m"])
                fi = np.minimum((di / M.HZ * G["fhz"]).round().astype(int), len(fs) - 1)
                gmed = float(np.median(fs[fi])) if len(fi) else 0.0
                ceil = max(lv["force_floor_N"], lv["force_k_gt"] * gmed)
                c_f = hold <= ceil
                out[n] = {
                    "rise_max_m": round(float(rise.max()), 4),
                    "gt_rise_max_m": round(float(drise.max()), 4),
                    "rise_ok": c_rise,
                    "window": [lo, hi],
                    "z_min_win_m": round(zmin, 4),
                    "z_floor_m": round(zfloor, 4),
                    "gt_z_min_rel_m": round(float(drise.min()), 4),
                    "z_ok": c_z,
                    "hold_force_med_N": round(hold, 2),
                    "gt_hold_force_med_N": round(gmed, 2),
                    "force_ceiling_N": round(ceil, 2),
                    "force_ok": c_f,
                }
                ok = ok and c_rise and c_z and c_f
            return dict(r, ok=ok, row=row if ok else None, v2=out)

        M.lift = lift_v2


def _install_travel(tr, rules, ctx, M, cache, ht, sc_roll, C):
    """Swap contact_travel for the tracked travel rule while this run is judged."""
    if tr == "tracked":
        sref = (rules.get("sweep") or {}).get("ref")

        def ct(sc, tool, tgt, ds, dc, pts=None):
            if sref == "data" and sc is ctx["kin"]:  # reference length L_ref from the data episode (same tracked rule)
                sc = _data_scene(M, ctx)
            k = (sc.label, tool, tgt)
            if k not in cache:
                cache[k] = point_dists(M, sc, tool, tgt, ds)
            D, X = cache[k]
            step, dmin = tracked_steps(D, X, dc)
            if (
                ht is not None and sc.force is not None
            ):  # the kinematic reference has no forces — grasp assumed (ref_contact assume)
                if sc is sc_roll:
                    h = ctx["hand"].get(tool)
                    if h is None:
                        raise SystemExit(f"{MARK} sweep.hold_tool — no hand force record for tool {tool} (meta.held)")
                    g = h[ht] > C["grasp_N"]
                # data ground-truth scene = data hand–tool force (both hands), same grasp_N
                elif sc is ctx.get("_dsc"):
                    if ht != "both":
                        raise SystemExit(
                            f"{MARK} sweep.ref=data · hold_tool={ht!r} — data forces exist for both hands only (both)"
                        )
                    if tool not in sc.force:
                        raise SystemExit(
                            f"{MARK} sweep.ref=data — data forces have no tool {tool} · have {sorted(sc.force)}"
                        )
                    g = sc.force[tool] > C["grasp_N"]
                else:  # frame-0 static scene = zero force
                    g = np.asarray(sc.force.get(tool, np.zeros(sc.T))) > C["grasp_N"]
                step = step * (g[1:] & g[:-1])
            cum = np.concatenate([[0.0], np.cumsum(step)])
            return float(cum[-1]), dmin, cum

        M.contact_travel = ct


def _add_run_fields(res, ctx, job, tr, lv, ht, rules):
    """The verdict plus the run, clip and rule fields of the judge record."""
    res.update(
        run=str(ctx["run"]),
        mode=ctx["mode"],
        clip=job["base"],
        dataset=job["dataset"],
        hand_use=job["hand_use"],
        variant=job.get("variant"),
        tree_clip=str(ctx["tc"]),
        release=ctx["rel_ev"],
        ref_oracle=ctx["chk"],
        missing_objects=ctx["info"]["missing"],
        objects_from_reference=ctx["info"]["from_reference"],
        frame_fit_m=ctx["info"]["frame_fit_m"],
        rows=ctx["sc"].T,
        cap_rows=ctx["info"]["meta"].get("cap_rows"),
        data_rows=ctx["info"]["meta"].get("data_rows"),
        sweep_travel=tr,
        lift_v2=lv,
        sweep_hold_tool=ht,
        sweep_ref=(rules.get("sweep") or {}).get("ref", "kinematic"),
        sweep_data_scene=ctx.get("_dsc_info"),
        rules_status=rules.get("status"),
    )
    return res


def judge_ctx(M, ctx: dict, rules: dict) -> dict:
    """Judge one run: v2 when the rules select it, else the frozen v1 (Sweep follows sweep.travel)."""
    lv, ht = v2_of(rules)
    C = rules["common"]
    orig_lift = M.lift
    sc_roll = ctx["sc"]
    _install_lift_v2(lv, orig_lift, sc_roll, M, ctx)
    tr = travel_of(rules)
    # several rule sets on the same run share the distance computation (scene label, tool, target)
    cache = ctx.setdefault("_pd", {})
    _install_travel(tr, rules, ctx, M, cache, ht, sc_roll, C)
    try:
        job = ctx["job"]
        res = M.judge(
            job["axis"],
            job["hand_use"],
            rules,
            ctx["ref_env"],
            ctx["kin"],
            ctx["sc"],
            job["dataset"],
            job["base"],
            ctx["release"],
        )
    finally:
        M.lift = orig_lift
        use_travel(M, "nearest")
    return _add_run_fields(res, ctx, job, tr, lv, ht, rules)


def judge_run_v2(M, run: Path, mode: str, rules: dict, home_map: str) -> dict:
    """Read and judge one run with the v2 rules."""
    return judge_ctx(M, load_ctx(M, run, mode, rules, home_map), rules)


if __name__ == "__main__":
    main()
