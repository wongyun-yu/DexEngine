#!/usr/bin/env python3
"""E5 offline success judge (v1, frozen): judges one closed-loop episode (closed_loop output) with a rules file.

No threshold lives in the code: every rule constant comes from the rules file (rules_v1.yaml / rules_v2.yaml), whose
`status` is copied into the result.
Rules: R1 threshold = min(literature value, 0.5 × the clip's reference value) · R2 judgeability gate (dropped scenes
are
counted under the table) · R3 achieved once with a short hold · Pour = reaching the reference pouring pose (variant
A) ·
Sweep = travel while in contact · Put release = auto.

One verdict = (judgeable?, success?, invalid?):
  judgeable (R2) = the reference (kinematic) passes the rule and the scene frozen at frame 0 fails it; otherwise the
                   reasons go to judgeable_why (counted under the table).
  invalid        = physics blow-up (object speed > blowup_mps) or a start jump (≥ jump_m within the first jump_s);
                   excluded from the denominator (re-run).
  success (R3)   = the task rule holds once (including its hold time).
Frames: the reference main objects (right, left) come from reference.npz (env-local frame written by the session — the
rollout's frame). Other objects and relative-pose rules use the kinematic npz (tree frame): the relative quantities
(rise, displacement, pose relative to the target's body, surface distance) are invariant to frame shifts and yaw.
The reference has no forces: grasp / release are assumed true on it (rules.common.ref_contact = assume, recorded).
Put release 'auto' is required only when the data episode released at its end, measured from the data force record
(judge_io.data_object_force; threshold release_N and window hold_s come from the rules file). A missing file raises.
Inputs: <run>/job_run.json (prep job + run fields: task, hand use, dataset, clip, seq_dir) · <run>/reference.npz ·
<run>/<mode>/episode.npz. Tree = job.seq_dir/<clip> mapped with --home-map; meshes from --data-root (both required).

The reason strings written into the record (why, judgeable_why, invalid) are stored data read by the counting and
report tools; they keep their original wording.

Usage:
  python -m experiments.sim_eval.judges.judge_v1 --run <session out> --mode replay|zero|policy-… --rules <yaml>
      --data-root <root> --home-map <build HOME>=<local copy> [--out <json>]
  python -m experiments.sim_eval.judges.judge_v1 summarize --runs <run list file | folder> --mode <mode> --out <json>
Marker: [E5_JUDGE] …
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from experiments.sim_eval.judges.geometry import (  # noqa: F401
    HZ,
    first_hold,
    longest,
    quat_R,
    rot_deg,
    speeds,
    tilt_deg,
)
from experiments.sim_eval.judges.judge_io import (  # noqa: F401 — re-exported: judge_v2 / pnp / scenes read them as judge_v1.<name>
    MARK,
    Scene,
    _paths,
    data_object_force,
    entity_map,
    env_from_tree,
    env_ref,
    kin_scene,
    map_home,
    mesh_pts,
    roll_scene,
    set_data_root,
    task_objects,
    tree_clip_of,
)
from experiments.sim_eval.judges.judge_v1_pair import (
    TARGET_RULE,
    contact_travel,
    label_of,
    pour,
    tool_target,
)
from experiments.sim_eval.judges.judge_v1_pair import (  # re-exported: judge_v2 / clip_motion read judge_v1.rel_pose
    rel_pose as rel_pose,
)


# ── task rules
# ────────────────────────────────────────────────────────────────
def lift(ref: Scene, sc: Scene, P: dict, C: dict, hand_use: str, assume: bool) -> dict:
    """Lift: every (or any, per rules) lifted main object rises ≥ h = min(cap, h_frac × reference rise) for the hold
    time.
    """
    cap = P["h_cap_m"]["two_hand_shared"] if hand_use == "bi-shared" else P["h_cap_m"]["one_hand"]
    out = {}
    for n, r in ref.o.items():
        if r["meta"].get("role") != "main":
            continue
        rise = r["p"][:, 2] - r["p"][0, 2]
        if rise.max() < P["min_ref_rise_m"] or n not in sc.o:
            continue
        h = min(cap, P["h_frac"] * float(rise.max()))
        H = max(P["hold_min_s"], min(P["hold_cap_s"], P["hold_frac"] * longest(rise >= h)))
        s = sc.o[n]
        m = (s["p"][:, 2] - s["p"][0, 2]) >= h
        if P["require_grasp"] and not (assume and sc.force is None):
            f = (sc.force or {}).get(n)
            if f is None:
                raise SystemExit(
                    f"{MARK} Lift grasp condition — no hand force row for main object {n} "
                    "(its entity is not in meta.held)"
                )
            m = m & (f > C["grasp_N"])
        out[n] = {"h_m": round(h, 4), "hold_s": round(H, 3), "row": first_hold(m, H)}
    if not out:
        return {"ok": False, "why": "no object lifted by the reference", "objs": {}}
    rows = [v["row"] for v in out.values()]
    ok = all(r is not None for r in rows) if P["multi"] == "all" else any(r is not None for r in rows)
    return {"ok": ok, "row": (max(r for r in rows if r is not None) if ok else None), "objs": out}


def put(ref: Scene, kin: Scene, sc: Scene, P: dict, C: dict, release: bool, assume: bool) -> dict:
    """Put: every moved main object rests within eps of the reference end place (world, or the receiver's body frame
    when the receiver is a main object), static, for hold_s (and released when required)."""
    out = {}
    for n, r in ref.o.items():
        if r["meta"].get("role") != "main" or n not in sc.o:
            continue
        disp = float(np.linalg.norm(r["p"][-1] - r["p"][0]))
        if disp < P["min_ref_disp_m"]:
            continue
        # receiver = the non-background object nearest to the moved object at the reference end (kinematic frame)
        kp = kin.o[n]["p"][-1]
        oth = [
            (float(np.linalg.norm(v["p"][-1] - kp)), m)
            for m, v in kin.o.items()
            if m != n and v["meta"].get("tag") != "background"
        ]
        recv = min(oth)[1] if oth else None
        body = recv is not None and kin.o[recv]["meta"].get("role") == "main" and recv in sc.o and recv in ref.o
        eps = min(P["eps_cap_m"], P["eps_frac"] * disp)
        s = sc.o[n]
        if body:
            g = ref.o[recv]
            tgt = g["R"][-1].T @ (r["p"][-1] - g["p"][-1])
            cur = np.einsum("tji,tj->ti", sc.o[recv]["R"], s["p"] - sc.o[recv]["p"])
        else:
            tgt, cur = r["p"][-1], s["p"]
        lin, ang = speeds(s["p"], s["R"])
        m = (np.linalg.norm(cur - tgt, axis=1) <= eps) & (lin <= P["static_lin_mps"]) & (ang <= P["static_ang_rps"])
        if release and not (assume and sc.force is None):
            f = (sc.force or {}).get(n)
            if f is None:
                raise SystemExit(
                    f"{MARK} Put release condition — no hand force row for main object {n} "
                    "(its entity is not in meta.held)"
                )
            m = m & (f < C["release_N"])
        out[n] = {
            "disp_ref_m": round(disp, 4),
            "eps_m": round(eps, 4),
            "receiver": recv,
            "frame": ("body" if body else "world"),
            "row": first_hold(m, P["hold_s"]),
            "min_dist_m": round(float(np.linalg.norm(cur - tgt, axis=1).min()), 4),
        }
    if not out:
        return {"ok": False, "why": "no object moved by the reference", "objs": {}}
    rows = [v["row"] for v in out.values()]
    ok = all(r is not None for r in rows)
    return {"ok": ok, "row": (max(rows) if ok else None), "objs": out}


def sweep(kin: Scene, sc: Scene, P: dict, C: dict, ds: str, pair: tuple[str, str]) -> dict:
    """Sweep: contact travel ≥ L = min(L_cap, L_frac × reference travel), with a reference travel ≥ min_ref_L_m."""
    tool, tgt = pair
    if tool not in sc.o or tgt not in sc.o:
        return {"ok": False, "why": f"tool/target {tool}/{tgt} not in the scene"}
    Lref, _, _ = contact_travel(kin, tool, tgt, ds, C["d_contact_m"])
    L = min(P["L_cap_m"], P["L_frac"] * Lref)
    Ls, dmin, cum = contact_travel(sc, tool, tgt, ds, C["d_contact_m"])
    ok = bool(Ls >= L and Lref >= P["min_ref_L_m"])
    # success row = first row where the cumulative contact travel reaches L (monotone, so it stays reached)
    row = int(np.argmax(cum >= L)) if ok else None
    return {
        "ok": ok,
        "row": row,
        "L_m": round(L, 4),
        "L_ref_m": round(Lref, 4),
        "L_m_got": round(Ls, 4),
        "tool": tool,
        "target": tgt,
        "ref_too_small": Lref < P["min_ref_L_m"],
        "dmin_min_m": round(float(dmin.min()), 4),
    }


def invalid(sc: Scene, C: dict) -> str | None:
    """Reason the rollout is invalid (physics blow-up or start jump), or None."""
    for n, v in sc.o.items():
        if v.get("entity") is None:  # objects posed from the reference (outside physics) cannot blow up or jump
            continue
        lin, _ = speeds(v["p"], v["R"])
        if lin.max() > C["blowup_mps"]:
            return f"blow-up {n} {lin.max():.2f} m/s"
        k = max(1, int(round(C["jump_s"] * HZ)))
        if np.linalg.norm(v["p"][: k + 1] - v["p"][0], axis=1).max() >= C["jump_m"]:
            return f"start jump {n} (≥ {C['jump_m']} m within the first {C['jump_s']} s)"
    return None


def judge(
    axis: str, hand_use: str, rules: dict, ref_env: Scene, kin: Scene, sc: Scene, ds: str, clip: str, release: bool
) -> dict:
    """One verdict: the task rule on the reference, on the frame-0 scene and on the rollout, plus the invalid check."""
    C = rules["common"]
    assume = C.get("ref_contact", "assume") == "assume"

    def one(s: Scene) -> dict:
        if axis == "Lift":
            return lift(
                kin, s, rules["lift"], C, hand_use, assume
            )  # threshold = kinematic relative rise (frame invariant)
        if axis == "Put":
            return put(ref_env, kin, s, rules["put"], C, release, assume)
        pair = tool_target(kin)
        if pair is None:
            return {"ok": False, "why": "no tool-target pair"}
        if axis == "Pour":
            return pour(kin, s, rules["pour"], label_of(clip, ds), ds, pair, False)
        if axis == "Sweep":
            return sweep(kin, s, rules["sweep"], C, ds, pair)
        raise SystemExit(f"{MARK} unknown task {axis}")

    # Put is a "stay at the reference end place" rule: the reference rests at its end like the session clock does
    # (padded by hold_s + 1 rows)
    ref_scene = (
        kin if axis in ("Pour", "Sweep", "Lift") else ref_env.padded(int(round(rules["put"]["hold_s"] * HZ)) + 1)
    )
    r_ref = one(ref_scene)
    r_f0 = one(sc.static())
    r = one(sc)
    why = []
    if C["judgeable_requires_ref_pass"] and not r_ref["ok"]:
        why.append("reference failed: " + str(r_ref.get("why") or _ref_fail_why(axis, r_ref)))
    if C["judgeable_requires_frame0_fail"] and r_f0["ok"]:
        why.append("frame 0 passes")
    judgeable = not why
    inv = invalid(sc, C)
    return {
        "axis": axis,
        "target_rule": TARGET_RULE,
        "judgeable": judgeable,
        "judgeable_why": why,
        "invalid": inv,
        "success": bool(r["ok"]) and judgeable and inv is None,
        "success_row": r.get("row"),
        "success_s": (None if r.get("row") is None else round(r["row"] / HZ, 2)),
        "detail": r,
        "reference": r_ref,
        "frame0": r_f0,
        "rules_status": rules.get("status"),
        "ref_contact": C.get("ref_contact", "assume"),
        "release_required": release,
    }


def _ref_fail_why(axis: str, r: dict) -> str:
    """One-phrase reason the reference failed its rule (the key of the under-table count), from the measurements."""
    if axis == "Sweep" and r.get("ref_too_small"):
        return "reference contact travel < min_ref_L_m"
    if axis == "Pour" and r.get("target_tipped"):
        return "the reference knocks the target over"
    if axis in ("Lift", "Put"):
        rows = {n: v.get("row") for n, v in (r.get("objs") or {}).items()}
        return f"hold not met {sorted(n for n, x in rows.items() if x is None)}"
    return "reference pose hold not met"


def release_of(job: dict, rules: dict, axis: str, emap: dict, ref_env: Scene) -> tuple[bool, dict]:
    """Put release condition (require_release) → (required?, evidence). auto = required only when the data episode
    released at its end: for every main object the reference moved, the max data force (data_object_force — sum of |F|
    over links) over the last hold_s window is below release_N."""
    rr = rules["put"]["require_release"]
    if axis != "Put":
        return False, {"rule": "outside Put"}
    if rr in ("on", "off"):
        return rr == "on", {"rule": rr}
    if rr != "auto":
        raise SystemExit(f"{MARK} put.require_release={rr!r} — only auto|on|off")
    series, hz, ff = data_object_force(job.get("episode") or {})
    k = max(1, int(round(rules["put"]["hold_s"] * hz)))
    thr = rules["common"]["release_N"]
    ev = {}
    for n, r in ref_env.o.items():
        if r["meta"].get("role") != "main":
            continue
        if float(np.linalg.norm(r["p"][-1] - r["p"][0])) < rules["put"]["min_ref_disp_m"]:
            continue
        ents = emap.get(n)
        ents = [ents] if isinstance(ents, str) else list(ents or [])
        got = [e for e in ents if e in series]
        if not got:
            raise SystemExit(f"{MARK} data force has no entity {ents} of {n} (have {sorted(series)})")
        fmax = max(float(np.max(series[e][-k:])) for e in got)
        ev[n] = {"entities": got, "end_max_n": round(fmax, 4), "released": fmax < thr}
    if not ev:
        return False, {"rule": "auto", "why": "no object moved by the reference"}
    on = all(v["released"] for v in ev.values())
    return on, {
        "rule": "auto",
        "window_rows": k,
        "release_N": thr,
        "objs": ev,
        "force_file": ff,
        "hz": hz,
        "measure": "force_object: sum of |F| per link (collect_force.link_mag)",
    }


def judge_run(run: Path, mode: str, rules: dict, home_map: str) -> dict:
    """One session output (run, mode) → the judge record (judge.json content). Meshes from the root set_data_root
    fixed.
    """
    job = json.loads((run / "job_run.json").read_text())
    tc = tree_clip_of(job, home_map)
    ti, objs = task_objects(tc)
    kin, hz = kin_scene(tc, objs)
    if abs(hz - HZ) > 1e-6:
        raise SystemExit(f"{MARK} kinematic rate {hz} != {HZ}")
    envo, chk = env_ref(run, objs, kin)
    ref_env = Scene(envo, None, "reference_env")
    emap = entity_map(tc, ti)
    sc, info = roll_scene(run / mode / "episode.npz", objs, emap, kin, envo, rules["common"])
    release, rel_ev = release_of(job, rules, job["axis"], emap, ref_env)
    res = judge(job["axis"], job["hand_use"], rules, ref_env, kin, sc, job["dataset"], job["base"], release)
    res.update(
        run=str(run),
        mode=mode,
        clip=job["base"],
        dataset=job["dataset"],
        hand_use=job["hand_use"],
        variant=job.get("variant"),
        tree_clip=str(tc),
        release=rel_ev,
        ref_oracle=chk,
        missing_objects=info["missing"],
        objects_from_reference=info["from_reference"],
        frame_fit_m=info["frame_fit_m"],
        rows=sc.T,
        cap_rows=info["meta"].get("cap_rows"),
        data_rows=info["meta"].get("data_rows"),
    )
    return res


def _dump(x):
    return x.item() if hasattr(x, "item") else str(x)


def summarize(runs: list[Path], mode: str) -> dict:
    """Per-task table: attempts n, invalid, judgeable (R2), dropped-reason counts (under the table), success /
    judgeable.
    A run without judge.json counts as 'no judge'."""
    import collections

    T: dict = {}
    for r in runs:
        f = r / mode / "judge.json"
        ax = json.loads((r / "job_run.json").read_text())["axis"] if (r / "job_run.json").is_file() else "?"
        t = T.setdefault(
            ax,
            {
                "n": 0,
                "no_judge": 0,
                "invalid": 0,
                "judgeable": 0,
                "success": 0,
                "dropped": collections.Counter(),
                "no_judge_runs": [],
            },
        )
        t["n"] += 1
        if not f.is_file():
            t["no_judge"] += 1
            t["no_judge_runs"].append(str(r))
            continue
        j = json.loads(f.read_text())
        if not j["judgeable"]:
            for w in j["judgeable_why"]:
                t["dropped"][w.split(":")[0] if w.startswith("reference failed") else w] += 1
            continue
        if j["invalid"]:
            t["invalid"] += 1
            continue
        t["judgeable"] += 1
        t["success"] += int(j["success"])
    for t in T.values():
        t["dropped"] = dict(t["dropped"])
        t["rate"] = round(t["success"] / t["judgeable"], 4) if t["judgeable"] else None
    return T


def main() -> None:
    """CLI: judge one run (default) or summarize judged runs (`summarize`)."""
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        ap = argparse.ArgumentParser(prog="judge_v1 summarize")
        ap.add_argument(
            "--runs", required=True, help="file listing run folders (one per line) or a folder holding them"
        )
        ap.add_argument("--mode", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args(sys.argv[2:])
        src = Path(a.runs)
        runs = (
            [Path(x) for x in src.read_text().split() if x]
            if src.is_file()
            else sorted(d for d in src.iterdir() if (d / "job_run.json").is_file())
        )
        T = summarize(runs, a.mode)
        Path(a.out).write_text(
            json.dumps({"mode": a.mode, "runs": len(runs), "tasks": T}, indent=1, ensure_ascii=False)
        )
        for ax, t in sorted(T.items()):
            print(
                f"{MARK} table {a.mode} {ax}: attempts {t['n']} · no judge {t['no_judge']} · dropped (R2) "
                f"{sum(t['dropped'].values())} {t['dropped']} · invalid {t['invalid']} · success "
                f"{t['success']}/{t['judgeable']} = {t['rate']}",
                flush=True,
            )
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--data-root", required=True, help="data root (holds data/ with the object meshes); read only")
    ap.add_argument(
        "--home-map", required=True, help="'<build-machine HOME>=<local copy>' — maps job paths (trees) here"
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    set_data_root(a.data_root)
    with open(a.rules) as fh:
        rules = yaml.safe_load(fh)
    run = Path(a.run)
    res = judge_run(run, a.mode, rules, a.home_map)
    out = Path(a.out) if a.out else run / a.mode / "judge.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=_dump))
    print(
        f"{MARK} {res['clip']} {a.mode} {res['axis']} target_rule={res['target_rule']} · judgeable {res['judgeable']} "
        f"{res['judgeable_why']} · invalid {res['invalid']} · success {res['success']} @ {res['success_s']} s → {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
