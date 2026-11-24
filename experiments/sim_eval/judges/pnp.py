#!/usr/bin/env python3
"""P&P judge pnp-a2nr (the paper's pick-and-place rule for the Lift task).

P&P success = judge-v2 Lift success (the run's judge_v2.json — judgeable and not invalid included) ∧ the episode ends
after the Lift success row ∧ for every main object
  ① the centre of the mesh bounding box at the last frame is within PNP['eps_m'] (5 cm) of the place target (the same
     point at the reference last frame) ∧
  ② not flipped — the tilt difference (angle between world-up seen in the body frame, u = R_endᵀ·ẑ, at the rollout
  end
     and at the reference end; rotation about z is ignored) ≤ PNP['flip_deg'] (135°).
  There is no release condition: ending within 5 cm above the place while still holding counts as success.
  Place target = place_ref.place_target (last frame of that variant's trajectory_kinematic.npz, put-in-place included);
  judged at the last rollout row; no static condition.
  Target frame: the receiver's end body frame when the receiver is a main object (frame independent; reference = the
  receiver's body frame at the tree's last frame), else the world moved by the tree → env rigid transform
  (judge_v1.env_from_tree).
  Mesh box centre = centre of the axis-aligned box of all visual.obj vertices (body frame), resolved like
  judge_v1.mesh_pts (main_root · data_layout.object_path).
  Reported only (not judged): body-origin end distance d_org_end_m · end height gap dz_end_m · end hand force
  force_end_N.
The record's rule text, why strings and `judge` name are stored data (counting reads `judge`); they keep their wording.
Usage: python -m experiments.sim_eval.judges.pnp --run R --mode M --data-root DR --home-map HM --out F"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.sim_eval.judges import place_ref as PR

RU = Path(__file__).resolve().parent / "rules_v2.yaml"
PNP = {
    "eps_m": 0.05,
    "flip_deg": 135.0,
    "point": "mesh_box_center",
    "rule": (
        "pnp-a2nr-20261102 (Lift v2 ∧ ends after the Lift row ∧ end-frame mesh box centre within 5 cm ∧ "
        "not flipped (tilt difference > 135°) · no release condition · no stop)"
    ),
}
MARK = "[E5_PNP]"
# judge name written to the record's `judge` field (pnp_batch → level counting reads it)
NAME = "e5_judge_pnp_20261102 (pnp-a2nr)"
_CEN = {}


def _md5(p) -> str:
    """md5 of a file (judge source marks in the record)."""
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _load():
    """(judge_v2 module, frozen v1 module, v2 rules) — pass as _ctx to judge_pnp to reuse across runs."""
    from experiments.sim_eval.judges import judge_v2 as SF  # noqa: PLC0415

    M = SF.load_frozen()
    with open(RU) as fh:
        return SF, M, yaml.safe_load(fh)


def tilt_diff_deg(R_roll: np.ndarray, R_ref: np.ndarray) -> float:
    """Tilt difference of two attitudes (body→world rotations): angle (degrees) between world-up seen in each body
    frame.
    Rotation about z (yaw) gives 0."""
    z = np.array([0.0, 0.0, 1.0])
    a, b = R_roll.T @ z, R_ref.T @ z
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def mesh_box_center(M, ds: str, mesh_dir: str) -> np.ndarray:
    """Centre of the axis-aligned box of all visual.obj vertices (body frame); same path as judge_v1.mesh_pts. A missing
    or
    empty mesh raises."""
    k = (ds, mesh_dir)
    if k not in _CEN:
        P = M._paths()
        f = P.main_root() / P.data_layout.object_path(ds, mesh_dir) / "visual.obj"
        if not f.is_file():
            raise SystemExit(f"{MARK} object mesh missing {f}")
        with open(f, encoding="utf-8", errors="replace") as fh:
            V = np.array([[float(t) for t in ln.split()[1:4]] for ln in fh if ln.startswith("v ")])
        if not len(V):
            raise SystemExit(f"{MARK} object mesh has no vertices {f}")
        _CEN[k] = (V.max(0) + V.min(0)) / 2
    return _CEN[k]


def _place_checks(tg, sc, M, ds, K, end, Rt, tt):
    """Per placed object: end distance and tilt against its place target, end force and why it fails."""
    objs, ok = {}, True
    for n, t in tg.items():
        if n not in sc.o:
            raise SystemExit(f"{MARK} main object {n} is not in the rollout")
        s = sc.o[n]
        cen = mesh_box_center(M, ds, K[n]["meta"]["mesh_dir"])
        Rr, pr = s["R"][end], s["p"][end]
        if t["frame"] == "body" and t["receiver"] in sc.o:
            g = sc.o[t["receiver"]]
            gR, gp = g["R"][end], g["p"][end]
            gRf = K[t["receiver"]]["R"][
                -1
            ]  # receiver attitude at the reference end (tree frame; Rt cancels in the body-frame difference)
            cur_o = gR.T @ (pr - gp)
            tgt_o = t["rel_end"]
            cur = gR.T @ (Rr @ cen + pr - gp)
            tgt = tgt_o + gRf.T @ (t["R_end"] @ cen)
            dz = float((gR @ (cur_o - tgt_o))[2])
        else:
            pf, Rf = Rt @ t["p_end"] + tt, Rt @ t["R_end"]
            cur_o, tgt_o = pr, pf
            cur, tgt = Rr @ cen + pr, Rf @ cen + pf
            dz = float(pr[2] - pf[2])
        d = float(np.linalg.norm(cur - tgt))
        dtl = tilt_diff_deg(Rr, Rt @ t["R_end"])
        f = (sc.force or {}).get(n)
        o_ok = d <= PNP["eps_m"] and dtl <= PNP["flip_deg"]
        why = (
            ""
            if o_ok
            else (
                "off the place spot (end-frame mesh centre)"
                if d > PNP["eps_m"]
                else f"flipped (tilt difference to the reference {dtl:.0f}° > {PNP['flip_deg']:.0f}°)"
            )
        )
        objs[n] = {
            "d_end_m": round(d, 4),
            "d_org_end_m": round(float(np.linalg.norm(cur_o - tgt_o)), 4),
            "dz_end_m": round(dz, 4),
            "dtilt_deg": round(dtl, 1),
            "eps_m": PNP["eps_m"],
            "frame": t["frame"],
            "receiver": t["receiver"],
            "disp_ref_m": round(t["disp_m"], 4),
            "center_body_m": [round(float(x), 4) for x in cen],
            "force_end_N": None if f is None else round(float(f[end]), 4),
            "ok": o_ok,
            "why": why,
        }
        ok = ok and o_ok
    return objs, ok


def _verdict(lift_ok, lrow, end, objs, ok, fit, j2, j2p, run, mode):
    """The P&P record: lift verdict, placement after the lift row, per-object checks and the judge marks."""
    after = lift_ok and lrow is not None and end > int(lrow)
    place_ok = bool(objs) and ok and after
    why = (
        ""
        if (lift_ok and place_ok)
        else (
            "Lift failed"
            if not lift_ok
            else (
                "ends before the Lift success row"
                if not after
                else next(v["why"] for v in objs.values() if not v["ok"])
            )
        )
    )
    return {
        "axis": "PnP",
        "judge": NAME,
        "success": lift_ok and place_ok,
        "why": why,
        "lift_success": lift_ok,
        "lift_row": lrow,
        "end_row": end,
        "after": after,
        "place_ok": place_ok,
        "objs": objs,
        "rule": PNP,
        "frame_fit_m": round(fit, 6),
        "marks": {
            "judge_pnp_md5": _md5(__file__),
            "place_ref_md5": _md5(PR.__file__),
            "judge_v2_marks": j2.get("v2_marks") or j2.get("marks"),
            "judge_v2_file": str(j2p),
        },
        "run": str(run),
        "mode": mode,
    }


def judge_pnp(run: str | Path, mode: str, data_root: str, home_map: str, _ctx=None) -> dict:
    """P&P verdict of one run (needs its <mode>/judge_v2.json). _ctx = a _load() result to reuse."""
    run = Path(run)
    j2p = run / mode / "judge_v2.json"
    if not j2p.is_file():
        raise SystemExit(f"{MARK} {j2p} missing — run judge v2 first")
    j2 = json.loads(j2p.read_text())
    SF, M, rules = _ctx or _load()
    M.set_data_root(data_root)
    ctx = SF.load_ctx(M, run, mode, rules, home_map)
    sc, kin = ctx["sc"], ctx["kin"]
    Rt, tt, fit = M.env_from_tree(ctx["ref_env"].o, kin)
    tg = PR.place_target(ctx["tc"])
    K = PR.kin_objects(ctx["tc"])
    ti = json.loads((Path(ctx["tc"]) / "task_info.json").read_text())
    ds = str(ti.get("dataset_name") or "")
    if not ds:
        raise SystemExit(f"{MARK} {ctx['tc']}/task_info.json has no dataset_name")
    lift_ok = bool(j2.get("success"))
    lrow = j2.get("success_row")
    end = sc.T - 1
    objs, ok = _place_checks(tg, sc, M, ds, K, end, Rt, tt)
    return _verdict(lift_ok, lrow, end, objs, ok, fit, j2, j2p, run, mode)


def main() -> None:
    """CLI: judge one run and write --out (outside the run folder)."""
    ap = argparse.ArgumentParser()
    for a in ("--run", "--mode", "--data-root", "--home-map", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.resolve().is_relative_to(Path(a.run).resolve()):
        raise SystemExit(f"{MARK} --out must not be inside the run folder (records are not overwritten)")
    r = judge_pnp(a.run, a.mode, a.data_root, a.home_map)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float))
    print(f"{MARK} {Path(a.run).name} success {r['success']} why {r['why'] or '-'}")


if __name__ == "__main__":
    main()
