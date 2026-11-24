#!/usr/bin/env python3
"""E5 kinematic baseline — solve the placement of clips that have no stage-1 placement record of ours (adopted)
**kinematically** (placement_src=kin_opt), so the baseline keeps every clip: same search space, objective = reach
ratio over all frames (not tied to frame 0).

Search space = our pipeline's placement freedom (the rigid_pl formula):
  · tree-plane rigid motion p' = Rz(yaw)·(p − pivot) + pivot + (dx, dy) — desk, objects and hands together
    (dexengine.placement.retime_variants.rigid_pl)
  · robot root z = robot_z_off over the dataset's layers (dexengine.placement.search_policy.dz_cells_for(ds, scene) ×
  CELL_M — our judge's layers)
  · pivot = centre of the active palms over all frames (tree plane) · (dx, dy) = the move of that centre to the work
    centre c (+ grid offsets u, v) (c = c_robot for two hands, c_<side> for one — the offsets table kin_offsets.json)
    · yaw = ψ_ds + grid (±YAW_SPAN)
Objective:
  A (fast proxy) — ratio of frames (all frames, min over active hands) inside the arm tip's (palm origin) reachable set
    (FK of random samples within joint limits, VOX cells, cells with samples ≥ the CORE_Q quantile = 'inside') — the
    whole grid in numpy.
  B (confirmation) — for the top TOPK candidates of A, naive IK (DLS from the previous solution, frame-0 multi-start)
    every STRIDE frames, ratio of frames within 1 cm / 10° (reach.py's tolerances, min over active hands) → the best
    candidate. Ties: A score, |layer − default layer|, |yaw − ψ_ds|.
  No choice by success (judge or RL), no physics, judge, retiming or filtering (scores are only recorded).
Output: <shadow root>/precompute/<ds>/robots/<combo>/placement_kinopt/<clip>_pl_rt.kinopt.json (offsets.kinopt_path) —
read by prep.py --placement pipeline. Marker [KIN_OPT] <clip> … scores A, B, seconds.
  python -m experiments.policies.kin_data.opt --rows <tsv ds·emb·clip> --inputs <base root> --shadow <shadow root>
      --offsets <kin_offsets.pipeline.json> --out <jsonl> [--part k/n]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SR

WT = Path(__file__).resolve().parents[3]  # repo root

MARK = "[KIN_OPT]"
ARM, HAND, HAND_DIR = "rby1", "wuji2", "wuji2"
VOX = 0.02  # reachable-set cell (m)
N_SAMPLES = 400_000  # FK samples per arm (uniform within joint limits, seed 0)
CORE_Q = 25  # inside cells = this quantile or more of the non-empty cells' sample counts
UV_SPAN, UV_STEP = 0.30, 0.03
YAW_SPAN, YAW_STEP = 90.0, 10.0
A_FRAMES = 80  # frames used by A (uniform)
TOPK = 10
STRIDE = 4  # IK frame stride of B
RZ_NEG90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _aik():
    from dexengine.placement import arm_ik as m  # noqa: PLC0415

    return m


def _dz():
    from dexengine.placement import search_policy as m  # noqa: PLC0415

    return m


class Arm:
    """One arm's model and reachable set (robot base frame = model origin, the frame of prep_tree.kin_ik)."""

    def __init__(self, AIK, side: str, cache: Path):
        import pinocchio as pin  # noqa: PLC0415

        self.pin = pin
        urdf, ee, joints, base_xyz, _ = AIK.resolve_combo(ARM, HAND, side)
        from dexengine.robots import urdf_load as UL  # noqa: PLC0415

        self.model = UL.build_model(str(urdf))
        self.data = self.model.createData()
        self.ee = self.model.getFrameId(ee)
        self.qidx = tuple(self.model.joints[self.model.getJointId(n)].idx_q for n in joints)
        self.base_xyz = np.asarray(base_xyz, float)
        key = hashlib.sha256(f"{urdf}|{ee}|{joints}|{N_SAMPLES}|{VOX}".encode()).hexdigest()[:12]
        f = cache / f"reach_{side}_{key}.npz"
        if f.is_file():
            z = np.load(f)
            self.lo, self.occ, self.thr = z["lo"], z["occ"], int(z["thr"])
        else:
            rng = np.random.default_rng(0)
            P = np.empty((N_SAMPLES, 3))
            q = pin.neutral(self.model)
            lo_q = np.array([self.model.lowerPositionLimit[i] for i in self.qidx])
            hi_q = np.array([self.model.upperPositionLimit[i] for i in self.qidx])
            Q = rng.uniform(lo_q, hi_q, size=(N_SAMPLES, len(self.qidx)))
            for n in range(N_SAMPLES):
                for j, qi in enumerate(self.qidx):
                    q[qi] = Q[n, j]
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacement(self.model, self.data, self.ee)
                P[n] = self.data.oMf[self.ee].translation
            self.lo = P.min(0) - VOX
            idx = np.floor((P - self.lo) / VOX).astype(int)
            shp = idx.max(0) + 2
            occ = np.zeros(shp, np.int32)
            np.add.at(occ, tuple(idx.T), 1)
            nz = occ[occ > 0]
            self.occ, self.thr = occ, int(np.percentile(nz, CORE_Q))
            cache.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
            np.savez(tmp, lo=self.lo, occ=self.occ, thr=np.int64(self.thr))
            os.replace(tmp, f)
        self.core = self.occ >= self.thr

    def inside(self, X: np.ndarray) -> np.ndarray:
        """X (..., 3) in the base frame → True inside cells."""
        idx = np.floor((X - self.lo) / VOX).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(self.core.shape)), axis=-1)
        out = np.zeros(X.shape[:-1], bool)
        ii = idx[ok]
        out[ok] = self.core[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out


def palms_tree(AIK, inputs: Path, ds: str, clip: str, side: str):
    """Raw wrist cache (base tree frame) → palm origin positions and rotations (tree frame) — the point kin_ik targets
    (palm_frame_offset).
    """
    w = np.load(inputs / f"precompute/{ds}/robots/{HAND_DIR}/wrist_poses_{side}/{clip}.npz")
    P, Qw = w["positions"].astype(float), w["quaternions"].astype(float)
    R = SR.from_quat(np.c_[Qw[:, 1], Qw[:, 2], Qw[:, 3], Qw[:, 0]])  # wxyz → xyzw
    return P + R.apply(AIK.palm_frame_offset(HAND)), R


def _levels(ds: str) -> tuple[list[float], float]:
    """Robot height layers searched for the dataset and the default layer (m)."""
    from experiments.policies.kin_data import offsets as KO  # noqa: PLC0415

    fl = KO.ds_flags(ds)
    dzm = _dz()
    levels = [float(c) * dzm.CELL_M for c in dzm.dz_cells_for(ds, fl["scene"]).split(",")]
    zdef = float(dzm.TACO_CANON) * dzm.CELL_M
    return levels, zdef


def _placement_grid(c: np.ndarray, piv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Work-centre offsets (U, V, scene) and the matching (dx, dy) move candidates in the tree plane."""
    uv = np.arange(-UV_SPAN, UV_SPAN + 1e-9, UV_STEP)
    U, V = np.meshgrid(uv, uv, indexing="ij")
    U, V = U.ravel(), V.ravel()
    # work centre (scene) → tree plane: scene = Rz(−90°)·tree → tree = Rz(+90°)·scene
    g_scene = np.stack([c[0] + U, c[1] + V, np.zeros_like(U)], 1)
    g_tree = (RZ_NEG90.T @ g_scene.T).T[:, :2]  # (G, 2)
    D = g_tree - piv  # (G, 2) = (dx, dy) candidates
    return U, V, D


def _score_a(arms: dict, act: list, pal: dict, fa, piv, yaws, levels, D, zoff: float) -> np.ndarray:
    """A: share of sampled palm frames inside the reachable core, per (yaw, layer, move); min over active hands."""
    scoreA = np.full((len(yaws), len(levels), len(D)), np.inf)
    for iy, yw in enumerate(yaws):
        cy, sy = np.cos(np.radians(yw)), np.sin(np.radians(yw))
        Rz = np.array([[cy, -sy], [sy, cy]])
        for s in act:  # min over active hands (two hands = the lower)
            p = pal[s][0][fa]
            pr = (p[:, :2] - piv) @ Rz.T + piv  # (F, 2) rotation only (before the move)
            for il, lz in enumerate(levels):
                base = arms[s].base_xyz + np.array([0.0, 0.0, lz])
                xy = pr[None, :, :] + D[:, None, :]  # (G, F, 2) tree plane
                X = np.empty(xy.shape[:2] + (3,))
                X[..., 0] = xy[..., 1]  # scene = Rz(−90°)·tree: (x, y) → (y, −x)
                X[..., 1] = -xy[..., 0]
                X[..., 2] = p[None, :, 2] + zoff
                scoreA[iy, il] = np.minimum(scoreA[iy, il], arms[s].inside(X - base).mean(1))  # (G,)
    return scoreA


def _top_candidates(scoreA: np.ndarray, levels, zdef: float, yaws, psi: float, U, V, D) -> list[tuple]:
    """The TOPK best A candidates with the tie break, at most one per (yaw, layer)."""
    # tie break: higher A, layer near the default, yaw near ψ, smaller move
    cand = [
        (
            -scoreA[iy, il, ig],
            abs(levels[il] - zdef),
            abs(yaws[iy] - psi),
            float(np.hypot(U[ig], V[ig])),
            iy,
            il,
            ig,
        )
        for iy in range(len(yaws))
        for il in range(len(levels))
        for ig in range(len(D))
    ]
    cand.sort()
    top, seen = [], set()
    for cnd in cand:  # diversity: one per (yaw, layer)
        k = (cnd[4], cnd[5])
        if k in seen:
            continue
        seen.add(k)
        top.append(cnd)
        if len(top) >= TOPK:
            break
    return top


def _score_b(AIK, arms: dict, act: list, pal: dict, top: list[tuple], T: int, piv, yaws, levels, D, zoff, zdef, psi):
    """B: naive IK hit rate of each top candidate; returns the best (key, record) and every tried record."""
    from experiments.policies.kin_data import reach as KR  # noqa: PLC0415

    # B: naive IK (DLS from previous, frame-0 multi-start) every STRIDE frames
    rz = SR.from_euler("z", -90, degrees=True)
    fb = np.arange(0, T, STRIDE)
    best = None
    tried = []
    for cnd in top:
        _a, _dzd, _dy, _r, iy, il, ig = cnd
        yw, lz, d = float(yaws[iy]), float(levels[il]), D[ig]
        Ryaw = SR.from_euler("z", yw, degrees=True)
        per = {}
        for s in act:
            arm = arms[s]
            base = arm.base_xyz + np.array([0.0, 0.0, lz])
            p, R = pal[s]
            q, hit = None, 0
            for t in fb:
                pt = np.r_[(np.asarray(p[t, :2]) - piv) @ Ryaw.as_matrix()[:2, :2].T + piv + d, p[t, 2]]
                tp = rz.apply(pt) + np.array([0.0, 0.0, zoff]) - base
                tr = (rz * Ryaw * R[t]).as_matrix()
                if q is None:
                    q, e_p, e_r, _ = AIK.solve_ik_multistart(
                        arm.model, arm.data, tp, tr, arm.ee, arm.qidx, num_starts=10, max_iters=300
                    )
                else:
                    q, e_p, e_r, _ = AIK.solve_ik_pose(
                        arm.model, arm.data, tp, tr, arm.ee, arm.qidx, max_iters=200, q0=q
                    )
                hit += int(e_p <= KR.POS_TOL_M and e_r <= KR.ROT_TOL_DEG)
            per[s] = hit / len(fb)
        sB = min(per.values())
        tried.append(
            {
                "yaw": round(yw, 2),
                "robot_z_off": round(lz, 5),
                "dx": round(float(d[0]), 6),
                "dy": round(float(d[1]), 6),
                "a": round(float(-_a), 4),
                "b": round(sB, 4),
                "b_per_side": {k: round(v, 4) for k, v in per.items()},
            }
        )
        key = (sB, -_a, -abs(lz - zdef), -abs(yw - psi))
        if best is None or key > best[0]:
            best = (key, tried[-1])
    return best, tried


def _placement_record(
    clip: str, best: tuple, tried: list, piv, act: list, yaws, levels, c, psi: float, zoff: float, T: int, t0: float
) -> dict:
    """kin_opt placement record of the best candidate, with every tried candidate, the search space and parameters."""
    from dexengine.core import paths as P  # noqa: PLC0415

    b = best[1]
    return {
        "sequence": P.tree_name(clip),
        "dx": b["dx"],
        "dy": b["dy"],
        "yaw": b["yaw"],
        "pivot": [round(float(piv[0]), 6), round(float(piv[1]), 6), 0.0],
        "robot_z_off": b["robot_z_off"],
        "placement_src": "kin_opt",
        "valid": True,
        "score_a": b["a"],
        "score_b": b["b"],
        "score_b_per_side": b["b_per_side"],
        "active_sides": act,
        "tried": tried,
        "space": {
            "yaw": [float(yaws[0]), float(yaws[-1]), YAW_STEP],
            "uv": [-UV_SPAN, UV_SPAN, UV_STEP],
            "levels": [round(x, 5) for x in levels],
            "center": [round(float(x), 5) for x in c],
            "psi": psi,
        },
        "params": {
            "VOX": VOX,
            "N_SAMPLES": N_SAMPLES,
            "CORE_Q": CORE_Q,
            "A_FRAMES": A_FRAMES,
            "TOPK": TOPK,
            "STRIDE": STRIDE,
        },
        "zoff": round(zoff, 5),
        "T": int(T),
        "s": round(time.time() - t0, 1),
    }


def solve_clip(AIK, arms: dict, offs: dict, inputs: Path, ds: str, emb: str, clip: str) -> dict:
    from dexengine.core import qpos_layout as QL  # noqa: PLC0415
    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415
    from dexengine.placement.coords import core as CO  # noqa: PLC0415

    t0 = time.time()
    src = inputs / f"data/{ds}/base/{HAND_DIR}/{emb}/{clip}"
    ti0 = json.loads((src / "task_info.json").read_text())
    act, _idle, _rule = IA.active_idle_for_tree(ti0.get("case"), QL.read(str(src), nq=None).hands)
    act = list(act)
    if not act:
        raise ValueError("no active hand")
    zoff = float(CO.scene_z_offset(src))
    levels, zdef = _levels(ds)
    psi = float(offs["datasets"][ds]["yaw_canon_deg"])
    c = np.asarray(offs["c_side"][act[0]] if len(act) == 1 else offs["c_robot"], float)
    pal = {s: palms_tree(AIK, inputs, ds, clip, s) for s in act}
    T = len(next(iter(pal.values()))[0])
    piv = np.mean(np.concatenate([pal[s][0] for s in act])[:, :2], axis=0)
    fa = np.unique(np.linspace(0, T - 1, min(A_FRAMES, T)).astype(int))
    yaws = psi + np.arange(-YAW_SPAN, YAW_SPAN + 1e-9, YAW_STEP)
    U, V, D = _placement_grid(c, piv)
    scoreA = _score_a(arms, act, pal, fa, piv, yaws, levels, D, zoff)
    top = _top_candidates(scoreA, levels, zdef, yaws, psi, U, V, D)
    best, tried = _score_b(AIK, arms, act, pal, top, T, piv, yaws, levels, D, zoff, zdef, psi)
    return _placement_record(clip, best, tried, piv, act, yaws, levels, c, psi, zoff, T, t0)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--shadow", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument(
        "--oracle-out",
        default="",
        help="comparison run — also solves clips that have an adopted record and writes only here (kin_prep "
        "does not read it · for comparison with adopted)",
    )
    return ap.parse_args()


def _read_part(a: argparse.Namespace) -> list[dict]:
    """Rows of the clip table that belong to part k/n."""
    with open(a.rows) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    k, n = (int(x) for x in a.part.split("/"))
    rows = rows[k::n]
    return rows


def _solve_and_write(a, AIK, arms: dict, offs: dict, inputs: Path, shadow: Path, commit: str, ds, emb, clip):
    """Solve one clip and write its kin_opt record (shadow root, or the oracle folder); returns (record, path)."""
    from experiments.policies.kin_data import offsets as KO  # noqa: PLC0415

    if KO.adopted_record(inputs, ds, clip) is not None and not a.oracle_out:
        raise ValueError("an adopted record exists — not a kin_opt clip")
    rec = solve_clip(AIK, arms, offs, inputs, ds, emb, clip)
    rec.update(code=commit, offsets_sha256=hashlib.sha256(Path(a.offsets).read_bytes()).hexdigest()[:16])
    f = (Path(a.oracle_out) / f"{ds}__{clip}.kinopt.json") if a.oracle_out else KO.kinopt_path(shadow, ds, clip)
    f.parent.mkdir(parents=True, exist_ok=True)
    if not a.oracle_out and not str(f.parent.resolve()).startswith(str(shadow) + os.sep):
        raise SystemExit(f"{MARK} output {f} is outside the shadow ({f.parent.resolve()}) — not written")
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(tmp, f)
    return rec, f


def _ok_row(ds: str, emb: str, clip: str, rec: dict, f: Path) -> dict:
    res = {
        "ds": ds,
        "emb": emb,
        "clip": clip,
        "ok": True,
        "score_a": rec["score_a"],
        "score_b": rec["score_b"],
        "yaw": rec["yaw"],
        "robot_z_off": rec["robot_z_off"],
        "s": rec["s"],
        "record": str(f),
    }
    print(
        f"{MARK} {clip} OK {rec['s']} s · A {rec['score_a']} · B {rec['score_b']} · ψ {rec['yaw']} · "
        f"z {rec['robot_z_off']}",
        flush=True,
    )
    return res


def _fail_row(ds: str, emb: str, clip: str, e: Exception) -> dict:
    """Result row of a failed clip; called inside the except block (the traceback is the active exception's)."""
    res = {
        "ds": ds,
        "emb": emb,
        "clip": clip,
        "ok": False,
        "err": f"{type(e).__name__}: {e}"[:600],
        "tb": traceback.format_exc()[-4000:],
    }
    print(f"{MARK} {clip} FAIL {res['err'][:300]}", flush=True)
    return res


def main() -> int:
    a = _parse_args()
    from experiments.policies.kin_data import offsets as KO  # noqa: PLC0415

    offs = KO.load(a.offsets)
    if offs.get("placement") != "pipeline":
        raise SystemExit(f"{MARK} offsets table {a.offsets}: placement {offs.get('placement')!r} ≠ pipeline")
    inputs, shadow = Path(a.inputs), Path(a.shadow).resolve()
    if not (shadow / "DATA_SHADOW.json").is_file():
        raise SystemExit(f"{MARK} --shadow {shadow} has no DATA_SHADOW.json — writes go to a shadow root only")
    commit = subprocess.run(
        ["git", "-C", str(WT), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    AIK = _aik()
    cache = shadow / "_kin_opt_cache"
    arms = {s: Arm(AIK, s, cache) for s in ("right", "left")}
    rows = _read_part(a)
    print(f"{MARK} start {len(rows)} clips · code {commit} · reach voxel {VOX} m · core quantile {CORE_Q}", flush=True)
    ok = fail = 0
    with open(a.out, "a") as fh:
        for r in rows:
            ds, emb, clip = r["ds"], r["emb"], r["clip"]
            try:
                rec, f = _solve_and_write(a, AIK, arms, offs, inputs, shadow, commit, ds, emb, clip)
                ok += 1
                res = _ok_row(ds, emb, clip, rec, f)
            except Exception as e:  # noqa: BLE001 — one clip's failure is recorded and the next one runs (the last line's fail count says it)
                fail += 1
                res = _fail_row(ds, emb, clip, e)
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"{MARK} DONE ok={ok} fail={fail} total={len(rows)}", flush=True)
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
