"""CHORD released metrics for the shared scorer (split from score.py): object vertices, reference lookup, ADD-AUC,
SP-SR / MP-SR through flash_chord.evaluation.metrics (imported, not copied).

Standard library + numpy (+ flash_chord at call time): score.py loads this file by path when it runs by file path
under `python -I`."""

from __future__ import annotations

import os

import numpy as np


# ---------- CHORD released metrics (owner: flash_chord/src/flash_chord/evaluation/metrics.py — imported, not copied)
# ---------- Every method hands the cohort roll = {t: (T,) original-motion time τ of each method step (the method's
# own time axis), obj_pos_<b> (W,T,3) · obj_quat_<b> (W,T,4 wxyz)} + ref = {dt, objs {b: (pos (F,3), quat wxyz
# (F,4))}} (original motion frames) + object vertices from the mesh the method simulates. Steps with τ in [0, τ_end]
# are scored, τ_end = the end of the reference or the method's own horizon if shorter. A rollout that stops before
# τ_end (e.g. EgoEngine's failed prefix) is not a full-sequence attempt: its missing steps count as ADD = ∞ and both
# SRs are 0 for it.
def chord_metrics_module():
    try:
        import flash_chord.evaluation.metrics as M
    except ImportError as e:
        raise SystemExit(
            f"[SCORE] CHORD metrics need the FlashCHORD venv (flash_chord.evaluation.metrics): {e}"
        ) from None
    return M


def sample_vertices(per_body: list) -> np.ndarray:
    """(B, ADD_VERTEX_COUNT, 3) — the count/seed are CHORD's constants and the draw is CHORD's rule
    (flash_chord.evaluation.rollout.sample_object_vertices: one default_rng(seed) over the bodies in order, choice
    without replacement unless a body has fewer vertices). That function needs a newton scene, so the rule is applied
    here to
    the vertices of the mesh the method itself simulates."""
    M = chord_metrics_module()
    g = np.random.default_rng(M.ADD_VERTEX_SEED)
    out = []
    for v in per_body:
        v = np.asarray(v, float)
        out.append(v[g.choice(v.shape[0], size=M.ADD_VERTEX_COUNT, replace=v.shape[0] < M.ADD_VERTEX_COUNT)])
    return np.stack(out)


def vertices_urdf(urdf: str) -> np.ndarray:
    """Visual mesh vertices of a single-link object URDF in the link frame (origin xyz/rpy and scale applied)."""
    import xml.etree.ElementTree as ET

    import trimesh
    from scipy.spatial.transform import Rotation as Rr

    vis = ET.parse(urdf).getroot().findall(".//visual")
    if not vis:
        raise SystemExit(f"[SCORE] {urdf}: no <visual> mesh")
    pts = []
    for v in vis:
        m = v.find(".//mesh")
        o = v.find("origin")
        f = os.path.join(os.path.dirname(urdf), m.get("filename"))
        sc = np.array([float(x) for x in (m.get("scale") or "1 1 1").split()])
        xyz = np.array([float(x) for x in ((o.get("xyz") if o is not None else None) or "0 0 0").split()])
        rpy = np.array([float(x) for x in ((o.get("rpy") if o is not None else None) or "0 0 0").split()])
        mesh = trimesh.load(f, force="mesh")
        pts.append(Rr.from_euler("xyz", rpy).apply(np.asarray(mesh.vertices, float) * sc) + xyz)
    return np.concatenate(pts)


def vertices_mjcf(scene_xml: str, n_objects: int) -> list:
    """Object meshes of a MuJoCo scene (SPIDER · EgoEngine) in each object body's frame: the bodies of the last
    n_objects free joints (qpos order = right, left as the qpos slices), their visual mesh geoms — no contact bit and
    in no explicit contact <pair> (SPIDER scenes collide only through pairs: every geom has contype = conaffinity = 0)
    — or all mesh geoms
    if a body has no visual one."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(scene_xml)
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][-n_objects:]
    paired = set(m.pair_geom1.tolist()) | set(m.pair_geom2.tolist())
    out = []
    for j in free:
        b = m.jnt_bodyid[j]
        g_all = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        g_vis = [g for g in g_all if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and g not in paired]
        pts = []
        for g in g_vis or g_all:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(v @ R.reshape(3, 3).T + m.geom_pos[g])
        if not pts:
            raise SystemExit(f"[SCORE] {scene_xml}: object body {b} has no mesh geom")
        out.append(np.concatenate(pts))
    return out


def ref_at(ref_pos, ref_quat, dt, tau):
    """Original-motion pose at τ: position linear, orientation slerp (wxyz in, wxyz out)."""
    from scipy.spatial.transform import Rotation as Rr
    from scipy.spatial.transform import Slerp

    tf = np.arange(len(ref_pos)) * dt
    tq = np.clip(tau, 0.0, tf[-1])
    p = np.stack([np.interp(tq, tf, ref_pos[:, i]) for i in range(3)], -1)
    q = Slerp(tf, Rr.from_quat(np.asarray(ref_quat)[:, [1, 2, 3, 0]]))(tq).as_quat()[:, [3, 0, 1, 2]]
    return p, q


def add_auc_best(M, per_rollout: list) -> tuple:
    """The table's ADD-AUC of a multi-rollout method = the best of the clip's rollouts (one rollout: best = pooled):
    each rollout's own AUC (CHORD's add_auc on that rollout alone · ∞ steps = missed) → the max (ties → the first).
    per_rollout = [(T_i, B) ADD arrays] → (best, index, all)."""
    a = [float(M.add_auc(np.asarray(x, dtype=np.float64)[:, None, :])[0]) for x in per_rollout]
    if not a or any(np.isnan(a)):
        raise ValueError(f"add_auc_best: {len(a)} rollouts, nan in {sum(map(np.isnan, a))}")
    i = int(np.argmax(a))
    return round(a[i], 4), i, [round(x, 4) for x in a]


def chord_metrics(roll: dict, ref: dict, verts: dict, step_dt: float, tau_end: float | None = None) -> dict:
    """ADD-AUC · SPIDER SR · ManipTrans SR (CHORD's functions) on the method's own time axis."""
    M = chord_metrics_module()
    bodies = sorted(ref["objs"])
    if sorted(verts) != bodies:
        raise ValueError(f"vertices for {sorted(verts)} ≠ reference bodies {bodies}")
    end_ref = (len(ref["objs"][bodies[0]][0]) - 1) * ref["dt"]
    tau_end = end_ref if tau_end is None else min(float(tau_end), end_ref)
    t = np.asarray(roll["t"], float)
    sel = (t >= -1e-9) & (t <= tau_end + 1e-9)
    n_exp = int(np.floor(tau_end / step_dt + 1e-6)) + 1  # method steps that cover [0, τ_end]
    n_have = int(sel.sum())
    missing = max(0, n_exp - n_have) if (n_have == 0 or t[sel][-1] < tau_end - 0.5 * step_dt) else 0
    W = roll[f"obj_pos_{bodies[0]}"].shape[0] if n_have else 1
    V = sample_vertices([verts[b] for b in bodies])
    if n_have:
        ach = np.stack(
            [
                np.concatenate([roll[f"obj_pos_{b}"][:, sel], roll[f"obj_quat_{b}"][:, sel][..., [1, 2, 3, 0]]], -1)
                for b in bodies
            ],
            2,
        ).transpose(1, 0, 2, 3)  # (T,W,B,7 xyzw)
        rr = [ref_at(*ref["objs"][b], ref["dt"], t[sel]) for b in bodies]
        refp = np.stack([np.concatenate([p, q[:, [1, 2, 3, 0]]], -1) for p, q in rr], 1)  # (T,B,7 xyzw)
        add = M.object_add(ach, refp, V)
        sp, sp_p, sp_r = M.spider_success(ach, refp)
        mt, mt_p, mt_r = M.maniptrans_success(ach, refp, np.arange(len(bodies)))
    else:
        add = np.zeros((0, W, len(bodies)))
        sp = mt = np.zeros(W, bool)
        sp_p = sp_r = mt_r = np.full(W, np.nan)
    if missing:
        add = np.concatenate([add, np.full((missing, W, len(bodies)), np.inf)], 0)
        sp = np.zeros(W, bool)
        mt = np.zeros(W, bool)
    auc, auc_b = M.add_auc(add)
    best, bw, per_w = add_auc_best(M, [add[:, w, :] for w in range(add.shape[1])]) if add.shape[0] else (None, None, [])
    return {
        "add_auc_best": best,
        "add_auc_best_world": bw,
        "add_auc_per_world": per_w,
        "add_auc_rule": "table = add_auc_best (best of the rollouts) · add_auc = pooled over rollouts and steps "
        "(record)",
        "add_auc": round(auc, 4),
        "add_auc_per_body": dict(zip(bodies, [round(x, 4) for x in auc_b], strict=False)),
        "spider_sr": float(np.mean(sp)),
        "maniptrans_sr": float(np.mean(mt)),
        "mean_add_m_scored_steps": round(float(np.mean(add[np.isfinite(add)])), 4) if np.isfinite(add).any() else None,
        "spider_pos_err_m": round(float(np.nanmean(sp_p)), 4) if n_have else None,
        "spider_rot_err_rad": round(float(np.nanmean(sp_r)), 4) if n_have else None,
        "maniptrans_rot_err_deg": round(float(np.nanmean(mt_r)), 2) if n_have else None,
        "steps_scored": n_have,
        "steps_missing": missing,
        "worlds": int(W),
        "tau_end_s": round(tau_end, 4),
        "step_dt": step_dt,
        "metrics_owner": "flash_chord.evaluation.metrics",
    }
