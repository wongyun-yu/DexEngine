#!/usr/bin/env python3
"""Ours rows of Tab. 2: CHORD metrics (ADD-AUC · SPIDER SR · ManipTrans SR, flash_chord.evaluation.metrics) from the
judge's object-pose file (``objtrace_recorder.py``, schema e123_objtrace/1). Per clip:
  episodes   real-batch ∧ valid rows of ep_table (warm 0 · valid 1) of that clip, ordered by env index, the first N
             (N = 64, fixed before any result: the '≥ 1 of 64' rule); fewer than N = the clip is not scored
  time       each step's reference tree row = ref_frame → original-motion time τ = (retime_src_idx[row] − pad) ×
  ref_dt
             (pad = task_info f0_prepad, else F0_REPAIR.json pad_new, else 0); τ < 0 (front pad) is not scored
  frame      env-local → CHORD world by ONE rigid transform per clip: Kabsch of the env reference positions (both
  objects,
             one episode — the reference is the same in every episode) onto the original motion at those τ; objects
             paired by TACO mesh id (tree obj_ids ↔ parquet object_mesh_paths). The fit residual is the oracle and is
             written per clip (align_pos_mm max/median · align_rot_deg)
  reference  every method is scored against the reference it was given (retiming is not an error): Ours = its own env
             reference (mode b · the table · score_ref = own_ref); mode a = the original motion (CHORD parquet, after
             the rigid map) is kept as a record with the per-clip gap between the two references. Mode b needs no map
             (ADD and both SRs are invariant to one rigid map applied to both poses); the map only serves mode a
  span       the original motion's span only (τ ≥ 0 — our front pad is not scored), the same window as the baselines
  missing    an episode that ended before the full span: its missing steps get ADD = inf and it fails SPIDER /
  ManipTrans
             success (score.py's rule). Full span in steps = (tree length − 1 − start frame) × ref_dt / step_dt + 1
             — checked against every completed episode (a mismatch fails the clip loudly)
  vertices   the mesh Ours simulates: <asset root>/<task_info mesh dir>/visual.obj (each method's own mesh, as in
             score.OBJECT_URDF); its bounding box is compared with CHORD's mesh of the same TACO id (recorded)
Reads the original motions under <workspace>/data/chord/assets/human_motion_data/taco/processed.
usage: python -m experiments.scoring.ours_objtrace <objtrace.npz> <tree root> <asset root> <out dir> [N=64]
       (FlashCHORD venv: flash_chord + pyarrow)"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# clip name ↔ tree name: owner dexengine.core.paths.clip_name — the objtrace carries tree names ('…_pl_rt'), the
# CHORD parquet is keyed by the clip's sequence id
from dexengine.core import paths as _OP
from experiments import workspace
from experiments.scoring import score as S
from experiments.scoring import score_chord as SC


def qmat(q):  # wxyz (...,4) → (...,3,3)
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_q(R):  # (...,3,3) → wxyz
    from scipy.spatial.transform import Rotation as Rr

    x = Rr.from_matrix(R.reshape(-1, 3, 3)).as_quat()  # xyzw
    return np.concatenate([x[:, 3:], x[:, :3]], 1).reshape(R.shape[:-2] + (4,))


def kabsch(A, Bp):
    ca, cb = A.mean(0), Bp.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def tree_info(root, clip):
    d = next((p for p in (f"{root}/{clip}_pl_rt", f"{root}/{clip}") if os.path.isdir(p)), None)
    if d is None:
        raise SystemExit(f"[OURS_OBJ] FAIL no tree for {clip} under {root}")
    ti = json.loads(Path(f"{d}/task_info.json").read_text())
    z = np.load(f"{d}/0/trajectory_kinematic.npz")
    fr0 = json.loads(Path(f"{d}/F0_REPAIR.json").read_text()) if os.path.exists(f"{d}/F0_REPAIR.json") else {}
    pad = float(ti.get("f0_prepad") or 0) or float(fr0.get("pad_new", 0) or 0)
    rsi = z["retime_src_idx"].astype(float) if "retime_src_idx" in z.files else np.arange(len(z["qpos"]), dtype=float)
    ids = {
        sd: str(os.path.basename(str(ti.get(f"{sd}_object_mesh_dir", ""))).split("_")[0]) for sd in ("right", "left")
    }
    return {
        "dir": d,
        "pad": pad,
        "dt": float(ti["ref_dt"]),
        "rsi": rsi,
        "n": len(z["qpos"]),
        "ids": ids,
        "mesh": {sd: ti[f"{sd}_object_mesh_dir"] for sd in ("right", "left")},
    }


def motion(clip):
    root = workspace.data("chord", "assets", "human_motion_data", "taco", "processed")
    f = glob.glob(f"{root}/sequence_id=*_{_OP.clip_name(clip).split('__')[1]}/robot_name=sharpa_wave/motion_v1.parquet")
    if len(f) != 1:
        raise SystemExit(f"[OURS_OBJ] FAIL motion parquet for {clip}: {len(f)}")
    r = pq.read_table(
        f[0], columns=["fps", "object_body_names", "object_body_position", "object_body_wxyz", "object_mesh_paths"]
    ).to_pylist()[0]
    ids = [os.path.basename(m).split("_")[0] for m in r["object_mesh_paths"]]
    return f[0], r["object_body_names"], ids, S.load_chord_motion_reference(f[0])


def mesh_vertices(path):
    import trimesh

    return np.asarray(trimesh.load(path, force="mesh").vertices, float)


def _reference_times(off, L, z, sidx, names, ti):
    """Per episode the original-motion time of each step (s, the RSI pad removed), from its reference frames."""
    tau = []
    for o, ep_len in zip(off, L, strict=True):
        rf = z["ref_frame"][o : o + ep_len, sidx[names[0]]].astype(int)
        tau.append((ti["rsi"][np.clip(rf, 0, ti["n"] - 1)] - ti["pad"]) * ti["dt"])
    return tau


def _align_reference(L, off, tau, z, sidx, names, ref, full):
    """Rigid map env-local to CHORD world from the longest episode reference, its position and rotation residuals."""
    e0 = int(np.argmax(L))
    o0, l0 = off[e0], L[e0]
    t0 = tau[e0]
    use = t0 >= 0
    A = np.concatenate([z["ref"][o0 : o0 + l0, sidx[n], :3][use] for n in names])
    Bp = np.concatenate([SC.ref_at(*ref["objs"][n], ref["dt"], t0[use])[0] for n in names])
    R, t = kabsch(A, Bp)
    res = np.linalg.norm(A @ R.T + t - Bp, axis=1)
    scored_full = full - int((t0 < 0).sum())  # the pad sits at the front: a full episode's scored steps
    rot = np.concatenate(
        [
            np.degrees(
                np.arccos(
                    np.clip(
                        (
                            np.einsum(
                                "nij,nij->n",
                                R @ qmat(z["ref"][o0 : o0 + l0, sidx[n], 3:7][use]),
                                qmat(SC.ref_at(*ref["objs"][n], ref["dt"], t0[use])[1]),
                            )
                            - 1
                        )
                        / 2,
                        -1,
                        1,
                    )
                )
            )
            for n in names
        ]
    )
    return R, t, res, scored_full, rot


def _mesh_frame_check(pqf, aroot, ti, side_of, names, mids):
    """Our object meshes, and their bbox difference to CHORD mesh of the same TACO id (mm)."""
    ids = S.chord_object_ids(pqf)
    verts = {
        n: mesh_vertices(f"{aroot}/{ti['mesh'][side_of[i]]}/visual.obj") for n, i in zip(names, mids, strict=False)
    }
    frame_chk = {}
    for n in names:  # our mesh vs CHORD's mesh of the same TACO id — same frame expected (bbox in mm)
        c = SC.vertices_urdf(S.object_urdf("chord", ids[n]))
        v = verts[n]
        frame_chk[n] = round(float(max(np.abs(v.min(0) - c.min(0)).max(), np.abs(v.max(0) - c.max(0)).max()) * 1000), 3)
    return verts, frame_chk


def _episode_poses(tau, e, z, o, ep_len, sidx, R, t, names, mode, ref):
    """One episode scored steps: its achieved object poses and the reference poses of the mode."""
    sel = tau[e] >= 0
    te = tau[e][sel]
    ach = np.stack(
        [
            np.concatenate(
                [
                    z["obj"][o : o + ep_len, sidx[n], :3][sel] @ R.T + t,
                    mat_q(R @ qmat(z["obj"][o : o + ep_len, sidx[n], 3:7][sel]))[:, [1, 2, 3, 0]],
                ],
                -1,
            )
            for n in names
        ],
        1,
    )
    if mode == "a":
        rr = [SC.ref_at(*ref["objs"][n], ref["dt"], te) for n in names]
        refp = np.stack([np.concatenate([p, q[:, [1, 2, 3, 0]]], -1) for p, q in rr], 1)
    else:
        refp = np.stack(
            [
                np.concatenate(
                    [
                        z["ref"][o : o + ep_len, sidx[n], :3][sel] @ R.T + t,
                        mat_q(R @ qmat(z["ref"][o : o + ep_len, sidx[n], 3:7][sel]))[:, [1, 2, 3, 0]],
                    ],
                    -1,
                )
                for n in names
            ],
            1,
        )
    return sel, ach, refp


def _score_modes(verts, names, off, L, tau, z, sidx, R, t, ref, scored_full):
    """ADD-AUC and success rates of every episode in mode a (original motion after the rigid map) and b (own ref)."""
    M = SC.chord_metrics_module()
    V = SC.sample_vertices([verts[n] for n in names])
    out = {}
    for mode in ("a", "b"):
        adds, sp, mp = [], [], []
        for e, (o, ep_len) in enumerate(zip(off, L, strict=True)):
            sel, ach, refp = _episode_poses(tau, e, z, o, ep_len, sidx, R, t, names, mode, ref)
            a = M.object_add(ach[:, None], refp, V)[:, 0]  # (steps, B)
            miss = max(0, scored_full - int(sel.sum()))  # scored steps the episode never reached (all at the end)
            if miss > 0:
                a = np.concatenate([a, np.full((miss, len(names)), np.inf)])
            adds.append(a)
            s_ok = bool(M.spider_success(ach[:, None], refp)[0][0]) and miss == 0
            m_ok = bool(M.maniptrans_success(ach[:, None], refp, np.arange(len(names)))[0][0]) and miss == 0
            sp.append(s_ok)
            mp.append(m_ok)
        auc, aucb = M.add_auc(np.concatenate(adds)[:, None, :])
        best, be, per_e = SC.add_auc_best(M, adds)  # the table's value: best of the rollouts (owner score_chord.py)
        out[mode] = {
            "add_auc_best": best,
            "add_auc_best_episode": be,
            "add_auc_per_episode": per_e,
            "add_auc": round(auc, 4),
            "add_auc_per_body": dict(zip(names, [round(x, 4) for x in aucb], strict=False)),
            "spider_sr": float(np.mean(sp)),
            "maniptrans_sr": float(np.mean(mp)),
            "sp_k": int(sum(sp)),
            "mp_k": int(sum(mp)),
        }
    return out


def score_clip(z, k, clip, root, N, aroot, step_dt):
    T = z["ep_table"]
    sides = [str(s) for s in z["side_names"]]
    rows = np.where((T[:, 0] == k) & (T[:, 3] == 0) & (T[:, 4] == 1))[0]
    rows = rows[np.argsort(T[rows, 2], kind="stable")][:N]
    if len(rows) < N:
        return {"clip": clip, "error": f"{len(rows)} valid real-batch episodes < {N}"}
    ti = tree_info(root, clip)
    pqf, names, mids, ref = motion(clip)
    side_of = {ti["ids"][sd]: sd for sd in ("right", "left")}
    if sorted(mids) != sorted(side_of):
        return {"clip": clip, "error": f"object ids parquet {mids} vs tree {ti['ids']}"}
    sidx = {n: sides.index(side_of[i]) for n, i in zip(names, mids, strict=False)}
    off, L = z["ep_offset"][rows], T[rows, 6]
    st0 = z["ep_start_frame"][rows] if "ep_start_frame" in z.files else np.zeros(len(rows), int)
    if len(set(st0.tolist())) != 1:
        return {
            "clip": clip,
            "error": f"episodes start at different tree frames {sorted(set(st0.tolist()))} "
            "— one span per clip expected",
        }
    full = int(round((ti["n"] - 1 - int(st0[0])) * ti["dt"] / step_dt)) + 1
    Lc = L[T[rows, 5] == 1]
    if len(Lc) and np.abs(Lc - full).max() > 1:
        return {
            "clip": clip,
            "error": f"completed episodes run {sorted(set(Lc.tolist()))} steps ≠ the span rule {full}",
        }
    tau = _reference_times(off, L, z, sidx, names, ti)
    # rigid map env-local → CHORD world from the longest episode's reference (identical across episodes)
    R, t, res, scored_full, rot = _align_reference(L, off, tau, z, sidx, names, ref, full)
    verts, frame_chk = _mesh_frame_check(pqf, aroot, ti, side_of, names, mids)
    out = _score_modes(verts, names, off, L, tau, z, sidx, R, t, ref, scored_full)
    return dict(
        clip=clip,
        worlds=N,
        rollouts_rule=f"first {N} valid real-batch episodes by env index",
        score_ref="own_ref",
        score_ref_note="Ours = its env reference (mode b) · mode_a = the original motion after the rigid map (record)",
        mesh_bbox_vs_chord_mm=frame_chk,
        park_complete=int(T[rows, 5].sum()),
        full_steps=full,
        episodes_short=int((full > L).sum()),
        align_pos_mm={"max": round(float(res.max()) * 1000, 2), "median": round(float(np.median(res)) * 1000, 2)},
        align_rot_deg={"max": round(float(rot.max()), 2), "median": round(float(np.median(rot)), 2)},
        tree=ti["dir"],
        pad=ti["pad"],
        parquet=pqf,
        metrics_owner="flash_chord.evaluation.metrics",
        mode_a=out["a"],
        mode_b=out["b"],
        **out["b"],
    )


def main(argv):
    p, root, aroot, od = argv[:4]
    N = int(argv[4]) if len(argv) > 4 else 64
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("schema") != "e123_objtrace/1":
        raise SystemExit(f"[OURS_OBJ] FAIL schema {meta.get('schema')}")
    if not meta.get("step_dt"):
        raise SystemExit("[OURS_OBJ] FAIL meta.step_dt missing — the span rule needs it")
    os.makedirs(od, exist_ok=True)
    n_ok = 0
    for k, clip in enumerate(str(c) for c in z["clip_names"]):
        r = score_clip(z, k, clip, root, N, aroot, float(meta["step_dt"]))
        r["objtrace"] = p
        # the record carries the CLIP name (the tables' key — E123_CLIPS_v1) and keeps the tree name it was scored on
        r["tree_name"] = clip
        r["clip"] = cn = _OP.clip_name(clip)
        with open(f"{od}/{cn}.json.tmp", "w") as fh:
            json.dump(r, fh, indent=1, default=float)
        os.replace(f"{od}/{cn}.json.tmp", f"{od}/{cn}.json")
        n_ok += "error" not in r
        print(
            f"[OURS_OBJ] {clip} "
            + (
                r["error"]
                if "error" in r
                else f"own-ref AUC {r['add_auc']} SP {r['sp_k']}/{N} MP {r['mp_k']}/{N} · orig-motion AUC "
                f"{r['mode_a']['add_auc']} · ref gap max {r['align_pos_mm']['max']} mm"
            ),
            flush=True,
        )
    print(f"[OURS_OBJ] DONE {n_ok}/{len(z['clip_names'])} clips scored → {od}")
    return 0 if n_ok == len(z["clip_names"]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
