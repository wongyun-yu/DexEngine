#!/usr/bin/env python3
"""HOT3D desk height recovery: finds the desk surface from resting objects over the whole source recording, recovers
each clip's grounding offset and writes desk_z (D10-h).

Why: 81 % of the contact-segmented clips (3,741/4,627) start with the object already held (support_audit cls=none →
    table_unknown) and drop out under first-round rule D4. The source world is fixed within a recording and tabletop
    objects rest at the same height, so there is **one desk surface per recording**. The source world's up axis differs
    per device (Aria +z · Quest3 +y; owner dexengine/data/readers/hot3d.py HEADSET_UP), and a Quest3 recording listed in
    configs/datasets/hot3d_up_level.tsv is further levelled on its desk plane (source_frames.recording_up), so source
    objects are turned with the rotation the reader wrote into clip task_info `source_frame.R_to_z_up` and z is read as
    height (fails without it).
Method: ⓪ the desk surface and footprint are decided by the one **work surface rule** (owner
        dexengine/data/support/work_surface.py, dataset-independent). This file only reads the source and fits clips to
        the source world.
        branch A (placement evidence) desk surface = the **lowest surface** among those the handled objects (main
          objects + objects moved within the window) and their receivers rest on within or around the clip (receiver =
          a still object under the placement by ground-truth heights on surface samples, containers included —
          work_surface.receivers_of). Objects resting higher stand on a pedestal (coords.compute_desk_support) ·
          untouched static objects below it are not spawned (extra_slots below_work_surface).
        branch B (no evidence, in hand throughout) HOT3D has no measured desk, so the given desk = the surface the main
          object rested on most recently (otherwise the per-object · per-recording mode from ① below, confidence low).
          When lifted objects or holding hands inside the footprint go below it, it is lowered below their lowest point
          (not_buried). ① serves only as that fallback and for diagnostics.
        footprint = convex hull of the footprints of objects resting on the desk surface in that recording (clip-frame
          XY): INTAKE `work_footprint_raw` · basis `work_footprint_basis` (placements · objects · clip-frame height z of
          the surface the hull was measured on; for clips lowered in branch B it is above desk_z, since the hull belongs
          to the given surface before lowering). A statement only, with no consumers (`desk_footprint_raw` is not
          written; that field turns D14′ on).
        ① per recording table_z_world = median of the modal bin (3 cm bins) of object bottoms (lowest mesh z) at rest
          (speed < V_REST) and away from hands; floor/shelf objects (≈−1.05) fall in other bins and drop out.
        ② clip dz = source main object z (clip start frame) − delivered clip object z (frame 0) (how far write_sequence
          lowered the first-frame object bottom to 0)
        ③ desk_z(clip) = table_z_world − dz. Check: lowest hand ≥ desk_z − 2 cm, lowest object ≥ desk_z − 1 cm →
          confidence high / medium (≤5 cm) / low.
Inputs: data/hot3d/raw/dataset/<seq>/{dynamic_objects.csv, mano_hand_pose_trajectory.jsonl}, assets/{<uid>.glb,
instance.json} · data/hot3d/mano/bimanual/<clip> · INTAKE/support_audit.json
Output: <out>/hot3d_desk_recover.{json,md}; with --write-intake, desk_mode=table_recovered · desk_z ·
work_surface_branch ·
        work_footprint_raw · work_footprint_basis are overwritten in clips[*] of INTAKE support_audit.json (high/medium
            only).
Usage: $RETARGET_PY dexengine/data/support/hot3d.py --out <out dir> [--write-intake] [--limit
N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import InputMissingError
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)
from dexengine.data.objects import recv_surface as RS  # owner of the surface samples (vertices + face grid)
from dexengine.data.readers import source_frames as SF  # noqa: E402  (the reader's parsing, device up axis, rotation)
from dexengine.data.support import hot3d_outputs as HO
from dexengine.data.support import (
    work_surface as WS,  # noqa: E402 — owner of the work surface rule (desk surface · footprint · branches)
)
from dexengine.data.support.bands import HIGH_REST_M

RAW = _data_root() / "data/hot3d/raw/dataset"
BIN_M, HAND_TOL, OBJ_TOL, MED_TOL, MIN_STILL = 0.03, 0.02, 0.01, 0.05, 30
#: Allowed up-axis mismatch of the clip ↔ source world fit (max |third row of R − ez|); the reader only moves the +z-up
#: world horizontally. Beyond it the clip's hand points and footprint are not used (main-object pose samples do not
#: match clip frame 0).
UP_TOL = 0.01
#: Sample step for measuring rest (hz of the work surface rule): one source object sample step is read as 1/30 s
#: (samples arrive several rows per frame at nearly the same time · rest window 15 samples).
STEP_HZ = 30.0
_MESH: dict[str, np.ndarray] = {}
_FULL: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_SURF: dict[str, np.ndarray] = {}


def _glb(uid: str) -> tuple[np.ndarray, np.ndarray]:
    """(body-frame vertices, triangles) of the whole mesh, read once."""
    if uid not in _FULL:
        m = trimesh.load(str(RAW / "assets" / f"{uid}.glb"), force="mesh", process=False)
        _FULL[uid] = (np.asarray(m.vertices, float), np.asarray(m.faces, int))
    return _FULL[uid]


def full_verts(uid: str) -> np.ndarray:
    """Body-frame vertices of the whole mesh — what a receiver's XY hull and bottom z read (work_surface.receivers_of
    `floor`). The 2000-vertex subsample sits up to 0.2 mm above the lowest vertex, and the lowest vertex lies on a real
    bottom face (measured on HOT3D: 16 of 16 receivers whose bottom moved)."""
    return _glb(uid)[0]


def surface_pts(uid: str) -> np.ndarray:
    """Body-frame surface samples of the whole mesh (recv_surface.surface_points: the vertices plus a face grid) — what
    the receiver rule's heights read (work_surface.receivers_of `surface`)."""
    if uid not in _SURF:
        _SURF[uid] = RS.surface_points(*_glb(uid))
    return _SURF[uid]


def mesh_pts(uid: str, n=2000) -> np.ndarray:
    if uid not in _MESH:
        v = full_verts(uid)
        _MESH[uid] = v[np.linspace(0, len(v) - 1, min(n, len(v))).astype(int)]
    return _MESH[uid]


def source_rotation(ti: dict, clip: str) -> np.ndarray:
    """Source → +z-up rotation the reader **wrote** as applied to this clip (task_info `source_frame.R_to_z_up`).
    Missing
    or not a proper rotation fails: output of a reader from before the up-axis rule would make Quest3 clips' heights a
    horizontal axis. A statement checked against the rule (`read_objects`' SF.world_up) as an oracle."""
    m = np.asarray(((ti.get("source_frame") or {}).get("R_to_z_up")) or np.zeros((0,)), float)
    if m.shape != (3, 3) or not np.allclose(m @ m.T, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(m), 1.0):
        raise InputMissingError(
            f"[hot3d_desk_recover] {clip}: no source_frame.R_to_z_up (reader up-axis rule) in "
            "task_info — output of a reader from before the up-axis rule. Rebuild the mano layer "
            "with the reader, then rerun"
        )
    return m


def read_objects(seq: str) -> tuple[dict, np.ndarray]:
    """({uid: (N,8) [t x y z qw qx qy qz]} in the same +z-up world as the reader, that rotation); source parsing, device
    up axis and rotation all call the reader's owner (SF = dexengine/data/readers/source_frames.py)."""
    rot = SF.recording_up(json.loads((RAW / seq / "metadata.json").read_text()).get("headset"), seq)[1]
    return SF.to_z_up(SF.object_samples(RAW / seq), rot), rot


def frame_ts(seq: str) -> np.ndarray:
    """30 fps frame times (ns) with the same dedupe rule as the reader (hot3d.py `_frames`)."""
    ts, seen = [], set()
    with open(RAW / seq / "mano_hand_pose_trajectory.jsonl") as f:
        for line in f:
            r = json.loads(line)
            k = int(round(r["timestamp_ns"] / 1e6))
            if k in seen:
                continue
            seen.add(k)
            ts.append(int(r["timestamp_ns"]))
    return np.sort(np.asarray(ts))


def _mode_z(b: np.ndarray):
    edges = np.arange(b.min() - BIN_M, b.max() + 2 * BIN_M, BIN_M)
    h, _ = np.histogram(b, edges)
    k = int(np.argmax(h))
    sel = b[(b >= edges[k] - BIN_M) & (b < edges[k + 1] + BIN_M)]
    return (
        float(np.median(sel)),
        float(h[k] / len(b)),
        {round(float(edges[i]), 2): int(h[i]) for i in range(len(h)) if h[i]},
    )


def clip_to_world(kp, side: str, a: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid transform clip (mano layer) ↔ source world, matching the main object's clip frame-0 pose to source sample
    j.
    Returns (R, t, up-axis mismatch): p_clip = R p_world + t."""
    oc = kp[f"qpos_obj_{side}"][0]
    R = Rot.from_quat(oc[[4, 5, 6, 3]]).as_matrix() @ Rot.from_quat(a[j, [5, 6, 7, 4]]).as_matrix().T
    return R, oc[:3] - R @ a[j, 1:4], float(np.abs(R[2] - np.array([0.0, 0.0, 1.0])).max())


def hands_world(kp, R: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """Hand keypoints (wrist + fingers) (T, K, 3) in the source world; a side whose wrist is all zeros (missing hand) is
    dropped. None without hands."""
    rows = []
    for x in ("right", "left"):
        if f"qpos_wrist_{x}" not in kp.files or np.abs(kp[f"qpos_wrist_{x}"][:, :3]).sum() == 0:
            continue
        fs = [kp[k][..., :3] for k in (f"qpos_finger_{x}", f"qpos_finger_distal_{x}") if k in kp.files]
        rows.append((np.concatenate([kp[f"qpos_wrist_{x}"][:, None, :3]] + fs, 1) - t) @ R)
    return np.concatenate(rows, 1) if rows else None


def table_z_world(objs: dict[str, np.ndarray]) -> tuple[float | None, dict]:
    """Median of the modal bin (3 cm) of resting object bottoms z in the recording + the **per-object** modal rest
    bottom
    (desk, floor and shelf mix within a recording, so each object rests on its own surface; P0001_10a27bf7: 5 objects
    −0.72, dumbbell −1.05). A clip uses its main object's surface, or the recording mode when that object has fewer than
    MIN_STILL rest samples."""
    bottoms = []
    per_obj = {}
    for uid, a in objs.items():
        p = a[:, 1:4]
        idx = np.where(WS.still_mask(a, STEP_HZ))[0][::10]
        if not len(idx):
            continue
        pts = mesh_pts(uid)
        bo = []
        for t in idx:
            R = Rot.from_quat(a[t, [5, 6, 7, 4]]).as_matrix()
            bo.append(float((pts @ R.T + p[t])[:, 2].min()))
        bottoms += bo
        if len(bo) >= MIN_STILL:
            z, frac, _ = _mode_z(np.asarray(bo))
            per_obj[uid] = {"z": round(z, 4), "frac": round(frac, 3), "n": len(bo)}
    if not bottoms:
        return None, {}
    z, frac, bins = _mode_z(np.asarray(bottoms))
    return z, {"n_still": int(len(bottoms)), "mode_frac": round(frac, 3), "bins": bins, "per_obj": per_obj}


def clip_surface(
    objs: dict, rest_rows: list, diag: dict, tz: float, uid: str, mains: list, j: int, kp, side: str, t_ns, t_end
) -> tuple:
    """Desk surface (source world z) · branch · footprint of one clip, through the work surface rule (WS). uid = the
    main object whose grounding offset is measured · j = its clip start sample. Returns (desk z, surface source, time
    gap
    to the placement s, branch, branch A result | None, z of the surface the main object rested on most recently |
    None, depth lowered in branch B, footprint dict (z_world = the surface it was measured on), footprint hull
    (clip-frame
    XY) | None, up-axis mismatch of the fit)."""
    a_ = objs[uid]
    near = WS.nearest_rest(a_, mesh_pts(uid), t_ns, t_end, STEP_HZ)  # comparison · branch B's given desk
    ws = WS.work_surface(
        objs, mesh_pts, mains, t_ns, t_end, STEP_HZ, surface=surface_pts, floor=full_verts
    )  # branch A: lowest surface of handled and receiving objects (receivers: ground-truth heights on surface samples)
    if ws is not None:
        rz, surf, gap_s = ws["z"], f"{ws['how']}:{'main' if ws['uid'] == uid else 'other'}", ws["gap_s"]
    elif near:  # branch B: no placement attached to the clip (starts or ends in hand)
        rz, surf, gap_s = near[0], f"{near[1]}:main_near", near[2]
    else:
        po = diag.get("per_obj", {}).get(uid)
        rz, surf, gap_s = (po["z"] if po else tz), ("object_mode" if po else "recording_mode"), None
    R_cw, t_cw, up_err = clip_to_world(kp, side, a_, j)
    fit = up_err <= UP_TOL
    hands = hands_world(kp, R_cw, t_cw) if fit else None
    fp = WS.footprint(rest_rows, objs, mesh_pts, mains, t_ns, t_end, rz, hands) | {
        "z_world": rz
    }  # measured on the given surface
    lowered = 0.0
    if ws is None:
        rz, lowered = WS.not_buried(objs, mesh_pts, mains, t_ns, t_end, rz, fp["eq"], hands)
    hull_c = None if fp["hull"] is None or not fit else np.round(fp["hull"] @ R_cw[:2, :2].T + t_cw[:2], 4).tolist()
    return (
        rz,
        surf,
        gap_s,
        WS.BRANCH_PLACED if ws is not None else WS.BRANCH_GIVEN,
        ws,
        near[0] if near else None,
        lowered,
        fp,
        hull_c,
        up_err,
    )


def _new_stats() -> dict:
    """Counters of one run (clips · skips · confidence · desk-surface sources · work-surface branches · footprint)."""
    S = {"clips": 0, "no_src": 0, "no_seq": 0, "no_obj": 0, "recovered": 0, "conf": defaultdict(int)}
    S |= dict.fromkeys(
        ("src_same_as_main", "src_lower_interaction", "src_fallback", "src_other"), 0
    )  # desk surface sources
    S |= dict.fromkeys(
        (
            "branch_" + WS.BRANCH_PLACED,
            "branch_" + WS.BRANCH_GIVEN,
            "given_lowered",
            "footprint",
            "footprint_none",
            "frame_fit_off",
            "conflict_obj",
            "conflict_hand",
        ),
        0,
    )  # branches · footprint
    return S


def _cache_recording(seq: str, per_seq: dict) -> None:
    """Read one recording once: objects, frame times, mode table height, rotation and rest windows."""
    if seq not in per_seq:
        objs, rot = read_objects(seq)
        tz, diag = table_z_world(objs)
        per_seq[seq] = (objs, frame_ts(seq), tz, diag, rot, WS.rests(objs, mesh_pts, STEP_HZ))


def _locate_clip(clip, src_of, emb_of, mano_root_of, name2uid, per_seq, S) -> tuple | None:
    """Source recording, task info, keypoints and main object of one clip (None = skipped; reason counted in S)."""
    src = src_of.get(clip)
    if not src:
        S["no_src"] += 1
        return None
    seq, span = src.split("__s", 1)
    start = int(span.split("_e")[0])
    mano_root = mano_root_of[emb_of.get(clip, "bimanual")]
    ti = json.loads((mano_root / clip / "task_info.json").read_text())
    declared = source_rotation(ti, clip)
    _cache_recording(seq, per_seq)
    objs, ts, tz, diag, rot, rest_rows = per_seq[seq]
    if not np.allclose(declared, rot, atol=1e-9):
        raise SystemExit(
            f"[hot3d_desk_recover] {clip}: rotation written by the reader ≠ rotation of the current rule "
            f"(recording {seq}) — "
            "output of a reader with another rule. Rebuild the mano layer with the reader, then rerun"
        )
    if tz is None:
        S["no_seq"] += 1
        return None
    kp = np.load(mano_root / clip / "0/trajectory_keypoints.npz")
    side = next((s for s in ("right", "left") if ti.get(f"{s}_object_mesh_dir")), None)
    if side is None:
        S["no_obj"] += 1
        return None
    uid = name2uid.get(Path(ti[f"{side}_object_mesh_dir"]).name)
    if uid is None or uid not in objs:
        S["no_obj"] += 1
        return None
    return seq, span, start, ti, kp, side, uid, objs, ts, tz, diag, rest_rows


def _desk_record(seq, start, uid, rec, diag, tz, a_, j, kp, side, cs) -> dict:
    """Desk height record of one clip from its work surface (cs = clip_surface result)."""
    rz, surf, gap_s, branch, ws, old_z, lowered, fp, hull_c, up_err = cs
    tz_use = rz
    dz = float(a_[j, 3] - kp[f"qpos_obj_{side}"][0, 2])
    desk_surf = tz_use - dz
    hz = rec.get("hand_zmin")
    oz = rec.get("obj_zmin")
    # Desk = the surface objects rest on (work_surface); putting something down on a lower surface within the clip
    # is caught by a rest window inside the window ('inside'). Frames going below that surface without resting (held
    # beside the table, tracking jitter) do not lower the desk: object ↔ top penetration is measured by D24
    # (env_dip_m reports its depth) and hands below the desk are D14's. confidence looks only at **surface
    # evidence** (time to the rest window).
    desk_z = desk_surf
    dip = max(0.0, desk_surf - oz) if oz is not None else 0.0
    viol = max(0.0, (desk_z - HAND_TOL) - hz if hz is not None else 0.0)
    conf = "high" if (gap_s is not None and gap_s <= 2.0) else ("medium" if gap_s is not None else "low")
    r = {
        "seq": seq,
        "start": start,
        "uid": uid,
        "table_z_world": round(tz_use, 4),
        "surface": surf,
        "surface_gap_s": gap_s,
        "recording_mode_z": round(tz, 4),
        "dz": round(dz, 4),
        "desk_surface_z": round(desk_surf, 4),
        "desk_z": round(desk_z, 4),
        "confidence": conf,
        "hand_below_desk_m": round(viol, 4),
        "mode_frac": diag.get("mode_frac"),
        "prev_desk_mode": rec.get("desk_mode"),
        "prev_desk_z": rec.get("desk_z"),
        "main_surface_z": None if old_z is None else round(old_z, 4),
        "obj_dip_m": round(dip, 4),
        "work_surface": None if ws is None else {k: ws[k] for k in ("uid", "how", "n_interaction", "surfaces")},
        "branch": branch,
        "given_lowered_m": round(lowered, 4),
        "frame_up_err": round(up_err, 5),
        "work_footprint_raw": hull_c,
        "footprint": {
            k: fp[k] for k in ("n_rests", "n_objects", "obj_below", "hand_below", "obj_conflict", "hand_conflict")
        }
        | {"z": round(fp["z_world"] - dz, 4)},
    }
    return r


def _count_recovery(S: dict, cs: tuple, conf: str) -> None:
    """Count one recovered clip in S: branch, footprint, frame fit, conflicts, desk-surface source, confidence."""
    rz, _, _, branch, ws, old_z, lowered, fp, hull_c, up_err = cs
    for k, on in (
        ("branch_" + branch, True),
        ("given_lowered", lowered > 0),
        ("frame_fit_off", up_err > UP_TOL),
        ("conflict_obj", fp["obj_conflict"] > 0),
        ("conflict_hand", fp["hand_conflict"] > 0),
    ):
        S[k] += on
    S["footprint" if hull_c else "footprint_none"] += 1
    # fallback = no attached placement (surface the main object rested on most recently · mode)
    src = (
        "fallback"
        if ws is None
        else "other"
        if old_z is None
        else "same_as_main"
        if abs(rz - old_z) <= HIGH_REST_M
        else "lower_interaction"
        if rz < old_z
        else "other"
    )
    S["src_" + src] += 1
    S["recovered"] += 1
    S["conf"][conf] += 1


def _recover_clip(clip, rec, src_of, emb_of, mano_root_of, name2uid, per_seq, S) -> dict | None:
    """Desk height of one clip from the raw recording (None = skipped; the reason is counted in S)."""
    found = _locate_clip(clip, src_of, emb_of, mano_root_of, name2uid, per_seq, S)
    if found is None:
        return None
    seq, span, start, ti, kp, side, uid, objs, ts, tz, diag, rest_rows = found
    t_ns = ts[min(start, len(ts) - 1)]
    a_ = objs[uid]
    j = int(np.argmin(np.abs(a_[:, 0] - t_ns)))
    end = int(span.split("_e")[1])
    t_end = ts[min(end, len(ts) - 1)]
    mains = [
        name2uid.get(Path(ti[f"{s_}_object_mesh_dir"]).name)
        for s_ in ("right", "left")
        if ti.get(f"{s_}_object_mesh_dir")
    ]
    cs = clip_surface(objs, rest_rows, diag, tz, uid, [m for m in mains if m], j, kp, side, t_ns, t_end)
    r = _desk_record(seq, start, uid, rec, diag, tz, a_, j, kp, side, cs)
    _count_recovery(S, cs, r["confidence"])
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-intake", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inst = json.loads((RAW / "assets/instance.json").read_text())  # HOT3D raw tree (external data)
    name2uid = {v["instance_name"]: k for k, v in inst.items()}
    ip = P.layer_path("intake", "hot3d") / "support_audit.json"
    SA = json.loads(ip.read_text())
    clips = SA["clips"]
    hand = json.loads(next((P.layer_path("intake", "hot3d")).glob("hot3d_self_*.json")).read_text())["clips"]
    src_of = {v["clip_id"]: v["source_task"] for v in hand.values() if "clip_id" in v}
    emb_of = {
        v["clip_id"]: v["emb"] for v in hand.values() if "clip_id" in v
    }  # 6 right-hand-only clips live in mano/right
    mano_root_of = {e: P.layer_path("mano", "hot3d", emb=e) for e in ("bimanual", "right", "left")}
    per_seq: dict[str, tuple] = {}
    results = {}
    S = _new_stats()
    for clip, rec in sorted(clips.items()):
        if a.limit and S["clips"] >= a.limit:
            break
        S["clips"] += 1
        r = _recover_clip(clip, rec, src_of, emb_of, mano_root_of, name2uid, per_seq, S)
        if r is not None:
            results[clip] = r
    HO.report(out, S, per_seq, results, t0)
    if a.write_intake and results:
        HO.write_intake(SA, clips, results, S, ip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
