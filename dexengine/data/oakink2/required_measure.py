"""Per-root measurement of the OakInk2 intake rule (required.py): source object tracks, MANO hand surfaces and
contact, per-window categories. The thresholds come from the requirement statement (scene_requirement) plus production
slack."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from dexengine.data.oakink2 import raw_base as RO
from dexengine.data.objects import (
    scene_requirement as SR,  # owner of the requirement statement (thresholds, categories)
)
from dexengine.data.readers import mano_conventions as MC

TREE_DT = 0.02  # base tree time axis (50 Hz)


TIP_VIDS = (
    745,
    317,
    444,
    556,
    673,
)  # MANO fingertip vertices, in delivered keypoint order (thumb · index · middle · ring · little)


HAND_ORACLE_MAX_M = 0.004  # self-oracle bound (delivered wrist/fingertips ↔ MANO median distance)


SURF_SPACING_M = 0.0025


NEAR_PAD_M = 0.03  # distance queries only for hand vertices within the object box + this pad


MEDFILT = 5


SIDES = (("lh", "left"), ("rh", "right"))


#: Production threshold = requirement threshold + production slack (both owned by scene_requirement; the audit uses the
#: requirement threshold without slack)
TOUCH_M = SR.TOUCH_M + SR.SLACK_TOUCH_M


TOUCH_MIN_S = SR.TOUCH_MIN_S - SR.SLACK_TOUCH_S


MOVE_M = SR.MOVE_M - SR.SLACK_MOVE_M


ROT_DEG = SR.ROT_DEG - SR.SLACK_ROT_DEG


RECV_M = SR.RECV_M + SR.SLACK_RECV_M


_SURF: dict = {}


_MANO: dict = {}


class RootError(Exception):
    """This root tree cannot be judged; counted, rc 1 (no silent pass)."""


def surface(rid: str):
    """rid → (surface points, cKDTree, centre, box lo, hi) | None (no mesh)."""
    if rid not in _SURF:
        import trimesh
        from scipy.spatial import cKDTree

        from dexengine.data.object_surface import sample_surface_with_normals

        mp = RO.RAW / "obj/object_repair/align_ds" / rid / "model.obj"
        if not mp.is_file():
            _SURF[rid] = None
        else:
            m = trimesh.load(mp, force="mesh", process=False)
            n = int(np.clip(m.area / SURF_SPACING_M**2, 4000, 60000))
            p = np.vstack([sample_surface_with_normals(m, n, seed=0)[0], np.asarray(m.vertices)]).astype(np.float64)
            _SURF[rid] = (p, cKDTree(p), p.mean(0), p.min(0), p.max(0))
    return _SURF[rid]


def mano_model(side: str):
    if side not in _MANO:
        import smplx
        import torch

        from dexengine.core import capacity  # noqa: PLC0415 — threads per process: the owner
        from dexengine.core import paths as P

        torch.set_num_threads(capacity.THREADS)
        d, kw = str(P.mano_model_dir()), {"use_pca": False, "flat_hand_mean": True}
        _MANO[side] = m = smplx.MANO(d, is_rhand=(side == "rh"), **kw)
        if side == "lh":
            MC.negate_left("oakink2", m.shapedirs, smplx.MANO(d, **kw).shapedirs)
    return _MANO[side]


def hand_surface(raw_mano: dict, frames: np.ndarray, side: str, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Frames of the source raw_mano → MANO surface vertices in the base frame (T, 778, 3). Joint 0 = tsl."""
    import torch
    from scipy.spatial.transform import Rotation as Rot

    q = np.stack([raw_mano[k][f"{side}__pose_coeffs"].reshape(16, 4).numpy() for k in frames])
    aa = Rot.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(frames), 16, 3).astype(np.float32)
    be = np.stack([raw_mano[k][f"{side}__betas"].reshape(10).numpy() for k in frames]).astype(np.float32)
    tsl = np.stack([raw_mano[k][f"{side}__tsl"].reshape(3).numpy() for k in frames]).astype(np.float64)
    with torch.no_grad():
        o = mano_model(side)(
            global_orient=torch.from_numpy(aa[:, 0]),
            hand_pose=torch.from_numpy(aa[:, 1:].reshape(-1, 45)),
            betas=torch.from_numpy(be),
            transl=torch.zeros(len(frames), 3),
        )
    v = o.vertices.numpy().astype(np.float64) - o.joints[:, :1].numpy().astype(np.float64) + tsl[:, None]
    return v @ R.T + t


def hand_oracle(ti: dict, V: dict, T: int, start: int) -> dict:
    """Self-oracle: MANO wrist (tsl) and fingertip vertices ↔ delivered keypoints. {side: (wrist, fingertips) median mm}
    · above the bound = RootError."""
    from scipy.spatial.transform import Rotation as Rot

    kp = np.load(RO.ROOT / ti["provenance"]["mano_dir"] / "0/trajectory_keypoints.npz")
    Rz = Rot.from_euler("z", ti["frame"]["yaw_deg"], degrees=True).as_matrix()
    out = {}
    for s, name in SIDES:
        w = kp[f"qpos_wrist_{name}"]
        idx = np.clip(np.arange(T) + start, 0, len(w) - 1)
        tips = kp[f"qpos_finger_{name}"][idx, :, :3] @ Rz.T
        dt = float(np.median(np.linalg.norm(V[s][:, list(TIP_VIDS)] - tips, axis=2)))
        dw = float(np.median(np.linalg.norm(V[s + "_wrist"] - w[idx, :3] @ Rz.T, axis=1)))
        out[name] = (round(dw * 1e3, 2), round(dt * 1e3, 2))
        if max(dw, dt) > HAND_ORACLE_MAX_M:
            raise RootError(f"hand oracle {name}: wrist {dw * 1e3:.1f} · fingertips {dt * 1e3:.1f} mm above the bound")
    return out


def object_track(rid: str, frames: dict, t50: np.ndarray, a0: int, b0: int, R, t, V: dict) -> dict | None:
    """Tree-timeline measurement of one source object (present · surface centre · rotation · min distance per hand).
    Fewer than 10 source frames in the window = None."""
    fr2 = [k for k in range(a0, b0) if k in frames]
    if len(fr2) < 10:
        return None
    pos, Rb = RO.pose_base(frames, fr2, t50, a0, R, t)
    tr = (np.array(fr2) - a0) / RO.MOCAP_HZ
    have = (t50 >= tr[0] - 1e-9) & (t50 <= tr[-1] + 1e-9)
    sf = surface(rid)
    if sf is None:
        return {"mesh": False, "have": have, "c": pos, "R": Rb}
    P_, kd, cen, lo, hi = sf
    dist = {}
    for s, _n in SIDES:
        d = np.full(len(t50), np.inf)
        loc = np.einsum("tvj,tji->tvi", V[s] - pos[:, None, :], Rb)  # hand vertices → object frame
        near = np.all((loc > lo - NEAR_PAD_M) & (loc < hi + NEAR_PAD_M), axis=2) & have[:, None]
        ti_, vi = np.nonzero(near)
        if len(ti_):
            np.minimum.at(d, ti_, kd.query(loc[ti_, vi], k=1)[0])
        dist[s] = d
    return {"mesh": True, "have": have, "c": np.einsum("tij,j->ti", Rb, cen) + pos, "R": Rb, "pos": pos, "dist": dist}


def _medfilt(x: np.ndarray, k: int = MEDFILT) -> np.ndarray:
    if len(x) < k:
        return x
    pad = np.concatenate([x[:1].repeat(k // 2, 0), x, x[-1:].repeat(k // 2, 0)])
    return np.median(np.stack([pad[i : i + len(x)] for i in range(k)]), axis=0)


def _surf_gap(tr: dict, a: str, b: str, i: int) -> tuple[float, np.ndarray]:
    """Minimum surface distance a ↔ b · a's centre in b's frame (sample i)."""
    Pa = surface(a)[0]
    Pa = Pa[:: max(1, len(Pa) // 1500)]
    pw = Pa @ tr[a]["R"][i].T + tr[a]["pos"][i]
    loc = (pw - tr[b]["pos"][i]) @ tr[b]["R"][i]
    return float(surface(b)[1].query(loc, k=1)[0].min()), loc.mean(0)


def window_categories(tr: dict, sel: np.ndarray) -> dict:
    """One window (sel, tree sample mask) → {rid: category} (first match of the scene_requirement rules; independent of
    hand activity)."""
    m = {}
    for rid, o in tr.items():
        ok = sel & o["have"]
        if not o["mesh"] or ok.sum() < 2:
            continue
        idx = np.flatnonzero(ok)
        c = _medfilt(o["c"][idx])
        mv = float(np.linalg.norm(c - c[0], axis=1).max())
        R0 = o["R"][idx[0]]
        rot = float(np.degrees(np.arccos(np.clip((np.einsum("ij,tij->t", R0, o["R"][idx]) - 1) / 2, -1, 1))).max())
        jump = float(np.linalg.norm(np.diff(o["c"][idx], axis=0), axis=1).max()) >= SR.JUMP_M
        ts = {s: float((o["dist"][s][idx] <= TOUCH_M).sum()) * TREE_DT for s, _n in SIDES}
        m[rid] = {
            "i0": int(idx[0]),
            "i1": int(idx[-1]),
            "move": mv,
            "jump": jump,
            "moved": mv >= MOVE_M or rot >= ROT_DEG,
            "any": SR.touched(ts, TOUCH_MIN_S),
        }
    cat = {}
    for rid, r in m.items():
        if r["moved"] and r["any"] and not r["jump"]:
            cat[rid] = "moved"
        elif r["any"]:
            cat[rid] = "touched"
        elif r["move"] >= MOVE_M and not r["any"] and not r["jump"]:
            cat[rid] = "carried"
    movers = [k for k, v in cat.items() if v == "moved"]
    for rid in m:
        if rid in cat:  # receive candidates = every object not chosen yet (production slack)
            continue
        lo, hi = surface(rid)[3], surface(rid)[4]
        for mv_ in movers:
            i0, i1 = m[mv_]["i0"], m[mv_]["i1"]
            if not (tr[rid]["have"][i0] and tr[rid]["have"][i1]):
                continue
            d0, _ = _surf_gap(tr, mv_, rid, i0)
            d1, cm = _surf_gap(tr, mv_, rid, i1)
            if d0 > SR.RECV_M - SR.SLACK_RECV_M and (d1 <= RECV_M or bool(np.all((cm > lo) & (cm < hi)))):
                cat[rid] = "receiver"
                break
    return cat


def windows_of(root_ti: dict, cuts: list, T: int) -> list:
    """Windows of a root (name, sample mask); a piece window (stage 50 Hz segment) outside the root axis is a
    RootError."""
    s0 = int(root_ti["segment"]["start"])
    grid = np.arange(T) + s0
    out = []
    for name, ti in [("<root>", root_ti)] + cuts:
        seg = ti["segment"]
        ws = int(seg["start"])
        we = s0 + T if name == "<root>" else int(seg["end"])
        if name != "<root>" and not (s0 <= ws < we <= s0 + T + 1):
            raise RootError(f"piece {name} window [{ws},{we}) outside the root axis [{s0},{s0 + T})")
        out.append((name, (grid >= ws) & (grid < we)))
    return out


def measure_root(root: str, ti: dict, cuts: list, d: dict, keys: list, base: Path) -> dict:
    """One root → {root, required: {rid: {cat, windows}}, in_tree, nomesh, hand_oracle, oracle}."""
    m = RO.SEQ_KEY.fullmatch(RO.intake_clip(ti))
    stage = int(m.group(4))
    if stage >= len(keys):
        raise RootError(f"stage {stage} ≥ program stage count {len(keys)}")
    ivs_raw = ast.literal_eval(keys[stage])
    ivs = [x for x in ivs_raw if x]
    a0, b0 = min(x[0] for x in ivs), max(x[1] for x in ivs)
    kin = np.load(base / root / "0/trajectory_kinematic.npz")["qpos"]
    T, start = len(kin), int(ti["segment"]["start"])
    t50 = (np.arange(T) + start) * TREE_DT
    R, t = RO.tree_transform(ti)
    res, osrc = RO.oracle_residual(ti, kin, d["obj_transf"], t50, a0, R, t)
    have_f = np.array(sorted(d["raw_mano"]))
    want = np.round(a0 + t50 * RO.MOCAP_HZ).astype(int)
    frames = have_f[np.clip(np.searchsorted(have_f, want), 0, len(have_f) - 1)]
    if np.abs(frames - want).max() > 2:
        raise RootError(f"source MANO frame gap > 2 (max {int(np.abs(frames - want).max())})")
    V = {}
    for s, _n in SIDES:
        V[s] = hand_surface(d["raw_mano"], frames, s, R, t)
        tsl = np.stack([d["raw_mano"][k][f"{s}__tsl"].reshape(3).numpy() for k in frames]).astype(np.float64)
        V[s + "_wrist"] = tsl @ R.T + t
    ho = hand_oracle(ti, V, T, start)
    tr, nomesh = {}, {}
    for rid, fr in d["obj_transf"].items():
        o = object_track(rid, fr, t50, a0, b0, R, t, V)
        if o is None:
            continue
        if not o["mesh"]:
            ok = o["have"]
            mv = float(np.linalg.norm(o["c"][ok] - o["c"][ok][0], axis=1).max()) if ok.sum() > 1 else 0.0
            nomesh[rid] = round(mv, 4)
            continue
        tr[rid] = o
    req: dict = {}
    for name, sel in windows_of(ti, cuts, T):
        for rid, c in window_categories(tr, sel).items():
            e = req.setdefault(rid, {"cat": c, "windows": []})
            e["windows"].append(name)
            if SR.CATEGORIES.index(c) < SR.CATEGORIES.index(e["cat"]):
                e["cat"] = c
    return {
        "root": root,
        "required": req,
        "in_tree": sorted(RO.tree_part_ids(ti)),
        "nomesh": nomesh,
        "hand_oracle": ho,
        "oracle": [round(res * 1e3, 3), osrc],
        "T": T,
        "stage": stage,
        "a0": a0,
        "b0": b0,
    }
