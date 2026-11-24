"""Scene object oracle — the dataset-independent half (one owner of the measurement reduction and the
classification — the requirement thresholds are read from scene_requirement; readers/raw/<ds> modules own only
their raw loader, clip -> window mapping and MANO convention).

A dataset module hands `measure_objects` the sample frames, the posed MANO vertices of both hands and each raw object's
pose stream; reduce_window turns a window of it into per-object scalars and classify sorts them against the tree's
objects[] (tree_scene) — the classes and the spawn rule are written in readers/raw/oakink2's module doc and apply to
every dataset unchanged.
Touch (the `touched` · `moved` categories and a receiver's movers) is the scene's one definition, scene_requirement's
six hand points, measured again on the raw capture (raw.touch). The hand-surface contact (<= TOUCH_M for
>= TOUCH_MIN_S) is a reference column `surf_*` for an object in the tree — never a defect there; an object the tree
does not hold is the intake's question, and the intake reads the hand-surface contact (classify · column touch_by).
`carried_distances` is the MANO-convention oracle every dataset module runs first: while an object translates >=
VAL_V m/s it is carried, so the nearest hand surface must touch it — the convention whose median distance is smallest
wins and the others are its control group.
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects.scene_requirement import (  # noqa: E402,F401
    JUMP_M,
    MOVE_M,
    RECV_M,
    ROT_DEG,
    TOUCH_M,
    TOUCH_MIN_S,
    touched,
)

#: the per-dataset MANO left-model statement (dexengine/data/readers/mano_conventions.py) — the readers pose
#: the delivered hands with it; this oracle reads it too and checks it on its own measurement (left_convention_check)
from dexengine.data.readers import mano_conventions as MC  # noqa: E402

# The requirement thresholds are a statement shared with the intake (oakink2.required) — one owner, scene_requirement
# (no imports there). This oracle reads the values only; its measurement and classification stay its own.
from dexengine.data.readers.raw import touch as TP  # noqa: E402  (six-point touch; also read by record as common.TP)
from dexengine.data.readers.raw.scene_classify import surface

VAL_V = 0.25  # carried-object speed (m/s) for the MANO-convention oracle


class NoParts:
    """Raw stand-in for datasets without an official part tree."""

    def root(self, rid):
        return None


_LAYERS: dict = {}


def mano_layer(mano_dir: str, side: str, dataset: str, *, invert_left: bool = False, **kw):
    """smplx MANO (cached per argument set). The left model is the one `dataset` was fitted with (MC.left_negate —
    the statement; no default). invert_left = the other reading, for left_convention_check only."""
    key = (mano_dir, side, dataset, invert_left, tuple(sorted(kw.items())))
    if key not in _LAYERS:
        import smplx
        import torch

        from dexengine.core import capacity  # noqa: PLC0415 — threads per process: the owner

        torch.set_num_threads(capacity.THREADS)
        m = smplx.MANO(mano_dir, is_rhand=(side == "rh"), **kw)
        if side == "lh":
            MC.negate_left(dataset, m.shapedirs, smplx.MANO(mano_dir, is_rhand=True, **kw).shapedirs)
            if invert_left:
                m.shapedirs[:, 0, :] *= -1
        _LAYERS[key] = m
    return _LAYERS[key]


#: left_convention_check: the stated left model must fit the reference better than the other reading by this (mm,
#: median over every checked frame) — the readings differ by 1–40 mm where shape matters (mano_conventions doc)
LEFT_CHECK_MARGIN_MM = 0.3


def _kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R A + t - B| (A, B (n, 3))."""
    ca, cb = A.mean(0), B.mean(0)
    U, _S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


#: hand_points: the wrist (MANO joint 0) and these fingertip vertices — the points the delivered keypoints carry
#: (qpos_wrist · qpos_finger). The oracle's own measurement choice; both readings share it, so it cannot pick one.
HAND_CHECK_TIPS = [745, 317, 445, 556, 673]


def hand_points(m, global_orient, hand_pose, betas, transl, recentre: bool) -> np.ndarray:
    """(n, 6, 3) wrist + HAND_CHECK_TIPS of smplx MANO `m`. recentre = vertices re-centred on joint 0, then + transl."""
    import torch

    def f(x):
        return torch.tensor(np.asarray(x, float), dtype=torch.float32)

    tr = np.asarray(transl, float)
    with torch.no_grad():
        o = m(
            global_orient=f(global_orient),
            hand_pose=f(hand_pose),
            betas=f(betas),
            transl=f(tr * (0 if recentre else 1)),
        )
    J0 = o.joints[:, :1].double().numpy()
    pts = np.concatenate([J0, o.vertices[:, HAND_CHECK_TIPS].double().numpy()], 1)
    return pts - J0 + tr[:, None] if recentre else pts


def delivered_points(z, sel) -> dict:
    """{rh, lh}: (n, 6, 3) wrist + five fingertips of a delivered trajectory_keypoints.npz at rows `sel`."""
    return {
        s: np.concatenate([z[f"qpos_wrist_{n}"][sel, None, :3], z[f"qpos_finger_{n}"][sel, :, :3]], 1)
        for s, n in (("rh", "right"), ("lh", "left"))
    }


def hand_check_cli(dataset: str, fn) -> int:
    """`python3 readers/raw/<ds>.py --hand-check [--invert] [--seqs N]` — fn(n_seq, invert) runs
    left_convention_check."""
    import argparse

    ap = argparse.ArgumentParser(
        description=f"{dataset}: the left-model statement (mano_conventions) against the "
        "delivered keypoints (scene_oracle_common.left_convention_check)"
    )
    ap.add_argument("--hand-check", action="store_true", required=True)
    ap.add_argument("--invert", action="store_true", help="negative test: the flipped statement must die")
    ap.add_argument("--seqs", type=int, default=12, help="raw sequences, spread over the sorted list")
    ap.add_argument("--selftest", action="store_true", help="the statement must pass and its inversion must die")
    a = ap.parse_args()
    if not a.selftest:
        fn(a.seqs, a.invert)
        return 0
    fn(a.seqs, False)
    try:
        fn(a.seqs, True)
    except SystemExit:
        print(f"[LEFT_CONVENTION] {dataset} selftest PASS (stated passes · inverted dies)", flush=True)
        return 0
    print(f"[LEFT_CONVENTION] {dataset} selftest FAIL: the inverted statement passed", flush=True)
    return 1


def left_convention_check(dataset: str, clips: list, invert: bool = False) -> dict:
    """The oracle's own test of the left-model statement (MC) — independent of it: a reference (the release's labels
    or the delivered keypoints) on the same frames. clips = [(ref_rh, ref_lh, rh, lh_stated, lh_other)], each (n, k, 3):
    the reference is put in the oracle's frame by one rigid fit on the right hand (the control) and the left hand is
    compared under both readings. Dies (SystemExit) unless the stated reading fits better by LEFT_CHECK_MARGIN_MM.
    invert = swap the two readings (the negative test: a flipped statement must die)."""
    e = {"stated": [], "other": [], "right": []}
    for ref_rh, ref_lh, rh, lh_st, lh_ot in clips:
        R, t = _kabsch(ref_rh.reshape(-1, 3), rh.reshape(-1, 3))

        def al(x, R=R, t=t):
            return x @ R.T + t

        e["right"].append(np.linalg.norm(al(ref_rh) - rh, axis=-1).mean(1))
        st, ot = (lh_ot, lh_st) if invert else (lh_st, lh_ot)
        e["stated"].append(np.linalg.norm(al(ref_lh) - st, axis=-1).mean(1))
        e["other"].append(np.linalg.norm(al(ref_lh) - ot, axis=-1).mean(1))
    med = {k: float(np.median(np.concatenate(v))) * 1e3 for k, v in e.items()}
    st_fix = MC.left_fix(dataset) != invert
    out = {
        "dataset": dataset,
        "stated_negated": st_fix,
        "n_clips": len(clips),
        "n_frames": int(sum(len(x) for x in e["right"])),
        "left_stated_mm": round(med["stated"], 2),
        "left_other_mm": round(med["other"], 2),
        "right_mm": round(med["right"], 2),
        "margin_mm": LEFT_CHECK_MARGIN_MM,
    }
    print(
        f"[LEFT_CONVENTION] {dataset} stated_negated={int(st_fix)} left stated {med['stated']:.2f} mm · other "
        f"{med['other']:.2f} mm · right {med['right']:.2f} mm ({out['n_clips']} clips · {out['n_frames']} frames)",
        flush=True,
    )
    if not med["stated"] + LEFT_CHECK_MARGIN_MM <= med["other"]:
        raise SystemExit(
            f"[LEFT_CONVENTION] {dataset}: the stated left model (negated={int(st_fix)}) does not fit "
            f"the reference better than the other reading by {LEFT_CHECK_MARGIN_MM} mm — "
            "mano_conventions is wrong"
        )
    return out


def measure_objects(frames, hv: dict, poses: dict, dt: float, six: dict | None = None) -> dict:
    """st = {frames, dt, objs}: per object {mesh, have, c, R, t, dist{lh, rh}, tipd{lh, rh}, palmd{lh, rh}, mp}.
    poses = {id: (mesh path | None, R (T,3,3), t (T,3), have (T,))}; hv = {"lh"|"rh": (T,778,3) world vertices, NaN
    where the hand is absent}. six = {"lh"|"rh": (fingertips (T,5,3), wrist (T,3))} when hv is not a MANO surface
    (TP.hand_points). dist = hand surface ↔ object surface (the reference columns); tipd (T,5) · palmd (T,) = the six
    points ↔ object surface (the touch — scene_requirement)."""
    pts = TP.hand_points(hv, six, HAND_CHECK_TIPS) if len(frames) else {}  # no sample = every object a no-mesh row
    objs = {}
    for rid, (mp, Rm, tt, have) in poses.items():
        have = np.asarray(have, bool)
        if mp is None or not len(frames):
            objs[rid] = {"mesh": False, "have": have}
            continue
        P, kd, cen, lo, hi = surface(str(mp))
        c = np.einsum("tij,j->ti", Rm, cen) + tt
        dist, tipd, palmd = {}, {}, {}
        for s in ("lh", "rh"):
            dd, d6 = np.full(len(frames), np.inf), np.full((len(frames), 6), np.inf)
            for i in np.flatnonzero(have):
                if np.isnan(hv[s][i, 0, 0]):
                    continue
                v = (hv[s][i] - tt[i]) @ Rm[i]
                near = np.all((v > lo - 0.03) & (v < hi + 0.03), axis=1)
                if near.any():
                    dd[i] = kd.query(v[near], k=1)[0].min()
                if not np.isnan(pts[s][i]).any():
                    d6[i] = kd.query((pts[s][i] - tt[i]) @ Rm[i], k=1)[0]
            dist[s], tipd[s], palmd[s] = dd, d6[:, :5], d6[:, 5]
        objs[rid] = {
            "mesh": True,
            "have": have,
            "c": c,
            "R": Rm,
            "t": tt,
            "dist": dist,
            "tipd": tipd,
            "palmd": palmd,
            "mp": str(mp),
        }
    return {"frames": np.asarray(frames, int), "dt": float(dt), "objs": objs}


def carried_distances(st: dict, t_s, v_min: float = VAL_V) -> np.ndarray:
    """MANO-convention oracle: min hand-surface distance (m) on the samples where an object's centroid moves at
    >= VAL_V m/s (and < 3 m/s — not a tracking jump). t_s = sample times (s). No vertex near = 0.08 (far)."""
    out = []
    for o in st["objs"].values():
        if not o["mesh"]:
            continue
        v = np.r_[0, np.linalg.norm(np.diff(o["c"], axis=0), axis=1) / np.maximum(np.diff(t_s), 1e-3)]
        sel = (v >= v_min) & (v < 3.0) & o["have"]
        d = np.minimum(o["dist"]["lh"], o["dist"]["rh"])[sel]
        out += list(np.where(np.isinf(d), 0.08, d))
    return np.asarray(out)


def summarize_distances(d: np.ndarray) -> dict:
    d = np.asarray(d, float)
    if not d.size:
        return {"n": 0}
    return {
        "n": int(d.size),
        "med_mm": round(float(np.median(d)) * 1e3, 1),
        "p90_mm": round(float(np.percentile(d, 90)) * 1e3, 1),
        "within10mm": round(float((d <= 0.01).mean()), 3),
    }


# ---------------------------------------------------------------- statement side (tree)


# ---------------------------------------------------------------- driver


def segment_window(clip_re, seq: dict, clip: str, ti: dict | None, fps: float, tree_hz: float, whole: str):
    """The sequence-wide window, or the tree's segment for a `_seg`/`_cut` clip (raw frames at `fps`, tree frames at
    `tree_hz`). Returns the dataset `window` tuple: (stage span, w0, w1, active hands, declared, how)."""
    m = clip_re.fullmatch(clip)
    w0, w1, how = 0, seq["T"], whole
    if m.group("seg") or m.group("cut"):
        seg = (ti or {}).get("segment") or {}
        if seg.get("start") is None or seg.get("end") is None:
            raise ValueError("segment/cut clip without tree segment.start/end")
        w0, w1, how = int(round(seg["start"] * fps / tree_hz)), int(round(seg["end"] * fps / tree_hz)), "segment"
    return (0, seq["T"]), w0, w1, {"lh": True, "rh": True}, set(), how
