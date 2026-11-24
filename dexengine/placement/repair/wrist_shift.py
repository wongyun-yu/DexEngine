#!/usr/bin/env python3
"""Moving the hand of a reference tree and writing it back — the one implementation the two reference repairs share:
the bake's cross repair (cross_repair.py) and the base-layer wrist-housing <-> desk repair (wrist_desk.py).

What lives here (each piece has exactly one implementation; neither caller keeps a copy):
  time profile   ramp_weights · smooth_ramp — 3-point smoothing inside the repaired episodes, a C¹ (cos) ramp of
                 ``W`` frames outside them, the spawn frame never cut by the smoothing · ramp_dilate — per-frame
                 requirements as one profile with the same ramp (no one-frame steps).
  backup / undo  source_of · mark_post · revert — one fingerprinted backup per file and tag (``<f>.<tag>.npz`` +
                 ``<f>.<tag>.meta.json`` holding the sha256 of what the repair wrote). A re-run recomputes from the
                 backup while the file is still the repair's own output; a changed upstream re-backs-up. ``revert`` is
                 the one way back to the original. Repairs that rewrite the same base-tree files run in the order
                 CHAIN; ``view`` peels the later repairs that still sit unchanged on top, so an earlier repair reads
                 the file as it left it (its re-run neither stacks on a later output nor takes it for an upstream
                 change) and reverting it reverts the later ones first.
  writing        shift_row — a translation of the hand root (+ fingers, + a carried object) on one qpos row ·
                 apply_delta · save_npz — the new qpos, the same delta on ``qpos_rollout``, the time derivative of the
                 delta on the named qvel columns (``gradient`` = central, the bake's rule; ``forward`` = the base
                 layer's rule, base_adapter's mj_differentiatePos), atomic replace (a new inode — the base layer
                 hard-links files with other layers, an in-place write would change them too).
  palm → EE      ee_target — the arm IK target of a palm pose (Rz(−90°)·(palm + R·palm_off) + scene z offset).
  finger range   finger_ranges — the reference finger joint range = the vendor range clipped by the retargeting IK's
                 hyperextension rule (owner dexengine/retarget/ik/prior.py — the IK imports it — reached
                 through its window ``dexengine.retarget.ik.hyperext_limits``, which runs in the bake/judge interpreter
                 too), then narrowed to the IK's branch windows (prior ``branch_window_deg``, ik/branch.py — the
                 window exports them as their own key; the sim asset does not read them). The f0 pre-pad, the
                 cross-repair finger box (cross_sets.HandLayout), the wrist repair's finger re-solve and the finger
                 range gate read it — a pose the retargeting cut away is never reintroduced by a repair.
  finger re-IK   finger_ik — the finger joints only (damped least squares, clamped to finger_ranges) put the chosen
                 fingertip sites on their targets; returns the worst residual.
  root move      move_root — applies a rigid transform x → R·x + c to a floating hand root by solving its six root
                 joints (MuJoCo body Jacobian) and checks the round trip (ROOT_TOL_M). A solve that does not
                 converge restarts from the reference it is given (the previous frame's answer; none → the row itself)
                 with bounded, damped steps; an answer whole turns away from the row is brought back (same pose).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from dexengine.placement.collision import det_math as DM
from dexengine.placement.collision.cross_sets import RZ_NEG90  # Rz(−90°) tree (capture) -> scene

ROOT_TOL_M = 1e-6  # root move round trip: position (m) and rotation (rad) residual allowed
ROOT_STEP_MAX_RAD = 0.5  # continued root solve: largest change of one root rotation coordinate in one step
ROOT_TURN_REPORT_DEG = 20.0  # root audit: a neighbour-frame real turn above this is counted (never judged)
ROOT_DAMP = 0.1  # continued root solve: damping = this × squared error (gone at the answer)
WRIST_DESK_TAG = "pre_ws"  # backup tag of stage 2e (wrist housing ↔ desk; placement/repair/wrist_desk.py)
#: Tag of a 2f repair output a tree may still hold (stage 2f writes none now): `view` peels it for 2e, and
#: placement/repair/hand_clear_stage refuses such a tree (base_write.undo_tree restores it).
HAND_CLEAR_TAG = "pre_hc"
#: The base-tree repairs that rewrite the same files, upstream first (stage-1 order 2e → 2f).
CHAIN = (WRIST_DESK_TAG, HAND_CLEAR_TAG)
_RANGES: dict = {}  # (spec dir, vendor ranges) → reference ranges (one window call per process and key)
_SERVER_KEEP = ("_RANGES",)  # pure cache (hand-keyed, clip-independent) — the resident worker keeps it


# ── time profile ────────────────────────────────────────────────────────────────────────────────────────────────


def ramp_weights(T: int, P: np.ndarray, W: int, det: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """P(bool T) → (w: per-frame ramp weight, src: per-frame index of the P frame it refers to). Inside the episode
    w=1, outside it is a cos ramp. det=True takes the cosine from det_math (the bits are the same on every host; numpy's
    may come from SVML on an AVX512 host) — the cross repair; the default keeps wrist_desk's bits."""
    w = np.zeros(T)
    src = np.full(T, -1, int)
    idx = np.where(P)[0]
    if len(idx) == 0:
        return w, src
    w[idx] = 1.0
    src[idx] = idx
    for t in range(T):
        if P[t]:
            continue
        k = int(np.min(np.abs(idx - t)))
        j = int(idx[np.argmin(np.abs(idx - t))])
        if k <= W:
            c = float(DM.sincos(np.pi * k / (W + 1))[1]) if det else np.cos(np.pi * k / (W + 1))
            w[t] = 0.5 * (1.0 + c)
            src[t] = j
    return w, src


def smooth_ramp(arrs: list, need: np.ndarray, W: int, det: bool = False) -> np.ndarray:
    """Per-frame displacement arrays (T, …) → in place: 3-point smoothing over the ``need`` frames (ends one-sided,
    the spawn frame kept as solved), then every frame takes its source frame's value times the ramp weight.
    Returns the weights (w > 0 = touched). det=True (the cross repair): det_math cosine (ramp_weights) and the
    neighbour mean summed left to right; the default keeps wrist_desk's bits."""
    T = len(need)
    w, src = ramp_weights(T, need, W, det=det)
    P = np.where(need)[0]
    for arr in arrs:
        sm = arr.copy()
        for t in P:
            nb = [u for u in (t - 1, t, t + 1) if 0 <= u < T and need[u]]
            if det:
                acc = arr[nb[0]].copy()
                for u in nb[1:]:
                    acc = acc + arr[u]
                sm[t] = acc / float(len(nb))
            else:
                sm[t] = arr[nb].mean(axis=0)
        if need[0]:
            sm[0] = arr[0]
        out = np.zeros_like(arr)
        for t in range(T):
            if src[t] >= 0:
                out[t] = sm[src[t]] * w[t]
        arr[...] = out
    return w


def ramp_dilate(e: np.ndarray, W: int) -> np.ndarray:
    """Per-frame requirements e (T,) ≥ 0 → the least profile that meets each with the ramp of ramp_weights around it:
    E[t] = max over t' of e[t']·k(|t − t'|), k(0) = 1, k(d) = ½(1 + cos(π·d/(W + 1))) for d ≤ W, 0 beyond. Continuous —
    no step larger than the ramp's own (a lone requirement never becomes a one-frame jump)."""
    e = np.asarray(e, float)
    out = e.copy()
    for d in range(1, W + 1):
        k = 0.5 * (1.0 + np.cos(np.pi * d / (W + 1)))
        out[d:] = np.maximum(out[d:], k * e[: len(e) - d])
        out[: len(e) - d] = np.maximum(out[: len(e) - d], k * e[d:])
    return out


# ── backup / undo ───────────────────────────────────────────────────────────────────────────────────────────────


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bk(cur: Path, tag: str) -> tuple[Path, Path]:
    return cur.with_suffix(f".{tag}.npz"), cur.with_suffix(f".{tag}.meta.json")


def _post(meta: Path):
    return json.loads(meta.read_text()).get("post_sha256")


def _later(tag: str) -> tuple:
    return CHAIN[CHAIN.index(tag) + 1 :] if tag in CHAIN else ()


def view(cur: Path, tag: str) -> Path:
    """The file as repair ``tag`` left it (read only): the later repairs of CHAIN that still sit on top unchanged (the
    recorded post sha equals what is there) peeled off, the last first. A tag outside CHAIN has none: ``cur``."""
    cur = Path(cur)
    f = cur
    for t in reversed(_later(tag)):
        bk, meta = _bk(cur, t)
        if bk.is_file() and meta.is_file() and _post(meta) == _sha(f):
            f = bk
    return f


def original(cur: Path, tag: str) -> Path:
    """Where the input of repair ``tag`` is now (read only): its backup while the file as it left it (``view``) is its
    own output (its sha equals the recorded post sha), else that view. A record that exists but cannot be read dies —
    read as 'no record' it would make the repaired file the new original and stack the repair on it."""
    v = view(cur, tag)
    bk, meta = _bk(Path(cur), tag)
    if bk.is_file() and meta.is_file() and _post(meta) == _sha(v):
        return bk
    return v


def wrote(cur: Path, tag: str) -> bool:
    """Is the file (as repair ``tag`` left it) that repair's own output?"""
    return original(cur, tag) == _bk(Path(cur), tag)[0]


def source_of(cur: Path, tag: str) -> Path:
    """The input of repair ``tag`` (``original``) — when that is not its backup, the file as the earlier repairs left it
    is backed up first as the new original (no backup yet, or upstream changed it since the repair wrote it)."""
    bk, meta = _bk(Path(cur), tag)
    o = original(cur, tag)
    if o == bk:
        return bk
    if bk.is_file():
        print(f"[wrist_shift] upstream change detected ({Path(cur).name}, {tag}): backup refreshed", flush=True)
    shutil.copy2(o, bk)
    meta.write_text(json.dumps({"src_sha256": _sha(o), "post_sha256": None}))
    return bk


def mark_post(cur: Path, tag: str) -> None:
    """Record the sha256 of what the repair just wrote (the next run then reads the backup as the original)."""
    _, meta = _bk(Path(cur), tag)
    rec = json.loads(meta.read_text())  # source_of wrote it — missing or unreadable dies (never a fresh {})
    rec["post_sha256"] = _sha(cur)
    meta.write_text(json.dumps(rec))


def revert(cur: Path, tag: str) -> bool:
    """Put the input of repair ``tag`` back (the one undo path) — only while the file as it left it is its own output;
    the later repairs of CHAIN on top are reverted first (they were built on it), then the backup and its record are
    removed. False when there is nothing of this repair to undo."""
    bk, meta = _bk(Path(cur), tag)
    if not wrote(cur, tag):
        return False
    for t in reversed(_later(tag)):
        revert(cur, t)
    tmp = Path(cur).with_suffix(".revert.tmp")
    shutil.copy2(bk, tmp)
    os.replace(tmp, cur)
    bk.unlink()
    meta.unlink()
    return True


# ── writing ─────────────────────────────────────────────────────────────────────────────────────────────────────


def apply_delta(z, new: np.ndarray, vel_cols, dt: float, diff: str) -> dict:
    """npz-like ``z`` (keys qpos[, qpos_rollout, qvel]) + new qpos → the arrays to write. ``qpos_rollout`` takes the
    same delta; qvel columns ``vel_cols`` [(qpos col, qvel col, width)] take its time derivative — ``gradient``
    (central, the bake) or ``forward`` (v[t] = (q[t+1]−q[t])/dt, last row repeated — base_adapter's rule)."""
    if diff not in ("gradient", "forward"):
        raise ValueError(f"diff={diff!r} — gradient | forward")
    qpos = np.asarray(z["qpos"], float)
    out = {k: z[k] for k in z.files}
    delta = np.asarray(new, float) - qpos
    out["qpos"] = np.asarray(new).astype(z["qpos"].dtype)
    if "qpos_rollout" in out and np.asarray(out["qpos_rollout"]).shape == delta.shape:
        ro = np.asarray(out["qpos_rollout"])
        out["qpos_rollout"] = (np.array(ro, float) + delta).astype(ro.dtype)
    if "qvel" in out:
        qv = np.array(out["qvel"], float)
        T = len(delta)
        if qv.shape[0] == T:
            if diff == "gradient":
                dd = np.gradient(delta, dt, axis=0) if T > 1 else np.zeros_like(delta)
            else:
                dd = np.zeros_like(delta)
                if T > 1:
                    dd[:-1] = np.diff(delta, axis=0) / dt
                    dd[-1] = dd[-2]
            for qc, vc, n in vel_cols:
                if qv.shape[1] >= vc + n:
                    qv[:, vc : vc + n] += dd[:, qc : qc + n]
            out["qvel"] = qv.astype(np.asarray(out["qvel"]).dtype)
    return out


def shift_row(row: np.ndarray, off: int, finger_off: int, H: int, dw, fingers, obj_col=None) -> np.ndarray:
    """One frame of tree qpos, in place: the hand root position (``row[off:off+3]`` — the rig's slide joints run
    along the tree axes) moves by ``dw`` (tree frame), the finger block takes ``fingers``, a held object (position
    column ``obj_col``) is carried by the same ``dw``. → the object displacement (zeros when nothing is carried)."""
    dw = np.asarray(dw, float)
    row[off : off + 3] += dw
    row[off + finger_off : off + H] = fingers
    if obj_col is None:
        return np.zeros(3)
    row[obj_col : obj_col + 3] += dw
    return dw


def save_npz(path: Path, arrays: dict, tag: str | None) -> None:
    """Atomic write (new inode) of ``arrays`` to ``path``; with a tag, record the post sha for that repair."""
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    if tag:
        mark_post(path, tag)


# ── palm → arm EE target ────────────────────────────────────────────────────────────────────────────────────────


def ee_target(palm_pos, palm_R, palm_off, z_off: float, det: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Palm pose in the tree (capture) frame → the arm EE target in the scene frame: (Rz(−90°)·(p + R·palm_off) +
    (0, 0, z_off), Rz(−90°)·R). Batched over leading axes. ``palm_off`` = arm_ik.palm_frame_offset(hand).
    det=True evaluates it with det_math (the bits are the same on every host) — the cross repair's tracking error;
    the default keeps the einsum bits of wrist_desk."""
    p = np.asarray(palm_pos, float)
    R = np.asarray(palm_R, float)
    if det:
        po = np.asarray(palm_off, float)
        pos = DM.matvec3(RZ_NEG90, p + DM.matvec3(R, po))
        return pos + np.array([0.0, 0.0, float(z_off)]), DM.matmul3(RZ_NEG90, R)
    pos = np.einsum("ij,...j->...i", RZ_NEG90, p + np.einsum("...ij,j->...i", R, np.asarray(palm_off, float)))
    return pos + np.array([0.0, 0.0, float(z_off)]), np.einsum("ij,...jk->...ik", RZ_NEG90, R)


# ── finger range (the retargeting IK's cut) ─────────────────────────────────────────────────────────────────────


def finger_ranges(hand: str, vendor: dict, emb: str) -> dict:
    """Hand id + vendor ranges {joint: (lo, hi)} (rad, every joint the caller's asset has) + the asset's embodiment
    (which sides it carries — a one-hand scene has no joint of the other side) → the same dict with the retargeting
    IK's hyperextension cut applied (owner prior.hyperext_clip_targets · prior.hyperext_clip_range, through the window,
    which reads the sides of ``emb`` from prior.emb_sides). The window imports the rule module, not the IK, and draws
    nothing (``MUJOCO_GL=disable``), so it runs in the retargeting interpreter and in the bake/judge one alike
    (``RETARGET_PY`` may be the Isaac interpreter). A hand whose prior does not cut (``off``) keeps its vendor ranges
    — that is the stated outcome of the window, not a fallback. Then the prior's branch windows (``branch_windows`` of
    the window, `branch_cut`) narrow their joints. A window failure dies. Marker ``[FINGER_RANGE] <spec>/<emb>: cut
    targets n · changed joints k`` (+ `` · branch windows <joint>=[lo,hi]deg …`` when the prior declares any)."""
    from dexengine.core import paths as P  # noqa: PLC0415 — the retargeting interpreter's and the spec dir's owner

    spec_dir = P.hand_spec_dir(hand)
    key = (spec_dir, emb, tuple(sorted((k, float(a), float(b)) for k, (a, b) in vendor.items())))
    if key not in _RANGES:
        repo = Path(__file__).resolve().parents[2]  # the dexengine package dir: the window lives in its retarget/
        env = dict(os.environ, MUJOCO_GL="disable")  # the window compiles a rig, it never renders
        r = subprocess.run(
            [P.require_retarget_py(), "-m", "dexengine.retarget.ik.hyperext_limits", spec_dir, emb, "--off-ok"],
            cwd=repo.parent,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            input=json.dumps({k: [a, b] for k, (a, b) in vendor.items()}),
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"[FINGER_RANGE] {spec_dir}/{emb}: cut window failed rc={r.returncode}: {r.stderr[-800:]}"
            )
        win = json.loads(r.stdout)
        if (repo / "retarget") not in Path(win["rule_file"]).resolve().parents:
            raise RuntimeError(f"[FINGER_RANGE] the cut rule was read from another tree: {win['rule_file']}")
        out = {k: (float(a), float(b)) for k, (a, b) in vendor.items()}
        for row in win["rows"]:
            out[row["joint"]] = (float(row["new_lo"]), float(row["new_hi"]))
        branch = branch_cut(out, win["branch_windows"], f"{spec_dir}/{emb}")
        _RANGES[key] = out
        moved = sum(out[k] != (float(a), float(b)) for k, (a, b) in vendor.items())
        print(
            f"[FINGER_RANGE] {spec_dir}/{emb}: cut targets {len(win['rows'])} · changed joints {moved}"
            + (f" (off: {win['off']})" if win.get("off") else "")
            + (f" · branch windows {branch}" if branch else "")
            + f" — owner {win['rule_file']}",
            flush=True,
        )
    return dict(_RANGES[key])


def branch_cut(ranges: dict, windows: list, what: str) -> str:
    """In place: each branch window {joint, lo, hi} (rad — the prior's ``branch_window_deg`` as the window exports it)
    narrows its joint's range to the intersection. A window joint the ranges lack, or an empty intersection, dies (a
    renamed joint must not leave the finger free to go back to the mirror branch). → the marker text ('' = none)."""
    said = []
    for w in windows:
        if w["joint"] not in ranges:
            raise RuntimeError(f"[FINGER_RANGE] {what}: branch window joint {w['joint']!r} is not in the asset ranges")
        lo, hi = max(ranges[w["joint"]][0], float(w["lo"])), min(ranges[w["joint"]][1], float(w["hi"]))
        if lo > hi:
            raise RuntimeError(f"[FINGER_RANGE] {what}: branch window {w} misses the range {ranges[w['joint']]}")
        ranges[w["joint"]] = (lo, hi)
        said.append(f"{w['joint']}=[{np.degrees(lo):.1f},{np.degrees(hi):.1f}]deg")
    return " ".join(said)


def scene_finger_ranges(hand: str, m, emb: str) -> dict:
    """The reference range of the finger joints a compiled scene ``m`` (MjModel) carries — {joint: (lo, hi)} for every
    side whose finger joints (core.hands names) the scene has, the vendor range = the scene's own (finger_ranges).
    A side with only some of its finger joints dies — a renamed joint must not leave a finger unbounded."""
    import mujoco as mj  # noqa: PLC0415

    from dexengine.core import hands as HS  # noqa: PLC0415

    spec = HS.HAND_REGISTRY[hand]
    have = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    vendor = {}
    for side_names in (tuple(spec.finger_joint_names_right), tuple(spec.left())):
        got = [n for n in side_names if n in have]
        if got and len(got) != len(side_names):
            raise ValueError(
                f"[FINGER_RANGE] {hand}: the scene has only some of the finger joints {len(got)}/{len(side_names)}"
            )
        vendor.update({n: (float(m.jnt_range[have[n], 0]), float(m.jnt_range[have[n], 1])) for n in got})
    if not vendor:
        raise ValueError(f"[FINGER_RANGE] {hand}: the scene has no finger joint")
    return finger_ranges(hand, vendor, emb)


# ── finger re-IK and root move (MuJoCo) ─────────────────────────────────────────────────────────────────────────


def finger_ik(m, d, qpos0, qadr, dadr, lo, hi, sids, targets, iters: int = 60, lam: float = 1e-3) -> tuple:
    """Finger joints (qpos addresses ``qadr``, dof addresses ``dadr``, clamped to [lo, hi]) move the sites ``sids``
    onto ``targets`` (model frame) from ``qpos0``; every other coordinate stays. → (qpos, worst residual m)."""
    import mujoco as mj  # noqa: PLC0415

    d.qpos[:] = qpos0
    if len(sids) == 0:
        mj.mj_kinematics(m, d)
        return d.qpos.copy(), 0.0
    jp = np.zeros((3, m.nv))
    tg = np.asarray(targets, float)
    for _ in range(iters):
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        e = (tg - d.site_xpos[sids]).ravel()
        if np.abs(e).max() < 2e-4:
            break
        J = []
        for s in sids:
            mj.mj_jacSite(m, d, jp, None, int(s))
            J.append(jp[:, dadr].copy())
        Jm = np.concatenate(J)
        dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam * np.eye(len(e)), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)
    mj.mj_kinematics(m, d)
    return d.qpos.copy(), float(np.linalg.norm(tg - d.site_xpos[sids], axis=1).max())


def _rotvec(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * (th / (2.0 * np.sin(th)))


def _root_solve(m, d, body: int, qadr, dadr, pt, Rt, iters: int, hinge=None) -> float:
    """Steps of the six root joints from the state in ``d.qpos`` toward body pose (pt, Rt) → round-trip residual.
    ``hinge`` None = the plain least-squares step. With ``hinge`` (mask of the rotation coordinates) the step is the
    continued one: damped least squares (ROOT_DAMP — next to the middle angle ±90° the plain step throws the first and
    third coordinate far apart in opposite directions) and no rotation coordinate moves more than ROOT_STEP_MAX_RAD."""
    import mujoco as mj  # noqa: PLC0415

    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        e = np.concatenate([pt - d.xpos[body], _rotvec(Rt @ d.xmat[body].reshape(3, 3).T)])
        if np.abs(e).max() < ROOT_TOL_M * 0.1:
            break
        mj.mj_jacBody(m, d, jp, jr, body)
        J = np.vstack([jp[:, dadr], jr[:, dadr]])
        if hinge is None:
            dq = np.linalg.lstsq(J, e, rcond=None)[0]
        else:
            dq = J.T @ np.linalg.solve(J @ J.T + ROOT_DAMP * float(e @ e) * np.eye(len(e)), e)
            dq *= min(1.0, ROOT_STEP_MAX_RAD / max(float(np.abs(dq[hinge]).max()), 1e-300))
        d.qpos[qadr] += dq
    mj.mj_kinematics(m, d)
    return max(
        float(np.linalg.norm(pt - d.xpos[body])), float(np.linalg.norm(_rotvec(Rt @ d.xmat[body].reshape(3, 3).T)))
    )


def root_hinges(m, qadr) -> np.ndarray:
    """Mask over the root joints ``qadr``: the rotation (hinge) coordinates."""
    return np.array([int(m.jnt_type[int(np.nonzero(m.jnt_qposadr == a)[0][0])]) == 3 for a in qadr])


def move_root(m, d, qpos_row, body: int, qadr, dadr, R: np.ndarray, c, iters: int = 30, ref=None):
    """Rigid transform x → R·x + c (model frame) of the hand rooted at ``body``: solves the six root joints (qpos
    ``qadr`` / dof ``dadr``) so the body pose equals the transformed pose. The plain solve from the row runs first;
    one that does not converge restarts from ``ref`` (the caller's previous frame carried to this one; None → the
    row) with the continued step. An answer whole turns away from the row's own coordinates is brought back (whole
    turns are the same pose — the only change of coordinates made here: the callers add the coordinate change to
    other arrays too); an answer that is not comes back bit for bit. Dies when the round trip misses ROOT_TOL_M."""
    import mujoco as mj  # noqa: PLC0415

    d.qpos[:] = qpos_row
    mj.mj_kinematics(m, d)
    pt = R @ d.xpos[body] + np.asarray(c, float)
    Rt = R @ d.xmat[body].reshape(3, 3)
    qa, hinge = np.asarray(qadr), root_hinges(m, qadr)
    row = np.asarray(qpos_row, float)[qa]
    res = _root_solve(m, d, body, qadr, dadr, pt, Rt, iters)
    if res > ROOT_TOL_M:
        d.qpos[qa] = row if ref is None else np.asarray(ref, float)
        res = _root_solve(m, d, body, qadr, dadr, pt, Rt, iters, hinge)
    if res <= ROOT_TOL_M:
        rq = qa[hinge]
        near = d.qpos[rq] + 2.0 * np.pi * np.round((row[hinge] - d.qpos[rq]) / (2.0 * np.pi))
        if not np.array_equal(near, d.qpos[rq]):
            d.qpos[rq] = near
            res = _root_solve(m, d, body, qadr, dadr, pt, Rt, iters, hinge)
    if res > ROOT_TOL_M:
        raise RuntimeError(
            f"[WRIST_SHIFT] hand root move round-trip residual {res:.2e} > {ROOT_TOL_M:g} — the "
            f"continued solve (step cap {ROOT_STEP_MAX_RAD:g} rad, damped) did not converge either"
        )
    return d.qpos.copy()
