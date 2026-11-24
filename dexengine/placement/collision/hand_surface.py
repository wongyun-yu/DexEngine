#!/usr/bin/env python3
"""The hand of the torso ruler — the palm and the finger bodies, placed by the clip's finger joint values.

Bodies (one rule for every hand): the URDF links under the last arm link split into rigid groups at the moving joints.
  palm    = the links the last arm link reaches through fixed joints (the arm link itself is an arm surface body)
  fingers = per moving joint, its child link and the links that child reaches through fixed joints
A body's shape = the convex hull (64 vertices, robots.wrist_hull.reduce_hull) of its links' visual meshes in the body's
root frame (a group without a visual mesh still carries the chain, it has no shape). A moving joint that is not one of
the hand spec's finger joints of that side dies — a mimic or an unknown joint cannot be placed from the tree.
Pose: finger values → FK along the URDF (joint origin · axis · revolute/continuous/prismatic), relative to the last arm
link; torso.torso_guard composes it with the arm FK on the guard's time axis (frames and retiming chords).
Finger values (guard_finger_q): an active hand = the tree's qpos hand block (qpos_layout — the finger block is the last
n_fingers columns, in the spec's order); an idle side or a hand the tree does not have = rest_finger_q, the rest pose
the bake writes and the one-hand layout (uni_layout) inserts.
Finger speed (finger_floor_for): the finger columns of every hand block the tree has (tree_finger_q — idle sides
included, an absent side has none; zero blocks die) against finger_budget (the HandSpec applied limit
core.hands.finger_velocity_vector times stretch.FRAME_DT) give the per-interval stretch floor that retiming, the judge
and the coarse path put under the arm stretch (stretch.finger_floor) — one assembly, three callers.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError

_CACHE: dict = {}
_SERVER_KEEP = ("_CACHE",)  # URDF-keyed pure cache (hand trees), clip-independent
MOVING = ("revolute", "continuous", "prismatic")


def _origin(j) -> np.ndarray:
    from dexengine.robots import wrist_hull as G

    o = j.find("origin")
    A = np.eye(4)
    A[:3, :3] = G.rpy_matrix((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split())
    A[:3, 3] = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    return A


def _spec(hand: str):
    from dexengine.core import hands as HR

    spec = HR.HAND_REGISTRY.get(hand)
    if spec is None:
        raise InputMissingError(f"[TORSO_HANDS] hand {hand!r} is not in the hand registry — cannot place its fingers")
    return spec


def finger_names(hand: str, side: str) -> tuple[str, ...]:
    """The finger joint names of one side in the spec's (= the tree block's) order."""
    spec = _spec(hand)
    return tuple(spec.finger_joint_names_right) if side == "right" else spec.left()


def finger_vmax(hand: str) -> np.ndarray:
    """(n_fingers,) applied finger speed limit (rad/s) in the finger-block order — the owner is HandSpec
    (core.hands.finger_velocity_vector = finger_velocity_applied, the value the sim actuator gets)."""
    from dexengine.core import hands as HR

    return np.asarray(HR.finger_velocity_vector(_spec(hand)), dtype=np.float64)


def rest_finger_q(hand: str) -> np.ndarray:
    """The fingers' rest pose of an idle hand — (n_fingers,) in the spec's order, the same for both sides: the hand
    spec's default pose (HandSpec.joint_pos_defaults — the sim's own default pose, which the park hold drives to), 0 for
    a joint it does not declare (a spec declares only the joints whose limits exclude 0). The bake, the one-hand layout
    (uni_layout.inject_arrays) and the guards (guard_finger_q) read this one function. A default for a joint that is
    not a finger joint cannot be placed in the finger block and dies. ``hand`` = hand id or spec dir (paths.hand_id)."""
    from dexengine.core import hands as HR
    from dexengine.core import paths as P

    hand = P.hand_id(hand)
    names = finger_names(hand, "right")
    dflt = dict(HR.HAND_REGISTRY[hand].joint_pos_defaults)
    if set(dflt) - set(names):
        raise InputMissingError(
            f"[TORSO_HANDS] {hand}: joint_pos_defaults {sorted(set(dflt) - set(names))} are not "
            "finger joints — cannot place them in the rest-pose finger block"
        )
    return np.array([float(dflt.get(n, 0.0)) for n in names])


def hand_tree(urdf: str, last: str, names: tuple[str, ...]) -> dict:
    """The hand under `last`: groups [{root, verts | None, links}] (0 = palm) and the moving joints in parent-first
    order [{g, pg, pT, axis, prism, col}] — col = the joint's column in the finger block (`names`)."""
    key = (str(urdf), last, names)
    if key in _CACHE:
        return _CACHE[key]
    from dexengine.robots import wrist_hull as G

    root = ET.parse(str(urdf)).getroot()
    mesh = {el.get("name") for el in root.findall("link") if el.find("visual/geometry/mesh") is not None}
    kids: dict[str, list] = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    col = {n: i for i, n in enumerate(names)}
    groups, joints, where, todo = [{"root": last}], [], {last: (0, np.eye(4))}, [last]
    while todo:
        ln = todo.pop(0)  # breadth first: a parent group exists before its child
        g, T = where[ln]
        for j in kids.get(ln, []):
            c, typ, name = j.find("child").get("link"), j.get("type"), j.get("name")
            if typ == "fixed":
                where[c] = (g, T @ _origin(j))
            elif typ in MOVING and j.find("mimic") is None and name in col:
                ax = j.find("axis")
                axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
                groups.append({"root": c})
                joints.append(
                    {
                        "g": len(groups) - 1,
                        "pg": g,
                        "pT": T @ _origin(j),
                        "axis": axis / np.linalg.norm(axis),
                        "prism": typ == "prismatic",
                        "col": col[name],
                    }
                )
                where[c] = (len(groups) - 1, np.eye(4))
            else:
                mim = ", mimic" if j.find("mimic") is not None else ""
                raise InputMissingError(
                    f"[TORSO_HANDS] {Path(urdf).name}: joint {name}({typ}{mim}) under {last} is "
                    "not a finger joint of the hand spec — cannot place it from tree values"
                )
            todo.append(c)
    if sorted(jj["col"] for jj in joints) != list(range(len(names))):
        raise SystemExit(
            f"[TORSO_HANDS] {Path(urdf).name}: only {len(joints)} of {len(names)} finger joints are under {last}"
        )
    for gi, gr in enumerate(groups):
        gr["links"] = [ln for ln, (g, _T) in where.items() if g == gi and ln != last and ln in mesh]
        pts = [G.visual_mesh(Path(urdf), ln)[0] @ where[ln][1][:3, :3].T + where[ln][1][:3, 3] for ln in gr["links"]]
        gr["verts"] = G.reduce_hull(np.concatenate(pts), 64)[0] if pts else None
    if groups[0]["verts"] is None:
        raise InputMissingError(f"[TORSO_HANDS] {Path(urdf).name}: no hand visual mesh fixed to {last} — no palm")
    _CACHE[key] = {"groups": groups, "joints": joints, "n": len(names)}
    return _CACHE[key]


def finger_frames(ht: dict, fq) -> np.ndarray:
    """(P, n_groups, 4, 4): each group's root frame in the last arm link's frame for finger values fq (P, n)."""
    from dexengine.placement.collision.geom import _rot_batch

    fq = np.asarray(fq, float)
    out = np.empty((len(fq), len(ht["groups"]), 4, 4))
    out[:, 0] = np.eye(4)
    for jj in ht["joints"]:
        M = np.repeat(np.eye(4)[None], len(fq), 0)
        th = fq[:, jj["col"]]
        if jj["prism"]:
            M[:, :3, 3] = th[:, None] * jj["axis"]
        else:
            M[:, :3, :3] = _rot_batch(jj["axis"], th)
        out[:, jj["g"]] = out[:, jj["pg"]] @ jj["pT"] @ M
    return out


def tree_finger_q(tree_dir, hand: str, *, qpos=None) -> dict[str, np.ndarray]:
    """{side: (T, n_fingers)} for the hand blocks the tree **has** (qpos_layout — an idle side included, an absent
    side has no entry: nothing is invented). The finger block is the last n_fingers columns of the block, in the
    spec's order; a block of another width dies (another hand's tree). qpos defaults to the tree's
    trajectory_kinematic qpos."""
    from dexengine.core.paths import qpos_layout
    from dexengine.placement.collision.hand_layout import WRIST_DIM

    tree_dir = Path(tree_dir)
    if qpos is None:
        qpos = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    qpos = np.asarray(qpos, float)
    n = len(finger_names(hand, "right"))
    lay = qpos_layout.read(str(tree_dir), nq=qpos.shape[1])
    out = {}
    for side in ("right", "left"):
        c = lay.hands.get(side)
        if c is None:
            continue
        H = int(lay.hand_qpos_dim)
        if H - n != WRIST_DIM:
            raise SystemExit(
                f"[TORSO_HANDS] {tree_dir.name}: hand width {H} − fingers {n} ({hand}) ≠ virtual wrist "
                f"{WRIST_DIM} — a tree of another hand"
            )
        out[side] = qpos[:, c + H - n : c + H]
    return out


def guard_finger_q(tree_dir, hand: str, *, idle=(), qpos=None) -> dict[str, np.ndarray]:
    """{side: (T, n_fingers)} finger values the torso guard places each hand with (module doc): tree_finger_q, with an
    idle side or a hand the tree does not have = rest_finger_q (the rest pose). qpos defaults to the tree's
    trajectory_kinematic qpos (final_guard passes the pre-retiming one for its attribution check)."""
    if qpos is None:
        qpos = np.load(Path(tree_dir) / "0/trajectory_kinematic.npz")["qpos"]
    have = tree_finger_q(tree_dir, hand, qpos=qpos)
    return {
        side: (
            have[side] if side in have and side not in idle else np.repeat(rest_finger_q(hand)[None], len(qpos), axis=0)
        )
        for side in ("right", "left")
    }


def finger_budget(hand: str, step: int = 1) -> np.ndarray:
    """(n_fingers,) per-frame finger budget (rad) = the applied limit (finger_vmax) × the tree frame interval
    (stretch.FRAME_DT — its one owner) × step (step > 1 = a strided time axis: frames step apart). The finger budget of
    the retiming floor, the retiming's own check and the gate speed check — no caller states the interval."""
    from dexengine.placement.retime import stretch as VST

    step = int(step)
    if step < 1:
        raise ValueError(f"[finger_budget] step {step} < 1")
    return finger_vmax(hand) * VST.FRAME_DT * step


def finger_floor_for(tree_dir, hand: str, safety: float, *, step: int = 1, fq=None, qpos=None) -> np.ndarray:
    """(T'−1,) per-interval stretch floor the fingers demand — stretch.finger_floor over the hand blocks the tree has
    (tree_finger_q), budget = finger_budget(hand, step). A tree with no hand block dies there (no floor of 1 is
    invented). step > 1 = the judge's sparse tier; fq = an already-read tree_finger_q dict (read once per clip)."""
    from dexengine.placement.retime import stretch as VST

    if fq is None:
        fq = tree_finger_q(tree_dir, hand, qpos=qpos)
    step = int(step)
    return VST.finger_floor({s: f[::step] for s, f in fq.items()}, finger_budget(hand, step), safety)
