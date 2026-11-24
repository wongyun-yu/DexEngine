#!/usr/bin/env python3
"""Measurements for the extra-object slot rule (`dexengine/data/objects/extra_slots.py`): motion, receiving, below the
desk surface, distribution line. The rule itself (class order, class names, slot count) lives in extra_slots; this file
holds only the measurements it reads.

Receiving (receivers) is the condition under which an untouched still extra stands in the scene (other untouched still
objects are added at render time only, except the ones S-envelope keeps next to the recorded motion):
  stands on  a spawned object (main object and moved extras at frame 0, rest intervals and the end frame; support
             objects, touched still extras and objects already receiving at frame 0) stands on it, by
             `coords._stands_on` (the pedestal geometry the judgement uses)
  arrives    a moved object (main object, moved extra) starts apart and at the end frame touches it or sits inside it
             (container): the receiver category of the scene requirement statement `scene_requirement`, loosened by the
             production slack (SLACK_RECV_M), same formula as `dexengine.data.oakink2.required`; the oracle
             `dexengine.data.readers.raw.common` measures without slack, so production ⊇ judgement.
The rule tests swap in a synthetic `geom` (removing the receiving object drops it).
"""

from __future__ import annotations

import os

import numpy as np

from dexengine.data.objects import (
    scene_requirement as SR,
)  # scene requirement statement: receiver threshold (RECV_M) and production slack (SLACK_RECV_M)
from dexengine.data.support import bands as _B


def motion(q: np.ndarray, st: int) -> tuple[float, float, float]:
    """(max motion m, max rotation °, max frame-to-frame jump m) of the 7-column pose (xyz · wxyz) at qpos column st."""
    pos = q[:, st : st + 3].astype(np.float64)
    qw = q[:, st + 3 : st + 7].astype(np.float64)
    n = np.linalg.norm(qw, axis=1, keepdims=True)
    if not np.all(n > 1e-9):
        raise ValueError(f"pose column {st}: has frames with a zero-norm quaternion")
    qw = qw / n
    move = float(np.linalg.norm(pos - pos[0], axis=1).max())
    rot = float(np.degrees(2 * np.arccos(np.clip(np.abs((qw * qw[0]).sum(1)), 0.0, 1.0))).max())
    step = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max()) if len(q) > 1 else 0.0
    return move, rot, step


def geom(tree_dir, ti: dict, q):
    """(placed(object, frame) → captured vertices (n,3), stands(upper vertices, lower vertices) → does upper stand on
    lower).
    The geometry belongs to coords (the same `_stands_on` the pedestal and the judgement use); rule tests replace it
    here."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    placed, _ = C._placers(tree_dir, ti, q)

    def stands(Vu, Vl) -> bool:
        # Box prefilter, same result: it skips layouts where support_contact must return None (the lower object has no
        # vertex within SUPPORT_REACH_M horizontally of the upper object's bottom, or none vertically in
        # [bottom − SUPPORT_GAP_MAX_M, bottom + SUPPORT_NEST_MAX_M]). Most still extras are far away; only close pairs
        # search vertices.
        lu, hu, ll, hl = Vu.min(0), Vu.max(0), Vl.min(0), Vl.max(0)
        if (
            np.any(lu[:2] - hl[:2] > _B.SUPPORT_REACH_M)
            or np.any(ll[:2] - hu[:2] > _B.SUPPORT_REACH_M)
            or hl[2] < lu[2] - _B.SUPPORT_GAP_MAX_M
            or ll[2] > lu[2] + _B.SUPPORT_NEST_MAX_M
        ):
            return False
        return C._stands_on(Vu, [Vl])

    return placed, stands


def stand_frames(tree_dir: str, ti: dict, q: np.ndarray, names: list) -> dict:
    """{moving spawned object: tree frames where it rests}: frame 0 · end frame · the middle of each of its rest
    intervals
    (support.rest_intervals). Applies the frame rule of `coords.main_rest_frames` to moved extras as well."""
    last = len(q) - 1
    fr = {n: {0, last} for n in names}
    ivs = (ti.get("support") or {}).get("rest_intervals") or []
    if ivs and fr:
        from dexengine.placement.coords import (
            core as C,  # noqa: PLC0415 — owner of rest-interval object names and the base → tree frame map
        )

        RRu, b2t = C._rest_rule(), C._base_to_tree(tree_dir, ti, len(q))
        for iv in ivs:
            n = RRu.rest_object_name(iv, ti)
            if n in fr:
                fr[n].add(min(last, max(0, int(b2t((int(iv["start"]) + int(iv["end"]) - 1) // 2)))))
    return {n: sorted(v) for n, v in fr.items()}


def arrives(Vm0: np.ndarray, Vm1: np.ndarray, Vc: np.ndarray) -> bool:
    """Production-side measurement of the scene requirement receiver (thresholds owned by scene_requirement, loosened by
    the production slack, same formula as oakink2.required): the moved object (start Vm0, end Vm1) starts farther than
    RECV_M − slack from the still object Vc and ends within RECV_M + slack, or with its centre inside the Vc box
    (inside a container). Distance = nearest vertex (`_within`)."""
    lo, hi = Vc.min(0), Vc.max(0)
    cm = Vm1.mean(0)
    if not (within(Vm1, Vc, lo, hi, SR.RECV_M + SR.SLACK_RECV_M) or np.all((cm > lo) & (cm < hi))):
        return False
    return not within(Vm0, Vc, lo, hi, SR.RECV_M - SR.SLACK_RECV_M)


def within(Vm: np.ndarray, Vc: np.ndarray, lo: np.ndarray, hi: np.ndarray, d: float) -> bool:
    """Whether any vertex of Vm lies within d of a vertex of Vc (nearest ≤ d). Only Vm points inside the Vc box (lo ·
    hi)
    grown by d are measured; points outside cannot be within d (same result; far objects end without a search)."""
    m = Vm[np.all((Vm >= lo - d) & (Vm <= hi + d), axis=1)]
    if not len(m):
        return False
    from scipy.spatial import cKDTree  # noqa: PLC0415

    return bool(np.isfinite(cKDTree(Vc).query(m, distance_upper_bound=d)[0]).any())


#: Receiving results per tree and pose; several readers call clip_rows repeatedly (the key holds the pose bytes of the
#: frames used).
MEMO: dict = {}


def receivers(tree_dir: str, ti: dict, q: np.ndarray, names: list, moved: list, still: list) -> dict:
    """{receiving still extra name: partner name}: stands on ∨ arrives (module docstring). `names` = untouched still
    extras
    (candidates) · `moved` = names of moving extra slots · `still` = still extras already standing (touched). Partners =
    spawned objects: moving ones (main object · moved; frame 0 · rest intervals · end) and fixed ones (support objects ·
    still · candidates already receiving; frame 0). A candidate standing on a receiver becomes a receiver too, so this
    iterates until nothing changes (spawned objects never stand on objects that are not placed)."""
    objs = {str(o["name"]): o for o in ti.get("objects") or [] if o.get("qpos_start") is not None and o.get("mesh_dir")}
    movers = [n for n, o in objs.items() if o.get("role") == "main"] + [n for n in moved if n in objs]
    fixed = [n for n, o in objs.items() if o.get("role") == "support"] + [n for n in still if n in objs]
    frames = stand_frames(tree_dir, ti, q, movers)
    last = len(q) - 1
    used = sorted({0, last} | {f for fs in frames.values() for f in fs})
    cols = sorted(int(objs[n]["qpos_start"]) for n in set(names) | set(movers) | set(fixed))
    key = (
        str(tree_dir),
        tuple(names),
        tuple(movers),
        tuple(fixed),
        tuple(used),
        hash(np.ascontiguousarray(q[used][:, [c + k for c in cols for k in range(7)]]).tobytes()),
    )
    if key in MEMO:
        return dict(MEMO[key])
    placed, stands = geom(tree_dir, ti, q)
    upper = [(n, placed(objs[n], f)) for n in movers for f in frames[n]] + [(n, placed(objs[n], 0)) for n in fixed]
    ends = {n: (placed(objs[n], 0), placed(objs[n], last)) for n in movers}
    V0 = {n: placed(objs[n], 0) for n in names}
    out = {n: m for n in names for m in [next((m for m in movers if arrives(*ends[m], V0[n])), None)] if m}
    upper += [(n, V0[n]) for n in out]
    left, grew = [n for n in names if n not in out], True
    while grew and left:
        grew = False
        for n in list(left):
            by = next((u for u, Vu in upper if stands(Vu, V0[n])), None)
            if by is not None:
                out[n] = by
                left.remove(n)
                upper.append((n, V0[n]))
                grew = True
    if len(MEMO) >= 64:
        MEMO.clear()
    MEMO[key] = dict(out)
    return out


def env_inputs(ti: dict, spawned_roles: tuple) -> dict:
    """What the env reads outside the tree because of this rule: {"mesh": [absolute object mesh paths] · "usd_ids":
    [obj_id of extras that can become slots]}. mesh = the files the geometry opens (receiving · below the desk · robot
    footprint, coords._placers), as declared by `coords.object_files` (all tree objects, the same declaration as the
    judge transfer list setup_inputs). usd_ids = objects not in a spawn role (extras) with a qpos column; which of them
    stand is the rule's decision (a subset). Trees where the rule does not run (no extras) give two empty lists."""
    from dexengine.placement.coords import (
        core as C,  # noqa: PLC0415 — owner of the object file declaration and physical paths (input_path)
    )

    extras = [o for o in ti.get("objects") or [] if o.get("role") not in spawned_roles]
    if not extras:
        return {"mesh": [], "usd_ids": []}
    ds = str(ti.get("dataset_name"))
    return {
        "mesh": [C.input_path(ds, p) for p in C.object_files(ti)],
        "usd_ids": list(dict.fromkeys(str(o["obj_id"]) for o in extras if o.get("qpos_start") is not None)),
    }


def below_work_surface(tree_dir: str, ti: dict, q: np.ndarray, o: dict) -> bool:
    """Whether the frame-0 bottom of an untouched, unmoved extra lies more than the support noise band
    (`coords.HIGH_REST_M`, the same band that raises pedestals) below the desk surface (work surface, tree
    support.desk_z).
    Desk scenes within the pedestal rule's scope only (`coords.pedestal_scene`); aerial scenes, datasets out of scope or
    no desk statement → false. True means it is not spawned: the desk surface is the lowest surface interacting objects
    rest on, so a still object below it (an object on the floor) would end up inside the desk volume."""
    from dexengine.placement.coords import (
        core as C,  # noqa: PLC0415 — owner of the support noise band, object vertices (_placers) and the desk-surface reading
    )

    if (ti.get("support") or {}).get("desk_z") is None or not C.pedestal_scene(
        ti
    ):  # desk scenes within the pedestal rule's scope only
        return False
    dz = C._qpos_layout_mod().desk_z_of(ti, str(tree_dir))[0]
    placed, _ = C._placers(tree_dir, ti, q)
    return float(placed(o, 0)[:, 2].min()) < float(dz) - _B.HIGH_REST_M


def held_intervals(tree_dir: str, ti: dict, name: str) -> dict:
    """Intervals in which the person holds the object, in **tree frames**: {side: [[start, end], …]} (end inclusive).
    Hold = tree stamp `case.grasp_s[name][side]` (stage 1 contact measurement, source time in seconds). The source frame
    of
    tree frame i = `derive_map[i]` of the contact sidecar `0/contact_mano_<side>.npz` (the map through f0 padding,
    retiming and tail trimming). A missing map is an error."""
    case = ti.get("case") or {}
    fps = float(case.get("fps") or 0.0)
    if fps <= 0:
        raise ValueError(f"{tree_dir}: no case.fps stamp — cannot convert hold times to frames")
    dmap = None
    for side in ("right", "left"):
        f = os.path.join(tree_dir, "0", f"contact_mano_{side}.npz")
        if os.path.isfile(f):
            z = np.load(f, allow_pickle=True)
            if "derive_map" in z.files:
                dmap = np.asarray(z["derive_map"], dtype=np.float64)
                break
    if dmap is None:
        raise ValueError(f"{tree_dir}: contact sidecar has no derive_map — tree frame ↔ source time map unknown")
    t_src = dmap / fps
    out = {}
    for side, iv in ((case.get("grasp_s") or {}).get(name) or {}).items():
        held = np.zeros(len(t_src), bool)
        for a, b in iv or []:
            held |= (t_src >= float(a)) & (t_src <= float(b))
        idx = np.flatnonzero(held)
        if len(idx):
            cuts = np.flatnonzero(np.diff(idx) > 1)
            starts, ends = np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]]
            out[side] = [[int(x), int(y)] for x, y in zip(starts, ends, strict=True)]
    return out


def dist_text(dist: dict, kinds: tuple) -> str:
    """Distribution line: `slots per clip (moving/touched static/receiving static) 0/0/0:<clips> …` + totals per kind ·
    max per clip."""
    tot = [sum(int(k.split("/")[i]) * v for k, v in dist.items()) for i in range(len(kinds))]
    mx = max((sum(int(x) for x in k.split("/")) for k in dist), default=0)
    return (
        "slots per clip ("
        + "/".join(kinds)
        + ") "
        + " ".join(f"{k}:{dist[k]}" for k in sorted(dist))
        + " · total "
        + " ".join(f"{k}={v}" for k, v in zip(kinds, tot, strict=True))
        + f" · max per clip {mx}"
    )
