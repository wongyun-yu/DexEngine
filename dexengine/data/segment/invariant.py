"""D36 piece invariant: is a registered piece = **a window piece of the current parent**. Owner of the comparison set
and the masking declaration.

segment.split builds pieces (piece_task_info · piece_support · window rule WINDOW_DECL · copy rule PIECE_COPY) and this
file measures whether a built piece matches the current parent (piece_stale_reason). Caller =
segment.build.reslice_stale
(`--reslice-stale`; stage-1 axis reslice_pieces). The staleness reason strings
(`task_info:<field>` · `quality:<field>` · `support:<field>` · scene · `<npz>:<key>`) go as is into the marker
`[RESLICE_STALE] … reasons {…}`.
"""

from __future__ import annotations

import json

import numpy as np

from dexengine.core.errors import InputMissingError
from dexengine.data.objects import (
    # owner of the scene file names (SCENE_XMLS); the callers (segment.split · tests) have already opened the paths
    scene_parts as SP,
)
from dexengine.data.segment import pieces as PS
from dexengine.data.segment import (
    # owner of the piece build rules (PIECE_COPY · window fields · support parent fields · layer locations)
    split as SS,
)

#: PIECE_COPY fields and subfields that later steps rewrite **per tree** (parent and piece separately); masked by
#: piece_stale_reason.
#: The statement lives in the writer's source (same form as WINDOW_DECL): (file, top-level tuple constant, slot).
#:   top = whole top-level field · objects = subfields of objects[] elements
#:   qpos_layout:object = subfields of the object elements of qpos_layout[] (kind == "object")
#:   provenance = subfields of provenance (build time, not a content statement)
RESTATED_DECL = (
    ("dexengine/data/object_roles.py", "ROLE_TOP", "top"),
    ("dexengine/data/object_roles.py", "ROLE_OBJECT", "objects"),
    ("dexengine/data/object_roles.py", "ROLE_LAYOUT", "qpos_layout:object"),
    ("dexengine/data/base_layer/common.py", "PROVENANCE_VOLATILE", "provenance"),
)


def _restated(slot: str) -> set:
    """One slot of RESTATED_DECL. A declaration pointing outside PIECE_COPY fails: a renamed field silently left
    unmasked
    or a declaration masking nothing would make pieces with other roles stale every time, or let real changes
    through."""
    if slot not in {sl for _p, _n, sl in RESTATED_DECL}:
        raise InputMissingError(f"[piece_invariant] RESTATED_DECL has no slot {slot!r}")
    got = {k for path, name, sl in RESTATED_DECL if sl == slot for k in PS._src_tuple(path, name)}
    if slot == "top" and not got <= set(PS.PIECE_COPY):
        raise SystemExit(
            f"[piece_invariant] RESTATED_DECL top {sorted(got - set(PS.PIECE_COPY))} is outside PIECE_COPY"
        )
    if slot != "top" and slot.split(":")[0] not in PS.PIECE_COPY:
        raise SystemExit(f"[piece_invariant] fields of RESTATED_DECL slot {slot!r} are outside PIECE_COPY")
    return got


def _canon(v) -> str:
    """Canonical string for comparison; json round-trip NaN is `nan != nan`, so comparing as is would make the same
    piece
    stale every time (measured: 84 pieces per hand with NaN quality.palm_*_mm)."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def piece_copy_view(ti: dict) -> dict:
    """Canonical copy of the **parent statement** the piece invariant compares: {"task_info:<field>" |
    "quality:<field>": canonical string}.
    = PIECE_COPY fields (minus fields and subfields masked by RESTATED_DECL) + the non-window quality fields (minus the
    producers' window fields; what piece_task_info receives). Missing fields have no key (presence is compared too).
    rules (PIECE_RULES_COPY) are left out; 2b and 2c rewrite those markers per tree."""
    top, obj, lay, prov = (_restated(s) for s in ("top", "objects", "qpos_layout:object", "provenance"))
    out = {}
    for k in PS.PIECE_COPY:
        if k in top or k not in ti:
            continue
        v = ti[k]
        if k == "objects" and isinstance(v, list):
            v = [{a: b for a, b in o.items() if a not in obj} if isinstance(o, dict) else o for o in v]
        elif k == "qpos_layout" and isinstance(v, list):
            v = [
                {a: b for a, b in e.items() if a not in lay} if isinstance(e, dict) and e.get("kind") == "object" else e
                for e in v
            ]
        elif k == "provenance" and isinstance(v, dict):
            v = {a: b for a, b in v.items() if a not in prov}
        out[f"task_info:{k}"] = _canon(v)
    qw = PS._window("quality")
    for k, v in (ti.get("quality") or {}).items():
        if k not in qw:
            out[f"quality:{k}"] = _canon(v)
    return out


def piece_stale_reason(ds: str, emb: str, parent: str, u: dict) -> str:
    """Whether the piece is **a window piece of the current parent** (an invariant, not a trigger). "" when equal, else
    the first difference (in the order below). The verdict compares the piece's **own copy** with the current parent,
    so only changed pieces become stale (a parent rebuild is not counted as an event).
    Compared (everything the parent states and the piece receives):
      ⓪ the piece's own fields are within the piece rule (segment.pieces.schema_stale_reason, the same function stage-1
         column 2 applies to parents)
      ① task_info, piece_copy_view: PIECE_COPY fields (provenance.adapter_version · objects · qpos_layout · object
         structure · hand assets …) + non-window quality fields. Masked = RESTATED_DECL (role, side, tag, slots,
             sharing,
         retarget keys rewritten per tree by role step 4 · provenance build time). Without this comparison, an adapter
         version bump that changes only task_info would let pieces carrying the old task_info pass as 'current'
         (base_layer.stale.tree_stale_reason measures pieces through their parent).
      ② parent-owned support fields (base_adapter SUPPORT_COPIED + SUPPORT_DERIVED: desk height, rule versions, …,
         window-independent). Without it, pieces of a parent whose desk alone changed (an OakInk2 desk rule revision;
         npz and scene unchanged) would keep the old desk_z.
      ③ scene*.xml bytes
      ④ tree arrays (SS.tree_arrays, excluding repair backups and half-written files): list, keys and arrays (the
         parent arrays sliced to [t0, t1), the same slicing as split_clip_registry). A rebuilt parent (stage-1
         regeneration) with pieces still cut from the old parent is caught here; otherwise old retargeted hand
         trajectories stay in pieces and judgement, bake and shared sidecars mix generations.
    Not compared: window fields (WINDOW_DECL, re-measured on the piece window) · window-bound support (rest_intervals,
    the job of --refresh-support) · support fields later steps write per piece (obj_zmin_traj etc.) · rules (D1g · D25;
    2b and 2c rewrite the markers per tree, and their substance is measured by ① and ④ through objects · qpos_layout ·
    nq · npz)."""
    src, dst = SS._base_dir(ds, emb, parent), SS._base_dir(ds, emb, u["name"])
    ti = json.loads((src / "task_info.json").read_text())
    cti = json.loads((dst / "task_info.json").read_text())
    # ⓪ the piece's own fields are outside the current piece rule (old-generation fields; ① looks only at PIECE_COPY, so
    # fields outside the rule are invisible there).
    why = PS.schema_stale_reason(cti)
    if why:
        return why
    pv, cv = piece_copy_view(ti), piece_copy_view(cti)
    for k in list(pv) + [k for k in cv if k not in pv]:
        if pv.get(k) != cv.get(k):
            return k
    T = int((ti.get("case") or {}).get("T") or 0) or int(np.load(src / "0/trajectory_kinematic.npz")["qpos"].shape[0])
    t0, t1 = max(0, int(u["t0"])), min(T, int(u["t1"]))
    ps, cs = ti.get("support"), cti.get("support")
    if (ps is None) != (cs is None):
        return "support presence"
    for k in PS._support_parent_keys() if ps is not None else ():
        if ps.get(k) != cs.get(k):
            return f"support:{k}"
    for x in SP.SCENE_XMLS:
        a, b = src / x, dst / x
        if a.is_file() != b.is_file() or (a.is_file() and a.read_bytes() != b.read_bytes()):
            return x
    pn = [q.name for q in SS.tree_arrays(src)]
    if pn != [q.name for q in SS.tree_arrays(dst)]:
        return "npz list"
    for name in pn:
        with np.load(src / "0" / name, allow_pickle=True) as zp, np.load(dst / "0" / name, allow_pickle=True) as zc:
            if sorted(zp.files) != sorted(zc.files):
                return f"{name} keys"
            for k in zp.files:
                a = zp[k]
                want = a[t0:t1] if getattr(a, "ndim", 0) >= 1 and a.shape[:1] == (T,) else a
                got = zc[k]
                same = want.shape == got.shape and (
                    np.array_equal(want, got, equal_nan=True) if want.dtype.kind in "fc" else np.array_equal(want, got)
                )
                if not same:
                    return f"{name}:{k}"
    return ""
