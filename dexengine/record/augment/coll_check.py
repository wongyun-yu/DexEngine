"""Per-variant checks of the augment collision gate (record.augment.coll_gate runs them in a process pool; its main is
the CLI run by path): pre_one (chain final_guard on the canonical or shadow tree + IK), post_one (the executed poses
of the rollout), and the recorded hand-hand minimum distance (hand_hand_min_mm, one CrossSetModel per clip scene).
Owns the gate marker MARK. Part of the augment cache fingerprint (AUG_CODE_FILES)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.core import stdio
from dexengine.core.errors import InputMissingError

MARK = "[AUG_COLL]"
#: hand-hand is outside the gate: the minimum hand-hand distance of the executed poses is only recorded. The post gate
#: rule is policy.post_ok.
HH_RECORD = True
#: hand-hand distance frame stride (same for pre and post): measured every this many frames (for the record).
HH_STRIDE = 2
HH_DISTMAX = 0.03


_HH: dict = {}


def _hh_model(tree: Path, ds: str, main: Path):
    """Once per clip; key = the real scene.xml (variant trees are symlinks to the original, the same model; multi-clip
    sessions get a different key per clip)."""
    from dexengine.placement.collision import cross_sets as CC

    key = str((tree / "scene.xml").resolve())
    if key not in _HH:
        with stdio.capture():
            L = CC.HandLayout(tree, quiet=True)
            assets = P.layer_path("assets_objects", ds).parent  # data/<ds>/assets (object USD root)
            _HH[key] = CC.CrossSetModel(
                L, tree_dir=tree, assets_root=main / assets.relative_to(P.main_root()), obj_geom="visual"
            )
    return _HH[key]


def hand_hand_min_mm(tree: Path, ds: str, main: Path, arm_names: dict, q: dict, fingers: dict, base) -> float:
    """Minimum hand-hand overlap (mm, negative = penetration) over frames every HH_STRIDE of q {side: (T,7)} · fingers
    {side: (T,n)}. The object block is the tree qpos (same frame, the last frame at the end; (1) has no objects)."""
    m = _hh_model(tree, ds, main)
    qpos = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    T = min(len(q["right"]), len(q["left"]))
    best = HH_DISTMAX
    for t in range(0, T, HH_STRIDE):
        _f, obj = m.layout.hand_blocks(qpos[min(t, len(qpos) - 1)])
        m.set_pose(
            {s: (arm_names[s], q[s][t]) for s in ("right", "left")},
            {s: fingers[s][t] for s in ("right", "left")},
            obj,
            base,
        )
        _mins, _h, rows = m.distances(HH_DISTMAX, sets=("1_cross",))
        for _s, cls, _a, _b, d, _ft in rows:
            if cls == "hand-hand" and d < best:
                best = d
    return round(best * 1000.0, 2)


def _ik(path: Path):
    z = np.load(path, allow_pickle=True)
    return (
        [str(x) for x in z["arm_joint_names"]],
        np.asarray(z["arm_joint_positions_all"], np.float64),
        np.asarray(z["base_xyz"], float),
    )


def _check(c: dict, tree: Path, q: dict, fingers=None) -> tuple[bool, str, float]:
    from dexengine.placement import final_guard as FG

    t0 = time.time()
    # fold guard markers such as [GUARD_BAND] (the verdict line is made here)
    with stdio.capture():
        ok, why = FG.check(c["arm"], c["hand"], tree, q, c["zoff"], fingers=fingers)
    return bool(ok), str(why), round(time.time() - t0, 1)


def pre_one(arg) -> dict:
    """One pre gate; kind 'canon' (canonical tree + canonical IK) | 'variant' (shadow)."""
    c, shadow, name, kind = arg
    from dexengine.placement.collision import hand_surface as HSF
    from dexengine.placement.collision import idle_arm as IA

    main = Path(c["main"])
    root = main if kind == "canon" else Path(shadow)
    tree = root / c["trees_rel"] / name
    names, q, base = {}, {}, None
    for s in ("right", "left"):
        names[s], q[s], b = _ik(root / c["ik_rel"][s] / f"{name}.npz")
        base = b if base is None else base
    ok, why, sec = _check(c, tree, q)
    fing = HSF.guard_finger_q(tree, c["hand"], idle=IA.idle_sides_of_tree(tree))
    t0 = time.time()
    hh = hand_hand_min_mm(tree, c["ds"], main, names, q, fing, base)
    return {
        "name": name,
        "kind": kind,
        "ok": ok,
        "why": why,
        "sec": sec,
        "hh_min_mm": hh,
        "hh_sec": round(time.time() - t0, 1),
    }


def _exec_cols(ents: list) -> dict:
    col = {}
    for e in ents:
        if e["kind"] != "artic":
            continue
        for i, n in enumerate(e["joint_names"]):
            if n in col:
                raise SystemExit(f"{MARK} joint name {n} in two articulations: no name matching")
            col[n] = e["lo"] + 7 + i
    return col


def post_one(arg) -> dict:
    c, shadow, run, j = arg
    from dexengine.placement.collision import hand_surface as HSF

    z = np.load(Path(run) / "rollouts.npz", allow_pickle=False)
    col = _exec_cols(json.loads(str(z["entities"])))
    ci = int(z["kept_clip"][j])
    name = str(z["clip_names"][ci])
    off, L = int(z["kept_offset"][j]), int(z["kept_length"][j])
    S = np.asarray(z["states"][off : off + L], np.float64)
    tree = Path(shadow) / c["trees_rel"] / name
    names, q, fing, base = {}, {}, {}, None
    for s in ("right", "left"):
        names[s], _q, b = _ik(Path(shadow) / c["ik_rel"][s] / f"{name}.npz")
        base = b if base is None else base
        fn = HSF.finger_names(c["hand"], s)
        miss = [n for n in list(names[s]) + list(fn) if n not in col]
        if miss:
            raise InputMissingError(f"{MARK} joints {miss[:3]} missing from the executed state: {run}")
        q[s] = S[:, [col[n] for n in names[s]]]
        fing[s] = S[:, [col[n] for n in fn]]
    t0 = time.time()
    with stdio.capture():
        from dexengine.placement import final_guard as FG

        vals = FG.values(c["arm"], c["hand"], tree, q, c["zoff"], fingers=fing)
    sec = round(time.time() - t0, 1)
    hh = hand_hand_min_mm(tree, c["ds"], Path(c["main"]), names, q, fing, base) if HH_RECORD else None
    return {"name": name, "j": j, "rank": int(z["kept_rank"][j]), "L": L, "values": vals, "hh_min_mm": hh, "sec": sec}
