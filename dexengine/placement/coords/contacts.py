"""Hand contact: the canonical contact sidecars of a tree, "holds", "touches at a frame", and the start state of the
main objects."""

from __future__ import annotations

import numpy as np

from dexengine.placement.coords.frames import _paths, capture_to_scene, place_objects
from dexengine.placement.coords.objects import (
    _base_to_tree,
    _main_of,
    _placers,
    _qpos_layout_mod,
    _rest_rule,
    _tree_key,
    artic_link_vertices,
    tree_f0_pad,
    tree_placement,
)
from dexengine.placement.coords.thresholds import HELD_CONTACT_FRAC, HIGH_REST_M, START_EMBED_M, START_HOLD_FRAMES

#: (sidecar path, mtime, size, base key) → checked arrays — a changed input is a new key.
_SIDECAR_CACHE: dict = {}


def _sf_mod():
    """`dexengine.data.contact.fingerprint` — the contact-sidecar input fingerprint (numpy + paths only)."""
    from dexengine.data.contact import fingerprint as m  # noqa: PLC0415

    return m


def sidecar_base_tree(tree_dir, ti: dict):
    """The **base** tree a tree (base · f0 shadow · baked) came from — the one the contact sidecar's fingerprint is
    relative to (its pose witness is the base qpos). Identity = the tree's own `hand` · `embodiment_type` · clip name
    (paths.ref_base_dir). Missing → raises."""
    import pathlib as _pl

    hand, emb = ti.get("hand"), ti.get("embodiment_type")
    if not hand or not emb:
        raise ValueError(
            f"{tree_dir}: task_info has no hand/embodiment_type — the sidecar fingerprint's base tree is unknown"
        )
    clip = _paths().clip_name(_pl.Path(tree_dir).name)
    return _paths().ref_base_dir(str(ti.get("dataset_name")), str(hand), str(emb)) / clip


def base_mismatch(tree_dir, ti: dict) -> dict:
    """{object name: (tree obj_id, base obj_id)} of the tree's objects whose obj_id differs from the same-named object
    of its base tree (`sidecar_base_tree`) — a base rebuilt after the tree was placed (e.g. an OakInk2 fixed-part merge
    re-keyed by a new part spec). Such a tree must be placed again: its base-relative inputs (contact sidecars,
    fingerprints) name the base's objects. Objects absent from the base are not compared."""
    import json as _json
    import pathlib as _pl

    bt = _pl.Path(sidecar_base_tree(tree_dir, ti))
    objs = _json.loads(_sf_mod().base_files(bt)[0].read_text())["objects"]
    base = {str(o.get("name")): str(o.get("obj_id")) for o in objs}
    return {
        str(o.get("name")): (str(o.get("obj_id")), base[str(o.get("name"))])
        for o in ti.get("objects") or []
        if str(o.get("name")) in base and base[str(o.get("name"))] != str(o.get("obj_id"))
    }


def sidecar_path(tree_dir, ti: dict, kind: str, side: str, spec: str):
    """One contact sidecar of a tree: kind `contact_base` (the side's main object) or `contact_all` (every object) of
    that side, at paths.sidecar_file for the tree's clip and hand spec folder `spec` — the place `_contact_sidecars`,
    `_all_sidecars` and `read_inputs` take it from."""
    import pathlib as _pl

    clip = _paths().clip_name(_pl.Path(tree_dir).name)
    return _paths().sidecar_file(kind, str(ti.get("dataset_name")), clip, side, spec)


def read_inputs(tree_dir, ti: dict) -> list:
    """The data-root files the contact readers open for a tree: `base_mismatch` (the base tree's task_info),
    `_contact_sidecars` (the contact_base sidecar of each side the base expects, when present) and `_all_sidecars`
    (the contact_all sidecar of each side the base expects), each sidecar with its fingerprint inputs
    (fingerprint.expected_inputs), and the hand's per-hand sidecar registry when it exists (paths.sidecar_file reads
    it to place every sidecar of that hand and dataset). What S-float-settle reads through them
    (data/objects/float_settle.inputs); the start-held statement (start_held_hands) reads a part of it."""
    SF = _sf_mod()
    bti_file = SF.base_files(sidecar_base_tree(tree_dir, ti))[0]
    spec = _paths().hand_spec_dir(_paths().hand_id(str(ti.get("hand"))))
    reg = _paths().main_root() / _paths().data_layout.sidecar_own_registry(str(ti.get("dataset_name")), spec)
    out = [bti_file] + [reg] * reg.is_file()
    for p, fp in checked_sidecars(tree_dir, ti):
        out += [p, *fp]
    return list(dict.fromkeys(out))


def checked_sidecars(tree_dir, ti: dict, layers=None) -> list:
    """(sidecar, the files its fingerprint check opens) for every contact sidecar the readers check against the tree's
    base: `_contact_sidecars` checks the contact_base sidecar of each side the base expects when the file exists,
    `_all_sidecars` the contact_all sidecar of each side the base expects (required). The check's inputs come from
    their owner, fingerprint.expected_inputs — the base tree's two files, its MANO keypoints, the dataset's generation
    declaration, the side's surface samples, the h2o plane. `layers` (default both) keeps the readers asked for. The
    base tree is read only when a sidecar of those layers can be checked: with no contact_base file
    `_contact_sidecars` returns before reading it. Read by read_inputs and coords.setup.sidecar_fp_inputs."""
    import json as _json

    SF = _sf_mod()
    rows = ((SF.LAYER_BASE, "contact_base", False), (SF.LAYER_ALL, "contact_all", True))
    rows = [r for r in rows if layers is None or r[0] in layers]
    spec = _paths().hand_spec_dir(_paths().hand_id(str(ti.get("hand"))))

    def present(kind: str) -> bool:
        return any(sidecar_path(tree_dir, ti, kind, s, spec).is_file() for s in ("right", "left"))

    if not any(always or present(kind) for _, kind, always in rows):
        return []
    bt = sidecar_base_tree(tree_dir, ti)
    bti = _json.loads(SF.base_files(bt)[0].read_text())
    root, ds = _paths().main_root(), str(ti.get("dataset_name"))
    out = []
    for layer, kind, always in rows:
        for s in SF.sides_for(bti, layer):
            p = sidecar_path(tree_dir, ti, kind, s, spec)
            if always or p.is_file():
                out.append((p, SF.expected_inputs(root, ds, bt, bti, layer, s)))
    return out


def _contact_sidecars(tree_dir, ti: dict) -> dict:
    """side → the canonical contact sidecar `precompute/<ds>/contact_mano_base/<clip>_<side>.npz` **checked against
    this tree's base** (the consumer rule is fingerprint.check):
      sides   only those the base expects (SF.sides_for — a side without a main object may carry a stale file)
      exists  → SF.check against the base's expected fingerprint — stale/unknown raises
      absent  → not in the result (callers read 'no sidecar' as undecidable → held, never a silent support)
    Value = {contact (T,) any finger · fingers (T, 5) per finger · tool_mesh_id | None · thr (contact_threshold_m) |
    None}."""
    ds = str(ti.get("dataset_name"))
    _spec = _paths().hand_spec_dir(_paths().hand_id(str(ti["hand"]))) if ti.get("hand") else ""  # unknown = ''
    files = {s: sidecar_path(tree_dir, ti, "contact_base", s, _spec) for s in ("right", "left")}
    files = {s: p for s, p in files.items() if p.is_file()}
    if not files:
        return {}
    SF = _sf_mod()
    bt = sidecar_base_tree(tree_dir, ti)
    bkey = _tree_key(bt)
    out = {}
    tin = None
    for s, p in files.items():
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size, bkey)
        if key not in _SIDECAR_CACHE:
            if tin is None:
                tin = SF.TreeInputs.load(
                    _paths().main_root(), ds, bt, _paths().hand_spec_dir(_paths().hand_id(str(ti.get("hand"))))
                )
            if s not in SF.sides_for(tin.ti, SF.LAYER_BASE):
                _SIDECAR_CACHE[key] = None
            else:
                SF.check(p, tin.expected(SF.LAYER_BASE, s), ds=ds)
                with np.load(p) as z:
                    _SIDECAR_CACHE[key] = {
                        "contact": np.any(z["contact"], axis=(1, 2)),
                        "fingers": np.any(z["contact"], axis=2),
                        "tool_mesh_id": str(z["tool_mesh_id"]) if "tool_mesh_id" in z.files else None,
                        "thr": float(z["contact_threshold_m"]) if "contact_threshold_m" in z.files else None,
                    }
        if _SIDECAR_CACHE[key] is not None:
            out[s] = _SIDECAR_CACHE[key]
    return out


def finger_contacts(tree_dir, ti: dict, T: int) -> np.ndarray:
    """(T, 10) per-finger contact of both hands (right 5, then left 5) on a tree that is not retimed yet, from the
    canonical per-side sidecars (`_contact_sidecars`, checked against the tree's base). Tree frame t reads base frame
    t - P (P = the f0 pad in front; the pad frames read base frame 0). A side the base expects without a sidecar dies;
    a side the base does not expect (no main object for that hand) holds nothing. A sidecar whose length is not T - P
    dies."""
    SF = _sf_mod()
    spec = _paths().hand_spec_dir(_paths().hand_id(str(ti.get("hand"))))
    tin = SF.TreeInputs.load(_paths().main_root(), str(ti.get("dataset_name")), sidecar_base_tree(tree_dir, ti), spec)
    sc = _contact_sidecars(tree_dir, ti)
    missing = sorted(set(SF.sides_for(tin.ti, SF.LAYER_BASE)) - set(sc))
    if missing:
        raise ValueError(f"{tree_dir}: the base expects a contact sidecar for {missing}, none found")
    P = tree_f0_pad(tree_dir, ti)
    src = np.clip(np.arange(T) - P, 0, None)
    out = np.zeros((T, 10))
    for k, side in enumerate(("right", "left")):
        if side not in sc:
            continue
        f = np.asarray(sc[side]["fingers"], bool)
        if f.shape != (T - P, 5):
            raise ValueError(f"{tree_dir}: {side} contact sidecar {f.shape} ≠ ({T - P}, 5) — tree length {T}, pad {P}")
        out[:, 5 * k : 5 * k + 5] = f[src]
    return out


def _all_sidecars(tree_dir, ti: dict) -> dict:
    """side → the all-objects contact sidecar (`contact_all`: every hand × every object of the base) **checked against
    this tree's base** like `_contact_sidecars` (SF.check on the base's expected fingerprint; stale/unknown raises).
    Every side the base expects (SF.sides_for, LAYER_ALL) must have its file: an absent one raises. Value = {names,
    ids (per object column), near_m (T, n_obj) m — the nearer of the fingertips' and the palm point's distance, thr}."""
    SF = _sf_mod()
    ds = str(ti.get("dataset_name"))
    spec = _paths().hand_spec_dir(_paths().hand_id(str(ti.get("hand"))))
    bt = sidecar_base_tree(tree_dir, ti)
    tin = SF.TreeInputs.load(_paths().main_root(), ds, bt, spec)
    out = {}
    for s in SF.sides_for(tin.ti, SF.LAYER_ALL):
        p = sidecar_path(tree_dir, ti, "contact_all", s, spec)
        if not p.is_file():
            raise ValueError(f"{tree_dir}: the base expects an all-objects contact sidecar for {s}, none at {p}")
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size, _tree_key(bt), SF.LAYER_ALL)
        if key not in _SIDECAR_CACHE:
            SF.check(p, tin.expected(SF.LAYER_ALL, s), ds=ds)
            with np.load(p) as z:
                tips = np.asarray(z["dist"], float).min(axis=2)
                _SIDECAR_CACHE[key] = {
                    "names": [str(n) for n in z["obj_names"]],
                    "ids": [str(n) for n in z["obj_ids"]],
                    "near_m": np.minimum(tips, np.asarray(z["palm_dist"], float)),
                    "thr": float(z["contact_threshold_m"]),
                }
        out[s] = _SIDECAR_CACHE[key]
    return out


def frame_near(tree_dir, ti: dict, obj_name: str):
    """**"A hand is within contact distance of that object" per base frame** → near(base_frame) → bool: a fingertip or
    the palm point of **either** hand within the sidecar's `contact_threshold_m` of the object (the all-objects layer,
    `_all_sidecars`). Wider than `frame_toucher` (the other hand and the palm count, no contact debouncing): the
    reading of an object already in a hand whose own contact flags pick it up late, or held by the other hand. An
    object absent from a sidecar's columns or under another obj_id there raises."""
    o = next((x for x in (ti.get("objects") or []) if x.get("name") == obj_name), None)
    if o is None:
        raise ValueError(f"{tree_dir}: object {obj_name!r} is not in objects[] — cannot read its distance")
    near = []
    for s, v in _all_sidecars(tree_dir, ti).items():
        if obj_name not in v["names"] or v["ids"][v["names"].index(obj_name)] != str(o.get("obj_id")):
            raise ValueError(f"{tree_dir}: the {s} all-objects sidecar has no column {obj_name}={o.get('obj_id')}")
        near.append(v["near_m"][:, v["names"].index(obj_name)] <= v["thr"])
    if not near:
        raise ValueError(f"{tree_dir}: no all-objects contact sidecar for {obj_name}")
    return lambda b: bool(any(0 <= b < len(n) and n[b] for n in near))


def _object_contact(tree_dir, ti: dict, obj_name: str) -> list:
    """Per-hand contact (any finger) frame arrays **of that object** (base frames): the checked sidecars whose
    `tool_mesh_id` is the object's obj_id — a hand holding a different object does not make this one held.
    An object not in `objects[]` raises, and so does a main object without any sidecar of its own (stage 1 writes one
    per main object, so its absence is a pipeline defect; reading it as "held" would silently drop the disc)."""
    o = next((x for x in (ti.get("objects") or []) if x.get("name") == obj_name), None)
    if o is None:
        raise ValueError(f"{tree_dir}: object {obj_name!r} is not in objects[] — cannot read its contact")
    want = str(o.get("obj_id"))
    sc = _contact_sidecars(tree_dir, ti)
    for side, v in sc.items():
        if v["tool_mesh_id"] is None:
            raise ValueError(
                f"{tree_dir}: the {side} contact sidecar has no tool_mesh_id — whose contact it is is unknown"
            )
    cts = [v["contact"] for v in sc.values() if v["tool_mesh_id"] == want]
    if not cts:
        raise ValueError(
            f"{tree_dir}: no contact sidecar of object {obj_name}({want}) — stage 1 (main-object "
            f"sidecars) should have written one (present sides {sorted(sc)}); not read as touching"
        )
    return cts


def _held_frac(cts: list, start: int, end: int) -> float:
    """Max over hands of the contact fraction on base frames [start, end] (inclusive). A window that falls outside
    every array reads 0 (no hand there)."""
    best = 0.0
    for ct in cts:
        a, e = max(0, int(start)), min(len(ct) - 1, int(end))
        if e >= a:
            best = max(best, float(ct[a : e + 1].mean()))
    return best


def _holds(cts: list, start: int, end: int) -> bool:
    """**"Holds"** — ≥ HELD_CONTACT_FRAC of base frames [start, end] in contact, over the given hands' contact arrays
    (every hand of the object, or one hand)."""
    return _held_frac(cts, start, end) >= HELD_CONTACT_FRAC


def _held_in_interval(tree_dir, ti: dict, obj_name: str, start: int, end: int) -> bool:
    """**"A hand holds the object" over base frames [start, end]**: contact (any finger) in that object's canonical
    contact sidecar on ≥ HELD_CONTACT_FRAC of the frames. Wrist-to-origin distance is not used: it cannot separate a
    gripping hand (15 cm) from a hovering one (15–20 cm)."""
    return _holds(_object_contact(tree_dir, ti, obj_name), start, end)


def start_held_hands(tree_dir, ti: dict) -> dict:
    """side → bool for both hands: the hand starts holding its own main object (the reset-wrist gate leaves that hand
    out). `_holds` over the start window [0, START_HOLD_FRAMES] base frames on **that hand's own** sidecar only.
    A hand without a main object → False. A main object without that side's sidecar, or a sidecar of another object,
    raises."""
    sc = _contact_sidecars(tree_dir, ti)
    out = {}
    for side in ("right", "left"):
        main = _main_of(ti.get("objects") or [], side)
        if main is None:
            out[side] = False
            continue
        own, want = sc.get(side), str(main.get("obj_id"))
        if own is None or own["tool_mesh_id"] != want:
            raise ValueError(
                f"{tree_dir}: no contact sidecar of the {side} hand's main object "
                f"{main.get('name')}({want}) (present {[(s, v['tool_mesh_id']) for s, v in sc.items()]})"
                f" — the start grip is unknown"
            )
        out[side] = _holds([own["contact"]], 0, START_HOLD_FRAMES)
    return out


def frame_toucher(tree_dir, ti: dict, q, obj_name: str, b2t):
    """**"The hand touches that object" per base frame** → touch(base_frame) → bool. An aerial support is not laid at a
    representative frame where this is true.
      rigid        the object's canonical sidecar contact (`_object_contact` — fingertip keypoint ↔ surface threshold
                   and smoothing are the sidecar's).
      articulated  a fingertip keypoint (MANO, base frame) within the sidecar threshold (`contact_threshold_m`, read
                   from the file) of the **base part** surface (articulation.parts.base) — while only the lid is
                   touched the body still rests on its support (measuring the whole object would wrongly drop many
                   arctic discs).
    No sidecar → raises (`_object_contact`)."""
    cts = _object_contact(tree_dir, ti, obj_name)
    o = next((x for x in (ti.get("objects") or []) if x.get("name") == obj_name), None)
    art = (o or {}).get("articulation")
    if not art:
        return lambda b: bool(any(0 <= b < len(c) and c[b] for c in cts))
    # articulated — fingertip distance to the base part surface
    from scipy.spatial.transform import Rotation as _R

    # threshold = the canonical sidecar's contact_threshold_m, read through the fingerprint-checked reader
    thr = next((v["thr"] for v in _contact_sidecars(tree_dir, ti).values() if v["thr"] is not None), None)
    if thr is None:
        raise ValueError(
            f"{tree_dir}: the contact sidecars have no contact_threshold_m — the articulated body contact "
            f"threshold is unknown"
        )
    fr = ti.get("frame") or {}
    if fr.get("h2o_tilt_d10e2"):
        raise ValueError(f"{tree_dir}: articulated body contact — the h2o tilt frame is not handled")
    yaw = np.radians(float(fr.get("yaw_deg") or 0.0))
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    if not (ti.get("provenance") or {}).get("mano_dir"):
        raise ValueError(f"{tree_dir}: no provenance.mano_dir — cannot measure the articulated body contact")
    kp = np.load(_sf_mod().mano_file(_paths().main_root(), ti))
    tips = {sd: kp[f"qpos_finger_{sd}"][..., :3] for sd in ("right", "left") if f"qpos_finger_{sd}" in kp.files}
    sg = ti.get("segment") or {}
    m0 = int(sg.get("start") or 0) + int((sg.get("frame_map") or {}).get("offset") or 0)
    Vb = artic_link_vertices(tree_dir, ti, o, "base")  # body = the base link (incl. fixed attached parts)
    s0 = int(o["qpos_start"])
    # The object vertices come from **this tree's qpos** (placed on a baked tree); the fingertips are in the base frame
    # (frame.yaw only). Move the fingertips by the same placement (place_objects — the bake's formula, z unchanged) so
    # both are measured in one frame. Unplaced trees (base · f0 shadow): identity.
    pl = tree_placement(ti)

    def _touch(b):
        f = b2t(b)
        R = _R.from_quat(q[f, [s0 + 4, s0 + 5, s0 + 6, s0 + 3]]).as_matrix()
        V = Vb @ R.T + q[f, s0 : s0 + 3]
        for tp in tips.values():
            if m0 + b >= len(tp):
                continue
            t = tp[m0 + b] @ Rz.T
            if pl is not None:
                t = np.concatenate([place_objects(t[:, :2], *pl), t[:, 2:]], axis=1)
            if float(np.min(np.linalg.norm(V[None, :, :] - t[:, None, :], axis=2))) < thr:
                return True
        return False

    return _touch


def start_gaps(tree_dir, ti: dict | None = None, q=None, *, aerial: bool, src_idx="tree") -> list:
    """The start state of each **main** object — one computation for the start support and the start-embed
    exclusion. Returns [{obj, side, g0_m, needs_support, held, still, x, y (scene), r, floor_capture}]:
      g0       table: (mesh floor at tree frame 0, visual.obj) − support.desk_z · aerial: None (no desk)
      held     `_held_in_interval` over the first START_HOLD_FRAMES base frames
      still    the mesh floor moves ≤ rest Z_STAB over those frames (support goes only under an object really at rest)
      needs_support  unheld and nothing else holds it: aerial → always · table → g0 ≥ HIGH_REST_M (below that the desk
               holds it)
    held/still are evaluated only where a rule acts (else None): aerial always · table where g0 ≥ HIGH_REST_M or
    g0 < −START_EMBED_M. `aerial` is the caller's scene (keyword, no default). Table: the desk plane =
    qpos_layout.desk_z_of (a tree without a desk-plane statement raises; TACO stage-1 staging reads 0.0)."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    if ti is None:
        ti = _json.loads((tree_dir / "task_info.json").read_text())
    dz = None if aerial else _qpos_layout_mod().desk_z_of(ti, str(tree_dir))[0]
    if q is None:
        q = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    _placed, _main_for = _placers(tree_dir, ti, q)
    b2t = _base_to_tree(tree_dir, ti, len(q), src_idx=src_idx)  # start window in tree frames (f0 pad · retime map)
    win = sorted({b2t(b) for b in range(START_HOLD_FRAMES)})
    out, seen = [], set()
    for side in ("right", "left"):
        o = _main_for(side)
        if o is None or str(o.get("name")) in seen:
            continue
        seen.add(str(o.get("name")))
        V0 = _placed(o, 0)
        fl = float(V0[:, 2].min())
        g0 = None if aerial else fl - float(dz)
        cxy = (V0[:, :2].min(0) + V0[:, :2].max(0)) / 2.0
        sxy = capture_to_scene(cxy[None, :])[0]
        acts = aerial or g0 >= HIGH_REST_M or g0 < -START_EMBED_M
        held = bool(_held_in_interval(tree_dir, ti, str(o.get("name")), 0, START_HOLD_FRAMES)) if acts else None
        high = aerial or g0 >= HIGH_REST_M
        still = None
        if high and held is False:
            fls = [float(_placed(o, f)[:, 2].min()) for f in win]
            still = bool(max(fls) - min(fls) <= _rest_rule().Z_STAB)
        out.append(
            {
                "obj": str(o.get("name")),
                "side": side,
                "g0_m": g0,
                "needs_support": bool(high and held is False),
                "held": held,
                "still": still,
                "x": float(sxy[0]),
                "y": float(sxy[1]),
                "r": float(np.linalg.norm(V0[:, :2] - cxy, axis=1).max()),
                "floor_capture": fl,
            }
        )
    return out


def start_embedded(tree_dir, ti: dict | None = None, q=None) -> list:
    """The unheld main objects whose start-pose mesh floor is more than START_EMBED_M below the desk → [start_gaps
    record, …] (empty = keep). Read at the first pass (train candidates)."""
    return [g for g in start_gaps(tree_dir, ti, q, aerial=False) if g["g0_m"] < -START_EMBED_M and g["held"] is False]
