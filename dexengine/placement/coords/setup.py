"""What the setup computation reads (code and data — the cache keys and transfer lists), and whether a stamped
support statement is still the one the current rule gives (staleness)."""

from __future__ import annotations

import numpy as np

from dexengine.core import stdio
from dexengine.placement.coords.contacts import _sf_mod, checked_sidecars, sidecar_base_tree
from dexengine.placement.coords.frames import _paths, scene_mode
from dexengine.placement.coords.objects import _taco_paths, is_baked_tree
from dexengine.placement.coords.stamp import (
    compute_aerial_support,
    compute_desk_support,
    desk_support_discs,
    read_aerial_support,
)
from dexengine.placement.coords.thresholds import SETUP_STALE_TOL_M, SETUP_STATEMENT_KEYS


def object_files(ti: dict) -> list:
    """The object mesh files of a tree by their **stored** names (logical `objects[].mesh_dir` + file — resolve with
    `input_path`): visual.obj of every object with a mesh_dir, then its articulated link parts (incl. fixed attached
    parts). The one declaration of what `_placers` and frame_toucher can open — read by setup_inputs (the judge's
    transfer list) and the extra-slot rule's env statement (the RL bundle)."""
    import pathlib as _pl

    ds, out = str(ti.get("dataset_name")), []
    for o in ti.get("objects") or []:
        if o.get("mesh_dir"):
            d = _pl.Path(o["mesh_dir"])
            out.append(d / "visual.obj")
            parts = (o.get("articulation") or {}).get("parts") or {}
            # part file names come from the one rule (object_part_ref), not by joining the stored text: a sibling part
            # (`../<id>/…`) joined as is would contain `..`, which input_path (object_path) does not accept
            out += [
                _pl.Path(_paths().data_layout.object_part_ref(ds, o["mesh_dir"], rel))
                for k in _taco_paths().ARTIC_LINKS
                if parts.get(k)
                for rel, _pose in _taco_paths().artic_link_files(parts, k)
            ]
    return out


def dataset_files(ds: str) -> list:
    """**Dataset-level** inputs — files a dataset's rules open whole, not per tree: OakInk2's official part tree json
    (oakink2 parts.input_files — opened by the extra-slot rule's part check). Read by the judge's transfer list and the
    RL layer (run_inputs.dataset_level). Deliberately not in setup_inputs (the staleness cache key) nor in a bundle's
    input set: no tree changed, so neither may turn stale for these. A dataset with no such files → []."""
    if ds == "oakink2":  # the dataset whose part tree extra_slots opens
        from dexengine.data.oakink2 import parts as _OP  # noqa: PLC0415

        return list(_OP.input_files())
    return []


def sidecar_inputs(tree_dir, ti: dict) -> list:
    """What a tree's contact-sidecar read (`_contact_sidecars`) needs so a remote root resolves the same place: the
    contact_base file of both sides (paths.sidecar_file — they may not exist; the reader decides which it wants) + the
    hand's per-hand registry when the clip is registered (the registry decides the place: a job given the files
    without it resolves to the shared place and finds nothing). The order and content are part of the judge
    fingerprint key. Read by setup_inputs (remote judge) and the render lane's sidecar inputs."""
    import pathlib as _pl

    ds = str(ti.get("dataset_name"))
    clip = _paths().clip_name(_pl.Path(tree_dir).name)
    _spec = _paths().hand_spec_dir(_paths().hand_id(str(ti["hand"]))) if ti.get("hand") else ""  # unknown = ''
    out = [_paths().sidecar_file("contact_base", ds, clip, s, _spec) for s in ("right", "left")]
    _dl, _mr = _paths().data_layout, _paths().main_root()
    if _spec and clip in _dl.sidecar_own_clips(_mr, ds, _spec):  # per-hand registration — ship the registry too
        out.append(_mr / _dl.sidecar_own_registry(ds, _spec))
    return out


def setup_inputs(tree_dir, ti: dict) -> list:
    """Every **data** file the setup computation (compute_desk_support · compute_aerial_support · the stamp readers)
    reads for this tree — the staleness cache keys on their (mtime, size). Object files are declared by their
    **stored** name (the logical `objects[].mesh_dir` + file); a consumer that opens, stats or ships an entry resolves
    it with `input_path`, never against the working folder."""
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    out = [tree_dir / "task_info.json", tree_dir / "0/trajectory_kinematic.npz", tree_dir / "F0_REPAIR.json"]
    out += sidecar_inputs(tree_dir, ti)
    md = (ti.get("provenance") or {}).get("mano_dir")
    if md:
        out.append(_paths().main_root() / md / "0/trajectory_keypoints.npz")  # frame_toucher (articulated)
    out += object_files(ti)
    out += sidecar_fp_inputs(tree_dir, ti, stated=out)
    return out


def sidecar_fp_inputs(tree_dir, ti: dict, stated=()) -> list:
    """What the sidecar fingerprint check (_contact_sidecars → fingerprint.expected) opens besides the sidecar files:
    the base tree (task_info · kinematic) · the object surface samples (SF.surface_path — surface_parts.npz for
    articulated objects) · the h2o plane, then what the check's owner states beyond them
    (contacts.checked_sidecars → fingerprint.expected_inputs, for the contact_base sidecars `_contact_sidecars`
    checks): the dataset's generation declaration (ds_ref reads its reference direction) and the base's MANO
    keypoints. Those are appended after the entries above, in the owner's order, only when not already among them or
    in `stated` (the caller's own entries — setup_inputs passes its list), so a tree whose check opens nothing new (no
    declaration file, no sidecar) keeps the list it had, byte for byte. Read by setup_inputs (remote judge · cache
    keys · the aerial f0 key through scene_z_offset_inputs) and the render lane's statement; a missing surface sample
    there gives surface_sha='missing' → a stale fingerprint."""
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    ds = str(ti.get("dataset_name"))
    out = []
    if ti.get("hand") and ti.get("embodiment_type"):
        bt = sidecar_base_tree(tree_dir, ti)
        out += [bt / "task_info.json", bt / "0/trajectory_kinematic.npz"]
    SF = _sf_mod()
    out += [SF.surface_path(_paths().main_root(), ds, o) for o in ti.get("objects") or [] if o.get("obj_id")]
    if ds == "h2o":
        from dexengine.data.readers import (
            h2o_frame as _HF,  # noqa: PLC0415 — the plane file (the fingerprint's tilt input)
        )

        out.append(_HF.plane_path())
    if ti.get("hand") and ti.get("embodiment_type"):
        have = {str(x) for x in (*stated, *out)}
        for _sidecar, fp in checked_sidecars(tree_dir, ti, layers=(SF.LAYER_BASE,)):
            for f in fp:
                if str(f) not in have:
                    out.append(f)
                    have.add(str(f))
    return out


def input_path(ds: str, p, root=None):
    """One setup_inputs entry → the file actually opened. Absolute entries as they are. A relative entry is an object
    file declared by its stored reference (`data/<ds>/assets/objects/<id>/visual.obj`), resolved by
    data_layout.object_path under the data root (`root`, else paths.main_root). A relative entry that is not an object
    reference raises. The declared text is never changed — the cache keys hold those names and their order, so moving
    the physical place does not move the keys."""
    import pathlib as _pl

    p = _pl.Path(p)
    if p.is_absolute():
        return p
    base = _pl.Path(root) if root is not None else _paths().main_root()
    return base / _paths().data_layout.object_path(ds, str(p))


def _supports_diff(a, b, tol: float) -> list:
    """Support objects of two statements: the same (support, main, holds) set and each frame-0 pose within tol
    (position and quaternion components). None = a stamp from before support objects — always a difference."""
    if a is None or b is None:
        return (
            []
            if a is None and b is None
            else [
                f"support objects field {'none' if a is None else len(a)}→"
                f"{'none' if b is None else len(b)} (stamp from before support objects)"
            ]
        )
    ka = {(str(x["obj"]), str(x["main"]), bool(x["holds"])): x["pose"] for x in a}
    kb = {(str(x["obj"]), str(x["main"]), bool(x["holds"])): x["pose"] for x in b}
    if set(ka) != set(kb):
        return [f"support objects {sorted(ka)}→{sorted(kb)}"]
    return [
        f"support object {k[0]} Δ{1000 * d:.1f}"
        for k in sorted(ka)
        if (d := max(abs(float(u) - float(v)) for u, v in zip(ka[k], kb[k], strict=False))) > tol
    ]


def setup_statement_diff(
    stamped: list,
    current: list,
    z_off_stamped=None,
    z_off_current=None,
    tol: float = SETUP_STALE_TOL_M,
    excluded_stamped=None,
    excluded_current=None,
    supports_stamped=(),
    supports_current=(),
) -> list:
    """Differences between two support statements, over what the env and the gate consume: disc x·y·z·r · kind ·
    collide · pedestal height · z_off · the exclusions (as a set of (obj, why)) · the support objects (which object
    holds which main, and its frame-0 pose; None = a stamp without that field). Discs are matched as a set (nearest
    unused, max-abs over x·y·z·r). A stamped disc without a kind is an older stamp → reported. [] = the same statement.
    The item texts become stale-reason text (dexengine.placement.train_ready `T5:`)."""
    out = _supports_diff(supports_stamped, supports_current, tol)
    if excluded_stamped is not None or excluded_current is not None:
        es = None if excluded_stamped is None else sorted((str(x["obj"]), str(x["why"])) for x in excluded_stamped)
        ec = None if excluded_current is None else sorted((str(x["obj"]), str(x["why"])) for x in excluded_current)
        if es != ec:
            out.append(f"D40 exclusions {es}→{ec}")
    if (z_off_stamped is None) != (z_off_current is None):
        out.append(f"z_off {z_off_stamped}→{z_off_current}")
    elif z_off_stamped is not None and abs(float(z_off_current) - float(z_off_stamped)) > tol:
        out.append(f"z_off Δ{1000 * (float(z_off_current) - float(z_off_stamped)):+.1f}mm")
    if len(stamped) != len(current):
        out.append(f"discs {len(stamped)}→{len(current)}")
        return out
    left = list(current)
    for a in stamped:
        dev = [max(abs(float(a[k]) - float(b[k])) for k in ("x", "y", "z", "r")) for b in left]
        j = int(np.argmin(dev))
        b = left.pop(j)
        if dev[j] > tol:
            out.append(f"disc {a.get('obj', '?')} Δ{1000 * dev[j]:.1f}mm")
        if "kind" not in a:
            out.append(f"disc {a.get('obj', '?')} no kind (stamp from before D40)")
        elif a["kind"] != b.get("kind"):
            out.append(f"disc {a.get('obj', '?')} kind {a['kind']}→{b.get('kind')}")
        if a.get("collide") != b.get("collide"):  # the disc ↔ pedestal reading (visible · collides) is a statement
            out.append(f"disc {a.get('obj', '?')} collide {a.get('collide')}→{b.get('collide')}")
        elif b.get("h") is not None and (a.get("h") is None or abs(float(a["h"]) - float(b["h"])) > tol):
            out.append(f"disc {a.get('obj', '?')} pedestal height {a.get('h')}→{b.get('h')}")
    return out


def setup_statement(tree_ti: dict, base_ti: dict) -> dict:
    """The task_info a **re-stamp of this baked tree under the current base statement** would compute with: the tree's
    own task_info (layout · placement · pad · objects) with the base's SETUP_STATEMENT_KEYS laid over it. A base
    without one of those keys cannot state the current setup → ValueError (never the tree's old copy)."""
    miss = [k for k in SETUP_STATEMENT_KEYS if k not in base_ti]
    if miss:
        raise ValueError(f"base task_info has no setup statement {miss} — cannot state the current setup")
    return {**tree_ti, **{k: base_ti[k] for k in SETUP_STATEMENT_KEYS}}


def setup_stale(tree_dir, statement_ti: dict, scene: str) -> dict:
    """Is a baked tree's stamped support statement the one the current rule gives on this tree?
      stamped  the one reader — aerial: read_aerial_support · table: desk_support_discs (a baked desk tree without the
               stamp is the old generation, read as 0 discs). An aerial tree without the stamp cannot be read by the
               env at all → stale (`aerial_stamp_absent`).
      current  compute_aerial_support / compute_desk_support — what the bake stamp calls — on **this tree's own qpos**
               with `statement_ti` = setup_statement(tree, current base): what a re-stamp would write, so the disc x·y
               are comparable to the stamp.
    scene = the ledger axis' scene. A tree whose own `scene` says otherwise is stale.
    Returns {diff: [...] ([] = current), stamp, n_stamped, n_current, z_off_stamped, z_off_current}.
    Raises ValueError when the current statement cannot be evaluated — the caller counts it."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    ti = _json.loads((tree_dir / "task_info.json").read_text())
    if not is_baked_tree(ti):
        raise ValueError(f"{tree_dir}: the staleness check looks at baked trees only (no placement_v2)")
    own = str(ti.get("scene") or "table")
    rec = {"diff": [], "stamp": "", "n_stamped": None, "n_current": None, "z_off_stamped": None, "z_off_current": None}
    if scene == "aerial":
        try:
            st = read_aerial_support(tree_dir)
        except ValueError:
            rec.update(diff=["aerial_stamp_absent"], stamp="absent")
            return rec
        # the height is the stamped one — a re-stamp keeps it, so a height re-measured on the final trajectory may
        # differ; whether the height matches the judge is the judge input fingerprint's and the gate's. Only the discs
        # and exclusions are compared here, on that height.
        new = compute_aerial_support(tree_dir, ti=statement_ti, height=(st["z_ref"], st["source"]))
        rec.update(
            stamp="stamp",
            n_stamped=len(st["discs"]),
            n_current=len(new["discs"]),
            z_off_stamped=float(st["z_off"]),
            z_off_current=float(new["z_off"]),
        )
        rec["diff"] = setup_statement_diff(
            st["discs"],
            new["discs"],
            st["z_off"],
            new["z_off"],
            excluded_stamped=st["excluded"],
            excluded_current=new["excluded"],
            supports_stamped=st["supports"],
            supports_current=new["supports"],
        )
        return rec
    if own != scene:
        rec.update(diff=[f"scene {own}≠{scene}"], stamp=own)
        return rec
    with stdio.capture():  # the old-generation marker is counted by the caller (one line)
        st = desk_support_discs(tree_dir)
    new = compute_desk_support(tree_dir, ti=statement_ti)
    rec.update(stamp=st["source"], n_stamped=len(st["discs"]), n_current=len(new["discs"]))
    rec["diff"] = setup_statement_diff(
        st["discs"],
        new["discs"],
        excluded_stamped=st["excluded"] if st["source"] == "stamp" else [],
        excluded_current=new["excluded"],
        supports_stamped=st["supports"] if st["source"] == "stamp" else [],
        supports_current=new["supports"],
    )
    return rec


def scene_z_offset_inputs(tree_dir, ti: dict) -> list:
    """Every **data** file `scene_z_offset` reads besides the tree's own task_info (the f0 cache key reads this list).
    Same dispatch as scene_z_offset (scene_mode):
      table   → []  (desk_z lives in task_info, already in the key)
      aerial  → setup_inputs (the z reference is the floor of the first rest disc, and which disc rests is decided
                by the contact sidecars · MANO keypoints · meshes · surfaces · base tree)
    An unknown scene raises (a new scene must declare what its offset reads)."""
    mode = scene_mode(tree_dir)
    if mode == "table":
        return []
    if mode == "aerial":
        return list(setup_inputs(tree_dir, ti))
    raise ValueError(f"{tree_dir}: unknown z-offset inputs of scene {mode!r} — declare them in scene_z_offset_inputs")
