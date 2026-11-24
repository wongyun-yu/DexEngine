"""Is a base tree current? Input fingerprints of the retarget output, the INTAKE support record and the supplements,
per tree (tree_stale_reason) and per dataset (freshness). Also owns which module states each dataset's desk-rule
version (DESK_RULE_OWNERS) and the INTAKE clips that version stamp applies to (stamp_targets)."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import InputMissingError
from dexengine.data.base_layer.common import (  # noqa: F401
    _PIECE_DEPTH,
    ADAPTER_VERSION,
    SUPPORT_COPIED,
    _dest,
    data_root,
    dl,
)
from dexengine.data.base_layer.supplements import (  # noqa: F401
    supplement_copy_reason,
    supplement_input,
    supplement_mark_reason,
    supplement_sample_reason,
)
from dexengine.data.objects import (
    scene_parts as SP,  # noqa: E402 — scene convex part list = decomposed now (owner of derivation and the stale check)
)


def _input_files(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.is_file())


def _input_stat(src: Path) -> str:
    """Cheap signature of a retargeting input folder: (relative path, size, mtime_ns) per file."""
    h = hashlib.sha256()
    for p in _input_files(src):
        st = p.stat()
        h.update(f"{p.relative_to(src).as_posix()}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def _abs_path_values(obj) -> set[str]:
    """String values (not keys) of a parsed JSON document that are absolute paths, at any depth."""
    if isinstance(obj, str):
        return {obj} if os.path.isabs(obj) else set()
    if isinstance(obj, dict):
        return set().union(*(_abs_path_values(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_abs_path_values(v) for v in obj))
    return set()


def content_bytes(p: Path) -> bytes:
    """The bytes the retargeting-input fingerprint hashes for one file — the file itself, except that in a JSON file
    every string value that is an absolute path is replaced by its basename: a recorded **location** is not content.
    The case that needs it: retarget_diagnostics.json `posture_prior.config_path` holds the absolute path of the code
    checkout that ran the retarget (a release or a rerun from another checkout path writes another path into
    byte-identical arrays). Only those values' bytes change; a file without one hashes as is. A value whose encoded
    form cannot be found in the bytes is an error (no silent fallback to the raw bytes)."""
    raw = p.read_bytes()
    if p.suffix != ".json":
        return raw
    try:
        paths = _abs_path_values(json.loads(raw))
    except ValueError:  # not JSON after all: its bytes are its identity
        return raw
    for v in sorted(paths, key=len, reverse=True):
        for enc in {json.dumps(v), json.dumps(v, ensure_ascii=False)}:
            if enc.encode() in raw:
                raw = raw.replace(enc.encode(), json.dumps(os.path.basename(v), ensure_ascii=False).encode())
                break
        else:
            raise ValueError(f"[base_layer] {p}: absolute path value {v!r} not found in its encoded form")
    return raw


def retarget_input(src: Path) -> dict:
    """**Fingerprint of the retargeting input** a base tree read: every file under the ingested clip folder (retarget
    layer), the whole folder rather than a list, so files adapt_clip starts reading are never missed: {stat: cheap
    signature (_input_stat), sha: (relative path, content sha256) per file, content = content_bytes (recorded absolute
    paths reduced to their basename, so the fingerprint does not depend on the checkout path that ran the retarget)}.
    The check (tree_stale_reason) treats an equal stat as the same input and otherwise decides by content (sha); a
    folder that was only touched is current."""
    return {"stat": _input_stat(src), "sha": fingerprint_sha(src)}


def fingerprint_sha(src: Path, read=content_bytes) -> str:
    """The `sha` of retarget_input: sha256 over (relative path, sha256 of `read(file)`) per file, 16 hex. `read` is
    content_bytes; the legacy-data migration passes the raw reader to verify a stamp made before content_bytes."""
    h = hashlib.sha256()
    for p in _input_files(src):
        h.update(p.relative_to(src).as_posix().encode() + b"\0" + hashlib.sha256(read(p)).digest())
    return h.hexdigest()[:16]


def input_stale_reason(prov: dict) -> str:
    """Whether the recorded retargeting input (provenance.retarget_dir · retarget_input) equals the current input: the
    reason when it differs, "" when equal. A missing record or input folder is stale."""
    rin, rdir = prov.get("retarget_input"), prov.get("retarget_dir")
    if not isinstance(rin, dict) or not rdir:
        return "no retargeting input fingerprint"
    src = data_root() / rdir
    if not src.is_dir():
        return f"retargeting input {rdir} missing"
    if _input_stat(src) == rin.get("stat") or retarget_input(src)["sha"] == rin.get("sha"):
        return ""
    return f"retargeting input {rdir} changed after base was built"


def tree_stale_reason(d: Path, depth: int = 0) -> str:
    """Whether one base tree was built with **the current ADAPTER_VERSION** from **the current retargeting input**: the
    first reason when stale, else "".

    A missing or different record (provenance.adapter_version) is stale (unknown is not current). Even on the same
    version, fields outside the current piece rule are stale (segment.pieces.schema_stale_reason, a content measurement
    independent of the version record). Even on the same version, an input fingerprint (provenance.retarget_input) that
    differs from the current retargeting folder is stale (input_stale_reason, so an old base is never silently reused
    after retargeting was rebuilt). A supplement pose fingerprint (provenance.supplement_input) that differs from the
    current INTAKE is stale (supplement_input: supplements exported later). INTAKE copied fields (SUPPORT_COPIED) that
    differ from the current INTAKE are stale (intake_stale_reason: desk recovery or the GRAB oracle rewritten after
    base).
    Pieces (trees where segment.split wrote `segment_source`) are window slices of their parent, not base_adapter
    output, so they are measured through **the parent tree**. Whether a piece copy equals its current parent is measured
    by dexengine.data.segment.invariant.piece_stale_reason (the PIECE_COPY copy, including provenance.adapter_version
    and hiding fields later steps rewrite per tree and build times · quality non-window fields · support parent fields ·
    scene · npz). The stage-1 driver (its piece re-slice step) re-slices **only the changed pieces** with that check at
    the start of each dataset and before 2d. If this column counted piece copies separately, one piece would put 2
    (rebuilding every parent of the dataset) into the plan."""
    try:
        ti = json.loads((d / "task_info.json").read_text())
    except (OSError, ValueError):
        return "task_info missing or unreadable"
    if "segment_source" in ti:
        parent = ti.get("parent_clip")
        if not parent or parent == d.name or depth >= _PIECE_DEPTH:
            return f"cannot follow the piece's parent {parent!r}"
        if not (d.parent / parent).is_dir():
            return f"the piece's parent {parent} is missing"
        why = tree_stale_reason(d.parent / parent, depth + 1)
        return f"parent {parent}: {why}" if why else ""
    prov = ti.get("provenance") or {}
    got = prov.get("adapter_version")
    if got != ADAPTER_VERSION:
        return f"adapter_version {got!r} ≠ {ADAPTER_VERSION}"
    # the version record is a **statement**; when the content (field set) is outside the current piece rule, it is an
    # old generation that changed fields without bumping the version. Pieces cannot be cut from such a parent
    # (segment.build.piece_task_info fails), so it counts as stale and 2 rebuilds it.
    from dexengine.data.segment import (
        pieces as PS,  # noqa: PLC0415 — home of the piece field declarations (no dependencies, the same declaration as segment.split)
    )

    why = PS.schema_stale_reason(ti)
    if why:
        return why
    why = input_stale_reason(prov)
    if why:
        return why
    sup = supplement_input(str(ti.get("dataset_name")), d.name)
    if prov.get("supplement_input") != sup:
        return (
            f"supplement pose fingerprint {prov.get('supplement_input')!r} ≠ current {sup!r}"
            " (INTAKE raw_objects changed after base)"
        )
    why = intake_stale_reason(ti, d.name)
    if why:
        return why
    why = supplement_copy_reason(ti)
    if why:
        return why
    why = supplement_mark_reason(ti, d.name)
    if why:
        return why
    why = supplement_sample_reason(ti)
    if why:
        return why
    from dexengine.data.artic import (
        convert_version as DR,  # noqa: PLC0415 — owner of the D1-g version (no dependencies)
    )

    got = ((ti.get("rules") or {}).get("D1g") or {}).get("version")
    if got is not None and got not in (DR.D1G_VERSION, DR.D1G_PER_TREE):
        # a tree merged by the in-place conversion (2b) under an old version; without the source part poses the
        # converter cannot rewrite it, so base is rebuilt.
        return f"D1-g old version {got!r} ≠ {DR.D1G_VERSION} (merged trees go from 2b after base is rebuilt)"
    return SP.stale_reason(
        d
    )  # scene convex part list ↔ decomposition now (re-decomposing an object makes the trees listing it stale)


#: Clip fields of INTAKE support_audit.json, per dataset (file key (path, mtime_ns, size), clips).
#: Read again when the file changes.
_INTAKE_SUPPORT: dict = {}


def _intake_support(ds: str) -> dict | None:
    p = _dest("intake", ds) / "support_audit.json"
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    key = (str(p), st.st_mtime_ns, st.st_size)
    if _INTAKE_SUPPORT.get(ds, (None,))[0] != key:
        _INTAKE_SUPPORT[ds] = (key, json.loads(p.read_text())["clips"])
    return _INTAKE_SUPPORT[ds][1]


#: dataset -> (module that owns its desk rule version RULE_VERSION, name of that module's attribute holding the marker
#: field its writer stamps | None). Versions and marker values belong to those modules; this table only says which
#: owner to ask. A clip carrying the marker field must also carry the version. Other datasets have no desk-rule owner.
DESK_RULE_OWNERS = {
    "oakink2": ("dexengine.data.support.oakink2_rule", None),
    "hrdexdb": ("dexengine.data.support.hrdexdb", "MARK"),
}


def desk_rule_owner(ds: str) -> tuple[str, str, str | None] | None:
    """(owner module, its RULE_VERSION, marker field | None) of `ds`'s desk rule, None for a dataset without one. An
    owner module that does not import or lacks the marker attribute raises (no invented field)."""
    if ds not in DESK_RULE_OWNERS:
        return None
    name, mark_attr = DESK_RULE_OWNERS[ds]
    mod = importlib.import_module(name)
    return name, mod.RULE_VERSION, (getattr(mod, mark_attr) if mark_attr else None)


def intake_clips(ds: str) -> dict:
    """The clip records of INTAKE support_audit.json of `ds`; a missing file raises."""
    sup = _intake_support(ds)
    if sup is None:
        raise InputMissingError(f"INTAKE {ds}/support_audit.json missing ({_dest('intake', ds)})")
    return sup


def stamp_targets(owner: tuple[str, str, str | None] | None, clips: dict) -> set[str]:
    """The clips the desk-rule version stamp applies to: clips with `desk_rule` plus clips carrying the owner writer's
    marker field (owner = desk_rule_owner(ds), read once by the caller)."""
    held = {c for c, r in clips.items() if "desk_rule" in r}
    mark = owner[2] if owner else None
    return held | ({c for c, r in clips.items() if mark in r} if mark else set())


def intake_stale_reason(ti: dict, tree_name: str) -> str:
    """The reason when a base tree's INTAKE copied fields (SUPPORT_COPIED) differ from that take's record in the current
    INTAKE support_audit.json, else "". Measured by support_drift alone. When INTAKE is fixed after base is built
    (desk recovery · GRAB desk oracle rewritten), the tree holds the old fields even though the retargeting input is
    unchanged, and the stage-1 column 2 rebuilds it. A missing INTAKE file is stale (base is never built without
    INTAKE)."""
    ds = str(ti.get("dataset_name"))
    sup = _intake_support(ds)
    if sup is None:
        return f"INTAKE {ds}/support_audit.json missing"
    drift = support_drift(sup.get(P.parent_clip(ti.get("parent_clip") or tree_name)), ti.get("support"))
    return (
        f"INTAKE copied fields {drift[:4]} differ from base (INTAKE changed after base, support_drift)" if drift else ""
    )


def freshness(ds: str, spec_dir: str) -> tuple[int, int]:
    """(trees built on the current version, all tree folders of this hand's base layer); owner of the data.stages
    stage-1 column 2.

    The layer location is opened through the same accessor (_dest) as the writer. Without any tree the denominator
    counts as 1: not '0/0 = complete' but an empty step (the stage-1 axis treats only x≠y as an empty step)."""
    fresh = total = 0
    for root in dict.fromkeys(_dest("base", ds, hand=spec_dir, emb=e) for e in dl.EMBODIMENTS):
        for d in root.iterdir() if root.is_dir() else ():
            if d.is_dir():
                total += 1
                fresh += not tree_stale_reason(d)
    return fresh, max(total, 1)


def support_drift(sup: dict | None, s: dict | None) -> list[str]:
    """The copied support fields where a tree (`s` = task_info.support) no longer equals its INTAKE record (`sup` —
    support_audit.json clips[<take>]): a field support_for copies must be present iff INTAKE has it, with the same
    value.
    Catches an INTAKE fixed after the hand's base was built (a desk recovery that never reached the tree). [] =
    current."""
    if sup is None or s is None:
        return ["no INTAKE record" if sup is None else "no support"]
    return [k for k in SUPPORT_COPIED if (k in sup) != (k in s) or sup.get(k) != s.get(k)]


def support_for(sup: dict | None, a: int, b: int, off: int, psi_deg: float) -> dict | None:
    if sup is None:
        return None
    keep = SUPPORT_COPIED
    # D14′ wrist depth is **not** a support field: it is measured per tree window by the case stage
    # (case.wrist_fp — dexengine.data.support.grab.wrist_footprint); the ledger keeps only the table-level
    # ORACLE_FIELDS.
    s = {k: sup.get(k) for k in keep if k in sup}
    fp = sup.get("footprint")
    if fp:
        c = np.array([[fp[0], fp[1]], [fp[2], fp[1]], [fp[2], fp[3]], [fp[0], fp[3]]])
        M = Rot.from_euler("z", psi_deg, degrees=True).as_matrix()[:2, :2]
        r = c @ M.T
        s["footprint"] = [round(float(x), 3) for x in (*r.min(0), *r.max(0))]
        s["footprint_capture"] = fp
    else:
        s["footprint"] = None
    # the window slicing has one owner, support.rest.slice_rest_intervals; the D36 split (segment.split) calls the same
    # function
    from dexengine.data.support.rest import slice_rest_intervals

    ri = slice_rest_intervals(sup.get("rest_intervals"), a, b, off)
    s["rest_intervals"] = ri
    s["source"] = "INTAKE/support_audit.json"
    s["frame"] = "base (yaw aligned, no z shift; the scene z offset is the placement stage)"
    return s
