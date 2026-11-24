"""f0 shadow inputs and cache key: the files f0 reads and rewrites, the hands present in a tree, the task_info
projection, the v2 content key (f0_input_key) and the reuse verdict (cache_verdict)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from dexengine.core import paths as _PATHS
from dexengine.data.objects import scene_parts as SP
from dexengine.placement.judge.f0_key import MARKER
from dexengine.retarget.f0 import repair_root as RT
from dexengine.retarget.f0.shadow_status import SIDES, SourceDefectError

# Source layer (--src-layer). `base` is the only source layer (grounding for 9 datasets).
SRC_LAYERS = ("base",)  # base is the only source layer


# The base layer's shadow scene.xml refers to meshes by absolute path (rewritten on copy).
SCENE_XMLS = SP.SCENE_XMLS


_COMPILER_TAG_RE = re.compile(r"<compiler\b[^>]*>")


_DIR_ATTR_RE = re.compile(r'\b(meshdir|texturedir)="([^"]*)"')


def rewrite_scene_dirs(xml: str, clip_dir: Path) -> str:
    """Turn the relative directories of `<compiler meshdir=… texturedir=…>` into **absolute paths
    anchored on the original clip**.

    If there is no compiler tag or no directory attribute, and if the directory pointed at does not
    exist, raise — silently returning the original text would leave mujoco unable to find the meshes
    at the shadow depth. The object convex part list must equal the current decomposition — otherwise the base
    tree is stale and the copy raises (scene_parts.verify_text).
    """
    SP.verify_text(xml, clip_dir, str(clip_dir))
    m = _COMPILER_TAG_RE.search(xml)
    if not m:
        raise ValueError(f"{clip_dir}: scene xml has no <compiler> tag")
    n = 0

    def _sub(mm):
        nonlocal n
        n += 1
        p = (clip_dir / mm.group(2)).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"{clip_dir}: {mm.group(1)}={mm.group(2)!r} → {p} is not a directory")
        return f'{mm.group(1)}="{p}/"'

    tag = _DIR_ATTR_RE.sub(_sub, m.group(0))
    if n == 0:
        raise ValueError(f"{clip_dir}: <compiler> has no meshdir/texturedir")
    return xml[: m.start()] + tag + xml[m.end() :]


def written_files(src_layer: str) -> set[str]:
    """Files that must be **real copies** in the shadow (a hard link would pollute the original, or the
    content has to be rewritten). The base layer rewrites the two scene xml files with absolute paths, so
    they are included here (and excluded from the inode comparison in shadow_is_current too)."""
    w = {
        "0/trajectory_kinematic.npz",
        "0/trajectory_ikrollout.npz",
        "0/contact_mano_right.npz",
        "0/contact_mano_left.npz",
    }
    if src_layer != "base":
        raise ValueError(f"unknown src_layer {src_layer!r} — {SRC_LAYERS}")
    w |= set(SCENE_XMLS)
    return w


# Name of the fingerprint convention. When the convention (what is hashed, and how) changes, bump this
# value — old fingerprints then automatically mismatch and are regenerated.
# v2: cache key = content hash of the inputs f0 actually reads (`f0_input_key`). v1 = (size, mtime) of every file in
# the input folder, a proxy under which a role stage rewriting an unchanged task_info invalidated the whole cache.
# v3: the pad finger speed (repair_cfg.HC.f_cap_rad_s = HandSpec applied limit / stretch.SAFETY) joins the key (an
# input outside the code).
# v4: only what f0 reads — the object part = the files ObjectGeom opens (repair_geom.object_input_files; v3 hashed the
# whole object folder, so a stray file beside surface_points.npz moved it), the hand part = what HandModel reads
# (repair_geom.hand_input_lines: collision blocks, their meshes, the base -> link joints, the reference finger ranges;
# v3 hashed the whole robot URDF and every mesh in it, so a material edit moved it), and task_info without the
# `reference` block (no f0 module reads it; its `time` stamp moved v3). Measured on canonical shadows: those
# three moved v3 keys of shadows whose f0 output was byte-identical on rerun. Each opened file's hash is recorded
# (input_parts.files), so the next STALE reason names the file. v3 markers are not migrated: they
# regenerate.
FP_DESIGN = "f0_in_v4"


FP_DESIGN_V1 = "src_fp_v1"  # old convention, read only when migrating v1 markers (cache_verdict)


def clip_sides(stage_src: Path, clip: str) -> tuple[str, ...]:
    """The hands actually present in the tree (owned by qpos_layout). A two-hand tree = ("right","left"),
    a single-hand emb tree = that one hand (2a).
    The wrist-cache requirement, the copying and f0.repair --side all follow this — requiring the cache of an absent
    hand makes every single-hand dataset FAIL_MIRROR."""
    from dexengine.core.paths import qpos_layout as _QL

    lay = _QL.read(str(stage_src / clip), nq=None)
    out = tuple(
        s for s in SIDES if s in lay.manip_sides()
    )  # D23: even when the hand exists, a side with no main object (inactive) is not an f0 target
    if not out:
        raise SourceDefectError(
            f"{stage_src / clip}: no manipulating hand (hands={lay.hands} main={lay.main}) — D23 inactive. "
            "A first-pass no_main_object exclusion"
        )
    return out


def src_fingerprint(clip: str, stage_src: Path, wrist_dirs: dict, ti_override: Path | None = None) -> str:
    """Fingerprint of f0's **two input branches** — the v1 cache key (cache_verdict uses it to migrate
    v1 markers; the v2 key is f0_input_key).

    Why it is needed: skipping on the marker (`F0_PREJUDGE.json`) alone means that even when the
    upstream staging or the wrist cache is repaired, the shadow **silently keeps using the stale
    values** (f0 then sees the world from before the repair and refuses, e.g. "74 pad frames needed").

    What goes in: **everything f0 actually reads**. Every file in the staging clip directory and
    **both sides of the wrist cache** — the wrist cache is exactly the side that can go stale while
    the staging shadow is current.

    What it measures with: (relative path, size, mtime_ns). Not a content hash but a cheap proxy,
    the same convention as make and rsync. It is the choice that avoids hashing a 6.6GB tree on
    every run, and "same size and mtime but different content" does not happen in this pipeline
    (the generator always writes anew).
    """
    parts: list[str] = []
    base = stage_src / clip
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        # v1 is now used only for migration (cache_verdict). ti_override = the task_info from f0 time kept in the
        # shadow; re-measuring with that file's (size, mtime) shows whether the marker's v1 value is reproduced.
        st = (ti_override if (ti_override is not None and str(p.relative_to(base)) == "task_info.json") else p).stat()
        parts.append(f"stage/{p.relative_to(base)}|{st.st_size}|{st.st_mtime_ns}")
    for side in clip_sides(stage_src, clip):
        w = wrist_dirs[side] / f"{clip}.npz"
        if w.exists():
            st = w.stat()
            parts.append(f"wrist/{side}|{st.st_size}|{st.st_mtime_ns}")
        else:
            parts.append(f"wrist/{side}|MISSING")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


#: task_info keys (top level and under support) that f0 does not read; excluded from the key (f0 output is unchanged
#: when they change). Evidence (code enumeration): none of these keys appears as a string in the repo modules the f0
#: process loads (prejudge, gpu, repair, coords, qpos_layout, paths, data_layout, hands, arms, datasets,
#: ledger); the "rules" in datasets is the name of its own table.
#: strace measurement (one clip): f0 opens task_info.json, 0/trajectory_kinematic.npz, 0/trajectory_ikrollout.npz, the
#: wrist cache (per side), the main object's visual.obj and surface_points.npz, and the hand URDF with its meshes. The
#: two scene xml files are moved with rewritten paths on mirroring (f0 does not read them), and body_positions and
#: retarget_diagnostics are only hard-linked; these four are pass-through files, kept out of the key and re-synced to
#: the current base on a cache hit (refresh_passthrough; the judge reads the shadow's task_info).
#: Every key not listed here is in the key (an unknown new field is conservatively an input, never wrongly reused).
TI_UNREAD_TOP = (
    # `reference` (the retargeting reference record: source, expected/observed sides, its time stamp): no f0 module
    # reads it (the readers are data.filtering and record.text.source_labels; guard test_f0_key_reads_no_reference);
    # its time stamp alone moved keys (measured, TACO dg5f_s).
    "reference",
    "case",
    "rules",
    "provenance",
    "segment",
    "segment_range",
    "segment_source",
    "clip_id",
    "parent_clip",
    "idle_hand_injected",
)


TI_UNREAD_SUPPORT = (
    "obj_zmin_traj",
    "obj_zmin_mesh_traj",
    "obj_zmin_hull_traj",  # D24 input (candidate rule only)
    # D36 piece window marker (segment.build.piece_support), read only by the auditor (not by coords or
    # f0); excluded so pieces that only gain the marker keep their f0 cache. rest_intervals itself
    # stays in the key (aerial f0 reads it through coords).
    "rest_intervals_window",
)


def ti_projection(ti: dict) -> dict:
    """The f0 input part of task_info (everything except TI_UNREAD_*)."""
    out = {k: v for k, v in ti.items() if k not in TI_UNREAD_TOP}
    if isinstance(out.get("support"), dict):
        out["support"] = {k: v for k, v in out["support"].items() if k not in TI_UNREAD_SUPPORT}
    return out


def f0_read_stage_files(src_layer: str) -> tuple:
    """Staging npz files f0 reads = the npz files f0 rewrites (it reads them to prepend frames). Owner:
    written_files."""
    return tuple(sorted(r for r in written_files(src_layer) if r.endswith(".npz")))


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


_GEOM_MEMO: dict = {}  # geometry hashes once per process (thousands of clips share the same objects and hands)


def _sha_memo(p: Path) -> str:
    """Content hash of one geometry file (or ABSENT), once per process (clips share an object and a hand)."""
    k = ("file", str(p))
    if k not in _GEOM_MEMO:
        _GEOM_MEMO[k] = _file_sha(p) if p.is_file() else "ABSENT"
    return _GEOM_MEMO[k]


def _f0_mod():
    """repair_cfg (single owner of the object and hand geometry paths), loaded once when needed. The hand is set by
    DEXENGINE_HAND (GPU workers load the same module with the same env, worker_env)."""
    from dexengine.retarget.f0 import repair_cfg  # noqa: PLC0415

    return repair_cfg


def object_key(mesh_dir, sp_dir, articulation) -> tuple:
    """(key text, {opened file: file hash}) of one main object — the files ObjectGeom opens (owner of the list:
    repair_geom.object_input_files), each named and hashed (ABSENT when an optional one is missing)."""
    from dexengine.retarget.f0 import repair_geom as G  # noqa: PLC0415

    fo = {n: _sha_memo(Path(p)) for n, p in sorted(G.object_input_files(mesh_dir, sp_dir, articulation).items())}
    return "|".join(f"{n}={v}" for n, v in fo.items()), {n: v[:16] for n, v in fo.items()}


def hand_key(side: str) -> tuple:
    """(hash, {mesh filename: file hash}) of what HandModel reads for this side (owner of the list:
    repair_geom.hand_input_lines). Once per process."""
    k = ("hand", side)
    if k not in _GEOM_MEMO:
        from dexengine.retarget.f0 import repair_geom as G  # noqa: PLC0415

        lines, files = G.hand_input_lines(side)
        fh = {n: _sha_memo(Path(p)) for n, p in sorted(files.items())}
        h = hashlib.sha256("\n".join(lines).encode())
        for n, v in fh.items():
            h.update(f"\nmesh|{n}|{v}".encode())
        _GEOM_MEMO[k] = (h.hexdigest(), {n: v[:16] for n, v in fh.items()})
    return _GEOM_MEMO[k]


def _ti_field_digests(proj: dict) -> dict:
    """Per-field short hashes of the f0 input part of task_info (ti_projection), the marker's `input_parts.ti_fields`.
    One entry per top-level key; a non-empty dict goes one level down as `<key>.<subkey>` (to show which field moved the
    key). Not part of the key computation; a diagnostic record (the key bytes are defined by f0_input_key)."""

    def _d(v) -> str:
        return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]

    out = {}
    for k, v in proj.items():
        if isinstance(v, dict) and v:
            out.update({f"{k}.{k2}": _d(v2) for k2, v2 in v.items()})
        else:
            out[k] = _d(v)
    return out


def input_key_diff(old_parts, new_parts: dict) -> str:
    """Difference between the marker's `input_parts` and the current components -> one line (appended to the STALE
    reason). Does not decide anything (the decision is input_key). An old marker without components, and all components
    equal with a different key (a difference outside the component list, i.e. the convention), each give a fixed reason
    word; reason words are recorded and keep their exact text."""
    if not isinstance(old_parts, dict) or not old_parts:
        return "no_component_record"
    comp = sorted(
        k
        for k in set(old_parts) | set(new_parts)
        if k not in ("ti_fields", "files") and old_parts.get(k) != new_parts.get(k)
    )
    of, nf = old_parts.get("ti_fields") or {}, new_parts.get("ti_fields") or {}
    fld = sorted(k for k in set(of) | set(nf) if of.get(k) != nf.get(k))
    oF, nF = old_parts.get("files") or {}, new_parts.get("files") or {}

    def _files(c):  # the opened files of an object or hand part that moved (v4 markers record them)
        a, b = oF.get(c) or {}, nF.get(c) or {}
        return sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))

    out = [
        f"task_info[{','.join(fld)}]"
        if (c == "task_info" and fld)
        else (f"{c}[{','.join(_files(c))}]" if _files(c) else c)
        for c in comp
    ]
    return " ".join(out) if out else "components_equal"


def f0_input_key(
    clip: str, stage_src: Path, wrist_dirs: dict, dataset: str, src_layer: str, parts: dict | None = None
) -> str:
    """Single owner of the f0 cache key (v2): content hash of the inputs f0 actually reads; no mtime, size or inode.
    Measured by the same function when writing (finish -> marker `input_key`) and when deciding reuse (cache_verdict).
    Contents: the staging npz files f0 reads (f0_read_stage_files), the f0 input part of task_info (ti_projection,
    sorted-key JSON), the wrist-cache bytes per manipulating side (clip_sides), the files the main object loader opens
    (object_key), what the hand model reads (hand_key) and the pad finger speed (HC.f_cap_rad_s; when the
    HandSpec changes, the shadows go stale with it) and, only for a clip the object repair acts on, its part
    (repair.obj_acted_part: policy + cooked hull digest per acting side — keyed on what the repair does, so every other
    clip keeps its key under the switch on or off).

    parts (default None = unchanged behaviour): a dict receives per-component hashes (design, stage, task_info,
    pad_vcap, wrist/<side>, obj/<side>, hand/<side>, zin, obj_repair), per-field task_info hashes (ti_fields) and the
    hash of each file the object and hand parts opened (files: {obj/<side> | hand/<side>: {file: hash}}). They
    stay in the marker as `input_parts`, so the next STALE reason says what changed (input_key_diff). The key bytes do
    not depend on parts: the same lines feed both the key hash and the component hashes."""
    _QL = _PATHS.qpos_layout
    src = stage_src / clip
    h = hashlib.sha256()
    _ph: dict = {}
    _files: dict = {}  # component -> {opened file: hash} (input_parts.files)

    def _feed(part: str, line: bytes) -> None:
        h.update(line)
        if parts is not None:
            _ph.setdefault(part, hashlib.sha256()).update(line)

    _feed("design", f"{FP_DESIGN}|{dataset}|{src_layer}\n".encode())
    for rel in f0_read_stage_files(src_layer):
        p = src / rel
        _feed("stage", f"stage/{rel}|{_file_sha(p) if p.is_file() else 'ABSENT'}\n".encode())
    ti = json.loads((src / "task_info.json").read_text())
    _proj = ti_projection(ti)
    _feed("task_info", b"task_info|" + json.dumps(_proj, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    lay = _QL.read(str(src), nq=None)
    F = _f0_mod()
    _feed("pad_vcap", f"pad_vcap|{F.HC.f_cap_rad_s!r}\n".encode())
    for side in clip_sides(stage_src, clip):
        w = wrist_dirs[side] / f"{clip}.npz"
        _feed(f"wrist/{side}", f"wrist/{side}|{_file_sha(w) if w.is_file() else 'MISSING'}\n".encode())
        ok, of = object_key(*F.main_object_dirs(src, lay, side, dataset))
        _feed(f"obj/{side}", f"obj/{side}|{ok}\n".encode())
        hk, hf = hand_key(side)
        _feed(f"hand/{side}", f"hand/{side}|{hk}\n".encode())
        _files[f"obj/{side}"] = of
        _files[f"hand/{side}"] = hf
    # Files opened by f0's z offset (coords.scene_z_offset): aerial scenes go through the support computation and read
    # the contact sidecar and MANO keypoints. Without them in the key, a rebuilt sidecar would let aerial shadows reuse
    # a z solved from the old sidecar as a 'hit'. The list is owned by coords.scene_z_offset_inputs (desk scene = [], so
    # desk keys are unchanged). The tree's own files are already in (task_info only as its f0 input part) and are
    # skipped.
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    root = _PATHS.main_root()
    for p in sorted({str(x) for x in C.scene_z_offset_inputs(src, ti)}):
        pp = Path(p)
        if str(pp).startswith(str(src) + os.sep):
            continue
        rel = os.path.relpath(str(pp), str(root)) if str(pp).startswith(str(root) + os.sep) else str(pp)
        fp = C.input_path(dataset, pp)  # opened file (relative = object ref -> data_layout); names, order as declared
        _feed("zin", f"zin/{rel}|{_file_sha(fp) if fp.is_file() else 'ABSENT'}\n".encode())
    _feed_obj_acted(_feed, clip, stage_src, wrist_dirs, dataset)
    if parts is not None:
        parts.update({k: v.hexdigest()[:16] for k, v in sorted(_ph.items())})
        parts["ti_fields"] = _ti_field_digests(_proj)
        parts["files"] = _files
    return h.hexdigest()[:32]


def _feed_obj_acted(_feed, clip: str, stage_src: Path, wrist_dirs: dict, dataset: str) -> None:
    """The object repair's key part (repair.obj_acted_part), fed only when the repair acts on this clip — the last line
    of the key, so a clip it leaves alone keeps its key byte for byte."""
    from dexengine.retarget.f0 import repair as R  # noqa: PLC0415 — the entry module (its gate), loaded when keyed

    sides = clip_sides(stage_src, clip)
    roots = {Path(wrist_dirs[s]).parent for s in sides}
    if len(roots) > 1 or any(Path(wrist_dirs[s]).name != f"wrist_poses_{s}" for s in sides):
        raise ValueError(f"[F0_OBJ] wrist caches outside one <root>/wrist_poses_<side>/ layout: {wrist_dirs}")
    acted = R.obj_acted_part(clip, stage_src, roots.pop(), dataset, sides) if sides else ""
    if acted:
        _feed("obj_repair", f"obj_repair|{acted}\n".encode())


#: cache_verdict's STALE reason for a marker that predates the input fingerprint (prejudge compares against it).
WHY_NO_FINGERPRINT = "no_fingerprint(pre_convention)"


def cache_verdict(
    clip: str,
    stage_src: Path,
    wrist_dirs: dict,
    dst_tree: Path,
    dataset: str,
    src_layer: str,
    parts_out: dict | None = None,
) -> tuple:
    """Single owner of the shadow reuse decision -> (verdict, reason, current key, marker dict | None).
    Verdicts: FRESH (v2 key matches), MIGRATE (strict migration of a v1 marker, below), STALE, NOMARK.

    Migration (v1 -> v2) does not guess. If the old key (size + mtime) still holds, the inputs f0 read are unchanged
    (exactly the criterion under which v1 returned CACHED): (1) re-measure the v1 fingerprint on the current base; equal
    means migrate. (2) taco case: when only the base task_info was os.replace'd, the shadow still hard-links the
    task_info from f0 time. Re-measure v1 with that file's (size, mtime); if the marker's value is reproduced, the other
    inputs are unchanged and the task_info of that time is the shadow file, and the migration happens only when the f0
    input parts (ti_projection) of the two are equal. If nothing is proven: STALE (solve again). parts_out (default
    None): a dict receives the per-component hashes of the current inputs (f0_input_key parts); the verdict is
    unchanged. Reason strings are recorded in the ledger and keep their exact text."""
    d = dst_tree / clip
    mark = d / MARKER
    key_now = (
        f0_input_key(clip, stage_src, wrist_dirs, dataset, src_layer)
        if parts_out is None
        else f0_input_key(clip, stage_src, wrist_dirs, dataset, src_layer, parts=parts_out)
    )
    if not mark.exists():
        return "NOMARK", "", key_now, None
    try:
        j = json.loads(mark.read_text())
    except (ValueError, OSError):
        return "STALE", "marker_damaged", key_now, None
    k_old, unwritten = str(j.get("input_key", "")), RT.unwritten(d)
    if k_old or unwritten:
        why = RT.WHY_UNWRITTEN if unwritten else ("" if k_old == key_now else "input_changed")
        return ("STALE" if why else "FRESH"), why, key_now, j
    fp_old = str(j.get("src_fingerprint", ""))
    if not fp_old:
        return "STALE", WHY_NO_FINGERPRINT, key_now, j
    if fp_old == src_fingerprint(clip, stage_src, wrist_dirs):
        return "MIGRATE", "v1→v2 migration (v1 fingerprint holds = inputs unchanged)", key_now, j
    sh_ti = d / "task_info.json"
    if sh_ti.is_file() and fp_old == src_fingerprint(clip, stage_src, wrist_dirs, ti_override=sh_ti):
        _now = json.loads((stage_src / clip / "task_info.json").read_text())
        if ti_projection(json.loads(sh_ti.read_text())) == ti_projection(_now):
            return (
                "MIGRATE",
                "v1→v2 migration (v1 reproduced with the shadow task_info · f0 input fields equal)",
                key_now,
                j,
            )
        return "STALE", "f0 input fields of task_info changed", key_now, j
    return "STALE", "v1→v2 migration impossible (no proof of the inputs)", key_now, j
