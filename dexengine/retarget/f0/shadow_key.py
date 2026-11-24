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
FP_DESIGN = "f0_in_v3"


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


#: Fields under reference that f0 does not read: the sibling-part record and raw-fit residual that the OakInk2 official
#: part supplement (dexengine.data.oakink2.raw_objects --parts, D1-e) adds to the axis-hand base. They changed only the
#: key, not the output bytes. Excluding them leaves current keys unchanged (no current base tree has these fields in
#: reference; measured over 65,776 trees).
TI_UNREAD_REFERENCE = ("sibling_parts", "raw_fit_residual_m")


def ti_projection(ti: dict) -> dict:
    """The f0 input part of task_info (everything except TI_UNREAD_*)."""
    out = {k: v for k, v in ti.items() if k not in TI_UNREAD_TOP}
    if isinstance(out.get("support"), dict):
        out["support"] = {k: v for k, v in out["support"].items() if k not in TI_UNREAD_SUPPORT}
    if isinstance(out.get("reference"), dict):
        out["reference"] = {k: v for k, v in out["reference"].items() if k not in TI_UNREAD_REFERENCE}
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


def _dir_sha(d: Path) -> str:
    """All regular files directly in a folder (name + bytes). For an object mesh folder this is wider than what f0 opens
    (visual.obj, surface_points.npz, articulated part files): wider only means an occasional extra solve, never a wrong
    reuse."""
    k = ("dir", str(d))
    if k not in _GEOM_MEMO:
        if not d.is_dir():
            _GEOM_MEMO[k] = "ABSENT"
        else:
            h = hashlib.sha256()
            for f in sorted(x for x in d.iterdir() if x.is_file()):
                h.update(f"{f.name}|{_file_sha(f)}\n".encode())
            _GEOM_MEMO[k] = h.hexdigest()
    return _GEOM_MEMO[k]


def _f0_mod():
    """repair_cfg (single owner of the object and hand geometry paths), loaded once when needed. The hand is set by
    DEXENGINE_HAND (GPU workers load the same module with the same env, worker_env)."""
    from dexengine.retarget.f0 import repair_cfg  # noqa: PLC0415

    return repair_cfg


def hand_geom_sha(side: str) -> str:
    """Hand geometry = the URDF f0 uses (repair_cfg.HC.urdf) and every mesh file in it (filename=, resolved like
    HandModel via paths.urdf_mesh) + the finger ranges changed by the retargeting cut (repair_cfg.finger_limits; for
    uncut hands the key is unchanged). Once per process."""
    k = ("hand", side)
    if k not in _GEOM_MEMO:
        F = _f0_mod()
        u = Path(F.HC.urdf(side))
        h = hashlib.sha256(f"{F.HC.name}|{u.name}|{_file_sha(u)}\n".encode())
        for fn in sorted(set(re.findall(r'filename="([^"]+)"', u.read_text()))):
            p = _PATHS.urdf_mesh(u, fn)
            h.update(f"{fn}|{_file_sha(p) if p.is_file() else 'ABSENT'}\n".encode())
        vendor, ref = F.finger_limits()  # finger box = reference range; a cut range enters the key
        cut = {j: ref[j] for j in sorted(ref) if ref[j] != vendor[j]}
        if cut:
            h.update(f"finger_range|{json.dumps(cut, sort_keys=True)}\n".encode())
        _GEOM_MEMO[k] = h.hexdigest()
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
        k for k in set(old_parts) | set(new_parts) if k != "ti_fields" and old_parts.get(k) != new_parts.get(k)
    )
    of, nf = old_parts.get("ti_fields") or {}, new_parts.get("ti_fields") or {}
    fld = sorted(k for k in set(of) | set(nf) if of.get(k) != nf.get(k))
    out = [f"task_info[{','.join(fld)}]" if (c == "task_info" and fld) else c for c in comp]
    return " ".join(out) if out else "components_equal"


def f0_input_key(
    clip: str, stage_src: Path, wrist_dirs: dict, dataset: str, src_layer: str, parts: dict | None = None
) -> str:
    """Single owner of the f0 cache key (v2): content hash of the inputs f0 actually reads; no mtime, size or inode.
    Measured by the same function when writing (finish -> marker `input_key`) and when deciding reuse (cache_verdict).
    Contents: the staging npz files f0 reads (f0_read_stage_files), the f0 input part of task_info (ti_projection,
    sorted-key JSON), the wrist-cache bytes per manipulating side (clip_sides), the main object mesh folders
    (repair_cfg.main_object_dirs), the hand geometry (hand_geom_sha) and the pad finger speed (HC.f_cap_rad_s; when the
    HandSpec changes, the shadows go stale with it).

    parts (default None = unchanged behaviour): a dict receives per-component hashes (design, stage, task_info,
    pad_vcap, wrist/<side>, obj/<side>, hand/<side>, zin) and per-field task_info hashes (ti_fields). They stay in the
    marker as `input_parts`, so the next STALE reason says what changed (input_key_diff). The key bytes do not depend on
    parts: the same lines feed both the key hash and the component hashes."""
    _QL = _PATHS.qpos_layout
    src = stage_src / clip
    h = hashlib.sha256()
    _ph: dict = {}

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
        mesh_dir, sp_dir, _art = F.main_object_dirs(src, lay, side, dataset)
        _feed(
            f"obj/{side}",
            f"obj/{side}|{_dir_sha(Path(mesh_dir))}|{_dir_sha(Path(sp_dir)) if sp_dir else '-'}\n".encode(),
        )
        _feed(f"hand/{side}", f"hand/{side}|{hand_geom_sha(side)}\n".encode())
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
    if parts is not None:
        parts.update({k: v.hexdigest()[:16] for k, v in sorted(_ph.items())})
        parts["ti_fields"] = _ti_field_digests(_proj)
    return h.hexdigest()[:32]


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
    k_old = str(j.get("input_key", ""))
    if k_old:
        return ("FRESH", "", key_now, j) if k_old == key_now else ("STALE", "input_changed", key_now, j)
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
