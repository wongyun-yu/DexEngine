"""Where each stage-1 artifact (table.ARTIFACTS) lives on a base tree, and the input key of a step that reuses a verdict
per tree: the bytes of every input it reads, so a verdict is reused only while all of them are the same.

Places (each artifact has one writer, table.check_inputs):
- task_info fields: the D1-g stamp ``rules.D1g`` (2b, oakink2.artic_convert), the case stamp ``case`` and the
  assignment record ``rules.ASSIGN`` (3, cases.taxonomy · assign_version.ASSIGN_KEY), the applied roles (4, the fields
  object_roles.apply rewrites: ROLE_TOP, ROLE_OBJECT of each ``objects[]`` element, ROLE_LAYOUT of each object element
  of ``qpos_layout[]``, ``rules.D23``, ``rules.D24``), the support-gap stamp under ``rules`` (5b, key
  support.gap.RULE_KEY).
- contact sidecar files: ``contact_all`` (2d) and ``contact_base`` (5), one file per arm side (paths.sidecar_file).
PLACES must name every ARTIFACTS entry (checked at import): an artifact the table gains without a place dies here.

`input_key(step, tree)` (the key a measured-only step stores per tree and compares to reuse its verdict) reads, all by
bytes (no time stamps): the kinematic qpos; the task_info as the steps up to `step` leave it (the whole file without the
fields of artifacts written by later steps — table.written_after); the sidecar files of the artifacts the step reads
(table.step_reads); the tree's scene files (scene.xml and every file it names, recursively for included xml); the
MANO keypoints of the tree's take (provenance.mano_dir); the folders of every object of the tree (its mesh_dir, its
surface samples folder, its USD folder — every regular file in each, f0's folder hash, once per process).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from dexengine.data.stage1 import table as T

TI, SIDECAR = "task_info", "sidecar"
#: artifact -> (kind, where): a task_info artifact names how to strip its fields (STRIP), a sidecar its layer.
PLACES = {
    "d1g": (TI, "d1g"),
    "contact_all": (SIDECAR, "contact_all"),
    "case": (TI, "case"),
    "assign": (TI, "assign"),
    "roles": (TI, "roles"),
    "contact_base": (SIDECAR, "contact_base"),
    "support_gap": (TI, "support_gap"),
}
KIN = "0/trajectory_kinematic.npz"
SCENE = "scene.xml"
_MEMO: dict = {}  # file sha by absolute path, once per process (objects, hands and takes are shared by many trees)


def owner_step(module: str, stages=None) -> str:
    """The id of the one step whose cell names `module` as its freshness owner (its cell is ``_INPUT_FP + module``)."""
    hit = [T.step_id(g) for g in (T.STAGES if stages is None else stages) if g[1] == T._INPUT_FP + module]
    if len(hit) != 1:
        raise SystemExit(f"[STAGE1_ORDER] {len(hit)} steps name {module} as their freshness owner (want 1): {hit}")
    return hit[0]


def step_reads(step: str, stages=None) -> tuple[str, ...]:
    """The artifacts step `step` declares it reads (its Run.reads, as check_inputs checked them)."""
    rows = [g for g in (T.STAGES if stages is None else stages) if T.step_id(g) == step and isinstance(g[3], T.Run)]
    if len(rows) != 1:
        raise SystemExit(f"[STAGE1_ORDER] step {step!r} is not one driven step")
    return tuple(rows[0][3].reads)


def written_after(step: str, stages=None) -> tuple[str, ...]:
    """The artifacts written by the driven steps after `step` in row order (what `step`'s input does not hold yet)."""
    ids = T.driven_steps(stages)
    if step not in ids:
        raise SystemExit(f"[STAGE1_ORDER] step {step!r} is not a driven step")
    later = set(ids[ids.index(step) + 1 :])
    return tuple(
        a
        for g in (T.STAGES if stages is None else stages)
        if T.step_id(g) in later
        for a in getattr(g[3], "writes", ())
    )


def _strip_rules(ti: dict, key: str) -> None:
    (ti.get("rules") or {}).pop(key, None)


def _strip_roles(ti: dict) -> None:
    from dexengine.data import object_roles as OR  # noqa: PLC0415 — owner of the fields apply rewrites

    for k in OR.ROLE_TOP:
        ti.pop(k, None)
    for o in ti.get("objects") or []:
        for k in OR.ROLE_OBJECT:
            o.pop(k, None)
    for e in ti.get("qpos_layout") or []:
        if e.get("kind") == "object":
            for k in OR.ROLE_LAYOUT:
                e.pop(k, None)
    for k in ("D23", "D24"):
        _strip_rules(ti, k)


def _strip(name: str, ti: dict) -> None:
    """Remove the fields of task_info artifact `name` from `ti` (in place)."""
    from dexengine.data.cases.assign_version import ASSIGN_KEY  # noqa: PLC0415
    from dexengine.data.support.gap import RULE_KEY as GAP_KEY  # noqa: PLC0415

    strip = {
        "d1g": lambda t: _strip_rules(t, "D1g"),
        "case": lambda t: t.pop("case", None),
        "assign": lambda t: _strip_rules(t, ASSIGN_KEY),
        "roles": _strip_roles,
        "support_gap": lambda t: _strip_rules(t, GAP_KEY),
    }
    strip[PLACES[name][1]](ti)


def _check_places() -> None:
    if set(PLACES) != set(T.ARTIFACTS):
        raise SystemExit(f"[STAGE1_INPUTS] artifact places {sorted(PLACES)} != table.ARTIFACTS {sorted(T.ARTIFACTS)}")


def file_sha(p: Path, memo: bool = False) -> str:
    """sha256 of a file's bytes ("-" when absent); `memo` keeps it for the process (files many trees share)."""
    k = str(Path(p).resolve())
    if memo and k in _MEMO:
        return _MEMO[k]
    if not Path(p).is_file():
        out = "-"
    else:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        out = h.hexdigest()
    if memo:
        _MEMO[k] = out
    return out


def ti_projection(ti: dict, after: tuple) -> str:
    """sha of task_info without the fields of the task_info artifacts in `after` (written by later steps)."""
    t = copy.deepcopy(ti)
    for a in after:
        if PLACES[a][0] == TI:
            _strip(a, t)
    return hashlib.sha256(json.dumps(t, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


_FILE_ATTR = re.compile(rb'\bfile="([^"]+)"')
_DIR_ATTR = re.compile(rb'\b(meshdir|texturedir|assetdir)="([^"]+)"')


def scene_files(xml: Path, seen: set | None = None) -> list[str]:
    """`name|sha` of an MJCF scene file and of every file it names (resolved like MuJoCo: meshdir/texturedir/assetdir
    relative to the file), recursively for included xml."""
    seen = set() if seen is None else seen
    xml = Path(xml)
    if str(xml) in seen:
        return []
    seen.add(str(xml))
    out = [f"{xml.name}|{file_sha(xml)}"]
    if not xml.is_file():
        return out
    b = xml.read_bytes()
    dirs = {m.group(1).decode(): m.group(2).decode() for m in _DIR_ATTR.finditer(b)}
    for m in _FILE_ATTR.finditer(b):
        rel = m.group(1).decode()
        base = dirs.get("meshdir") or dirs.get("assetdir") or ""
        cand = [xml.parent / base / rel, xml.parent / dirs.get("texturedir", base) / rel, xml.parent / rel]
        f = next((c for c in cand if c.is_file()), cand[0])
        if f.suffix == ".xml":
            out += scene_files(f, seen)
        else:
            out.append(f"{rel}|{file_sha(f, memo=True)}")
    return out


def _object_dirs(root: Path, ds: str, o: dict) -> list[Path]:
    """The folders an object's geometry is read from: its mesh_dir (visual.obj, convex parts, articulated links), its
    surface samples (fingerprint_core.surface_path), its USD (artifact_paths: the usd layer, usd_artic when
    articulated)."""
    from dexengine.core import artifact_paths as AP  # noqa: PLC0415 — owner of the asset id and the USD layers
    from dexengine.data.contact import fingerprint_core as FC  # noqa: PLC0415 — owner of the surface sample file
    from dexengine.placement.coords import setup as CSU  # noqa: PLC0415 — owner of the object file resolution

    out = []
    if o.get("mesh_dir"):
        out.append(Path(CSU.input_path(ds, Path(o["mesh_dir"]) / "visual.obj", root)).parent)
    if o.get("obj_id") is not None:
        out.append(FC.surface_path(root, ds, o).parent)
        tp, aid = AP.get(ds), AP.object_asset_id(o["obj_id"], o.get("articulation"))
        out.append(Path(tp.artic_usd_root() if o.get("articulation") else tp.usd_root()) / aid)
    return out


def _dir_sha(d: Path) -> str:
    """All regular files directly in a folder (name + bytes), once per process. For an object mesh folder this is wider
    than what a step opens: wider only means an occasional extra run, never a wrong reuse. (The f0 key no longer hashes
    whole folders, so the 2f key owns this hash; the bytes are those of the folder hash it read before.)"""
    k = ("dir", str(d))
    if k not in _MEMO:
        h = hashlib.sha256()
        for f in sorted(x for x in d.iterdir() if x.is_file()) if d.is_dir() else ():
            h.update(f"{f.name}|{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
        _MEMO[k] = h.hexdigest() if d.is_dir() else "ABSENT"
    return _MEMO[k]


def input_key(step: str, tree: Path, hand_dir: str) -> str:
    """The per-tree input key of `step` (module doc), 32 hex; files under the data root (paths.main_root, read now)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    t, root = Path(tree), Path(P.main_root())
    ti = json.loads((t / "task_info.json").read_text(encoding="utf-8"))
    ds = str(ti["dataset_name"])
    with np.load(t / KIN) as z:
        qpos = hashlib.sha256(np.ascontiguousarray(z["qpos"]).tobytes()).hexdigest()
    parts = [f"qpos|{qpos}", f"task_info|{ti_projection(ti, written_after(step))}"]
    for a in step_reads(step):
        if PLACES[a][0] == SIDECAR:
            parts += [
                f"{a}/{s}|{file_sha(P.sidecar_file(PLACES[a][1], ds, t.name, s, hand_dir))}" for s in ("right", "left")
            ]
    parts += [f"scene|{x}" for x in scene_files(t / SCENE)]
    mano = (ti.get("provenance") or {}).get("mano_dir")
    parts.append(f"mano|{file_sha(root / mano / '0/trajectory_keypoints.npz', memo=True) if mano else '-'}")
    for o in sorted(ti.get("objects") or [], key=lambda o: str(o.get("name"))):
        parts += [f"object/{o.get('name')}/{i}|{_dir_sha(d)}" for i, d in enumerate(_object_dirs(root, ds, o))]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


_check_places()
