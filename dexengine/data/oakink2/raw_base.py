"""OakInk2 raw objects, shared layer: roots, part ids, the raw -> base transform, the oracle residual and the
supplement export (owner for raw_objects.py and required.py; standard roots resolved once at import)."""

from __future__ import annotations

import collections
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P
from dexengine.data import raw_sample as RS
from dexengine.data import rules as RR
from dexengine.data.oakink2 import parts as OP
from dexengine.data.support.oakink2 import (
    MOCAP_HZ,
    RES_P95_MAX_M,
    RULE_VERSION,
    ClipError,
    clip_fit_residual,
    clip_transform,
)

# The reference base is the axis hand's layer (RR.AXIS_HAND), not the running hand: the outputs (INTAKE/raw_objects)
# are shared by the dataset and base_layer.hand_dir.supplements_for adds them to every hand (the base frame is
# hand-independent).
dl = P.data_layout
ROOT = P.main_root()
HAND = P.hand_spec_dir(RR.AXIS_HAND)
RAW = ROOT / dl.render("raw", "oakink2") / "hf"


rid_of = OP.rid_of  # part identity rule (right_/left_/raw_ prefixes)


def tree_part_ids(ti: dict) -> set:
    """Raw object ids the tree already carries: objects[] ids, the parts merged into articulations (articulation.parts
    base_id, moving_id, fixed[].id) and the parts a merged mesh covers (fixed_merge.parts). D1-g removes merged parts
    from objects[]; without this set they would be supplemented again. Every export filters with it."""
    out = set()
    for o in ti.get("objects") or []:
        fm = o.get("fixed_merge")
        for i in fm["parts"] if fm else [o["obj_id"]]:
            out.add(rid_of(i))
        parts = (o.get("articulation") or {}).get("parts") or {}
        for i in [parts.get("base_id"), parts.get("moving_id"), *[f.get("id") for f in parts.get("fixed") or []]]:
            if i:
                out.add(rid_of(i))
    return out


def exported_rids(tree_name: str) -> set:
    """rids already exported under this tree name (any tag); they count as present although the base has not been
    rebuilt yet, so the same (tree, rid) is never written twice."""
    import glob as _glob  # noqa: PLC0415

    return {
        f.stem.rsplit("__", 2)[2]
        for f in raw_objects_dir().glob(f"{_glob.escape(tree_name)}__*.npz")
        if not f.name.startswith(".")
    }


def retire_unowned(req: dict, roots: set, export: bool, full: bool) -> tuple[collections.Counter, set]:
    """Retire supplement files with an ownerless tag (outside OP.SUPP_TAGS: right · left of the old contact branch). For
    a
    required object of a measured root: delete, and this run rewrites it under the category tag (same transform); not a
    required object: delete; not a measured root: delete only in a full run. Ledger = INTAKE/raw_objects_retired.tsv
    (appended); without --export, count only. req = {(root, rid): category} (dexengine.data.oakink2.required's required
    objects) · roots = measured roots → (count, (root, rid) to rewrite)."""
    S, led, again = collections.Counter(), [], set()
    for f in sorted(raw_objects_dir().glob("*.npz")):
        root, tag, rid = f.stem.rsplit("__", 2)
        if f.name.startswith(".") or tag in OP.SUPP_TAGS:
            continue
        if root in roots:
            why = f"reexport:{req[(root, rid)]}" if (root, rid) in req else "drop:not_required"
        elif full:
            why = "drop:no_root"
        else:
            S["unowned_unmeasured"] += 1
            continue
        S["retire_" + why.split(":")[0]] += 1
        led.append((f.name, root, tag, rid, why))
        if why.startswith("reexport"):
            again.add((root, rid))
    if export and led:
        ledger = raw_objects_dir().parent / "raw_objects_retired.tsv"
        head = "" if ledger.exists() else "time\tfile\troot\ttag\trid\tdecision\n"
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(head + "".join("\t".join((time.strftime("%Y-%m-%dT%H:%M:%S%z"),) + x) + "\n" for x in led))
        for x in led:
            (raw_objects_dir() / x[0]).unlink()
    return S, again


def pose_base(frames: dict, fr2: list, t50, a0: int, R, t):
    """Raw per-frame 4x4 poses -> 50 Hz base-frame series (base = R * raw + t; raw_sample grid: linear pos, slerp)."""
    T2 = np.array([frames[k] for k in fr2])
    tr2 = (np.array(fr2) - a0) / MOCAP_HZ
    pos, Rm = RS.sample(T2, tr2, t50)
    return pos @ R.T + t, np.einsum("ij,tjk->tik", R, Rm)


def raw_objects_dir() -> Path:
    """INTAKE/raw_objects: where the supplement pose series live (read by base_layer.hand_dir.supplements_for)."""
    return ROOT / dl.render("intake", "oakink2") / "raw_objects"


def export_npz(
    tree_name: str,
    tag: str,
    rid2: str,
    pos_b,
    Rb,
    frac: float,
    res: float,
    oracle: str,
    dest: Path | None = None,
    yaw: float | None = None,
):
    """Base-frame pose sequence to <dest>/<tree>__<tag>__<rid>.npz (no dest = raw_objects_dir); tag ∈ right/left =
    contact
    supplement, part = sibling-part supplement D1-e, support = support supplement D42. fit_residual_m · oracle = p95
    residual of the transform oracle and its source (oracle_residual) · transform = transform version (desk_recover
    RULE_VERSION) · sample = sampling rule version (raw_sample; readers reject old samples by this field). Write = temp
    file → replace.
    yaw = the scene rotation this pose sequence was placed with (scene_yaw); written as a field (YAW_KEY) only with the
    repair switch (raw_sample.YAW_FIX_ENV) on (off: same fields as old files · on without a yaw: fails)."""
    extra = {}
    if RS.yaw_fix_on():
        if yaw is None:
            raise SystemExit(f"{RS.YAW_MARK} {tree_name}__{tag}__{rid2}: switch on but no scene rotation was given")
        extra[RS.YAW_KEY] = np.float64(yaw)
    qw = Rot.from_matrix(Rb).as_quat()[:, [3, 0, 1, 2]]
    ed = Path(dest) if dest is not None else raw_objects_dir()
    ed.mkdir(parents=True, exist_ok=True)
    f = ed / f"{tree_name}__{tag}__{rid2}.npz"
    tmp = f.with_name(f".{f.name}.tmp.npz")  # dot prefix = outside the base_adapter glob (<tree>__*.npz)
    np.savez(
        tmp,
        rid=rid2,
        pos=pos_b.astype(np.float64),
        quat_wxyz=qw.astype(np.float64),
        contact_frac=np.float32(frac),
        mesh_src=str((RAW / "obj/object_repair/align_ds" / rid2 / "model.obj").relative_to(ROOT)),
        frame="base(yaw)",
        fit_residual_m=np.float32(res),
        oracle=oracle,
        transform=f"Rz(yaw)·(Rx·raw + t_clip) desk_rule {RULE_VERSION}",
        sample=RS.RAW_SAMPLE,
        **extra,
    )
    os.replace(tmp, f)


SEQ_KEY = re.compile(r"oakink2_(.+?)__scene_(\d+)_([0-9a-f]{12})_s(\d+)")


def intake_clip(ti: dict) -> str:
    """INTAKE clip key of a tree = the `task` base_adapter wrote (segments and D36 pieces keep the source clip).
    `parent_clip` is the direct parent, i.e. the segment name for a segment's pieces, which is not an INTAKE key. A key
    that
    does not match the naming rule (SEQ_KEY) or a parent that does not start with it fails."""
    k = ti.get("task") or ""
    if not SEQ_KEY.fullmatch(k) or not str(ti.get("parent_clip", "")).startswith(k):
        raise ClipError(f"cannot determine the INTAKE clip key: task={k!r} · parent_clip={ti.get('parent_clip')!r}")
    return k


_VERTS: dict = {}


def _raw_verts(rid: str) -> np.ndarray:
    """Vertices of the source object mesh (align_ds); the ruler that turns pose differences into surface displacement
    (mm)."""
    if rid not in _VERTS:
        import trimesh  # noqa: PLC0415 — read only by the rewrite report

        _VERTS[rid] = np.asarray(
            trimesh.load(RAW / "obj/object_repair/align_ds" / rid / "model.obj", force="mesh", process=False).vertices
        )
    return _VERTS[rid]


_OWNER_YAW: dict = {}


def scene_yaw(ti: dict) -> float:
    """Scene rotation (°) for placing supplements. Off (default) = tree record frame.yaw_deg (old behaviour). On
    (raw_sample.YAW_FIX_ENV) = owner base_layer.hand_dir.yaw_for (reference hand direction dexengine.data.align.REF_DEG
    − dataset hand
    direction); more than YAW_TOL_DEG from the tree record fails (that tree's delivered objects sit at another rotation;
    supplement vs delivered object direction; rebuild base first)."""
    rec = float(ti["frame"]["yaw_deg"])
    if not RS.yaw_fix_on():
        return rec
    ds = ti.get("dataset_name") or "oakink2"
    if ds not in _OWNER_YAW:
        from dexengine.data.base_layer import hand_dir as BA  # noqa: PLC0415 — scene rotation owner (only when on)

        _OWNER_YAW[ds] = BA.yaw_for(ds)[0]
    want = _OWNER_YAW[ds]
    RS.yaw_mark(f"oakink2_raw_objects ds={ds}", want)
    if abs(rec - want) > RS.YAW_TOL_DEG:
        raise ClipError(
            f"{RS.YAW_MARK} tree rotation {rec:.4f}° ≠ reference {want:.4f}° (base_adapter.yaw_for) — this tree's "
            "delivered objects sit at another rotation; rebuild base first"
        )
    return want


def tree_transform(ti: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of raw → base: base = Rz(yaw)·(Rx·raw + t_clip) · yaw = scene_yaw(ti). t_clip is the source clip's
    (intake_clip), so segments and pieces share the transform."""
    Rx, tc = clip_transform(intake_clip(ti))
    Rz = Rot.from_euler("z", scene_yaw(ti), degrees=True).as_matrix()
    return Rz @ Rx, Rz @ tc


def oracle_residual(ti: dict, kin, ot: dict, t50, a0: int, R, t) -> tuple[float, str]:
    """Transform oracle: (p95 residual m, source). Source order (the most direct evidence available):
      tree_rigid      move the origin of the tree's delivered rigid body (not a supplement) with the same transform and
          compare with the base trajectory
      tree_artic_root compare the root pose of a delivered articulated object (D1-g; = the pose of that obj_id part)
      clip            the tree has no delivered object pose (only supplement-based articulations) → the transform
          residual desk_recover measured on the same clip (INTAKE desk_rule)
    Above the threshold (RES_P95_MAX_M): ClipError. The source is written to the output (which evidence passed is not
    hidden).
    The **rotation** of the same object is measured too: when our rotation (pose_base) and the tree's disagree by more
    than a constant (beyond raw_sample.DRIFT_MAX_DEG), our time grid or interpolation differs from the delivery →
    ClipError (measuring origins alone would pass misaligned rotation samples)."""
    for want_artic in (False, True):
        for o in ti["objects"]:
            if o.get("supplement") or bool(o.get("articulation")) != want_artic:
                continue
            rid = rid_of(o["obj_id"])
            if rid not in ot:
                continue
            fr = sorted(ot[rid])
            Ts = np.array([ot[rid][k] for k in fr])
            traw = (np.array(fr) - a0) / MOCAP_HZ
            ok = (t50 >= traw[0]) & (t50 <= traw[-1])
            if ok.sum() < 10:
                continue
            rp = np.stack([np.interp(t50[ok], traw, Ts[:, k, 3]) for k in range(3)], 1) @ R.T + t
            q0 = int(o["qpos_start"])
            res = float(np.percentile(np.linalg.norm(kin[ok, q0 : q0 + 3] - rp, axis=1), 95))
            if res > RES_P95_MAX_M:
                raise ClipError(
                    f"transform oracle: {o['obj_id']} origin residual p95 {res * 1000:.2f} mm > "
                    f"{RES_P95_MAX_M * 1000:.1f}"
                )
            drift = RS.drift_deg(
                pose_base(ot[rid], fr, t50[ok], a0, R, t)[1],
                Rot.from_quat(kin[ok, q0 + 3 : q0 + 7][:, [1, 2, 3, 0]]).as_matrix(),
            )
            if drift.max() > RS.DRIFT_MAX_DEG:
                raise ClipError(
                    f"{RS.MARK} transform oracle: {o['obj_id']} rotation is a pose from another time than the "
                    f"delivery — max {drift.max():.2f}° > "
                    f"{RS.DRIFT_MAX_DEG}° ({int((drift > RS.DRIFT_MAX_DEG).sum())}/{len(drift)} frames)"
                )
            return res, ("tree_artic_root" if want_artic else "tree_rigid")
    return clip_fit_residual(intake_clip(ti)), "clip"


def render_only_tracks(seq: tuple, sk: tuple, ti: dict, kin, rids: list) -> tuple[dict, dict]:
    """Base trajectories of render-only objects (dexengine/record/scene/render_only_build.py: source objects outside the
    tree), with **the same transform and time grid** as the support supplement (--supports) (tree_transform · pose_base
    · oracle
    oracle_residual against that base tree's kin). seq = (program, anno_preview) of
    dexengine.data.readers.raw.oakink2.load_seq · sk = (stage, a0, b0) of that window.
    -> ({rid: {pos (T,3) | None, wxyz (T,4) | None, have (T,) bool, mesh}}, {residual_m, oracle}). have = the row's time
    lies within that object's raw frame range; pose_base's sampling extrapolates outside it, so those rows are not
    poses."""
    ot = seq[1]["obj_transf"]
    a0, b0 = int(sk[1]), int(sk[2])
    t50 = (np.arange(len(kin)) + ti["segment"]["start"]) * 0.02
    R, t = tree_transform(ti)
    res, osrc = oracle_residual(ti, kin, ot, t50, a0, R, t)
    out = {}
    for rid in rids:
        frames = ot.get(rid) or {}
        fr2 = [k for k in range(a0, b0) if k in frames]
        mp = RAW / "obj/object_repair/align_ds" / rid / "model.obj"
        if len(fr2) < 2:
            out[rid] = {"pos": None, "wxyz": None, "have": np.zeros(len(kin), bool), "mesh": str(mp)}
            continue
        pos_b, Rb = pose_base(frames, fr2, t50, a0, R, t)
        traw = (np.asarray(fr2, float) - a0) / MOCAP_HZ
        q = Rot.from_matrix(Rb).as_quat()
        out[rid] = {
            "pos": pos_b,
            "wxyz": q[:, [3, 0, 1, 2]],
            "have": (t50 >= traw[0]) & (t50 <= traw[-1]),
            "mesh": str(mp),
        }
    return out, {"residual_m": float(res), "oracle": osrc}


def root_tree(base: Path, p: Path, ti: dict) -> str:
    """Name of the root tree base_adapter built; D36 pieces (segment_source) climb the parent_clip chain (supplements
    load
    under the root name and pieces inherit objects from their parent). A broken or cyclic chain fails."""
    name, cur = p.name, ti
    for _ in range(4):
        if "segment_source" not in cur:
            return name
        name = str(cur.get("parent_clip") or "")
        if not (base / name / "task_info.json").is_file():
            raise ClipError(f"piece {p.name}: parent {name!r} not in base")
        cur = json.loads((base / name / "task_info.json").read_text())
    raise ClipError(f"parent chain of piece {p.name} longer than 4 levels")


def raw_supports_under(Vm, raw_v0: dict) -> list:
    """The one selection rule of support supplements: raw objects (rid → that tree's first-frame vertices, capture
    frame)
    that hold up the main-object vertices Vm (coords.on_support_object; same geometry as the support kind check
    support_of, with polygon slack). No name branches."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of the support geometry (placement layer)

    return [rid for rid, V in raw_v0.items() if C.on_support_object(Vm, [V])]


def merge_tree_reference(tif: Path, fields: dict) -> bool:
    """The only writer of fields onto a tree's task_info.reference. The file is read again **right before** writing and
    the
    fields go on top; equal values are not written. --parts reads every axis-hand tree at start and writes tens of
    minutes
    later while iterating sequences, so writing back the start-time copy would drop quality markers (4b · 5b) the axis
    hand's stage 1 wrote in between (lost update). This only narrows the read–write window to milliseconds and is not a
    lock; keeping clear of the axis hand's stage 1 is up to launch order. Atomic replace (tmp → os.replace) as
    before. Returns whether it wrote."""
    cur = json.loads(tif.read_text())
    ref = cur.setdefault("reference", {})
    if all(k in ref and ref[k] == v for k, v in fields.items()):
        return False
    ref.update(fields)
    tmp = tif.with_name(tif.name + ".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
    os.replace(tmp, tif)
    return True
