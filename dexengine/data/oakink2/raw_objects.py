#!/usr/bin/env python3
"""OakInk2 raw-object supplements: raw -> base transform, export, root trees, part ids a tree already has, exported
files.

Source = anno_preview obj_transf (poses of every object of the scene). Which objects to supplement depends on the
branch:
  required objects (moved, touched or received by the person)  oakink2/required.py: raw object poses + raw MANO contact
  sibling parts (--parts)                                       the official part tree (oakink2/parts), contact-free
  supports (--supports)                                         under the main object at the first frame (support
      geometry)
The raw -> base transform is read, not fitted: base = Rz(yaw) * (Rx * raw + t_clip). Rx and t_clip come from
support.oakink2.clip_transform (INTAKE desk rule), yaw from the tree's frame.yaw_deg. Oracle: the delivered objects'
origins are moved with the same transform and compared with the base trajectory; residual p95 > RES_P95_MAX_M is an
error for that tree.
--retransform: rewrite every exported INTAKE/raw_objects/*.npz with the exact transform, keeping the same selection
      (the same file set). Done = rewritten == existing files; `--dry` only measures the change against the old poses.
--supports [--export]: support supplements (D42). For every tree of the axis hand's base (D36 pieces included) whose
      main object has no support candidate in the tree (coords.support_of), place the raw objects that are not in the
      tree at its first frame with the same transform and pick the object underneath with the support geometry
      (coords.on_support_object: support_contact, support polygon SUPPORT_POLY_PAD_M). With --export, each chosen
      (root tree, rid) gets the whole pose series of its root tree (pieces follow the parent_clip chain) as
      INTAKE/raw_objects/<root>__support__<rid>.npz; base_adapter loads it as a supplement and object_roles gives it
          role
      support. Done = written files == chosen pairs; per-tree exceptions are counted and give rc 1.

    $(paths.py --what retarget-py) dexengine/data/oakink2/raw_objects.py --parts [--export] [--limit N]
    $(paths.py --what retarget-py) dexengine/data/oakink2/raw_objects.py --supports [--export] [--report <json>]
    $(paths.py --what retarget-py) dexengine/data/oakink2/raw_objects.py --retransform [--dry --report <json>]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P
from dexengine.data import raw_sample as RS  # noqa: E402  (time grid and interpolation rule)
from dexengine.data.oakink2 import parts as OP  # noqa: E402
from dexengine.data.oakink2.raw_base import (
    HAND,
    RAW,
    ROOT,
    SEQ_KEY,
    _raw_verts,
    dl,
    export_npz,
    exported_rids,
    intake_clip,
    merge_tree_reference,
    oracle_residual,
    pose_base,
    raw_objects_dir,
    raw_supports_under,
    root_tree,
    scene_yaw,
    tree_part_ids,
    tree_transform,
)
from dexengine.data.support.oakink2 import (
    ClipError,
)


def retransform(dry: bool, report: str, only: set | None = None) -> int:
    """Recompute every existing INTAKE/raw_objects/*.npz (same selection) with the exact transform. Done = files
    rewritten (or measured) == existing files."""
    ed = ROOT / dl.render("intake", "oakink2") / "raw_objects"
    base = ROOT / dl.render("base", "oakink2", hand=HAND, emb="bimanual")
    files = sorted(f for f in ed.glob("*.npz") if not f.name.startswith(".") and not f.name.endswith(".tmp.npz"))
    if only is not None:  # for tests and partial re-measurement (python calls only; completion counts that subset)
        files = [f for f in files if f.name in only]
    seqs = {re.search(r"seq__([0-9a-f]+)__", f.stem).group(1)[:12]: f for f in (RAW / "anno_preview").glob("*.pkl")}
    by_seq = collections.defaultdict(list)
    rows = {}
    S = collections.Counter()
    for f in files:
        tree, tag, rid = f.stem.rsplit("__", 2)
        tif = base / tree / "task_info.json"
        if not tif.is_file():
            rows[f.name] = {"error": f"no tree in the axis-hand base ({HAND})"}
            continue
        ti = json.loads(Path(tif).read_text())
        m = SEQ_KEY.fullmatch(intake_clip(ti))
        by_seq[m.group(3)].append((f, tree, tag, rid, ti, int(m.group(4))))
    t0 = time.time()
    for i, (h, lst) in enumerate(sorted(by_seq.items()), 1):
        if h not in seqs:
            for f, *_ in lst:
                rows[f.name] = {"error": "source pkl missing"}
            continue
        ot = pickle.loads(Path(seqs[h]).read_bytes())["obj_transf"]
        keys = list(json.loads(Path(RAW / "program/program_info" / f"{seqs[h].stem}.json").read_text()))
        for f, tree, tag, rid, ti, stage in lst:
            _retransform_file(f, tree, tag, rid, ti, stage, ot, keys, base, dry, rows, S)
        if i % 50 == 0:
            print(f"  … seq {i}/{len(by_seq)} · {dict(S)} · {time.time() - t0:.0f}s", flush=True)
    return _retransform_summary(files, rows, S, dry, report)


def _retransform_file(f, tree, tag, rid, ti, stage, ot, keys, base, dry, rows, S) -> None:
    """Measure one exported file against the exact transform: its row goes into `rows` (an error row on a clip,
    key or value error) and, unless dry, the file is rewritten; `S["done"]` counts the files that passed."""
    try:
        iv = ast.literal_eval(keys[stage])
        ivs = [x for x in iv if x]
        a0 = min(x[0] for x in ivs)
        b0 = max(x[1] for x in ivs)
        kin = np.load(base / tree / "0/trajectory_kinematic.npz")["qpos"]
        T = len(kin)
        t50 = (np.arange(T) + ti["segment"]["start"]) * 0.02
        R, t = tree_transform(ti)
        res, osrc = oracle_residual(ti, kin, ot, t50, a0, R, t)
        if rid not in ot:
            raise ClipError(f"no {rid} pose in raw")
        fr2 = [k for k in range(a0, b0) if k in ot[rid]]
        if len(fr2) < 10:
            raise ClipError("too few raw frames")
        pos_b, Rb = pose_base(ot[rid], fr2, t50, a0, R, t)
        with np.load(f, allow_pickle=True) as z:
            old_pos, old_q, frac = z["pos"], z["quat_wxyz"], float(z["contact_frac"])
        if old_pos.shape != pos_b.shape:
            raise ClipError(f"pose length {old_pos.shape} ≠ {pos_b.shape}")
        dpos = np.linalg.norm(old_pos - pos_b, axis=1)
        Ro = Rot.from_quat(old_q[:, [1, 2, 3, 0]]).as_matrix()
        dang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Ro, Rb) - 1) / 2, -1, 1)))
        k = int(dang.argmax())
        V = _raw_verts(
            rid
            # surface displacement at that frame = max vertex difference of the mesh placed with the old and new
            # pose
        )
        dsurf = float(np.linalg.norm(V @ (Ro[k] - Rb[k]).T + (old_pos[k] - pos_b[k]), axis=1).max())
        rows[f.name] = {
            "tag": tag,
            "dpos_max_mm": round(float(dpos.max()) * 1000, 2),
            "dz_med_mm": round(float(np.median(pos_b[:, 2] - old_pos[:, 2])) * 1000, 2),
            "dang_max_deg": round(float(dang.max()), 2),
            "dsurf_max_mm": round(dsurf * 1000, 2),
            "oracle_res_mm": round(res * 1000, 3),
            "oracle": osrc,
        }
        if not dry:
            export_npz(tree, tag, rid, pos_b, Rb, frac, res, osrc, yaw=scene_yaw(ti))
        S["done"] += 1
    except (ClipError, KeyError, ValueError) as e:
        rows[f.name] = {"error": f"{type(e).__name__}: {e}"[:200]}


def _retransform_summary(files, rows, S, dry, report) -> int:
    """Marker, summary line and optional report of --retransform; rc 0 only when every file was done without an
    error."""
    err = {k: v for k, v in rows.items() if "error" in v}
    d = np.array([v["dpos_max_mm"] for v in rows.values() if "dpos_max_mm" in v])
    src = collections.Counter(v["oracle"] for v in rows.values() if "oracle" in v)
    print(RS.mark(S["done"], f"oakink2_raw_objects --retransform dry={int(dry)}"), flush=True)
    print(
        f"[RAW_OBJ_RETRANSFORM] files={len(files)} done={S['done']} errors={len(err)} dry={int(dry)} oracle="
        f"{dict(src)} "
        + (f"dpos_max_mm p50/p95/max={np.median(d):.2f}/{np.percentile(d, 95):.2f}/{d.max():.2f}" if d.size else ""),
        flush=True,
    )
    if report:
        Path(report).write_text(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "dry": dry,
                    "files": len(files),
                    "done": S["done"],
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=0,
            )
        )
    if err:
        print(f"[RAW_OBJ_RETRANSFORM] ✗ errors {len(err)} — e.g. {list(err.items())[:3]}", flush=True)
    return 0 if (S["done"] == len(files) and not err) else 2


def support_supplements(dest: Path | None, report: str, limit: int = 0) -> int:
    """Body of --supports (see the module docstring). Marker [RAW_SUPPORT] (main objects · tree candidates · raw support
    count · chosen pairs · files written)."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of the support geometry (placement layer)

    base = ROOT / dl.render("base", "oakink2", hand=HAND, emb="bimanual")
    seqs = {re.search(r"seq__([0-9a-f]+)__", f.stem).group(1)[:12]: f for f in (RAW / "anno_preview").glob("*.pkl")}
    by_seq = collections.defaultdict(list)
    for p in sorted(x for x in base.iterdir() if (x / "task_info.json").is_file()):
        ti = json.loads((p / "task_info.json").read_text())
        m = SEQ_KEY.fullmatch(intake_clip(ti))
        by_seq[m.group(3)].append((p, ti, int(m.group(4))))
    S, out, verts, picked = collections.Counter(), {}, {}, collections.defaultdict(set)
    for h, lst in sorted(by_seq.items()):
        if h not in seqs:
            S["seq_missing"] += len(lst)
            continue
        with open(seqs[h], "rb") as fh:
            ot = pickle.load(fh)["obj_transf"]
        keys = list(json.loads((RAW / "program/program_info" / f"{seqs[h].stem}.json").read_text()))
        for p, ti, stage in lst:
            _support_tree(C, base, p, ti, stage, keys, ot, verts, picked, out, S)
            if limit and S["mains"] >= limit:
                break
        if dest is not None:
            _export_supports(dest, lst, picked, keys, ot, S)
        if limit and S["mains"] >= limit:
            break
    return _support_summary(dest, picked, out, S, report)


def _support_tree(C, base, p, ti, stage, keys, ot, verts, picked, out, S) -> None:
    """Support candidates of one tree: for each main object without a tree candidate, the raw objects under it at
    the first frame go into `out` and `picked[root]`; a failure is counted in `S` (the first few printed)."""
    try:
        q = np.load(p / "0/trajectory_kinematic.npz")["qpos"]
        root = root_tree(base, p, ti)
        placed, _ = C._placers(p, ti, q)
        ivs = [x for x in ast.literal_eval(keys[stage]) if x]
        a0, b0 = min(x[0] for x in ivs), max(x[1] for x in ivs)
        t50 = (np.arange(len(q)) + ti["segment"]["start"]) * 0.02
        R, t = tree_transform(ti)
        have = tree_part_ids(ti) | exported_rids(root)
        mains = list({str(o["name"]): o for o in ti["objects"] if o.get("role") == "main"}.values())
        for o in mains:
            S["mains"] += 1
            if C.support_of(p, ti, q, o)["candidates"]:
                S["tree_candidate"] += 1
                continue
            raw_v0 = _raw_first_frame(ot, have, a0, b0, t50, R, t, verts)
            hits = raw_supports_under(placed(o, 0), raw_v0)
            S[f"raw_{min(len(hits), 2)}"] += 1
            out[f"{p.name}/{o['name']}"] = {"root": root, "raw": hits}
            if hits:
                picked[root].update(hits)
    except Exception as e:  # noqa: BLE001 — a single tree's failure is counted and the first few are printed
        S["error"] += 1
        if S["error"] <= 3:
            print(f"[RAW_SUPPORT] {p.name}: {type(e).__name__}: {str(e)[:160]}", flush=True)


def _raw_first_frame(ot, have, a0, b0, t50, R, t, verts) -> dict:
    """Base-frame vertices at the first frame of every raw object not already in the tree that has a mesh and at
    least 10 frames in the stage (meshes cached in `verts`)."""
    raw_v0 = {}
    for rid2, frames in ot.items():
        mp = RAW / "obj/object_repair/align_ds" / rid2 / "model.obj"
        fr2 = [k for k in range(a0, b0) if k in frames]
        if rid2 in have or len(fr2) < 10 or not mp.is_file():
            continue
        if rid2 not in verts:
            import trimesh  # noqa: PLC0415 — only the branch that reads meshes (--status reads clip keys only)

            verts[rid2] = np.asarray(trimesh.load(mp, force="mesh", process=False).vertices, np.float64)
        pos_b, Rb = pose_base(frames, fr2, t50, a0, R, t)
        raw_v0[rid2] = verts[rid2] @ Rb[0].T + pos_b[0]
    return raw_v0


def _export_supports(dest, lst, picked, keys, ot, S) -> None:
    """Chosen pair → pose sequence over the root tree's whole timeline (within the same sequence)."""
    for p, ti, stage in lst:
        for rid2 in sorted(picked.get(p.name, ())):
            _export_support(dest, p, ti, stage, rid2, keys, ot, S)


def _export_support(dest, p, ti, stage, rid2, keys, ot, S) -> None:
    """Write one chosen (tree, rid) support supplement; a failure is counted in `S` and printed."""
    try:
        kin = np.load(p / "0/trajectory_kinematic.npz")["qpos"]
        ivs = [x for x in ast.literal_eval(keys[stage]) if x]
        a0, b0 = min(x[0] for x in ivs), max(x[1] for x in ivs)
        t50 = (np.arange(len(kin)) + ti["segment"]["start"]) * 0.02
        R, t = tree_transform(ti)
        res, osrc = oracle_residual(ti, kin, ot, t50, a0, R, t)
        fr2 = [k for k in range(a0, b0) if k in ot[rid2]]
        pos_b, Rb = pose_base(ot[rid2], fr2, t50, a0, R, t)
        export_npz(p.name, "support", rid2, pos_b, Rb, -1.0, res, osrc, dest=dest, yaw=scene_yaw(ti))
        S["written"] += 1
    except Exception as e:  # noqa: BLE001 — one pair's failure is counted (rc 1) and printed; other pairs still write
        S["error"] += 1
        print(f"[RAW_SUPPORT] write {p.name}/{rid2}: {type(e).__name__}: {str(e)[:160]}", flush=True)


def _support_summary(dest, picked, out, S, report) -> int:
    """Marker line and optional report of --supports; rc 1 on any error or when fewer files were written than
    pairs chosen."""
    n_pairs = sum(len(v) for v in picked.values())
    short = dest is not None and S["written"] != n_pairs
    print(
        f"[RAW_SUPPORT] axis hand {HAND} · main objects {S['mains']} · with tree candidates "
        f"{S['tree_candidate']} · raw supports "
        f"0 {S['raw_0']} · one {S['raw_1']} · several {S['raw_2']} · chosen (root, rid) {n_pairs} · "
        f"root trees {len(picked)} · files written {S['written'] if dest is not None else '— (count only)'} · "
        f"no sequence {S['seq_missing']} · errors {S['error']}",
        flush=True,
    )
    if short:
        print(f"[RAW_SUPPORT] ✗ files written {S['written']} ≠ chosen pairs {n_pairs}", flush=True)
    if report:
        Path(report).write_text(
            json.dumps(
                {
                    "summary": dict(S),
                    "pairs": n_pairs,
                    "roots": {k: sorted(v) for k, v in sorted(picked.items())},
                    "mains": out,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
    return 1 if (S["error"] or short) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--export",
        action="store_true",
        help="with --parts/--supports: write the chosen objects' base-frame pose sequences to INTAKE/raw_objects/",
    )
    ap.add_argument(
        "--parts",
        action="store_true",
        help="D1-e: when a tree has any official part (dexengine.data.oakink2.parts), export **all sibling "
        "parts** with raw poses and meshes, regardless of contact (tag=part)",
    )
    ap.add_argument(
        "--retransform",
        action="store_true",
        help="rewrite every existing export with the same selection and the exact transform (bug repair)",
    )
    ap.add_argument(
        "--dry",
        action="store_true",
        help="with --retransform: write nothing, only measure the difference from the old poses",
    )
    ap.add_argument("--report", default="", help="per-file differences of --retransform · results of --supports (json)")
    ap.add_argument(
        "--supports",
        action="store_true",
        help="choose support supplements (D42, see the module docstring) · with --export, write to INTAKE",
    )
    a = ap.parse_args()
    if a.retransform:
        return retransform(a.dry, a.report)
    if a.supports:
        return support_supplements(raw_objects_dir() if a.export else None, a.report, a.limit)
    if a.dry or a.report:
        raise SystemExit("[raw_obj] --dry only with --retransform · --report only with --retransform/--supports")
    if not a.parts:
        raise SystemExit(
            "[raw_obj] choose a branch: --parts | --supports | --retransform — supplements of required "
            "objects (moved · touched · received) "
            "are in dexengine/data/oakink2/required.py"
        )
    return part_supplements(a.export, a.limit)


def part_supplements(export: bool, limit: int) -> int:
    """Body of --parts (D1-e): when a tree has any official part, all sibling parts with raw poses and meshes
    (regardless
    of contact, tag=part). Written into the tree's task_info.reference.sibling_parts and summarised in
    INTAKE/raw_objects_check.json."""
    trees = []
    for p in sorted((ROOT / dl.render("base", "oakink2", hand=HAND, emb="bimanual")).iterdir()):
        if not (p / "task_info.json").exists():
            continue
        ti = json.loads((p / "task_info.json").read_text())
        if any(OP.is_part(r) for r in tree_part_ids(ti) | exported_rids(p.name)):
            trees.append((p, ti))
    if limit:
        trees = trees[:limit]
    by_seq = collections.defaultdict(list)
    for p, ti in trees:
        m = SEQ_KEY.fullmatch(intake_clip(ti))
        by_seq[m.group(3)].append((p, ti, int(m.group(4))))
    seqs = {re.search(r"seq__([0-9a-f]+)__", f.stem).group(1)[:12]: f for f in (RAW / "anno_preview").glob("*.pkl")}
    print(f"[raw_obj] target trees {len(trees)} · sequences {len(by_seq)}", flush=True)
    out = {}
    S = collections.Counter()
    t0 = time.time()
    for i, (h, lst) in enumerate(sorted(by_seq.items()), 1):
        if h not in seqs:
            S["seq_missing"] += len(lst)
            continue
        d = pickle.loads(Path(seqs[h]).read_bytes())
        ot = d["obj_transf"]
        keys = list(json.loads(Path(RAW / "program/program_info" / f"{seqs[h].stem}.json").read_text()))
        for p, ti, stage in lst:
            _part_tree(p, ti, stage, keys, ot, export, out, S)
        if i % 20 == 0:
            print(f"  … seq {i}/{len(by_seq)} · {dict(S)} · {time.time() - t0:.0f}s", flush=True)
    return _write_parts_check(trees, out, S, limit)


def _part_tree(p, ti, stage, keys, ot, export, out, S) -> None:
    """Sibling parts of one tree: pose each sibling with raw poses and a mesh (exported with `export`), merge the
    list into the tree's task_info reference and record it in `out`; counters and failures go into `S`."""
    try:
        iv = ast.literal_eval(keys[stage])
        ivs = [x for x in iv if x]
        a0 = min(x[0] for x in ivs)
        b0 = max(x[1] for x in ivs)
        kin = np.load(p / "0/trajectory_kinematic.npz")["qpos"]
        T = len(kin)
        t50 = (np.arange(T) + ti["segment"]["start"]) * 0.02
        R, t = tree_transform(ti)
        res, osrc = oracle_residual(ti, kin, ot, t50, a0, R, t)
        have = tree_part_ids(ti) | exported_rids(p.name)
        sibs = sorted({s2 for h_ in have for s2 in OP.siblings(h_)} - have)
        parts_out = []
        for rid2 in sibs:
            if rid2 not in ot:
                S["part_no_pose"] += 1
                continue
            if not (RAW / "obj/object_repair/align_ds" / rid2 / "model.obj").is_file():
                S["part_no_mesh"] += 1
                continue
            fr2 = [k for k in range(a0, b0) if k in ot[rid2]]
            if len(fr2) < 10:
                S["part_short"] += 1
                continue
            pos_b, Rb = pose_base(ot[rid2], fr2, t50, a0, R, t)
            if export:
                export_npz(p.name, "part", rid2, pos_b, Rb, -1.0, res, osrc, yaw=scene_yaw(ti))
            parts_out.append({"raw_obj": rid2, "name": OP.name_of(rid2), "instance": OP.name_of(OP.root_of(rid2))})
            S["part_exported"] += 1
        fields = {"sibling_parts": parts_out, "raw_fit_residual_m": round(res, 4)}
        S["written" if merge_tree_reference(p / "task_info.json", fields) else "unchanged"] += 1
        out[p.name] = {"fit_residual_m": round(res, 4), "sibling_parts": parts_out}
        S["checked"] += 1
        S["found"] += bool(parts_out)
    except Exception as e:  # noqa: BLE001 — a single tree's failure goes into its row and is counted (rc 1)
        out[p.name] = {"error": f"{type(e).__name__}: {e}"}
        S["error"] += 1


def _write_parts_check(trees, out, S, limit) -> int:
    """INTAKE/raw_objects_check.json summary of --parts (a scoped name under --limit); rc 1 on any error."""
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "dexengine/data/oakink2/raw_objects.py --parts",
        "summary": dict(S),
        "n_trees": len(trees),
        "trees": out,
    }
    chk = P.clips_record(
        ROOT / dl.render("intake", "oakink2") / "raw_objects_check.json", bool(limit)
    )  # --limit = scoped run
    chk.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"[raw_obj] summary {dict(S)} → INTAKE/{chk.name}")
    return 1 if S["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
