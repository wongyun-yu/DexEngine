#!/usr/bin/env python3
"""OakInk2 rigid-joint output: merges the fixed parts of the per-object joint spec (oakink2.part_kinds) with their body
into **one mesh and one rigid body** (the consumer contract stays: one rigid body · one mesh · obj_id = asset id).

Why: one thing with missing connecting geometry (lamp base – head) left as two rigid bodies lets the head float or
fall. The articulated path assumes one joint, and multi-mesh rigid bodies are outside the contract of 30 consumers, so
the asset side solves it.

Rules
  relative pose = **clip median** of the part pose in the body frame (position median · rotation chordal mean); only the
             representative point of the asset pose, not a judgement statistic. It differs per clip even for one object
             (lamp heads turn; SharpA 82 trees: between-clip deviation median 65 mm · 61°, within clip 71/82 inside the
             band) → one asset **per relative pose** (the spec stage assigns them in sorted order; published assets are
                 fixed).
  statistic = series_err alone (max difference from the asset pose over all frames, m · °); assignment (spec stage
             merge), oracle (converter pick) and wobble (rigid) all read it. Judge = bake contract: the judgement is
             "inside the band during the clip" (frame max), so assignment uses the frame max too (assigning by distance
             between medians would give trees to assets the judgement rejects).
  thresholds = audit (oracle) = the band itself (position POS_TOL_M · rotation ROT_TOL_DEG) · production (reuse) = band
      ×
             REUSE_FRAC. Production is looser **toward covering** (when unsure it makes a new asset, so production never
             drops a tree the audit would accept). A tree outside the audit band even against its own asset (wobble >
             band) gets no asset, and the converter fails with the reason (merge_attach_moves).
  asset id  = raw_<root rid>_f<pose sha 8> (e.g. raw_o02_0045_00003_f1a2b3c4); it holds the root rid (rid_of gives the
             root, not a part). Mesh = body visual + part visuals (moved by the relative pose) in one visual.obj (gaps
                 kept;
             two separate meshes are still one rigid body and do not float). Convex decomposition, USD and cooked dump
             follow the rigid path (bake owner).
  oracle    = per tree, series_err ≤ the audit band (D1-g thresholds: position 1 cm · rotation 5°); otherwise that tree
      is
             not that asset (without a matching asset the converter fails).
Provenance: FIXED.json in the asset folder (root · body · parts · relative poses · band · source mesh sha) + the `fixed`
section of the spec file.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.data.oakink2 import part_kinds as PK  # noqa: E402
from dexengine.data.oakink2 import parts as OP  # noqa: E402
from dexengine.data.oakink2.artic_fit import POS_TOL_M, ROT_TOL_DEG, block  # noqa: E402

TAG = "[FIXED_MERGE]"
SECTION = "fixed"  # section of the spec file (artic_ranges.json), separate from the joint assets (objects)
#: Production (reuse) threshold = audit band × this ratio. Both read the same statistic, so the slack only has to cover
#: per-hand tree copies (the spec stage reads the first hand's, the converter its own) and pose rounding.
REUSE_FRAC = 0.9
#: Decimal digits of the vertices in the merged visual.obj (build_mesh); the corner-match tolerance of the cover
#: generator
#: derives from it (half a step × 2 = 10^-OBJ_DIGITS · dexengine.robots.objects.render_overlay._st), so changing it
#: moves that tolerance too.
OBJ_DIGITS = 6
POS_ATOL = 10.0**-OBJ_DIGITS


def rel_series(q: np.ndarray, base_o: dict, part_o: dict):
    """Part pose sequence in the body frame (positions (T,3), rotations Rotation)."""
    a, b = block(q, base_o), block(q, part_o)
    Ra, Rb = Rot.from_quat(a[:, [4, 5, 6, 3]]), Rot.from_quat(b[:, [4, 5, 6, 3]])
    return Ra.inv().apply(b[:, :3] - a[:, :3]), Ra.inv() * Rb


def clip_pose(series) -> dict:
    p, R = series
    w = R.mean().as_quat()
    return {"p": [round(float(x), 6) for x in np.median(p, 0)], "q": [round(float(x), 6) for x in w[[3, 0, 1, 2]]]}


def series_err(series, pose: dict) -> tuple[float, float]:
    """**The one statistic**: max difference (m, °) from pose `pose` over all frames of the relative pose sequence; read
    by
    assignment, the oracle and wobble."""
    p, R = series
    Rc = Rot.from_quat(np.asarray(pose["q"])[[1, 2, 3, 0]])
    return (
        float(np.linalg.norm(p - np.asarray(pose["p"]), axis=1).max()),
        float(np.degrees((Rc.inv() * R).magnitude()).max()),
    )


def within(err: tuple[float, float], frac: float = 1.0) -> bool:
    """The one threshold: audit = frac 1 · production (reuse) = REUSE_FRAC."""
    return err[0] <= POS_TOL_M * frac and err[1] <= ROT_TOL_DEG * frac


def record(cluster: dict, q: np.ndarray, pin: tuple = ()) -> dict:
    """One fixed_only cluster of part_clusters → tree record {root, body, body_obj, members: {rid: {obj, pose,
    series}}}.
    series (the relative pose sequence) stays in memory (not stored in the asset; merge drops it). pin = part rids fixed
    in
    their recorded pose (clip median) without the band check; when present the record has a pinned field (read by the
    asset id, shape check and oracle)."""
    body = cluster["anchor"]
    rb = OP.rid_of(str(body["obj_id"]))
    mem = {}
    for o in cluster["fixed"]:
        ser = rel_series(q, body, o)
        mem[OP.rid_of(str(o["obj_id"]))] = {"obj": str(o["obj_id"]), "pose": clip_pose(ser), "series": ser}
    return {
        "root": OP.root_of(rb),
        "body": rb,
        "body_obj": str(body["obj_id"]),
        "members": mem,
        **({"pinned": sorted(pin)} if pin else {}),
    }


def rigid(rec: dict) -> bool:
    """Whether it is inside the audit band at its own pose (clip median); otherwise no asset can take this tree."""
    return all(
        within(series_err(m["series"], m["pose"])) for r, m in rec["members"].items() if r not in rec.get("pinned", ())
    )


def out_of_band(rec: dict) -> list[str]:
    """Part rids outside the audit band at their own pose (clip median): candidates for a lever pin."""
    return sorted(r for r, m in rec["members"].items() if not within(series_err(m["series"], m["pose"])))


def pose_err(a: dict, b: dict) -> tuple[float, float]:
    """Difference (m, °) between two poses; read by the asset assignment of pinned parts."""
    Ra, Rb = (Rot.from_quat(np.asarray(x["q"])[[1, 2, 3, 0]]) for x in (a, b))
    return float(np.linalg.norm(np.subtract(a["p"], b["p"]))), float(np.degrees((Ra.inv() * Rb).magnitude()))


def _err(rec: dict, sp: dict) -> dict:
    """Record ↔ asset difference per part: ordinary parts by relative pose sequence, pinned parts by recorded pose."""
    pin = set(rec.get("pinned", ()))
    return {
        r: (
            pose_err(m["pose"], sp["members"][r]["pose"])
            if r in pin
            else series_err(m["series"], sp["members"][r]["pose"])
        )
        for r, m in rec["members"].items()
    }


def moved_word(rec: dict) -> str:
    """Exclusion reason word of a record that is not one rigid body ('' = one rigid body); the kind of the part that
    exceeds the audit band most decides it (oakink2.part_kinds.ATTACH_MOVED, words not restated here)."""
    if rigid(rec):
        return ""
    err = {r: series_err(m["series"], m["pose"]) for r, m in rec["members"].items()}
    worst = max(sorted(err), key=lambda r: max(err[r][0] / POS_TOL_M, err[r][1] / ROT_TOL_DEG))
    return PK.ATTACH_MOVED[PK.kind_of(worst)[0]]


#: Merged asset id form (the form asset_id builds; the builder checks itself): raw_<root rid lower-case · @→_>_f<sha 8>
ASSET_ID_RE = re.compile(r"raw_(o\d\d_\d{4}_\d{5})_f[0-9a-f]{8}")


def asset_id(rec: dict) -> str:
    key = json.dumps({r: m["pose"] for r, m in sorted(rec["members"].items())}, sort_keys=True)
    if rec.get("pinned"):  # assets with pinned parts get ids distinct from ordinary assets
        key += json.dumps(rec["pinned"])
    aid = f"raw_{rec['root'].lower().replace('@', '_')}_f{hashlib.sha256(key.encode()).hexdigest()[:8]}"
    if not ASSET_ID_RE.fullmatch(aid):
        raise ValueError(f"[FIXED_MERGE] merged asset id {aid!r} does not match ASSET_ID_RE — root {rec['root']!r}")
    return aid


def is_asset_id(obj_id: str) -> bool:
    """Whether a tree obj_id is a merged asset (asset_id form)."""
    return bool(ASSET_ID_RE.fullmatch(str(obj_id)))


def fixed_json(obj_id: str) -> Path:
    """FIXED.json location of a merged asset: <obj_id>/FIXED.json in the assets_objects layer (paths.layer_path)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    return Path(P.layer_path("assets_objects", "oakink2")) / str(obj_id) / "FIXED.json"


def body_rid(obj_id: str) -> str:
    """Rigid-body frame rid of a merged asset = body in FIXED.json (the merged asset's frame = the body frame; parts
    attach
    by their relative pose in it). A missing FIXED.json (fixed_json) or body fails."""
    fx = fixed_json(obj_id)
    if not is_asset_id(obj_id) or not fx.is_file():
        raise ValueError(f"[FIXED_MERGE] {obj_id!r}: not a merged asset, or no FIXED.json {fx}")
    body = json.loads(fx.read_text(encoding="utf-8")).get("body")
    if not body:
        raise ValueError(f"[FIXED_MERGE] {fx}: no body")
    return str(body)


def same_shape(sp: dict, rec: dict) -> bool:
    return (
        sp["root"] == rec["root"]
        and sp["body"] == rec["body"]
        and sorted(sp["members"]) == sorted(rec["members"])
        and sp.get("pinned", []) == rec.get("pinned", [])
    )


def fits(sp: dict, rec: dict) -> bool:
    """Assignment rule (spec stage, production): the tree record's relative pose sequence lies within the production
    band (audit × REUSE_FRAC) of the asset pose."""
    pin = set(rec.get("pinned", ()))  # pinned parts compare recorded poses (pose_err)
    return same_shape(sp, rec) and all(
        within(series_err(m["series"], sp["members"][r]["pose"]), REUSE_FRAC)
        if r not in pin
        else within(pose_err(m["pose"], sp["members"][r]["pose"]), REUSE_FRAC)
        for r, m in rec["members"].items()
    )


def merge(published: dict, recs: list[tuple[str, dict]]) -> tuple[dict, dict]:
    """Published rigid assets + tree records (sorted (clip, record)) → (asset table, {asset: changed}). Published assets
    stay;
    only records without a matching asset get a new asset (that record's pose, reference clip); records whose wobble is
    outside the audit band get none (rigid)."""
    out, changed = dict(published), {}
    for clip, rec in recs:
        if not rigid(rec) or any(fits(sp, rec) for sp in out.values()):
            continue
        aid = asset_id(rec)
        out[aid] = {
            **rec,
            "members": {r: {"obj": m["obj"], "pose": m["pose"]} for r, m in rec["members"].items()},
            "ref_clip": clip,
            "band": {"pos_m": POS_TOL_M, "rot_deg": ROT_TOL_DEG},
            "kinds": {r: PK.kind_of(r)[0] for r in rec["members"]},
            **({"pinned": rec["pinned"]} if rec.get("pinned") else {}),
        }
        changed[aid] = "new"
    return out, changed


def pick(assets: dict, cluster: dict, q: np.ndarray, pin: tuple = ()) -> tuple[str | None, dict]:
    """Converter oracle (audit): among assets whose series_err is inside the audit band, the one with the smallest error
    (asset id, {part: [m, °]})."""
    rec = record(cluster, q, pin)
    best, best_err = None, {}
    for aid, sp in sorted(assets.items()):
        if not same_shape(sp, rec):
            continue
        err = _err(rec, sp)
        if all(within(e) for e in err.values()):
            if best is None or max(e[0] for e in err.values()) < max(e[0] for e in best_err.values()):
                best, best_err = aid, err
    return best, best_err


def static_rigid(cluster: dict, q: np.ndarray, cj: dict) -> dict | None:
    """When the clip range of the canonical joint values (cj = result of the converter's canon_joint) of one-body
    cluster
    `cluster` is at or below the threshold (convert_version.static_artic): {kind, range, joint_asset}; above
    it None (stays articulated). At or below the threshold but outside the audit band (rigid), the cluster cannot be one
    rigid body: the same shape + skip (part exceeding the band most · kind · position · rotation). The skip text is
    stored in rules.D1g and stays as written."""
    from dexengine.data.artic import (
        convert_version as D1G,  # noqa: PLC0415 — owner of the rule and threshold (dependency-free module)
    )

    th = np.asarray(cj["theta"], float)
    ok, rng = D1G.static_artic(cj["art"]["type"], float(th.min()), float(th.max()))
    if not ok:
        return None
    st = {"kind": cj["art"]["type"], "range": rng, "joint_asset": cj["asset"]}
    rec = record(cluster, q)
    if rigid(rec):
        return st
    err = {r: series_err(m["series"], m["pose"]) for r, m in rec["members"].items()}
    w = max(sorted(err), key=lambda r: max(err[r][0] / POS_TOL_M, err[r][1] / ROT_TOL_DEG))
    return {
        **st,
        "skip": f"{w}({PK.kind_of(w)[0]}) {err[w][0] * 100:.2f}cm · {err[w][1]:.2f}° — outside the fixed-joint band",
    }


def pinned(sp: dict, o: dict) -> bool:
    """Whether asset `sp` carries part object `o` as a pinned part (outside the oracle; its displacement is
    recorded)."""
    return OP.rid_of(str(o["obj_id"])) in sp.get("pinned", ())


def lever_pin(cluster: dict, q: np.ndarray, st: dict | None, names: frozenset) -> list[str]:
    """Part rids outside the band (fixed in their recorded pose) when the static-articulation check skipped (static
    articulated object, cluster outside the rigid-joint band) and stage 1 listed this cluster's body name in the lever
    pin
    request (artic.convert_version.lever_for); otherwise []."""
    if not st or "skip" not in st or cluster["anchor"]["name"] not in names:
        return []
    return out_of_band(record(cluster, q))


def decision(assets: dict, cluster: dict, q: np.ndarray, who: dict, ds: str, pin: tuple = ()) -> dict:
    """One rigid-joint decision of the converter (fixed_only and static articulated objects take this path): the asset
    inside the audit band (pick) · excluded (moved_word) without a rigid pose · fails without an in-band asset (the spec
    stage did not see this tree)."""
    aid, err = pick(assets, cluster, q, pin)
    if aid is None and (w := moved_word(record(cluster, q, pin))):  # no rigid pose
        return {"status": f"excluded:{w}", "exclude": w, **who, "why": "the attached part wobbles outside the band"}
    if aid is None:
        raise ValueError(
            f"rigid joint {who['members']}: no spec asset inside the band — the spec stage"
            f" (python -m dexengine.data.artic.spec_stage --dataset {ds}) did not see this tree"
        )
    return {"fixed": cluster, "asset": aid, "err": err}


def _obj_vf(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    """OBJ vertices and faces (vertex indices only, 0-based). Polygons are fan-triangulated."""
    V, F = [], []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) for t in ln.split()[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            F += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.asarray(V, float), F


def merged_parts(sp: dict) -> list[tuple[str, str, dict | None]]:
    """Part order of the merged mesh: [(source part id, part asset folder name, pose | None)], body first, members
    sorted.
    Merging (build_mesh) and the render cover (dexengine.robots.textures.rules._oakink2_raw) read this one order."""
    return [(sp["body"], sp["body_obj"], None)] + [
        (r, sp["members"][r]["obj"], sp["members"][r]["pose"]) for r in sorted(sp["members"])
    ]


def raw_parts(asset_dir: str, align_root: str) -> list | None:
    """With FIXED.json in the asset folder, the render cover's source = the part list
    [(align_root/<rid>/model.obj | None, rid, pose | None)] (merged_parts order); otherwise None (not a merged
    asset)."""
    fx = Path(asset_dir) / "FIXED.json"
    if not fx.is_file():
        return None
    out = []
    for rid, _, pose in merged_parts(json.loads(fx.read_text(encoding="utf-8"))):
        q = Path(align_root) / rid / "model.obj"
        out.append((str(q) if q.is_file() else None, rid, pose))
    return out


def place(V: np.ndarray, pose: dict | None) -> np.ndarray:
    """Part vertices into the body frame; pose = {p, q(wxyz)} · None = the body itself (unchanged)."""
    if pose is None:
        return V
    return Rot.from_quat(np.asarray(pose["q"])[[1, 2, 3, 0]]).apply(V) + np.asarray(pose["p"])


def build_mesh(objects_root: Path, aid: str, sp: dict) -> Path:
    """Write the merged visual.obj + FIXED.json into the asset folder (objects_root/aid); existing ones are compared and
    a difference fails."""
    d = objects_root / aid
    srcs = {
        sp["body"]: objects_root / sp["body_obj"] / "visual.obj",
        **{r: objects_root / m["obj"] / "visual.obj" for r, m in sp["members"].items()},
    }
    miss = [str(p) for p in srcs.values() if not p.is_file()]
    if miss:
        raise FileNotFoundError(f"{TAG} {aid}: source meshes missing {miss}")
    Vs, Fs, off = [], [], 0
    for r, _, pose in merged_parts(sp):
        V, F = _obj_vf(srcs[r])
        Vs.append(place(V, pose))
        Fs.append(np.asarray(F, int) + off)
        off += len(V)
    V, F = np.vstack(Vs), np.vstack(Fs)
    d_ = OBJ_DIGITS
    text = (
        "# FIXED_MERGE "
        + aid
        + "\n"
        + "".join(f"v {x:.{d_}f} {y:.{d_}f} {z:.{d_}f}\n" for x, y, z in V)
        + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F)
    )
    prov = {
        "asset": aid,
        **{k: sp[k] for k in ("root", "body", "body_obj", "members", "ref_clip", "band", "pinned") if k in sp},
        "src_sha": {r: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for r, p in sorted(srcs.items())},
        "n_vert": int(len(V)),
        "n_face": int(len(F)),
    }
    d.mkdir(parents=True, exist_ok=True)
    vis, pj = d / "visual.obj", d / "FIXED.json"
    if vis.is_file() and vis.read_text(encoding="utf-8") != text:
        raise ValueError(
            f"{TAG} {aid}: the existing visual.obj differs from this statement's merged mesh (the asset id is "
            "the pose sha)"
        )
    if not vis.is_file():
        tmp = vis.with_suffix(".obj.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(vis)
    pj.write_text(json.dumps(prov, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return vis


def lever_decision(assets: dict, cluster: dict, q: np.ndarray, who: dict, ds: str, st: dict, pin: list) -> dict:
    """Lever-pin conversion decision: rigid joint (decision) + out-of-band parts `pin` fixed in their recorded pose ·
    field
    static_artic.lever_rigid · the static check's reason (skip) moves to band_out (the object became rigid; skip means
    'stays articulated')."""
    sa = {k: v for k, v in st.items() if k != "skip"}
    return {
        **decision(assets, cluster, q, who, ds, tuple(pin)),
        "static_artic": {**sa, "lever_rigid": list(pin), "band_out": st["skip"]},
    }


def pin_shift(pin: dict) -> dict:
    """Displacement (name → [m, °]) of pinned parts left out of the converter FK oracle; tree record
    rules.D1g.pinned."""
    return {"pinned": {n: [round(e[0], 4), round(e[1], 2)] for n, e in pin.items()}} if pin else {}
