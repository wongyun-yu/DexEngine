"""Artic spec collection: per-tree articulations, the merged spec, physical-range entries and the spec
update (driven by spec_stage.ensure)."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from dexengine.core import capacity
from dexengine.core import paths as P
from dexengine.data.artic import limits as AL
from dexengine.data.artic import (
    range_src as RS,  # owner of the source records (hand-independent, measured once and fixed)
)
from dexengine.data.artic import (
    ranges as AR,  # owner of the asset spec rule (asset_spec · statement · window)
)
from dexengine.data.oakink2 import (
    fixed_merge as FM,  # owner of rigid-joint assets (merged mesh, pose assignment)
)


def _tree_arts(
    ds: str, f: Path, ti: dict, published: dict, rest_only: bool = False
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]], int, list[dict]]:
    """One tree → (the joints the tree states: arctic input, or the consumer copy in merged OakInk2 trees; source
    records:
    statements of unmerged OakInk2 trees (converter per_tree_record · per_tree_rest for frozen trees, posed with
    `published`); fit count; rigid-joint records (per_tree_fixed, per instance)). rest_only = frozen statements and
    rigid joints only."""
    out = [(str(o["obj_id"]), o["articulation"]) for o in ti.get("objects") or [] if o.get("articulation")]
    if ds != "oakink2" or "D1g" in (ti.get("rules") or {}):
        return out, [], 0, []
    from dexengine.data.oakink2.artic_fit import part_instances  # owner of the pairing rule

    if not any(len(v) >= 2 for v in part_instances(ti.get("objects") or []).values()):
        return out, [], 0, []
    from dexengine.data.oakink2 import (
        artic_convert as CV,  # owner of the fit rule (it reads this module, hence the late import)
    )

    with np.load(f.parent / "0/trajectory_kinematic.npz") as z:
        rest = CV.per_tree_rest(
            ti, z["qpos"], published
        )  # tree with the articulated part frozen: statement posed with the spec joint
        fx = CV.per_tree_fixed(ti, z["qpos"], published)
        if rest_only:
            return out, rest, 0, fx
        rec = CV.per_tree_record(ti, z["qpos"])
    return out, ([rec] if rec else []) + rest, int(rec is not None), fx


def collect_all(published: dict | None = None, rest_only: bool = False) -> tuple[dict, dict, list, dict]:
    """Every base-layer tree of every registered hand (a broken task_info fails · fit failures are counted) → (stats,
    per =
    arctic tree joints {(ds, asset): (object id, [(name, statement)])}, fixed, src = OakInk2 source records {(ds,
    asset):
    (object id, {name: statement})}). Name = <emb>/<tree> (no hand; differing between hands fails, RS.one_record);
    independent of hand order and of which hands have the tree."""
    from dexengine.core import artifact_paths as tp
    from dexengine.data import rules as RR

    per: dict = {}
    src: dict = {}
    stats: dict = {}
    fixed: dict = {}
    for hand in [P.hand_spec_dir(h) for h in P.HAND_IDS]:
        n = fits = 0
        errs: list = []
        for ds in RR.ALL:
            for emb in P.data_layout.EMBODIMENTS:
                root = Path(P.layer_path("base", ds, hand=hand, emb=emb))
                if not root.is_dir():
                    continue
                for f in sorted(root.glob("*/task_info.json")):
                    n += 1
                    try:
                        out, recs, k, fx = _tree_arts(ds, f, json.loads(f.read_text()), published or {}, rest_only)
                    except Exception as e:  # noqa: BLE001 — a tree whose fit fails is counted and skipped (2b reports it, rc≠0)
                        errs.append(f"{hand}:{f.parent.name}: {type(e).__name__}: {str(e)[:120]}")
                        continue
                    fits += k
                    name = f"{emb}/{f.parent.name}"
                    for rec in fx:
                        fixed.setdefault((name, rec["root"]), rec)
                    for oid, art in out if ds != "oakink2" else []:
                        RS.one_record(per, (ds, tp.object_asset_id(oid, art)), oid, name, art, hand, "tree joint")
                    for oid, art in recs:
                        RS.one_record(src, (ds, tp.object_asset_id(oid, art)), oid, name, art, hand, "source")
        stats[hand] = [n, fits, errs]
    per = {k: (oid, sorted(v.items())) for k, (oid, v) in sorted(per.items())}
    return stats, per, [(name, rec) for (name, _root), rec in sorted(fixed.items())], dict(sorted(src.items()))


def kind_conflict(key: str, sp: dict) -> str:
    """Reason an OakInk2 asset disagrees with the per-object joint spec (oakink2.part_kinds), else '': the moving part's
    spec kind is not a joint (hinge · slide), the kind differs, or its parent is not the base."""
    if not key.startswith("oakink2/"):
        return ""
    from dexengine.data.oakink2 import part_kinds as PK  # owner of the per-object joint kinds
    from dexengine.data.oakink2 import parts as OP

    parts = sp.get("parts") or {}
    mov, base = OP.rid_of(str(parts.get("moving_id"))), OP.rid_of(str(parts.get("base_id")))
    kind, parent = PK.kind_of(mov)
    want = {"revolute": "hinge", "prismatic": "slide"}.get(kind)
    if want != sp.get("type") or parent != base:
        return f"spec {OP.name_of(mov)} = {kind}(parent {parent}) · asset {sp.get('type')}(base {base})"
    return ""


PHYS_ONLY = (
    "phys"  # change reason: only the physical range was added and the window (bake limits) is unchanged, so no rebake
)


STEP = "dexengine.data.artic.spec_collect"  # capacity.NEEDS key: the physical-range pool (assets are independent)


def phys_entry(k: str, e: dict, meshes=None) -> tuple[dict, str]:
    """Physical range of an OakInk2 joint asset (dexengine.data.artic.limits): measured from the link meshes when
    missing or of another version, and the window = data window ∩ effective limits. Returns (entry, change reason | "").
    Other datasets get joint
    values from their source (outside the converter's clamping) and stay as they are.
    meshes = (entry, link base|moving) → link mesh (tests swap it; default = object_surface.link_mesh of the asset
    folder)."""
    if not k.startswith("oakink2/"):
        return e, ""
    ph = e.get("phys")
    if not ph or ph.get("version") != AL.VERSION:
        if meshes is None:
            from dexengine.data.object_surface import (
                link_mesh,  # owner of the link meshes (parts ∪ fixed attachments)
            )

            root = Path(P.layer_path("assets_objects", "oakink2"))

            def meshes(x, link):
                return link_mesh(root / x["obj_id"], x["parts"], link, process=False)

        ph = AL.measure(e, meshes(e, "base"), meshes(e, "moving"))
        print(f"{AL.MARK} {k}: {AL.verdict(ph)} · limits {[ph['lo'], ph['hi']]} · scan {ph['scan']}", flush=True)
    n = AL.narrow({x: v for x, v in e.items() if x != "phys"}, ph)
    if n == e:
        return e, ""
    if abs(n["lo"] - e["lo"]) <= AR.WRAP_TOL and abs(n["hi"] - e["hi"]) <= AR.WRAP_TOL:
        return n, PHYS_ONLY
    return n, f"narrowed ({e['lo']:.4f}, {e['hi']:.4f}) → ({n['lo']:.4f}, {n['hi']:.4f})"


def _phys_job(kv: tuple[str, dict]) -> tuple[dict, str]:
    return phys_entry(*kv)


def merge(published: dict, per: dict) -> tuple[dict, dict]:
    """Published spec + collection → (spec asset table, {asset: change}). New asset = AR.asset_spec (tree reproducing
    the
    most) · existing asset = statement fixed, window = published window ∪ fitting trees (asset_spec prev). Assets whose
    window did not widen keep **the published entry as is** (bytes unchanged) · published assets missing from the
    collection stay too (windows never shrink) · only published assets disagreeing with the per-object joint spec are
    removed (kind_conflict)."""
    objs, changed = dict(published), {}
    for k in sorted(
        objs
    ):  # published assets disagreeing with the per-object joint spec are removed (kind or pairing differs)
        why = kind_conflict(k, objs[k])
        if why:
            del objs[k]
            changed[k] = f"dropped ({why})"
    for (ds, asset), (oid, recs) in sorted(per.items()):
        k = f"{ds}/{asset}"
        if kind_conflict(
            k, recs[0][1]
        ):  # joints of trees merged under an older version (bottle-cap hinges etc.) are not new assets
            continue
        prev = objs.get(k)
        if prev is None:
            objs[k] = AR.asset_spec(ds, oid, asset, recs)
            changed[k] = "new"
            continue
        e = AR.asset_spec(ds, oid, asset, recs, prev)
        if e["lo"] < prev["lo"] - AR.WRAP_TOL or e["hi"] > prev["hi"] + AR.WRAP_TOL:
            objs[k] = e
            changed[k] = f"extended ({prev['lo']:.4f}, {prev['hi']:.4f}) → ({e['lo']:.4f}, {e['hi']:.4f})"
    need = [
        k for k in sorted(objs) if k.startswith("oakink2/") and (objs[k].get("phys") or {}).get("version") != AL.VERSION
    ]
    got = {}
    if len(need) > 1:  # several assets to measure (a first-time spec): split across workers
        with (
            capacity.sized(STEP, len(need)) as cap,
            multiprocessing.get_context("fork").Pool(cap.workers, capacity.worker_init, cap.initargs) as pool,
        ):
            got = dict(zip(need, pool.map(_phys_job, [(k, objs[k]) for k in need], chunksize=1), strict=True))
        print(f"{AL.MARK} assets with measured physical range {len(got)} · workers {cap.workers}", flush=True)
    for k in sorted(
        objs
    ):  # physical range: measure (again for another version) and narrow the window; the bake limits are this window
        objs[k], why = got.get(k) or phys_entry(k, objs[k])
        if why and (why != PHYS_ONLY or k not in changed):
            changed[k] = why if k not in changed else f"{changed[k]} · {why}"
    return objs, changed


def update(write: bool = True, out: Path | None = None, remeasure: bool = False) -> dict:
    """Spec update: OakInk2 = dexengine.data.artic.range_src.oak_entries (verify | measure) · arctic = merge. Nothing is
    **written** when
    nothing changed (generation time and sha stay put). Returns {changed, stats, wrote, out, sha, objects, fixed,
    mode}."""
    from dexengine.data import rules as RR

    out = Path(out) if out is not None else RR.axis_output(RR.ARTIC_SPEC, must_exist=False)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    r = RS.oak_entries(doc, remeasure, collect_all, kind_conflict)
    stats, changed = r["stats"], r["changed"]
    base = {**{k: v for k, v in (doc.get("objects") or {}).items() if not k.startswith("oakink2/")}, **r["oak"]}
    objs, mchanged = merge(base, r["per"])
    for k, why in mchanged.items():  # a physical-range-only change (PHYS_ONLY) does not overwrite a remeasure reason
        if k not in changed:
            changed[k] = why
        elif why != PHYS_ONLY:
            changed[k] = f"{changed[k]} · {why}"
    fixed, fchanged = FM.merge(doc.get(FM.SECTION) or {}, r["fixed"])
    changed.update({f"oakink2/{FM.SECTION}/{k}": v for k, v in fchanged.items()})
    src_doc = RS.src_doc(r["src"])
    wrote = False
    if (changed or src_doc != (doc.get(RS.SRC) or {})) and write:
        res = {
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "tool": "dexengine/data/artic/spec_stage.py",
            "hand": RR.axis_hand(),
            "rule": "D1-j (a) — USD joint limit = clip max opening (canonical coordinates) · OakInk2 = measured once "
            "on the "
            "hand-independent raw record (src) and fixed · canonical joint = most reproduced tree (D1-g) · "
            "arctic = tree joint (not narrowed)",
            "tol": {"pos_m": AR.POS_TOL_M, "rot_deg": AR.ROT_TOL_DEG},
            "tol_slide": dict(zip(("pos_m", "rot_deg"), AR.band("slide"), strict=True)),
            "n_src_trees": len({n for _o, recs in r["src"].values() for n in recs}),
            "n_objects": len(objs),
            "n_restate": sum(v["n_restate"] for v in objs.values()),
            "objects": objs,
            FM.SECTION: fixed,
            RS.SRC: src_doc,
        }
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(out)
        wrote = True
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12] if out.is_file() else "-"
    return {
        "changed": changed,
        "stats": stats,
        "wrote": wrote,
        "out": out,
        "sha": sha,
        "objects": objs,
        "fixed": fixed,
        "mode": r["mode"],
    }
