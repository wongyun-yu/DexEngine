#!/usr/bin/env python3
"""Bake the arm-collider revision of the robot USDs — a sim-only derivative of each body-generation USD.

What changes (owner: ``dexengine.core.arms.ARM_SIM_COLLIDER_REV``): on each side, the arm links listed in
``replace`` lose their vendor capsule and get the convex hull of their URDF ``<visual>`` mesh; the links in ``add``
(the flange arm_6) get such a hull next to their existing hand colliders; on the static root body ``world`` the vendor
torso capsules (``torso_replace``) leave and the torso hulls are added — one per part of every torso link
(``torso_replace`` + ``torso_add``) as the decomposition rule cuts them (sim_collider.torso_bodies), after its gate
(torso_split.torso_split_check) passes on exactly those bodies. The hull vertices are those of the one generator of the
arm colliders (``dexengine.placement.collision.sim_collider`` — the shapes the judge measures with), and each removed
capsule is checked against the URDF capsule it came from. Nothing else changes — kinematics, masses, visuals,
instancing and every other collider are byte-identical to the source (checked below, prim by prim).

Where things go: source ``<combo><body variant>/<combo><body variant>.usd`` is opened read-only and copied to
``<combo><body variant><suffix>/`` (a new directory — the source and the runs spawning it are never touched),
together with the body's bake meta (``config.yaml``, it still names the URDF the body was baked from) and a
provenance record (``dexengine/robots/derived_usd.py``) that the spawn check reads. Then the split (two-articulation)
copy is made from it with ``split_arm.py``.

Every hand gets its two bodies baked (``arms.hand_bodies``) — the body its sim spawns (the canon body, or the
canon body with that hand's joint-limit variant, ``HandSpec.sim_finger_limits`` → ``arms.HAND_LIMIT_SUFFIX``) and
the render body (its repaint by ``render_body.py``, ``arms.ARM_RENDER_GEN``); a missing one dies with the
command that makes it. A render plate must equal its training plate except for materials (``render_plate.check`` —
after the bake and in ``--verify-only``, the merge gate).

Usage (the retarget venv has pxr/trimesh/scipy; the data root is where the USD cache lives)::

    DEXENGINE_MAIN_ROOT=<main checkout> .venv/bin/python dexengine/robots/usd/collider_rev.py --arm rby1 [--hand ...]
    ... --verify-only        # re-check what is on disk, write nothing
    ... --dry-run            # list the (hand, body variant) jobs and their source/destination, write nothing
    ... --out-root <dir>     # write the derivatives under <dir> instead (a test bake; sources stay where they are)

Completion = every (hand, body variant) pair produced and verified (count printed in the last line).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics

from dexengine.core import arms as AS
from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.core.repo import REPO_ROOT as REPO
from dexengine.placement.collision import sim_collider as SC
from dexengine.placement.collision import torso_split as TSP
from dexengine.robots import derived_usd as DU
from dexengine.robots.usd import render_plate as RP
from dexengine.robots.usd.collider_edit import edit_link, edit_world


def usd_root() -> Path:
    return P.main_root() / "data/usd_cache/robots/full_robot_gen"


def _links(arm: str, rev) -> list[tuple[str, str]]:
    spec = AS.ARM_REGISTRY[arm]
    return [
        (spec.body_name(side, k), mode)
        for side in spec.sides
        for mode, ks in (("replace", rev.replace), ("add", rev.add))
        for k in ks
    ]


def _prims(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    return {str(p.GetPath()): p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())}


def _hulls(arm: str, urdf: Path) -> list:
    """The revision hulls of both sides — sim_collider.bodies, the one generator of the arm colliders."""
    return [b for side in AS.ARM_REGISTRY[arm].sides for b in SC.bodies(arm, side, urdf) if isinstance(b, SC.Hull)]


def verify(arm: str, src: Path, dst: Path, rev, urdf: Path) -> dict:
    """Composed-stage comparison: the only differences are the colliders of the revised links, and each revised link's
    hull points are sim_collider's vertices (float32 in the USD → 1e-6 m)."""
    a, b = Usd.Stage.Open(str(src)), Usd.Stage.Open(str(dst))
    root = str(b.GetDefaultPrim().GetPath())
    pa, pb = _prims(a), _prims(b)
    # replace: anything under that link's collisions may differ · add: only the new prim (the hand colliders stay)
    allowed = tuple(
        f"{root}/{ln}/collisions/" + ("" if mode == "replace" else f"{rev.prim_name}") for ln, mode in _links(arm, rev)
    ) + ((f"{root}/world/collisions/",) if rev.torso_replace else ())
    removed = sorted(set(pa) - set(pb))
    added = sorted(set(pb) - set(pa))
    stray = [p for p in removed + added if not p.startswith(allowed)]
    changed = [
        p
        for p in set(pa) & set(pb)
        if not p.startswith(allowed)
        and (
            pa[p].GetTypeName() != pb[p].GetTypeName()
            or RP.attrs(pa[p]) != RP.attrs(pb[p])
            or list(pa[p].GetAppliedSchemas()) != list(pb[p].GetAppliedSchemas())
        )
    ]
    per_link, want = {}, {h.link: h.verts for h in _hulls(arm, urdf)}
    for ln, mode in _links(arm, rev):
        col = b.GetPrimAtPath(f"{root}/{ln}/collisions")
        shapes = [d for d in Usd.PrimRange(col, Usd.TraverseInstanceProxies()) if d.HasAPI(UsdPhysics.CollisionAPI)]
        caps = [d for d in Usd.PrimRange(col, Usd.TraverseInstanceProxies()) if d.IsA(UsdGeom.Capsule)]
        hull = b.GetPrimAtPath(f"{root}/{ln}/collisions/{rev.prim_name}/node_hull/mesh")
        n = len(hull.GetAttribute("points").Get()) if hull.IsValid() else 0
        dv = (
            float(np.abs(np.asarray(hull.GetAttribute("points").Get(), float) - want[ln]).max())
            if hull.IsValid() and n == len(want[ln])
            else float("inf")
        )
        ok = (
            hull.IsValid()
            and not caps
            and 4 <= n <= rev.hull_vertex_cap
            and (mode == "add" or len(shapes) == 1)
            and dv <= 1e-6
        )
        per_link[ln] = {"colliders": len(shapes), "capsules": len(caps), "hull_points": n, "max_dv_m": dv, "ok": ok}
    col = f"{root}/world/collisions"
    torso = SC.torso_bodies(arm, urdf)
    shapes = [
        d
        for d in Usd.PrimRange(b.GetPrimAtPath(col), Usd.TraverseInstanceProxies())
        if d.HasAPI(UsdPhysics.CollisionAPI)
    ]
    n_caps = sum(1 for d in shapes if d.IsA(UsdGeom.Capsule))
    for t in torso:  # every part, and nothing else on world (colliders = parts)
        mesh = b.GetPrimAtPath(f"{col}/{rev.torso_prim(t.link, t.part)}/node_hull/mesh")
        pts = np.asarray(mesh.GetAttribute("points").Get(), float) if mesh.IsValid() else np.zeros((0, 3))
        dv = float(np.abs(pts - t.verts).max()) if len(pts) == len(t.verts) else float("inf")
        per_link[f"world/{t.name}"] = {
            "hull_points": len(pts),
            "max_dv_m": dv,
            "world_capsules": n_caps,
            "world_colliders": len(shapes),
            "ok": n_caps == 0 and len(shapes) == len(torso) and 4 <= len(pts) <= rev.hull_vertex_cap and dv <= 1e-6,
        }
    res = {
        "removed": len(removed),
        "added": len(added),
        "stray": stray[:10],
        "changed": sorted(changed)[:10],
        "n_changed": len(changed),
        "links": per_link,
        "ok": not stray and not changed and all(v["ok"] for v in per_link.values()),
    }
    return res


def _src(arm: str, hand: str, variant: str, root: Path | None = None) -> Path:
    """The plate ``<root>/<combo><variant>/<combo><variant>.usd`` — a body under the data root by default."""
    c = P.combo_id(arm, hand, variant)
    return (root or usd_root()) / c / f"{c}.usd"


def bake(arm: str, hand: str, body_variant: str, rev, force: bool, out_root: Path) -> dict:
    combo = P.combo_id(arm, hand)
    src, urdf = _src(arm, hand, body_variant), SC.combo_urdf(arm, hand, body_variant)
    dv = body_variant + rev.suffix
    out = out_root / f"{combo}{dv}"
    dst = out / f"{combo}{dv}.usd"
    if not src.is_file() or not urdf.is_file():
        raise InputMissingError(
            f"[wrist_hull] source missing: usd {src.is_file()} {src} · urdf {urdf.is_file()} {urdf}"
        )
    if out.exists() and not force:
        raise SystemExit(
            f"[wrist_hull] already exists: {out} (re-bake only with --force · the source is never touched)"
        )
    try:  # the torso decomposition rule (D-1) gates every torso bake
        TSP.torso_split_check(arm, urdf, SC.torso_bodies(arm, urdf))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tdst = tmp / dst.name
    try:
        return _bake_into(arm, src, urdf, tmp, tdst, out, combo, body_variant, dv, rev)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # only this run's own temp dir (success renamed it away)


def _bake_into(arm, src, urdf, tmp, tdst, out, combo, body_variant, dv, rev) -> dict:
    shutil.copy2(src, tdst)
    stage = Usd.Stage.Open(str(tdst))
    root = str(stage.GetDefaultPrim().GetPath())
    uroot = ET.parse(str(urdf)).getroot()
    links = {h.link: edit_link(stage, root, h, rev, SC.vendor_capsules(uroot, h.link)) for h in _hulls(arm, urdf)}
    if rev.torso_replace or rev.torso_add:
        links["world"] = edit_world(stage, root, rev, uroot, SC.torso_bodies(arm, urdf))
    stage.GetRootLayer().Save()
    del stage
    mat = AS.generation_of(arm, body_variant).material_of
    # a repaint's body · bake meta = its source's
    body = src.parent if mat is None else src.parent.parent / f"{combo}{mat}"
    if (body / "config.yaml").is_file():  # bake meta: still names the URDF the body was baked from
        shutil.copy2(body / "config.yaml", tmp / "config.yaml")
    rev_rec = {
        k: getattr(rev, k)
        for k in ("suffix", "replace", "add", "hull_vertex_cap", "prim_name", "torso_replace", "torso_add")
    }
    prov = {
        "variant": dv,
        "source_variant": body_variant,
        "kind": "sim_collider_rev",
        "rev": rev_rec,
        "source_usd": str(src),
        "source_md5": DU.file_md5(src),
        "urdf": str(urdf),
        "links": links,
        DU.DIGEST_KEY: SC.digest(arm, [urdf]),  # the spawn check (DU.verify) re-measures it with sim_collider
        "generator": "dexengine/robots/usd/collider_rev.py",
        "generator_commit": _git_head(),
        DU.GENERATOR_KEY: DU.generator_digest(REPO),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if mat is not None:  # render plate: the training body its repaint came from
        prov[DU.MATERIAL_SOURCE_KEY] = DU.file_md5(body / f"{body.name}.usd")
    (tmp / DU.PROVENANCE_NAME).write_text(
        json.dumps(prov, ensure_ascii=False, indent=1, default=float), encoding="utf-8"
    )
    v = verify(arm, src, tdst, rev, urdf)
    if not v["ok"]:
        raise SystemExit(
            f"[wrist_hull] check failed {combo}{dv}: {json.dumps(v, ensure_ascii=False, default=str)[:900]}"
        )
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return {
        "combo": f"{combo}{dv}",
        "verify": v,
        "links": {k: {kk: vv for kk, vv in r.items() if kk != "meshes"} for k, r in links.items()},
    }


def _git_head() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "?"


def split(dst: Path, force: bool) -> None:
    cmd = [sys.executable, str(REPO / "dexengine/robots/usd/split_arm.py"), "--src", str(dst)]
    cmd += ["--force"] if force else []
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[wrist_hull] split failed {dst.name}: {(r.stdout + r.stderr)[-800:]}")


def _missing_body(arm: str, hand: str, bv: str) -> str:
    """A registered body USD that is not on disk, with the command that makes it."""
    g = AS.generation_of(arm, bv)
    if g.material_of is not None:
        how = (
            f"python dexengine/robots/usd/render_body.py --src-dir {_src(arm, hand, g.material_of).parent} "
            f"--out-dir {_src(arm, hand, bv).parent}"
        )
    elif g.limits_of is not None:
        how = (
            f"python3 dexengine/robots/urdf/truncate.py --hand {hand} (joint-limit copy URDF) → "
            f"python scripts/build_robot_usd.py --combos {P.combo_id(arm, hand, bv)}"
        )
    else:
        how = f"python scripts/build_robot_usd.py --arm {arm} --hand {hand} (canonical body)"
    return f"{_src(arm, hand, bv)} — make it: {how}"


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--arm", required=True, help="arm id (no default) — its collider revision arms.ARM_SIM_COLLIDER_REV"
    )
    ap.add_argument("--hand", action="append", help="hand (repeatable). Default = every registered combo of this arm")
    ap.add_argument("--force", action="store_true", help="re-bake existing derivatives (the source always stays)")
    ap.add_argument("--verify-only", action="store_true", help="write nothing, only check the derivatives on disk")
    ap.add_argument(
        "--dry-run", action="store_true", help="write nothing, only print the job list (source → derivative)"
    )
    ap.add_argument("--no-split", action="store_true", help="do not make the split copies")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="root to write derivatives under (default = the robot USD place of the data root) — for test bakes. "
        "Sources always come from the data root",
    )
    return ap.parse_args()


def main() -> int:
    a = _args()
    arm = a.arm
    if arm not in AS.ARM_SIM_COLLIDER_REV:
        raise SystemExit(f"[wrist_hull] arm {arm!r} has no collider revision (arms.ARM_SIM_COLLIDER_REV)")
    rev = AS.ARM_SIM_COLLIDER_REV[arm]
    out_root = a.out_root or usd_root()
    # each hand's two bodies (sim body · its repaint — the repaint after the plate it is checked against)
    jobs, done = [(h, bv) for h, bv, _dv in DU.baked_bodies(arm, a.hand)], []
    missing = [_missing_body(arm, h, bv) for h, bv in jobs if not _src(arm, h, bv).is_file()]
    if missing:  # every registered body (canon · render) is required for every hand — no skipping
        raise InputMissingError("[wrist_hull] body USDs missing:\n  " + "\n  ".join(missing))
    for h, bv in jobs:
        combo, dst = P.combo_id(arm, h) + bv + rev.suffix, _src(arm, h, bv + rev.suffix, out_root)
        if a.dry_run:
            print(f"[WRIST_HULL_DRY] {_src(arm, h, bv)} → {dst}{'' if a.no_split else ' (+ _split)'}", flush=True)
            done.append(combo)
            continue
        mat = AS.generation_of(arm, bv).material_of
        if a.verify_only:
            v = verify(arm, _src(arm, h, bv), dst, rev, SC.combo_urdf(arm, h, bv))
            DU.verify(
                str(dst),
                bv + rev.suffix,
                bv,
                str(_src(arm, h, bv)),
                arm=arm,
                hand=h,
                material_source_usd=None if mat is None else str(_src(arm, h, mat)),
            )
            print(
                f"[WRIST_HULL_VERIFY] {combo} ok={v['ok']} links {json.dumps(v['links'], ensure_ascii=False)}",
                flush=True,
            )
            ok = v["ok"] and (mat is None or RP.check(_src(arm, h, mat + rev.suffix, out_root), dst))
            done += [combo] if ok else []
            continue
        r = bake(arm, h, bv, rev, a.force, out_root)
        if not a.no_split:
            split(dst, a.force)
        if mat is not None and not RP.check(_src(arm, h, mat + rev.suffix, out_root), dst, not a.no_split):
            shutil.rmtree(dst.parent)  # a render plate ≠ its training plate never stays
            shutil.rmtree(dst.parent.with_name(dst.parent.name + "_split"), ignore_errors=True)
            raise SystemExit(
                f"[wrist_hull] render plate {combo} differs from the training plate outside materials — "
                f"removed (see {RP.MARK} above)"
            )
        tor = r["links"].get("world", {}).get("links", {})
        rows = {
            **{k: x for k, x in r["links"].items() if k != "world"},
            **{f"world/{k}": dict(x, mode="torso") for k, x in tor.items()},
        }
        print(
            f"[WRIST_HULL] {r['combo']} "
            + " · ".join(
                f"{k}: {x['mode']} {x['vertices']}v max inward error {x['max_inward_mm']}mm "
                f"volume ratio {x['volume_ratio']}"
                for k, x in rows.items()
            ),
            flush=True,
        )
        done.append(combo)
    print(
        f"WRIST_HULL_{'DRY' if a.dry_run else 'DONE'} {len(done)}/{len(jobs)} rev={rev.suffix} ({' '.join(done)})",
        flush=True,
    )
    return 0 if len(done) == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
