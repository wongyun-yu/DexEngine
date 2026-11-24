#!/usr/bin/env python
"""Merged surface points and keypoints for articulated objects: base part ∪ FK(moving part).

Background: the env observation (ObjectStateV2 35-dim, fingertip → nearest surface point) and the keypoint reward read a
single `<meshes>/<asset>/surface_points.npz` · `keypoints_fps2.npy` (per asset; asset id =
dexengine.core.artifact_paths.object_asset_id) and
**rigidly transform it with the root (base_link) pose**. Building it from the base part alone would drop the moving part
(lid) from observations and rewards entirely, so both parts are merged into one point cloud here.

Merge pose (--theta): there is one file per object but the joint angle varies per frame, so no static pose fits every
frame.
  closed (default, θ=0) — the time-weighted mode for 8 of 11 arctic objects (θ<0.15·hi in 57–76 % of frames), the USD
                          native frame (localPos0 = localPos1), and the repo convention (f0.repair:
                              "surface_points.npz is for the closed pose").
  hi                    — the object's max opening over its clips (dexengine.data.artic.ranges `hi`, D1-j (a)); the
  wrong pose for most
      frames.
  <float>               — any angle (rad), for experiments.
Runtime FK is the real fix, so the metadata it needs (part_id · joint_axis · joint_anchor · joint_type · merge_theta) is
stored too: the observation (ObjectStateV2) reads it through `_shared/mdp/artic_parts.SurfaceParts` and rotates the
moving
points to the current θ (fails without part_id). joint_axis and joint_anchor are in the **base frame** (the spec's
child-frame axis and anchor with child_pose applied; equal when there is no child_pose): the point cloud is in the base
frame, so the rotation turns about the base-frame axis too.

Location: asset folder `<meshes>/<asset>/`. Paired articulations (OakInk2) use a folder different from the object id, so
they never mix with the point cloud of trees using the same id as a rigid body (path owner
dexengine.core.artifact_paths).

Owners: parts · axis · anchor · limits = artic_ranges.json in the axis hand's audit folder (generator
dexengine/data/artic/ranges.py) · FK convention = dexengine.data.artic.fk.artic_moving_vertices · surface sampling =
object_surface.sample_surface_with_normals · keypoint rule = object_surface.fps_keypoints · paths = data_layout
(assets_objects → meshes).
Ledger: artic_surface_ledger.json in the hand-independent layer (exp_shared) (count check: spec count ≠ generated count
→ rc 1).

Usage:
    $ISAAC_PY dexengine/data/artic/surface.py --dataset arctic [--theta closed] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from dexengine.core import paths as P  # noqa: E402
from dexengine.data import rules as RR  # noqa: E402  owner of output locations (audit_path)
from dexengine.data.artic.fk import artic_moving_vertices
from dexengine.data.artic.ranges import (
    base_frame_joint,  # noqa: E402 — base-frame image of the spec axis and anchor (owner = the spec)
)
from dexengine.data.object_surface import KP_K, N_SURF, fps_keypoints, link_mesh, sample_surface_with_normals

dl = P.data_layout
ROOT = P.main_root()
# root of the run layer (spec · ledger): this code tree for the canonical checkout, the data root for a generation root
# (owner paths.run_root)
RUN_ROOT = P.run_root()
# Spec = per-hand audit output from the hands' base trees; this tool has no running hand, so it reads the **axis
# hand's** (owner of shared outputs).
# Ledger = generation record of object assets (hand-independent) → hand-independent layer exp_shared.
SPEC_FILE = RR.axis_output(RR.ARTIC_SPEC, must_exist=False)
LEDGER_FILE = P.run_path("exp_shared") / "artic_surface_ledger.json"


def stated(p) -> str:
    """Run-layer path as written in records and logs: relative to the run root (RUN_ROOT). In the canonical checkout
    this
    equals the repo-relative path, and a generation root gives the same string (the root never leaks into records). A
    path outside the run root raises ValueError (never falls back to an absolute path)."""
    return str(Path(p).relative_to(RUN_ROOT))


def _load_specs(ds: str) -> dict[str, dict]:
    if not SPEC_FILE.is_file():
        raise FileNotFoundError(f"{SPEC_FILE} missing — run dexengine/data/artic/ranges.py first")
    with open(SPEC_FILE, encoding="utf-8") as f:
        d = json.load(f)
    out = {k.split("/", 1)[1]: v for k, v in d["objects"].items() if v.get("dataset") == ds}
    if not out:
        raise ValueError(f"artic_ranges.json has no entries for dataset={ds!r}")
    return out


#: Default merge-pose policy, read by the CLI default and the staleness check (surface_problem).
THETA_DEFAULT = "closed"
#: Tolerance of the staleness check, comparing the float32-stored axis, anchor and merge angle with values re-derived
#: from the spec.
STALE_TOL = 1e-5


def _assets_root(ds: str) -> Path:
    return ROOT / dl.render("assets_objects", ds)


def _theta_of(policy: str, spec: dict) -> float:
    if policy == "closed":
        return 0.0
    if policy == "hi":
        return float(spec["hi"])
    return float(policy)


def build_one(obj_id: str, spec: dict, mesh_dir: Path, theta: float, seed: int = 0) -> dict:
    """Surface points of base ∪ FK(moving, θ) (area-proportional split, N_SURF in total) · normals · part ids ·
    keypoints. Coordinates in the base part (root) frame."""
    parts = spec["parts"]
    mb = link_mesh(mesh_dir, parts, "base", process=False)  # link = parts ∪ fixed attachments (artic_link_files)
    mm = link_mesh(mesh_dir, parts, "moving", process=False)
    a_b, a_m = float(mb.area), float(mm.area)
    n_m = int(round(N_SURF * a_m / (a_b + a_m)))
    n_m = min(max(n_m, 1), N_SURF - 1)
    n_b = N_SURF - n_m
    pts_b, nrm_b = sample_surface_with_normals(mb, n_b, seed)
    pts_m0, nrm_m0 = sample_surface_with_normals(mm, n_m, seed + 1)
    # FK is not reimplemented: points as is, normals from the image difference of (p+n) and p (exact for a rigid
    # transform).
    pts_m = artic_moving_vertices(spec, pts_m0.astype(np.float64), theta)
    nrm_m = artic_moving_vertices(spec, (pts_m0 + nrm_m0).astype(np.float64), theta) - pts_m
    nrm_m = nrm_m / np.clip(np.linalg.norm(nrm_m, axis=1, keepdims=True), 1e-12, None)
    points = np.concatenate([pts_b, pts_m.astype(np.float32)], 0)
    normals = np.concatenate([nrm_b, nrm_m.astype(np.float32)], 0)
    part_id = np.concatenate([np.zeros(n_b, np.int8), np.ones(n_m, np.int8)])
    kps = fps_keypoints(points, KP_K)
    # which part each keypoint lies on (exactly one point of the cloud, so an equality search)
    kp_part = [int(part_id[int(np.argmin(np.linalg.norm(points - kp, axis=1)))]) for kp in kps]
    ax, anc = base_frame_joint(spec)
    return {
        "points": points,
        "normals": normals,
        "part_id": part_id,
        "keypoints": kps,
        "kp_part": kp_part,
        "n_base": n_b,
        "n_moving": n_m,
        "area_base": a_b,
        "area_moving": a_m,
        "joint_type": str(spec["type"]),
        "joint_axis": ax.astype(np.float32),
        "joint_anchor": anc.astype(np.float32),
        "merge_theta": float(theta),
        "spec_hi": float(spec["hi"]),
    }


def surface_problem(spec: dict, sp_path: Path, policy: str = THETA_DEFAULT) -> str:
    """Reason the asset's surface point and keypoint files are not what this generator would write for the current spec
    ('' = current). The metadata runtime FK reads (kind · base-frame axis · anchor · merge angle), the source location
    and
    the point count are re-derived from the spec and compared, and the keypoint file must hold this cloud's keypoints
    (the
    two files are replaced in turn, so a crash in between leaves them split). Measured from state, so leftovers of a run
    that died during the surface step are caught too. Consumer: the spec stage's unfinished check
    (dexengine.data.artic.spec_stage._unfinished)."""
    sp_path = Path(sp_path)
    kp_path = sp_path.parent / f"keypoints_fps{KP_K}.npy"
    if not sp_path.is_file() or not kp_path.is_file():
        return f"file missing ({sp_path.name if not sp_path.is_file() else kp_path.name})"
    need = ("points", "part_id", "joint_type", "joint_axis", "joint_anchor", "merge_theta", "source")
    with np.load(sp_path, allow_pickle=False) as z:
        if set(need) - set(z.files):
            return f"fields missing {sorted(set(need) - set(z.files))}"
        pts, jt, src = z["points"], str(z["joint_type"]), str(z["source"])
        ax, anc, th = z["joint_axis"], z["joint_anchor"], float(z["merge_theta"])
    ax0, anc0 = base_frame_joint(spec)
    want_src = str((_assets_root(spec["dataset"]) / spec["obj_id"]).relative_to(ROOT))
    for bad, why in (
        (jt != str(spec["type"]), f"kind {jt} ≠ spec {spec['type']}"),
        (pts.shape[0] != N_SURF, f"point count {pts.shape[0]} ≠ {N_SURF}"),
        (not np.allclose(ax, ax0, atol=STALE_TOL), f"axis {ax.round(5).tolist()} ≠ spec"),
        (not np.allclose(anc, anc0, atol=STALE_TOL), f"anchor {anc.round(5).tolist()} ≠ spec"),
        (abs(th - _theta_of(policy, spec)) > STALE_TOL, f"merge angle {th:.5f} ≠ policy {policy}"),
        (src != want_src, f"source location {src} ≠ {want_src}"),
    ):
        if bad:
            return why
    if not np.allclose(np.load(kp_path), fps_keypoints(pts, KP_K), atol=STALE_TOL):
        return "keypoints do not belong to this point cloud"
    return ""


def main() -> int:
    args = _parse_args()

    specs = _load_specs(args.dataset)
    want = sorted(specs) if not args.ids else [t.strip() for t in args.ids.split(",") if t.strip()]
    missing = [w for w in want if w not in specs]
    if missing:
        print(f"[FATAL] asset ids not in the spec: {missing}", file=sys.stderr)
        return 2
    assets_root = _assets_root(args.dataset)
    meshes_root = ROOT / dl.render("meshes", args.dataset)
    spec_sha = hashlib.sha256(SPEC_FILE.read_bytes()).hexdigest()[:12]
    print(
        f"[artic_surface] dataset={args.dataset} spec {len(specs)} · targets {len(want)} · theta={args.theta} · "
        f"assets={assets_root} → meshes={meshes_root} · spec sha {spec_sha}",
        flush=True,
    )
    ledger = _new_ledger(args, specs, spec_sha)
    done = 0
    for oid in want:
        spec = specs[oid]
        mesh_dir = assets_root / spec["obj_id"]
        out_dir = meshes_root / oid
        theta = _theta_of(args.theta, spec)
        sp_path = out_dir / "surface_points.npz"
        kp_path = out_dir / f"keypoints_fps{KP_K}.npy"
        prev = None
        if sp_path.is_file():
            z = np.load(sp_path)
            prev = {
                "n_points": int(z["points"].shape[0]),
                "had_part_id": bool("part_id" in z.files),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(sp_path.stat().st_mtime)),
            }
            if not args.force:
                print(f"[SKIP] {oid}: {sp_path} exists (--force overwrites)", flush=True)
                continue
        r = build_one(oid, spec, mesh_dir, theta, args.seed)
        entry = _ledger_entry(r, mesh_dir, spec, prev)
        print(
            f"[artic_surface] {oid:16s} base {r['n_base']:4d} + moving {r['n_moving']:4d} = {N_SURF} "
            f"(area {r['area_base']:.4f}/{r['area_moving']:.4f} m²) θ={theta:+.4f} kp_part={r['kp_part']} "
            f"prev={prev}",
            flush=True,
        )
        if not args.dry_run:
            _write_surface(r, out_dir, sp_path, kp_path, mesh_dir, spec_sha)
        ledger["objects"][oid] = entry
        done += 1
    ledger["n_done"] = done
    led = P.clips_record(LEDGER_FILE, bool(args.ids))  # a --ids subset run does not overwrite the full-spec ledger
    if not args.dry_run:
        led.write_text(json.dumps(ledger, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ok = (done == len(specs)) if not args.ids else (done == len(want))
    print(
        f"[artic_surface] done {done}/{len(specs)} (whole spec) · ledger {stated(led)} · {'OK' if ok else 'MISMATCH'}",
        flush=True,
    )
    return 0 if ok else 1


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--theta", default=THETA_DEFAULT, help="closed (default, θ=0) | hi (max opening over clips) | <rad>"
    )
    ap.add_argument(
        "--ids",
        default="",
        help="comma-separated asset ids (subset); the ledger is still checked against the whole spec",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing surface_points.npz/keypoints_fps2.npy")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    return args


def _new_ledger(args, specs, spec_sha) -> dict:
    """Ledger of this run (objects filled per asset)."""
    ledger = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "tool": "dexengine/data/artic/surface.py",
        "dataset": args.dataset,
        "theta_policy": args.theta,
        "n_points": N_SURF,
        "keypoints_k": KP_K,
        "spec_file": stated(SPEC_FILE),
        "spec_sha256_12": spec_sha,
        "fk_owner": "scripts/placement/coords.artic_moving_vertices",
        "n_spec": len(specs),
        "n_done": 0,
        "objects": {},
    }
    return ledger


def _ledger_entry(r, mesh_dir, spec, prev) -> dict:
    """Ledger record of one built asset (counts, areas, joint, keypoints, source, previous file)."""
    entry = {
        "n_base": r["n_base"],
        "n_moving": r["n_moving"],
        "area_base": round(r["area_base"], 6),
        "area_moving": round(r["area_moving"], 6),
        "merge_theta": r["merge_theta"],
        "spec_hi": r["spec_hi"],
        "joint_type": r["joint_type"],
        "joint_axis": r["joint_axis"].round(6).tolist(),
        "joint_anchor": r["joint_anchor"].round(6).tolist(),
        "keypoints": r["keypoints"].round(4).tolist(),
        "kp_part": r["kp_part"],
        "source": str(mesh_dir.relative_to(ROOT)),
        "parts": dict(spec["parts"]),
        "prev": prev,
    }
    return entry


def _write_surface(r, out_dir, sp_path, kp_path, mesh_dir, spec_sha) -> None:
    """Write the surface point and keypoint files of one asset (each replaced atomically, in that order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = sp_path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        points=r["points"],
        normals=r["normals"],
        n_points=np.int64(N_SURF),
        part_id=r["part_id"],
        joint_type=r["joint_type"],
        joint_axis=r["joint_axis"],
        joint_anchor=r["joint_anchor"],
        merge_theta=np.float32(r["merge_theta"]),
        source=str(mesh_dir.relative_to(ROOT)),
        generator="dexengine/data/artic/surface.py",
        spec_sha256_12=spec_sha,
    )
    os.replace(tmp, sp_path)
    tmpk = kp_path.with_suffix(".tmp.npy")
    np.save(tmpk, r["keypoints"])
    os.replace(tmpk, kp_path)


if __name__ == "__main__":
    sys.exit(main())
