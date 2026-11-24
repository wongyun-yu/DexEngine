#!/usr/bin/env python
"""The **only producer** of object surface samples: rigid surface pairs (surface_points.npz + keypoints_fps<k>.npy) and
articulated part surfaces (surface_parts.npz). The data layer (dexengine.data.contact.geom.ensure_surface) and this CLI
call the same functions.

  ensure_rigid(mesh, out)          out = surface_points.npz (points · normals · n_points) + keypoints_fps<k>.npy beside
      it
  ensure_parts(src, parts, out, …) out = surface_parts.npz (base_/moving_ points · normals · source); no keypoint pair
                                   (training keypoints of articulated objects live beside dexengine.data.artic.surface's
                                   merged point
                                       cloud)
  Existing files are kept, since point identity enters the trained contact and keypoint rewards (overwrite only with
  force_*). Keypoints are drawn from the folder's **actual** surface points (with only a surface, only the keypoints are
  filled in). Writes go through a temp file + os.replace.

Usage:
    python dexengine/data/object_surface.py --mesh data/taco/meshes/071/visual.obj [--n_points 2048]

Coordinate frame: object-local (whatever the .obj uses).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh

from dexengine.core import artifact_paths

N_SURF = 2048  # surface points per object (observation N_surface_pts); articulated part surfaces use half per part
KP_K = 2  # keypoints_fps<k>; the keypoint reward reads it beside surface_points.npz


def sample_surface_with_normals(mesh: trimesh.Trimesh, n_points: int, seed: int = 0):
    """Sample n_points uniformly on the mesh surface; return (points, normals).

    Uses trimesh.sample.sample_surface_even for uniform spacing (Poisson-disk-like).
    Falls back to sample_surface if even-sampling can't reach n_points.
    """
    # Try sample_surface_even first — produces uniform spacing
    try:
        points, face_idx = trimesh.sample.sample_surface_even(mesh, n_points, seed=seed)
        if len(points) < n_points:
            extra = n_points - len(points)
            extra_points, extra_face_idx = trimesh.sample.sample_surface(mesh, extra, seed=seed + 1)
            points = np.vstack([points, extra_points])
            face_idx = np.concatenate([face_idx, extra_face_idx])
    except Exception as e:  # noqa: BLE001  any failure of the even sampler falls back to the plain sampler
        print(f"[precompute] sample_surface_even failed ({e}); falling back to sample_surface")
        points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)

    points = points[:n_points]
    face_idx = face_idx[:n_points]

    # Per-point normal = the normal of the source face
    face_normals = mesh.face_normals[face_idx]
    return points.astype(np.float32), face_normals.astype(np.float32)


def fps_keypoints(points: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point keypoints on a surface cloud — **the single owner of the keypoint rule**
    (start = the point farthest from the centroid, then iterated farthest-from-set). Returns (k, 3) float32.
    One function, so dexengine/data/artic/surface.py reuses the same rule."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    kp_idx = [int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))]
    while len(kp_idx) < int(k):
        d = np.min(np.linalg.norm(pts[:, None] - pts[kp_idx][None], axis=-1), axis=1)
        kp_idx.append(int(np.argmax(d)))
    return pts[kp_idx].astype(np.float32)


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mesh not found: {path}")
    mesh = trimesh.load(path, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"loaded object is not a single Trimesh: {type(mesh)} ({path})")
    return mesh


def _atomic(path: Path, write) -> None:
    """write(temp path), then os.replace; never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    write(tmp)
    os.replace(tmp, path)


def ensure_rigid(
    mesh: str | Path,
    out: str | Path,
    *,
    n_points: int = N_SURF,
    seed: int = 0,
    k: int = KP_K,
    force_surface: bool = False,
    force_keypoints: bool = False,
) -> list[Path]:
    """Rigid surface pair; builds only what is missing (or forced). k = 0 builds no keypoints. Returns the files written
    now."""
    out = Path(out)
    kp = out.parent / f"keypoints_fps{k}.npy" if k > 0 else None
    wrote: list[Path] = []
    if force_surface or not out.exists():
        points, normals = sample_surface_with_normals(_load_mesh(mesh), n_points, seed)
        _atomic(out, lambda f: np.savez(f, points=points, normals=normals, n_points=np.int64(len(points))))
        wrote.append(out)
    if kp is not None and (force_keypoints or not kp.exists()):
        _atomic(kp, lambda f: np.save(f, fps_keypoints(np.load(out)["points"], k)))
        wrote.append(kp)
    return wrote


def link_mesh(src: str | Path, parts: dict, link: str, *, process: bool) -> trimesh.Trimesh:
    """Mesh of an articulated link (base|moving) = the link's parts ∪ parts rigidly attached to that link (moved by
    their
    poses), in the link frame. List owner = artifact_paths.artic_link_files · pose convention = coords.pose_points.
    Without attachments, the link parts as is."""
    from dexengine.data.artic.fk import pose_points  # noqa: PLC0415

    ms = []
    for rel, pose in artifact_paths.artic_link_files(parts, link):
        if not (Path(src) / rel).is_file():
            raise FileNotFoundError(f"mesh not found: {Path(src) / rel}")
        m = trimesh.load(
            Path(src) / rel, force="mesh", process=process
        )  # process = the caller's original reading as is
        ms.append(
            m
            if pose is None
            else trimesh.Trimesh(
                vertices=pose_points(np.asarray(m.vertices, float), pose), faces=np.asarray(m.faces), process=False
            )
        )
    return ms[0] if len(ms) == 1 else trimesh.util.concatenate(ms)


def ensure_parts(
    src: str | Path, parts: dict, out: str | Path, source: str, *, n_points: int = N_SURF, seed: int = 0
) -> list[Path]:
    """Articulated part surfaces: links base · moving (including rigid attachments, link_mesh) · n_points // 2 per link.
    Kept when present."""
    out = Path(out)
    if out.exists():
        return []
    arrays = {}
    for part in ("base", "moving"):
        pts, nrm = sample_surface_with_normals(link_mesh(src, parts, part, process=False), n_points // 2, seed)
        arrays[f"{part}_points"], arrays[f"{part}_normals"] = pts, nrm
    _atomic(out, lambda f: np.savez(f, **arrays, source=source))
    return [out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Path to .obj mesh file")
    ap.add_argument("--n_points", type=int, default=N_SURF)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None, help="Output .npz path (default: <mesh_dir>/surface_points.npz)")
    ap.add_argument(
        "--keypoints_k",
        type=int,
        default=KP_K,
        help="Also emit keypoints_fps<k>.npy (deterministic farthest-point sampling on "
        "the surface cloud; seed 0 start = point farthest from the centroid). 0 = skip. "
        "NOTE: assets made before this producer existed keep their keypoints — "
        "regenerating them changes keypoint identity, which trained rewards depend on.",
    )
    ap.add_argument(
        "--force_surface",
        action="store_true",
        help="Overwrite an existing surface_points.npz (default: keep — point identity feeds trained contact rewards).",
    )
    ap.add_argument(
        "--force_keypoints", action="store_true", help="Overwrite an existing keypoints file (default: keep)."
    )
    args = ap.parse_args()

    out = Path(args.output or Path(args.mesh).parent / "surface_points.npz")
    want = [out] + ([out.parent / f"keypoints_fps{args.keypoints_k}.npy"] if args.keypoints_k > 0 else [])
    wrote = ensure_rigid(
        args.mesh,
        out,
        n_points=args.n_points,
        seed=args.seed,
        k=args.keypoints_k,
        force_surface=args.force_surface,
        force_keypoints=args.force_keypoints,
    )
    print(f"[precompute] mesh: {args.mesh}")
    for f in want:
        print(f"[precompute] {'saved' if f in wrote else 'kept (--force_* to overwrite)'}: {f}")


if __name__ == "__main__":
    main()
