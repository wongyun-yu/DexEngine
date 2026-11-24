"""OakInk2 source colour — when the official repair mesh has no texture (the `NO_TEXTURE` mark), carry the colour of the
source scan (`object_raw`) as a derived texture OBJ. Rule "source colour, else one colour per object": a source colour
(texture · vertex colours · material Kd) comes before the name rule (dexengine.robots.textures.colors). Reader:
dexengine.robots.textures.rules._oakink2_raw (the source re-intake entry — it treats the returned OBJ as the source).
  ① vertex colours — when the header of `object_raw/align_ds/<rid>/*.ply` has red · green · blue, unwrap the repair
     mesh (uv_robust) and give each texel the inverse-distance-weighted mean of the IDW_K nearest scan vertex colours
     at that object point (sRGB as is).
  ② material colour — when the material with the most faces (main material) of `object_raw/align_ds/<rid>/*.obj` has
     only Kd (no map_Kd) and Kd ≠ (1,1,1): that one Kd colour. The Kd of a material with map_Kd multiplies a texture
     and is not a colour (even when the image is not distributed) · Kd (1,1,1) is the MTL default.
  Neither → None — the last step (one colour per object · dexengine.robots.textures.fill).
The derived OBJ copies the repair `v` lines verbatim and keeps the face order (the generator checks corner positions at
1e-6). A source scan that does not match the repair mesh (surface distance p95 > ALIGN_P95_M) is an error — another
object's colour is never carried silently.
"""

from __future__ import annotations

import glob
import os

import numpy as np

ALIGN_P95_M = 0.005  # cap of the repair vertex → source scan surface distance p95 (glass 5 · S10021 ≤ 3.0 mm)
TEX_PX = 1024  # vertex-colour atlas side
IDW_K = 4  # texel colour = inverse-distance-weighted mean of the k nearest scan vertices
NO_TEX_MARK = "NO_TEXTURE"


def _ply_has_color(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(4096).split(b"end_header")[0].decode("ascii", "replace").split("\n")
    props = {ln.split()[-1] for ln in head if ln.startswith("property")}
    return {"red", "green", "blue"} <= props


def _read_obj(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(raw v lines · V (n,3) · F (m,3) 0-based) — triangles with positive indices only (anything else is an error)."""
    vlines, faces = [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                vlines.append(ln.rstrip("\n"))
            elif ln.startswith("f "):
                ids = [int(t.split("/")[0]) for t in ln.split()[1:]]
                if len(ids) != 3 or min(ids) < 1:
                    raise ValueError(f"{path}: a face that is not a triangle with positive indices {ln.strip()!r}")
                faces.append([i - 1 for i in ids])
    V = np.array([[float(x) for x in ln.split()[1:4]] for ln in vlines])
    return vlines, V, np.array(faces, dtype=np.int64)


def kd_color(raw_obj: str) -> tuple[tuple[float, float, float], str] | None:
    """Kd of the main material (② above) — (colour, reason) or None."""
    from dexengine.robots.textures import rules as OT

    d = os.path.dirname(raw_obj)
    mats, cnt, cur = {}, {}, None
    with open(raw_obj) as f:
        for ln in f:
            if ln.startswith("mtllib"):
                p = os.path.join(d, ln.split(None, 1)[1].strip())
                if os.path.isfile(p):
                    mats.update(OT.parse_mtl(p))
            elif ln.startswith("usemtl"):
                cur = ln.split(None, 1)[1].strip()
            elif ln.startswith("f "):
                cnt[cur] = cnt.get(cur, 0) + 1
    if not cnt:
        return None
    main = max(cnt, key=cnt.get)
    m = mats.get(main) or {}
    kd = tuple(float(x) for x in (m.get("kd") or "").split()[:3])
    if "map_kd" in m or len(kd) != 3 or kd == (1.0, 1.0, 1.0):
        return None
    return kd, (
        f"source material colour Kd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f} (main material {main} faces "
        f"{cnt[main]}/{sum(cnt.values())})"
    )


def _check_align(V: np.ndarray, scan_mesh, rid: str) -> None:
    import trimesh

    _, dist, _ = trimesh.proximity.closest_point(scan_mesh, V[:: max(1, len(V) // 2000)])
    p95 = float(np.percentile(dist, 95))
    if p95 > ALIGN_P95_M:
        raise ValueError(
            f"oakink2/{rid}: the source scan does not match the repair mesh (surface distance p95 "
            f"{p95 * 1e3:.1f} mm > {ALIGN_P95_M * 1e3:.0f} mm) — the colour is not carried"
        )


def _raster(V: np.ndarray, F: np.ndarray, UV: np.ndarray, px: int):
    """UV-space raster — (texel rows · columns · object points P)."""
    rows, cols, pts = [], [], []
    for f in range(len(F)):
        xy = np.stack([UV[f, :, 0] * px - 0.5, (1.0 - UV[f, :, 1]) * px - 0.5], 1)
        x0, y0 = np.floor(xy.min(0)).astype(int)
        x1, y1 = np.ceil(xy.max(0)).astype(int)
        gx, gy = np.meshgrid(np.arange(max(x0, 0), min(x1, px - 1) + 1), np.arange(max(y0, 0), min(y1, px - 1) + 1))
        q = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
        a, b, c = xy
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12 or len(q) == 0:
            continue
        w0 = ((b[1] - c[1]) * (q[:, 0] - c[0]) + (c[0] - b[0]) * (q[:, 1] - c[1])) / den
        w1 = ((c[1] - a[1]) * (q[:, 0] - c[0]) + (a[0] - c[0]) * (q[:, 1] - c[1])) / den
        w = np.stack([w0, w1, 1.0 - w0 - w1], 1)
        ok = (w >= -1e-3).all(1)
        if ok.any():
            rows.append(q[ok, 1].astype(int))
            cols.append(q[ok, 0].astype(int))
            pts.append(w[ok] @ V[F[f]])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(pts)


def _write(out: str, vlines: list[str], F: np.ndarray, uv: np.ndarray, img: np.ndarray, png: str) -> str:
    """Derived OBJ (raw v lines · three vt per face) + MTL + image."""
    from PIL import Image

    os.makedirs(out, exist_ok=True)
    Image.fromarray(img).save(os.path.join(out, png))
    with open(os.path.join(out, "model.mtl"), "w") as f:
        f.write(f"newmtl orig\nKd 1 1 1\nmap_Kd {png}\n")
    with open(os.path.join(out, "model.obj"), "w") as f:
        f.write("mtllib model.mtl\n" + "\n".join(vlines) + "\n")
        f.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in uv.reshape(-1, 2)))
        f.write("usemtl orig\n")
        f.write(
            "".join(f"f {a + 1}/{3 * i + 1} {b + 1}/{3 * i + 2} {c + 1}/{3 * i + 3}\n" for i, (a, b, c) in enumerate(F))
        )
    return os.path.join(out, "model.obj")


def derive(rid: str, repair_obj: str, raw_dir: str, scratch: str) -> str | None:
    """Path of the derived texture OBJ when there is a source colour, else None (① then ② above)."""
    import trimesh

    out = os.path.join(scratch, "oakink2_raw_color", rid)
    vlines, V, F = _read_obj(repair_obj)
    plys = [p for p in sorted(glob.glob(os.path.join(raw_dir, "*.ply"))) if _ply_has_color(p)]
    if plys:
        s = trimesh.load(plys[0], force="mesh", process=False)
        _check_align(V, s, rid)
        from scipy.ndimage import distance_transform_edt
        from scipy.spatial import cKDTree

        UV, _ = uv_robust(V, F)
        r, c, P = _raster(V, F, UV, TEX_PX)
        col = np.asarray(s.visual.vertex_colors)[:, :3]
        img = np.zeros((TEX_PX, TEX_PX, 3), np.uint8)
        d, j = cKDTree(np.asarray(s.vertices)).query(P, k=IDW_K)  # IDW mean of the k nearest scan vertices
        w = 1.0 / np.maximum(d, 1e-6)
        img[r, c] = np.round((col[j] * w[..., None]).sum(1) / w.sum(1)[:, None]).astype(np.uint8)
        hole = np.ones((TEX_PX, TEX_PX), bool)
        hole[r, c] = False
        idx = distance_transform_edt(hole, return_distances=False, return_indices=True)
        return _write(out, vlines, F, UV, img[idx[0], idx[1]], "scan_vertex_color.png")
    objs = sorted(glob.glob(os.path.join(raw_dir, "*.obj")))
    k = kd_color(objs[0]) if objs else None
    if k is None:
        return None
    _check_align(V, trimesh.load(objs[0], force="mesh", process=False), rid)
    lin = np.array(k[0])  # Kd = linear diffuse colour (read like the material colour rule) → the image is sRGB
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    rgb = np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)
    return _write(out, vlines, F, np.full((len(F), 3, 2), 0.5), np.tile(rgb, (4, 4, 1)), "raw_kd_color.png")


def for_repair(repair_obj: str, raw_root: str, rid: str, scratch: str) -> str:
    """The `_oakink2_raw` entry — the source-colour derived OBJ (when there is one) if the repair folder has the
    NO_TEXTURE mark, else the repair OBJ as is."""
    if not os.path.isfile(os.path.join(os.path.dirname(repair_obj), NO_TEX_MARK)):
        return repair_obj
    raw_dir = os.path.join(raw_root, "hf", "obj", "object_raw", "align_ds", rid)
    return derive(rid, repair_obj, raw_dir, scratch) or repair_obj


def uv_parameterize_uvatlas(
    vertices, faces, size=1024, gutter=2.5, max_stretch=0.1666666716337204, parallel_partitions=16, nthreads=0
):
    """open3d uvatlas → per-face UV (#F, 3, 2). Source: MV-Adapter (huanngzh/MV-Adapter, Apache-2.0),
    mvadapter/utils/mesh_utils/mesh_process.py, the function of the same name with the same arguments and defaults."""
    import open3d as o3d

    device = o3d.core.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32), o3d.core.float32, device)
    mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int64), o3d.core.int64, device)
    mesh.compute_uvatlas(
        size=size, gutter=gutter, max_stretch=max_stretch, parallel_partitions=parallel_partitions, nthreads=nthreads
    )
    return mesh.triangle.texture_uvs.numpy()


def uv_robust(V, F):
    """Per-face UV atlas robust to duplicate and zero-area faces → (uv (#F, 3, 2) float32, method used). Vertex and
    face order stay those of the input mesh (per-face UV of the training mesh).

    open3d uvatlas. When it refuses duplicate faces (same three vertices) as non-manifold, the duplicates are removed
    for unwrapping and each removed face takes its twin's UV (vertex/face order unchanged). When uvatlas still refuses,
    xatlas (face order preserved) — the method is returned (result.json uv_method)."""
    try:
        return uv_parameterize_uvatlas(V, F).astype(np.float32), "uvatlas"
    except RuntimeError as e:
        if "Non-manifold" not in str(e):
            raise
    key = np.sort(F, 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    keep = np.sort(first)
    pos = np.empty(len(F), np.int64)
    pos[keep] = np.arange(len(keep))
    try:
        uvk = uv_parameterize_uvatlas(V, F[keep]).astype(np.float32)
        how = f"uvatlas_dedup(dup_faces={len(F) - len(keep)})"
    except RuntimeError:
        # uvatlas refuses the partition itself (many zero-area faces) — unwrap with xatlas (face order preserved)
        import xatlas

        _, idx, uvs = xatlas.parametrize(V.astype(np.float32), F[keep].astype(np.uint32))
        assert len(idx) == len(keep)
        uvk = uvs[idx].astype(np.float32)
        how = f"xatlas_dedup(dup_faces={len(F) - len(keep)}, uvatlas_failed)"
    rep = first[inv.reshape(-1)]  # representative (first occurrence) face per face
    uv = uvk[pos[rep]]
    # rotate the UV to the face's vertex order when it differs from the representative's
    for i in np.where(rep != np.arange(len(F)))[0]:
        r = rep[i]
        uv[i] = uvk[pos[r]][[list(F[r]).index(v) for v in F[i]]]
    return uv, how
