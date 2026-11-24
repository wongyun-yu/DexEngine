"""UV transfer from a raw textured mesh onto our visual mesh: the raw re-intake branch of the render overlay generator
(render_overlay.py) for links whose topology differs from the raw mesh, so faces do not correspond (ARCTIC
`mesh_tex.obj`).

The shape of our visual mesh is kept: the overlay only adds face-vertex UVs (`primvars:st`) and a material binding to
the visual mesh (collision is untouched). The raw UV is carried over by nearest points.

Rules (for the links that share one raw OBJ):
  1. Scale = a power of ten (raw size / our size). Each raw face goes to the link nearest its face centre (part split,
     used only for alignment).
  2. Rigid alignment: per part, ICP onto our link surface (rotation and translation only, closest ICP_TRIM fraction of
     matches). The pose is used only when it lowers the residual p95 by more than ICP_MIN_GAIN (a pose that slid on a
     symmetric shape does not lower the residual, so the texture does not rotate). A pose beyond ICP_MAX_ROT_DEG or
     ICP_MAX_T_MM raises (not the same object in the same pose: the raw location rule is wrong).
  3. Residual = distance from our link surface (N_RESID area samples) to the whole raw surface in that link's pose
     (transfer direction: is there raw surface next to ours to read from; the whole surface so that a wrong part split
     at part borders is not measured). Surfaces our mesh encloses from all sides (part contacts, inner faces, absent
     from the raw scan) are excluded: only samples beyond XFER_P95_MM cast ENCLOSED_RAYS rays (normal + a 60° ring
     around it), and a sample whose rays all hit our mesh is enclosed. Gate = p95 of the remaining samples
     <= XFER_P95_MM for every link (p99 is only recorded).
  4. UV: N_UV area samples of the whole raw mesh in that pose (each sample's UV by barycentric interpolation; chart =
     faces connected in UV). Each of our faces takes the chart of the sample nearest its centre, and each face vertex
     takes the UV of the nearest sample of that same chart (searching only that chart's samples when none is within
     K_CHART). So no face crosses a UV seam.
Failure = (None, RAW_MISMATCH + alignment detail); the generator then uses the final-stage uniform
colour (rules.OK_ALIGN_UNIFORM). Returned status and `why` strings are recorded data.
"""

from __future__ import annotations

import numpy as np

from dexengine.robots.textures import rules as OT

#: Transfer gate: p95 (mm) of the distance from our unenclosed surface to the raw surface. A wrong correspondence
#: (another object, another pose, a slipped alignment) moves most of the surface away, which p95 catches. On 22 ARCTIC
#: links the ICP corrections were all <= 0.23° and 0.14 mm (already aligned) while p99 spread over 0.09–3.2 mm: the
#: tail is local shape difference (scan vs our mesh) along the normal, which does not shift the texture sideways.
XFER_P95_MM = 2.5
ICP_TRIM, ICP_ITERS, ICP_MIN_GAIN = 0.9, 50, 0.1
ICP_MAX_ROT_DEG, ICP_MAX_T_MM = 10.0, 5.0
N_RESID, N_ICP, N_UV = 20000, 20000, 2_000_000
ENCLOSED_RAYS, K_CHART = 9, 32


def _surfaces(train_usd: str, links: list[str]):
    """Link -> visual mesh in world coordinates (trimesh), plus all visual meshes of the object merged (for the
    enclosure test). Raises on non-triangle faces."""
    import trimesh

    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(train_usd)
    allv = {}
    for p in st.Traverse():
        if not str(p.GetPath()).endswith("/visuals/mesh"):
            continue
        m = UsdGeom.Mesh(p)
        if set(np.asarray(m.GetFaceVertexCountsAttr().Get())) != {3}:
            raise ValueError(f"{p.GetPath()} has non-triangle faces — face-vertex UVs cannot be written as (3F,2)")
        M = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        w = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64) @ M[:3, :3] + M[3, :3]  # USD uses row vectors
        allv[str(p.GetPath()).split("/")[2]] = trimesh.Trimesh(
            w, np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False
        )
    miss = [L for L in links if L not in allv]
    if miss:
        raise ValueError(f"no visual prim {miss}")
    return {L: allv[L] for L in links}, trimesh.util.concatenate(list(allv.values()))


def _kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R (no reflection) and translation t with b ≈ a @ R.T + t."""
    ca, cb = a.mean(0), b.mean(0)
    u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ d @ u.T
    return r, cb - ca @ r.T


def _icp(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree, r, t = cKDTree(tgt), np.eye(3), np.zeros(3)
    for _ in range(ICP_ITERS):
        cur = src @ r.T + t
        d, j = tree.query(cur)
        k = d <= np.quantile(d, ICP_TRIM)
        dr, dt = _kabsch(cur[k], tgt[j[k]])
        r, t = dr @ r, t @ dr.T + dt
        if np.linalg.norm(dt) < 1e-8 and np.trace(dr) > 3 - 1e-12:
            break
    return r, t


def _enclosed(mesh, pts: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """True per sample when the normal ray and the 60° ring of rays all hit our mesh (surface enclosed on all sides)."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=nrm.shape)
    tan = u - (u * nrm).sum(1, keepdims=True) * nrm
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    bit = np.cross(nrm, tan)
    out = np.ones(len(pts), bool)
    for i in range(ENCLOSED_RAYS):
        if i == 0:
            d = nrm
        else:
            a = 2 * np.pi * (i - 1) / (ENCLOSED_RAYS - 1)
            d = 0.5 * nrm + 0.866 * (np.cos(a) * tan + np.sin(a) * bit)
        out &= mesh.ray.intersects_any(pts + nrm * 1e-5, d)
        if not out.any():
            break
    return out


def _resid(raw, link, whole, samples) -> dict:
    import trimesh

    pts, fi = samples
    d = trimesh.proximity.closest_point(raw, pts)[1] * 1e3
    enc = np.zeros(len(d), bool)
    far = d > XFER_P95_MM
    if far.any():
        enc[far] = _enclosed(whole, pts[far], link.face_normals[fi[far]])
    k = ~enc
    return {"p95": float(np.percentile(d[k], 95)), "p99": float(np.percentile(d[k], 99)), "enc": float(enc.mean())}


def _uv_samples(raw, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw area samples (raw coordinates), their UVs (barycentric interpolation) and charts (index of UV-connected face
    groups)."""
    import trimesh

    ch = np.empty(len(raw.faces), np.int64)
    for i, c in enumerate(trimesh.graph.connected_components(raw.face_adjacency, nodes=np.arange(len(ch)))):
        ch[c] = i
    pts, fi = trimesh.sample.sample_surface(raw, N_UV, seed=0)
    bary = trimesh.triangles.points_to_barycentric(raw.triangles[fi], pts)
    return pts, np.einsum("nk,nkj->nj", bary, uv[raw.faces[fi]]), ch[fi]


def _uv(pts: np.ndarray, suv: np.ndarray, sch: np.ndarray, link) -> tuple[np.ndarray, float]:
    """Face-vertex UVs (3F,2) from the nearest sample of the same chart (rule 4), and the fraction of vertices with no
    same-chart sample within K_CHART (chart borders)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    want = np.repeat(sch[tree.query(link.triangles_center)[1]], 3)
    cor = link.vertices[link.faces].reshape(-1, 3)
    j = tree.query(cor, k=K_CHART)[1]
    ok = sch[j] == want[:, None]
    pick = j[np.arange(len(j)), ok.argmax(1)]
    miss = np.flatnonzero(~ok.any(1))  # no same-chart sample within K_CHART: search that chart's samples only
    if miss.size:
        order = np.argsort(sch, kind="stable")
        srt, wm = sch[order], want[miss]
        for c in np.unique(wm):
            q = miss[wm == c]
            idx = order[np.searchsorted(srt, c, "left") : np.searchsorted(srt, c, "right")]
            pick[q] = idx[cKDTree(pts[idx]).query(cor[q])[1]]
    return suv[pick].astype(np.float32), float(miss.size / len(cor))


def transfer(train_usd: str, links: list[str], raw_obj: str) -> tuple[dict | None, str]:
    """-> ({link: {"st" (3F,2) float32, "why"}}, "") | (None, RAW_MISMATCH + detail). One link failing the gate
    fails them all."""
    import trimesh
    from scipy.spatial import cKDTree

    surf, whole = _surfaces(train_usd, links)
    r = trimesh.load(raw_obj, force="mesh", process=False)
    uv = getattr(r.visual, "uv", None)
    if uv is None or len(uv) != len(r.vertices):
        return None, f"{OT.RAW_MISMATCH}:no raw UV"
    ours = trimesh.util.concatenate(list(surf.values()))
    rv = r.vertices * 10.0 ** np.round(np.log10(ours.extents.max() / r.extents.max()))
    raw = trimesh.Trimesh(rv, r.faces, process=False)
    lab = np.stack([cKDTree(surf[L].vertices).query(rv[r.faces].mean(1))[0] for L in links], 1).argmin(1)
    spts, suv, sch = _uv_samples(raw, np.asarray(uv, dtype=np.float64))
    out, bad = {}, []
    for i, L in enumerate(links):
        f = r.faces[lab == i]
        if not len(f):
            return None, f"{OT.RAW_MISMATCH}:link {L} has 0 raw faces"
        samples = trimesh.sample.sample_surface(surf[L], N_RESID, seed=0)
        r0 = _resid(raw, surf[L], whole, samples)
        rot, t = _icp(
            trimesh.sample.sample_surface(trimesh.Trimesh(rv, f, process=False), N_ICP, seed=1)[0],
            trimesh.sample.sample_surface(surf[L], 10 * N_ICP, seed=2)[0],
        )
        deg = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
        tmm = float(np.linalg.norm(t) * 1e3)
        if deg > ICP_MAX_ROT_DEG or tmm > ICP_MAX_T_MM:
            raise ValueError(
                f"{L} ICP pose {deg:.2f}° · {tmm:.2f} mm is beyond the limit ({ICP_MAX_ROT_DEG}° · "
                f"{ICP_MAX_T_MM} mm) — raw {raw_obj} is not the same object in the same pose"
            )
        r1 = _resid(trimesh.Trimesh(rv @ rot.T + t, r.faces, process=False), surf[L], whole, samples)
        use = r1["p95"] < (1 - ICP_MIN_GAIN) * r0["p95"]
        res = r1 if use else r0
        if res["p95"] > XFER_P95_MM:
            bad.append(f"{L} p95 {res['p95']:.2f} mm")
        st, off = _uv(spts @ rot.T + t if use else spts, suv, sch, surf[L])
        out[L] = {
            "st": st,
            "why": (
                f"UV transfer · residual p95/p99 {res['p95']:.2f}/{res['p99']:.2f} mm (before ICP "
                f"{r0['p95']:.2f}/{r0['p99']:.2f}) · ICP {deg:.2f}° {tmm:.2f} mm "
                f"{'adopted' if use else 'dropped'} · occluded {res['enc'] * 100:.1f} % · "
                f"chart-border vertices {off * 100:.2f} %"
            ),
        }
    if bad:
        return None, f"{OT.RAW_MISMATCH}:alignment {' · '.join(bad)} > {XFER_P95_MM}"
    return out, ""


# Render-cover atlas: several part images (merged OakInk2 assets: a different source texture per part) carried as one
# link = one material. Cell = the source image as is (no resampling) · cells side by side · ATLAS_GUTTER px edge
# replication left and right of each cell (stops bilinear bleeding) · differing sizes are an error · wrap = clamp
# (the generator write_overlay). Consumer: dexengine/robots/objects/render_overlay.py.
#: Edge replication width (px) on each side of an atlas cell.
ATLAS_GUTTER = 8


def mtl_image(obj: str) -> str:
    t = OT.find_texture(obj)
    if t["status"] != OT.OK:
        raise ValueError(f"atlas: the image of {obj} does not pass the rules ({t['status']})")
    return t["image"]


def atlas_size(images: list[str]) -> tuple[int, int]:
    from PIL import Image

    sz = set()
    for p in images:
        with Image.open(p) as im:
            sz.add(im.size)
    if len(sz) != 1:
        raise ValueError(f"atlas: part images differ in size {sorted(sz)} — outside the rule (no resampling)")
    return sz.pop()


def atlas_st(uvs: list[np.ndarray], images: list[str]) -> np.ndarray:
    """Part k's (u, v) → atlas ((k·(W+2g) + g + u·W) / (N·(W+2g)), v). W = part image width (must be equal)."""
    W, _ = atlas_size(images)
    g, n = ATLAS_GUTTER, len(uvs)
    out = []
    for k, uv in enumerate(uvs):
        a = uv.copy()
        a[..., 0] = (k * (W + 2 * g) + g + uv[..., 0] * W) / (n * (W + 2 * g))
        out.append(a)
    return np.concatenate(out)


def atlas_image(images: list[str]) -> np.ndarray:
    """Part images → horizontal atlas (H, N·(W+2g), 3) — g px edge replication left and right of each cell · pixels as
    in the source."""
    from PIL import Image

    atlas_size(images)
    g = ATLAS_GUTTER
    arr = []
    for p in images:
        with Image.open(p) as im:
            arr.append(np.pad(np.asarray(im.convert("RGB")), ((0, 0), (g, g), (0, 0)), mode="edge"))
    return np.concatenate(arr, axis=1)
