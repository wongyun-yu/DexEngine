"""S-true-shape (CPU): the collider-shape owner (pinned still kinds → triangles, everything else → convex parts, the
per-dataset opt-out) and the triangle ruler (nearest-face depth, surface samples, both-way pair reading on open shells,
the visual-mesh prim selection)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dexengine.core import datasets as RR
from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import extra_slots as XS
from dexengine.data.objects import triangle_geom as TGM


def _box_mesh(half, c=(0.0, 0.0, 0.0)):
    """A closed box as 12 outward-wound triangles."""
    import trimesh  # noqa: PLC0415

    m = trimesh.creation.box(extents=2 * np.asarray(half, float))
    return np.asarray(m.vertices) + np.asarray(c, float), np.asarray(m.faces)


def _sheet(half=0.05):
    """One open square sheet in z = 0 (two triangles, normal +z)."""
    V = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]])


def test_shape_owner_pinned_still_kinds_only():
    for k in ("receiver", "touched", "touched_solid", "envelope", CS.SUPPORT):
        assert CS.shape_of("hot3d", k, True) == CS.TRIANGLES
    # negative: the same kinds not pinned, and the moving kinds pinned or not, stay convex
    for k in ("receiver", "touched", "touched_solid", "envelope"):
        assert CS.shape_of("hot3d", k, False) == CS.PARTS
    for k in (XS.KIND_MOVING, XS.KIND_ATTACHED, None):
        assert CS.shape_of("hot3d", k, True) == CS.PARTS
    assert CS.row_shape("hot3d", {"kind": XS.KIND_RECEIVER, "static": True}) == CS.TRIANGLES
    assert CS.row_shape("hot3d", {"kind": XS.KIND_RECEIVER}) == CS.PARTS


def test_triangle_kinds_are_known_names():
    assert set(CS.TRIANGLE_KINDS) - {CS.SUPPORT} <= {*XS.KINDS, *XS.SCOPED_KINDS}
    assert CS.SUPPORT in XS.SPAWNED_ROLES


def test_opt_out_keeps_convex(monkeypatch):
    p = RR.DATASET_PROFILE["hot3d"]
    off = dict(p.scene_rules_off, **{CS.RULE_ID: "test opt-out"})
    monkeypatch.setitem(RR.DATASET_PROFILE, "hot3d", dataclasses.replace(p, scene_rules_off=off))
    assert CS.shape_of("hot3d", "receiver", True) == CS.PARTS
    assert CS.shape_of("taco", "receiver", True) == CS.TRIANGLES


def test_body_refuses_unknown_shape():
    with pytest.raises(ValueError, match="unknown collider shape"):
        CS.body("hot3d", "x", "hull")


def test_triangle_depth_closed_box():
    T = CS.Triangles(*_box_mesh((0.05, 0.05, 0.05)))
    P = np.array([[0, 0, 0.047], [0, 0, 0.0], [0, 0, 0.06], [0.2, 0, 0]])
    d = T.depths(P)
    assert d[0] == pytest.approx(0.003, abs=1e-6) and d[1] == pytest.approx(0.05, abs=1e-6)
    assert d[2] == 0.0 and d[3] == 0.0


def test_samples_cover_the_surface():
    """Every surface point is within SAMPLE_M of a sample (large faces get a grid, not only their corners)."""
    import trimesh  # noqa: PLC0415

    V, F = _box_mesh((0.1, 0.02, 0.07))
    T = CS.Triangles(V, F)
    pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), 4000, seed=0)
    d, _ = T.tree.query(pts)
    assert d.max() <= CS.SAMPLE_M


def test_open_sheet_straddle_reads_deep():
    """A convex body (20 mm thick) straddling an open sheet reads deep through the sheet's samples inside it, even
    through the middle of a large two-triangle quad (no vertex nearby) — at most SAMPLE_M low (the sample spacing);
    moved off the sheet it reads 0."""
    sheet = CS.Triangles(*_sheet())
    box = CS.Parts([np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * 0.01])
    M = np.eye(4)
    assert CS.pair_depth(sheet, box, M) >= 0.01 - CS.SAMPLE_M
    M[:3, 3] = (0.0, 0.0, 0.0101)
    assert CS.pair_depth(sheet, box, M) == 0.0


def test_thin_wall_crossing_reads_deep():
    """A convex body pushed through a 2 mm closed wall reads its own depth through the wall samples, not the wall's
    half thickness."""
    wall = CS.Triangles(*_box_mesh((0.06, 0.001, 0.06)))
    box = CS.Parts([np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * 0.008])
    assert wall.depth(box.points) <= 0.001 + 1e-9  # one way alone saturates at half the wall
    assert CS.pair_depth(wall, box, np.eye(4)) >= 0.008 - CS.SAMPLE_M


def _usd(tmp_path, visuals: int = 1, quads: bool = False):
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    f = tmp_path / "object.usd"
    st = Usd.Stage.CreateNew(str(f))
    UsdGeom.Xform.Define(st, "/Object")
    b = UsdGeom.Xform.Define(st, "/Object/base_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(b)
    V, F = _box_mesh((0.02, 0.02, 0.02))
    for i in range(visuals):
        m = UsdGeom.Mesh.Define(st, f"/Object/base_link/visuals/mesh{i}")
        m.CreatePointsAttr([tuple(v) for v in V])
        if quads:
            m.CreateFaceVertexCountsAttr([4])
            m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        else:
            m.CreateFaceVertexCountsAttr([3] * len(F))
            m.CreateFaceVertexIndicesAttr([int(x) for x in F.ravel()])
    c = UsdGeom.Mesh.Define(st, "/Object/base_link/collision_0")
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    st.Save()
    return st, b


def test_visual_mesh_is_the_one_mesh_under_visuals(tmp_path):
    _st, b = _usd(tmp_path)
    assert str(CS.visual_mesh(b).GetPath()) == "/Object/base_link/visuals/mesh0"


@pytest.mark.parametrize(("visuals", "quads", "why"), [(2, False, "want 1"), (1, True, "not a triangle")])
def test_visual_mesh_refuses_ambiguous_or_quads(tmp_path, visuals, quads, why):
    _st, b = _usd(tmp_path, visuals, quads)
    with pytest.raises(ValueError, match=why):
        CS.visual_mesh(b)


def test_rule_card_registered():
    assert RR.rule_on(CS.RULE_ID, "hot3d")


# ── inside = winding number > 1/2 (the sign rule of Triangles.signed) ─────────────────────────────────────────────


def _jar(a=0.05, h=0.05, c=0.02, d=0.04, t=0.01):
    """A closed block (half width a, height h) with a cavity open at the top through a narrower mouth: mouth half width
    c at z = h, floor half width d at z = t (the inner walls lean in, so a cavity point under the mouth is nearest to
    the mouth's rim edge or corner, where the rim face and the inner wall tie). Outward-wound, watertight."""
    import trimesh  # noqa: PLC0415

    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    V = np.concatenate(
        [
            np.c_[sq * a, np.zeros(4)],
            np.c_[sq * a, np.full(4, h)],
            np.c_[sq * c, np.full(4, h)],
            np.c_[sq * d, np.full(4, t)],
        ]
    )
    ob, ot, it, ib = 0, 4, 8, 12
    F = [[ob + 0, ob + 2, ob + 1], [ob + 0, ob + 3, ob + 2], [ib + 0, ib + 1, ib + 2], [ib + 0, ib + 2, ib + 3]]
    for k in range(4):
        n = (k + 1) % 4
        F += [[ob + k, ob + n, ot + n], [ob + k, ot + n, ot + k]]  # outer wall
        F += [[ot + k, ot + n, it + n], [ot + k, it + n, it + k]]  # rim
        F += [[it + k, it + n, ib + n], [it + k, ib + n, ib + k]]  # inner wall
    m = trimesh.Trimesh(V, np.array(F), process=False)
    m.fix_normals()
    assert m.is_watertight and m.volume > 0
    return np.asarray(m.vertices), np.asarray(m.faces)


def _single_face_signed(T, P):
    """The rule before the winding number: the sign of the first nearest candidate face's normal (the negative)."""
    k = min(CS.K_NEAR, len(T.points))
    _d, idx = T.tree.query(P, k=k)
    faces = T.sample_face[idx.reshape(len(P), k)]
    Tr = T.V[T.F[faces]]
    Q = TGM.closest_point(np.repeat(P[:, None], k, 1), Tr[..., 0, :], Tr[..., 1, :], Tr[..., 2, :])
    dist = np.linalg.norm(P[:, None] - Q, axis=-1)
    j = np.argmin(dist, axis=1)
    r = np.arange(len(P))
    s = ((P - Q[r, j]) * T.normals[faces[r, j]]).sum(-1)
    return np.where(s < 0, -dist[r, j], dist[r, j])


def _cavity_points():
    """Points in the jar's cavity just under the mouth: nearest to a rim edge (edge-nearest) or, toward the corners, a
    rim corner (vertex-nearest). All outside the solid."""
    g = np.linspace(-0.018, 0.018, 7)
    return np.array([[x, y, 0.0485] for x in g for y in g])


def test_winding_matches_brute_force_and_reads_cavity_outside():
    """(b) a closed mesh with an open cavity: every cavity point under the mouth reads outside (depth 0); the
    hierarchical winding number equals the brute-force sum."""
    T = CS.Triangles(*_jar())
    P = _cavity_points()
    assert len(P) == 49 and (T.depths(P) == 0).all()
    Q = np.random.default_rng(0).uniform(T.lo, T.hi, (200, 3))
    assert np.abs(T.winding(Q) - TGM.solid(T.V[T.F], Q)).max() < 1e-9


def test_single_face_sign_misreads_the_cavity():
    """(e) negative: the old single-face sign reads some of the same cavity points inside (the defect the winding
    number fixes) — the points are not trivially outside."""
    T = CS.Triangles(*_jar())
    P = _cavity_points()
    inb = np.all((T.lo < P) & (T.hi > P), axis=1)
    old = np.maximum(-_single_face_signed(T, P), 0.0) * inb
    assert (old > 0.005).any(), "the old rule should misread at least one rim point 5 mm deep"


def test_true_inside_reads_inside_at_its_depth():
    """(d) points inside the jar's wall and floor read inside at their distance to the nearest face."""
    T = CS.Triangles(*_jar())
    P = np.array([[0.045, 0.0, 0.025], [0.0, 0.0, 0.004], [-0.047, 0.047, 0.03]])
    want = np.array([0.005, 0.004, 0.003])
    assert T.depths(P) == pytest.approx(want, abs=1e-6)


def test_open_mesh_and_split_vertices():
    """(c) a box without its top face (not watertight): deep inside reads inside (winding near 1), just above the
    opening outside (winding under 1/2); the same box with every vertex split per face (a triangle soup) reads the
    same."""
    V, F = _box_mesh((0.05, 0.05, 0.05))
    top = np.flatnonzero(np.all(V[F][:, :, 2] > 0.049, axis=1))
    assert len(top) == 2
    Fo = np.delete(F, top, axis=0)
    T = CS.Triangles(V, Fo)
    P = np.array([[0.0, 0.0, -0.045], [0.0, 0.0, 0.06], [0.0, 0.0, 0.0]])
    w = T.winding(P)
    assert w[0] > 0.8 and w[1] < 0.5 and 0.5 < w[2] < 1.0
    assert T.depths(P[:1]) == pytest.approx([0.005], abs=1e-6)
    soup = CS.Triangles(V[Fo].reshape(-1, 3), np.arange(3 * len(Fo)).reshape(-1, 3))
    assert np.allclose(soup.winding(P), w, atol=1e-9)
    assert np.allclose(soup.depths(P), T.depths(P), atol=1e-9)


def _brute(T, P):
    return TGM.solid(T.V[T.F], P)


def _crossing_boxes():
    """Two closed boxes as one mesh, crossing each other (x in [-0.03, 0.03] and [-0.01, 0.05]): watertight and
    outward-wound part by part, the shape of the oakink2 mesh whose two shells overlap."""
    a, b = _box_mesh((0.03, 0.02, 0.02)), _box_mesh((0.03, 0.02, 0.02), (0.02, 0.0, 0.0))
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1] + len(a[0])])


def _meshes():
    """(name, V, F): closed, crossing shells, open, split vertices (a soup), inside-out."""
    import trimesh  # noqa: PLC0415

    s = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    V, F = _box_mesh((0.05, 0.05, 0.05))
    top = np.flatnonzero(np.all(V[F][:, :, 2] > 0.049, axis=1))
    return [
        ("jar", *_jar()),
        ("sphere", np.asarray(s.vertices), np.asarray(s.faces)),
        ("crossing", *_crossing_boxes()),
        ("open box", V, np.delete(F, top, axis=0)),
        ("open sphere", np.asarray(s.vertices), np.asarray(s.faces)[40:]),
        ("soup", V[F].reshape(-1, 3), np.arange(3 * len(F)).reshape(-1, 3)),
        ("inside-out", V, F[:, ::-1]),
    ]


def _probe(T, rng, n=600):
    d = rng.normal(size=(n, 3))
    S = T.points[rng.choice(len(T.points), n)]
    near = S + d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(0.0, 0.004, (n, 1))
    return np.concatenate([rng.uniform(T.lo - 0.01, T.hi + 0.01, (n, 3)), near])


def test_inside_equals_the_winding_test_on_every_mesh():
    """One rule: signed < 0 exactly where |winding number| > 1/2 (the brute-force sum over every face) on closed,
    crossing, open, split-vertex and inside-out meshes, at points spread over and beyond the box and points within 4 mm
    of the surface; the region shortcut (cover, closure) gives the same signs as summing every point."""
    rng = np.random.default_rng(3)
    for name, V, F in _meshes():
        T = CS.Triangles(V, F)
        assert T.sign_rule == "winding", name
        P = _probe(T, rng)
        if name == "jar":
            P = np.concatenate([P, _cavity_points()])
        want = np.abs(_brute(T, P)) > 0.5
        assert np.array_equal(T.signed(P) < 0, want), name
        summed = CS.Triangles(V, F)
        summed.cover = np.inf  # no free ball: every point summed
        assert np.array_equal(summed.signed(P) < 0, want), name
    V, F = _box_mesh((0.05, 0.05, 0.05))
    Q = rng.uniform(-0.06, 0.06, (400, 3))
    assert np.array_equal(CS.Triangles(V, F[:, ::-1]).depths(Q) > 0, CS.Triangles(V, F).depths(Q) > 0)


def test_crossing_shells_read_inside_where_the_nearest_face_is_the_other_shell():
    """(f) a point inside one closed shell and nearest to a face of the other reads inside at its depth; negative:
    the pseudo-normal of the nearest feature (here a face interior, so that face's normal: `_single_face_signed`) reads
    every such point outside — the shortcut the ruler took on closed meshes before."""
    T = CS.Triangles(*_crossing_boxes())
    g = np.linspace(-0.012, -0.011, 3)
    P = np.array([[x, y, z] for x in g for y in (-0.01, 0.0, 0.01) for z in (-0.01, 0.0, 0.01)])
    assert (np.abs(_brute(T, P)) > 0.5).all()
    assert T.depths(P) == pytest.approx(-0.01 - P[:, 0], abs=1e-9)  # to the other shell's face x = -0.01
    assert (_single_face_signed(T, P) > 0).all()


def test_within_equals_signed_at_most_r():
    """within(P, r) is signed(P) <= r on every point (crossing shells, an open sphere, the jar), r 0, 1 and 5 mm;
    a negative r is refused."""
    rng = np.random.default_rng(5)
    for name, V, F in _meshes()[:5]:
        T = CS.Triangles(V, F)
        P = _probe(T, rng)
        s = T.signed(P)
        for r in (0.0, 0.001, 0.005):
            assert np.array_equal(T.within(P, r), s <= r), (name, r)
    with pytest.raises(ValueError, match="r >= 0"):
        T.within(P, -0.001)


def test_closure_closes_the_surface():
    """The closure of an open mesh leaves surface ∪ closure without boundary edges; a closed mesh has none."""
    V, F = _box_mesh((0.05, 0.05, 0.05))
    assert CS.Triangles(V, F)._winding().closure is None
    W = CS.Triangles(V, F[2:])._winding()
    C = W.closure
    assert C is not None and len(C) > 0
    Vc = np.concatenate([V, C.reshape(-1, 3)])
    Fc = np.concatenate([F[2:], len(V) + np.arange(3 * len(C)).reshape(-1, 3)])
    assert not len(TGM.Winding(Vc, Fc)._boundary(np.arange(len(Fc)))[0])


def _pose(rng, spread):
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    M = np.eye(4)
    M[:3, :3] = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    M[:3, 3] = rng.uniform(-spread, spread, 3)
    return M


def test_pair_depth_within_is_pair_depth_or_none():
    """pair_depth_within(a, b, M, tol) equals pair_depth wherever pair_depth <= tol and is None wherever it exceeds
    tol, on random poses of a convex box and a triangle body against an open and a closed mesh."""
    import trimesh  # noqa: PLC0415

    rng = np.random.default_rng(7)
    s = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    closed, open_ = CS.Triangles(np.asarray(s.vertices), np.asarray(s.faces)), CS.Triangles(*_meshes()[4][1:])
    box = CS.Parts([np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * 0.01])
    seen = {"within": 0, "over": 0}
    for a in (closed, open_):
        for b in (box, CS.Triangles(*_box_mesh((0.01, 0.008, 0.012)))):
            poses = [_pose(rng, 0.05) for _ in range(40)]
            depths = [CS.pair_depth(a, b, M) for M in poses]
            tol = float(np.median(depths))
            for M, d in zip(poses, depths, strict=True):
                got = CS.pair_depth_within(a, b, M, tol)
                if d <= tol:
                    assert got == d
                    seen["within"] += 1
                else:
                    assert got is None
                    seen["over"] += 1
    assert min(seen.values()) > 0


def _depths_unbounded(T, P):
    """`Triangles.depths` with the nearest-sample search unbounded (the body before DEPTH_QUERY_M)."""
    out = np.zeros(len(P))
    inb = np.flatnonzero(np.all((T.lo < P) & (T.hi > P), axis=1))
    if len(inb):
        Q, d1 = P[inb], T.tree.query(P[inb], k=1)[0]
        ins = T.inside(Q, d1)
        need = np.flatnonzero(ins | (d1 <= T._reach(0.0)))
        d = T._nearest(Q[need])[0]
        out[inb[need]] = np.maximum(-np.where(ins[need], -d, d), 0.0)
    return out


def test_bounded_nearest_sample_search_gives_the_same_depths_bit_for_bit():
    """The search stops at DEPTH_QUERY_M (a speed setting): on every mesh, and on a large box whose inner points lie
    farther than that from every sample (so the bound is what they read), the depths are the bytes of the unbounded
    search's."""
    import trimesh  # noqa: PLC0415

    rng = np.random.default_rng(5)
    big = trimesh.creation.icosphere(subdivisions=4, radius=0.12)
    cases = [*_meshes(), ("big box", *_box_mesh((0.1, 0.08, 0.06))), ("big sphere", big.vertices, big.faces)]
    capped = 0
    for name, V, F in cases:
        T = CS.Triangles(np.asarray(V, float), np.asarray(F))
        P = np.concatenate([_probe(T, rng), rng.uniform(T.lo * 0.6, T.hi * 0.6, (400, 3))])
        assert T.depths(P).tobytes() == _depths_unbounded(T, P).tobytes(), name
        capped += int((T.tree.query(P, k=1)[0] > CS.DEPTH_QUERY_M).sum())
    assert capped > 500  # the bound is reached: deep points read it, not their nearest sample
