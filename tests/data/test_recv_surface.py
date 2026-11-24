"""Receiving distance on surfaces: an untouched still extra receives a moved object when the object ends
within RECV_M + slack of its surface (vertex-to-vertex distance missed objects put on broad faces — a keyboard) or with
its centre inside the extra's convex hull (a container — the same answer at any pose)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import recv_surface as RS

PLATE_V = np.array([[-0.2, -0.1, 0.0], [0.2, -0.1, 0.0], [0.2, 0.1, 0.0], [-0.2, 0.1, 0.0]])
PLATE_F = np.array([[0, 1, 2], [0, 2, 3]])
FAR, NEAR = np.array([[0.0, 0.0, 0.3]]), np.array([[0.0, 0.0, 0.012]])  # NEAR: inside RECV_M + slack, outside − slack
TOUCHING = np.array([[0.0, 0.0, 0.005]])  # inside RECV_M − slack: already at the receiver


def test_vertices_alone_miss_an_object_on_the_middle_of_a_broad_face():
    """The old measurement's gap: 12 mm above the centre of a 0.4 × 0.2 m plate is > 100 mm from every vertex."""
    assert not XG.arrives(FAR, NEAR, PLATE_V)


def test_surface_samples_see_it():
    S = RS.surface_points(PLATE_V, PLATE_F)
    assert len(S) > len(PLATE_V) and XG.arrives(FAR, NEAR, S)


@pytest.mark.parametrize(("start", "end"), [(TOUCHING, NEAR), (FAR, FAR)])
def test_near_from_the_start_or_far_at_the_end_is_not_arriving(start, end):
    assert not XG.arrives(start, end, RS.surface_points(PLATE_V, PLATE_F))


def test_the_given_inside_test_is_the_one_read():
    S = RS.surface_points(PLATE_V, PLATE_F)
    assert XG.arrives(FAR, FAR, S, inside=lambda c: True) and not XG.arrives(FAR, FAR, S, inside=lambda c: False)


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _tri_prism():
    """A triangular container (0.2 m legs, 0.1 m tall): half of its axis-aligned box lies outside it."""
    V = np.array([[x, y, z] for x, y in ((0, 0), (0.2, 0), (0, 0.2)) for z in (0.0, 0.1)])
    F = np.array([[0, 2, 4], [1, 5, 3], [0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4], [4, 5, 1], [4, 1, 0]])
    return V.astype(float), F


@pytest.mark.parametrize("yaw", [0.0, 30.0, 45.0, 90.0, 137.0])
def test_inside_is_the_hull_at_any_yaw(yaw):
    """Centre in the prism (0.05, 0.05 — 50 mm from its nearest face) receives; centre in the empty half of its box
    (0.15, 0.15 — 71 mm from the slanted face) does not — at every yaw, with the placed points' own test (the
    articulated path) and with arrives' default."""
    V, F = _tri_prism()
    R = _rz(yaw)
    S = RS.surface_points(V, F) @ R.T
    start = np.array([[0.0, 0.0, 1.0]])
    for c, want in (((0.05, 0.05, 0.05), True), ((0.15, 0.15, 0.05), False)):
        end = np.asarray(c)[None] @ R.T
        assert XG.arrives(start, end, S) is want
        assert XG.arrives(start, end, S, inside=RS.inside_test(V @ R.T)) is want


def test_the_world_box_alone_would_take_the_empty_half():
    """The case the hull fixes: the empty-half centre is inside the prism's box at yaw 0."""
    V, _F = _tri_prism()
    c = np.array([0.15, 0.15, 0.05])
    assert np.all((c > V.min(0)) & (c < V.max(0))) and not RS.inside_test(V)(c)


def test_inside_test_matches_an_independent_hull_membership():
    """Random clouds and probes against Delaunay's simplex search (a separate hull construction)."""
    from scipy.spatial import Delaunay  # noqa: PLC0415

    rng = np.random.default_rng(3)
    for _ in range(5):
        P = rng.normal(size=(60, 3)) * rng.uniform(0.02, 0.2, 3)
        T, D = RS.inside_test(P), Delaunay(P)
        probe = rng.normal(size=(400, 3)) * P.std(0) * 1.5
        assert [T(c) for c in probe] == [bool(D.find_simplex(c) >= 0) for c in probe]


def test_a_flat_object_has_no_inside_and_a_volume_is_never_skipped():
    """Rank < 3 (a plate): every point outside, no error; a hull failure on a volume would die (not tested here —
    Qhull hulls every non-flat cloud the box screen passes)."""
    flat = np.c_[np.random.default_rng(4).uniform(-0.1, 0.1, (30, 2)), np.full(30, 0.02)]
    assert RS.hull_planes(flat) is None
    assert not RS.inside_test(flat)(np.array([0.0, 0.0, 0.02]))
    assert RS.hull_planes(np.r_[flat, [[0.0, 0.0, 0.05]]]) is not None


def test_a_face_is_sampled_no_coarser_than_the_spacing():
    S = RS.surface_points(PLATE_V, PLATE_F)
    from scipy.spatial import cKDTree

    probe = np.random.default_rng(0).uniform([-0.2, -0.1, 0.0], [0.2, 0.1, 0.0], (500, 3))
    assert cKDTree(S).query(probe)[0].max() <= RS.RECV_SURF_M


def test_obj_faces_fans_polygons_and_starts_at_one(tmp_path):
    f = tmp_path / "quad.obj"
    f.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n")
    assert RS.obj_faces(f).tolist() == [[0, 1, 2], [0, 2, 3]]


@pytest.mark.parametrize(("body", "match"), [("v 0 0 0\n", "no faces"), ("v 0 0 0\nf -1 -2 -3\n", "relative")])
def test_obj_faces_dies_on_what_it_cannot_read(tmp_path, body, match):
    f = tmp_path / f"bad_{match.split()[0]}.obj"
    f.write_text(body)
    with pytest.raises(ValueError, match=match):
        RS.obj_faces(f)


def test_receivers_measure_on_surfaces_and_hulls():
    """Structure: receivers builds near_placer and calls arrives with the surface samples and its inside test."""
    tree = ast.parse(Path(XG.__file__).read_text(encoding="utf-8"))
    f = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "receivers")
    calls = [ast.unparse(c) for c in ast.walk(f) if isinstance(c, ast.Call)]
    assert any(c.startswith("RS.near_placer(") for c in calls)
    assert any(c.startswith("arrives(") and "S0[n]" in c and "I0[n]" in c for c in calls)
    assert any(c.startswith("near.inside(") for c in calls)


def test_the_pose_fit_is_rigid_or_dies():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(40, 3))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    R2, t2 = RS._rigid(V, V @ R.T + [0.1, -0.2, 0.3])
    assert np.allclose(R2, R) and np.allclose(t2, [0.1, -0.2, 0.3])
    with pytest.raises(ValueError, match="not a rigid copy"):
        RS._rigid(V, 1.01 * V)


def test_the_oracle_reads_the_hull_of_its_own_samples(tmp_path):
    """The scene oracle (readers.raw.scene_classify) builds its own hull: the prism's empty half is outside, its middle
    inside; a flat mesh has none."""
    from dexengine.data.readers.raw import scene_classify as CL  # noqa: PLC0415

    V, F = _tri_prism()
    prism, plate = tmp_path / "prism.obj", tmp_path / "plate.obj"
    for f, (v, fs) in ((prism, (V, F)), (plate, (PLATE_V, PLATE_F))):
        lines = [f"v {a} {b} {c}" for a, b, c in v] + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in fs]
        f.write_text("\n".join(lines) + "\n")
    H = CL.hull(str(prism))
    inside = [bool(np.all(H[:, :3] @ c + H[:, 3] < 0)) for c in ([0.05, 0.05, 0.05], [0.15, 0.15, 0.05])]
    assert inside == [True, False] and CL.hull(str(plate)) is None
