"""The cross repair gives the same bits on every host and GPU (the measured causes are in det_math's docstring).

- det_math: fixed bits on fixed inputs (the golden digests below are host-independent by construction — a change of
  arithmetic order changes them) and agreement with numpy within a few ulps.
- DetFK agrees with pinocchio (revolute RX · RY · RZ · unaligned) to 1e-12, Jacobian included.
- GpuDist.query drops only GJK's separating-plane certificate: with the GPU kernel replaced by adversarial outputs
  (an EPA failure, a far estimate of a near pair) the rows equal the CPU-only rows.
- Structure: the decision-path functions contain no BLAS / dispatched-math call (``@``, np.linalg, np.einsum, np.dot,
  np.sin/np.cos/np.arccos …) and no pinocchio FK call — checked on the syntax tree, with a planted violation caught.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

import numpy as np
import pytest

from dexengine.placement.collision import det_math as D


def _h(*xs) -> str:
    return hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(x, np.float64)).tobytes() for x in xs)).hexdigest()


def _inputs():
    i = np.arange(3000, dtype=np.int64)
    u = ((i * 2654435761) % 1000003).astype(np.float64) / 1000003.0 - 0.5  # integer arithmetic + one division
    return u


# sha256 of the outputs on _inputs() — the same on every x86-64 host (measured on Intel Core Ultra, AMD EPYC and
# Intel Xeon CPUs)
GOLDEN = {
    "sincos": "fa15f7c4b8962bce103528f2e720648b8df58c321d585f2b71d2111560714bad",
    "linalg": "e2be7fed00c25a09a318b6d5c38a862979b889b88b4b3c42cd2f86c08a552e63",
    "acos": "3bc51b8a3c513399e5a5c80cb5577615ae554e7ac6ab279de8e4df29a361927b",
    "frames": "b7b0f25170fc68edf38b19a7cf6fbb26f12aff35bf063e91de794ed65afdb0b1",
}


def _digests() -> dict:
    u = _inputs()
    x = u * 25.0
    s, c = D.sincos(x)
    A = u[:2700].reshape(300, 3, 3)
    v = u[:900].reshape(300, 3)
    spd = D.matmul3(A, np.swapaxes(A, -1, -2)) + 0.1 * np.eye(3)
    lin = (D.dot3(v, v[::-1]), D.norm(v), D.cross3(v, v[::-1]), D.matvec3(A, v), D.matTvec3(A, v), D.matmul3(A, A))
    return {
        "sincos": _h(s, c),
        "linalg": _h(
            *lin, D.solve3(spd, v), D.trace_AtB(A, A[::-1]), D.rot_axis(np.array([0.6, 0.0, 0.8]), s[:5], c[:5])
        ),
        "acos": _h(D.acos_deg(np.clip(u * 2.0, -1, 1))),
        "frames": _h(*_frames(u)),
    }


def _frames(u):
    """rpy rotations, 4×4 compositions and row means — the collider geometry's transforms (sim_collider · torso_split ·
    wrist_hull)."""
    rpy = (u[:300].reshape(100, 3) * 6.0).tolist()
    R = [D.rpy_matrix(r) for r in rpy]
    T = []
    for i, r in enumerate(R):
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = r, u[300 + 3 * i : 303 + 3 * i]
        T.append(t)
    H = [D.hom(T[i], T[i + 1]) for i in range(len(T) - 1)]
    return (*R, *H, D.mean_rows(u[:900].reshape(300, 3)))


def test_det_math_bits_are_the_golden_ones():
    got = _digests()
    for k, want in GOLDEN.items():
        assert want is not None, f"golden {k} not set — `python tests/placement/test_xr_determinism.py --print`"
        assert got[k] == want, k


def test_det_math_agrees_with_numpy():
    x = np.linspace(-12.0, 12.0, 200001)
    s, c = D.sincos(x)
    assert np.abs(s - np.sin(x)).max() < 4e-15 and np.abs(c - np.cos(x)).max() < 4e-15
    cc = np.linspace(-1.0, 1.0, 20001)
    assert np.abs(D.acos_deg(cc) - np.degrees(np.arccos(cc))).max() < 1e-12
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 3, 3))
    b = rng.normal(size=(200, 3))
    spd = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(3)
    assert np.allclose(D.solve3(spd, b), np.linalg.solve(spd, b[..., None])[..., 0], rtol=1e-12, atol=1e-12)
    assert np.allclose(D.matmul3(A, A[::-1]), A @ A[::-1], rtol=0, atol=1e-14)
    assert np.allclose(D.norm(b), np.linalg.norm(b, axis=1), rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match="DET_MATH"):
        D.sincos(np.array([1e7]))


URDF = """<robot name="t">
<link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>
<joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0.1 0 0.2" rpy="0.1 0.2 0.3"/>
<axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
<joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0.3 0" rpy="0 0 0.5"/>
<axis xyz="0 0.9396926 0.3420201"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
<joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.2 0 0" rpy="0.3 0 0"/>
<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
<joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.15" rpy="0 0.4 0"/>
<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
</robot>"""


def test_det_fk_matches_pinocchio():
    pin = pytest.importorskip("pinocchio")
    from dexengine.placement.collision.det_fk import DetFK

    m = pin.buildModelFromXML(URDF)
    kinds = {m.joints[i].shortname() for i in range(1, m.njoints)}
    assert {"JointModelRX", "JointModelRY", "JointModelRZ", "JointModelRevoluteUnaligned"} <= kinds
    d = m.createData()
    F = DetFK(m)
    rng = np.random.default_rng(1)
    Q = rng.uniform(-3, 3, size=(40, m.nq))
    R, p = F.frames(Q, range(m.nframes))
    fid = m.nframes - 1
    jids = list(range(1, m.njoints))
    Jd = F.frame_jac_lin(Q, fid, jids)
    for b in range(len(Q)):
        pin.framesForwardKinematics(m, d, Q[b])
        for f in range(m.nframes):
            assert np.abs(d.oMf[f].rotation - R[b, f]).max() < 1e-12
            assert np.abs(d.oMf[f].translation - p[b, f]).max() < 1e-12
        pin.computeJointJacobians(m, d, Q[b])
        pin.updateFramePlacements(m, d)
        J = pin.getFrameJacobian(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        assert np.abs(J[:, [m.joints[j].idx_v for j in jids]] - Jd[b]).max() < 1e-12


# ── the GPU filter drops only the certificate ─────────────────────────────────────────────────────────────────────
XML = """<mujoco><asset>
<mesh name="box" vertex="-0.03 -0.02 -0.01 0.03 -0.02 -0.01 -0.03 0.02 -0.01 0.03 0.02 -0.01
                         -0.03 -0.02 0.01 0.03 -0.02 0.01 -0.03 0.02 0.01 0.03 0.02 0.01"/>
<mesh name="tet" vertex="0 0 0 0.04 0 0 0 0.04 0 0 0 0.04"/>
</asset><worldbody>
<body name="a" mocap="true"><geom name="g0" type="mesh" mesh="box"/></body>
<body name="b" mocap="true"><geom name="g1" type="mesh" mesh="tet"/></body>
<body name="c" mocap="true"><geom name="g2" type="capsule" size="0.01 0.03"/></body>
</worldbody></mujoco>"""


class _FakeModel:
    """The parts of CrossSetModel GpuDist reads: m · d · pairs · _fromto · _requery_zero."""

    def __init__(self):
        import mujoco

        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)
        self.pairs = [(0, 1, "1_cross", "hand-hand"), (0, 2, "1_cross", "hand-hand"), (1, 2, "2_hand_obj", "hand-obj")]
        self._fromto = np.zeros(6)

    def _requery_zero(self, gi, gj, distmax):
        return 0.0


def _poses(M, B=96, seed=3):
    rng = np.random.default_rng(seed)
    mpos = np.zeros((B, M.m.nmocap, 3))
    mquat = np.zeros((B, M.m.nmocap, 4))
    mquat[..., 0] = 1.0
    mpos[:, 1] = rng.uniform(-0.07, 0.07, (B, 3))  # near, apart and penetrating
    mpos[:, 2] = rng.uniform(-0.07, 0.07, (B, 3))
    gx = mpos.copy()  # one geom per body, at the body origin
    gR = np.broadcast_to(np.eye(3), (B, 3, 3, 3)).copy()
    return gx, gR, (mpos, mquat)


def _same(r1, r2):
    m1, h1, a = r1
    m2, h2, b = r2
    assert np.array_equal(m1, m2) and np.array_equal(h1, h2)
    for k in ("qb", "qk", "d", "ft", "worst_v", "worst_i"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_query_rows_equal_cpu_only_under_adversarial_gpu(monkeypatch):
    """Whatever the GPU returns besides a truthful certificate, the rows are the CPU-only rows."""
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    from dexengine.placement.repair import xr_gpu as XG

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    M = _FakeModel()
    G = XG.GpuDist(M)
    gx, gR, moc = _poses(M)
    dm, ws = 0.03, ("1_cross", "2_hand_obj")
    full = {}
    o_cpu = G._cpu

    def rec(qb, g1, g2, sel, distmax, robust_zero, mocap):
        out = o_cpu(qb, g1, g2, sel, distmax, robust_zero, mocap)
        full["d"] = out[0]
        return out

    monkeypatch.setattr(G, "_cpu", rec)
    cpu = G.query(gx, gR, dm, moc, worst_sets=ws)
    dfull = full["d"]
    assert (dfull < 0).any() and ((dfull > 0) & (dfull < dm)).any() and (dfull >= dm).any()
    monkeypatch.setattr(G, "_cpu", o_cpu)
    far = XG._warp()["FAR"]
    G.use_gpu = True

    def garbage(qg1, qg2, qdx, p2, r1, r2, cutoff):  # no certificate: EPA failures, huge / negative estimates
        assert cutoff == dm + XG.EXACT_MARGIN
        n = len(qg1)
        i = np.arange(n) % 4
        d = np.select([i == 0, i == 1, i == 2], [far, 1e6, -0.5], 0.0)
        return d, np.zeros((n, 3)), np.zeros((n, 3)), np.where(i == 0, 0, 1)

    monkeypatch.setattr(G, "_launch", garbage)
    _same(cpu, G.query(gx, gR, dm, moc, worst_sets=ws))

    def certify_far(qg1, qg2, qdx, p2, r1, r2, cutoff):  # certificate exactly where the CPU says distmax
        n = len(qg1)
        cert = dfull >= dm
        return np.where(cert, far, 5.0), np.zeros((n, 3)), np.zeros((n, 3)), np.ones(n, int)

    monkeypatch.setattr(G, "_launch", certify_far)
    _same(cpu, G.query(gx, gR, dm, moc, worst_sets=ws))


def test_query_drop_rule_keeps_epa_failures_and_estimates(monkeypatch):
    """The selection itself: (FLOAT_MAX, 1) is dropped; (FLOAT_MAX, 0) and any finite value — however large — are
    measured on the CPU."""
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    from dexengine.placement.repair import xr_gpu as XG

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    M = _FakeModel()
    G = XG.GpuDist(M)
    far = XG._warp()["FAR"]
    seen = {}

    def launch(qg1, qg2, qdx, p2, r1, r2, cutoff):
        n = len(qg1)
        assert n == 3
        return np.array([far, far, 5.0]), np.zeros((n, 3)), np.zeros((n, 3)), np.array([1, 0, 1])

    def cpu(qb, g1, g2, sel, distmax, robust_zero, mocap):
        seen["sel"] = np.asarray(sel).tolist()
        n = len(qb)
        return np.full(n, distmax), np.zeros((n, 3)), np.zeros((n, 3))

    G.use_gpu = True
    monkeypatch.setattr(G, "_launch", launch)
    monkeypatch.setattr(G, "_cpu", cpu)
    gx = np.zeros((1, 3, 3))  # every pair inside its bounding spheres → 3 queries
    gR = np.broadcast_to(np.eye(3), (1, 3, 3, 3)).copy()
    G.query(gx, gR, 0.03, (np.zeros((1, 3, 3)), np.zeros((1, 3, 4))))
    assert seen["sel"] == [1, 2]  # 0 = certificate (dropped) · 1 = EPA failed · 2 = a 5 m estimate


def test_collider_rpy_is_det_math():
    """wrist_hull.rpy_matrix (every collider frame: sim_collider.origin_matrix · visual meshes) is det_math's."""
    from dexengine.robots import wrist_hull as WG

    for r in ([0.1, -2.0, 3.0], ["1.5707963", "0", "-0.2618"], [0, 0, 0]):
        assert np.array_equal(WG.rpy_matrix(r), D.rpy_matrix(r))
    assert np.array_equal(D.rpy_matrix([0, 0, 0]), np.eye(3))


# ── structure: no BLAS / dispatched math / pinocchio FK in the decision path ────────────────────────────────────────
BANNED_ATTR = {
    ("np", "einsum"),
    ("np", "dot"),
    ("np", "matmul"),
    ("np", "inner"),
    ("np", "tensordot"),
    ("np", "sin"),
    ("np", "cos"),
    ("np", "tan"),
    ("np", "arccos"),
    ("np", "arcsin"),
    ("np", "arctan2"),
    ("np", "exp"),
    ("np", "log"),
    ("np", "degrees"),
    ("np", "trace"),
}
BANNED_PIN = {"framesForwardKinematics", "forwardKinematics", "computeJointJacobians", "getFrameJacobian"}


def _violations(src: str) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            out.append(f"@ at line {node.lineno}")
        if isinstance(node, ast.Attribute):
            v = node.value
            np_linalg = isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "np"
            if np_linalg and v.attr == "linalg":
                out.append(f"np.linalg.{node.attr} at line {node.lineno}")
            if isinstance(v, ast.Name) and (v.id, node.attr) in BANNED_ATTR:
                out.append(f"{v.id}.{node.attr} at line {node.lineno}")
            if node.attr in BANNED_PIN:
                out.append(f"pinocchio {node.attr} at line {node.lineno}")
    return out


def _decision_path():
    from dexengine.placement.collision import cross_sets as CS
    from dexengine.placement.collision import det_fk
    from dexengine.placement.reach_map import det_ik
    from dexengine.placement.repair import cross_steps, cross_util, xr_gpu, xr_solver

    M = CS.CrossSetModel
    return [
        ("det_math", inspect.getsource(D)),
        ("det_fk", inspect.getsource(det_fk)),
        ("xr_solver", inspect.getsource(xr_solver)),
        ("xr_gpu.GpuDist.query", inspect.getsource(xr_gpu.GpuDist.query)),
        ("xr_gpu.GpuDist._cpu", inspect.getsource(xr_gpu.GpuDist._cpu)),
        ("xr_gpu.PoseBatch", inspect.getsource(xr_gpu.PoseBatch)),
        ("cross_steps", inspect.getsource(cross_steps)),
        ("cross_util._palm_tracking_error", inspect.getsource(cross_util._palm_tracking_error)),
        ("cross_util.obj_rel_shift_mm", inspect.getsource(cross_util.obj_rel_shift_mm)),
        *(
            (f"CrossSetModel.{f}", inspect.getsource(getattr(M, f)))
            for f in ("build_q", "hand_frame_pos", "arm_dq", "mocap_pose", "mocap_from_links")
        ),
        ("det_ik (the CPU bake's arm IK)", inspect.getsource(det_ik)),
    ]


def _collider_geometry():
    from dexengine.placement.collision import sim_collider as SC
    from dexengine.placement.collision import torso_split as TS
    from dexengine.robots import wrist_hull as WG

    return [
        *(
            (f"wrist_hull.{f}", inspect.getsource(getattr(WG, f)))
            for f in ("rpy_matrix", "visual_mesh", "_outside", "_reduce_hull", "hull_triangles", "capsule_outside_mm")
        ),
        *(
            (f"sim_collider.{f}", inspect.getsource(getattr(SC, f)))
            for f in ("origin_matrix", "fixed_frame", "shape_dicts", "vendor_capsules")
        ),
        ("sim_collider.Capsule", inspect.getsource(SC.Capsule)),
        ("torso_split", inspect.getsource(TS)),
    ]


def test_collider_geometry_has_no_host_dependent_call():
    """The torso decomposition, its cache key and the arm / torso hulls are host-independent (the cut once differed
    between OpenBLAS kernels: key 57a2b9 · 11d166 on one CPU)."""
    bad = {name: v for name, src in _collider_geometry() if (v := _violations(src))}
    assert not bad, bad


def test_decision_path_has_no_host_dependent_call():
    bad = {name: v for name, src in _decision_path() if (v := _violations(src))}
    assert not bad, bad


def test_the_structure_check_catches_a_planted_violation():
    planted = "def f(a, b, pin, m, d, q):\n    x = a @ b\n    y = np.linalg.norm(a)\n    z = np.cos(a)\n"
    planted += "    pin.framesForwardKinematics(m, d, q)\n    return x, y, z\n"
    v = _violations(planted)
    assert len(v) == 4, v
    assert not _violations("def f(a, b):\n    return D.matvec3(a, b)\n")


if __name__ == "__main__":
    import sys

    if "--print" in sys.argv:
        for k, v in _digests().items():
            print(f'    "{k}": "{v}",')
