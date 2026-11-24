"""The bake's host-independent arm IK (reach_map.det_ik.DetArmIK).

- the Jacobian (linear and angular rows) equals pinocchio's LOCAL_WORLD_ALIGNED frame Jacobian;
- one DLS pass equals a float64 reference written with pinocchio and numpy.linalg (to rounding);
- the bits of a fixed DLS pass and of the arm-flip seed draw (torch CPU generator, float64) are the golden ones — the
  same on every x86-64 host (measured on three CPU families, see GOLDEN);
- the CPU bake selects it (coarse_path.dp._arm_ik), and the nullspace term it does not implement dies.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
torch = pytest.importorskip("torch")

# a 7-joint arm: axis-aligned and unaligned revolute joints, offsets and rotated origins (every joint kind of the
# rby1 combo URDFs: RX · RY · RZ · RevoluteUnaligned)
_J = [
    ("0.0 0.0 0.10", "0 0 0", "0 0 1"),
    ("0.0 0.05 0.12", "0.1 0 0", "0 1 0"),
    ("0.0 0.0 0.25", "0 0.2 0", "1 0 0"),
    ("0.02 0.0 0.22", "0 0 0.3", "0 0.6 0.8"),
    ("0.0 0.0 0.20", "-0.2 0 0", "0 1 0"),
    ("0.0 0.03 0.15", "0 0 -0.4", "0.48 0.6 0.64"),
    ("0.0 0.0 0.08", "0.3 0.1 0", "0 0 1"),
]


def _urdf() -> str:
    links = "".join(f'<link name="l{i}"/>' for i in range(len(_J) + 1)) + '<link name="ee"/>'
    joints = "".join(
        f'<joint name="j{i + 1}" type="revolute"><parent link="l{i}"/><child link="l{i + 1}"/>'
        f'<origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="{ax}"/><limit lower="-2.5" upper="2.4" effort="1" velocity="1"/>'
        "</joint>"
        for i, (xyz, rpy, ax) in enumerate(_J)
    )
    ee = (
        f'<joint name="jee" type="fixed"><parent link="l{len(_J)}"/><child link="ee"/>'
        '<origin xyz="0.03 0 0.11" rpy="0.2 -0.1 0.4"/></joint>'
    )
    return f'<robot name="arm">{links}{joints}{ee}</robot>'


NAMES = [f"j{i + 1}" for i in range(len(_J))]


def _ik(tmp_path):
    from dexengine.placement.reach_map.det_ik import DetArmIK

    f = tmp_path / "arm.urdf"
    f.write_text(_urdf())
    m = pin.buildModelFromUrdf(str(f))
    return m, DetArmIK("right", m, f, "ee", NAMES), f


def _inputs(n=96):
    """Targets and seeds from integer arithmetic only (no host-dependent function in the inputs)."""
    i = np.arange(n * 7, dtype=np.int64)
    u = ((i * 2654435761) % 1000003).astype(np.float64) / 1000003.0  # [0, 1)
    qt = (u.reshape(n, 7) - 0.5) * 3.0
    qs = qt + (((i * 40503) % 997).astype(np.float64).reshape(n, 7) / 997.0 - 0.5) * 0.4
    return qt, qs


def test_jacobian_matches_pinocchio(tmp_path):
    m, ik, _ = _ik(tmp_path)
    d = m.createData()
    fid = m.getFrameId("ee")
    qt, _ = _inputs(24)
    T = ik.fk(torch.tensor(qt)).numpy()
    J = ik.jac(torch.tensor(qt)).numpy()
    for b in range(len(qt)):
        pin.computeJointJacobians(m, d, qt[b])
        pin.updateFramePlacements(m, d)
        assert np.abs(d.oMf[fid].homogeneous - T[b]).max() < 1e-12
        Jp = pin.getFrameJacobian(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        assert np.abs(Jp - J[b]).max() < 1e-12


def test_the_arm_chain_pass_has_detfk_bits(tmp_path):
    """The DLS loop's one-pass pose and Jacobian (fixed prefix + arm chain) = DetFK on the full configuration, bit for
    bit — on an arm whose chain starts at the universe and on one with a fixed prefix joint."""
    _, ik, _ = _ik(tmp_path)
    qt, _ = _inputs(16)
    R, p, J = ik.pose_jac(qt)
    R2, p2 = ik.F.frames(ik._full(qt), [ik.fid])
    assert np.array_equal(R, R2[:, 0]) and np.array_equal(p, p2[:, 0])
    assert np.array_equal(J, ik.F.frame_jac6(ik._full(qt), ik.fid, ik.jids))
    from dexengine.placement.reach_map.det_ik import DetArmIK

    f = tmp_path / "arm_prefix.urdf"
    f.write_text(_urdf())
    m = pin.buildModelFromUrdf(str(f))
    ik2 = DetArmIK("right", m, f, "ee", NAMES[1:])  # j1 off the arm: a fixed prefix at neutral
    R, p, J = ik2.pose_jac(qt[:, 1:])
    full = ik2._full(qt[:, 1:])
    R2, p2 = ik2.F.frames(full, [ik2.fid])
    assert np.array_equal(R, R2[:, 0]) and np.array_equal(p, p2[:, 0])
    assert np.array_equal(J, ik2.F.frame_jac6(full, ik2.fid, ik2.jids))


def test_dls_equals_a_float64_reference(tmp_path):
    """One pass of 6 iterations, compared with the same DLS written with pinocchio and numpy.linalg (float64)."""
    m, ik, _ = _ik(tmp_path)
    d = m.createData()
    fid = m.getFrameId("ee")
    qt, qs = _inputs(32)
    Tt = ik.fk(torch.tensor(qt))
    q, pe, red, mg = (v.numpy() for v in ik._dls(Tt, 6, 0.01, 0.0, torch.tensor(qs)))
    lo, hi = ik._lo, ik._hi
    for b in range(len(qt)):
        qb = qs[b].copy()
        Tb = Tt[b].numpy()
        for _ in range(6):
            pin.computeJointJacobians(m, d, qb)
            pin.updateFramePlacements(m, d)
            M = d.oMf[fid]
            e6 = np.concatenate([Tb[:3, 3] - M.translation, pin.log3(Tb[:3, :3] @ M.rotation.T)])
            J = pin.getFrameJacobian(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            qb = np.clip(qb + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e6), lo, hi)
        assert np.abs(qb - q[b]).max() < 1e-9
    assert np.isfinite(pe).all() and np.isfinite(red).all() and np.isfinite(mg).all()


def _digests(tmp_path) -> dict:
    _, ik, _ = _ik(tmp_path)
    qt, qs = _inputs()
    Tt = ik.fk(torch.tensor(qt))
    out = ik._dls(Tt, 8, 0.01, 0.0, torch.tensor(qs))
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    rq = ik.lo + (ik.hi - ik.lo) * torch.rand((64, ik.n), dtype=torch.float64, device="cpu", generator=g)
    h = lambda *xs: hashlib.sha256(b"".join(np.ascontiguousarray(x.numpy()).tobytes() for x in xs)).hexdigest()  # noqa: E731
    return {"fk": h(Tt), "dls": h(*out), "flip_seeds": h(rq)}


# sha256 on _inputs() — the same on every x86-64 host (measured on Intel Core Ultra, Intel Xeon and AMD EPYC CPUs)
# with the pinned numpy, scipy and torch (the retarget_torch fixture skips otherwise)
GOLDEN = {
    "fk": "fc73cc92d90750177eeddaf1c74cadf8dc5c55be6de1307826c8b7406564b808",
    "dls": "9f3896ec3d358324e224859dd8508142be9a4efc3f643cd32b69940d418d25fe",
    "flip_seeds": "57020271f3f40cd948416be0fcbc869eebcaf652aad4e2e13f6f1b9e1cd4e25b",
}


def test_bits_are_the_golden_ones(tmp_path, retarget_torch):
    got = _digests(tmp_path)
    for k, want in GOLDEN.items():
        assert want is not None, f"golden {k} not set — `python tests/placement/test_det_ik.py --print`"
        assert got[k] == want, (k, got[k])


def test_nullspace_term_dies(tmp_path):
    _, ik, _ = _ik(tmp_path)
    qt, qs = _inputs(4)
    with pytest.raises(ValueError, match="null_alpha"):
        ik._dls(ik.fk(torch.tensor(qt)), 2, 0.01, 0.1, torch.tensor(qs))


def test_the_cpu_bake_selects_det_ik(tmp_path, monkeypatch, capsys):
    import types

    from dexengine.placement.coarse_path import dp
    from dexengine.placement.reach_map import batch_ik as OPG
    from dexengine.placement.reach_map.det_ik import DetArmIK

    _, _, f = _ik(tmp_path)
    monkeypatch.setattr(OPG, "DEV", "cpu")
    monkeypatch.setattr(dp, "_IK_MEMO", {})
    a = types.SimpleNamespace(side="right", sequence="s")
    _model, ik, DT = dp._arm_ik(f, a, "ee", NAMES, OPG)
    assert isinstance(ik, DetArmIK) and torch.float64 == DT
    assert "[IK_ENGINE] s/right: det-f64" in capsys.readouterr().out


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    if "--print" in sys.argv:
        with tempfile.TemporaryDirectory() as t:
            for k, v in _digests(Path(t)).items():
                print(f'    "{k}": "{v}",')
