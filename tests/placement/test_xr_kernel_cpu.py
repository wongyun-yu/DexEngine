"""The cross-repair GJK/EPA kernel (xr_kernels._kernel) gives the stored bits on Warp's CPU device.

Capsule and convex-mesh geoms of a small MuJoCo model are queried for every type pair the filter specialises
(capsule-capsule, capsule-mesh, mesh-capsule, mesh-mesh), each with pairs apart within the cutoff, beyond it, and
penetrating. `python tests/placement/test_xr_kernel_cpu.py --write` rewrites the golden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).with_name("golden") / "xr_kernel_cpu.npz"
KINDS = ((3, 3), (3, 7), (7, 3), (7, 7))
EPA_V = ("ev", "ev1", "ev2", "evi1", "evi2")
EPA_F = ("ef", "epr", "en2", "eix", "emap")
CUTOFF = 0.05
XML = """<mujoco><asset>
<mesh name="box" vertex="-0.03 -0.02 -0.01 0.03 -0.02 -0.01 -0.03 0.02 -0.01 0.03 0.02 -0.01
                         -0.03 -0.02 0.01 0.03 -0.02 0.01 -0.03 0.02 0.01 0.03 0.02 0.01"/>
<mesh name="tet" vertex="0 0 0 0.04 0 0 0 0.04 0 0 0 0.04"/>
<mesh name="ico" vertex="0 0.0162 0.0262 0 -0.0162 0.0262 0 0.0162 -0.0262 0 -0.0162 -0.0262
                         0.0162 0.0262 0 -0.0162 0.0262 0 0.0162 -0.0262 0 -0.0162 -0.0262 0
                         0.0262 0 0.0162 -0.0262 0 0.0162 0.0262 0 -0.0162 -0.0262 0 -0.0162"/>
</asset><worldbody>
<geom name="c0" type="capsule" size="0.01 0.03"/>
<geom name="c1" type="capsule" size="0.015 0.02"/>
<geom name="m0" type="mesh" mesh="box"/>
<geom name="m1" type="mesh" mesh="tet"/>
<geom name="m2" type="mesh" mesh="ico"/>
</worldbody></mujoco>"""
PAIRS = {(3, 3): [(0, 1)], (3, 7): [(0, 2), (1, 4)], (7, 3): [(3, 1), (4, 0)], (7, 7): [(2, 3), (4, 2), (3, 4)]}


def _rot(rng):
    import mujoco

    q = rng.normal(size=4)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q / np.linalg.norm(q))
    return out.reshape(3, 3)


def _queries(rng, pairs, n):
    """n queries over the pairs: centre distance 0..14 cm along a random direction (apart, within the cutoff,
    beyond it, and overlapping), geom1 nudged along x as the filter does."""
    g = np.array([pairs[i % len(pairs)] for i in range(n)])
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return {
        "g1": g[:, 0],
        "g2": g[:, 1],
        "dx": rng.uniform(-0.002, 0.002, n),
        "p2": dirs * np.linspace(0.0, 0.14, n)[:, None],
        "r1": np.stack([_rot(rng) for _ in range(n)]),
        "r2": np.stack([_rot(rng) for _ in range(n)]),
    }


def _run() -> dict:
    import mujoco

    from dexengine.placement.repair import xr_kernels as XK

    W = XK._warp()
    wp = W["wp"]
    m = mujoco.MjModel.from_xml_string(XML)
    rng = np.random.default_rng(5)
    d, f32, n = "cpu", np.float32, 24
    it, tol = int(m.opt.ccd_iterations), float(m.opt.ccd_tolerance)

    def a(x, dt):
        return wp.array(np.ascontiguousarray(x), dtype=dt, device=d)

    model = XK.struct_of(
        "model",
        {
            "geom_type": a(m.geom_type.astype(np.int32), int),
            "geom_dataid": a(m.geom_dataid.astype(np.int32), int),
            "geom_size": a(m.geom_size.astype(f32), wp.vec3),
            "mesh_vertadr": a(m.mesh_vertadr.astype(np.int32), int),
            "mesh_vertnum": a(m.mesh_vertnum.astype(np.int32), int),
            "mesh_graphadr": a(m.mesh_graphadr.astype(np.int32), int),
            "mesh_vert": a(m.mesh_vert.astype(f32), wp.vec3),
            "mesh_graph": a(m.mesh_graph.astype(np.int32), int),
        },
    )
    out = {}
    for kind in KINDS:
        q = _queries(rng, PAIRS[kind], n)
        nv, nf = 5 + it, 6 + W["NF"] * it
        buf = {k: wp.zeros((n, nv), dtype=t, device=d) for k, t in zip(EPA_V, (wp.vec3,) * 3 + (int,) * 2, strict=True)}
        buf |= {
            k: wp.zeros((n, nf), dtype=t, device=d)
            for k, t in zip(EPA_F, (wp.vec3i, wp.vec3, float, int, int), strict=True)
        }
        buf["ehor"] = wp.zeros((n, 2 * W["NH"]), dtype=int, device=d)
        res = [wp.zeros(n, dtype=float, device=d), wp.zeros(n, dtype=wp.vec3, device=d)]
        res.append(wp.zeros(n, dtype=wp.vec3, device=d))
        query = {
            "q_g1": a(q["g1"].astype(np.int32), int),
            "q_g2": a(q["g2"].astype(np.int32), int),
            "q_dx": a(q["dx"].astype(f32), float),
            "q_p2": a(q["p2"].astype(f32), wp.vec3),
            "q_r1": a(q["r1"].astype(f32), wp.mat33),
            "q_r2": a(q["r2"].astype(f32), wp.mat33),
        }
        args = [model, XK.struct_of("query", query), tol, CUTOFF, it, XK.struct_of("epa", buf)]
        args.append(XK.struct_of("out", dict(zip(("out_d", "out_w1", "out_w2"), res, strict=True))))
        wp.launch(XK._kernel(*kind), dim=n, device=d, inputs=args)
        for name, r in zip(("d", "w1", "w2"), res, strict=True):
            out[f"{kind[0]}_{kind[1]}_{name}"] = r.numpy()
    return out


@pytest.fixture(scope="module")
def cpu_warp(tmp_path_factory):
    wp = pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp_cache"))
    return wp


def test_ccd_kernel_matches_the_golden_bits(cpu_warp):
    got = _run()
    want = np.load(GOLDEN)
    assert sorted(got) == sorted(want.files)
    for k in want.files:
        assert got[k].dtype == want[k].dtype and got[k].shape == want[k].shape, k
        assert got[k].tobytes() == want[k].tobytes(), k


def test_the_queries_reach_every_outcome():
    """Every type pair has pairs beyond the cutoff (1e30), apart within it, and overlapping."""
    want = np.load(GOLDEN)
    for k1, k2 in KINDS:
        d = want[f"{k1}_{k2}_d"]
        assert (d > 1e29).any() and ((d > 0) & (d < CUTOFF)).any() and (d < 0).any(), (k1, k2)


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez(GOLDEN, **_run())
    print(f"wrote {GOLDEN}")
