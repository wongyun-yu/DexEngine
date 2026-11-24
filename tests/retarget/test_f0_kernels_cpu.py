"""The f0 point and nearest-sample kernels (pts_kernel, nn_kernel) give the stored bits on Warp's CPU device.

Synthetic inputs exercise every branch: two objects (sample, kd-tree and hull offsets), points below the desk, points
outside the AABB and the 26-DOP, nearest-sample ties (duplicate samples with opposite normals, so the lowest-index
rule decides the inside verdict), and points inside and outside the hulls on both sides of the clearance and
penetration margins. `python tests/retarget/test_f0_kernels_cpu.py --write` rewrites the golden. GPU_FUSE_FP is off,
so the IEEE operations run in written order on every device; this test proves the CPU result only.
The inputs use numpy matmuls (BLAS), so the golden holds under the f0 worker environment (shadow_gpu.worker_env: one
BLAS thread, the pinned OpenBLAS kernel). OpenBLAS picks its kernel when it loads, so the bits are computed in a child
process started with that environment, as the f0 workers are; where the pinned kernel does not load, the test is
skipped with the reason.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).with_name("golden") / "f0_kernels_cpu.npz"
DOP_DIRS = np.array(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float
)
DOP_DIRS /= np.linalg.norm(DOP_DIRS, axis=1, keepdims=True)
TIES = 5


def _object(rng, radii, n):
    """An ellipsoid point cloud as a GpuObjectSpec: samples, outward normals, hull planes, AABB, DOP, kd-tree. The
    first TIES samples repeat at the end with inward normals."""
    from scipy.spatial import ConvexHull

    from dexengine.retarget.f0.gpu_specs import GpuObjectSpec

    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    o = GpuObjectSpec.__new__(GpuObjectSpec)
    nrm = u / np.asarray(radii)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    o.s_pts = np.ascontiguousarray(np.vstack([u * radii, (u * radii)[:TIES]]))
    o.s_nrm = np.ascontiguousarray(np.vstack([nrm, -nrm[:TIES]]))
    o.hull_eq = ConvexHull(o.s_pts).equations
    o.lo, o.hi = o.s_pts.min(0), o.s_pts.max(0)
    o.dop_D = DOP_DIRS
    o.dop_hi = (o.s_pts @ DOP_DIRS.T).max(0)
    o._build_kd()
    return o


def _rot(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _inputs():
    rng = np.random.default_rng(7)
    objs = [_object(rng, (0.05, 0.03, 0.02), 300), _object(rng, (0.02, 0.02, 0.04), 200)]
    n_links, n_pts = 3, 40
    link_pts = [rng.normal(scale=0.03, size=(n_pts, 3)) for _ in range(n_links)]
    n_prob, n_cand = 4, 24
    prob = {
        "obj": np.array([0, 1, 0, 1], np.int32),
        "opos": rng.normal(scale=0.01, size=(n_prob, 3)),
        "oR": np.stack([_rot(rng) for _ in range(n_prob)]),
        "floor": np.array([-0.02, -0.01, -0.03, -0.015]),
        "clr": np.array([0.002, 0.004, 0.003, 0.001]),
        "pm": np.array([0.001, 0.0005, 0.002, 0.001]),
        "g": np.full(n_prob, 1e-5, np.float32),
    }
    cand = {
        "prob": rng.integers(0, n_prob, n_cand).astype(np.int32),
        "LR": np.stack([[_rot(rng) for _ in range(n_links)] for _ in range(n_cand)]),
        "Lt": rng.normal(scale=0.03, size=(n_cand, n_links, 3)),
    }
    return objs, link_pts, prob, cand


def _point_arrays(a, wp, objs, link_pts, prob) -> dict:
    """The evaluator's device arrays that pts_kernel reads (names as in gpu_upload), for one hand and the objects."""
    pts = np.concatenate(link_pts)
    pt_a = np.cumsum([0] + [len(p) for p in link_pts[:-1]]).astype(np.int32)
    return {
        "h_l_off": a([0], wp.int32),
        "h_nl": a([len(link_pts)], wp.int32),
        "h_p_off": a([0], wp.int32),
        "pt_a": a(pt_a, wp.int32),
        "pt_b": a(pt_a + np.array([len(p) for p in link_pts], np.int32), wp.int32),
        "pts": a(pts, wp.vec3d),
        "pts32": a(pts.astype(np.float32), wp.vec3f),
        "p_obj": a(prob["obj"], wp.int32),
        "p_opos": a(prob["opos"], wp.vec3d),
        "p_oR": a(prob["oR"], wp.mat33d),
        "p_opos32": a(prob["opos"].astype(np.float32), wp.vec3f),
        "p_oR32": a(prob["oR"].astype(np.float32), wp.mat33f),
        "p_floor": a(prob["floor"], wp.float64),
        "p_clr": a(prob["clr"], wp.float64),
        "p_g": a(prob["g"], wp.float32),
        "o_lo": a([o.lo for o in objs], wp.vec3d),
        "o_hi": a([o.hi for o in objs], wp.vec3d),
        "o_lo32": a(np.array([o.lo for o in objs], np.float32), wp.vec3f),
        "o_hi32": a(np.array([o.hi for o in objs], np.float32), wp.vec3f),
        "dop_D32": a(np.concatenate([o.dop_D for o in objs]).astype(np.float32), wp.vec3f),
        "dop_hi32": a(np.concatenate([o.dop_hi for o in objs]).astype(np.float32), wp.float32),
    }


def _point_pass(a, wp, K, g, cand, L, npts) -> dict:
    """Both passes of pts_kernel (count, then write at the scanned offsets)."""
    d = "cpu"
    n = len(cand["prob"])
    out = {k: wp.zeros(n * L, dtype=t, device=d) for k, t in (("dsum", wp.float64), ("dcnt", wp.int32))}
    out |= {k: wp.zeros(n * L, dtype=wp.int32, device=d) for k in ("qcnt", "qoff")}
    mid = [
        a(cand["prob"], wp.int32),
        a(np.zeros(n, np.int32), wp.int32),
        a(cand["LR"], wp.mat33d),
        a(cand["Lt"], wp.vec3d),
        a(cand["LR"].astype(np.float32), wp.mat33f),
        a(cand["Lt"].astype(np.float32), wp.vec3f),
        *K.point_pass_structs(g),
        out["dsum"],
        out["dcnt"],
        out["qcnt"],
        out["qoff"],
    ]
    q_k = wp.zeros(n * L * npts, dtype=wp.int32, device=d)
    q_loc = wp.zeros(n * L * npts, dtype=wp.vec3d, device=d)
    wp.launch(K.pts_kernel, dim=(n, L), device=d, inputs=[0, L, *mid, q_k, q_loc])
    qcnt = out["qcnt"].numpy()
    out["qoff"].assign(np.concatenate([[0], np.cumsum(qcnt)[:-1]]).astype(np.int32))
    wp.launch(K.pts_kernel, dim=(n, L), device=d, inputs=[1, L, *mid, q_k, q_loc])
    nq = int(qcnt.sum())
    res = {"desk_sum": out["dsum"].numpy(), "desk_cnt": out["dcnt"].numpy(), "q_cnt": qcnt}
    return {**res, "q_k": q_k.numpy()[:nq], "q_loc": q_loc.numpy()[:nq]}


def _probes(objs, prob, cand):
    """Points on, inside and just outside each candidate's object surface, plus one next to a duplicated sample."""
    rng = np.random.default_rng(11)
    probe_k, probe_loc = [], []
    for k in range(len(cand["prob"])):
        o = objs[prob["obj"][cand["prob"][k]]]
        s = o.s_pts[rng.integers(0, len(o.s_pts), 6)]
        probe_loc += [s[0], s[1] * 0.97, s[2] * 1.01, s[3] * 0.999, s[4] * 1.001, s[5] * 0.5, o.s_pts[k % TIES] * 0.995]
        probe_k += [k] * 7
    return np.array(probe_k), np.array(probe_loc)


def _nearest_pass(a, wp, K, g, objs, prob, cand, qk, ql) -> dict:
    """nn_kernel over the queries: penetration count, deep and close depths."""
    from dexengine.retarget.f0.gpu_specs import HULL_TOL

    d = "cpu"

    def cat(attr, dt):
        return a(np.concatenate([getattr(o, attr) for o in objs]), dt)

    def offsets(attr):
        return a(np.cumsum([0] + [len(getattr(o, attr)) for o in objs[:-1]]), wp.int32)

    heq = np.concatenate([o.hull_eq for o in objs])
    r = {k: wp.zeros(len(qk), dtype=t, device=d) for k, t in (("cnt", wp.int32), ("deep", wp.float64))}
    r["close"] = wp.zeros(len(qk), dtype=wp.float64, device=d)
    kd = [cat(f, wp.vec3f) for f in ("kd_lo", "kd_hi")]
    kd += [cat(f, wp.int32) for f in ("kd_left", "kd_right", "kd_miss", "kd_a", "kd_b", "kd_perm")]
    inputs = [a(qk, wp.int32), a(ql, wp.vec3d), a(cand["prob"], wp.int32), g["p_obj"], g["p_clr"]]
    inputs += [a(prob["pm"], wp.float64), g["p_g"], offsets("s_pts"), a([len(o.s_pts) for o in objs], wp.int32)]
    inputs += [offsets("kd_lo"), offsets("kd_perm"), *kd]
    inputs += [a(np.concatenate([o.s_pts for o in objs]).astype(np.float32), wp.vec3f), cat("s_pts", wp.vec3d)]
    inputs += [cat("s_nrm", wp.vec3d), offsets("hull_eq"), a([len(o.hull_eq) for o in objs], wp.int32)]
    inputs += [a(heq[:, :3], wp.vec3d), a(heq[:, 3], wp.float64), a(heq[:, :3].astype(np.float32), wp.vec3f)]
    inputs += [a(heq[:, 3].astype(np.float32), wp.float32), HULL_TOL, r["cnt"], r["deep"], r["close"]]
    wp.launch(K.nn_kernel, dim=len(qk), device=d, inputs=inputs)
    return {"r_cnt": r["cnt"].numpy(), "r_deep": r["deep"].numpy(), "r_close": r["close"].numpy()}


def _run():
    import warp as wp

    from dexengine.retarget.f0 import gpu_kernels as K

    wp.config.quiet = True
    objs, link_pts, prob, cand = _inputs()

    def a(x, dt):
        return wp.array(np.ascontiguousarray(x), dtype=dt, device="cpu")

    g = _point_arrays(a, wp, objs, link_pts, prob)
    pts = _point_pass(a, wp, K, g, cand, len(link_pts), sum(len(p) for p in link_pts))
    probe_k, probe_loc = _probes(objs, prob, cand)
    qk = np.concatenate([pts["q_k"], probe_k]).astype(np.int32)
    ql = np.concatenate([pts["q_loc"].reshape(-1, 3), probe_loc])
    return {**pts, **_nearest_pass(a, wp, K, g, objs, prob, cand, qk, ql)}


@pytest.fixture(scope="module")
def cpu_warp(tmp_path_factory):
    wp = pytest.importorskip("warp")
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp_cache"))
    return wp


def _openblas_cores() -> list[str]:
    """Kernel names of the OpenBLAS libraries this process has loaded (empty when none can be read)."""
    with open("/proc/self/maps", encoding="utf-8") as f:
        libs = sorted({ln.split()[-1] for ln in f if "openblas" in ln.lower() and ln.rstrip().endswith(".so")})
    cores = []
    for p in libs:
        lib = ctypes.CDLL(p)
        for name in ("openblas_get_corename", "openblas_get_corename64_"):
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.restype = ctypes.c_char_p
                cores.append(fn().decode())
                break
    return cores


def _run_in_worker_env(tmp_path) -> tuple[dict, list[str], str]:
    """(_run() of a child started with the f0 worker environment, the child's loaded OpenBLAS kernels, the pin)."""
    from dexengine.core import blas_pin
    from dexengine.core.repo import REPO_ROOT
    from dexengine.retarget.f0.shadow_gpu import worker_env

    out, cache = tmp_path / "got.npz", tmp_path / "warp_cache"
    env = worker_env()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    cmd = [sys.executable, __file__, "--out", str(out), "--cache", str(cache)]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    got = dict(np.load(out))
    cores = [str(c) for c in got.pop("blas_cores")]
    return got, cores, blas_pin.KERNEL


def test_pts_and_nn_kernels_match_the_golden_bits(cpu_warp, tmp_path):
    got, cores, pin = _run_in_worker_env(tmp_path)
    if cores and any(c.lower() != pin.lower() for c in cores):
        pytest.skip(
            f"known limit: the f0 worker kernel pin OPENBLAS_CORETYPE={pin} does not load here (loaded {cores})"
        )
    want = np.load(GOLDEN)
    assert sorted(got) == sorted(want.files)
    for k in want.files:
        assert got[k].dtype == want[k].dtype and got[k].shape == want[k].shape, k
        assert got[k].tobytes() == want[k].tobytes(), k


def test_the_inputs_reach_every_verdict():
    """The golden is not trivial: queries exist, some points are below the desk, and both deep and close verdicts
    occur next to free points."""
    want = np.load(GOLDEN)
    assert want["q_cnt"].sum() > 0 and want["desk_cnt"].sum() > 0
    assert (want["r_deep"] > 0).any() and (want["r_close"] > 0).any() and (want["r_cnt"] == 0).any()


if __name__ == "__main__" and "--out" in sys.argv:  # the child of the golden test (worker environment)
    import warp as wp

    wp.config.kernel_cache_dir = sys.argv[sys.argv.index("--cache") + 1]
    res = _run()
    np.savez(sys.argv[sys.argv.index("--out") + 1], blas_cores=np.array(_openblas_cores()), **res)
elif __name__ == "__main__" and "--write" in sys.argv:  # run it under the f0 worker environment (module doc)
    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez(GOLDEN, **_run())
    print(f"wrote {GOLDEN}")
