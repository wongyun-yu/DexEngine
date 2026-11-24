"""Warp kernel of the cross-repair GPU filter: mujoco_warp's GJK/EPA (ccd) called directly, one kernel per geom-type
pair (part of xr_gpu; the module-level kernel cache _WP is kept by the resident bake server between clips).

The kernel's Warp-struct annotations are closure objects of _ccd_kernel, so this module keeps evaluated (not
postponed) annotations."""

_WP = {}
_SERVER_KEEP = ("_WP",)  # Warp kernels (keyed by geom-type pair) — a clip-independent cache the resident worker keeps


def _warp():
    if _WP:
        return _WP
    import importlib.util as _ilu
    import sys as _sys
    import types as _types

    from dexengine.core import runtime_env

    wp = runtime_env.warp()

    if "mujoco_warp" not in _sys.modules:
        # The top-level package __init__ pulls in the whole physics engine (forward, solver, io ...) at ~1.9 s. Only
        # _src's GJK/EPA and its dependencies (types, math, warp_util, collision_primitive) are used, so the top level
        # is stood up as an empty package.
        _spec = _ilu.find_spec("mujoco_warp")
        if _spec is None:
            raise RuntimeError(
                # the head "[XR_GPU] mujoco_warp" marks a GPU-path failure in bake logs
                "[XR_GPU] mujoco_warp is not installed in this interpreter — install it into the Isaac env with "
                "`pip install --no-deps git+https://github.com/google-deepmind/mujoco_warp@dde89450`"
            )
        _pkg = _types.ModuleType("mujoco_warp")
        _pkg.__path__ = list(_spec.submodule_search_locations)
        _sys.modules["mujoco_warp"] = _pkg
    import numpy as np
    from mujoco_warp._src.collision_gjk import FLOAT_MAX
    from mujoco_warp._src.types import MJ_MAX_EPAFACES, MJ_MAX_EPAHORIZON

    if FLOAT_MAX != 1e30:  # the separating-plane sentinel the filter reads (xr_gpu.GpuDist.query) — a change dies
        raise RuntimeError(f"[XR_GPU] mujoco_warp FLOAT_MAX is {FLOAT_MAX!r}, the filter was proven for 1e30")
    wp.config.quiet = True
    _WP.update(
        wp=wp,
        NF=MJ_MAX_EPAFACES,
        NH=MJ_MAX_EPAHORIZON,
        kernels={},
        commit=_mjw_commit(),
        FAR=float(np.float32(FLOAT_MAX)),  # the sentinel as the float32 out_d reads back
    )
    return _WP


def _structs() -> dict:
    """The Warp structs of the ccd kernel's arguments, built once per process: the model's geom and mesh arrays, the
    queries, the EPA work buffers and the outputs."""
    W = _warp()
    if "structs" not in W:
        wp = W["wp"]
        a, a2 = wp.array, wp.array2d
        fields = {
            "model": {
                "geom_type": a(dtype=int),
                "geom_dataid": a(dtype=int),
                "geom_size": a(dtype=wp.vec3),
                "mesh_vertadr": a(dtype=int),
                "mesh_vertnum": a(dtype=int),
                "mesh_graphadr": a(dtype=int),
                "mesh_vert": a(dtype=wp.vec3),
                "mesh_graph": a(dtype=int),
            },
            "query": {
                "q_g1": a(dtype=int),
                "q_g2": a(dtype=int),
                "q_dx": a(dtype=float),
                "q_p2": a(dtype=wp.vec3),
                "q_r1": a(dtype=wp.mat33),
                "q_r2": a(dtype=wp.mat33),
            },
            "epa": {
                "ev": a2(dtype=wp.vec3),
                "ev1": a2(dtype=wp.vec3),
                "ev2": a2(dtype=wp.vec3),
                "evi1": a2(dtype=int),
                "evi2": a2(dtype=int),
                "ef": a2(dtype=wp.vec3i),
                "epr": a2(dtype=wp.vec3),
                "en2": a2(dtype=float),
                "eix": a2(dtype=int),
                "emap": a2(dtype=int),
                "ehor": a2(dtype=int),
            },
            "out": {
                "out_d": a(dtype=float),
                "out_w1": a(dtype=wp.vec3),
                "out_w2": a(dtype=wp.vec3),
                "out_n": a(dtype=int),
            },
        }
        W["structs"] = {
            k: wp.struct(type(f"Xr_{k}", (), {"__annotations__": f, "__module__": __name__})) for k, f in fields.items()
        }
    return W["structs"]


def struct_of(kind: str, arrays: dict):
    """A ccd-kernel struct argument (`model`, `query`, `epa` or `out`) whose fields are taken by name from `arrays`."""
    cls = _structs()[kind]
    st = cls()
    for name in cls.vars:
        setattr(st, name, arrays[name])
    return st


def _kernel(T1: int, T2: int):
    """A kernel specialised to (geom type 1, geom type 2): the types go in as wp.static constants so the compiler drops
    the other types' branches, and no backward code is generated (enable_backward=False — the same approach as
    mujoco_warp's ccd_kernel_builder). Compiled and cached once per type pair."""
    W = _warp()
    key = (int(T1), int(T2))
    if key not in W["kernels"]:
        W["kernels"][key] = _ccd_kernel(W["wp"], _structs(), *key)
    return W["kernels"][key]


def _ccd_kernel(wp, S: dict, T1: int, T2: int):
    """The (T1, T2) kernel: per query, geom 1 at its x offset and geom 2 at its relative position, both rotated, then
    mujoco_warp's ccd -> signed distance, the two witness points and ccd's contact count (which tells the
    separating-plane certificate from an EPA failure — both return FLOAT_MAX)."""
    from mujoco_warp._src.collision_gjk import ccd
    from mujoco_warp._src.collision_primitive import Geom

    Model, Query, Epa, Out = S["model"], S["query"], S["epa"], S["out"]

    @wp.kernel(enable_backward=False, module="unique")
    def _xr_ccd(model: Model, q: Query, tol: float, cutoff: float, iters: int, epa: Epa, out: Out):
        tid = wp.tid()
        g1 = q.q_g1[tid]
        g2 = q.q_g2[tid]
        geom1 = Geom()
        geom2 = Geom()
        # origin = geom1's (unperturbed) world position — float32 precision is not spent on a large common offset
        # (the CPU subtracted the relative coordinates in float64)
        geom1.pos = wp.vec3(q.q_dx[tid], 0.0, 0.0)
        geom1.rot = q.q_r1[tid]
        geom1.size = model.geom_size[g1]
        geom1.normal = wp.vec3(geom1.rot[0, 2], geom1.rot[1, 2], geom1.rot[2, 2])
        geom2.pos = q.q_p2[tid]
        geom2.rot = q.q_r2[tid]
        geom2.size = model.geom_size[g2]
        geom2.normal = wp.vec3(geom2.rot[0, 2], geom2.rot[1, 2], geom2.rot[2, 2])
        t1 = wp.static(T1)
        t2 = wp.static(T2)
        if wp.static(T1 == 7):
            d1 = model.geom_dataid[g1]
            geom1.vertadr = model.mesh_vertadr[d1]
            geom1.vertnum = model.mesh_vertnum[d1]
            geom1.graphadr = model.mesh_graphadr[d1]
            geom1.vert = model.mesh_vert
            geom1.graph = model.mesh_graph
            geom1.mesh_polyadr = -1
            geom1.mesh_polynum = -1
        if wp.static(T2 == 7):
            d2 = model.geom_dataid[g2]
            geom2.vertadr = model.mesh_vertadr[d2]
            geom2.vertnum = model.mesh_vertnum[d2]
            geom2.graphadr = model.mesh_graphadr[d2]
            geom2.vert = model.mesh_vert
            geom2.graph = model.mesh_graph
            geom2.mesh_polyadr = -1
            geom2.mesh_polynum = -1
        geom1.index = -1
        geom1.margin = 0.0
        geom2.index = -1
        geom2.margin = 0.0
        dist, ncon, w1, w2, idx = ccd(
            tol,
            cutoff,
            iters,
            geom1,
            geom2,
            t1,
            t2,
            geom1.pos,
            geom2.pos,
            epa.ev[tid],
            epa.ev1[tid],
            epa.ev2[tid],
            epa.evi1[tid],
            epa.evi2[tid],
            epa.ef[tid],
            epa.epr[tid],
            epa.en2[tid],
            epa.eix[tid],
            epa.emap[tid],
            epa.ehor[tid],
        )
        out.out_d[tid] = dist
        out.out_w1[tid] = w1
        out.out_w2[tid] = w2
        out.out_n[tid] = ncon  # (FLOAT_MAX, 1) = GJK's separating-plane exit at the cutoff · (FLOAT_MAX, 0) = EPA fail

    return _xr_ccd


def _mjw_commit() -> str:
    """Git commit of the installed mujoco_warp (for the marker; the filter was validated with dde89450), else its
    version; "unregistered" when it is not installed as a distribution."""
    import importlib.metadata as _md
    import json as _js

    try:
        d = _md.distribution("mujoco-warp")
    except _md.PackageNotFoundError:
        return "unregistered"
    du = d.read_text("direct_url.json")
    return _js.loads(du).get("vcs_info", {}).get("commit_id", d.version)[:8] if du else d.version
