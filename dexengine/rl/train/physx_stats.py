"""PhysX scene GPU statistics: heap bytes per allocator group and the last simulated frame's constraint counts.

Read through ``omni.physx`` ``PhysicsSceneStats`` (the binding of ``PxSimulationStatistics``). The GPU solver's
per-frame constraint buffers grow by doubling when a frame needs more and never shrink, so a frame with many contacts
raises the device memory floor for the rest of the process; ``physx_stats_line`` puts heap bytes and contact counts on
one line so the two can be read together.
"""

from __future__ import annotations

HEAP_KEYS = (
    ("gpu_mem_heap", "heap"),
    ("gpu_mem_heap_broadphase", "bp"),
    ("gpu_mem_heap_narrowphase", "np"),
    ("gpu_mem_heap_solver", "solver"),
    ("gpu_mem_heap_articulation", "artic"),
    ("gpu_mem_heap_simulation", "sim"),
    ("gpu_mem_heap_other", "other"),
)
COUNT_KEYS = (
    ("nb_active_constraints", "constr"),
    ("nb_axis_solver_constaints", "axis"),
    ("nb_discrete_contact_pairs_total", "pairs"),
    ("nb_discrete_contact_pairs_with_contacts", "pairs_c"),
    ("nb_new_touches", "new_t"),
    ("nb_lost_touches", "lost_t"),
    ("nb_partitions", "part"),
    ("nb_active_dynamic_rigids", "dyn"),
)
MEM_KEYS = (
    ("compressed_contact_size", "cc"),
    ("required_contact_constraint_memory", "req_cc"),
    ("peak_constraint_memory", "peak_c"),
)
CAP_KEYS = (
    ("gpu_mem_rigid_contact_count", "contacts"),
    ("gpu_mem_rigid_patch_count", "patches"),
    ("gpu_mem_found_lost_pairs", "fl_pairs"),
    ("gpu_mem_found_lost_aggregate_pairs", "fl_agg"),
    ("gpu_mem_total_aggregate_pairs", "agg_total"),
    ("gpu_mem_temp_buffer_capacity", "tmp"),
    ("gpu_mem_collision_stack_size", "cstack"),
)

_cache: dict = {"stage_id": None, "enc": None}


def physx_scene_stats() -> dict:
    """PhysX scene statistics as a dict; {"error": reason} outside Isaac, without a scene, or on a failed call."""
    try:
        import omni.usd
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import PhysicsSchemaTools, UsdPhysics
    except Exception as e:  # noqa: BLE001 — outside the Isaac runtime
        return {"error": f"{type(e).__name__}: {e}"}
    try:
        ctx = omni.usd.get_context()
        stage_id = ctx.get_stage_id()
        # Find the scene prim once: traversing a 4096-env stage cannot run every step.
        if _cache["enc"] is None or _cache["stage_id"] != stage_id:
            scene_path = None
            for prim in ctx.get_stage().Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    scene_path = prim.GetPath()
                    break
            if scene_path is None:
                return {"error": "no scene prim"}
            _cache["stage_id"] = stage_id
            _cache["enc"] = PhysicsSchemaTools.sdfPathToInt(scene_path)  # the binding takes one int
        st = PhysicsSceneStats()
        ok = get_physx_statistics_interface().get_physx_scene_statistics(stage_id, _cache["enc"], st)
        if not ok:
            return {"error": "statistics failed"}
        d = {}
        for keys in (HEAP_KEYS, COUNT_KEYS, MEM_KEYS, CAP_KEYS):
            for k, _ in keys:
                d[k] = int(getattr(st, k, -1))
        return d
    except Exception as e:  # noqa: BLE001 — the stats are a report; the error goes into the returned dict
        return {"error": f"{type(e).__name__}: {e}"}


def _gb(v: int) -> str:
    return f"{v / 2**30:.2f}"


def _mb(v: int) -> str:
    return f"{v / 2**20:.1f}M"


def physx_heap_line(d: dict) -> str:
    if "error" in d:
        return f"physx: {d['error']}"
    return "physx " + " ".join((f"{s}={_gb(d[k])}G" if s == "heap" else f"{s}={_gb(d[k])}") for k, s in HEAP_KEYS)


def physx_counts_line(d: dict) -> str:
    if "error" in d:
        return ""
    return " ".join(f"{s}={d[k]}" for k, s in COUNT_KEYS) + " " + " ".join(f"{s}={_mb(d[k])}" for k, s in MEM_KEYS)


def physx_stats_line(d: dict | None = None) -> str:
    """One line: heap bytes per group | last-frame constraint counts | capacities."""
    if d is None:
        d = physx_scene_stats()
    if "error" in d:
        return f"physx: {d['error']}"
    # omni.physx may leave the capacity fields unset in this build (all 0 when measured): print raw ints.
    caps = " ".join(f"{s}={d[k]}" for k, s in CAP_KEYS)
    return f"{physx_heap_line(d)} | {physx_counts_line(d)} | caps: {caps}"
