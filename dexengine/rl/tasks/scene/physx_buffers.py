"""The PhysX GPU collision stack size of a run (`collision_stack`, the one rule; rl/cli._construct_env writes it once
the env count is known) and `[PHYSX_BUFFERS]`: the buffer sizes the stage's physics scene holds, read back after the
sim cfg wrote them (Isaac Lab authors cfg.sim.physx onto the scene prim's PhysxSceneAPI), against the cfg statement and
against the rule for the env's count — any mismatch raises (an entry that built the cfg without the cli is caught).
Standard library only. A PhysX overflow at run time is the launcher's (launch/process).
"""

from __future__ import annotations

#: The narrow-phase collision stack needed per env count. Measured (oakink2 8 clips, NE 4096, 5,120
#: triangle-collider cells): the S-true-shape triangle bodies needed up to 136,653,960 B within 64 steps at NE 4096, and
#: Isaac Lab's default 2^26 overflowed ("Contacts have been dropped"). Every env holds the same kind of scene, so the
#: need scales with the env count: 2^30 at NE 4096 (7.9x the peak; 3.3x after a scene with 2.4x the triangle cells per
#: env) and the same ratio at any NE, never below Isaac Lab's default (NE <= 256 keep 2^26: byte-equal there).
PHYSX_COLLISION_STACK_LOG2 = 30
PHYSX_COLLISION_STACK_REF_NE = 4096
#: Isaac Lab's PhysxCfg.gpu_collision_stack_size (the floor; the value before this rule).
ISAAC_COLLISION_STACK = 2**26
#: The measured peak the value is sized from (the [PHYSX_BUFFERS] marker states it).
PHYSX_COLLISION_STACK_PEAK_B = 136_653_960


def collision_stack(num_envs: int) -> int:
    """Collision stack bytes for `num_envs` envs: 2^30 x NE / 4096 rounded up to a power of two, at least 2^26."""
    n = int(num_envs)
    if n < 1:
        raise ValueError(f"[PHYSX_BUFFERS] num_envs {num_envs!r} < 1")
    need = -(-(2**PHYSX_COLLISION_STACK_LOG2) * n // PHYSX_COLLISION_STACK_REF_NE)
    return max(ISAAC_COLLISION_STACK, 1 << (need - 1).bit_length())


RULE = (
    f"2^{PHYSX_COLLISION_STACK_LOG2} x NE/{PHYSX_COLLISION_STACK_REF_NE}, floor 2^26 · measured peak "
    f"{PHYSX_COLLISION_STACK_PEAK_B:,} B at NE {PHYSX_COLLISION_STACK_REF_NE}"
)
MARK = "[PHYSX_BUFFERS]"
#: (short name, PhysxSceneAPI attribute, cfg.sim.physx field)
BUFFERS = (
    ("collision_stack", "physxScene:gpuCollisionStackSize", "gpu_collision_stack_size"),
    ("contact", "physxScene:gpuMaxRigidContactCount", "gpu_max_rigid_contact_count"),
    ("patch", "physxScene:gpuMaxRigidPatchCount", "gpu_max_rigid_patch_count"),
    ("found_lost", "physxScene:gpuFoundLostPairsCapacity", "gpu_found_lost_pairs_capacity"),
    ("found_lost_agg", "physxScene:gpuFoundLostAggregatePairsCapacity", "gpu_found_lost_aggregate_pairs_capacity"),
    ("total_agg", "physxScene:gpuTotalAggregatePairsCapacity", "gpu_total_aggregate_pairs_capacity"),
    ("temp", "physxScene:gpuTempBufferCapacity", "gpu_temp_buffer_capacity"),
)


def _pow2(v) -> str:
    v = int(v)
    return f"2^{v.bit_length() - 1}" if v > 0 and v & (v - 1) == 0 else str(v)


def stage_values(stage, scene_path: str) -> dict:
    """{short name: value | None} of the physics scene prim at `scene_path`."""
    prim = stage.GetPrimAtPath(scene_path)
    if not prim.IsValid():
        raise RuntimeError(f"{MARK} no physics scene prim at {scene_path}")
    out = {}
    for name, attr, _field in BUFFERS:
        a = prim.GetAttribute(attr)
        out[name] = a.Get() if a and a.HasAuthoredValue() else None
    return out


def line(stage, physx_cfg, scene_path: str, num_envs: int) -> str:
    """The marker line; raises when the stage and the cfg statement differ on any buffer, or the collision stack is not
    the rule's value for `num_envs`."""
    got = stage_values(stage, scene_path)
    bad = {n: (got[n], getattr(physx_cfg, f)) for n, _a, f in BUFFERS if got[n] != getattr(physx_cfg, f)}
    if bad:
        raise RuntimeError(f"{MARK} stage {scene_path} differs from cfg.sim.physx (stage, cfg): {bad}")
    want = collision_stack(num_envs)
    if got["collision_stack"] != want:
        raise RuntimeError(
            f"{MARK} collision stack {got['collision_stack']} on the stage, the rule gives {want} for NE {num_envs} "
            f"({RULE}): this env's cfg did not go through rl/cli._construct_env"
        )
    rest = " · ".join(f"{n} {_pow2(got[n])}" for n, _a, _f in BUFFERS[1:])
    return (
        f"{MARK} collision_stack {_pow2(ISAAC_COLLISION_STACK)} → {_pow2(want)} at NE {num_envs} ({RULE}) · {rest} · "
        f"stage {scene_path} = cfg"
    )


def print_line(env) -> str:
    """The marker of a built env (its stage, cfg.sim.physx, its env count)."""
    s = line(env.scene.stage, env.cfg.sim.physx, env.cfg.sim.physics_prim_path, env.scene.num_envs)
    print(s, flush=True)
    return s
