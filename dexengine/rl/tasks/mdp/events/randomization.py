"""Domain randomization events for object manipulation training.

Object gravity ramp: The object starts with disable_gravity=True and receives
a ramped external gravitational force (0 → mg). Scene gravity remains at 9.81
throughout, so the hand always experiences full gravity and the policy operates
in consistent conditions.

Friction decay: Object friction starts at 3x nominal and decays to 1x nominal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from dexengine.core.physics import GRAVITY_MPS2
from dexengine.rl.tasks.mdp import physx_props

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def gravity_progress(step: int, ramp_steps: int, start_fraction: float) -> float:
    """Gravity ramp progress at `step`: start_fraction → 1 over ramp_steps (the one formula `ramp_object_gravity` and
    the carried slots' hold (scene.extra_carry) read)."""
    raw_progress = min(step / max(ramp_steps, 1), 1.0)
    return start_fraction + (1.0 - start_fraction) * raw_progress


def ramp_object_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # required; the effective value is resolved at runtime from --gravity_ramp_fraction in rl/train/runner.py
    # train_cfg_overrides (train) / pinned at eval. Every EventTerm caller passes it explicitly.
    ramp_steps: int,
    gravity_magnitude: float = GRAVITY_MPS2,
    object_name: str = "object_right",
    start_fraction: float = 0.0,
) -> None:
    """Ramp object gravity from start_fraction to full via external force.

    The object has disable_gravity=True in its rigid body config, so PhysX
    scene gravity does not affect it. Instead, we apply an external downward
    force F = -m * g * progress to simulate gradually increasing gravity.

    IMPORTANT: this function MUST run every step — it cannot early-return after
    the ramp completes. The object's ``permanent_wrench_composer`` is cleared
    by ``RigidObject.reset()`` on every env reset (Isaac Lab zeros both the
    permanent and instantaneous wrench composers in asset.reset). If we skip
    the update after the ramp, the object becomes weightless the moment its
    episode resets and stays that way for the rest of training / eval.
    WHEN it runs is owned by ``mdp/gravity_wrench`` (event mode ``MODE``, applied by the
    env right before physics, after every pose write): an interval call at the end of the
    step left the first step after a gym ``env.reset()`` weightless and missed the pose
    writes (park / spawn hold) that happen between the end of a step and the next physics.

    The body-frame force is built HERE from the current orientation. Passing
    ``is_global=True`` instead lets the WrenchComposer rotate world->body with a link
    pose it caches at the first set after its last reset and keeps until the next one,
    so between resets the "gravity" turned with the object.

    Args:
        start_fraction: Initial gravity fraction (0.0 = weightless, 0.5 = half gravity).
            The ramp goes from start_fraction to 1.0 over ramp_steps.
            Default 0.0 for backward compatibility (hand-only task env).
            The arm-involved task env uses 0.5 for stable early contact.
    """
    progress = gravity_progress(env.common_step_counter, ramp_steps, start_fraction)

    obj = env.scene[object_name]
    num_envs = env.num_envs

    # The object's CURRENT mass, per object (the old shared `env._object_mass_total` gave the LEFT bowl the RIGHT
    # tool's mass). randomize_object_mass_at_reset rewrites it every episode (default_mass × scale, identity during
    # the frozen eval) through physx_props, which keeps this device mirror equal to PhysX — so the ramp's weight is
    # m·g of the body that is actually simulated, in training and in eval. (A mass read once at the first call
    # froze each env's first randomized sample: 0.77–1.43 g effective, also inside the frozen eval.)
    masses = physx_props.read_device(env, object_name, "mass", env.device)
    object_mass_total = masses.reshape(num_envs, -1).sum(dim=1)
    _logged = f"_object_mass_logged_{object_name}"
    if not getattr(env, _logged, False):
        setattr(env, _logged, True)
        # Task-5 runtime mass log: the ACTUAL physical mass per object once, so the loaded mass is always in the log.
        print(
            f"[object_mass] '{object_name}' total mass = "
            f"{float(object_mass_total.float().mean().item()):.5f} kg (runtime, per-object, current)"
        )

    # F = m * g * progress (downward = negative z)
    force_z = -object_mass_total * gravity_magnitude * progress

    # World-frame force -> root-body frame with the CURRENT orientation (root_link_quat_w is fresh both after a
    # pose write at reset and after scene.update). The composer then holds a body-frame force for the next step.
    f_w = torch.zeros(num_envs, 3, device=env.device)
    f_w[:, 2] = force_z
    forces = math_utils.quat_apply_inverse(obj.data.root_link_quat_w, f_w).unsqueeze(1)

    obj.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torch.zeros_like(forces),
        body_ids=[0],
        is_global=False,
    )


def randomize_object_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # required; the effective value is resolved at runtime from --gravity_ramp_fraction in rl/train/runner.py
    # train_cfg_overrides (train) / pinned at eval. Every EventTerm caller passes it explicitly.
    ramp_steps: int,
    nominal_friction: float = 2.0,
    initial_multiplier: float = 3.0,
    final_multiplier: float = 1.0,
    object_name: str = "object_right",
) -> None:
    """Linearly decay object friction from high to nominal.

    Note: unlike the gravity ramp, friction is a material property written to
    the physx view and PERSISTS across asset resets (Isaac Lab's
    RigidObject.reset clears external wrenches but does NOT touch material
    properties). So it is safe (though not harmful) to early-return after the
    ramp completes. We keep the optimization to avoid per-step material writes
    once the schedule is settled.
    """
    # Early return if ramp already completed — material props persist across resets.
    if getattr(env, "_friction_ramp_done", False):
        return

    step = env.common_step_counter
    progress = min(step / max(ramp_steps, 1), 1.0)
    current_multiplier = initial_multiplier + (final_multiplier - initial_multiplier) * progress
    current_friction = nominal_friction * current_multiplier

    obj = env.scene[object_name]
    num_envs = env.num_envs
    num_shapes = obj.root_physx_view.max_shapes
    mat_props = torch.zeros(num_envs, num_shapes, 3, device="cpu")
    mat_props[:, :, 0] = current_friction  # static friction
    mat_props[:, :, 1] = current_friction  # dynamic friction
    mat_props[:, :, 2] = 0.0  # restitution
    physx_props.write(env, object_name, "material", mat_props, torch.arange(num_envs, device="cpu"))

    # Keep the **absolute value** the ramp last wrote: the only basis for the per-reset DR to multiply "this base value"
    # instead of "the current value".
    if not hasattr(env, "_friction_base"):
        env._friction_base = {}
    env._friction_base[object_name] = float(current_friction)

    if progress >= 1.0:
        env._friction_ramp_done = True


# ============================================================================
# Per-reset DR functions (Gate 0 cross-stage parity)
# Operate on a single
# "object" entity in the scene. Compatible with TACO env where the right-hand
# object is named "object".
# ============================================================================
def randomize_object_mass_at_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str = "object_right",
    mass_scale_range: tuple[float, float] = (0.7, 1.3),
) -> None:
    """Multiplicatively scale object mass at each episode reset.

    Applies a per-env random scale drawn from *mass_scale_range* to the
    PhysX default mass and recomputes the inertia tensor proportionally
    (same approach as Isaac Lab's randomize_rigid_body_mass with
    operation="scale" and recompute_inertia=True).

    Args:
        mass_scale_range: (lo, hi) multiplicative scale on the default mass.
            Training range: (0.7, 1.3).
            Near-OOD: (1.4, 1.6).  Far-OOD: (1.8, 2.2).
    """
    obj = env.scene[object_name]
    n_reset = len(env_ids)
    env_ids_cpu = env_ids.cpu()

    # Sample per-env scale factors
    lo, hi = mass_scale_range
    scales = torch.empty(n_reset, device="cpu").uniform_(lo, hi)

    # Get current masses (num_envs, num_bodies) on CPU
    masses = physx_props.read(env, object_name, "mass")  # CPU copy (bit-identical with PhysX)

    # Reset to default first, then scale (avoids accumulation across resets).
    # default_mass shape is (num_envs, num_bodies). Index by env_ids_cpu to
    # get only the resetting envs' default values, then scale per env.
    default_mass = obj.data.default_mass  # (num_envs, num_bodies)
    masses[env_ids_cpu] = default_mass[env_ids_cpu].cpu() * scales.unsqueeze(-1)

    physx_props.write(env, object_name, "mass", masses, env_ids_cpu)

    # Recompute inertia proportionally (uniform-density assumption).
    # Shape depends on asset type:
    #   - Articulation: (num_envs, num_bodies, 9)
    #   - RigidObject:  (num_envs, 9)
    inertias = obj.root_physx_view.get_inertias()  # CPU
    default_inertia = obj.data.default_inertia
    default_idx = default_inertia[env_ids_cpu].cpu()
    if default_idx.dim() == 2:
        # RigidObject: (n_reset, 9) — scale via (n_reset, 1) broadcast
        inertias[env_ids_cpu] = default_idx * scales.unsqueeze(-1)
    else:
        # Articulation: (n_reset, num_bodies, 9) — scale via (n_reset, 1, 1)
        inertias[env_ids_cpu] = default_idx * scales.view(n_reset, 1, 1)
    obj.root_physx_view.set_inertias(inertias, env_ids_cpu)


def randomize_object_friction_at_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str = "object_right",
    static_friction_scale_range: tuple[float, float] = (0.5, 1.5),
    dynamic_friction_scale_range: tuple[float, float] = (0.5, 1.5),
) -> None:
    """Multiplicatively scale object static and dynamic friction at each episode reset.

    Each env gets an independent random scale for static and dynamic friction
    drawn from their respective ranges.  Restitution is left unchanged.

    ⚠ INERT WHENEVER friction_decay IS ACTIVE (which is the default — only
    --disable_friction_decay / the no10pctphys recipe turns it off). This term scales
    the *current* material properties, but friction_decay is an interval event with
    interval_range_s=(0,0), i.e. it runs EVERY step and assigns an absolute
    nominal_friction × multiplier to all envs. So whatever this term writes at reset is
    erased on the first step of the episode and the object's friction is identical in
    every env. Do not read a nonzero static/dynamic_friction_scale_range as evidence
    that object friction is being randomized — check whether friction_decay is present
    first. (Sister effect: because friction_decay writes restitution 0.0 as well, the
    per-reset restitution DR is inert on the same grounds.)

    Because it scales the current value rather than a stored default, this term would
    also COMPOUND across resets if friction_decay were ever removed while it stayed on
    (unlike randomize_object_mass_at_reset, which resets to default first). Fix that
    before relying on it standalone.

    Args:
        static_friction_scale_range:  (lo, hi) multiplicative scale.
        dynamic_friction_scale_range: (lo, hi) multiplicative scale.
    """
    # ── Friction DR as an absolute write ────────────────────────────────
    # The original implementation **multiplied the current material value**. It was harmless only because
    # friction_decay (an interval event) overwrote the absolute value every step and so erased it. But once the ramp
    # ends friction_decay sets `_friction_ramp_done` and **returns early for good** (at 10% of the total steps, or
    # iteration 50 with a frozen eval); from then on nothing erases it and the multiplication **compounds**.
    # Measured (NE=2048 · 32 clips · step 11,737, per env without padding):
    #   min 0.0003 · p05 0.047 · p50 0.70 · p95 7.62 · max 176.6   (nominal 2.0)
    #   min 0.0020->0.0003, p50 0.83->0.70 keep drifting apart; nothing pulls them back.
    # Evaluation always fixes the nominal 2.0, so training and eval physics disagree (the situation this function's
    # docstring already warned about: "Fix that before relying on it").
    #
    # Fix (unconditional): multiply the base value and **write the absolute result** (no compounding possible).
    n_reset = len(env_ids)
    env_ids_cpu = env_ids.cpu()

    mat_props = physx_props.read(env, object_name, "material")  # (num_envs, num_shapes, 3) CPU copy

    # Sample scales — broadcast across shapes
    s_lo, s_hi = static_friction_scale_range
    d_lo, d_hi = dynamic_friction_scale_range
    s_scales = torch.empty(n_reset, device="cpu").uniform_(s_lo, s_hi)  # (n_reset,)
    d_scales = torch.empty(n_reset, device="cpu").uniform_(d_lo, d_hi)  # (n_reset,)

    # Apply — scale the current value (which after decay is the nominal value)
    # [:, :, 0] = static_friction, [:, :, 1] = dynamic_friction, [:, :, 2] = restitution
    # **Absolute write** from the base value. The current value is never read, so compounding is impossible.
    # Base value = the absolute friction the ramp last wrote (nominal 2.0 without one, for runs with the ramp off).
    _base = float(getattr(env, "_friction_base", {}).get(object_name, 2.0))
    _cur = mat_props[env_ids_cpu, :, 0]
    _live = _cur > 0  # padding slots without a shape (0) stay 0
    _zero = torch.zeros_like(_cur)
    mat_props[env_ids_cpu, :, 0] = torch.where(_live, _base * s_scales.unsqueeze(-1).expand_as(_cur), _zero)
    mat_props[env_ids_cpu, :, 1] = torch.where(_live, _base * d_scales.unsqueeze(-1).expand_as(_cur), _zero)

    physx_props.write(env, object_name, "material", mat_props, env_ids_cpu)


def randomize_object_restitution_at_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str = "object_right",
    restitution_scale_range: tuple[float, float] = (0.5, 1.5),
) -> None:
    """Multiplicatively scale object restitution at each episode reset.

    The nominal restitution base is set to 0.1 via RigidBodyMaterialCfg in the
    task env cfg (beaker USD bakes restitution=0.0, so the material override
    is required for multiplicative DR to have effect).

    Args:
        restitution_scale_range: (lo, hi) multiplicative scale.
            Training range: (0.5, 1.5).
            Near-OOD: (1.6, 1.8).  Far-OOD: (1.9, 2.0).
    """
    n_reset = len(env_ids)
    env_ids_cpu = env_ids.cpu()

    mat_props = physx_props.read(env, object_name, "material")  # (num_envs, num_shapes, 3) CPU copy

    lo, hi = restitution_scale_range
    scales = torch.empty(n_reset, device="cpu").uniform_(lo, hi)

    mat_props[env_ids_cpu, :, 2] = mat_props[env_ids_cpu, :, 2] * scales.unsqueeze(-1)

    physx_props.write(env, object_name, "material", mat_props, env_ids_cpu)


def randomize_object_com_at_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: object = None,
    offset_range_m: tuple[float, float] = (-0.02, 0.02),
    object_name: str = "object_right",
    com_offset_range_m: tuple[float, float] | None = None,
) -> None:
    """Apply a random per-axis COM offset at each episode reset.

    Samples an independent (x, y, z) offset for each resetting env from
    the uniform range and adds it to the default COM position via PhysX's
    set_coms API (confirmed present in IsaacLab 2.3.0,
    isaaclab/envs/mdp/events.py:432,438).

    The COM offset is expressed in the body's local frame (same convention
    as PhysX root_physx_view.get_coms()).

    Args:
        asset_cfg: Unused; kept for signature parity with randomize_object_mass_at_reset.
        offset_range_m: (lo, hi) uniform offset per axis in meters.
            Training range: (-0.02, 0.02) = ±2 cm.
            Near-OOD: (-0.03, 0.03).  Far-OOD: (-0.05, 0.05).
        object_name: Scene key for the object asset.
        com_offset_range_m: Deprecated alias for offset_range_m (backward compat).
    """
    # Backward-compat: com_offset_range_m overrides offset_range_m if provided
    if com_offset_range_m is not None:
        offset_range_m = com_offset_range_m

    obj = env.scene[object_name]
    n_reset = len(env_ids)
    env_ids_cpu = env_ids.cpu()

    lo, hi = offset_range_m

    # Get default COM positions. Shape depends on asset type:
    #   - Articulation: (num_envs, num_bodies, 7) — (x,y,z, qw,qx,qy,qz)
    #   - RigidObject:  (num_envs, 7) — single body, no bodies dim
    # The beaker is a RigidObject so we handle the 2D case.
    coms = obj.root_physx_view.get_coms().clone()

    # Default COM: stored at first call, PER OBJECT (keyed by scene name). A shared
    # cache would store the RIGHT object's default COM and then write it onto the LEFT
    # object on reset (same shared-cache class of bug as ramp_object_gravity), so the
    # left object would inherit the right object's COM before the random offset.
    _com_attr = f"_object_default_coms_{object_name}"
    if not hasattr(env, _com_attr):
        setattr(env, _com_attr, coms.clone())
    _default_coms = getattr(env, _com_attr)

    # Sample offsets: (n_reset, 3)
    rand_samples = torch.empty(n_reset, 3, device="cpu").uniform_(lo, hi)

    # Reset to default then apply offset
    if coms.dim() == 2:
        # RigidObject: (num_envs, 7) — (x,y,z, qw,qx,qy,qz)
        coms[env_ids_cpu] = _default_coms[env_ids_cpu].clone()
        coms[env_ids_cpu, :3] += rand_samples
    else:
        # Articulation: (num_envs, num_bodies, 7)
        n_bodies = coms.shape[1]
        coms[env_ids_cpu] = _default_coms[env_ids_cpu].clone()
        coms[env_ids_cpu, :, :3] += rand_samples.unsqueeze(1).expand(-1, n_bodies, -1)

    obj.root_physx_view.set_coms(coms, env_ids_cpu)
