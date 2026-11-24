"""Single owner of the spawn hold (SPAWN_HOLD) window and of the articulated joint pin.

Window = N policy steps after a reset (env DEXENGINE_OBJ_SPAWN_HOLD; canonical table scripts/_rl_recipe.sh). During the
window `ArmResidualTrainEnv.step` rewrites the objects at their reference pose every step
(`pin_objects_to_reference(joints=True)`), and the articulated sanity termination
(`mdp.terminations.ArticulatedObjectSanity`) does not count the speed rule for steps that started in the window. Both
read the same function, so the window never splits in two.

Joint pin: pinning only the root sends the depenetration impulse of a reset penetration into the free hinge (stiffness
0), which spins up to the speed limit. So during the window the joint state is rewritten too (reference θ, ω 0), with
the same function `pin_joints` as the reset (`mdp.reset_object_to_trajectory`). The judge park (PARK, rl.judge.park)
writes the root only (`joints=False`; the basis of PARK gates C and D is unchanged).
"""

from __future__ import annotations

import torch

from dexengine.rl.tasks import env_flags

HOLD_VAR = "DEXENGINE_OBJ_SPAWN_HOLD"
JOINTS_MARK = "[SPAWN_HOLD] joints"


def spawn_hold_steps() -> int:
    """Window length N (policy steps); unset or empty = 0 (off). Anything but an integer ≥ 0 fails."""
    v = (env_flags.get(HOLD_VAR) or "0").strip()
    if not v.isdigit():
        raise ValueError(f"[SPAWN_HOLD] {HOLD_VAR}={v!r} — integers ≥ 0 only")
    return int(v)


def pinned_before_physics(el: torch.Tensor, n: int) -> torch.Tensor:
    """(num_envs,) bool: envs pinned **just before** this policy step's physics (pre-physics counter el < N). Reader =
    the pin."""
    return el < n


def step_began_pinned(el_post: torch.Tensor, n: int) -> torch.Tensor:
    """(num_envs,) bool: envs where the policy step that just ended **started** pinned. Terminations read after IsaacLab
    advanced the counter by 1, so the pre-physics value = el_post − 1 -> the same window as `pinned_before_physics`
    (1 ≤ el_post ≤ N). Reader = the articulated sanity termination."""
    return pinned_before_physics(el_post - 1, n)


def pin_joints(env, ids: torch.Tensor, side: str, obj, ref: dict) -> int:
    """Joint state of the `side` articulated object for ids = (reference θ, ω 0). Reference θ = the trajectory manager's
    `{side}_obj_arti` (the same `ref`, the same frame as the root pose). ω 0 = the same kinematic approximation as the
    root pin (velocity 0). Returns the number of joints written (rigid = 0: no joint write window or 0 joints). Joints
    whose count differs from the reference columns, or a missing reference column, fail."""
    if not hasattr(obj, "write_joint_state_to_sim"):
        return 0
    n_j = int(getattr(obj, "num_joints", 0) or 0)
    if n_j < 1:
        return 0
    key = f"{side}_obj_arti"
    if key not in ref:
        raise KeyError(
            f"[SPAWN_HOLD] {side}: articulated but the reference state has no '{key}' — a loader without the joint "
            f"columns (rule D1)"
        )
    jp = ref[key].to(env.device).reshape(len(ids), -1)
    if jp.shape[1] != n_j:
        raise ValueError(
            f"[SPAWN_HOLD] {side}: object joints {n_j} ≠ reference joint columns {jp.shape[1]} — one-joint convention "
            f"broken"
        )
    obj.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
    return n_j
