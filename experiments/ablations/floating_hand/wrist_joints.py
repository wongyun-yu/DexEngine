"""Floating hand, joint wrist drive (``DEXENGINE_FH_WRIST=joint``, off by default): six virtual wrist joints.

Why: the external wrench sees state and load one physics step late, and its model inertia (the root body's own) is
smaller than the articulation's, so the wrist ROTATION tracking error grows with dt. An arm's joint PD is solved
implicitly by PhysX in the same step with the true inertia. The joint drive puts, per hand, a world-fixed anchor →
3 prismatic → 3 revolute (Y→X→Z) virtual joints → the hand root, driven by PhysX implicit joint drives
(ImplicitActuator, gains = WRIST_GAINS) — the 6-DoF wrist of DexMachina (Zhao et al., 2025). The anchor is put at each
env's start reference wrist at every reset (virtual joints 0).
Rotation order Y→X→Z: on 62 TACO clips the middle angle is
smallest about X (max 48° / 55°); other datasets are re-measured before use.

Env flags (read inside the Isaac process — a process boundary, so they stay env vars):
  DEXENGINE_FH_WRIST      '' (default: wrench) | 'joint'
  DEXENGINE_FH_NAN_RESET  '' / '0' (default) | '1': an env whose reward is non-finite gets reward 0 for that step and is
                          terminated the next step (floating hand only; the arm body has no consumer and dies)
  DEXENGINE_FH_PALM_POINT '' (default) | 'arm_ee': the palm point of the palm rewards (palm_point_mode)
"""

from __future__ import annotations

from dexengine.core.floating import (  # noqa: F401 — the asset-path rule is the release registry's
    VJ_STEM_SUFFIX,
    WRIST_DRIVE_ENV,
    WRIST_DRIVES,
    joint_drive,
    mode_canon_path,
    vj_canon_path,
    wrist_drive,
)
from dexengine.rl.tasks import switches
from experiments.ablations.floating_hand.identity import MARK, active
from experiments.ablations.floating_hand.wrench import WRIST_GAINS, _quat_conj, _quat_mul, _quat_rotate

#: virtual joints (name tail, kind, parent axis) in chain order. Rotation order Y→X→Z (measured on TACO: the middle
#: angle is smallest about X).
VJ_AXES = (
    ("px", "prismatic", (1.0, 0.0, 0.0)),
    ("py", "prismatic", (0.0, 1.0, 0.0)),
    ("pz", "prismatic", (0.0, 0.0, 1.0)),
    ("ry", "revolute", (0.0, 1.0, 0.0)),
    ("rx", "revolute", (1.0, 0.0, 0.0)),
    ("rz", "revolute", (0.0, 0.0, 1.0)),
)
VJ_LIN_LIMIT_M = 1.0  # prismatic limit ± (m) — measured max 0.33 m from the start
#: Y · Z limit ± (rad) — above ±π so that PhysX does not wrap the angle (a limited revolute joint = an unwrapped
#: angle)
VJ_ROT_OUTER_LIMIT = 2.0 * 3.141592653589793 - 0.01
VJ_ROT_MID_LIMIT = 1.50  # X (middle) limit ± (rad) — before the gimbal (±π/2) · measured max 0.96 rad
VJ_GIMBAL_WARN = 1.309  # 75° — physics steps above it are counted in the marker
VJ_LINK_MASS = 1e-3  # virtual link mass (kg) — 3 orders below the hand (0.6 kg)
VJ_LINK_INERTIA = 1e-6  # virtual link inertia diagonal (kg·m²)
#: Virtual-joint armature rule: revolute = kd²/(16·kp) (WRIST_GAINS — ζ = 2 if that joint's inertia alone) ·
#: prismatic = 0 (the linear mass as it is). Why: with light virtual links (the mass / inertia above) the training solve
#: (8/0 iterations) does not resolve a load statically — in a contact-free test a 20 N·m torque leaked into 73 mm of
#: translation and 22° about another axis; with armature 0.375 kg·m² 15/15 cases pass. Heavier links (0.2× hand
#: mass, 100× inertia) also pass but double the linear mass (0.62 → 1.25 kg) and leak 18× more torque, so they were
#: not chosen. Slow pole kp/kd 10.0 → 10.7 rad/s (tracking nearly unchanged). Marker [FH_WRIST] armature rot=… pris=0
#: rule=kd2/16kp.
VJ_ARMATURE_RULE = "kd2/16kp"
#: Same flag family (off by default): an env whose reward is non-finite → reward 0 for that step, terminate next step
#: (one reward NaN with a finite state was measured at 200 Hz on an OakInk2 clip; the state sanitizer only fixes
#: non-finite / absurd states).
NAN_RESET_ENV = "DEXENGINE_FH_NAN_RESET"


def vj_joint_names(side: str) -> list[str]:
    """Virtual joint names in chain order — ``{side}_fh_wrist_{px,py,pz,ry,rx,rz}`` (the side token first, so that
    split_artic.side_of splits them)."""
    return [f"{side}_fh_wrist_{a}" for a, _k, _x in VJ_AXES]


def vj_link_names(side: str) -> list[str]:
    """Virtual link names (anchor = articulation root, world-fixed) — anchor + 5 links (the last revolute joint's child
    is the hand root)."""
    return [f"{side}_fh_wrist_anchor"] + [f"{side}_fh_wrist_l{i}" for i in range(1, len(VJ_AXES))]


def vj_armature(sides=("right", "left")) -> dict:
    """Virtual joint armature (joint name → value) — the rule VJ_ARMATURE_RULE."""
    g = WRIST_GAINS
    rot = g.rot_kd**2 / (16.0 * g.rot_kp)
    out = {}
    for side in sides:
        for name, (_a, kind, _x) in zip(vj_joint_names(side), VJ_AXES, strict=True):
            out[name] = 0.0 if kind == "prismatic" else rot
    return out


def vj_armature_mark() -> str:
    """Marker of the armature rule — a joint-drive run without it in its log is void."""
    rot = WRIST_GAINS.rot_kd**2 / (16.0 * WRIST_GAINS.rot_kp)
    return f"[FH_WRIST] armature rot={rot:g} pris=0 rule={VJ_ARMATURE_RULE}"


def vj_actuator_gains(sides=("right", "left")) -> dict:
    """Virtual joint actuator values (joint name → value): linear gains on prismatic, rotational on revolute."""
    g = WRIST_GAINS
    kp, kd, eff = {}, {}, {}
    for side in sides:
        for name, (_a, kind, _x) in zip(vj_joint_names(side), VJ_AXES, strict=True):
            lin = kind == "prismatic"
            kp[name] = g.lin_kp if lin else g.rot_kp
            kd[name] = g.lin_kd if lin else g.rot_kd
            eff[name] = g.lin_force_max if lin else g.rot_torque_max
    return {"stiffness": kp, "damping": kd, "effort_limit_sim": eff}


def nan_reset() -> bool:
    """Reward-NaN reset on? '' · '0' = off (default) · '1' = on (floating hand only — the arm body dies)."""
    raw = switches.text(NAN_RESET_ENV)
    if raw in ("", "0"):
        return False
    if raw != "1":
        raise RuntimeError(f"{MARK} {NAN_RESET_ENV}={raw!r} — only 0 | 1")
    if not active():
        raise RuntimeError(f"{MARK} {NAN_RESET_ENV}=1 is for the floating hand only — the arm body has no consumer")
    return True


#: Palm point of the floating hand's palm rewards and diagnostics (owner of the env read and the value check; the offset
#: math is the RL task's, dexengine.rl.tasks.mdp.naming.fh_palm_point_offset). '' (default) = the hand root (the wrist)
#: · 'arm_ee' = the arm body's EE origin (wrist + R_wrist · (−R_mᵀ t), mount = the data-world arm's), the same
#: physical point as the arm body's palm body (rby1 × wuji2: 15.5 cm from the wrist).
PALM_POINT_ENV = "DEXENGINE_FH_PALM_POINT"
PALM_POINT_MODES = ("arm_ee",)


def palm_point_mode() -> str:
    """'' (unset · off) | 'arm_ee'. Any other value dies; 'arm_ee' on the arm body dies (it has no consumer there)."""
    raw = switches.text(PALM_POINT_ENV).strip()
    if not raw:
        return ""
    if raw not in PALM_POINT_MODES:
        raise ValueError(f"{PALM_POINT_ENV}={raw!r} — only {'|'.join(PALM_POINT_MODES)} (or unset)")
    if not active():
        from dexengine.core.arms import active_arm_name  # noqa: PLC0415

        raise ValueError(f"{PALM_POINT_ENV}={raw} is for the floating hand only — active arm {active_arm_name()!r}")
    return raw


def euler_yxz(q):
    """Rotation q (wxyz, anchor frame) → intrinsic Y→X→Z angles (a, b, c) with R = Ry(a)·Rx(b)·Rz(c);
    b ∈ [−π/2, π/2] · a, c ∈ (−π, π]."""
    import torch  # noqa: PLC0415

    w, x, y, z = q.unbind(-1)
    m02 = 2 * (x * z + w * y)
    m22 = 1 - 2 * (x * x + y * y)
    m12 = 2 * (y * z - w * x)
    m10 = 2 * (x * y + w * z)
    m11 = 1 - 2 * (x * x + z * z)
    b = torch.asin((-m12).clamp(-1.0, 1.0))
    return torch.stack((torch.atan2(m02, m22), b, torch.atan2(m10, m11)), dim=-1)


def vj_targets(anchor_pos_w, anchor_quat_w, tgt_pos_w, tgt_quat_w, q_cur, lin_limit: float = VJ_LIN_LIMIT_M):
    """Hand-root world target → virtual joint targets (N,6): prismatic = position in anchor axes · revolute = Y→X→Z
    angles relative to the anchor. Y · Z take the 2π branch nearest the current joint angle (a wrapped target does
    not spin the
    other way) and are clipped to the limits; X is clipped to its limit. Pure torch."""
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415

    qa_inv = _quat_conj(anchor_quat_w)
    p = _quat_rotate(qa_inv, tgt_pos_w - anchor_pos_w).clamp(-lin_limit, lin_limit)
    e = euler_yxz(_quat_mul(qa_inv, tgt_quat_w))
    cur = q_cur[:, 3:6]
    two_pi = 2.0 * math.pi
    out_ac = cur + torch.remainder(e - cur + math.pi, two_pi) - math.pi  # nearest branch
    rot = torch.stack(
        (
            out_ac[:, 0].clamp(-VJ_ROT_OUTER_LIMIT, VJ_ROT_OUTER_LIMIT),
            e[:, 1].clamp(-VJ_ROT_MID_LIMIT, VJ_ROT_MID_LIMIT),
            out_ac[:, 2].clamp(-VJ_ROT_OUTER_LIMIT, VJ_ROT_OUTER_LIMIT),
        ),
        dim=-1,
    )
    return torch.cat((p, rot), dim=-1)
