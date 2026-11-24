"""Floating hand, default wrist drive: an external SE(3) spring-damper wrench on the hand root body (pure torch).

The wrench pulls the hand root toward the wrist target (reference wrist pose + policy residual). Gains, the inertia
rule of the implicit damping and the load observer live here; the Isaac action term (action.py) only wires them.
torch is imported inside the functions so that this module loads without it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WristGains:
    """Wrist SE(3) spring-damper gains, SI units. Starting point = FlashCHORD's gains (read from an external CHORD
    checkout; no FlashCHORD code is copied here)."""

    lin_kp: float  # N/m        — prismatic PD stiffness (ke) of the 3 linear axes
    lin_kd: float  # N·s/m      — prismatic PD damping (implicit, the order of MuJoCo implicitfast)
    lin_force_max: float  # N          — per world axis, on spring + damper (prismatic effort_limit)
    rot_kp: float  # N·m/rad    — rotational spring on the body-frame rotvec error
    rot_kd: float  # N·m·s/rad  — rotational passive damping (MuJoCo dof_passive_damping — implicit)
    rot_torque_max: float  # N·m        — per body axis, on the spring torque
    source: str


#: Wrist gains: FlashCHORD's stiff tracking set as the starting point (its default kp 25 set left a 6 cm p95 wrist error
#: at our termination thresholds), then a first fit on a CPU rigid-body tracking grid over the 64 B0r clips:
#: linear kp 1500 · ζ 1.0 (kd 88.66) · rotational kp 600 · kd 60 · limits 2000. 64-clip wrist position p95 3.14 →
#: 1.44 cm · rotation p95 7.7 → 3.5° (against the stiff set). The paper reports no gain selection.
WRIST_GAINS = WristGains(
    lin_kp=1500.0,
    lin_kd=88.66,
    lin_force_max=2000.0,
    rot_kp=600.0,
    rot_kd=60.0,
    rot_torque_max=2000.0,
    # the files cited below are those of an external CHORD checkout (FlashCHORD, commit in baselines/TREES.txt)
    source=(
        "first fit on a CPU rigid-body tracking grid (64 B0r clips) — starting point = FlashCHORD stiff set "
        "src/flash_chord/embodiments/sharpa_hands.py:67-74 SharpaHandsConfig.stiff_tracking "
        "(wrist_[xyz] kp 6000 · kd 60 · effort_limit 2000 / wrist_rotvec mode=effort kd 60 — in effort mode "
        "_apply_control_config (sharpa_hands.py:167-175) drops kp (ke=0) and uses kd as dof_passive_damping) · "
        "configs/action/residual_hand_pose_stiff.yaml:5-6 (wrist_kp 1200 · wrist_tau_max 2000 — the rotational spring; "
        "runtime/actions/hand_pose.py:551-556 passes them to WristController) · runtime/control.py:51-56 "
        "(error = rotvec(q_cur⁻¹ ⊗ q_tgt) · per-axis clamp ±tau_max) · base.py:27-28 (effort-mode kd = "
        "passive damping) · "
        "sharpa_hands.py:190-197 (gravity_compensation — gravcomp 1.0 on every body) · "
        "pair = scripts/debug/replay.py:11-12"
    ),
)


#: Inertia of the implicit rotational damping = the hand ROOT body's own inertia (fingers excluded, about its own centre
#: of mass). Deadbeat correction with the whole articulation's (composite) inertia over-corrects: the effective inertia
#: a torque on the root moves in one step is about 1/2.3 of it (the fingers are joined by joints and do not follow
#: within one step), so the angular-velocity error grows by (1 − r) per step and diverges. The root's own inertia is a
#: lower bound of the effective inertia, so r ≤ 1 and the error (1 − r) ∈ [0, 1) converges. The load observer uses
#: the same lower-bound inertia; the linear mass rule is MASS_RULE.
DAMP_INERTIA_RULE = "root_body_own"
#: Mass of the linear wrench = the root body's own mass (a lower bound of the effective mass) — the convergence
#: condition of the load observer M/M_true ∈ (0, 1] (a 1-axis toy diverges for M_true < M/2). With loose fingers the
#: effective mass drops toward the root, so the whole hand mass (0.623 kg for wuji2) is not a lower bound. The observer
#: fills the difference to the true mass as load.
MASS_RULE = "root_body_own"


def damping_inertia_w(link_quat_w0, inertia_b0):
    """Inertia of the implicit rotational damping (DAMP_INERTIA_RULE): the root body's own inertia about its centre of
    mass, in world axes (N,3,3). link_quat_w0 (N,4 wxyz) = root link pose · inertia_b0 (N,3,3) = root body
    ``default_inertia`` (about the centre of mass, link frame)."""
    import torch  # noqa: PLC0415

    w, x, y, z = link_quat_w0.unbind(-1)
    R = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )  # (N,3,3)
    return R @ inertia_b0 @ R.transpose(-1, -2)


def _quat_conj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def _quat_mul(a, b):
    import torch  # noqa: PLC0415

    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_rotate(q, v):
    """Rotate v by the wxyz quaternion q (the formula of isaaclab.utils.math.quat_apply)."""
    import torch  # noqa: PLC0415

    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def rotvec_error_body(q_cur, q_tgt):
    """Body-frame rotation error rotvec(q_cur⁻¹ ⊗ q_tgt), short arc (w ≥ 0) — FlashCHORD runtime/control.py:51."""
    import torch  # noqa: PLC0415

    q = _quat_mul(_quat_conj(q_cur), q_tgt)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = torch.linalg.norm(v, dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(s, q[..., :1].clamp(min=0.0))
    # s → 0 limit: 2·v/w (small angle) — avoids 0/0
    scale = torch.where(s > 1e-9, ang / s.clamp(min=1e-12), 2.0 / q[..., :1].clamp(min=1e-12))
    return v * scale


def wrist_wrench(
    pos_w,
    quat_w,
    lin_vel_w,
    ang_vel_w,
    tgt_pos_w,
    tgt_quat_w,
    inertia_w,
    mass,
    dt: float,
    gains: WristGains = WRIST_GAINS,
    d_force=None,
    d_torque=None,
):
    """World-frame wrench (F, τ) on the hand root's CENTRE OF MASS, both (N,3).

    The spring is explicit (the error of this step's pose) and the damping is IMPLICIT (backward-Euler equivalent) —
    the same order as FlashCHORD (MuJoCo implicitfast), which integrates the joint-PD damping and the ball joint's
    passive damping implicitly. PhysX integrates external forces explicitly, so damping applied as kd·v diverges at
    kd·dt/I ≈ 410 (rotation, I ≈ 1.4e-3 kg·m², kd 60). MuJoCo also puts the LOAD (contacts, a held object, finger
    coupling) into the same implicit solve; without that load d (the wrench the controller does not know, measured by
    LoadObserver from the last step's velocity change) the stiffness against load collapses to kp·M/(M + dt·kd).
    linear:  F_s = kp·(p* − p) ;  v⁺ = (M·v + dt·(F_s + d_F))/(M + dt·kd) ;  F = clip(M·(v⁺ − v)/dt − d_F,
    ±F_max)
    angular: τ_s = R · clip(kp · rotvec(q⁻¹ q*), ±τ_max)  (per body axis — FlashCHORD control.py:51-56)
             ω⁺ = (I + dt·kd·1)⁻¹ (I·ω + dt·(τ_s + d_τ)) ;  τ = I·(ω⁺ − ω)/dt − d_τ.
    At static equilibrium (v⁺ = v = 0) F_s + d = 0 → the stiffness against load is the declared kp. M, I = the
    controller inertia the caller passes (MASS_RULE · DAMP_INERTIA_RULE). d = None → 0 · kd → 0 → F = F_s, τ =
    τ_s.
    """
    import torch  # noqa: PLC0415

    g = gains
    m = mass.reshape(-1, 1)
    d_force = torch.zeros_like(pos_w) if d_force is None else torch.clamp(d_force, -g.lin_force_max, g.lin_force_max)
    d_torque = (
        torch.zeros_like(pos_w) if d_torque is None else torch.clamp(d_torque, -g.rot_torque_max, g.rot_torque_max)
    )
    F_s = g.lin_kp * (tgt_pos_w - pos_w)
    v_next = (m * lin_vel_w + dt * (F_s + d_force)) / (m + dt * g.lin_kd)
    F = torch.clamp(m * (v_next - lin_vel_w) / dt - d_force, -g.lin_force_max, g.lin_force_max)
    e_b = rotvec_error_body(quat_w, tgt_quat_w)
    tau_s = _quat_rotate(quat_w, torch.clamp(g.rot_kp * e_b, -g.rot_torque_max, g.rot_torque_max))
    eye = torch.eye(3, dtype=inertia_w.dtype, device=inertia_w.device)
    A = inertia_w + (dt * g.rot_kd) * eye
    rhs = (inertia_w @ ang_vel_w.unsqueeze(-1)).squeeze(-1) + dt * (tau_s + d_torque)
    w_next = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    tau = (inertia_w @ (w_next - ang_vel_w).unsqueeze(-1)).squeeze(-1) / dt - d_torque
    return F, tau


class LoadObserver:
    """Load observer (the wrench d the controller does not know): this step's d from the last step's record (world
    axes, about the centre of mass).

    d_F = M·(v_k − v_{k−1})/dt − F_{k−1} · d_τ = I·(ω_k − ω_{k−1})/dt − τ_{k−1} (F, τ = the wrench
    applied last step). With M, I below the true values (the lower-bound rule) d also fills that difference, so the
    steady state converges
    to the true inertia. A reset env has d = 0 for one step (the reset wrote the velocity — the record is broken)."""

    def __init__(self, num_envs: int, device):
        import torch  # noqa: PLC0415

        z = torch.zeros(num_envs, 3, device=device)
        self.v, self.w, self.F, self.T = z.clone(), z.clone(), z.clone(), z.clone()
        self.ok = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)

    def estimate(self, lin_vel_w, ang_vel_w, inertia_w, mass, dt: float):
        import torch  # noqa: PLC0415

        m = mass.reshape(-1, 1)
        dF = torch.where(self.ok, m * (lin_vel_w - self.v) / dt - self.F, 0.0)
        dT = torch.where(self.ok, (inertia_w @ (ang_vel_w - self.w).unsqueeze(-1)).squeeze(-1) / dt - self.T, 0.0)
        return dF, dT

    def commit(self, lin_vel_w, ang_vel_w, F, tau) -> None:
        self.v, self.w, self.F, self.T = lin_vel_w.clone(), ang_vel_w.clone(), F.clone(), tau.clone()
        self.ok[:] = True

    def reset(self, env_ids) -> None:
        self.ok[env_ids] = False


def wrench_to_link_frame(F_w, tau_c_w, link_quat_w):
    """Rotate the world-axis (F, τ_c) into THIS step's root link axes (wxyz) — the wrench is applied with
    is_global=False.

    Isaac Lab 0.53.1's ``permanent_wrench_composer.set_forces_and_torques(is_global=True)`` CACHES the link pose it
    uses for the world → link transform (``wrench_composer.py:292-297`` — the cache clears only in ``reset()``, which
    ``Articulation.reset`` calls at env reset). Passing world axes would turn the wrench by R_now·R_cached⁻¹ as the
    hand rotates. Rotating here with the current pose and passing link-axis values bypasses the cache (the kernels
    ``cast_*_to_link_frame`` keep the values as given when is_global=False — the path the object-gravity code uses)."""
    qi = _quat_conj(link_quat_w)
    return _quat_rotate(qi, F_w), _quat_rotate(qi, tau_c_w)
