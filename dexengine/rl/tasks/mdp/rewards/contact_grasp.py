"""Approach, pre-grasp and grasp-shape rewards with their diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from dexengine.rl.tasks.mdp.rewards.contact_match import ContactMatchBase


class RewardGraspTargetSoft(ContactMatchBase):
    """Ref-gated SOFT contact reward on the witness_obj GT-target distance.

    Unlike the removed nearest-surface reward, ``d`` is the robot fingertip -> per-finger GT
    contact target distance (``_robot_tip_to_grasp_target_dist``), NOT the
    nearest-surface distance. The GT target is the specific grasp point on the
    LIVE object (object-relative), so this rewards approaching the CORRECT region
    rather than any nearby surface. Active only on annotated contact frames.

    reward(N,) = sum over ref-contact fingers of exp(-k * d_target)  in [0, 5].
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._k = float(cfg.params.get("softness_k", 15.0))
        self._sanity_done = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        softness_k: float = 15.0,
    ) -> torch.Tensor:
        ref = self._ref_for_frame(env)  # (N,5) bool
        d = self._robot_tip_to_grasp_target_dist(env)  # (N,5) GT-target meters
        prox = torch.exp(-self._k * d)
        reward = (ref.float() * prox).sum(dim=-1)  # (N,) in [0,5]
        if not self._sanity_done:
            print(
                f"[RewardGraspTargetSoft][sanity] side={self._side} k={self._k:.1f} env0: "
                f"ref={ref[0].cpu().tolist()} "
                f"d_cm={[round(x * 100, 2) for x in d[0].cpu().tolist()]} "
                f"reward={reward[0].item():.3f}"
            )
            if self._tip_offsets is not None:  # tip-site fix: show pre-fix DP-origin d for comparison
                hand = env.scene[self._hand_key]
                d_o = self._robot_tip_to_grasp_target_dist(
                    env, tips_w=hand.data.body_link_pos_w[:, self._tip_body_ids, :]
                )
                print(
                    f"  [tip-site fix] DP-origin d_cm={[round(x * 100, 2) for x in d_o[0].cpu().tolist()]} "
                    f"-> tip-site d_cm={[round(x * 100, 2) for x in d[0].cpu().tolist()]} "
                    f"(per-finger |delta| <= offset norm {['2.76', '2.60', '2.60', '2.60', '2.60']} cm)"
                )
            self._sanity_done = True
        return reward


class RewardApproachSoft(ContactMatchBase):
    """Dense (NOT ref-gated) approach reward on the witness_obj GT-target distance.

    For every finger that contacts the object at SOME point in the clip
    (``_will_contact_mask``), reward ``exp(-k * d_target)`` EVERY frame — so the
    hand is shaped toward a contact-ready configuration during the approach,
    before the annotated contact frames begin. Uses the same GT target as
    RewardGraspTargetSoft (object-relative grasp point), never nearest-surface.

    reward(N,) = sum over will-contact fingers of exp(-k * d_target)  in [0, 5].
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._k = float(cfg.params.get("softness_k", 12.0))
        self._sanity_done = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        softness_k: float = 12.0,
    ) -> torch.Tensor:
        mask = self._will_contact_mask(env)  # (N,5) bool (static per clip)
        d = self._robot_tip_to_grasp_target_dist(env)  # (N,5) GT-target meters
        prox = torch.exp(-self._k * d)
        reward = (mask.float() * prox).sum(dim=-1)  # (N,) in [0,5]
        if not self._sanity_done:
            print(
                f"[RewardApproachSoft][sanity] side={self._side} k={self._k:.1f} env0: "
                f"will_contact={mask[0].cpu().tolist()} "
                f"d_cm={[round(x * 100, 2) for x in d[0].cpu().tolist()]} "
                f"reward={reward[0].item():.3f}"
            )
            if self._tip_offsets is not None:  # tip-site fix: show pre-fix DP-origin d for comparison
                hand = env.scene[self._hand_key]
                d_o = self._robot_tip_to_grasp_target_dist(
                    env, tips_w=hand.data.body_link_pos_w[:, self._tip_body_ids, :]
                )
                print(
                    f"  [tip-site fix] DP-origin d_cm={[round(x * 100, 2) for x in d_o[0].cpu().tolist()]} "
                    f"-> tip-site d_cm={[round(x * 100, 2) for x in d[0].cpu().tolist()]} "
                    f"(per-finger |delta| <= offset norm {['2.76', '2.60', '2.60', '2.60', '2.60']} cm)"
                )
            self._sanity_done = True
        return reward


class RewardPreGraspPalm(ContactMatchBase):
    """PRE-GRASP reward: pull the PALM (``{side}_wrist_rot_z``) to within
    finger reach of the LIVE grasp centroid (mean of the will-contact ``witness_obj``
    GT targets, re-anchored to the current object pose), so the fingers CAN wrap.

    Motivation (grasp-wall probe): the wrist residual is already active
    (scale 2.0, ‖mu‖wrist growing) but the reward landscape steers it to track the
    morphology-mismatched MANO reference (RH56F1 ~2.2x MANO), holding the hand in a
    non-graspable downward-poke pose. This term gives the wrist a *reason* to deviate
    toward a graspable configuration. The existing force-direction reward then has
    real contact frames to align on.

    One-sided so it never incentivises penetration:
        reward(N,) = exp(-k * relu(d_palm - d_grasp))  in (0, 1].
    d_palm = ‖palm - grasp_centroid‖; flat (=1) once palm is within ``d_grasp`` of the
    centroid (≈ finger reach), gradient only while too far. Dense (will-contact clips).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._k = float(cfg.params.get("pregrasp_k", 8.0))
        self._d_grasp = float(cfg.params.get("pregrasp_dist", 0.12))
        self._palm_id = None
        self._sanity_done = False

    def _grasp_centroid_w(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """(N,3) world-frame mean of the will-contact witness GT targets on the
        LIVE object (same anchoring as ``_robot_tip_to_grasp_target_dist``)."""
        return grasp_centroid_world(self, env)

    def _palm_pos_w(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return _palm_point_w(self, env.scene[self._hand_key])  # (N,3)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        pregrasp_k: float = 8.0,
        pregrasp_dist: float = 0.12,
    ) -> torch.Tensor:
        d_palm = torch.norm(self._palm_pos_w(env) - self._grasp_centroid_w(env), dim=-1)
        reward = torch.exp(-self._k * torch.clamp(d_palm - self._d_grasp, min=0.0))
        if not self._sanity_done:
            print(
                f"[RewardPreGraspPalm][sanity] side={self._side} k={self._k} "
                f"d_grasp={self._d_grasp:.3f} env0: d_palm_cm={d_palm[0].item() * 100:.2f} "
                f"reward={reward[0].item():.3f}"
            )
            self._sanity_done = True
        return reward


def _palm_point_w(term, hand) -> torch.Tensor:
    """The point (N,3) the palm rewards measure: the palm body origin, or on the floating hand with
    DEXENGINE_FH_PALM_POINT=arm_ee the same physical point as the arm body (offset from the plugin). Prints
    [FH_PALM_POINT] once when the offset is on."""
    from isaaclab.utils.math import quat_apply

    from dexengine.rl.tasks import hooks
    from dexengine.rl.tasks.mdp.naming import resolve_palm_body_id

    if term._palm_id is None:
        term._palm_id = resolve_palm_body_id(hand, term._side)
        off = hooks.floating_palm_point(hand, term._side)
        term._palm_off = (
            None if off is None else torch.tensor(off, dtype=torch.float32, device=hand.data.body_link_pos_w.device)
        )
        if off is not None:
            print(
                f"[FH_PALM_POINT] {type(term).__name__} side={term._side} body={hand.data.body_names[term._palm_id]} "
                f"mode=arm_ee offset_palm_frame=({off[0]:.5f},{off[1]:.5f},{off[2]:.5f}) m",
                flush=True,
            )
    p = hand.data.body_link_pos_w[:, term._palm_id, :]
    if getattr(term, "_palm_off", None) is None:
        return p
    q = hand.data.body_link_quat_w[:, term._palm_id, :]
    return p + quat_apply(q, term._palm_off.expand(p.shape[0], 3))


class RewardDiagPalmDist(ContactMatchBase):
    """Diag: mean palm -> grasp-centroid distance (m). Should DECREASE toward
    d_grasp as the pre-grasp reward pulls the wrist into a graspable pose."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._palm_id = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        self._ensure_grasp_targets(env)
        hand = env.scene[self._hand_key]
        obj = env.scene[self._object_key]
        clip = self._traj_mgr.clip_ids
        c_local = self._grasp_targets[clip]
        mask = self._will_contact[clip].float()
        N = c_local.shape[0]
        q = obj.data.root_quat_w.unsqueeze(1).expand(N, 5, 4).reshape(-1, 4)
        t_world = quat_apply(q, c_local.reshape(-1, 3)).reshape(N, 5, 3) + obj.data.root_pos_w.unsqueeze(1)
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        centroid = (t_world * mask.unsqueeze(-1)).sum(dim=1) / denom
        return torch.norm(_palm_point_w(self, hand) - centroid, dim=-1)


def grasp_centroid_world(term: ContactMatchBase, env) -> torch.Tensor:
    """(N,3) world centroid of the will-contact witness GT targets on the LIVE
    object (same anchoring as RewardPreGraspPalm._grasp_centroid_w)."""
    from isaaclab.utils.math import quat_apply

    term._ensure_grasp_targets(env)
    obj = env.scene[term._object_key]
    clip = term._traj_mgr.clip_ids
    c_local = term._grasp_targets[clip]  # (N,5,3) object-local
    mask = term._will_contact[clip].float()  # (N,5)
    N = c_local.shape[0]
    q = obj.data.root_quat_w.unsqueeze(1).expand(N, 5, 4).reshape(-1, 4)
    t_world = quat_apply(q, c_local.reshape(-1, 3)).reshape(N, 5, 3) + obj.data.root_pos_w.unsqueeze(1)
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (t_world * mask.unsqueeze(-1)).sum(dim=1) / denom  # (N,3)


def safe_normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class RewardSlipPenalty(ContactMatchBase):
    """P1 pose-agnostic grasp-stability: penalize object-hand relative
    linear velocity (SLIP) while the hand is actually holding the object.

        slip(N,)   = || v_obj - v_palm ||                       (m/s, world frame)
        holding(N) = (#fingertips with object-filtered force >= slip_f_gate) >= slip_min_fingers
        reward(N,) = -min(slip, slip_cap)/slip_cap  where holding, else 0.   in [-1, 0]

    Rationale (unimanual knife diagnosis): the only pose-agnostic grasp
    reward (closure_force) is reference-window-gated and cap-saturated — nothing pays
    for KEEPING a stable hold. This term rewards a slip-free grip regardless of finger
    pose, grasp location, or reference timeline; approach/release phases (no contact)
    are unaffected by construction (holding gate). State-based, no MANO annotation use.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._palm_id = None
        self._sanity_done = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        slip_cap: float = 0.5,
        slip_f_gate: float = 0.05,
        slip_min_fingers: int = 2,
    ) -> torch.Tensor:
        from dexengine.rl.tasks.mdp.naming import resolve_palm_body_id

        hand = env.scene[self._hand_key]
        if self._palm_id is None:
            self._palm_id = resolve_palm_body_id(hand, self._side)
        obj = env.scene[self._object_key]
        F = self._robot_obj_force_mag(env)  # (N,5) Newtons
        holding = (float(slip_f_gate) <= F).sum(dim=-1) >= int(slip_min_fingers)  # (N,)
        v_palm = hand.data.body_link_lin_vel_w[:, self._palm_id, :]  # (N,3)
        v_obj = obj.data.root_lin_vel_w  # (N,3)
        slip = torch.norm(v_obj - v_palm, dim=-1)  # (N,)
        pen = -torch.clamp(slip, max=float(slip_cap)) / float(slip_cap)
        out = torch.where(holding, pen, torch.zeros_like(pen))
        if not self._sanity_done:
            print(
                f"[RewardSlipPenalty][sanity] side={self._side} cap={slip_cap}m/s "
                f"f_gate={slip_f_gate}N min_fingers={slip_min_fingers} env0: "
                f"holding={bool(holding[0])} slip={slip[0].item():.4f}m/s "
                f"reward={out[0].item():.3f}"
            )
            self._sanity_done = True
        return out
