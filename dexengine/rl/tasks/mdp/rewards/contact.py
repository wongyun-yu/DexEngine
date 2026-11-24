# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 contact-aware lag compensation MDP terms.

Three classes, all gated by training-time flags so default-off behavior
is byte-identical to the pre-Phase-1 code:

  - ``LookaheadWristTarget`` (obs):
      adds the reference wrist target at frame [t + lookahead_k] for each
      configured lookahead horizon. Pure obs augmentation; ArmIK keeps
      tracking the t-th target. Frame: WORLD frame (env-local delta
      relative to the current-t target, so obs is shift-invariant).

  - ``tip_to_surface_dist (module fn)`` (obs):
      10D (5 right + 5 left fingertips) signed distance to the env's
      ``object`` rigid body surface. Frame: WORLD throughout (tip pos
      world, object surface_points transformed by object pose).

  - ``HandContactMismatchSustained`` (termination):
      counts consecutive frames where (NEW per-finger annotation marks any-tip
      contact) AND (robot has zero OBJECT-filtered contact force) for one hand.
      Terminates when counter ≥ grace_steps (default 50 = 1s at 50 Hz policy
      rate, exposed via ``--contact_term_grace_steps``). Robot contact uses
      ``force_matrix_w`` (object-filtered) > force_threshold.

Coordinate frames (BLOCKING — every position input is annotated):
  - Robot tip pos: fingertip SITE = body_link_pos_w + R(body_link_quat_w)·offset
    (WORLD; offset from naming.resolve_fingertip_site_offsets — SharpA
    ~2.6 cm DP-local +x to the physical pad tip, all-zero = body origin for
    RH56F1, byte-identical legacy path)
  - Object surface points: cached object-local, transformed to WORLD
    via current obj.data.root_pose_w
  - Reference wrist pos/quat: from trajectory_manager.get_wrist_{pos,quat}_w
    (WORLD, after Rz(-90°) + traj_offset_z per the wrist verification note)
  - All distance computations done in WORLD frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from dexengine.rl.tasks.mdp.rewards.contact_match import (
    ContactMatchBase,
)

# ============================================================================
# Helper: resolve fingertip body names + sensor indices on a hand articulation.
# ============================================================================

# Fingertip body names come from naming.resolve_tip_body_names (spec-based); per-hand name tables here failed for any
# hand missing from them, so they are gone.


# ----------------------------------------------------------------------------
# Fingertip SITE positions (tip-point fix).
# SharpA tip bodies are the *_DP distal phalanges whose ORIGINS sit ~2.6 cm
# PROXIMAL of the physical fingertip (URDF *_fingertip frames: thumb +0.027611,
# fingers +0.026009 along DP-local +x). Every LIVE object-distance quantity in
# this module measures from the tip SITE = body origin + R_body·offset so that
# exp(-k·d) rewards reach ~1.0 at physical touch and d_cm diagnostics read
# pad-true. RH56F1 (and dataset hands) resolve all-zero offsets and take the
# early-return path below — BYTE-IDENTICAL to the historical body-origin
# behavior (no quat math executed). Fingertip-TRACKING rewards/terminations
# (taco_train/mdp/rewards.py, FingertipTrackingExceeded) are deliberately NOT
# routed through this: they compare live DP-origin vs dataset DP-origin
# (like-for-like) and must stay consistent.
# ----------------------------------------------------------------------------


# ============================================================================
# OBS: Lookahead wrist target
# ============================================================================


# ============================================================================
# OBS: Bimanual per-fingertip distance to object surface
# ============================================================================


# ============================================================================
# TERMINATION: hand-level contact mismatch sustained >= grace_steps
# (NEW per-finger annotation + object-filtered contact force)
# ============================================================================


# NOTE: the legacy nearest-surface ``RewardContactSoft`` was REMOVED here
# (dead code). It was unreachable: the CLI routes
# ``--contact_reward_mode soft`` to RewardGraspTargetSoft unconditionally and
# nothing else imported/instantiated it (verified by grep). It was
# region-agnostic (rewarded approaching ANY surface) and reward-hackable.


# ============================================================================
# Wrap-ready pose + object-relative grasp geometry.
# Reuse ContactMatchBase helpers (_robot_tip_dist, _ref_for_frame, _will_contact,
# _robot_contact, _tip_body_ids). Gated on the REFERENCE contact label (robust;
# no dependence on three_phase metadata). Tip order = [thumb, index, middle, ring, little].
# ============================================================================


class RewardForceDirection(ContactMatchBase):
    """Force-direction reward — hand-applied object force aligned with the
    **force the reference motion requires** (a_ref + g), during the post-contact grasp/lift window,
    soft-scaled by thumb+index contact (anti one-finger poke).

    reward(N,) = clip(0.5 + 0.5*proj/F_cap, 0, 1) * gate2 * 1[window] * 1[reference travel>eps]
    proj = F_hand_on_obj · F̂_req,  F_req/m = a_ref + g·ẑ (the mass cancels). All quantities from ``_force_quantities``
    (shared,
    memoized). The RewTerm weight (force_reward_weight) scales it. Targets the
    observed failure: the hand presses DOWN (opposing the lift) instead of applying
    force along the desired object motion. mg is NOT in the reward (logged as diag)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
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
        force_cap: float = 30.0,
        force_k: int = 3,
        force_motion_eps: float = 0.003,
        force_window_pre: int = 3,
        force_window_post: int = 50,
    ) -> torch.Tensor:
        q = self._force_quantities(env)
        if not self._sanity_done:
            print(
                f"[RewardForceDirection][sanity] side={self._side} F_cap={self._force_cap} "
                f"k={self._force_k} window=[fc-{self._force_pre},fc+{self._force_post}] "
                f"eps={self._force_eps} env0: proj={q['proj'][0].item():.2f}N Fz={q['Fz'][0].item():.2f}N "
                f"gate2={q['gate2'][0].item():.2f} window={bool(q['window_ok'][0])} "
                f"motion={bool(q['motion_ok'][0])} reward={q['reward'][0].item():.3f}"
            )
            self._sanity_done = True
        return q["reward"]


# Force-direction DIAGNOSTICS (logged at weight 1e-9; recover ×1e9). Share the
# memoized _force_quantities, so they add no extra force/ref computation per step.
class _ForceDiagBase(ContactMatchBase):
    """Force diagnostics: one __call__ (the term parameters) on _force_quantities; subclasses pick the value."""

    def __call__(
        self,
        env,
        side="right",
        contact_source="mano",
        seq_dir="",
        mesh_dir="",
        F_threshold=0.5,
        d_threshold=0.01,
        object_key="object",
        force_cap=30.0,
        force_k=3,
        force_motion_eps=0.003,
        force_window_pre=3,
        force_window_post=50,
    ):
        return self._diag(self._force_quantities(env))


class RewardDiagForceProj(_ForceDiagBase):
    """Signed projection F_hand·d̂_ref (Newtons) — only meaningful in the active window."""

    def _diag(self, q):
        return torch.where(q["window_ok"] & q["motion_ok"], q["proj"], torch.zeros_like(q["proj"]))


class RewardDiagForceZ(_ForceDiagBase):
    """Hand-on-object Fz (Newtons; >0 = upward/support). Active-window only."""

    def _diag(self, q):
        return torch.where(q["window_ok"] & q["motion_ok"], q["Fz"], torch.zeros_like(q["Fz"]))


class RewardDiagForcePos(_ForceDiagBase):
    """1 if (proj>0 AND active) — episode-mean ≈ %frames the force helps the desired motion."""

    def _diag(self, q):
        return ((q["proj"] > 0) & q["active"]).float()


class RewardDiagForceActive(_ForceDiagBase):
    """1 if the force-reward gate is active (window AND motion AND ≥1 finger). Active ratio."""

    def _diag(self, q):
        return q["active"].float()


class RewardDiagTIBoth(ContactMatchBase):
    """1 if thumb AND index both have object contact — thumb+index simultaneous rate."""

    def __call__(
        self,
        env,
        side="right",
        contact_source="mano",
        seq_dir="",
        mesh_dir="",
        F_threshold=0.5,
        d_threshold=0.01,
        object_key="object",
        force_cap=30.0,
        force_k=3,
        force_motion_eps=0.003,
        force_window_pre=3,
        force_window_post=50,
    ):
        return self._force_quantities(env)["ti_both"]


class RewardContactPenalty(ContactMatchBase):
    """PENALTY-form contact reward (contact pilot, Batch 2).

    Same binary robot-contact gate as RewardContactMatch, but PENALIZES every
    reference-contact finger that is NOT touching:
        reward(N,) = - sum over fingers of (ref AND NOT robot_contact)  in [-5, 0].
    Gives a non-zero (negative) gradient even when robot contact is ~0 — each
    missed reference-contact finger costs -1 (×weight) — whereas the positive
    match reward is inert at zero contact. The per-finger reference label is the
    corrected fingertip-region annotation (not the old degenerate broadcast bit).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
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
    ) -> torch.Tensor:
        ref, robot = self._ref_and_robot(env)  # (N,5),(N,5) bool
        miss = ref & ~robot  # (N,5)
        reward = -miss.float().sum(dim=-1)  # (N,) in [-5,0]
        if not self._sanity_done:
            print(
                f"[RewardContactPenalty][sanity] side={self._side} "
                f"source={self._contact_source} env0: "
                f"ref={ref[0].cpu().tolist()} robot={robot[0].cpu().tolist()} "
                f"miss={miss[0].cpu().tolist()} reward={reward[0].item():.2f}"
            )
            self._sanity_done = True
        return reward


# --- Contact diagnostic terms (logged via Info/Episode_Reward/<name>) --------
# Each returns a (N,) scalar metric. The cfg-override registers them with a
# NEGLIGIBLE non-zero weight (~1e-9): Isaac Lab's RewardManager.compute() SKIPS
# terms with weight==0 (their func is never called and they log 0), so a literal
# weight=0 would not log. The ~1e-9 weight makes them execute + log while their
# contribution to the total reward is numerically zero. The logged
# Episode_Reward value is scaled by that weight (multiply back by 1/weight to
# read the true metric). See dexengine/rl/eval/cfg_overrides.py and
# dexengine/rl/train/runner.py.


class RewardDiagContactMatchRate(ContactMatchBase):
    """Mean over the 5 fingers of (ref == robot) — per-finger agreement."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        ref, robot = self._ref_and_robot(env)
        return (ref == robot).float().mean(dim=-1)  # (N,)


class RewardDiagContactPrecision(ContactMatchBase):
    """sum(ref & robot) / max(sum(robot), 1) — of fingers the robot touches,
    fraction that should be touching."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        ref, robot = self._ref_and_robot(env)
        tp = (ref & robot).float().sum(dim=-1)
        denom = robot.float().sum(dim=-1).clamp(min=1.0)
        return tp / denom  # (N,)


class RewardDiagContactRecall(ContactMatchBase):
    """sum(ref & robot) / max(sum(ref), 1) — of fingers that should touch,
    fraction the robot actually touches."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        ref, robot = self._ref_and_robot(env)
        tp = (ref & robot).float().sum(dim=-1)
        denom = ref.float().sum(dim=-1).clamp(min=1.0)
        return tp / denom  # (N,)


class RewardDiagRefContactFrac(ContactMatchBase):
    """Mean reference contact (fraction of fingers the annotation marks touching)."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        ref, _ = self._ref_and_robot(env)
        return ref.float().mean(dim=-1)  # (N,)


class RewardDiagRobotContactFrac(ContactMatchBase):
    """Mean robot contact (fraction of fingers actually touching)."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        _, robot = self._ref_and_robot(env)
        return robot.float().mean(dim=-1)  # (N,)


class RewardDiagThumbContact(ContactMatchBase):
    """Per-env thumb OBJECT-contact (0/1) — episode-mean = thumb contact rate."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        return self._robot_contact(env)[:, 0].float()  # (N,) thumb


class RewardDiagIndexContact(ContactMatchBase):
    """Per-env index OBJECT-contact (0/1) — episode-mean = index contact rate."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        return self._robot_contact(env)[:, 1].float()  # (N,) index


class RewardDiagGraspTargetDist(ContactMatchBase):
    """Mean GT-target distance (meters) over will-contact fingers. Lower = closer
    to the correct grasp points; the episode-summed trend should DECREASE as the
    new GT-target rewards pull the fingers in."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        mask = self._will_contact_mask(env).float()  # (N,5)
        d = self._robot_tip_to_grasp_target_dist(env)  # (N,5) meters
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return (mask * d).sum(dim=-1) / denom  # (N,) mean over will-contact


# ============================================================================
# Closure bootstrap A/B: force-onset vs closing-potential.
# Both reuse ContactMatchBase machinery (_robot_obj_force_mag = per-fingertip
# OBJECT-filtered force magnitudes, _robot_tip_dist = tip→object-surface
# distance, _ref_for_frame = reference-contact gating). Finger order
# [thumb, index, middle, ring, little].
# ============================================================================


# ============================================================================
# CURRICULUM: linear fingertip-hinge schedule (contact pilot --hinge_schedule)
# ----------------------------------------------------------------------------
# Minimal but functional: linearly ramps the fingertip tip-reward
# `hinge_threshold_m` param from 0.0 -> target over the first `ramp_fraction`
# of `max_iterations` (measured by env.common_step_counter). Mutates the live
# reward-term cfg params via env.reward_manager so the change takes effect on
# the next reward evaluation. Default-OFF (only registered when --hinge_schedule
# 1 is passed in the cfg-override).
# ============================================================================
