"""Closure-force rewards (force and potential), their hinge threshold schedule and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, RewardTermCfg

from dexengine.rl.swallow import note_once

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from dexengine.rl.tasks.mdp.rewards.contact_match import ContactMatchBase


class RewardClosureForce(ContactMatchBase):
    """FORCE-ONSET reward: pay for ANY per-fingertip object force
    the moment it exists. NO robot-contact gate — that is the point: force_dir's
    thumb+index gate never opens for middle/ring/little, so no existing term has
    gradient that pays those fingers to press.

        reward(N,) = (Σ_f w_f·min(F_f, cap)/cap) / Σ_f w_f
        w_f = nonthumb_scale for index..little, 1.0 for thumb;
        active on reference-contact frames (``_ref_for_frame(env).any(-1)``).

    F_f = object-filtered per-fingertip force magnitude (``_robot_obj_force_mag``,
    desk/self contact excluded). ``cap`` bounds the per-finger payout so slamming
    saturates instead of escalating (anti-slam; obj_vel watched in the audit).
    Range [0, 1] before the RewTerm weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._cap = float(cfg.params.get("cap", 5.0))
        self._nonthumb = float(cfg.params.get("nonthumb_scale", 1.5))
        # A1 per-finger mode (default OFF = legacy hand-level any() gate, baseline-reproducible).
        self._perfinger = bool(int(cfg.params.get("perfinger", 0)))
        self._f_contact = float(cfg.params.get("contact_f_min", 0.05))
        self._margins = torch.tensor(
            list(cfg.params.get("margins", [2, 5, 5, 5, 5])), device=env.device, dtype=torch.long
        )  # [thumb,index,middle,ring,pinky]
        # B2 (immediate force): drop the af>=ref_start[f] lower bound — pay force as soon as object-contact
        # force exists within [.., ref_end[f]-margin_f] for will-contact fingers. Default 0 = A1 (ref_start gate).
        self._force_immediate = bool(int(cfg.params.get("force_immediate", 0)))
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
        cap: float = 5.0,
        nonthumb_scale: float = 1.5,
        perfinger: int = 0,
        contact_f_min: float = 0.05,
        margins=(2, 5, 5, 5, 5),
        force_immediate: int = 0,
    ) -> torch.Tensor:
        F = self._robot_obj_force_mag(env)  # (N,5) Newtons
        w = torch.ones_like(F)
        w[:, 1:] = self._nonthumb  # thumb 1.0, fingers 1-4 scaled
        if not self._perfinger:
            # Legacy hand-level any() gate (baseline).
            r = (w * F.clamp(max=self._cap)).sum(dim=-1) / (self._cap * w.sum(dim=-1))
            active = self._ref_for_frame(env).any(dim=-1).float()
            if not self._sanity_done:
                print(
                    f"[RewardClosureForce][sanity legacy] cap={self._cap}N nonthumb={self._nonthumb} "
                    f"env0: F_N={[round(x, 3) for x in F[0].cpu().tolist()]} r={r[0].item():.4f}"
                )
                self._sanity_done = True
            return r * active
        # A1 PER-FINGER: force only on [ref_start[f], ref_end[f]-margin_f] and only once contacting.
        af, will, gfirst, rstart, hi = self._perfinger_windows(env, self._margins)
        if self._force_immediate:
            active_force = will & (af <= hi) & (self._f_contact <= F)  # B2: no ref_start gate
        else:
            active_force = will & (af >= rstart) & (af <= hi) & (self._f_contact <= F)  # (N,5) bool
        force_raw = F.clamp(max=self._cap) / self._cap  # (N,5) in [0,1]
        num = (active_force.float() * w * force_raw).sum(dim=-1)
        den = (active_force.float() * w).sum(dim=-1).clamp(min=1e-6)
        r = num / den  # normalize by ACTIVE weight
        env._closure_force_active = active_force  # diag stash
        if not self._sanity_done:
            print(
                f"[RewardClosureForce][sanity A1] cap={self._cap}N nonthumb={self._nonthumb} "
                f"margins={self._margins.tolist()} force_immediate={int(self._force_immediate)} env0: "
                f"F_N={[round(x, 3) for x in F[0].cpu().tolist()]} "
                f"active_force={active_force[0].int().cpu().tolist()} r={r[0].item():.4f}"
            )
            self._sanity_done = True
        return r


class RewardClosurePotential(ContactMatchBase):
    """CLOSING-POTENTIAL reward: potential-difference on the
    tip→object-surface distance for NON-contacting fingers — dense gradient from
    any distance, hover-proof (holding position yields 0, retreating costs).

        reward(N,) = (Σ_f m_f·clamp(d_f(t-1)−d_f(t), −c, +c)/c) / Σ_f m_f
        m_f = (F_f < contact_f_min)  non-contact mask (force terms take over at
              contact — no reward for already-pressing fingers);
        c   = clamp_cm/100 m; active on reference-contact frames; emits 0 when
              ALL fingers contact (Σm_f = 0) and on the first reward step after
              a reset (prev-dist stale).

    Stateful: per-env (N,5) previous-distance buffer, seeded on first call.
    Reset handling: IsaacLab step order is elen += 1 → terminations → rewards →
    _reset_idx (elen = 0), so the first reward call of a NEW episode sees
    elen == 1 with prev from the PREVIOUS episode. Mask elen <= 1: emit 0 there
    and re-seed prev with the current distance (elen == 0 never occurs at reward
    time but is kept for safety — same convention family as
    HandContactMismatchSustained's counter reset on episode_length_buf == 0).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._clamp_m = float(cfg.params.get("clamp_cm", 1.0)) / 100.0
        self._f_contact = float(cfg.params.get("contact_f_min", 0.05))
        self._nonthumb = float(cfg.params.get("nonthumb_scale", 1.5))
        # A1 per-finger mode (default OFF = legacy hand-level any() gate + nearest-surface distance).
        self._perfinger = bool(int(cfg.params.get("perfinger", 0)))
        self._margins = torch.tensor(
            list(cfg.params.get("margins", [2, 5, 5, 5, 5])), device=env.device, dtype=torch.long
        )
        # B1 (distance metric for the A1 per-finger potential): 'gt_target' (default, A1) = tip->per-finger GT
        # contact target; 'nearest_surface' = tip->nearest object-surface point (legacy distance logic).
        self._pot_dist_mode = str(cfg.params.get("pot_dist_mode", "gt_target"))
        self._prev_d: torch.Tensor | None = None  # (N,5) m
        self._prev_active: torch.Tensor | None = None  # (N,5) bool: pot-active last step (window-entry reseed)
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
        clamp_cm: float = 1.0,
        contact_f_min: float = 0.05,
        perfinger: int = 0,
        nonthumb_scale: float = 1.5,
        margins=(2, 5, 5, 5, 5),
        pot_dist_mode: str = "gt_target",
    ) -> torch.Tensor:
        if not self._perfinger:
            # Legacy hand-level any() gate + nearest-surface distance (baseline).
            d = self._robot_tip_dist(env)
            if self._prev_d is None:
                self._prev_d = d.clone()
            fresh = env.episode_length_buf <= 1
            prev = torch.where(fresh.unsqueeze(-1), d, self._prev_d)
            delta = (prev - d).clamp(-self._clamp_m, self._clamp_m) / self._clamp_m
            m = (self._robot_obj_force_mag(env) < self._f_contact).float()
            msum = m.sum(dim=-1)
            r = (delta * m).sum(dim=-1) / msum.clamp(min=1.0)
            r = torch.where(msum > 0, r, torch.zeros_like(r))
            active = self._ref_for_frame(env).any(dim=-1)
            r = torch.where(fresh, torch.zeros_like(r), r * active.float())
            self._prev_d = d.clone()
            if not self._sanity_done:
                print(
                    f"[RewardClosurePotential][sanity legacy] clamp={self._clamp_m * 100:.1f}cm "
                    f"env0 d_cm={[round(x * 100, 2) for x in d[0].cpu().tolist()]} r={r[0].item():.4f}"
                )
                self._sanity_done = True
            return r
        # A1 PER-FINGER: approach toward the GT contact target (NOT nearest-surface), per-finger window,
        # non-contacting fingers only. Reseed prev_d on any pot-active 0->1 transition (window entry /
        # force->potential return / reset) so the first step in-window emits 0 (no spurious delta).
        if self._pot_dist_mode == "nearest_surface":
            d = self._robot_tip_dist(env)  # B1: tip->nearest object-surface (N,5) m
        else:
            d = self._robot_tip_to_grasp_target_dist(env)  # (N,5) m, GT-target distance
        F = self._robot_obj_force_mag(env)  # (N,5) N
        af, will, gfirst, rstart, hi = self._perfinger_windows(env, self._margins)
        in_win = (af >= gfirst) & (af <= hi)  # (N,5)
        active = will & in_win & (self._f_contact > F)  # (N,5) bool
        if self._prev_d is None:
            self._prev_d = d.clone()
        if self._prev_active is None:
            self._prev_active = torch.zeros_like(active)
        fresh = (env.episode_length_buf <= 1).unsqueeze(-1)  # (N,1)
        newly = active & (~self._prev_active | fresh)  # reseed mask
        prev = torch.where(newly, d, self._prev_d)
        delta = (prev - d).clamp(-self._clamp_m, self._clamp_m) / self._clamp_m  # (N,5) in [-1,1]
        w = torch.ones_like(d)
        w[:, 1:] = self._nonthumb
        af_w = active.float() * w
        r = (af_w * delta).sum(dim=-1) / af_w.sum(dim=-1).clamp(min=1e-6)  # normalize by ACTIVE weight
        self._prev_d = d.clone()
        self._prev_active = active.clone()
        env._closure_pot_active = active  # diag stash
        env._closure_pot_d = d
        if not self._sanity_done:
            print(
                f"[RewardClosurePotential][sanity A1] clamp={self._clamp_m * 100:.1f}cm "
                f"f_contact={self._f_contact}N margins={self._margins.tolist()} "
                f"pot_dist_mode={self._pot_dist_mode} env0: "
                f"d_cm={[round(x * 100, 2) for x in d[0].cpu().tolist()]} "
                f"active_pot={active[0].int().cpu().tolist()} r={r[0].item():.4f}"
            )
            self._sanity_done = True
        return r


class HingeThresholdSchedule(ManagerTermBase):
    """Linearly ramp fingertip hinge_threshold_m 0 -> target over a window.

    params:
        tip_terms: list[str] of reward-term names to update (the 5 tips).
        target_m:  final hinge threshold in metres.
        ramp_steps: number of common_step_counter steps to reach target.
    Returns the current threshold (m) for TensorBoard/W&B logging.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._announced = False

    def __call__(
        self, env: ManagerBasedRLEnv, env_ids, tip_terms=(), target_m: float = 0.015, ramp_steps: int = 1
    ) -> float:
        step = env.common_step_counter
        frac = min(step / max(ramp_steps, 1), 1.0)
        cur = frac * target_m
        rm = env.reward_manager
        for name in tip_terms:
            try:
                term_cfg = rm.get_term_cfg(name)
            except ValueError as _exc:
                note_once("CLOSURE_HINGE_RAMP term", _exc)  # RewardManager.get_term_cfg: ValueError
                continue
            term_cfg.params["hinge_threshold_m"] = cur
        if not self._announced:
            print(
                f"[HingeThresholdSchedule] ramping {list(tip_terms)} "
                f"hinge_threshold_m 0 -> {target_m:.4f} m over {ramp_steps} steps"
            )
            self._announced = True
        return cur


# Closure-force DIAGNOSTICS (×1e-9 family — stripped unless
# --enable_reward_diagnostics 1). Answer "how hard / how saturated is the grip":
# per-step object-filtered per-fingertip force stats. Recovery: ×1e9, then the
# usual Episode_Reward/dt/eplen per-step normalization.
class RewardDiagClosureFN(ContactMatchBase):
    """Mean over the 5 fingertips of min(F_f, cap) — capped Newtons."""

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
        cap: float = 5.0,
    ) -> torch.Tensor:
        return self._robot_obj_force_mag(env).clamp(max=float(cap)).mean(dim=-1)


class RewardDiagClosureSat(ContactMatchBase):
    """Fraction of the 5 fingertips at/above the closure cap (F_f >= cap)."""

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
        cap: float = 5.0,
    ) -> torch.Tensor:
        return (self._robot_obj_force_mag(env) >= float(cap)).float().mean(dim=-1)


class RewardDiagClosureFmax(ContactMatchBase):
    """Max fingertip object-filtered force, UNCAPPED Newtons (slam magnitude)."""

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
        cap: float = 5.0,
    ) -> torch.Tensor:
        return self._robot_obj_force_mag(env).max(dim=-1).values
