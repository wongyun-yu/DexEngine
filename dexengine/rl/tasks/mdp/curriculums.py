"""Curriculum terms for bimanual training.

RSI (Reference State Initialization) ramps up the probability of starting
from random trajectory frames.

Threshold tightening uses an explicit linear schedule to anneal termination
thresholds from start (loose) to end (final/tight) over training.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RsiSchedule(ManagerTermBase):
    """Linearly ramp RSI fraction from 0 to target over a number of episodes.

    RSI (Reference State Initialization) allows the policy to start from
    random frames in the reference trajectory, improving robustness.
    Initially all episodes start from frame 0; over training, more episodes
    start from random frames.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._total_episodes = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        ramp_episodes: int = 10000,
        target_fraction: float = 1.0,
        fraction_start: float | None = None,
        lo_frac: float = 0.0,
        hi_frac: float = 1.0,
        lo_frac_end: float | None = None,
        hi_frac_end: float | None = None,
        band_ramp_episodes: int | None = None,
        ramp_steps: int | None = None,
        band_ramp_steps: int | None = None,
    ) -> float:
        """Update RSI fraction AND the RSI band based on total episodes completed.

        rsi_fraction ramps ``fraction_start`` (default 0) -> ``target_fraction`` over
        ``ramp_episodes`` (so it can anneal DOWN, e.g. 0.8 -> 0.25, when fraction_start
        > target_fraction). The RSI band [lo_frac, hi_frac] * (clip_len-1) ramps to
        [lo_frac_end, hi_frac_end] over ``band_ramp_episodes`` (reverse curriculum:
        slide the lower edge toward 0). Defaults (lo=0, hi=1, no ends, fraction_start
        None, target depending on cfg) reproduce the legacy behavior exactly.

        Args:
            ramp_episodes: episodes to ramp rsi_fraction.
            target_fraction: final RSI fraction.
            fraction_start: initial RSI fraction (None -> 0, legacy linear-from-0 ramp).
            lo_frac/hi_frac: initial band edges (fraction of clip length).
            lo_frac_end/hi_frac_end: final band edges (None -> constant band).
            band_ramp_episodes: episodes to ramp the band (None -> ramp_episodes).

        Returns:
            Current RSI fraction (for TensorBoard logging).
        """
        self._total_episodes += len(env_ids)
        tm = env.trajectory_manager
        # Progress driver: step-based (deterministic, preferred) when ramp_steps is set,
        # else episode-count-based (legacy). Step-based mirrors LinearThresholdSchedule so the
        # band reaches its end value at a KNOWN fraction of training.
        if ramp_steps is not None:
            progress = min(env.common_step_counter / max(ramp_steps, 1), 1.0)
        else:
            progress = min(self._total_episodes / max(ramp_episodes, 1), 1.0)
        if fraction_start is not None:
            tm.rsi_fraction = fraction_start + (target_fraction - fraction_start) * progress
        else:
            tm.rsi_fraction = progress * target_fraction
        # Band reverse curriculum (step-based when band_ramp_steps set, else episode-based)
        if band_ramp_steps is not None or ramp_steps is not None:
            _bs = band_ramp_steps if band_ramp_steps is not None else ramp_steps
            band_progress = min(env.common_step_counter / max(_bs, 1), 1.0)
        else:
            band_progress = min(self._total_episodes / max(band_ramp_episodes or ramp_episodes, 1), 1.0)
        le = lo_frac if lo_frac_end is None else lo_frac_end
        he = hi_frac if hi_frac_end is None else hi_frac_end
        tm.rsi_lo_frac = lo_frac + (le - lo_frac) * band_progress
        tm.rsi_hi_frac = hi_frac + (he - hi_frac) * band_progress
        return tm.rsi_fraction


class LinearThresholdSchedule(ManagerTermBase):
    """Linear threshold schedule driver. Sets traj_mgr.schedule_progress in [0,1]:
    0 at step 0 (start/loose thresholds), ramping linearly to 1 at `anneal_steps`
    (end/final/tight thresholds), then held at 1. Termination terms compute
    effective threshold = start + (end-start)*schedule_progress."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], anneal_steps: int = 1, stairs: int = 0) -> float:
        progress = min(env.common_step_counter / max(anneal_steps, 1), 1.0)
        # Stairs (--tighten_stairs N>1): split [0,1] into N stretches, hold the target **fixed** (a plateau) inside each
        # and tighten only at the boundaries. Linear moves the target before the policy settles a solution, so this is
        # the control variable for the collapse-in-the-tightening hypothesis. The last stretch is the final gate
        # (progress 1.0). Default 0 = linear = byte-identical.
        if stairs and stairs > 1:
            import math as _m

            progress = min(_m.floor(progress * stairs) / (stairs - 1), 1.0)
        env.trajectory_manager.schedule_progress = progress
        return progress


class ObjFollowSchedule(ManagerTermBase):
    """exp/obj-reward-ramp: ramp the OBJECT-FOLLOWING reward WEIGHTS from 1x to ``target_scale`` over the
    first ``ramp_steps`` (= 75% of training, in env.common_step_counter units), then hold at ``target_scale``.

        scale = 1 + (target_scale - 1) * min(step / ramp_steps, 1.0)

    Multiplies the BASE weights of ``obj_terms`` (obj_pos/obj_rot/obj_vel/obj_ang_vel), captured on the first
    call, via the live reward-manager term cfgs (RewardManager.compute reads ``term_cfg.weight`` each step).
    Returns the current scale for TensorBoard logging (``Curriculum/<name>``)."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._base = None  # {term_name: base_weight}
        self._announced = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        obj_terms: Sequence[str] = (),
        target_scale: float = 1.0,
        ramp_steps: int = 1,
    ) -> float:
        rm = env.reward_manager
        if self._base is None:
            self._base = {}
            for n in obj_terms:
                try:
                    self._base[n] = float(rm.get_term_cfg(n).weight)
                except ValueError:  # noqa: PERF203 — RewardManager.get_term_cfg: term not found
                    print(f"[ObjFollowSchedule] reward term {n!r} not found: not ramped", flush=True)
        frac = min(env.common_step_counter / max(int(ramp_steps), 1), 1.0)
        scale = 1.0 + (float(target_scale) - 1.0) * frac
        for n, b in self._base.items():
            try:
                rm.get_term_cfg(n).weight = b * scale
            except ValueError:  # noqa: PERF203 — RewardManager.get_term_cfg: term not found
                print(f"[ObjFollowSchedule] reward term {n!r} not found: weight not set", flush=True)
        if not self._announced:
            print(
                f"[ObjFollowSchedule] ramp 1.0 -> {target_scale} over {ramp_steps} steps (75% of train); "
                f"base weights = {self._base}"
            )
            self._announced = True
        return scale


class Level12RampDown(ManagerTermBase):
    """C5: ramp level_1 and level_2 reward weights from their base values DOWN to 0 over the
    first ``ramp_steps`` (= 75% of training, in env.common_step_counter units), then hold at 0.

        weight = base_weight * (1.0 - min(step / ramp_steps, 1.0))

    Base weights are captured on the first call from the live reward-manager term cfgs.
    ``level_terms`` is a sequence of reward-term names to ramp (e.g. ['level_1', 'level_2']).
    Returns the current ramp fraction (0 -> 1) for TensorBoard logging (``Curriculum/<name>``)."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._base = None  # {term_name: base_weight}
        self._announced = False

    def __call__(
        self, env: ManagerBasedRLEnv, env_ids: Sequence[int], level_terms: Sequence[str] = (), ramp_steps: int = 1
    ) -> float:
        rm = env.reward_manager
        if self._base is None:
            self._base = {}
            for n in level_terms:
                try:
                    self._base[n] = float(rm.get_term_cfg(n).weight)
                except ValueError:  # noqa: PERF203 — RewardManager.get_term_cfg: term not found
                    print(f"[Level12RampDown] reward term {n!r} not found: not ramped", flush=True)
        frac = min(env.common_step_counter / max(int(ramp_steps), 1), 1.0)
        for n, b in self._base.items():
            try:
                rm.get_term_cfg(n).weight = b * (1.0 - frac)
            except ValueError:  # noqa: PERF203 — RewardManager.get_term_cfg: term not found
                print(f"[Level12RampDown] reward term {n!r} not found: weight not set", flush=True)
        if not self._announced:
            print(
                f"[Level12RampDown] ramp base -> 0 over {ramp_steps} steps (ramp_level12_frac window); "
                f"base weights = {self._base}"
            )
            self._announced = True
        return frac


class WristResidualFreeZoneRamp(ManagerTermBase):
    """Ramp the tax-free band of the wrist **residual-L2 penalty**.

    ⚠ **This is NOT a tracking hinge.** It was called ``WristHingeRamp`` and printed
    ``WRIST-HINGE``, which reads as "the wrist tracking reward has a hinge" — it does
    not. Two different things share the word "hinge":

      · **tracking hinge** — ``eef_pos.hinge_threshold_m`` / fingertip
        ``hinge_threshold_m``: error within the threshold scores FULL reward.
        The wrist one is **OFF** (the recipe does not
        set ``--enable_hinge``).
        Only the fingertips have it, at 2 cm.
      · **residual free zone** (this class) — ``per_dim_deadzone`` on
        ``residual_l2_wrist``: how large the residual ACTION may grow before the
        L2 magnitude penalty starts charging. Says nothing about tracking error.

    Ramps ``per_dim_deadzone`` from 0 (full-L2 guide) to ``target`` over a step window.

        dz(t) = target * clamp((t - start_step) / (end_step - start_step), 0, 1)

    Phase 1 (t < start_step): dz = 0 — per-dim full L2, a strict wrist anchor that
    guides the residual onto the reference while the value function bootstraps.
    Phase 2 (ramp): the tax-free band widens linearly to ``target`` (fraction of the
    hard cap, e.g. 0.7), freeing the morphology-gap wrist deviation the lift needs
    (the residual_l2_wrist lift-tax finding).
    Divergence pressure near cap saturation (|r_i| -> 1) is kept at ALL times.

    Mutates ``term_cfg.params['per_dim_deadzone']`` of every term in ``terms`` via the
    live reward manager (params are re-read each RewardManager.compute call — same
    mechanism as ObjFollowSchedule's weight mutation). Terms built with the final
    target value in cfg, so EVAL/RENDER (--freeze_curriculum removes this term)
    keep the trained-final hinge. Returns dz/target for TensorBoard.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._announced = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        terms: Sequence[str] = (),
        target: float = 0.7,
        start_step: int = 0,
        end_step: int = 1,
    ) -> float:
        span = max(int(end_step) - int(start_step), 1)
        frac = (env.common_step_counter - int(start_step)) / span
        dz = float(target) * min(max(frac, 0.0), 1.0)
        rm = env.reward_manager
        for n in terms:
            try:
                rm.get_term_cfg(n).params["per_dim_deadzone"] = dz
            except ValueError:  # noqa: PERF203 — RewardManager.get_term_cfg: term not found
                if not self._announced:
                    print(f"[WristResidualFreeZoneRamp] reward term {n!r} not found: not ramped", flush=True)
        if not self._announced:
            print(
                f"[WristResidualFreeZoneRamp] terms={list(terms)} residual-L2 free zone "
                f"0 -> {target} of cap over steps [{start_step}, {end_step}] "
                f"(phase1 full-L2 guide, then linear ramp). "
                f"NOT a tracking hinge — wrist tracking hinge stays OFF."
            )
            self._announced = True
        return dz / max(float(target), 1e-9)


# Old name, kept so callers that still import it keep working.
WristHingeRamp = WristResidualFreeZoneRamp
