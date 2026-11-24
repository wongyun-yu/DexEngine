"""Arm-in-loop training environment classes.

Provides env classes for arm-in-loop training: ``ArmTrainEnv`` (hand-tracking)
and ``ArmResidualTrainEnv`` (task env, residual on a kinematic-reference base).

The arm is positioned at the trajectory frame-0 wrist pose by the reset event
(``reset_arm_to_trajectory``, a single IK step) and then PD-converged toward the
target during the first few policy steps by the ``ArmIKFingersPositionAction``.
Per-env iterative IK inside ``_reset_idx`` is intentionally NOT used (it would
``sim.step`` ALL envs and corrupt mid-episode ones — see ``_reset_idx``).

The residual action term's base action is the kinematic reference
(compute_reference_base_action) — no base checkpoint is involved.
"""

from __future__ import annotations

import torch

from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.env_base import EvalEnv
from dexengine.rl.tasks.env_load_steps import audit_collision_and_preset, reset_scheme_and_collcurr
from dexengine.rl.tasks.env_objects import ObjectPinMixin
from dexengine.rl.tasks.env_side_reset import PerSideResetMixin
from dexengine.rl.tasks.scene.extra_slots import pre_physics as _extra_slots_pre_physics


class ArmTrainEnv(EvalEnv):
    """Arm-in-loop hand-tracking training environment.

    Arm positioning at the trajectory frame-0 wrist pose is handled by the reset
    event + PD convergence (see the module docstring); ``_reset_idx`` is a thin
    override that documents why per-env iterative IK is unsafe here.
    """

    # ── Collision authoring lives in one place: collision_authoring.py (a pre-parse hook over every env) ──

    def _reset_idx(self, env_ids):
        """Reset with single-step IK for arm wrist positioning.

        The event manager fires reset_arm_to_trajectory (single IK step)
        which gives ~5cm initial wrist error. The ArmIKFingersPositionAction
        converges the arm toward the trajectory target during the first few
        policy steps via PD control.

        NOTE: Iterative IK (calling sim.step inside reset) is NOT safe here
        because it advances ALL environments, corrupting non-resetting envs
        that are mid-episode. The single IK step + PD convergence is the
        correct approach for parallel training.
        """
        super()._reset_idx(env_ids)


class ArmResidualTrainEnv(PerSideResetMixin, ObjectPinMixin, ArmTrainEnv):
    """Arm-in-loop task training environment.

    Extends ArmTrainEnv (inheriting its reset). The residual action term's
    base action is the kinematic reference (single-stage pipeline).
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        """Init env, optionally enabling SELECTIVE cross-hand-only self-collision.

        When ``cfg.bimanual_self_collision_filter`` is True (bimanual builds), we
        temporarily rebind the ``InteractiveScene`` symbol that
        ``ManagerBasedEnv.__init__`` resolves so the scene authors cross-hand-only
        ``FilteredPairsAPI`` exclusions on the env_0 robot links BEFORE cloning
        (see dexengine/rl/tasks/scene/collision_filter.py). Pairs with the bimanual-gated
        ``enabled_self_collisions=True`` set in the env cfg; together they yield
        left_hand<->right_hand collision while keeping every inner-hand and
        inner-arm pair OFF (current convention). The single-hand path (flag
        False/absent) calls the parent unchanged — byte-identical.
        """
        _desk_nocoll = env_flags.get("DEXENGINE_DESK_ROBOT_NOCOLL") == "1"
        # DEXENGINE_BI_SELFCOLL defaults to 0 (off), the same gate as cfg/residual_scene.py. The self-collision
        # scene (=1) was harmful; =1 reproduces the old physics.
        if getattr(cfg, "bimanual_self_collision_filter", False) and env_flags.get("DEXENGINE_BI_SELFCOLL") == "1":
            import isaaclab.envs.manager_based_env as _mbe

            from dexengine.rl.tasks.scene.collision_selfcoll import BimanualSelfCollisionScene

            _orig_scene_cls = _mbe.InteractiveScene
            _mbe.InteractiveScene = BimanualSelfCollisionScene
            # COLLISION CURRICULUM (flag-gated --collision_curriculum, bimanual train
            # only): request start-OFF cross-pair filters via class attrs (ctor
            # kwargs cannot pass the symbol swap). Defaults False = byte-identical.
            # objpair (--collision_curriculum_objpair): FULL ISOLATION — also filter
            # objR<->objL at start (re-enabled at the same flip).
            _collcurr = bool(getattr(cfg, "collision_curriculum", False))
            _collcurr_obj = _collcurr and bool(getattr(cfg, "collision_curriculum_objpair", False))
            BimanualSelfCollisionScene.collcurr_start_off = _collcurr
            BimanualSelfCollisionScene.collcurr_objpair = _collcurr_obj
            print(
                "[ArmResidualTrainEnv] bimanual_self_collision_filter=True -> using "
                "BimanualSelfCollisionScene (cross-hand-only self-collision"
                + ("; +robot<->desk EXCLUDED" if _desk_nocoll else "")
                + ("; +COLLCURR cross-pairs start OFF" if _collcurr else "")
                + ("; +objpair FULL ISOLATION" if _collcurr_obj else "")
                + ")."
            )
            try:
                super().__init__(cfg, render_mode=render_mode, **kwargs)
            finally:
                # Restore the module symbol no matter what so other (e.g.
                # single-hand) envs constructed later are unaffected.
                _mbe.InteractiveScene = _orig_scene_cls
                BimanualSelfCollisionScene.collcurr_start_off = False
                BimanualSelfCollisionScene.collcurr_objpair = False
        else:
            # Canonical path: a hook scene authors the collision filters of every env before PhysX parses them
            # (right after the spawners).
            import isaaclab.envs.manager_based_env as _mbe

            from dexengine.rl.tasks.scene.collision_authoring import PostSpawnHookScene, author_collision_filters

            # Set the marker dict on self first, so the POSTINIT pass (load_managers runs inside super().__init__)
            # does not author a fallback; the hook fills it.
            self._coll_author_info = {}
            _pre_info = self._coll_author_info
            _orig_scene_cls = _mbe.InteractiveScene
            PostSpawnHookScene.hooks = [
                (
                    lambda _sc: _pre_info.update(author_collision_filters(_sc, cfg)),
                    "collision filter authoring (pre-parse)",
                )
            ]
            _mbe.InteractiveScene = PostSpawnHookScene
            try:
                super().__init__(cfg, render_mode=render_mode, **kwargs)
            finally:
                _mbe.InteractiveScene = _orig_scene_cls
                PostSpawnHookScene.hooks = []

    def load_managers(self):
        """Load managers (no base-checkpoint wiring in the single-stage pipeline)."""
        super().load_managers()

        # Single-stage reference-base pipeline: there is no base-checkpoint concept
        # (the legacy frozen-base loader is gone).

        # --voc 1 (cfg.voc_enable) has no consumer: the virtual object controller was removed. It could not run
        # anyway (it and the pre-physics gravity wrench write the same object composer), so this stays a hard stop.
        if getattr(self.cfg, "voc_enable", False):
            raise RuntimeError(
                "--voc 1: the virtual object controller was removed (it overwrote the pre-physics "
                "gravity wrench); run without --voc"
            )

        # TERMDIAG — print-only per-term termination counts. cfg.termdiag_enable is
        # set by rl/train/runner.py for every --bimanual 1 TRAIN run (no extra flag);
        # default (attribute absent) is byte-identical. Counts accumulate in step()
        # from TerminationManager._term_dones; first print after 1 iter (32 steps,
        # wiring proof), then every cfg.termdiag_steps.
        self._termdiag_counts = None
        if getattr(self.cfg, "termdiag_enable", False):
            names = list(self.termination_manager._term_names)
            self._termdiag_counts = torch.zeros(len(names), dtype=torch.long, device=self.device)
            self._termdiag_steps = max(int(getattr(self.cfg, "termdiag_steps", 800)), 1)
            self._termdiag_last = 0
            self._termdiag_first = True
            print(
                f"[TERMDIAG] armed: terms={names} (counts every {self._termdiag_steps} steps; first at 32)", flush=True
            )

        self._reset_scheme_and_collcurr()
        # The start-OFF authoring is done by collision_authoring.author_collision_filters (the pre-parse hook);
        # the scene clone hook is not called by the current Isaac Lab, so nothing may rely on it.

        self._audit_collision_and_preset()

    _reset_scheme_and_collcurr = reset_scheme_and_collcurr

    _audit_collision_and_preset = audit_collision_and_preset

    def step(self, action):
        """Step; first snap the sides that failed their gates last step (_snap_failed_sides)."""
        self._snap_failed_sides()
        # Extra object slots (DEXENGINE_EXTRA_SLOTS; off does nothing): teleport to the next reference frame pose.
        _extra_slots_pre_physics(self)
        ret = super().step(action)
        # TERMDIAG (bimanual train, print-only): accumulate this step's per-term dones
        # (TerminationManager.compute refreshed _term_dones inside super().step()).
        if getattr(self, "_termdiag_counts", None) is not None:
            tm = self.termination_manager
            self._termdiag_counts += tm._term_dones.sum(dim=0).long()
            c = int(self.common_step_counter)
            interval = 32 if self._termdiag_first else self._termdiag_steps
            if c - self._termdiag_last >= interval:
                vals = self._termdiag_counts.tolist()
                body = " ".join(f"{n}={v}" for n, v in zip(tm._term_names, vals, strict=True))
                print(
                    f"[TERMDIAG] step {c} (last {c - self._termdiag_last} steps): {body} total={sum(vals)}", flush=True
                )
                self._termdiag_last = c
                self._termdiag_counts.zero_()
                self._termdiag_first = False
        # PER-SIDE ADVANTAGE: publish reward_right/reward_left into extras (the
        # info dict returned this step). Total reward untouched. The first 64
        # steps are debug-checked: right+left must reproduce reward_buf.
        if getattr(self.cfg, "per_side_reward", False):
            r_r, r_l = self.compute_per_side_rewards()
            if getattr(self, "_park_on", False):
                # Parked side reward 0; subtract it from the total too so the split check (r_R+r_L==total) holds
                _dr = r_r * self._parked["right"].float()
                _dl = r_l * self._parked["left"].float()
                r_r = r_r - _dr
                r_l = r_l - _dl
                self.reward_buf = self.reward_buf - _dr - _dl
            self.extras["reward_right"] = r_r
            self.extras["reward_left"] = r_l
            n = int(getattr(self, "_psr_steps", 0))
            if n < 64:
                err = float((r_r + r_l - self.reward_buf).abs().max().item())
                tol = 1e-4 + 1e-4 * float(self.reward_buf.abs().max().item())
                assert err <= tol, (
                    f"[PER-SIDE REWARD] split mismatch at step {n}: max|r_R+r_L-total|={err:.3e} > tol {tol:.3e}"
                )
                self._psr_err = max(float(getattr(self, "_psr_err", 0.0)), err)
                self._psr_steps = n + 1
                if self._psr_steps == 64:
                    print(f"[PER-SIDE REWARD] split check PASS over 64 steps: max|r_R+r_L-total|={self._psr_err:.3e}")
        # RESET SCHEME=decoupled: publish done_right/done_left extras from the masks
        # the snap-only gates stashed during this step's termination compute, and
        # queue the corresponding snaps for the top of the next step.
        if getattr(self, "_pss_on", False):
            self._pss_after_step()
        # ABSURD rule B: make the rewards, per-side rewards and observations finite only for the envs cut this step
        # by a non-finite or absurd state; the per-side rewards (extras) must be set by then. Without such envs it
        # writes nothing (owner guards).
        from dexengine.rl.tasks.guards import sanitize_step as _abs_sanitize

        return _abs_sanitize(self, ret)

    def _snap_failed_sides(self):
        """RESET SCHEME=decoupled: snap sides that failed their gates LAST step back onto the reference before this
        step's physics (same pre-physics injection point as the collision-curriculum flip).
        """
        if getattr(self, "_pss_on", False):
            self._pss_apply_pending()
        # DEXENGINE_OBJ_SPAWN_HOLD=N: for N physics steps after reset both objects are re-pinned every step to the
        # reference pose (zero velocity), a kinematic approximation. They are released to physics once the finger PD
        # has settled into the grasp (0.2 s at N=10), which removes the spawn depenetration pop (1-5 cm).
        # Default (unset) = byte-identical. Owner of the window length and test = spawn_hold (the articulated-object
        # sanity termination reads the same window).
        from dexengine.rl.tasks.scene.spawn_hold import JOINTS_MARK as _hjm
        from dexengine.rl.tasks.scene.spawn_hold import pinned_before_physics as _pbp
        from dexengine.rl.tasks.scene.spawn_hold import spawn_hold_steps as _shs

        _hold_n = _shs()
        if _hold_n > 0:
            _mask = _pbp(self.episode_length_buf, _hold_n)
            if _mask.any():
                _wj = self.pin_objects_to_reference(_mask.nonzero(as_tuple=False).squeeze(-1), joints=True)
                if not getattr(self, "_spawn_hold_logged", False):
                    self._spawn_hold_logged = True
                    print(
                        f"[SPAWN_HOLD] on: objects re-pinned to the reference for {_hold_n} steps after reset",
                        flush=True,
                    )
                    print(
                        f"{_hjm} articulated joints θ·ω (reference θ · ω 0) re-pinned in the same window — joints per "
                        f"side {_wj}{' (no articulated object — rigid only)' if not any(_wj.values()) else ''}",
                        flush=True,
                    )
        # COLLISION CURRICULUM: one-shot cross-pair enable BETWEEN physics steps
        # (pre-physics of this policy step). env_0 template edit -> all clones live.
        # Gradual switch: with DEXENGINE_COLLCURR_RAMP="start,end" (training progress 0..1) the **share of envs**
        # with collision on rises linearly 0->1 over that window; unset keeps the one-shot switch below (default off
        # = byte-identical). Why a share of envs: clips bind statically to envs (clip_ids = arange(num_envs) %
        # num_clips), so raising the env share raises the share of clips with collision on. With one clip some envs
        # of the same clip switch on (a difficulty mix); with many clips it is exactly a per-clip ramp.
        if getattr(self, "_collcurr_pending", False) and getattr(self, "_collcurr_ramp", None):
            _s, _e = self._collcurr_ramp
            _prog = self.common_step_counter / max(1, self._collcurr_total)
            _f = 0.0 if _prog <= _s else (1.0 if _prog >= _e else (_prog - _s) / max(1e-9, _e - _s))
            if _f > 0.0:
                # A module function, independent of the scene class (a `hasattr(self.scene, ...)` gate was always
                # closed on the canonical bimanual scene, which made the ramp a no-op).
                from dexengine.rl.tasks.scene.collision_curriculum import collcurr_enable_fraction as _cc_frac_fn

                _added, _on = _cc_frac_fn(self.scene, _f)
                if _on >= self.num_envs:
                    self._collcurr_pending = False
                    print(
                        f"[COLLCURR] gradual switch done — cross collision ON in every env "
                        f"(step {self.common_step_counter}/{self._collcurr_total})",
                        flush=True,
                    )
        elif getattr(self, "_collcurr_pending", False) and self.common_step_counter >= self._collcurr_flip_step:
            self._collcurr_pending = False
            # No `hasattr` gate either: the module function only looks at the scene._collcurr_pairs record, so it
            # does not depend on the scene class.
            from dexengine.rl.tasks.scene.collision_curriculum import collcurr_enable_cross_pairs as _cc_flip_fn

            _n = _cc_flip_fn(self.scene)
            _cats = "RHxLH, RHxobjL, LHxobjR" + (", objRxobjL" if getattr(self, "_collcurr_objpair", False) else "")
            print(
                f"[COLLCURR] cross-pairs ENABLED ({_cats}) at step "
                f"{self.common_step_counter}/{self._collcurr_total} "
                f"({_n} filtered-pair targets removed; the edit range and env count are printed as a result by "
                f"the [COLLCURR] one-shot line just above)",
                flush=True,
            )
