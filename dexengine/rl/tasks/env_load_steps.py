"""Two steps of ArmResidualTrainEnv.load_managers, bound back as its methods: the per-side reset scheme with the
collision-curriculum switches, and the collision authoring audit with the COLL_PRESET check."""

from __future__ import annotations

import torch

from dexengine.rl.tasks import env_flags


def reset_scheme_and_collcurr(self):
    """Per-side reset scheme (snap / decoupled / park) and the collision curriculum switches."""
    _apply_reset_scheme(self)

    # COLLISION CURRICULUM — TRAIN-ONLY, flag-gated: cfg fields set by
    # rl/train/runner.py under --collision_curriculum 1 --bimanual 1; default
    # (attributes absent) is byte-identical. One-shot OFF->ON flip in step().
    self._collcurr_pending = False
    if getattr(self.cfg, "collision_curriculum", False):
        self._collcurr_flip_step = int(getattr(self.cfg, "collision_curriculum_flip_step", 0))
        # Parse the ramp window; a failure is never skipped silently. The cfg ramp
        # (--collision_curriculum_on_frac "start,end") wins; otherwise the env variable is read.
        _cfg_ramp = getattr(self.cfg, "collision_curriculum_ramp", None)
        _cr = f"{_cfg_ramp[0]},{_cfg_ramp[1]}" if _cfg_ramp else ""
        self._collcurr_ramp = None
        if _cr:
            try:
                _a, _b = (float(x) for x in _cr.split(","))
            except ValueError as _e_cr:
                raise ValueError(
                    f'DEXENGINE_COLLCURR_RAMP={_cr!r} cannot be read — the form is "start,end" '
                    f"(training progress 0..1): {_e_cr}"
                ) from _e_cr
            if not (0.0 <= _a < _b <= 1.0):
                raise ValueError(f"DEXENGINE_COLLCURR_RAMP={_cr!r}: needs 0<=start<end<=1")
            self._collcurr_ramp = (_a, _b)
            print(
                f"[COLLCURR] gradual ramp: over progress {_a:.2f}->{_b:.2f} the "
                f"share of enabled envs goes 0->100% (instead of a one-shot switch)",
                flush=True,
            )
        self._collcurr_total = int(getattr(self.cfg, "collision_curriculum_total_steps", 0))
        self._collcurr_objpair = bool(getattr(self.cfg, "collision_curriculum_objpair", False))
        self._collcurr_pending = True
        _cats = "RHxLH, RHxobjL, LHxobjR" + (", objRxobjL" if self._collcurr_objpair else "")
        print(
            f"[COLLCURR] armed: cross-pairs ({_cats}) start OFF; "
            f"one-shot ENABLE at step {self._collcurr_flip_step}/{self._collcurr_total}",
            flush=True,
        )


def _apply_reset_scheme(self):
    """RESET SCHEME=decoupled — TRAIN-ONLY, flag-gated: cfg.per_side_reset set by rl/train/runner.py under
    --reset_scheme decoupled (requires bimanual + per-side advantage). Side gates run in SNAP-ONLY mode (they stash
    env._side_fail_{side} masks instead of terminating); this env then snaps the failed side back onto its reference
    at the current clip frame at the TOP of the next step (same injection point as VOC/collcurr) and publishes
    done_right/done_left extras for the per-side GAE. Known 1-step approximations: the action applied on the snap
    step was computed from pre-snap obs, and closure_potential's prev-dist gives one clamped (±1cm) stale sample.
    Default (attribute absent) is byte-identical.
    """
    _prm = getattr(self.cfg, "per_side_reset", False)
    self._pss_mode = ("snap" if _prm is True else str(_prm)) if _prm else ""
    self._pss_on = self._pss_mode in ("snap", "decoupled", "park")
    # RESET_SCHEME=park: a side that breaks its gate neither ends the episode (OR) nor jumps back to a new
    # segment (DEC snap). It emits done_side (GAE cut) and is then **parked**: every step it follows the current
    # frame's reference kinematically (one clock, time sync kept) with reward 0, while the other side keeps
    # learning to the clip's end. When both sides are parked the gate returns termination and the episode ends.
    # Frozen/strict evaluation releases snap_only and measures pure OR (the honest metric).
    self._park_on = self._pss_mode == "park"
    if self._park_on:
        self._parked = {
            s_: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for s_ in ("right", "left")
        }
    if self._pss_on:
        _N = self.num_envs
        self._pss_pending = {
            "right": torch.zeros(_N, dtype=torch.bool, device=self.device),
            "left": torch.zeros(_N, dtype=torch.bool, device=self.device),
        }
        self._pss_counts = {"right": 0, "left": 0}
        self._pss_last_log = 0
        self._pss_warned_arm = set()
        # Per-side segment statistics (decoupled metric):
        # segment completion rate = comp / (comp + fail) per side. Two counter sets:
        # _pss_win (rolling, emitted into extras['log'] every 32 steps -> W&B
        # 'Info / SideStat/...') and _pss_cum (cumulative, snapshot/reset by the
        # frozen-eval hook for the per-side frozen metrics).
        self._pss_win = {"fail": [0, 0], "comp": [0, 0], "seglen": [0, 0]}
        self._pss_cum = {"fail": [0, 0], "comp": [0, 0], "seglen": [0, 0]}
        self._pss_win_last = 0
        # POST-SNAP GATE GRACE: after a side snap the object may be
        # teleported to an AIRBORNE reference pose (mid-clip RSI segment start)
        # and needs a catch window before its gates re-arm — without it the
        # object free-falls past the pos gate in ~7 steps and the side enters a
        # snap treadmill (observed: 848->1500 snaps/step runaway on one rank,
        # 6.7k PhysX articulation warnings). Same rationale/value as the uni
        # episode-start grace_period. Gates read these attrs in snap_only mode.
        self._pss_grace = int(getattr(self.cfg, "per_side_reset_grace", 20))
        self._pss_last_snap = torch.full((_N, 2), -(10**9), dtype=torch.long, device=self.device)
        print(f"[SIDESNAP] post-snap gate grace = {self._pss_grace} steps", flush=True)
        if self._pss_mode == "decoupled":
            # TRUE independent per-side clip clocks: side resets RESAMPLE a fresh
            # RSI start for that side alone (uni-equivalent segment cycling); side
            # clip completion also cycles. All reference lookups become side-aware
            # inside the TrajectoryManager (per_side_clock).
            self.trajectory_manager.enable_per_side_clock()
        print(
            f"[SIDESNAP] armed: reset_scheme={self._pss_mode} — side gate fails "
            + (
                "RESAMPLE the failed side onto a fresh RSI segment (independent "
                "side clocks; clip completion cycles too)"
                if self._pss_mode == "decoupled"
                else "snap the failed side onto its reference at the current frame"
            )
            + " (episode ends on time_out / non-side terms only)",
            flush=True,
        )


def audit_collision_and_preset(self):
    """Collision authoring audit (desk, truth table, wrist collider, USD variant limits) and the COLL_PRESET check."""
    # EVAL PHYSICS ABLATION (DEXENGINE_EVAL_CROSS_OFF=1): evaluate an existing policy with only hand <-> own
    # object collisions left on. The eval env never authors cfg.collision_curriculum (a training-only field), so
    # a SELFCOLL=0 policy would otherwise be scored with the cross pairs ON, which is not its training physics.
    # The same cross-pair set (RHxLH, RHxobjL, LHxobjR, objRxobjL) is authored permanently OFF by the same
    # function (author_collcurr_cross_pair_exclusions; no flip, since _collcurr_pending is never set), for
    # bimanual builds only. Default (unset) = byte-identical.
    # The collision filters are authored by PostSpawnHookScene (pre-parse). The legacy scene branch (SELFCOLL=1,
    # proxy) does not use the hook scene, so it runs the same function here (POSTINIT) as before: one authoring
    # function.
    if getattr(self, "_coll_author_info", None) is None:
        from dexengine.rl.tasks.scene.collision_authoring import author_collision_filters as _acf_post

        self._coll_author_info = _acf_post(self.scene, self.cfg)
        print(
            "[COLL_AUTHOR] ★legacy scene branch — collision filters authored at POSTINIT (after parsing; object "
            "prim authoring risks reset kicks).",
            flush=True,
        )
    _ci = self._coll_author_info
    from dexengine.rl.tasks.scene.collision_groups import assert_done as _robot_off_done

    _robot_off_done(self.scene)  # robot-off groups still pending = robot <-> slot on: fail
    _eval_cross_off = bool(_ci.get("cross_off"))
    _arm_obj_nocoll = bool(_ci.get("arm_obj"))
    _arm_obj_pairs = int(_ci.get("arm_obj_pairs", 0))
    _split_off = set(_ci.get("split_off", set()))

    from dexengine.rl.tasks.scene.desk_check import audit_desk as _audit_desk

    _audit_desk(self)

    from dexengine.rl.tasks.scene.collision_check import audit_and_print as _coll_audit

    # An audit that raises kills the boot (preset or not): no training or judging in a scene whose truth table
    # was not measured.
    self._coll_audit_rows = _coll_audit(  # RO row expectation = the disc owner's statement (_build_disc_slots)
        self.scene,
        disc_owner=getattr(self.cfg, "support_disc_owner", None),
        shared=bool(getattr(self.cfg, "shared_object", False)),
        idle=str(getattr(self.cfg, "idle_side", "") or ""),
        extra_kinds=getattr(self.cfg, "extra_slot_kinds", None),
    )  # RE · OE expectation = the extra slot kinds
    # Is the wrist collider revision really on the stage (dies if it differs from the spawn variant)? Marker
    # [ARM_WRIST_COLLIDER]
    from dexengine.rl.tasks.scene.asset_checks import audit_wrist_collider_and_print as _wrist_audit

    self._wrist_collider = _wrist_audit(self.scene)
    _check_spawned_joint_limits(self, _arm_obj_nocoll, _arm_obj_pairs, _eval_cross_off, _split_off)


def _check_spawned_joint_limits(self, _arm_obj_nocoll, _arm_obj_pairs, _eval_cross_off, _split_off):
    """Do the spawned joint limits match that variant's URDF (hand joint-limit variants included; dies if not)? Marker
    [USD_VARIANT].
    """
    from dexengine.rl.tasks.scene.asset_checks import audit_and_print as _limits_audit

    self._usd_variant_limits = _limits_audit(self.scene)

    # Consolidated one-line summary of the eval physics ablation toggles.
    # Printed for BOTH train and eval instantiations (this env class serves both
    # gym ids) — harmless extra log line, no behavior change. selfcoll reflects
    # the code-default gate read at __init__ time (DEXENGINE_BI_SELFCOLL, default "0").
    print(
        f"[EVAL_PHYS] split_off={','.join(sorted(_split_off)) or '-'} cross_off={int(_eval_cross_off)} "
        f"desk_robot_nocoll={env_flags.get('DEXENGINE_DESK_ROBOT_NOCOLL')} "
        f"selfcoll={env_flags.get('DEXENGINE_BI_SELFCOLL')}",
        flush=True,
    )
    # Physics-ablation recipe summary — arm<->object + desk<->robot +
    # self-collision toggles + the pair count author_arm_obj_nocollision just
    # authored (0 when the flag is unset). Same common point as [EVAL_PHYS]
    # (train AND eval instantiations both hit this line).
    print(
        f"[PHYS_ABL] arm_obj_nocoll={int(_arm_obj_nocoll)} "
        f"desk_robot_nocoll={env_flags.get('DEXENGINE_DESK_ROBOT_NOCOLL')} "
        f"selfcoll={env_flags.get('DEXENGINE_BI_SELFCOLL')} "
        f"cross_pairs={_arm_obj_pairs}",
        flush=True,
    )

    # ══ Collision preset check: the **real stage audit** against the expected table ══
    # audit_and_print above reads the effective ON/OFF from the prims of env_0, env_1 and the last env and
    # compares it with the expected table (collision_presets.EXPECTED); any env that differs fails. (Deriving the
    # matrix from the env variables would compare intent with intent and never catch an env_0-only authoring hole.)
    _cp = env_flags.get("DEXENGINE_COLL_PRESET").strip().lower()
    if _cp:
        from dexengine.rl.tasks.scene.collision_preset_check import check_preset as _cp_check
        from dexengine.rl.tasks.scene.collision_preset_check import normalize as _cp_norm
        from dexengine.rl.tasks.scene.collision_presets import EXPECTED as _cp_expect

        if _cp not in _cp_expect:
            raise RuntimeError(
                f"DEXENGINE_COLL_PRESET={_cp!r} is not a preset — only {sorted(_cp_expect)} exist "
                "(COLL_PRESET of the launcher, dexengine/rl/launch/plan.py)."
            )
        _ct_names = [
            _n
            for _n in getattr(getattr(self, "termination_manager", None), "_term_names", []) or []
            if str(_n).startswith("contact_")
        ]
        _contact_term = 1 if _ct_names else 0
        if not self._coll_audit_rows:
            raise RuntimeError(f"[COLL_PRESET] {_cp}: no stage audit result — cannot verify")
        _actual = _cp_norm(self._coll_audit_rows[0]["effective"])
        print(
            f"[COLL_PRESET] {_cp}: "
            + " ".join(f"{k} {v}" for k, v in _actual.items())
            + f" contact_term={_contact_term}  (real stage audit, envs "
            f"{[r['env'].rsplit('/', 1)[-1] for r in self._coll_audit_rows]} agree)",
            flush=True,
        )
        _mism = _cp_check(self._coll_audit_rows, _cp, getattr(self.cfg, "support_disc_collide", None))
        if _mism or _contact_term != 0:
            raise RuntimeError(
                f"[COLL_PRESET] {_cp} preset but the real physics differs — "
                + (f"mismatch {_mism} " if _mism else "")
                + (f"contact_term={_contact_term} (must be 0) " if _contact_term else "")
                + f"| expected {_cp_expect[_cp]}. A preset holds only if training, the frozen judge and render "
                "see the **same** physics (the recipe's COLL_PRESET); passing this silently would train "
                "on a train-test mismatch."
            )
