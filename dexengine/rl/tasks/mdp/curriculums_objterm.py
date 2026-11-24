"""Adaptive object-tracking termination curriculum (ObjTermAdaptive): after each frozen eval it loosens or tightens the
position and rotation gates of the object-tracking termination.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from dexengine.rl.swallow import note_once

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#: Returned by a split-out step that did not return early (its caller then goes on).
_CONTINUE = object()


class ObjTermAdaptive(ManagerTermBase):
    """adaptive-term v3 (v2 revived plus upgrades): INDEPENDENT per-axis
    performance-gated tightening of the object_tracking termination thresholds. Unlike time-based
    schedules (which collapsed by outpacing the policy in earlier runs), each axis only
    tightens when the policy is demonstrably improving at a FIXED-100% probe gate AND the axis'
    peak-error q80 fits inside the proposed gate.

    v3 upgrades over v2 (whose verdict was "responsive but gate gamed via low-skim"):
      - PHYSICAL per-axis steps: pos_step_cm / rot_step_deg (e.g. 0.2cm / 3deg);
        step<=0 disables that axis (posonly/rotonly ablations).
      - PHYSICAL floors: pos_floor_cm / rot_floor_deg (replaces shared fractional s_min).
      - TIP-DRAG VETO (anti-low-skim fix): tighten only while the eval population actually LIFTS
        the knife tip — recent mean drag must be <= tip_veto (reason 8). Uses the v3 tip-drag
        capture in ObjectTrackingExceeded. tip_veto<0 disables (v2-replica ablation).
      - Optional BACKOFF (backoff=1): if the latest eval completion crashes below
        floor_complete-0.20, loosen BOTH axes one step (never above base), reason 9.

    Mechanics: writes pos_end / rot_end_deg on the object_tracking term every step; with
    threshold_tightening anneal_steps=1 the TRAINING gate == pos_end/rot_end. The frozen eval
    probes at schedule_progress=0 == pos_start/rot_start == 100% baseline (clean decision metric).
    Decisions fire only when frozen_eval_snapshot appends a new tuple to env._adaptive_eval_log:
    (completion, mean_ep_len, q80_pos_cm, q80_rot_deg[, drag_q50]).

    Per-axis reason codes (W&B adaptive_term/*_no_tighten_reason_code):
      0 tightened | 1 cooldown | 2 improvement failed | 3 recent downtrend | 4 completion floor
      5 axis q80 capacity veto | 6 at floor/disabled | 7 warmup | 8 tip-drag veto | 9 backoff fired
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._pos_cm = None  # current thresholds in PHYSICAL units (init from base on first call)
        self._rot_deg = None
        self._trans_reductions = 0
        self._rot_reductions = 0
        self._backoffs = 0
        self._last_trans_idx = -(10**9)
        self._last_rot_idx = -(10**9)
        self._seen = 0
        self._trans_reason = 7
        self._rot_reason = 7
        self._announced = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        base_pos: float = 0.10,
        base_rot: float = 180.0,
        pos_step_cm: float = 0.2,
        rot_step_deg: float = 3.0,
        pos_floor_cm: float = 3.0,
        rot_floor_deg: float = 30.0,
        n: int = 2,
        cooldown: int = 1,
        dC: float = 0.01,
        dP: float = 2.0,
        floor_complete: float = 0.55,
        q: float = 0.80,
        margin_pos_cm: float = 0.0,
        margin_rot_deg: float = 0.0,
        tolC: float = 0.05,
        tolP: float = 10.0,
        tip_veto: float = 0.30,
        backoff: int = 0,
        unconditional: int = 0,
        term_name: str = "object_tracking",
    ) -> float:
        try:
            term_cfg = env.termination_manager.get_term_cfg(term_name)
        except ValueError as _exc:
            note_once("OBJTRACK_CURRICULUM term", _exc)  # TerminationManager.get_term_cfg: ValueError
            return 1.0
        if self._pos_cm is None:
            self._pos_cm = float(base_pos) * 100.0
            self._rot_deg = float(base_rot)
        # Apply CURRENT thresholds to the TRAINING gate every step (pos_start/rot_start stay 100%).
        term_cfg.params["pos_end"] = self._pos_cm / 100.0
        term_cfg.params["rot_end_deg"] = self._rot_deg
        if not self._announced:
            print(
                f"[ObjTermAdaptive:v3] ENABLED (independent axes, PHYSICAL steps): "
                f"pos {self._pos_cm:.1f}cm -{pos_step_cm}cm/step floor {pos_floor_cm}cm | "
                f"rot {self._rot_deg:.0f}deg -{rot_step_deg}deg/step floor {rot_floor_deg}deg | "
                f"n={n} cooldown={cooldown} dC={dC} dP={dP} floorC={floor_complete} q={q} "
                f"margins={margin_pos_cm}cm/{margin_rot_deg}deg tip_veto={tip_veto} backoff={backoff}",
                flush=True,
            )
            self._announced = True
        self._stash_call_frozen_eval(env)
        _ret_ = self._adapt_on_new_eval(
            env,
            base_pos,
            base_rot,
            pos_step_cm,
            rot_step_deg,
            pos_floor_cm,
            rot_floor_deg,
            n,
            cooldown,
            dC,
            dP,
            floor_complete,
            margin_pos_cm,
            margin_rot_deg,
            tolC,
            tolP,
            tip_veto,
            backoff,
            unconditional,
            term_cfg,
        )
        if _ret_ is not _CONTINUE:
            return _ret_
        return self._pos_cm / max(float(base_pos) * 100.0, 1e-9)

    def _adapt_on_new_eval(
        self,
        env,
        base_pos,
        base_rot,
        pos_step_cm,
        rot_step_deg,
        pos_floor_cm,
        rot_floor_deg,
        n,
        cooldown,
        dC,
        dP,
        floor_complete,
        margin_pos_cm,
        margin_rot_deg,
        tolC,
        tolP,
        tip_veto,
        backoff,
        unconditional,
        term_cfg,
    ):
        """Read the adaptive eval log; on a new entry, back off or step the thresholds, then log them."""
        log = getattr(env, "_adaptive_eval_log", None)
        if not log or len(log) <= self._seen:
            return self._pos_cm / max(float(base_pos) * 100.0, 1e-9)
        self._seen = len(log)
        idx = len(log) - 1
        C = [float(e[0]) for e in log]
        P = [float(e[1]) for e in log]
        Q80pos = [float(e[2]) for e in log]
        Q80rot = [float(e[3]) for e in log]
        Drag = [float(e[4]) if len(e) > 4 else float("nan") for e in log]

        def _mean(a):
            return sum(a) / max(len(a), 1)

        _ret = self._backoff_or_step(
            base_pos,
            base_rot,
            pos_step_cm,
            rot_step_deg,
            pos_floor_cm,
            rot_floor_deg,
            n,
            cooldown,
            dC,
            dP,
            floor_complete,
            margin_pos_cm,
            margin_rot_deg,
            tolC,
            tolP,
            tip_veto,
            backoff,
            unconditional,
            C,
            Drag,
            P,
            Q80pos,
            Q80rot,
            _mean,
            idx,
            log,
            term_cfg,
        )
        if _ret is not _CONTINUE:
            return _ret
        _dr = Drag[-1] if Drag else float("nan")
        print(
            f"[ObjTermAdaptive:v3] eval#{idx} pos={self._pos_cm:.2f}cm rot={self._rot_deg:.1f}deg "
            f"q80pos={_mean(Q80pos[-n:]):.2f}cm q80rot={_mean(Q80rot[-n:]):.1f}deg drag={_dr:.2f} "
            f"reasons(t/r)={self._trans_reason}/{self._rot_reason} "
            f"reductions(t/r)={self._trans_reductions}/{self._rot_reductions}",
            flush=True,
        )
        return _CONTINUE

    def _backoff_or_step(
        self,
        base_pos,
        base_rot,
        pos_step_cm,
        rot_step_deg,
        pos_floor_cm,
        rot_floor_deg,
        n,
        cooldown,
        dC,
        dP,
        floor_complete,
        margin_pos_cm,
        margin_rot_deg,
        tolC,
        tolP,
        tip_veto,
        backoff,
        unconditional,
        C,
        Drag,
        P,
        Q80pos,
        Q80rot,
        _mean,
        idx,
        log,
        term_cfg,
    ):
        """Completion crash: loosen both axes one step (backoff); otherwise run the global gate and the axis steps."""
        # --- BACKOFF (v3): completion crash -> loosen both axes one step, cooldown ---
        if (
            int(backoff) == 1
            and len(log) >= 2 * n
            and C[-1] < float(floor_complete) - 0.20
            and (self._trans_reductions > 0 or self._rot_reductions > 0)
        ):
            self._pos_cm = min(self._pos_cm + float(pos_step_cm), float(base_pos) * 100.0)
            self._rot_deg = min(self._rot_deg + float(rot_step_deg), float(base_rot))
            self._apply_backoff(C, idx, term_cfg)
            return self._pos_cm / max(float(base_pos) * 100.0, 1e-9)

        self._global_gate(
            pos_step_cm,
            rot_step_deg,
            pos_floor_cm,
            rot_floor_deg,
            n,
            cooldown,
            dC,
            dP,
            floor_complete,
            margin_pos_cm,
            margin_rot_deg,
            tolC,
            tolP,
            tip_veto,
            unconditional,
            C,
            Drag,
            P,
            Q80pos,
            Q80rot,
            _mean,
            idx,
            log,
        )

        term_cfg.params["pos_end"] = self._pos_cm / 100.0
        term_cfg.params["rot_end_deg"] = self._rot_deg
        return _CONTINUE

    def _apply_backoff(self, C, idx, term_cfg):
        """Loosen both axes one step after a completion crash and print the backoff."""
        self._backoffs += 1
        self._last_trans_idx = idx
        self._last_rot_idx = idx
        self._trans_reason = 9
        self._rot_reason = 9
        term_cfg.params["pos_end"] = self._pos_cm / 100.0
        term_cfg.params["rot_end_deg"] = self._rot_deg
        print(
            f"[ObjTermAdaptive:v3] eval#{idx} BACKOFF (C={C[-1]:.2f} < floor-0.20): "
            f"pos->{self._pos_cm:.2f}cm rot->{self._rot_deg:.1f}deg (backoffs={self._backoffs})",
            flush=True,
        )

    def _stash_call_frozen_eval(self, env):
        """W&B stash every call (frozen_eval_snapshot logs it; 1-eval lag is fine)."""
        env._adaptive_term_wandb = {
            "adaptive_term/trans_threshold_cm": self._pos_cm,
            "adaptive_term/rot_threshold_deg": self._rot_deg,
            "adaptive_term/trans_no_tighten_reason_code": self._trans_reason,
            "adaptive_term/rot_no_tighten_reason_code": self._rot_reason,
            "adaptive_term/trans_reductions": self._trans_reductions,
            "adaptive_term/rot_reductions": self._rot_reductions,
            "adaptive_term/backoffs": self._backoffs,
        }

    def _global_gate(
        self,
        pos_step_cm,
        rot_step_deg,
        pos_floor_cm,
        rot_floor_deg,
        n,
        cooldown,
        dC,
        dP,
        floor_complete,
        margin_pos_cm,
        margin_rot_deg,
        tolC,
        tolP,
        tip_veto,
        unconditional,
        C,
        Drag,
        P,
        Q80pos,
        Q80rot,
        _mean,
        idx,
        log,
    ):
        """GLOBAL gate (shared by both axes) unconditional=1 (adaptivity ABLATION): skip ALL gates
        (improvement/floor/tip/q80 via the per-axis branches below still see global_ok=True and margins are bypassed
        by callers setting them < 1e8) -> pure eval-clocked fixed-rate ramp (linear-in-evals schedule).
        """
        if int(unconditional) == 1:
            global_ok, global_reason = True, 0
        elif len(log) < 2 * n:
            global_ok, global_reason = False, 7
        else:
            Cp, Cc = _mean(C[-2 * n : -n]), _mean(C[-n:])
            Pp, Pc = _mean(P[-2 * n : -n]), _mean(P[-n:])
            _drag_recent = _mean(Drag[-n:])
            if not (Cc >= Cp + dC and Pc >= Pp + dP):
                global_ok, global_reason = False, 2
            elif C[-1] < Cc - tolC or P[-1] < Pc - tolP:
                global_ok, global_reason = False, 3
            elif Cc < floor_complete:
                global_ok, global_reason = False, 4
            elif float(tip_veto) >= 0.0 and not (_drag_recent <= float(tip_veto)):
                # nan drag (capture off/failed) also lands here -> HOLD, never tighten blindly
                global_ok, global_reason = False, 8
            else:
                global_ok, global_reason = True, 0

        self._translation_axis(
            pos_step_cm,
            rot_step_deg,
            pos_floor_cm,
            rot_floor_deg,
            n,
            cooldown,
            margin_pos_cm,
            margin_rot_deg,
            unconditional,
            Q80pos,
            Q80rot,
            _mean,
            idx,
            log,
            global_ok,
            global_reason,
        )

    def _translation_axis(
        self,
        pos_step_cm,
        rot_step_deg,
        pos_floor_cm,
        rot_floor_deg,
        n,
        cooldown,
        margin_pos_cm,
        margin_rot_deg,
        unconditional,
        Q80pos,
        Q80rot,
        _mean,
        idx,
        log,
        global_ok,
        global_reason,
    ):
        """TRANSLATION axis."""
        if float(pos_step_cm) <= 0.0:
            self._trans_reason = 6  # axis disabled == permanently at floor
        elif int(unconditional) != 1 and len(log) < 2 * n:
            self._trans_reason = 7
        elif idx - self._last_trans_idx < cooldown:
            self._trans_reason = 1
        elif not global_ok:
            self._trans_reason = global_reason
        elif self._pos_cm <= float(pos_floor_cm) + 1e-9:
            self._trans_reason = 6
        else:
            # CLAMP the step to the floor (a step that would overshoot the floor would otherwise be
            # REJECTED, stranding the gate above the floor forever, e.g. 6.6-1.7=4.9<5.0).
            proposed_cm = max(self._pos_cm - float(pos_step_cm), float(pos_floor_cm))
            q80p = _mean(Q80pos[-n:])
            if int(unconditional) != 1 and (q80p != q80p or proposed_cm < q80p + float(margin_pos_cm)):
                self._trans_reason = 5
            else:
                self._pos_cm = proposed_cm
                self._trans_reductions += 1
                self._last_trans_idx = idx
                self._trans_reason = 0

        # --- ROTATION axis ---
        if float(rot_step_deg) <= 0.0:
            self._rot_reason = 6
        elif int(unconditional) != 1 and len(log) < 2 * n:
            self._rot_reason = 7
        elif idx - self._last_rot_idx < cooldown:
            self._rot_reason = 1
        elif not global_ok:
            self._rot_reason = global_reason
        elif self._rot_deg <= float(rot_floor_deg) + 1e-9:
            self._rot_reason = 6
        else:
            proposed_deg = max(self._rot_deg - float(rot_step_deg), float(rot_floor_deg))
            q80r = _mean(Q80rot[-n:])
            if int(unconditional) != 1 and (q80r != q80r or proposed_deg < q80r + float(margin_rot_deg)):
                self._rot_reason = 5
            else:
                self._rot_deg = proposed_deg
                self._rot_reductions += 1
                self._last_rot_idx = idx
                self._rot_reason = 0
