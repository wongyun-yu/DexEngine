"""Object-tracking overrides of the env cfg: reference-following objects, the object-tracking gate and its schedule,
fingertip and diagnostic terms."""

from __future__ import annotations

from dexengine.core.errors import InputMissingError
from dexengine.rl.tasks import env_flags


def apply_phase1_contact_aware_overrides(cfg, args_cli, ctx: str) -> None:
    """Wire contact-aware lag compensation Phase 1 flags into the env cfg.


    All overrides are gated by flag (default no-op). Slow-down + lookahead obs
    apply to hand-tracking AND task terms. Contact reward/obs/termination are task-level only
    (the hand-tracking base MDP has no object).
    """
    # (1) Slow-down: applies to both stages via trajmgr
    if hasattr(cfg, "trajectory") and cfg.trajectory is not None:
        if args_cli.slowdown_factor != 1.0:
            cfg.trajectory.slowdown_factor = float(args_cli.slowdown_factor)
            print(f"[phase1_contact_aware] slowdown_factor = {args_cli.slowdown_factor}")

    # (2) Lookahead wrist target obs (both stages)
    if getattr(args_cli, "enable_lookahead_obs", 0):
        from isaaclab.managers import ObservationTermCfg as _ObsTerm

        from dexengine.rl.tasks.mdp.rewards.contact_base import LookaheadWristTarget as _LookaheadWristTarget

        _lookahead = tuple(int(x) for x in str(args_cli.lookahead_frames).split(","))
        # Append one term per side that has its own obs group on cfg.observations
        # For TACO arm-in-loop the policy obs group is cfg.observations.policy
        # (single bimanual obs group); add ONE term covering the right side
        # (TACO's primary). If you want bimanual lookahead, add both.
        _pgroup = cfg.observations.policy
        _pgroup.lookahead_wrist_right = _ObsTerm(
            func=_LookaheadWristTarget, params={"lookahead_frames": _lookahead, "side": "right"}
        )
        print(f"[phase1_contact_aware] lookahead_wrist (right) added: frames={_lookahead}")


# Fingertip reward-term names (parent hand-tracking cfg). Used by the contact pilot
# hinge wiring. These exist via inheritance on the task rewards cfg.
CONTACT_PILOT_TIP_TERMS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
# Bimanual TACO qpos object slots (verified layout, width 50; half-open column ranges): right object pos 36..39,
# quat 39..43; left object pos 43..46, quat 46..50. Used ONLY by --auto_passive_hand motion detection.
_BIMANUAL_TACO_OBJ_SLOTS = {"right": 36, "left": 43}


def _reference_object_motion(seq_name, side):
    """Max object displacement (cm) + geodesic rotation (deg) over the reference trajectory for one
    side. Reads <data_layout.taco_source_root()>/<seq>/0/trajectory_kinematic.npz (bimanual TACO qpos).
    Returns (disp_cm, rot_deg). Only --auto_passive_hand reads it: a missing file or a qpos that does not carry the
    bimanual TACO object slots dies — the old None kept the hand ACTIVE, so a requested switch silently did
    nothing."""
    import os as _os

    import numpy as _np

    from dexengine.core import layout as _dl  # owner of the data-root rule (never read relative to the working folder)

    p = _dl.data_abspath(_os.path.join(_dl.taco_source_root(), seq_name, "0", "trajectory_kinematic.npz"))
    q = _np.load(p, allow_pickle=True)["qpos"]
    if q.ndim != 2 or q.shape[1] < 50:
        raise ValueError(
            f"[auto_passive_hand] {p}: qpos {q.shape} has no bimanual TACO object columns — the motion "
            "cannot be measured"
        )
    s = _BIMANUAL_TACO_OBJ_SLOTS[side]
    pos = q[:, s : s + 3]
    quat = q[:, s + 3 : s + 7]
    disp_cm = float(_np.linalg.norm(pos - pos[0], axis=1).max() * 100.0)
    qn = quat / _np.linalg.norm(quat, axis=1, keepdims=True)
    dot = _np.abs(qn @ qn[0])
    rot_deg = float((2.0 * _np.degrees(_np.arccos(_np.clip(dot, -1.0, 1.0)))).max())
    return disp_cm, rot_deg


def auto_passive_sides(args_cli):
    """Return the set of sides ('right'/'left') that --auto_passive_hand marks PASSIVE (object
    near-static in the reference). A side is passive iff disp < pos_cm AND rot < rot_deg. The switch is on, so
    motion that cannot be read dies instead of quietly leaving every hand ACTIVE. Prints the
    per-side decision."""
    seq = getattr(args_cli, "sequence", None)
    if not seq:
        raise InputMissingError(
            "[auto_passive_hand] --auto_passive_hand 1 without --sequence — no clip to measure the motion on"
        )
    # Track B multi-clip (comma-list): all clips are the same task → passive side is shared; use the
    # first clip for motion detection.
    seq = seq.split(",")[0].strip()
    pos_thr = float(getattr(args_cli, "auto_passive_pos_cm", 5.0))
    rot_thr = float(getattr(args_cli, "auto_passive_rot_deg", 45.0))
    passive = set()
    for side in ("right", "left"):
        disp, rot = _reference_object_motion(seq, side)
        is_passive = disp < pos_thr and rot < rot_thr
        if is_passive:
            passive.add(side)
        print(
            f"[auto_passive_hand] {side}: disp={disp:.2f}cm rot={rot:.1f}deg "
            f"(thr {pos_thr:.1f}cm/{rot_thr:.1f}deg) → {'PASSIVE (GT-following)' if is_passive else 'ACTIVE'}"
        )
    return passive


def apply_gt_following(act, side_label, ctx):
    """Zero ALL residual authority on an actor cfg (wrist+finger scales/caps) → pure reference
    execution. Shared by --left_gt_following and --auto_passive_hand."""
    act.finger_residual_scale = 0.0
    for attr in ("wrist_residual_scale", "wrist_pos_residual_max_m", "wrist_rot_residual_max_deg"):
        if hasattr(act, attr):
            setattr(act, attr, 0.0)
    print(f"[{ctx}] {side_label} GT-FOLLOWING: all residual scales/caps = 0 (pure reference execution)")


def apply_obj_track_overrides(cfg, args_cli, ctx: str) -> None:
    """Exp-4 (H2): loosen the ObjectTrackingExceeded termination thresholds/grace.

    All gated by default-no-op flags (pos/rot scale 1.0, grace None) so default
    behavior is unchanged. Lets the un-grasped object diverge further from the
    moving reference before terminating, so the episode survives into the
    manipulation phase where a grasp can be learned.
    """
    disable_rot, grace, lag_window, pos_const, pos_scale, rot_const, rot_scale = _read_obj_track_flags(
        cfg, args_cli, ctx
    )
    if (
        pos_scale == 1.0
        and rot_scale == 1.0
        and grace is None
        and not disable_rot
        and rot_const <= 0.0
        and pos_const <= 0.0
        and lag_window <= 0
    ):
        return
    term = getattr(getattr(cfg, "terminations", None), "object_tracking", None)
    if term is None:
        print(f"[{ctx}] WARNING: --obj_track_* set but cfg.terminations.object_tracking missing")
        return
    _apply_loosen, _l_pos_start, _l_rot_start, _lgr, _rotmode, _terms, pos_const_left, rot_const_left = _prepare_loosen(
        cfg, args_cli, disable_rot, grace, pos_const, pos_scale, rot_const, rot_scale, term
    )
    if _lgr:
        _l_pos_start, _l_rot_start = (float(x) for x in _lgr.split(","))
        print(
            f"[{ctx}] LEFT_GATE_RAMP: left ramps {_l_pos_start}m/{_l_rot_start}deg -> "
            f"{pos_const_left if pos_const_left > 0 else pos_const}m/"
            f"{rot_const_left if rot_const_left > 0 else rot_const}deg"
        )
    if lag_window > 0:
        # Lag-tolerant pose-match gate: both side gates get the SAME trailing
        # window; thresholds themselves are set by the const/ramp logic below.
        # lag_window_ahead: forward extension [F+1, F+ahead] pairing the wrist
        # lookahead — an early-arriving object (phase-lead overcorrection) must not be
        # punished. Set --obj_lag_window_ahead = --wrist_ref_lookahead_frames. 0 = off.
        _lag_ahead = int(getattr(args_cli, "obj_lag_window_ahead", 0) or 0)
        for _name, _term in _terms:
            _term.params["lag_window"] = lag_window
            if _lag_ahead > 0:
                _term.params["lag_window_ahead"] = _lag_ahead
        print(
            f"[{ctx}] object-tracking LAG WINDOW: {lag_window} frames "
            f"({lag_window * 0.02:.2f}s @ 50Hz)"
            + (f" + AHEAD {_lag_ahead} frames (wrist-lookahead pairing)" if _lag_ahead > 0 else "")
            + " — survive on pose-match with ANY window frame"
        )
    for _name, _term in _terms:
        p = _term.params
        if _name == "object_tracking_left" and (pos_const_left > 0.0 or rot_const_left > 0.0):
            _apply_loosen(
                p,
                pos_const=(pos_const_left if pos_const_left > 0.0 else pos_const),
                rot_const=(rot_const_left if rot_const_left > 0.0 else rot_const),
                pos_start_o=_l_pos_start,
                rot_start_o=_l_rot_start,
            )
        else:
            _apply_loosen(p)
        print(
            f"[{ctx}] {_name.upper()} loosen: pos_scale={pos_scale} rot_scale={rot_scale} "
            f"grace={grace} rot_termination={_rotmode} -> "
            f"pos_start={p.get('pos_start')} pos_end={p.get('pos_end')} "
            f"rot_start={p.get('rot_start_deg')} rot_end={p.get('rot_end_deg')}"
        )


def _read_obj_track_flags(cfg, args_cli, ctx):
    """Add the bimanual left velocity_sanity term, then read the --obj_track_* loosening flags."""
    # Bimanual: the shared velocity_sanity DoneTerm defaults to side='right' (the parent cfg
    # is used by single-hand too, where scene['left_hand'] doesn't exist). When bimanual, flip
    # it to 'both' so the LEFT arm/hand is also guarded against velocity explosions (the func
    # already has the complete left branch). Runs in both train + eval (before the early return).
    if bool(getattr(args_cli, "bimanual", 0)):
        _vs = getattr(getattr(cfg, "terminations", None), "velocity_sanity", None)
        if _vs is not None and _vs.params.get("side") != "both":
            _vs.params["side"] = "both"
            print(f"[{ctx}] (bimanual) velocity_sanity side -> 'both' (left arm/hand now guarded)")

    pos_scale = float(getattr(args_cli, "obj_track_pos_scale", 1.0) or 1.0)
    rot_scale = float(getattr(args_cli, "obj_track_rot_scale", 1.0) or 1.0)
    grace = getattr(args_cli, "obj_track_grace", None)
    disable_rot = bool(getattr(args_cli, "disable_obj_rot_termination", 0))
    rot_const = float(getattr(args_cli, "obj_rot_thresh_const", 0.0) or 0.0)
    pos_const = float(getattr(args_cli, "obj_pos_thresh_const", 0.0) or 0.0)
    lag_window = int(getattr(args_cli, "obj_lag_window", 0) or 0)
    return disable_rot, grace, lag_window, pos_const, pos_scale, rot_const, rot_scale


def _prepare_loosen(cfg, args_cli, disable_rot, grace, pos_const, pos_scale, rot_const, rot_scale, term):
    """Terms to loosen (right; left when present), left constants, start overrides and the loosen function."""
    # Terms to loosen: RIGHT object_tracking (always) + LEFT object_tracking_left (bimanual).
    # Mirroring to the left keeps the per-hand termination leash SYMMETRIC. Without it the left
    # stays on the tight default schedule (e.g. 3cm/90deg) while the right is loosened (10cm/180deg)
    # -> the left object trips its tight threshold early and caps episode length, starving left
    # learning (real left/right asymmetry).
    _terms = [("object_tracking", term)]
    _tl = getattr(getattr(cfg, "terminations", None), "object_tracking_left", None)
    if _tl is not None:
        _terms.append(("object_tracking_left", _tl))

    # LEFT-specific const overrides (0 = follow the shared consts). ObjTermAdaptive drives the RIGHT
    # gate only, so training-parity strict evals need right=floor while left stays at its baseline.
    pos_const_left = float(getattr(args_cli, "obj_pos_thresh_const_left", 0.0) or 0.0)
    rot_const_left = float(getattr(args_cli, "obj_rot_thresh_const_left", 0.0) or 0.0)

    # RIGHT-side ramp-start overrides (0 = off): with the consts set, express a LINEAR schedule
    # start->const (threshold_tightening drives progress; frozen-eval forces progress=1 -> end gate).
    pos_start_ovr = float(getattr(args_cli, "obj_pos_thresh_start", 0.0) or 0.0)
    rot_start_ovr = float(getattr(args_cli, "obj_rot_thresh_start", 0.0) or 0.0)

    def _apply_loosen(
        p, pos_const=pos_const, rot_const=rot_const, pos_start_o=pos_start_ovr, rot_start_o=rot_start_ovr
    ):
        if pos_const > 0.0:
            # CONSTANT position threshold (no annealing), METERS: pos_start == pos_end == pos_const, so
            # effective_pos_thresh = pos_const at every schedule_progress, in BOTH train and eval. e.g.
            # 0.07 = 7cm. Used to EXTREMELY loosen pose termination (with rot const 180 + contact term)
            # so the hand has max freedom to lift; the contact term guards against non-grasping. Takes
            # precedence over pos_scale.
            p["pos_start"] = pos_const
            p["pos_end"] = pos_const
            if pos_start_o > 0.0:
                # LINEAR ramp: loose START -> const END, annealed by threshold_tightening.
                p["pos_start"] = pos_start_o
        elif pos_scale != 1.0:
            for k in ("pos_start", "pos_end"):
                if k in p:
                    p[k] = float(p[k]) * pos_scale
        if rot_scale != 1.0:
            for k in ("rot_start_deg", "rot_end_deg"):
                if k in p:
                    p[k] = float(p[k]) * rot_scale
        if rot_const > 0.0:
            # CONSTANT rotation threshold (no annealing): rot_start == rot_end == rot_const, so
            # effective_rot_thresh = rot_const at every schedule_progress, in BOTH train and eval.
            # e.g. 180 deg = the max possible geodesic angle_err, so rotation termination effectively
            # never fires (a real physical constant, vs the 1e6 disable sentinel). Position schedule
            # is untouched. Takes precedence over disable_rot.
            p["rot_start_deg"] = rot_const
            p["rot_end_deg"] = rot_const
            if rot_start_o > 0.0:
                # LINEAR rot ramp: loose START (e.g. 180deg) -> const END (e.g. 60deg).
                p["rot_start_deg"] = rot_start_o
        elif disable_rot:
            # DISABLE rotation termination (NOT 0 deg — that would fire on any rotation). Set the rot
            # threshold effectively infinite; geodesic angle_err <= 180 deg can never exceed it, so
            # rot_exceeded is always False and object_tracking reduces to POSITION-only. Applied in
            # both train and eval (this fn runs in both paths).
            p["rot_start_deg"] = 1.0e6
            p["rot_end_deg"] = 1.0e6
        if grace is not None:
            p["grace_period"] = int(grace)

    _rotmode = f"CONST {rot_const:.0f}deg" if rot_const > 0.0 else ("DISABLED" if disable_rot else "on")
    # Symmetric left-hand training PoC: DEXENGINE_LEFT_GATE_RAMP="pos_start_m,rot_start_deg" gives the
    # LEFT term its own linear ramp (start -> left consts) even when the RIGHT consts are loose.
    # Needed because the left-const branch hardcodes ramp starts to 0 (const-only), but symmetric
    # left training wants LEFT ramped 10cm/180 -> 5cm/60 while the GT-following RIGHT stays loose.
    _lgr = env_flags.get("DEXENGINE_LEFT_GATE_RAMP")
    _l_pos_start, _l_rot_start = 0.0, 0.0
    return _apply_loosen, _l_pos_start, _l_rot_start, _lgr, _rotmode, _terms, pos_const_left, rot_const_left


def verify_objtrack_schedule(cfg, ctx: str, frozen: bool) -> None:
    """Print the EFFECTIVE object_tracking thresholds across the schedule, so a silent
    freeze / curriculum mismatch is caught loudly. effective = start + (end-start)*progress.
    TRAIN (not frozen): progress anneals 0->1 over the first ~50% of training.
    EVAL/frozen: progress pinned at 1.0 -> the fixed *_end thresholds."""
    term = getattr(getattr(cfg, "terminations", None), "object_tracking", None)
    if term is None or not hasattr(term, "params"):
        print(f"[{ctx}] OBJTRACK SCHEDULE VERIFY: object_tracking term MISSING!")
        return
    p = term.params
    ps, pe = float(p.get("pos_start", 0.0)), float(p.get("pos_end", 0.0))
    rs, re = float(p.get("rot_start_deg", 0.0)), float(p.get("rot_end_deg", 0.0))

    def eff(a, b, g):
        return a + (b - a) * g

    rot_disabled = re >= 360.0  # effectively-infinite threshold => rotation termination DISABLED

    def rfmt(deg):
        return "DISABLED" if rot_disabled else f"{deg:.0f}deg"

    if frozen:
        print(
            f"[{ctx}] OBJTRACK SCHEDULE VERIFY (FROZEN, progress=1.0): "
            f"FIXED pos={eff(ps, pe, 1.0) * 100:.1f}cm / rot={rfmt(eff(rs, re, 1.0))}"
        )
    else:
        print(
            f"[{ctx}] OBJTRACK SCHEDULE VERIFY (annealed progress 0->1 over first ~50% of train): "
            f"train-start(p=0)={ps * 100:.1f}cm/{rfmt(rs)} | "
            f"train-mid(p=0.5)={eff(ps, pe, 0.5) * 100:.1f}cm/{rfmt(eff(rs, re, 0.5))} | "
            f"train-end(p=1)={pe * 100:.1f}cm/{rfmt(re)}  [EVAL pins p=1 -> {pe * 100:.1f}cm/{rfmt(re)}]"
        )


def apply_judge_objtrack_term(cfg, launch_ctx: str, ctx: str) -> None:
    """Judge object-tracking termination on/off (owner judge_terms: OBJTRACK_VAR; standard = on). Only a fresh-process
    judge consumes it (in any other context dexengine.rl.judge.launch.check_launch has already failed on the input). Off
    (explicit 0) sets judge_off=True on each side's termination terms (judge_terms.OBJTRACK_TERMS); no such term means
    no consumer, which fails. Prints the marker [JUDGE_TERM] objtrack=on|off when fresh (measured = the [TERMDIAG]
    counts of the judge log). On leaves the cfg untouched (byte-identical)."""
    if launch_ctx != "fresh":
        return
    from dexengine.rl.judge import terms as judge_terms

    val, src = judge_terms.objtrack_term()
    terms = getattr(cfg, "terminations", None)
    found = [n for n in judge_terms.OBJTRACK_TERMS if getattr(terms, n, None) is not None]
    if val == 0:
        if not found:
            raise RuntimeError(
                f"[JUDGE_TERM] {judge_terms.OBJTRACK_VAR}=0 but the cfg has none of the termination "
                f"terms {judge_terms.OBJTRACK_TERMS} (no consumer)"
            )
        for n in found:
            tc = getattr(terms, n)
            tc.params = dict(tc.params or {}, judge_off=True)
    print(
        judge_terms.objtrack_marker(val, src, found), flush=True
    )  # line head = [JUDGE_TERM] (the marker form) · caller = ctx


def wire_objtrack_split_diags(cfg, args_cli, ctx: str) -> None:
    """Log the object_tracking termination's pos vs rot components separately
    as zero-weight diagnostic reward terms. Reads the LIVE (post-scale) thresholds off
    cfg.terminations.object_tracking so the split matches the real termination exactly.
    Original termination behavior is untouched. Call AFTER apply_obj_track_overrides."""
    if not int(getattr(args_cli, "log_objtrack_split", 1)):
        return
    term = getattr(getattr(cfg, "terminations", None), "object_tracking", None)
    rewards = getattr(cfg, "rewards", None)
    if term is None or rewards is None:
        print(f"[{ctx}] objtrack split-diag SKIPPED (object_tracking term / rewards missing)")
        return
    from isaaclab.managers import RewardTermCfg as _RewTerm

    from dexengine.rl.tasks.mdp.terms import ObjectTrackingSplitDiag as _OTS

    tp = term.params
    base = {
        "pos_start": float(tp.get("pos_start", 0.04)),
        "pos_end": float(tp.get("pos_end", 0.02)),
        "rot_start_deg": float(tp.get("rot_start_deg", 60.0)),
        "rot_end_deg": float(tp.get("rot_end_deg", 30.0)),
        "side": str(tp.get("side", "right")),
        "grace_period": int(tp.get("grace_period", 8)),
    }
    for nm, comp in (
        ("diag_objtrack_pos_err", "pos_err"),
        ("diag_objtrack_rot_err", "rot_err"),
        ("diag_objtrack_pos_exc", "pos_exc"),
        ("diag_objtrack_rot_exc", "rot_exc"),
    ):
        setattr(rewards, nm, _RewTerm(func=_OTS, weight=1e-9, params=dict(base, component=comp)))
    print(
        f"[{ctx}] objtrack split-diag added: diag_objtrack_{{pos_err,rot_err(deg),pos_exc,rot_exc}} "
        f"(pos {base['pos_start']:.3f}->{base['pos_end']:.3f}m, "
        f"rot {base['rot_start_deg']:.0f}->{base['rot_end_deg']:.0f}deg, ×1e-9 recover ×1e9)"
    )


def apply_fingertip_term_disable(cfg, args_cli, ctx: str) -> None:
    """REMOVE the FingertipTrackingExceeded termination (set the term to None) so the
    hand can leave the morphology-mismatched MANO fingertip poses and discover a robot grasp.
    Robust: scans cfg.terminations by func name (no import). object_tracking (task) + the velocity
    explosion guards stay; the fingertip-tracking REWARD is untouched. Default-no-op."""
    if not int(getattr(args_cli, "disable_fingertip_termination", 0)):
        return
    terms = getattr(cfg, "terminations", None)
    if terms is None:
        print(f"[{ctx}] disable_fingertip_termination set but cfg.terminations missing")
        return
    removed = []
    for name in list(vars(terms).keys()):
        t = getattr(terms, name, None)
        f = getattr(t, "func", None)
        if t is not None and getattr(f, "__name__", "") == "FingertipTrackingExceeded":
            setattr(terms, name, None)
            removed.append(name)
    print(f"[{ctx}] FINGERTIP TERMINATION DISABLED: removed {removed or '[NONE FOUND — WARN]'}")


def gate_reward_diagnostics(cfg, args_cli) -> None:
    """Strip weight-0 diagnostic-only reward terms (``diag_*``) unless --enable_reward_diagnostics.

    Diagnostics are logging-only (weight 1e-9 → zero gradient): they appear in the reward manager
    purely so W&B records them (contact precision/recall, force projections, object-tracking error,
    keypoint error, palm distance, …). They do NOT shape the policy. Default OFF removes ~32 per-step
    reward-term computations in large-scale TACO training; --enable_reward_diagnostics 1 keeps them
    for debugging/monitoring. Behavior-neutral: only zero-gradient terms are removed, so shaping
    rewards, obs/action layout, and training semantics are unchanged.

    SAFETY: only a term that is BOTH named ``diag_*`` AND has |weight| <= 1e-6 is stripped, so a
    (hypothetical) gradient-bearing term named diag_* is never silently removed.
    """
    rewards = getattr(cfg, "rewards", None)
    if rewards is None or bool(getattr(args_cli, "enable_reward_diagnostics", 0)):
        return
    try:
        from isaaclab.managers import RewardTermCfg as _RTC
    except ImportError:
        _RTC = None
    stripped, kept_nonzero = [], []
    for _nm in list(vars(rewards)):  # instance attrs incl. dynamically setattr'd diag terms
        if not _nm.startswith("diag_"):
            continue
        # TIPZ exemption: diag_object_tip_z_cm / diag_object_tip_drag are the PRIMARY
        # knife quality metrics (completion is blind to tip drag) — always keep them so W&B records
        # Diag/object_tip_z_cm + Diag/object_tip_drag in every run. Cost: 2 small terms per step.
        if _nm.startswith("diag_object_tip_"):
            kept_nonzero.append((_nm, "tipz-exempt"))
            continue
        _term = getattr(rewards, _nm, None)
        if _term is None:
            continue
        _w = getattr(_term, "weight", None)
        if _w is None or (_RTC is not None and not isinstance(_term, _RTC)):
            continue
        if abs(float(_w)) <= 1e-6:
            setattr(rewards, _nm, None)  # not registered → not computed/logged
            stripped.append(_nm)
        else:
            kept_nonzero.append((_nm, _w))
    if stripped:
        print(
            f"[reward_diagnostics] OFF (default): stripped {len(stripped)} weight-0 diag_* "
            f"logging-only terms (set --enable_reward_diagnostics 1 to keep for W&B/debug)"
        )
    if kept_nonzero:
        print(
            f"[reward_diagnostics] KEPT {len(kept_nonzero)} diag_* term(s) with non-zero weight "
            f"(NOT logging-only): {kept_nonzero}"
        )


def apply_finger_imit_release(cfg, args_cli, ctx: str) -> None:
    """The finger-imitation release multiplier was removed; refuse ``--imit_release_min/start`` if given."""
    v = getattr(args_cli, "imit_release_min", None)
    p0 = getattr(args_cli, "imit_release_start", None)
    if v is None and p0 is None:
        return
    raise SystemExit(f"[IMIT_RELEASE] {ctx}: --imit_release_min/--imit_release_start were removed (got {v}, {p0})")
