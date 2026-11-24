"""Object-tracking termination and thresholds and the stability anchor (the rest: tracking_ppo).

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case)."""

from __future__ import annotations


def add_tracking_args(parser) -> None:
    """Add the tracking flags to ``parser`` (in launch-contract order)."""
    _add_obj_track_args(parser)
    _add_obj_track_schedule_args(parser)
    _add_obj_lag_and_tip_term_args(parser)
    _add_residual_l2_and_ramp_args(parser)
    _add_residual_floor_and_passive_args(parser)
    _add_left_residual_l2_args(parser)


def _add_obj_track_args(parser) -> None:
    """Object-tracking termination loosening and constant thresholds (right side)."""
    # -------- object_tracking loosening so episodes survive past grasp onset --------
    parser.add_argument(
        "--obj_track_pos_scale",
        type=float,
        default=1.0,
        help="multiply ObjectTrackingExceeded pos_start/pos_end "
        "thresholds. >1 lets the sim object diverge further from the "
        "reference before terminating, so the episode survives into the "
        "manipulation phase where a grasp can be learned. Default 1.0 (off).",
    )
    parser.add_argument(
        "--obj_track_rot_scale",
        type=float,
        default=1.0,
        help="multiply ObjectTrackingExceeded rot_start_deg/rot_end_deg. Default 1.0 (off).",
    )
    parser.add_argument(
        "--obj_track_grace",
        type=int,
        default=None,
        help="override ObjectTrackingExceeded grace_period (steps "
        "before termination can fire). None = keep cfg default.",
    )
    parser.add_argument(
        "--disable_obj_rot_termination",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 (default) = ROTATION component of object_tracking active. 1 = DISABLE rotation "
        "termination (sets the rot threshold effectively infinite so geodesic angle_err "
        "<=180deg never exceeds it; object_tracking reduces to POSITION-only). NOT a 0-deg "
        "threshold (that would fire on any rotation). Applied in both train and eval.",
    )
    parser.add_argument(
        "--obj_rot_thresh_const",
        type=float,
        default=90.0,
        help="Canonical 90°. 0 = use the cfg rot schedule. >0 = set a CONSTANT object-tracking "
        "rotation threshold (deg) with NO annealing (rot_start=rot_end=value, same in train "
        "and eval). e.g. 180 = max geodesic angle, so rotation effectively never terminates "
        "(a real physical constant vs the 1e6 disable sentinel). Takes precedence over "
        "--disable_obj_rot_termination. Position schedule untouched.",
    )
    parser.add_argument(
        "--obj_pos_thresh_const",
        type=float,
        default=0.05,
        help="Canonical 0.05 m. 0 = use the cfg pos schedule. >0 = set a CONSTANT object-tracking "
        "POSITION threshold (METERS) with NO annealing (pos_start=pos_end=value, same in "
        "train and eval). e.g. 0.07 = 7cm. Used to extremely loosen pose termination (with "
        "rot const 180 + contact term) so the hand has max freedom to lift. Takes precedence "
        "over --obj_track_pos_scale.",
    )


def _add_obj_track_schedule_args(parser) -> None:
    """Left constant thresholds, tightening start values/stairs and the look-ahead lag window."""
    parser.add_argument(
        "--obj_rot_thresh_const_left",
        type=float,
        default=0.0,
        help="0 (default) = LEFT object_tracking_left follows --obj_rot_thresh_const. >0 = set the "
        "LEFT rotation threshold (deg) independently. Needed because ObjTermAdaptive drives the "
        "RIGHT gate only — training-parity strict evals want right=floor, left=baseline.",
    )
    parser.add_argument(
        "--obj_pos_thresh_const_left",
        type=float,
        default=0.0,
        help="0 (default) = LEFT object_tracking_left follows --obj_pos_thresh_const. >0 = set the "
        "LEFT POSITION threshold (METERS) independently (e.g. 0.10 baseline while right=0.05).",
    )
    parser.add_argument(
        "--obj_pos_thresh_start",
        type=float,
        default=0.0,
        help="With --obj_pos_thresh_const: set the RIGHT pos_START (METERS) while pos_end stays at "
        "the const -> LINEAR ramp start->const driven by threshold_tightening "
        "(--tighten_steps_scale; frozen-eval still forces progress=1 = end gate). "
        "LEFT (explicit left consts) unaffected. 0 (default) = off (const start=end).",
    )
    parser.add_argument(
        "--tighten_stairs",
        type=int,
        default=0,
        help="0 (default, off) = LINEAR threshold_tightening. N>1 = STAIRCASE: split "
        "[0,1] into N intervals, hold the gate constant inside each (plateau) and "
        "tighten only at boundaries; the last interval sits at the final gate. "
        "Tests whether the policy needs a fixed target to consolidate before the "
        "next tightening step.",
    )
    parser.add_argument(
        "--obj_rot_thresh_start",
        type=float,
        default=0.0,
        help="With --obj_rot_thresh_const: set the RIGHT rot_START (deg) similarly for a linear "
        "rot ramp (e.g. 180 -> const 60). LEFT (explicit left consts) unaffected. 0 = off.",
    )
    parser.add_argument(
        "--obj_lag_window_ahead",
        type=int,
        default=5,
        help="Canonical 5. Forward extension of the lag-window gate: [F-W, F+ahead] instead of "
        "[F-W, F]. Pair with --wrist_ref_lookahead_frames (same value) so an "
        "early-arriving object under the phase-lead base is not punished. "
        "Applies train AND eval. 0 = trailing-only.",
    )


def _add_obj_lag_and_tip_term_args(parser) -> None:
    """Lag window, the object-tracking split log and the fingertip termination switch."""
    parser.add_argument(
        "--obj_lag_window",
        type=int,
        default=25,
        help="Canonical 25 frames (0.5 s window gate; the judge gate). "
        "0 = instantaneous object-tracking gate (legacy). "
        ">0 = lag-tolerant pose-match gate: the env survives if ANY reference frame in the "
        "trailing window [F-W, F] matches the CURRENT object pose (pos_err<=thresh AND "
        "rot_err<=thresh at the SAME window frame). W in reference frames = policy steps "
        "(25 = 0.5 s @ 50 Hz). Applied to BOTH side gates, train and eval alike "
        "(the windowed gate IS the verdict gate).",
    )
    parser.add_argument(
        "--log_objtrack_split",
        type=int,
        default=1,
        choices=[0, 1],
        help="log object_tracking pos-vs-rot split as diag reward terms "
        "(diag_objtrack_{pos_err,rot_err,pos_exc,rot_exc}, weight 1e-9; recover "
        "×1e9). Matches the live (post-scale) ObjectTrackingExceeded thresholds. "
        "Default 1 (on). Original termination behavior unchanged.",
    )
    parser.add_argument(
        "--disable_fingertip_termination",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 (default)=OFF. 1 = REMOVE the FingertipTrackingExceeded "
        "termination so the hand can leave the (morphology-mismatched MANO) "
        "fingertip poses and discover a robot-appropriate grasp. object_tracking "
        "(task) + velocity guards stay. Fingertip-tracking REWARD unchanged.",
    )


def _add_residual_l2_and_ramp_args(parser) -> None:
    """Residual L2 anchor and the residual ramp (*_end) flags."""
    # -------- stability (soft anchor against actor-gain collapse) --------
    parser.add_argument(
        "--residual_l2_weight",
        type=float,
        default=0.0,
        help="coefficient for the soft residual-L2 penalty term "
        "(0 = off). Penalty = -weight*relu(||residual||-deadzone)^2. "
        "Softly discourages actor-gain drift. Try 0.05-0.2.",
    )
    parser.add_argument(
        "--residual_l2_deadzone",
        type=float,
        default=0.5,
        help="dead-zone for the residual-L2 penalty. ||residual|| "
        "below this is unpenalized (policy free to deviate). Try 0.5-0.9 "
        "(healthy plateau ||mu||~0.2-0.9; collapse onset ~1.3).",
    )
    parser.add_argument(
        "--wrist_pos_residual_max_m",
        type=float,
        default=0.07,
        help="Hard +/- cap (m) on wrist POSITION residual from reference; "
        "clamp(raw,-1,1)*cap. Canonical 0.07 (a checkpoint contract: train, eval and render must match). "
        "Checkpoints of the older 0.05/60 generation need those values passed.",
    )
    parser.add_argument(
        "--wrist_rot_residual_max_deg",
        type=float,
        default=120.0,
        help="Hard +/- cap (deg) on wrist ROTATION residual (geodesic, Rodrigues). "
        "Canonical 120 (a checkpoint contract).",
    )
    # Residual ramps: each *_end flag interpolates linearly from the start value to the end value over the gate
    # curriculum progress (schedule_progress: 0 -> 1 over the first 50 % of training; eval fixed at 1.0).
    # None (default) = no ramp. Wrist authority is set by the caps (with a cap set, wrist_residual_scale is ignored).
    parser.add_argument(
        "--finger_residual_scale_end",
        type=float,
        default=None,
        help="Ramp end for FINGER residual scale (start = --finger_residual_scale).",
    )
    parser.add_argument(
        "--wrist_pos_residual_max_m_end",
        type=float,
        default=None,
        help="Ramp end (m) for the wrist POSITION residual cap.",
    )
    parser.add_argument(
        "--wrist_rot_residual_max_deg_end",
        type=float,
        default=None,
        help="Ramp end (deg) for the wrist ROTATION residual cap.",
    )


def _add_residual_floor_and_passive_args(parser) -> None:
    """Finger ROM floor, wrist residual L2, left GT following and the auto-passive hand."""
    parser.add_argument(
        "--finger_rom_floor",
        type=float,
        default=None,
        help="Floor the FINGER action-mapping LOWER limits at this "
        "value (rad) — human-ROM approximation (blocks wuji's -27deg "
        "hyperextension). AA joints (finger[2-5]_joint2) excluded by default. "
        "ACTION-CONTRACT param (train/eval must match). None = unchanged.",
    )
    parser.add_argument(
        "--wrist_residual_l2_weight",
        type=float,
        default=None,
        help="If set, SPLIT residual_l2 into wrist(0:6) + finger (span/weight scale = "
        "--finger_penalty_span, per side); this is the WRIST l2 weight. "
        "None=combined l2 (legacy).",
    )
    parser.add_argument(
        "--left_gt_following",
        type=int,
        default=0,
        choices=[0, 1],
        help="Bimanual only: LEFT hand executes the pure reference (GT) trajectory — wrist AND "
        "finger residual scales forced to 0 so the left tower's output has no effect. Use when "
        "the left object is static in the reference (hold/no-contact): halves the effective "
        "action space and removes left reward noise from the advantage. Right hand unaffected.",
    )
    parser.add_argument(
        "--auto_passive_hand",
        type=int,
        default=0,
        choices=[0, 1],
        help="Bimanual only: auto-decide per-hand passivity from the REFERENCE object motion. For "
        "each side, if the reference object's max displacement < --auto_passive_pos_cm AND max "
        "rotation < --auto_passive_rot_deg, that hand is made GT-following (residual scales/caps "
        "= 0, like --left_gt_following) — a hand whose object barely moves executes pure "
        "reference. Generalizes --left_gt_following to whichever hand is (near-)static; supports "
        "left-manipulates and both-active clips automatically. Motion read from "
        "<data_layout.taco_source_root()>/<seq>/0/trajectory_kinematic.npz "
        "(the TACO base layer · bimanual TACO qpos layout).",
    )
    parser.add_argument(
        "--auto_passive_pos_cm",
        type=float,
        default=5.0,
        help="auto_passive_hand: per-side max object displacement (cm) below which the hand is passive.",
    )
    parser.add_argument(
        "--auto_passive_rot_deg",
        type=float,
        default=45.0,
        help="auto_passive_hand: per-side max object rotation (deg) below which the hand is passive.",
    )


def _add_left_residual_l2_args(parser) -> None:
    """Left-side residual L2 scale."""
    parser.add_argument(
        "--residual_l2_left_scale",
        type=float,
        default=1.0,
        help="Bimanual only: multiplier on the LEFT residual-L2 anchor weights (finger+wrist). "
        "The left tower's advantage is noise-dominated (summed reward, near-static left object), "
        "so it needs a stronger anchor than the right (which keeps --residual_l2_weight as-is).",
    )
