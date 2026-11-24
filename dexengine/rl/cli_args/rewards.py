"""Reward / observation experiment flags, first half: fixed object-observation layout, force-direction, pre-grasp,
grasp geometry, pinch-aperture and adaptive termination tightening (the rest: rewards_closure_voc).

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case)."""

from __future__ import annotations


def add_reward_args(parser) -> None:
    """Add the rewards flags to ``parser`` (in launch-contract order)."""
    _add_obs_layout_and_force_dir_args(parser)
    _add_pregrasp_args(parser)
    _add_grasp_geometry_args(parser)
    _add_aperture_and_follow_args(parser)
    _add_ramp_level_args(parser)
    _add_adaptive_term_args(parser)
    _add_adaptive_backoff_and_keypoint_args(parser)


def _add_obs_layout_and_force_dir_args(parser) -> None:
    """Fixed object-observation layout and the force-direction reward."""
    # -------- fixed object-observation layout (rigid and articulated clips, one width) --------
    parser.add_argument(
        "--obj_obs_fixed",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = FIXED 87+3 for every object: (θ, ref−θ, valid bit), rigid = 0,0,0 — so rigid "
        "and articulated clips share one width. 0 = the object observation is 87 dims, 89 for an "
        "articulated object (θ, ref−θ appended) — the layout earlier checkpoints were trained with. "
        "An observation-layout (checkpoint) contract: training and every eval/render of that checkpoint must use "
        "the same value "
        "(a reload of another width dies, [CKPT-OBS]). Owner = the recipe key OBJ_OBS_FIXED "
        "(configs/rl/recipe.yaml); the launcher always passes the recipe value, so this argparse "
        "default (0) only applies to a bare trainer call.",
    )
    # -------- force-direction reward (hand-on-object force aligned with reference motion) --------
    parser.add_argument(
        "--enable_force_dir_reward",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 (default)=OFF. 1 = add RewardForceDirection: clip(0.5+0.5*F_hand·F̂_req/F_cap,0,1)*gate2. "
        "F_req/m = a_ref + g·ẑ — the direction of the force the reference REQUIRES (the motion direction "
        "would flip sign while decelerating); the mass cancels out of the direction. "
        "gated to the post-contact window + reference-motion, soft-scaled by thumb+index contact. "
        "Requires --enable_contact_match_reward 1. Adds force diagnostics too.",
    )
    parser.add_argument(
        "--force_reward_weight",
        type=float,
        default=2.5,
        help="weight of the force-direction reward. Default 2.5 (≈obj_pos weight).",
    )
    parser.add_argument(
        "--force_cap",
        type=float,
        default=30.0,
        help="projection cap (N) for force reward normalization clip(proj/cap,0,1). Default 30.",
    )
    parser.add_argument(
        "--force_k",
        type=int,
        default=3,
        help="look-ahead frames for the reference object-motion direction d_ref. Default 3.",
    )
    parser.add_argument(
        "--force_motion_eps",
        type=float,
        default=0.003,
        help="min ‖d_ref‖ (m over k frames) for the force reward to be active (skip static frames). Default 0.003 "
        "(0.3cm).",
    )
    parser.add_argument(
        "--force_window_pre",
        type=int,
        default=3,
        help="force-window start = first_ref_contact_frame - this. Default 3.",
    )


def _add_pregrasp_args(parser) -> None:
    """Force post window and the pre-grasp palm reward."""
    parser.add_argument(
        "--force_window_post",
        type=int,
        default=100000,
        help="force-window end = first_ref_contact_frame + this. Canonical 100000 (= to the clip end; "
        "50 covers only grasp->lift).",
    )
    # -------- pre-grasp palm reward (give the wrist a graspable-pose incentive) --------
    parser.add_argument(
        "--enable_pregrasp_reward",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 (default)=OFF. 1 = add RewardPreGraspPalm: exp(-k*relu(d_palm-d_grasp)), "
        "pulls palm (wrist_rot_z) to within finger reach of the live witness grasp centroid so "
        "fingers can wrap. Gives the active wrist residual a reason to leave the MANO reference. "
        "Requires --enable_contact_match_reward 1. Adds diag_palm_dist.",
    )
    parser.add_argument(
        "--pregrasp_weight",
        type=float,
        default=2.0,
        help="weight of the pre-grasp palm reward. Default 2.0. Must compete with tracking pull.",
    )
    parser.add_argument(
        "--pregrasp_k",
        type=float,
        default=8.0,
        help="softness k for exp(-k*relu(d_palm-d_grasp)). Default 8 (gradient meaningful at ~10cm).",
    )
    parser.add_argument(
        "--pregrasp_dist",
        type=float,
        default=0.12,
        help="target palm->grasp-centroid distance (m, about one finger reach). Reward flat once within it. "
        "Default 0.12.",
    )


def _add_grasp_geometry_args(parser) -> None:
    """Wrap-ready pose and object-relative grasp geometry rewards."""
    # -------- wrap-ready pose + object-relative grasp geometry --------
    parser.add_argument(
        "--enable_grasp_geom_reward",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF. 1 = add RewardGraspSurface (will-contact fingertips->object surface, "
        "non-thumb up-weighted) + RewardPreGraspOrient (palm faces grasp centroid, pre-contact). "
        "Adds diag_n_fingers + diag_oppo. Requires --enable_contact_match_reward 1.",
    )
    parser.add_argument(
        "--grasp_surface_weight",
        type=float,
        default=3.0,
        help="weight of the multi-finger surface-proximity reward (H-A). Default 3.0.",
    )
    parser.add_argument(
        "--grasp_surface_k",
        type=float,
        default=40.0,
        help="softness for exp(-k*relu(d_tip_surface)). Default 40 (~2.5cm scale).",
    )
    parser.add_argument(
        "--grasp_nonthumb_scale",
        type=float,
        default=1.5,
        help="up-weight non-thumb fingers in the surface reward (drive >thumb+index). Default 1.5.",
    )
    parser.add_argument(
        "--pregrasp_orient_weight",
        type=float,
        default=2.0,
        help="weight of the wrap-ready palm-orientation reward (pre-contact). Default 2.0.",
    )
    parser.add_argument(
        "--enable_opposition",
        type=int,
        default=0,
        choices=[0, 1],
        help="0=OFF (15a), 1=ON (15b) add RewardGraspOpposition (thumb-vs-finger opposition, "
        "proximity-gated). Requires --enable_grasp_geom_reward 1.",
    )
    parser.add_argument(
        "--grasp_oppo_weight",
        type=float,
        default=2.0,
        help="weight of the opposition reward (15b only). Default 2.0.",
    )
    parser.add_argument(
        "--grasp_oppo_near_m",
        type=float,
        default=0.03,
        help="proximity gate (m) for the opposition reward (both thumb & a finger within this). Default 0.03.",
    )


def _add_aperture_and_follow_args(parser) -> None:
    """Pinch-aperture pose reward, EEF weight scale and object-follow schedule."""
    # -------- pinch-aperture pose reward (wrist-pose lever) --------
    parser.add_argument(
        "--enable_aperture_reward",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF. 1 = add RewardGraspAperture: exp(-k*||grasp_centroid - pinch_midpoint||), "
        "pulls the wrist so the object sits BETWEEN thumb & finger tips (closing pinches it, not pokes from "
        "above). Adds diag_aperture_dist. Requires --enable_contact_match_reward 1.",
    )
    parser.add_argument(
        "--aperture_weight",
        type=float,
        default=4.0,
        help="weight of the pinch-aperture pose reward. Default 4.0.",
    )
    parser.add_argument(
        "--aperture_k",
        type=float,
        default=10.0,
        help="softness for exp(-k*||centroid-midpoint||). Default 10 (~10cm scale).",
    )
    parser.add_argument(
        "--aperture_target",
        type=str,
        default="mano",
        choices=["mano", "surface"],
        help="Re-anchor: target for the pinch midpoint. 'mano'=MANO witness grasp_centroid ("
        "non-graspable poke spot); 'surface'=nearest OBJECT-SURFACE point (reachable; reward pinch closing "
        "ON the object).",
    )
    parser.add_argument(
        "--eef_weight_scale",
        type=float,
        default=1.0,
        help="multiply the eef_pos & eef_rot wrist-tracking reward weights by this (default 1.0=off). "
        "<1 loosens the wrist-pin so the policy can deviate to a graspable pose (trades reference-tracking). "
        "Applied in the cfg-override block.",
    )
    parser.add_argument(
        "--obj_follow_scale",
        type=float,
        default=1.0,
        help="exp/obj-reward-boost: multiply the OBJECT-FOLLOWING reward weights (obj_pos, obj_rot, obj_vel, "
        "obj_ang_vel — all reference-tracking exp(-k*|ref-cur|) terms) by this (default 1.0=off). >1 makes "
        "object-following dominate more vs closure_force. Does NOT touch closure/contact/pregrasp/force_dir/"
        "residual_l2 or termination thresholds. Applied in the cfg-override block.",
    )
    parser.add_argument(
        "--obj_follow_schedule_target",
        type=float,
        default=0.0,
        help="exp/obj-reward-ramp: if >1, SCHEDULE the object-following weight scale 1x -> this target over the "
        "first 75%% of training, then hold (ObjFollowSchedule curriculum). Scales the SAME 4 terms as "
        "--obj_follow_scale (obj_pos/obj_rot/obj_vel/obj_ang_vel). Mutually exclusive with the fixed "
        "--obj_follow_scale. Logs the live scale to TensorBoard (Curriculum/obj_follow_schedule). 0=off.",
    )


def _add_ramp_level_args(parser) -> None:
    """Level-1/2 reward ramps."""
    parser.add_argument(
        "--ramp_level12",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "0 (default) = OFF, behavior identical to current (level_2 removed by contact-pilot, "
            "level_1 static at base weight). 1 = re-enable level_2 (weight 0.3 kept in contact-pilot "
            "block) and ramp BOTH level_1 (base 0.5) and level_2 (base 0.3) weights linearly from "
            "their base values down to 0 over the first 75%% of training (same window as "
            "ObjFollowSchedule), then hold at 0. Implemented via Level12RampDown curriculum term "
            "registered as cfg.curriculum.level12_ramp_down."
        ),
    )
    parser.add_argument(
        "--ramp_level12_frac",
        type=float,
        default=0.75,
        help=(
            "fraction of total training over which level_1/level_2 ramp from base down to 0 "
            "(only used when --ramp_level12 1). Default 0.75 (= same window as ObjFollowSchedule). "
            "Smaller = STEEPER ramp that finishes earlier (e.g. 0.30 = finger-pose guidance only "
            "over the first 30%% of training, then pure object-following). Held at 0 after the window."
        ),
    )
    parser.add_argument(
        "--ramp_level1",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Variant: ramp ONLY level_1 (base 0.5) from base down to 0 over --ramp_level12_frac of "
            "training, then hold at 0. Unlike --ramp_level12, this does NOT re-enable level_2 (level_2 "
            "stays removed by contact_pilot default). Use when the baseline already has level_2 off and "
            "you only want to fade out the level_1 finger-pose guidance. Mutually-exclusive with --ramp_level12."
        ),
    )


def _add_adaptive_term_args(parser) -> None:
    """Performance-gated adaptive termination tightening (steps, floors, margins, tip veto)."""
    # ---- adaptive-term v3: performance-gated INDEPENDENT
    # pos/rot termination tightening with PHYSICAL steps/floors + tip-drag veto + optional backoff.
    parser.add_argument(
        "--obj_term_adaptive",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = enable ObjTermAdaptive v3: after each frozen eval (fixed-100%% probe gate), tighten the "
        "object_tracking pos/rot thresholds INDEPENDENTLY by fixed physical steps when completion+ep_len "
        "improve, no recent downtrend, completion floor holds, tip-drag veto passes, and the axis q80 "
        "peak error fits inside the proposed gate. Requires --eval_threshold_progress 0.",
    )
    parser.add_argument(
        "--obj_term_adaptive_n",
        type=int,
        default=2,
        help="Comparison window (evals): previous n vs current n probe means. "
        "v2 result: n=2 stable, n=1 over-tightens.",
    )
    parser.add_argument(
        "--obj_term_adaptive_pos_step_cm",
        type=float,
        default=0.2,
        help="Translation tighten step (cm) per decision. <=0 DISABLES the translation axis (rot-only ablation).",
    )
    parser.add_argument(
        "--obj_term_adaptive_rot_step_deg",
        type=float,
        default=3.0,
        help="Rotation tighten step (deg) per decision. <=0 DISABLES the rotation axis (pos-only ablation).",
    )
    parser.add_argument(
        "--obj_term_adaptive_pos_floor_cm",
        type=float,
        default=3.0,
        help="Translation threshold floor (cm) — never tightens below this.",
    )
    parser.add_argument(
        "--obj_term_adaptive_rot_floor_deg", type=float, default=30.0, help="Rotation threshold floor (deg)."
    )
    parser.add_argument(
        "--obj_term_adaptive_margin_pos_cm",
        type=float,
        default=0.0,
        help="q80 capacity-veto margin, translation (cm). 0 = none (v2-final).",
    )
    parser.add_argument(
        "--obj_term_adaptive_margin_rot_deg",
        type=float,
        default=0.0,
        help="q80 capacity-veto margin, rotation (deg). 0 = none (v2-final).",
    )
    parser.add_argument(
        "--obj_term_adaptive_tip_veto",
        type=float,
        default=0.30,
        help="Tip-drag veto: tighten only when recent eval drag_q50 <= this (anti-low-skim, v3 fix for the "
        "v2 'gate gamed via low-skim' failure). Negative = veto OFF (v2-replica ablation).",
    )


def _add_adaptive_backoff_and_keypoint_args(parser) -> None:
    """Adaptive-term backoff and the object keypoint reward."""
    parser.add_argument(
        "--obj_term_adaptive_backoff",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 (default) = on completion crash (< floor-0.20) loosen both axes one step (reason 9). 0 = off.",
    )
    parser.add_argument(
        "--obj_term_adaptive_unconditional",
        type=int,
        default=0,
        choices=[0, 1],
        help="ADAPTIVITY ABLATION: 1 = skip ALL gates (improvement/floor/tip/q80) -> pure eval-clocked "
        "fixed-rate ramp (-step per eval until floor). Isolates whether performance-GATING matters "
        "vs plain scheduled tightening.",
    )
    parser.add_argument(
        "--obj_keypoint_reward",
        type=int,
        default=0,
        choices=[0, 1],
        help="exp/obj-keypoint-reward: 0(default)=OFF. 1=add K=2 FPS object-keypoint position rewards "
        "(RewardObjKeypoint, kp0+kp1 from keypoints_fps2.npy) each with weight=1.25 (=W/2 of obj_pos "
        "base weight 2.5). Ramps identically to obj_pos via ObjFollowSchedule. k=80. Also registers "
        "3 diagnostic terms (weight 1e-9): object_keypoint_1_err, object_keypoint_2_err, "
        "object_keypoint_mean_err (cm). Requires --obj_follow_schedule_target >1 to ramp.",
    )
    parser.add_argument(
        "--obj_keypoint_mode",
        type=str,
        default="both",
        choices=["both", "farthest"],
        help="exp/obj-keypoint-reward keypoint set. 'both'(default)=register BOTH FPS keypoints. 'farthest'="
        "register ONLY the single keypoint whose object-local position is FARTHEST from the object "
        "centroid (the hardest extremity to track; for the knife = the long handle end). Avoids the "
        "policy satisfying the easy keypoint while missing the hard one.",
    )
    parser.add_argument(
        "--obj_keypoint_weight",
        type=float,
        default=1.25,
        help="exp/obj-keypoint-reward per-keypoint base weight (default 1.25 = W/2 of obj_pos 2.5). In "
        "'farthest' mode use 2.5 to match the center weight (center:keypoint = 2.5:2.5). Ramped "
        "identically to obj_pos via ObjFollowSchedule.",
    )
