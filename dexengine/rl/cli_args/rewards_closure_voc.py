"""Reward experiment flags, second half: closure bootstrap, RSI, finger gains, force curriculum, tracking hinge,
virtual object controller and the bimanual collision curriculum.

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case). dexengine.rl.args calls this right after rewards.add_reward_args."""

from __future__ import annotations


def add_closure_voc_args(parser) -> None:
    """Add the second half of the reward flags to ``parser`` (in launch-contract order)."""
    _add_closure_force_and_slip_args(parser)
    _add_closure_potential_args(parser)
    _add_rsi_args(parser)
    _add_finger_gain_args(parser)
    _add_force_curriculum_and_hinge_args(parser)
    _add_voc_args(parser)
    _add_voc_gate_args(parser)
    _add_voc_schedule_args(parser)
    _add_collision_curriculum_args(parser)


def _add_closure_force_and_slip_args(parser) -> None:
    """Closure force-onset reward and the slip penalty."""
    # -------- closure bootstrap A/B: force-onset vs closing-potential --------
    parser.add_argument(
        "--enable_closure_force",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF. 1 = add RewardClosureForce: (Σ_f w_f·min(‖F_f^obj‖,cap)/cap)/Σw_f "
        "on reference-contact frames, NO robot-contact gate — pays for any per-fingertip object-"
        "filtered force the moment it exists (closure bootstrap from the no-contact state). "
        "Requires --enable_contact_match_reward 1.",
    )
    parser.add_argument(
        "--closure_force_weight",
        type=float,
        default=2.0,
        help="weight of the force-onset finger-closure reward term. Default 2.0 "
        "(between contact_match 3.0 and approach 1.5).",
    )
    parser.add_argument(
        "--closure_force_cap",
        type=float,
        default=5.0,
        help="per-finger force cap (N) — payout saturates here (anti-slam). Default 5.0.",
    )
    parser.add_argument(
        "--closure_force_nonthumb_scale",
        type=float,
        default=1.5,
        help="up-weight index/middle/ring/little vs thumb 1.0 (drive >thumb+index). Default 1.5.",
    )
    parser.add_argument(
        "--enable_slip_penalty",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF. 1 = add RewardSlipPenalty: -min(||v_obj - v_palm||, cap)/cap "
        "while >= slip_min_fingers fingertips press the object (object-filtered force >= slip_f_gate). "
        "Pose-agnostic grasp-STABILITY term — pays for a slip-free hold regardless of finger pose, "
        "grasp location, or reference timeline. Requires the per-fingertip object contact sensors.",
    )
    parser.add_argument(
        "--slip_penalty_weight",
        type=float,
        default=2.0,
        help="weight of the slip penalty (term returns [-1,0]). Default 2.0.",
    )
    parser.add_argument(
        "--slip_cap",
        type=float,
        default=0.5,
        help="slip speed (m/s) at which the penalty saturates at -1. Default 0.5.",
    )
    parser.add_argument(
        "--slip_f_gate",
        type=float,
        default=0.05,
        help="per-fingertip object-filtered force (N) for a finger to count as holding. Default 0.05.",
    )


def _add_closure_potential_args(parser) -> None:
    """Slip finger count and the closing-potential ablation toggles."""
    parser.add_argument(
        "--slip_min_fingers",
        type=int,
        default=2,
        help="minimum holding fingertips for the slip gate to be active. Default 2 (thumb+one).",
    )
    parser.add_argument(
        "--enable_closure_potential",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF. 1 = add RewardClosurePotential: potential-difference "
        "(Σ_f m_f·clamp(d_f(t-1)-d_f(t),±c)/c)/Σm_f over NON-contacting fingers (F<0.05N), "
        "d = tip→object-surface distance, reference-contact frames only; stateful prev-dist "
        "buffer, reset-aware; hover yields 0, retreating costs. "
        "Requires --enable_contact_match_reward 1.",
    )
    parser.add_argument(
        "--closure_potential_weight",
        type=float,
        default=2.0,
        help="weight of the closing-potential reward. Default 2.0.",
    )
    parser.add_argument(
        "--closure_potential_clamp_cm",
        type=float,
        default=1.0,
        help="per-step distance-delta clamp ±c (cm); also the normalizer (c=clamp_cm/100 m). Default 1.0.",
    )
    parser.add_argument(
        "--closure_perfinger",
        type=int,
        default=0,
        choices=[0, 1],
        help="A1: 0(default)=legacy hand-level any() gate. 1 = PER-FINGER closure windows: each finger f "
        "rewarded only on [ref_start[f], ref_end[f]-margin_f]; closure_potential uses the GT contact "
        "TARGET distance (not nearest-surface) and is active pre-contact (F<0.05) from global_first; "
        "closure_force active once contacting (F>=0.05) and only after ref_start[f]; both end margin_f "
        "frames before ref_end[f]. Normalized by active-finger weight. Pair with --enable_approach_reward 0.",
    )
    parser.add_argument(
        "--closure_margins",
        type=str,
        default="2,5,5,5,5",
        help="A1: per-finger release margins (reference frames) 'thumb,index,middle,ring,pinky'. Each finger's "
        "closure ends margin_f frames before its ref-contact end, leaving room for natural release. "
        "Default '2,5,5,5,5' (thumb load-bearing → small margin).",
    )
    # --- Baseline-search ablation toggles (only affect the A1 per-finger closure path; default = A1 baseline B0) ---
    parser.add_argument(
        "--closure_force_immediate",
        type=int,
        default=0,
        choices=[0, 1],
        help="in the per-finger closure_force, DROP the af>=ref_start[f] lower bound — pay force as soon as "
        "object-contact force exists within [.., ref_end[f]-margin_f] for will-contact fingers. "
        "0(default)=A1 (force only after ref_start[f]). Still excludes never-ref-contact fingers.",
    )


def _add_rsi_args(parser) -> None:
    """Reference state initialization (RSI) start-frame sampling flags."""
    parser.add_argument(
        "--closure_potential_dist",
        type=str,
        default="gt_target",
        choices=["gt_target", "nearest_surface"],
        help="distance metric for the per-finger closure_potential. 'gt_target'(default,A1)=tip→per-finger "
        "GT contact target (object-relative); 'nearest_surface'=tip→nearest object-surface point "
        "(the previous/legacy distance logic).",
    )
    # -------- RSI (Reference State Initialization) experiment flags (arm path) --------
    # Requires the all-frame arm-IK precompute (the arm IK stage, all frames) so the arm
    # resets to the reference at the sampled start_frame. Patches cfg.curriculum.rsi_schedule.
    parser.add_argument(
        "--rsi_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF (start frame 0). 1 = enable RSI: reset episodes from frames sampled in "
        "the band [rsi_lo_frac,rsi_hi_frac]*(clip_len-1), with probability rsi_target_fraction.",
    )
    parser.add_argument(
        "--rsi_target_fraction",
        type=float,
        default=1.0,
        help="RSI: final fraction of episodes that start in the band. Default 1.0.",
    )
    parser.add_argument(
        "--rsi_fraction_start",
        type=float,
        default=None,
        help="RSI: initial rsi_fraction (None=linear ramp from 0). Set to anneal DOWN (e.g. 0.8->0.25).",
    )
    parser.add_argument(
        "--rsi_ramp_episodes",
        type=int,
        default=1,
        help="RSI: episodes to ramp rsi_fraction (default 1 = instant, for the constant-band diagnostic).",
    )
    parser.add_argument(
        "--rsi_lo_frac",
        type=float,
        default=0.0,
        help="RSI: initial band lower edge (fraction of clip_len-1). Default 0.0.",
    )
    parser.add_argument(
        "--rsi_hi_frac",
        type=float,
        default=1.0,
        help="RSI: initial band upper edge (fraction of clip_len-1). Default 1.0.",
    )
    parser.add_argument(
        "--rsi_lo_frac_end",
        type=float,
        default=None,
        help="RSI: final band lower edge (None=constant band). Reverse curriculum slides this to ~0.",
    )
    parser.add_argument(
        "--rsi_hi_frac_end", type=float, default=None, help="RSI: final band upper edge (None=constant band)."
    )


def _add_finger_gain_args(parser) -> None:
    """RSI band ramp and the finger actuation gains."""
    parser.add_argument(
        "--rsi_band_ramp_episodes",
        type=int,
        default=None,
        help="RSI: episodes to ramp the band (None=rsi_ramp_episodes).",
    )
    parser.add_argument(
        "--rsi_ramp_frac",
        type=float,
        default=None,
        help="RSI: ramp the fraction AND band over this FRACTION of total training timesteps "
        "(step-based, deterministic; preferred over episode-count). E.g. 0.7 = anneal over the "
        "first 70%% of training. Sets ramp_steps=band_ramp_steps=int(frac*total_timesteps).",
    )
    # -------- finger actuation gains (responsiveness ablation) --------
    parser.add_argument(
        "--finger_effort",
        type=float,
        default=None,
        help="Build cap: override the finger actuator effort_limit_sim (N·m, every finger joint) on the training "
        "path. Default None = the HandSpec applied cap (core.hands.finger_effort_applied). With "
        "--finger_effort_anneal_start it is the force-curriculum start cap. Per-hand values live only in "
        "core.hands.HAND_RL_RECIPES (dexengine.rl.launch.hand_recipe).",
    )
    parser.add_argument(
        "--finger_velocity",
        type=float,
        default=None,
        help="override the legacy RH56F1 finger velocity_limit_sim (rad/s). Default None=cfg (2.0). "
        "The pilot used 10.0. Real HW 10.",
    )
    parser.add_argument(
        "--finger_kd",
        type=float,
        default=None,
        help="override RH56F1 finger damping (Kd). Default None=cfg (30). The finger-gain ablation used 10.",
    )
    parser.add_argument(
        "--finger_kp",
        type=float,
        default=None,
        help="Optional: override RH56F1 finger stiffness (Kp). Default None=cfg (200). "
        "Not changed in the finger-gain ablation (raising Kp without effort headroom just saturates more).",
    )


def _add_force_curriculum_and_hinge_args(parser) -> None:
    """Finger effort anneal, the tracking hinge and the wandb run name."""
    # Force curriculum (only the final behaviour must stay within the hardware): early training uses
    # --finger_effort (above the hardware cap); once schedule_progress (gate curriculum progress: 0 -> 1 over the
    # TIGHTEN fraction of training, frozen eval fixed at 1.0) passes this value, the cap goes down linearly to the
    # hardware cap (core.hands.finger_effort_applied), reached at 1.0. Eval and render run at progress 1.0, so
    # always at the hardware cap. None (default) = off.
    parser.add_argument(
        "--finger_effort_anneal_start",
        type=float,
        default=None,
        help="Force curriculum: schedule_progress in [0,1) at which the finger effort cap starts ramping "
        "from the build cap (--finger_effort, required) down to the hand's hardware cap "
        "(core.hands.finger_effort_applied), reached at progress 1.0. Eval pins progress 1.0 => "
        "always hardware caps. Default None = off.",
    )
    # The next flags are the real TRACKING hinge (full reward while the error is inside it), unlike
    # --wrist_l2_hinge. The uni launch path does not set them; only the fingertip hinge is on there
    # (--fingertip_hinge_cm 2.0, a separate path).
    parser.add_argument(
        "--enable_hinge",
        type=int,
        default=0,
        choices=[0, 1],
        help="Contact-pilot TRACKING hinge: 0 (default) = OFF. 1 = set hinge_threshold_m on the 5 "
        "fingertip rewards (--hinge_fingertip_cm) and eef_pos (--hinge_wrist_cm). "
        "Not used on the uni launch path. "
        "Distinct from --wrist_l2_hinge, which is a residual-magnitude free zone.",
    )
    parser.add_argument(
        "--hinge_fingertip_cm",
        type=float,
        default=1.5,
        help="Contact-pilot: fingertip hinge threshold in cm (used when --enable_hinge 1 "
        "and/or --hinge_schedule 1). Default 1.5 cm. NOTE: the campaign path instead uses "
        "--fingertip_hinge_cm (2.0), applied in train_cfg_overrides; that one is live.",
    )
    parser.add_argument(
        "--hinge_wrist_cm",
        type=float,
        default=2.0,
        help="Contact-pilot: wrist (eef_pos) hinge threshold in cm (used when "
        "--enable_hinge 1). Default 2.0 cm. Applies to POSITION only — eef_rot accepts a "
        "hinge_threshold_rad but nothing ever sets it (no wiring in env_setup).",
    )
    parser.add_argument(
        "--hinge_schedule",
        type=int,
        default=0,
        choices=[0, 1],
        help="Contact-pilot: 0 (default) = no schedule. 1 = register a curriculum term "
        "linearly ramping the fingertip hinge_threshold_m from 0 → "
        "hinge_fingertip_cm/100 over the first ~40%% of --max_iterations.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Override the W&B run name. Default None = use --exp_name or the "
        "auto-generated name (no behavior change when unset).",
    )


def _add_voc_args(parser) -> None:
    """Virtual object controller (VOC) gains, decay and checks."""
    # -------- VOC: DexMachina-style Virtual Object Controller (arXiv 2505.24853) --------
    parser.add_argument(
        "--voc",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF. 1 = TRAIN-ONLY virtual object controller: each present object "
        "(right always; left when bimanual) is pulled toward its per-frame reference pose "
        "by an external PD wrench (world frame, root body) so a from-scratch policy can "
        "learn motion/contact alignment before taking over. kp decays x--voc_decay via a "
        "PERFORMANCE-GATED schedule (only while the tracking-error EMA is under the "
        "--voc_err_* gates); wrench is zeroed permanently once kp_pos < 1%% of initial. "
        "Never constructed in eval/render (wired only by the train runner).",
    )
    parser.add_argument(
        "--voc_kp_pos",
        type=float,
        default=200.0,
        help="VOC translational stiffness (N/m). kd_pos = 2*sqrt(kp_pos*m), m = cfg object mass (0.2 kg).",
    )
    parser.add_argument(
        "--voc_kp_rot",
        type=float,
        default=5.0,
        help="VOC rotational stiffness (Nm/rad). kd_rot = voc_kd_rot_scale*sqrt(kp_rot).",
    )
    parser.add_argument(
        "--voc_decay", type=float, default=0.8, help="VOC multiplicative kp decay applied on each PASSED gate check."
    )
    parser.add_argument(
        "--voc_check_iters",
        type=int,
        default=25,
        help="VOC gate-check interval in POLICY ITERATIONS (1 iter = 32 rollout steps; "
        "the train runner converts to common_step_counter units).",
    )
    parser.add_argument(
        "--voc_err_pos",
        type=float,
        default=0.03,
        help="VOC EMA position-error gate (m): decay fires only when the EMA pos err "
        "of the assisted objects is below this AND the rot gate passes.",
    )
    parser.add_argument(
        "--voc_err_rot_deg", type=float, default=20.0, help="VOC EMA geodesic rotation-error gate (deg)."
    )
    parser.add_argument(
        "--voc_kd_rot_scale", type=float, default=0.3, help="VOC rotational damping scale: kd_rot = scale*sqrt(kp_rot)."
    )


def _add_voc_gate_args(parser) -> None:
    """VOC force cap and contact gate."""
    # -------- VOC v2 (defaults keep v1 behavior: --voc 1 alone is unchanged) --------
    parser.add_argument(
        "--voc_force_cap_mg",
        type=float,
        default=0.0,
        help="VOC v2: cap the assist FORCE magnitude at cap*m*g per object (0=off=v1 "
        "uncapped). E.g. 0.6 -> VOC can stabilize a held object but never lift it "
        "alone (fixes v1 assist exploitation). Torque capped analogously at "
        "cap*kp_rot_init*0.2 Nm (= the torque a 0.2 rad rot error produces at the "
        "initial kp_rot, scaled by the same cap factor).",
    )
    parser.add_argument(
        "--voc_gate_mode",
        type=str,
        default="ema",
        choices=["ema", "moving"],
        help="VOC v2 decay-gate mode. 'ema' (default) = v1: population-mean tracking-error "
        "EMA over ALL frames. 'moving' = the EMA accumulates ONLY on (env,object) "
        "samples where the REFERENCE object moves >1cm/s (one-frame ref pose delta / "
        "step_dt) AND the mean object-filtered fingertip contact force on that side "
        "exceeds --voc_gate_contact_n — kills the v1 trivial-frame gate dilution.",
    )
    parser.add_argument(
        "--voc_gate_contact_n",
        type=float,
        default=0.3,
        help="VOC v2 'moving' gate: minimum mean object-filtered fingertip force (N, over "
        "the 5 per-fingertip object-filtered sensors) on the assisted side for a "
        "frame to count toward the decay EMA. Default 0.3 N.",
    )


def _add_voc_schedule_args(parser) -> None:
    """VOC flush fraction and schedule."""
    # -------- VOC v2.1 (default 0 keeps v2 behavior) --------
    parser.add_argument(
        "--voc_flush_frac",
        type=float,
        default=0.0,
        help="VOC v2.1 forced final flush (0=off=v2). During the LAST this-fraction of "
        "training (common_step_counter vs max_iterations*32, the ramp clock) the "
        "performance gate is OVERRIDDEN: kp decays by a per-check factor sized so "
        "kp_pos reaches <0.5%% of initial within the FIRST HALF of the flush window, "
        "then the wrench is hard-zeroed permanently — training always ends with an "
        "assist-free finetune tail (v2 defect: gated decay never reached zero -> "
        "assist-dependent policies). Also restricts best_agent.pt: frozen-eval best "
        "updates (and skrl's reward fallback) are suppressed until assist is OFF, "
        "since pre-flush evals run WITH assist.",
    )
    # -------- VOC v3 (default 'global' keeps v2.1 behavior) --------
    parser.add_argument(
        "--voc_schedule",
        type=str,
        default="global",
        choices=["global", "uniform"],
        help="VOC v3 gain schedule. 'global' (default) = single kp for all envs (v1/v2/v2.1 "
        "behavior). 'uniform' = DexMachina-default per-env gain randomization: the "
        "curriculum kp becomes the UPPER bound kp_hi (gated decay + flush machinery "
        "unchanged, operating on kp_hi); lower bound kp_lo = kp_hi*max(0, 1-progress/0.25) "
        "(exact 0 by 25%% of training, stays 0); each env samples kp_env ~ U[kp_lo, kp_hi] "
        "at ITS reset — zero-assist envs are in-batch from early training on, so the "
        "zero-assist regime is in-distribution before kp_hi reaches it. Gains are NOT in "
        "observations. Status print at gate-check cadence: '[VOC] uniform: kp_hi=... "
        "kp_lo=... zero-env-frac=...'.",
    )


def _add_collision_curriculum_args(parser) -> None:
    """Bimanual collision curriculum."""
    # -------- Collision curriculum (bimanual TRAIN only) --------
    parser.add_argument(
        "--collision_curriculum",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF. 1 = bimanual TRAIN-ONLY "
        "collision curriculum: cross-pairs (right-hand<->left-hand links incl. link7, "
        "right-hand<->LEFT object, left-hand<->RIGHT object) start FILTERED OFF and are "
        "one-shot re-ENABLED at --collision_curriculum_on_frac of training via a runtime "
        "FilteredPairsAPI edit on the env_0 template (propagates to all clones, no sim "
        "re-init). Hand<->own-object, desk/ground and all uni pairs stay ON. Requires "
        "--bimanual 1; never wired in eval/render (full collision there).",
    )
    parser.add_argument(
        "--collision_curriculum_on_frac",
        type=str,
        default="1.0",
        help="Training-progress fraction (of max_iterations*rollouts common steps) at which "
        "the cross-pair filters are removed. Canonical 1.0 = cross pairs isolated for the whole training. "
        'Two values "start,end" make a RAMP — e.g. 0.25,0.75 = start enabling at 25%% progress, every '
        "env ON at 75%% (final physics = strict). The ramp enables per env (clip<->env binding is static, "
        "so it is the fraction of clips with collision on). The env var DEXENGINE_COLLCURR_RAMP gives the "
        "same; this flag wins. The cross-pair start-OFF authoring always applies to every env.",
    )
    parser.add_argument(
        "--collision_curriculum_objpair",
        type=int,
        default=0,
        choices=[0, 1],
        help="0(default)=OFF (base curriculum unchanged). 1 = FULL ISOLATION: ALSO filter "
        "the object_right<->object_left rigid-body pair at start (closes the remaining "
        "cross channel — partner mishandling its object into the other side's reference "
        "path — so joint training decomposes into two independent uni-equivalent "
        "problems) and re-enable it at the SAME --collision_curriculum_on_frac flip. "
        "Hand<->own-object and everything<->desk/ground stay ON. Requires "
        "--collision_curriculum 1 (fail-fast otherwise).",
    )
