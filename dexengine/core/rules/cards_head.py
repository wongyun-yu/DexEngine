"""Rule list, first half in declaration order (assembled in dexengine.core.rules.cards)."""

from __future__ import annotations

from dexengine.core.rules.profiles import _DESK_OWNER
from dexengine.core.rules.reader_calls import READER_CALL_SCOPE
from dexengine.core.rules.spec import Rule

RULES_HEAD = [
    Rule(
        "D1-g",
        "oakink2 part cluster → one articulated object (fixed attachments included)",
        "base",
        "dexengine/data/oakink2/artic_convert.py:convert_tree",
        datasets=("oakink2",),
        thresholds=(
            ("fit residual constant", "dexengine/data/contact/geom.py:ROT_MIN_DEG"),
            ("one-body rotation threshold (°)", "dexengine/data/oakink2/artic_fit.py:ROT_TOL_DEG"),
            ("one-body translation threshold (m)", "dexengine/data/oakink2/artic_fit.py:POS_TOL_M"),
            ("sliding-rail play, rotation (°)", "dexengine/data/oakink2/part_kinds.py:SLIDE_PLAY_ROT_DEG"),
            ("sliding-rail play, translation (m)", "dexengine/data/oakink2/part_kinds.py:SLIDE_PLAY_POS_M"),
            ("lever tip displacement cap (m)", "dexengine/data/oakink2/part_kinds.py:LEVER_TIP_M"),
            (
                "cap on the lever's share in driving the jointed part",
                "dexengine/data/oakink2/part_kinds.py:LEVER_DRIVE_SHARE",
            ),
        ),
    ),
    Rule(
        "D12-a",
        "First-pass drop: segments with no manipulation (BI_NONE)",
        "candidates(first pass)",
        "dexengine/data/candidate_rules.py:CASE_DROP_PREFIX",
    ),
    Rule(
        "D12-c",
        "First-pass drop: poor source quality (reference mismatch · keypoint glitch · retarget error · long "
        "GT gap of a moving object, etc.)",
        "candidates(first pass)",
        "dexengine/data/candidate_rules.py:QUALITY_FLAGS_DROP",
        thresholds=(
            ("retargeting fingertip error cap (mm)", "dexengine/data/filtering.py:TIP_ERR_MM"),
            ("finger glitch size (mm)", "dexengine/data/filtering.py:GLITCH_MM"),
            ("wrist rotation cap for a glitch (°)", "dexengine/data/filtering.py:GLITCH_WRIST_DEG"),
            ("glitch minimum frames", "dexengine/data/filtering.py:GLITCH_MIN_FRAMES"),
        ),
    ),
    Rule(
        "D12-g",
        "Object source GT gaps (short or static = interpolate / long gap while moving = quality flag object_gt_gap)",
        "readers·quality(4b)",
        "dexengine/data/readers/hot3d.py:_fill_gt_gaps",
        scope_from=READER_CALL_SCOPE,  # every dataset whose reader calls the owner (the gap fill)
        thresholds=(
            ("max gap to interpolate (s)", "dexengine/data/readers/hot3d.py:OBJ_GAP_INTERP_MAX_S"),
            ("static test translation (m)", "dexengine/data/readers/hot3d.py:OBJ_GAP_STATIC_POS_M"),
            ("static test rotation (°)", "dexengine/data/readers/hot3d.py:OBJ_GAP_STATIC_ROT_DEG"),
        ),
    ),
    Rule(
        "D12-d",
        "First-pass drop: shorter than 1 s",
        "candidates(first pass)",
        "dexengine/data/candidate_rules.py:MIN_DURATION_S",
        thresholds=(("minimum length s", "dexengine/data/candidate_rules.py:MIN_DURATION_S"),),
    ),
    Rule(
        "D10-b",
        "First-pass drop: GRAB raw desk differs from the oracle",
        "candidates(first pass)",
        "dexengine/data/support/grab.py:ORACLE_DROP_M",
        datasets=("grab",),
        scenes=("table",),
        thresholds=(("oracle difference", "dexengine/data/support/grab.py:ORACLE_DROP_M"),),
    ),
    Rule(
        "D14'",
        "First-pass drop: wrist below the desk **inside the table footprint**",
        "candidates(first pass)",
        "dexengine/data/candidate_rules.py:WRIST_BELOW_M",
        scope_from="profile:desk_outline=True",
        scenes=("table",),
        thresholds=(("depth", "dexengine/data/candidate_rules.py:WRIST_BELOW_M"),),
    ),
    Rule(
        "D14-fb",
        "First-pass drop: wrist below the desk in trees whose table footprint **cannot be computed** (D14' fallback)",
        "candidates(first pass)",
        "dexengine/data/candidate_rules.py:WRIST_BELOW_M",
        scope_from="dexengine/core/datasets.py:D14_FB_SCOPE",
        scenes=("table",),
        thresholds=(("depth", "dexengine/data/candidate_rules.py:WRIST_BELOW_M"),),
    ),
    Rule(
        "D23-key",
        "Each dataset declares the **object key source** of the training set (tree = D23 main object · "
        "clip = last field of the clip id)",
        "training set selection",
        "dexengine/placement/train_set/object_shape.py:target_object",
        scope_from="dexengine/core/datasets.py:DATASET_PROFILE",
    ),
    Rule(
        "D31",
        "First-pass drop: main-object pose jump (fast object + slow MANO wrist = source tracking defect)",
        "candidates(first pass)",
        "dexengine/data/candidates.py:_d31_fast_object_slow_wrist",
        thresholds=(
            ("main-object speed in a fast stretch (m/s)", "dexengine/data/candidate_rules.py:OBJ_SPEED_MAX_MPS"),
            ("slow-wrist threshold (m/s)", "dexengine/data/candidate_rules.py:WRIST_SLOW_MPS"),
        ),
    ),
    Rule(
        "D24",
        "First-pass drop: object–desk penetration (lowest point of the collision hull)",
        "candidates(first pass)",
        "dexengine/data/candidate_rules.py:OBJ_BELOW_M",
        scenes=("table",),
        thresholds=(("depth", "dexengine/data/candidate_rules.py:OBJ_BELOW_M"),),
    ),
    Rule(
        "D24′",
        "First-pass drop: a main object not grasped at the start is embedded in the desk (mesh)",
        "candidates(first pass)",
        "dexengine/placement/coords/contacts.py:start_embedded",
        scenes=("table",),
        thresholds=(
            ("embed depth", "dexengine/data/support/bands.py:START_EMBED_M"),
            ("hold test window (frames)", "dexengine/placement/coords/thresholds.py:START_HOLD_FRAMES"),
            ("contact fraction", "dexengine/placement/coords/thresholds.py:HELD_CONTACT_FRAC"),
        ),
    ),
    Rule(
        "D16",
        "Idle hand = rest pose (only the active hand manipulates)",
        "judge·bake",
        "dexengine/placement/collision/idle_arm.py:active_idle_for_tree",
        thresholds=(("minimum active time", "dexengine/data/cases/hand_object_assign.py:TH_ACTIVE_MIN_S"),),
    ),
    Rule(
        "D19",
        "Inject the bimanual layout into a one-hand emb (idle-hand block)",
        "bake entry",
        "dexengine/placement/bake/combo.py:_inject_missing_hand",
        datasets=("dexcanvas", "hrdexdb", "hocap", "hot3d"),
    ),
    Rule(
        "D5",
        "Per-dataset frame (yaw) alignment",
        "base",
        "dexengine/data/base_layer/hand_dir.py:hand_dir_stats",
        thresholds=(
            ("reference direction (°)", "dexengine/data/align.py:REF_DEG"),
            ("allowed difference from a re-measured value (°)", "dexengine/data/align.py:TOL_DEG"),
        ),
    ),
    Rule(
        "D5-b",
        "Egocentric data is yaw-aligned **per clip**",
        "base",
        "dexengine/data/base_layer/hand_dir.py:PER_CLIP_YAW",
        datasets=("hot3d",),
    ),
    Rule(
        "D10-h",
        "HOT3D desk recovery (table_recovered)",
        "base",
        "dexengine/data/support/plane_audit.py:audit_clip",
        datasets=("hot3d",),
    ),
    Rule(
        "D10-hrd",
        "HRDexDB grasp-start desk = frame-0 floor (table z 0) · the object-level h0 is the oracle",
        "base",
        "dexengine/data/support/hrdexdb.py:decide",
        datasets=("hrdexdb",),
        thresholds=(
            ("resting-clip LOO max (m)", "dexengine/data/support/hrdexdb.py:LOO_MAX_M"),
            ("|h0| max (m)", "dexengine/data/support/hrdexdb.py:H0_MAX_M"),
            ("hand ≥ desk − tolerance (m)", "dexengine/data/support/hrdexdb.py:HAND_TOL"),
            ("base start gap |g0| max (m)", "dexengine/data/support/hrdexdb_verify.py:G0_MAX_M"),
        ),
    ),
    Rule(
        "D1-h",
        "Supplementary parts are visual copies only (no collision) · parts already in the delivered set are not loaded",
        "bake",
        "dexengine/data/base_layer/hand_dir.py:supplements_for",
        datasets=("oakink2",),
    ),
    Rule(
        "D1-i",
        "Articulated-object RL init = joint values at the clip start",
        "env",
        "dexengine/rl/tasks/scene/spawn_hold.py:pin_joints",
        datasets=("arctic", "oakink2"),
    ),
    Rule(
        "D32",
        "Mid-air grasp start aid = spawn hold after reset (DEXENGINE_OBJ_SPAWN_HOLD, currently 0.2 s)",
        "env",
        "dexengine/rl/tasks/env.py:ArmResidualTrainEnv",
    ),
    Rule(
        "D44",
        "Floating static render objects: lowered to the desk surface if no hand approaches · removed from the "
        "render if one does",
        "visualization",
        "dexengine/placement/scene/static_desk.py:float_rule",
        thresholds=(
            ("float threshold (m)", "dexengine/placement/scene/static_desk.py:FLOAT_MAX_M"),
            ("hand approach threshold (m)", "dexengine/placement/scene/static_desk.py:HAND_NEAR_M"),
        ),
    ),
    Rule(
        "D20",
        "Scenes without a desk = support on a static disk",
        "judge·bake·visualization·env",
        "dexengine/placement/coords/stamp.py:aerial_support",
        datasets=("arctic",),
        scenes=("aerial",),
        thresholds=(
            ("match with the declared floor (m)", "dexengine/data/support/rest.py:Z_STAB"),
            ("minimum rest segment length (s)", "dexengine/data/support/rest.py:REST_MIN_S"),
            ("contact fraction", "dexengine/placement/coords/thresholds.py:HELD_CONTACT_FRAC"),
        ),
    ),
    Rule(
        "D22",
        "Scene desk = one canonical tabletop per workspace (tabletop 0.76×1.40 m)",
        "judge·bake·visualization",
        "dexengine/placement/coords/frames.py:desk_bounds_scene_for",
        thresholds=(
            ("tabletop top center x (m)", _DESK_OWNER + ":DESK_X_TASK"),
            ("tabletop top half-width — depth direction (m)", _DESK_OWNER + ":DESK_TOP_HALF_X"),
            ("tabletop top half-width — width direction (m)", _DESK_OWNER + ":DESK_TOP_HALF_Y"),
            ("edge margin (m)", "dexengine/placement/coords/thresholds.py:DESK_EDGE_MARGIN"),
        ),
    ),
    Rule(
        "D23",
        "Object roles (main/background · side · shared) = hand–object assignment (carrying)",
        "base",
        "dexengine/data/cases/hand_object_assign.py:assign",
        thresholds=(
            (
                "main object minimum carry time (continuous)",
                "dexengine/data/cases/hand_object_assign.py:TH_CARRY_MIN_S",
            ),
            ("carry relative speed ratio", "dexengine/data/cases/hand_object_assign.py:TH_CARRY_REL"),
            ("carry object speed floor", "dexengine/data/cases/hand_object_assign.py:TH_OBJ_STILL_MPS"),
            ("carry in-place rotation (deg/s)", "dexengine/data/cases/hand_object_assign.py:TH_ROT_DPS"),
            ("grasp minimum length (s)", "dexengine/data/cases/hand_object_assign.py:TH_GRASP_MIN_S"),
            ("grasp minimum fingertip count", "dexengine/data/cases/hand_object_assign.py:TH_GRASP_MIN_TIPS"),
            (
                "timeline grasp co-motion floor (m/s)",
                "dexengine/data/cases/hand_object_assign.py:TH_TIMELINE_COMOVE_MPS",
            ),
            ("small object longest side (m)", "dexengine/data/cases/hand_object_assign.py:TH_SMALL_OBJ_M"),
            ("used object minimum total grasp (s)", "dexengine/data/cases/hand_object_assign.py:TH_USED_MIN_S"),
            ("timeline minimum piece (s)", "dexengine/data/cases/hand_object_assign.py:TH_TIMELINE_MIN_S"),
            ("gap bridging (s)", "dexengine/data/cases/hand_object_assign.py:TH_GAP_CLOSE_S"),
            ("lag between joint lift and put-down (s)", "dexengine/data/cases/hand_object_assign.py:TH_SIM_STAGGER_S"),
            ("handover max overlap (s)", "dexengine/data/cases/hand_object_assign.py:TH_HANDOVER_MAX_OV_S"),
            ("speed smoothing window (s)", "dexengine/data/cases/hand_object_assign.py:TH_SMOOTH_S"),
            ("contact-free move (m)", "dexengine/data/objects/scene_requirement.py:MOVE_M"),
            ("static support tag move cap (m)", "dexengine/data/cases/hand_object_assign.py:TH_STATIC_SUPPORT_M"),
        ),
    ),
    Rule(
        "D35",
        "Contact source for cases · roles · active side = MANO",
        "cases·base",
        "dexengine/data/contact/source.py:CONTACT_SOURCE",
        thresholds=(("MANO fingertip threshold", "configs/contact/contact_source_distance.json:best_thr_m"),),
    ),
    Rule(
        "D36",
        "Time split = registry at the source (MANO) stage · all parents · hand-independent",
        "base",
        "dexengine/data/segment/split_steps.py:compute_units",
        thresholds=(("minimum piece length s", "dexengine/data/segment/split.py:--min-seconds"),),
    ),
    Rule(
        "D37",
        "Budget-exhausted unproven (UNPROVEN) is retried once = budget ×2",
        "judge",
        "dexengine/placement/judge/retry.py:classify",
        thresholds=(
            ("canonical budget", "configs/placement/judge.yaml:exhaust_max"),
            ("retry multiplier", "dexengine/core/datasets.py:EXHAUST_RETRY"),
        ),
    ),
    Rule(
        "D38",
        "One ruleset (RULESET_VERSION) — chunks with an older stamp are retired in the declared datasets "
        "(RULESET_RETIRE) · rulesets chained by a predicate (RULESET_STEPS) touch only the rows the change reaches",
        "fold",
        "dexengine/core/datasets.py:ruleset_retire_reason",
        scope_from="dexengine/core/datasets.py:RULESET_RETIRE",
    ),
    Rule(
        "D25",
        "arctic joint values = source sidecar (smoothing removed)",
        "base",
        "dexengine/data/artic/arctic_align.py:align",
        datasets=("arctic",),
    ),
    Rule(
        "D26",
        "Rest segment above the desk → disk (stands in for a holder)",
        "bake(stamp)·gate·env·visualization",
        "dexengine/placement/coords/stamp.py:compute_desk_support",
        scenes=("table",),
        thresholds=(
            ("height", "dexengine/data/support/bands.py:HIGH_REST_M"),
            ("hold test window (frames)", "dexengine/placement/coords/thresholds.py:START_HOLD_FRAMES"),
            ("contact fraction", "dexengine/placement/coords/thresholds.py:HELD_CONTACT_FRAC"),
            ("match with the declared floor (m)", "dexengine/data/support/rest.py:Z_STAB"),
        ),
    ),
    Rule(
        "D40",
        "Start support — an ungrasped main object starts on a support (the support stays to the end)",
        "bake(stamp)·gate·env·visualization",
        "dexengine/placement/coords/support_kind.py:start_support",
        thresholds=(
            ("hold test window (frames)", "dexengine/placement/coords/thresholds.py:START_HOLD_FRAMES"),
            ("contact fraction", "dexengine/placement/coords/thresholds.py:HELD_CONTACT_FRAC"),
            ("height that needs a support (desk scene)", "dexengine/data/support/bands.py:HIGH_REST_M"),
            ("floor change for the rest test (m)", "dexengine/data/support/rest.py:Z_STAB"),
        ),
    ),
    Rule(
        "D42",
        "Support objects — a static tracked object under an ungrasped main object goes into the scene as a "
        "fixed support (no disk where it supports)",
        "cases·bake(stamp)·gate·env",
        "dexengine/placement/coords/support_kind.py:support_kinds",
        thresholds=(
            ("floor band (m)", "dexengine/data/support/bands.py:SUPPORT_BAND_M"),
            ("horizontal reach distance (m)", "dexengine/data/support/bands.py:SUPPORT_REACH_M"),
            ("nesting limit (m)", "dexengine/data/support/bands.py:SUPPORT_NEST_MAX_M"),
            ("gap cap (m)", "dexengine/data/support/bands.py:SUPPORT_GAP_MAX_M"),
            ("support vertex count", "dexengine/data/support/bands.py:SUPPORT_MIN_VERTS"),
            ("support polygon margin (m)", "dexengine/data/support/bands.py:SUPPORT_POLY_PAD_M"),
            ("still displacement (m)", "dexengine/data/support/bands.py:SUPPORT_STILL_M"),
            ("still rotation (deg)", "dexengine/data/support/bands.py:SUPPORT_STILL_DEG"),
        ),
    ),
    Rule(
        "D45",
        "Pedestals — a pillar rises from the desk top under objects resting above the desk surface (main objects · "
        "support objects · static extras · where moved extras are put down · where the clip ends with the main "
        "object released), and from the floor under a fixed object standing outside the desk box (visible · collides "
        "with arm, hand and objects · measured by the judge's arm guard) · static extras below the desk surface that "
        "no hand touches are not spawned · static extras that no hand touches and that receive nothing are not "
        "placed in the scene (visual only — extra_slots VISUAL_ONLY)",
        "bake(stamp)·judge·gate·env",
        "dexengine/placement/coords/pedestals.py:static_pedestals",
        scope_from="profile:raised_rest=True",
        scenes=("table",),
        thresholds=(
            ("support band (m)", "dexengine/data/support/bands.py:HIGH_REST_M"),
            ("overhang fraction off the desk", "dexengine/placement/scene/static_desk.py:OVERHANG_MAX_FRAC"),
        ),
    ),
    Rule(
        "D46",
        "Collision partners of extra-object slots — a receiving static object collides with the main objects, "
        "not with the robot",
        "env",
        "dexengine/rl/tasks/scene/collision_presets.py:EXTRA_KIND_COLLIDE",
    ),
    Rule(
        "D48",
        "Placement rule — the outlines of objects the hand handles (first place · put-down place) and of static "
        "objects placed in the training scene do not overlap the robot's standing area and do not stick out of the "
        "desk beyond the threshold · if the only violator is a grazed static object, that object is removed from the "
        "spawn",
        "placement filter·env",
        "dexengine/placement/scene/place_rule.py:verdict",
        scenes=("table",),
        thresholds=(
            ("area fraction off the desk", "dexengine/placement/scene/static_desk.py:OVERHANG_MAX_FRAC"),
            ("grazed object threshold (s)", "dexengine/placement/scene/place_rule.py:GRAZE_MAX_S"),
            ("put-down place low (m)", "dexengine/placement/scene/place_rule.py:PUT_LOW_M"),
            ("put-down place still (m)", "dexengine/placement/scene/place_rule.py:PUT_STILL_M"),
            ("put-down place still time (s)", "dexengine/placement/scene/place_rule.py:PUT_STILL_S"),
            ("put-down place high (m)", "dexengine/placement/scene/place_rule.py:PUT_HIGH_M"),
            ("area sample count", "dexengine/placement/scene/static_desk.py:AREA_POINTS_N"),
        ),
    ),
    Rule(
        "D41",
        "Aerial tail cut — in an aerial scene (scene_of) the parent's last candidate window ends before the "
        "stretch where the main object stays unsupported to the end",
        "base",
        "dexengine/placement/coords/stamp.py:aerial_tail_start",
        scope_from="profile:workspace=room",
        scenes=("aerial",),
    ),
    Rule(
        "C640",
        "First-pass drop: aerial support gap (head · middle · whole)",
        "candidates(first pass)",
        "dexengine/data/support/gap.py:exclude_kind",
        scenes=("aerial",),
        thresholds=(("rest height tolerance (m)", "dexengine/data/support/rest.py:Z_STAB"),),
    ),
    Rule(
        "D27",
        "Articulation drive = the joint turns while touched → grasp",
        "cases",
        "dexengine/data/cases/hand_object_assign.py:TH_ARTIC_DRIVE_RADPS",
        thresholds=(("joint speed", "dexengine/data/cases/hand_object_assign.py:TH_ARTIC_DRIVE_RADPS"),),
    ),
    Rule(
        "T3",
        "An articulated object is trainable only with an articulated USD + env wiring",
        "trainable",
        "dexengine/core/datasets.py:ARTIC_ENV_WIRED",
        datasets=("arctic", "oakink2"),
    ),
    Rule(
        "D28",
        "Hand–object contact is measured on the MANO hand: fingertip keypoint ↔ object surface (with a skin pad) and "
        "the palm proxy point",
        "cases",
        "dexengine/data/contact/geom.py:contact_from_dist",
        thresholds=(
            ("contact", "dexengine/data/cases/classify.py:TH_CONTACT_M"),
            ("palm", "dexengine/data/cases/hand_object_assign.py:TH_PALM_M"),
        ),
    ),
    Rule(
        "D29",
        "Elbow swing cap (0.5 s window) — placements above it are rejected",
        "judge",
        "configs/placement/judge.yaml:swivel_max_deg",
        thresholds=(("swivel max over a 0.5 s window (deg)", "configs/placement/judge.yaml:swivel_max_deg"),),
    ),
    Rule(
        "dz",
        "Search dz layers (per dataset · scene)",
        "judge",
        "dexengine/placement/search_policy.py:dz_cells_for",
        scope_from="dexengine/placement/search_policy.py:DZ_CELLS",
    ),
    Rule(
        "center_scene",
        "Judge search origin (per-dataset target point · off when not in the table)",
        "judge",
        "dexengine/placement/search_policy.py:target_for",
        scope_from="dexengine/placement/search_policy.py:CENTER_SCENE",
    ),
    Rule(
        "D3",
        "Extra objects (extras) are left out of spawn and reward (main objects only)",
        "env",
        "dexengine/core/qpos_layout.py:Layout",
    ),
    Rule(
        "D4",
        "First-pass drop: clips with no/unknown support surface (desk datasets) · desk-less datasets take the D20 path",
        "candidates(first pass)",
        "dexengine/data/candidate_rules.py:DESK_OK",
    ),
    Rule(
        "D1-c",
        "Articulated object = base pose + joint values, all tracked and observed",
        "env",
        "dexengine/rl/tasks/mdp/rewards/task.py:RewardObjArtic",
        datasets=("arctic", "oakink2"),
    ),
]
