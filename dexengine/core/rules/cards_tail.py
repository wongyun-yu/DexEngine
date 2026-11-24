"""Rule list, second part in declaration order — data, placement and bake rules (assembled in
dexengine.core.rules.cards)."""

from __future__ import annotations

from dexengine.core.rules.spec import Rule

RULES_TAIL = [
    Rule(
        "D6-a",
        "Shared object = one prim · the same pose in the left/right observation blocks",
        "env",
        "dexengine/rl/env_setup/build.py:_apply_shared_object",
    ),
    Rule(
        "D1-d",
        "oakink2 part identity = the official part tree (only instances of the same root form one part group)",
        "base",
        "dexengine/data/oakink2/parts.py:same_instance",
        datasets=("oakink2",),
    ),
    Rule(
        "D1-e",
        "oakink2 part-group completeness = add the sibling parts that have a pose and a mesh in the source",
        "base",
        "dexengine/data/oakink2/parts.py:siblings",
        datasets=("oakink2",),
    ),
    Rule(
        "D10-a",
        "Desk surface = source frame z 0 (official · by construction, hocap D10-a · dexcanvas D10-d)",
        "base",
        "dexengine/data/support/plane.py:OFFICIAL_ZERO",
        scope_from="dexengine/data/support/plane.py:OFFICIAL_ZERO",
        thresholds=(
            ("warning: deviation from the rest estimate (m)", "dexengine/data/support/plane.py:OFFICIAL_ZERO_TOL"),
        ),
    ),
    Rule(
        "D47",
        "Desk height = resting height · lowered halfway when much lower · not lowered when too low",
        "base",
        "dexengine/data/support/desk_rule.py:desk_height",
        thresholds=(
            ("lowering starts (m)", "dexengine/data/support/desk_rule.py:DESK_MID_START_M"),
            ("no-lowering limit (m)", "dexengine/data/support/desk_rule.py:DESK_MID_STOP_M"),
            ("lowering fraction", "dexengine/data/support/desk_rule.py:DESK_MID_FRACTION"),
        ),
    ),
    Rule(
        "D10-e''",
        "h2o desk = the source depth plane (plane height after undoing the tilt)",
        "base",
        "dexengine/data/readers/h2o_frame.py:tilt_transform",
        datasets=("h2o",),
    ),
    Rule(
        "D10-g'",
        "arctic = no desk (desk_mode none)",
        "base",
        "dexengine/data/support/plane.py:NO_DESK",
        scope_from="dexengine/data/support/plane.py:NO_DESK",
    ),
    Rule(
        "D15",
        "HOT3D contact segmentation — clips are cut at fingertip↔object-surface contact windows",
        "source readers",
        "dexengine/data/readers/hot3d.py:load_clips",
        datasets=("hot3d",),
        thresholds=(
            ("contact distance (m)", "dexengine/data/readers/hot3d.py:CONTACT_M"),
            ("gap bridging (frames)", "dexengine/data/readers/hot3d.py:BRIDGE"),
            ("per-hand dominant object minimum fraction", "dexengine/data/readers/hot3d.py:MIN_FRAC"),
            ("padding before/after (frames)", "dexengine/data/readers/hot3d.py:PAD"),
            ("minimum clip length (s)", "dexengine/data/readers/hot3d.py:MIN_CLIP_S"),
        ),
    ),
    Rule(
        "D1-j",
        "Articulated USD joint limits = per-object clip maximum (lo 0)",
        "assets",
        "dexengine/data/artic/spec_collect.py:update",
        datasets=("arctic", "oakink2"),
    ),
    Rule(
        "D34",
        "Joint velocity limits = the arm vendor URDF as is (wrist roll included) · finger limits as applied in sim",
        "judge·bake·gate",
        "dexengine/placement/retime/retime.py:main",
    ),
    Rule(
        "D39",
        "The wrist D14′ measures = the MANO (human) wrist",
        "candidates(first pass)",
        "dexengine/data/mano_frame.py:mano_keypoints_base",
        scenes=("table",),
        thresholds=(("wrist measurement frame", "dexengine/data/support/grab.py:WRIST_FRAME"),),
    ),
    Rule(
        "T5",
        "Stale setup tree = not trainable — the stamped initial setup ≠ the value recomputed with the current rules",
        "trainable",
        "dexengine/placement/coords/setup.py:setup_stale",
        thresholds=(("stamp comparison tolerance (m)", "dexengine/placement/coords/thresholds.py:SETUP_STALE_TOL_M"),),
    ),
    Rule(
        "J-desk-pen",
        "Cap on the penetration depth of the arm collision surface into the desk · support objects",
        "judge·bake·gate·final guard",
        "dexengine/placement/collision/core.py:DESK_SURF_PEN_MM",
        thresholds=(
            ("penetration depth cap (mm)", "dexengine/placement/collision/core.py:DESK_SURF_PEN_MM"),
            ("support object collision pairs", "dexengine/rl/tasks/scene/collision_presets.py:SUPPORT_COLLIDE"),
            ("hand↔support handling", "dexengine/placement/collision/support_surface.py:HAND_SUPPORT_MODE"),
        ),
    ),
    Rule(
        "J-self-clr",
        "Cap on the overlap of the arm collision surface · hand (palm · finger) hull with the torso surface",
        "judge·bake·gate·final guard",
        "dexengine/placement/collision/torso.py:TORSO_OVERLAP_MAX_MM",
        thresholds=(
            ("torso surface overlap cap (mm)", "dexengine/placement/collision/torso.py:TORSO_OVERLAP_MAX_MM"),
            ("link pairs measured against the torso", "dexengine/placement/collision/torso.py:TORSO_PAIRS"),
            ("torso ruler", "dexengine/placement/collision/arm_surface.py:TORSO_RULER"),
        ),
    ),
    Rule(
        "J-cross-clr",
        "Cap on the overlap of the two arms' collision surfaces — every arm-link body pair (wrist included)",
        "judge·bake·gate·final guard",
        "dexengine/placement/collision/core.py:ARM_OVERLAP_MAX_MM",
        thresholds=(("arm–arm surface overlap cap (mm)", "dexengine/placement/collision/core.py:ARM_OVERLAP_MAX_MM"),),
    ),
    Rule(
        "J-track-err",
        "Wrist tracking error p99 cap — above it the placement is unreachable",
        "judge·bake",
        "dexengine/placement/coarse_path/dp.py:--max-err-mm",
        thresholds=(("wrist tracking error p99 cap (mm)", "dexengine/placement/coarse_path/dp.py:--max-err-mm"),),
    ),
    Rule(
        "J-retime",
        "Retiming — velocity-limit safety factor · cap on the total clip stretch",
        "judge(prediction)·bake",
        "dexengine/placement/retime/stretch.py:SAFETY",
        thresholds=(
            ("velocity safety factor", "dexengine/placement/retime/stretch.py:SAFETY"),
            ("total stretch factor cap", "dexengine/placement/retime/retime.py:--max-stretch"),
        ),
    ),
    Rule(
        "J-gate-speed",
        "Gate — cap on the joint speed / limit ratio of the finished motion (p99 · max)",
        "gate",
        "dexengine/placement/gate/core.py:GATE_SPEED_P99_MAX",
        thresholds=(
            ("speed/limit ratio p99 cap", "dexengine/placement/gate/core.py:GATE_SPEED_P99_MAX"),
            ("speed/limit ratio max cap", "dexengine/placement/gate/core.py:GATE_SPEED_MAX_MAX"),
        ),
    ),
    Rule(
        "J-gate-grasp",
        "Gate — cap on the fraction of grasp-and-carry frames with a large wrist deviation",
        "gate",
        "dexengine/placement/gate/core.py:GATE_GRASP_BAD_FRAC",
        thresholds=(
            ("deviating frame fraction cap", "dexengine/placement/gate/core.py:GATE_GRASP_BAD_FRAC"),
            ("wrist error that counts as deviating (mm)", "dexengine/placement/gate/core.py:GATE_GRASP_ERR_MM"),
            ("minimum frames to measure the fraction", "dexengine/placement/gate/core.py:GATE_GRASP_MIN_FRAMES"),
            (
                "speed at which the object counts as moving (m/s)",
                "dexengine/placement/gate/core.py:GATE_OBJ_MOVING_MPS",
            ),
        ),
    ),
    Rule(
        "J-gate-f0pen",
        "Gate — robot colliders buried in objects at the first frame (spawn) (arm↔every main object "
        "· hand↔its own main object)",
        "gate",
        "dexengine/placement/gate/f0_pen.py:frame0_penetration",
        thresholds=(
            ("arm↔object burial depth (⑤)", "dexengine/placement/coords/thresholds.py:F0_ARM_OBJ_PEN_M"),
            ("hand↔own object (⑥) handling", "dexengine/placement/coords/thresholds.py:F0_HAND_OWNOBJ_MODE"),
            (
                "systematic unmeasurable gate — minimum count",
                "dexengine/placement/ledger_states.py:ERROR_SYSTEMATIC_MIN",
            ),
            (
                "systematic unmeasurable gate — fraction of measured clips",
                "dexengine/placement/ledger_states.py:ERROR_SYSTEMATIC_FRAC",
            ),
        ),
    ),
    Rule(
        "J-gate-f0objobj",
        "Gate — collision pieces of two main objects buried in each other at the first frame (spawn)",
        "gate",
        "dexengine/placement/gate/f0_pen.py:frame0_penetration",
        thresholds=(("object–object first-frame depth", "dexengine/placement/coords/thresholds.py:F0_OBJ_OBJ_PEN_M"),),
    ),
    Rule(
        "J-coverage",
        "Floor on the fraction of frames with a pose candidate (volume filter · final coverage)",
        "judge·bake",
        "dexengine/placement/coarse_path/dp.py:--min-cov",
        thresholds=(
            ("minimum fraction of frames with a pose candidate", "dexengine/placement/coarse_path/dp.py:--min-cov"),
            ("volume candidate frame fraction", "configs/placement/judge.yaml:vol_frac"),
        ),
    ),
    Rule(
        "J-jump",
        "Cap on arm joint jumps between adjacent frames (filters pose flips)",
        "judge",
        "configs/placement/judge.yaml:jump_max",
        thresholds=(("adjacent-frame joint jump cap (rad)", "configs/placement/judge.yaml:jump_max"),),
    ),
    Rule(
        "J-desk-area",
        "Object first-rest place = on the desk (robot-side edge trimmed) · first frame only · spawned "
        "objects only (coords.desk_mask_slots — D3 main objects, scene props excluded)",
        "judge",
        "configs/placement/judge.yaml:desk_shrink_robot",
        thresholds=(
            ("robot-side edge trim width (m)", "configs/placement/judge.yaml:desk_shrink_robot"),
            ("first frame only (1=yes)", "configs/placement/judge.yaml:desk_obj_frame0"),
        ),
    ),
    Rule(
        "J-prefilter",
        "Pre-filter margin of placement candidates (arm · hand–desk) — looser than the judge",
        "judge",
        "dexengine/placement/reach_map/clearance.py:LOOSE_MARGIN_M",
        thresholds=(
            ("arm margin (m)", "dexengine/placement/reach_map/clearance.py:LOOSE_MARGIN_M"),
            ("hand margin (m)", "configs/placement/judge.yaml:hand_margin"),
        ),
    ),
    Rule(
        "J-search",
        "Placement search range — fore-aft · lateral cells · rotation range · cells · start points · "
        "number of candidates to collect",
        "judge",
        "dexengine/placement/judge/core.py:main",
        thresholds=(
            ("fore-aft radius (cells)", "configs/placement/judge.yaml:half_cells_x"),
            ("lateral radius (cells)", "configs/placement/judge.yaml:half_cells_y"),
            ("rotation cell count", "configs/placement/judge.yaml:n_yaw"),
            ("rotation lower bound (°)", "configs/placement/judge.yaml:yaw_lo"),
            ("rotation upper bound (°)", "configs/placement/judge.yaml:yaw_hi"),
            ("frame stride", "configs/placement/judge.yaml:stride"),
            ("start points (anchors)", "configs/placement/judge.yaml:anchors"),
            ("first-pass candidate count", "configs/placement/judge.yaml:n_candidates"),
            ("exhaustive candidate count", "configs/placement/judge.yaml:exhaust_cands"),
            ("cell exclusion radius (mm)", "configs/placement/judge.yaml:prune_radius_mm"),
        ),
    ),
    Rule(
        "J-budget",
        "Judge budget — cap on cells tried per clip · cap on search time",
        "judge",
        "configs/placement/judge.yaml:exhaust_max",
        thresholds=(
            ("cap on cells tried per clip", "configs/placement/judge.yaml:exhaust_max"),
            ("cap on search time per clip (s)", "configs/placement/judge.yaml:clip_budget_s"),
        ),
    ),
    Rule(
        "J-cross-repair",
        "Bake cross-penetration repair — repair depth · target clearance · wrist travel cap",
        "bake",
        "dexengine/placement/repair/cross_repair.py:--repair-mm",
        thresholds=(
            ("repaired penetration depth (mm)", "dexengine/placement/repair/cross_repair.py:--repair-mm"),
            ("repair target clearance (mm)", "dexengine/placement/repair/cross_repair.py:--clearance-mm"),
            ("wrist travel cap (mm)", "dexengine/placement/repair/cross_repair.py:--max-wrist-mm"),
        ),
    ),
    Rule(
        "WRIST_SLACK",
        "Reference with the wrist housing inside the tabletop — tilt the wrist about the contact "
        "fingertips + re-solve the fingers (stage 1, 2e)",
        "base",
        "dexengine/placement/repair/wrist_desk.py:Side",
        scenes=("table",),
        thresholds=(
            ("hand tilt total rotation cap (°)", "dexengine/placement/repair/wrist_desk.py:WRIST_TILT_MAX_DEG"),
            ("lift cap after tilting (mm)", "dexengine/placement/repair/wrist_desk.py:LIFT_MAX_MM"),
            ("housing ↔ tabletop clearance (mm)", "dexengine/placement/repair/wrist_desk.py:WRIST_DESK_CLEAR_MM"),
            ("contact fingertip residual cap (mm)", "dexengine/placement/repair/wrist_desk.py:TIP_RESID_MM"),
            ("hand–object penetration increase cap (mm)", "dexengine/placement/repair/wrist_desk.py:OBJ_PEN_INC_MM"),
            (
                "lift weight (°/mm, 0 = least rotation first)",
                "dexengine/placement/repair/wrist_desk.py:LIFT_WEIGHT_DEG_PER_MM",
            ),
            ("per-frame added wrist speed cap (m/s)", "dexengine/placement/repair/wrist_desk.py:WRIST_JUMP_MAX_MPS"),
            ("penetration depth to repair (mm)", "dexengine/placement/collision/core.py:DESK_SURF_PEN_MM"),
        ),
    ),
    Rule(
        "J-gate-fingerrange",
        "Gate — finger angles of the baked reference lie within the retargeting clip range (f0 "
        "start fill · cross repair included)",
        "gate",
        "dexengine/placement/gate/core.py:finger_range_reason",
        thresholds=(
            ("allowed overshoot of the range (°)", "dexengine/placement/gate/core.py:GATE_FINGER_RANGE_TOL_DEG"),
            (
                "systematic unmeasurable gate — minimum count",
                "dexengine/placement/ledger_states.py:ERROR_SYSTEMATIC_MIN",
            ),
            (
                "systematic unmeasurable gate — fraction of measured clips",
                "dexengine/placement/ledger_states.py:ERROR_SYSTEMATIC_FRAC",
            ),
        ),
    ),
    Rule(
        "J-f0-shallow",
        "Shallow first-frame penetration is not repaired (the first pose is repaired only when deep)",
        "bake(f0)",
        "dexengine/retarget/f0/repair_side.py:$DEXENGINE_F0_SKIP_DEPTH_MM",
        thresholds=(
            (
                "max depth that skips the repair (mm)",
                "dexengine/retarget/f0/repair_side.py:$DEXENGINE_F0_SKIP_DEPTH_MM",
            ),
        ),
    ),
    Rule(
        "J-hand-oracle",
        "New-hand registration gate — the fingertip position computation matches the reference computation",
        "hand registration·post-bake gate",
        "dexengine/placement/gate/palm_check.py:--tol-mm",
        thresholds=(
            ("fingertip position difference tolerance (mm)", "dexengine/placement/gate/palm_check.py:--tol-mm"),
            ("compared-frame wrist position error (mm)", "dexengine/placement/gate/palm_check.py:--trk-pos-mm"),
            ("compared-frame wrist rotation error (°)", "dexengine/placement/gate/palm_check.py:--trk-rot-deg"),
        ),
    ),
    Rule(
        "J-arm-flip-select",
        "Arm-flip selection rule — selection drops placement candidates that cannot avoid an arm flip",
        "judge",
        "dexengine/placement/arm_flip.py:JUDGE_GATE_DEFAULT",
    ),
    Rule(
        "J-arm-flip",
        "Arm-flip repair — joint stretches where the bake's arm solution turns a lot in a short time "
        "are replaced by a continuous solution",
        "bake",
        "dexengine/placement/arm_flip.py:fix",
    ),
    Rule(
        "J-root-branch",
        "Hand-root orientation of frames inserted by retiming — only frames that left the true "
        "value are re-joined to the nearer representation",
        "bake",
        "dexengine/placement/retime/warp.py:root_branch_apply",
    ),
]
