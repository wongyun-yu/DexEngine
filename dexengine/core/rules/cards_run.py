"""Rule list, third part in declaration order — scene, queue and eval rules (S-*, Q-*, E-*; assembled in
dexengine.core.rules.cards)."""

from __future__ import annotations

from dexengine.core.rules.spec import Rule

RULES_RUN = [
    Rule(
        "S-spawn",
        "Objects placed in the training scene — only main objects · support objects · extras the hand "
        "moved · receiving objects · static objects a hand touched",
        "env",
        "dexengine/data/objects/extra_slots.py:slot_rows",
        thresholds=(
            ("cap on extra slots per clip", "dexengine/data/objects/extra_slots.py:K_MAX"),
            ("moved: translation (m)", "dexengine/data/objects/scene_requirement.py:MOVE_M"),
            ("moved: rotation (°)", "dexengine/data/objects/scene_requirement.py:ROT_DEG"),
            ("receiving distance (m)", "dexengine/data/objects/scene_requirement.py:RECV_M"),
            ("tracking glitch (m)", "dexengine/data/objects/extra_slots.py:JUMP_M"),
            ("micro-rotation cap (°)", "dexengine/data/objects/extra_slots.py:MICRO_ROT_DEG"),
            ("rule version", "dexengine/data/objects/extra_slots.py:RULE"),
            ("switch default (1 = on)", "dexengine/data/objects/extra_slots.py:DEFAULT"),
        ),
    ),
    # ── Scene rules (scope `default:scene_rules` = every dataset except those whose Profile.scene_rules_off names the
    #    id with a reason; empty = none) ──
    Rule(
        "S-carried",
        "Carried extras follow their carrier, not the world-recorded trajectory — carry relation "
        "(carrier · interval · relative pose) from the trajectory · every carried extra is attached (kinematic) to "
        "the simulated carrier at the stated relative pose for the whole clip",
        "env·RL eval",
        "dexengine/data/objects/extra_slots_geom.py:carry_of",
        scope_from="default:scene_rules",
        thresholds=(
            ("constant relative pose: translation (m)", "dexengine/data/objects/extra_slots_geom.py:CARRY_REL_M"),
            ("constant relative pose: rotation (°)", "dexengine/data/objects/extra_slots_geom.py:CARRY_REL_DEG"),
            ("carrier contact: receiving distance (m)", "dexengine/data/objects/scene_requirement.py:RECV_M"),
            ("carrier contact: production slack (m)", "dexengine/data/objects/scene_requirement.py:SLACK_RECV_M"),
            ("attached collision partners", "dexengine/rl/tasks/scene/collision_presets.py:EXTRA_ATTACHED_COLLIDE"),
            ("rule versions", "dexengine/data/objects/scene_rules.py:VERSIONS"),
        ),
    ),
    Rule(
        "S-touch-gate",
        "A still extra a hand touched collides with the main objects; a recorded overlap deeper than the true-shape "
        "value (the shapes physics collides, per main object) drops the clip",
        "env",
        "dexengine/data/objects/touch_gate.py:gate_row",
        scope_from="default:scene_rules",
        thresholds=(
            ("true-shape overlap (m)", "dexengine/data/objects/scene_requirement.py:TRUE_SHAPE_OVERLAP_M"),
            ("collision partners", "dexengine/rl/tasks/scene/collision_presets.py:EXTRA_TOUCHED_SOLID_COLLIDE"),
            ("rule versions", "dexengine/data/objects/scene_rules.py:VERSIONS"),
        ),
    ),
    Rule(
        "S-envelope",
        "A still extra next to where the objects move stands in the scene and collides with the objects (not the "
        "robot) — within the envelope margin of the recorded sweep of the main objects and moving extras; when the "
        "slots run out, the farthest envelope objects stay visual only",
        "env",
        "dexengine/data/objects/envelope.py:envelope_pass",
        scope_from="default:scene_rules",
        thresholds=(
            ("envelope margin (m)", "dexengine/data/objects/scene_requirement.py:ENVELOPE_M"),
            ("true-shape overlap (m)", "dexengine/data/objects/scene_requirement.py:TRUE_SHAPE_OVERLAP_M"),
            ("cap on extra slots per clip", "dexengine/data/objects/extra_slots.py:K_MAX"),
            ("collision partners", "dexengine/rl/tasks/scene/collision_presets.py:EXTRA_ENVELOPE_COLLIDE"),
            ("rule versions", "dexengine/data/objects/scene_rules.py:VERSIONS"),
        ),
    ),
    Rule(
        "S-true-shape",
        "A pinned still body (receiver · touched · touched_solid · envelope · support) stands as its visual mesh "
        "(triangles), not its convex parts; moving slots and dynamic objects keep the convex parts. Every reader "
        "measures the shape physics uses",
        "env",
        "dexengine/data/objects/collider_shape.py:shape_of",
        scope_from="default:scene_rules",
        thresholds=(
            ("triangle kinds", "dexengine/data/objects/collider_shape.py:TRIANGLE_KINDS"),
            ("triangle sample spacing (m)", "dexengine/data/objects/collider_shape.py:SAMPLE_M"),
        ),
    ),
    Rule(
        "S-robot-static",
        "The robot (hand and arm) collides with the still extras (touched · receiver · touched_solid · envelope) in "
        "training; self-collision, the desk, supports and discs keep their own rows",
        "env",
        "dexengine/rl/tasks/scene/collision_presets.py:extra_collide",
        scope_from="default:scene_rules",
        thresholds=(("still kinds", "dexengine/rl/tasks/scene/collision_presets.py:ROBOT_STATIC_KINDS"),),
    ),
    Rule(
        "S-hand-clear",
        "A reference hand inside a still extra the robot collides with is measured only: the side keeps its "
        "reference as recorded (reason hand_clear_kept_as_recorded, the max depth per side) — per-pair ruler: "
        "reference hand hulls × the extra's shape physics uses",
        "trainable",
        "dexengine/data/objects/hand_clear.py:clip_verdicts",
        scope_from="default:scene_rules",
        thresholds=(
            ("true-shape overlap (m)", "dexengine/data/objects/scene_requirement.py:TRUE_SHAPE_OVERLAP_M"),
            ("target kinds (robot channel ON)", "dexengine/data/objects/hand_clear.py:target_kinds"),
        ),
    ),
    Rule(
        "S-float-settle",
        "A free main object that rests above the desk or its disc is set onto it for the rest window (before the "
        "first touch); a reference whose object would jump more than the cap at the first touch is dropped (reason "
        "float_settle_jump) — supports: desk, own discs and every collision-on body under the object's lowest patch",
        "trainable·env",
        "dexengine/data/objects/float_settle.py:statement",
        scope_from="default:scene_rules",
        thresholds=(
            ("float above the support (m)", "dexengine/data/objects/scene_requirement.py:FLOAT_REST_M"),
            ("settle cap (m)", "dexengine/data/objects/scene_requirement.py:FLOAT_SETTLE_MAX_M"),
        ),
    ),
    Rule(
        "S-dyn-pairs",
        "A main object deeper than the free-pair tolerance into a body it collides with at frame 0 that keeps its "
        "pose to it over the whole clip is one body with it: a still rides the main object (the carry path), and a "
        "pair with no exact path is dropped (reason together_no_path); two main objects that do not move together "
        "are separated at frame 0 by the F0-1d move (one rigid object, at most the separation cap, nothing new "
        "overlapping) and keep their collision; a pair that move does not free has the collision of that pair only "
        "off, and the trajectory is dropped when the recorded overlap on the visual meshes, or a delivered output's, "
        "is deeper than the true-shape value (reason dyn_pair_visual_overlap); a pair the cross repair turns off or "
        "deepens is read again on the final rows by the gate (J-gate-dynpairs-xr)",
        "trainable·env",
        "dexengine/data/objects/dyn_pairs.py:dyn_pairs",
        scope_from="default:scene_rules",
        thresholds=(
            ("free pair tolerance (m)", "dexengine/data/objects/hand_clear.py:CLEAR_TOL_M"),
            ("moves together, extra pair: translation (m)", "dexengine/data/objects/dyn_pairs.py:EXTRA_TOGETHER_POS_M"),
            ("moves together, extra pair: rotation (°)", "dexengine/data/objects/dyn_pairs.py:EXTRA_TOGETHER_ROT_DEG"),
            ("moves together, main pair: translation (m)", "dexengine/data/objects/dyn_pairs.py:MAIN_TOGETHER_POS_M"),
            ("moves together, main pair: rotation (°)", "dexengine/data/objects/dyn_pairs.py:MAIN_TOGETHER_ROT_DEG"),
            ("frame-0 separation cap (m)", "dexengine/retarget/f0/repair_sep.py:CAP_M"),
            ("true-shape overlap (m)", "dexengine/data/objects/scene_requirement.py:TRUE_SHAPE_OVERLAP_M"),
            ("pair-off collision", "dexengine/rl/tasks/scene/collision_presets.py:DYN_PAIR_COLLIDE"),
            ("rule versions", "dexengine/data/objects/scene_rules.py:VERSIONS"),
        ),
    ),
    Rule(
        "S-desk-lite",
        "A clip whose recorded hand goes under the desk top inside the scene desk box is gated (the desk is the "
        "dataset's fixed box, not the captured table); spawned objects standing off the box are flagged",
        "trainable",
        "dexengine/placement/scene/desk_lite.py:measure",
        scope_from="default:scene_rules",
        scenes=("table",),
        thresholds=(
            ("wrist depth below the desk top (m)", "dexengine/data/candidate_rules.py:WRIST_BELOW_M"),
            ("off-desk footprint fraction", "dexengine/placement/scene/static_desk.py:OVERHANG_MAX_FRAC"),
        ),
    ),
    Rule(
        "S-levels",
        "Observed supports carry what rests on them at their measured height — no rest or start disc "
        "under a main object standing on a static extra that receives it (discs only where no support is observed)",
        "bake(stamp)·judge·gate·env",
        "dexengine/placement/coords/observed_support.py:drop_carried",
        scope_from="default:scene_rules",
        scenes=("table",),
    ),
    Rule(
        "S-prox",
        "Rollout overlap gate of the render also measures the main objects and extra slot bodies against "
        "the visual-only extras by penetration (episode gate, no collider)",
        "visualization",
        "dexengine/record/scene/overlap.py:gate_rows",
        scope_from="default:scene_rules",
        thresholds=(("penetration trigger (m)", "dexengine/record/scene/overlap.py:PEN_TRIGGER_M"),),
    ),
    Rule(
        "Q-same-content",
        "Same clip = a clip whose content read by training is byte-identical — only a same clip "
        "reuses an earlier checkpoint as is",
        "bundling",
        "dexengine/data/tree_content.py:parts",
        thresholds=(("record fields not counted as content", "dexengine/data/tree_content.py:RECORD_PATHS"),),
    ),
    Rule(
        "Q-turnover",
        "Declaring the clips to retrain — clips whose content changed · clips unchanged in the same "
        "generation · new clips",
        "bundling",
        "dexengine/rl/slot_generation.py:TURN_KINDS",
        thresholds=(("turnover kinds", "dexengine/rl/slot_generation.py:TURN_KINDS"),),
    ),
    Rule(
        "Q-issue-hold",
        "Hold list — clips caught by an open issue are not put into bundles",
        "bundling",
        "site:run/rl_queue_label.py:issue_hold",
        thresholds=(
            ("hold label", "site:run/rl_queue_label.py:HOLD"),
            ("hold list folder", "site:run/rl_queue_label.py:ISSUE_DIR"),
        ),
    ),
    Rule(
        "Q-bundle",
        "Bundle cutting — separately per dataset × run kind · several clips of the same object in one bundle",
        "bundling",
        "dexengine/placement/train_set/bundle_pack.py:pack",
        thresholds=(
            ("bundle size (clips)", "dexengine/rl/manifest_rules.py:N_DEFAULT"),
            ("minimum clips of the same object", "dexengine/placement/train_set/bundle_pack.py:CO_MIN"),
            ("split unit for large objects (clips)", "dexengine/rl/manifest_rules.py:BLOCK_DEFAULT"),
            ("rare object pairing threshold (clips)", "dexengine/placement/train_set/bundle_pack.py:REKEY_MIN"),
            ("shape family size edges (m)", "dexengine/placement/train_set/bundle_pack.py:FAM_SIZE_EDGES"),
            ("thin/flat ratio", "dexengine/placement/train_set/bundle_pack.py:FAM_THIN"),
            ("rule version", "dexengine/placement/train_set/bundle_pack.py:PACK_RULE"),
        ),
    ),
    Rule(
        "Q-canon-use",
        "Checkpoint-use check — was the loaded checkpoint trained on that clip with that content?",
        "RL eval",
        "dexengine/rl/checkpoints.py:check_args",
    ),
    Rule(
        "E-judge-shape",
        "RL eval shape — a fresh process that loads the checkpoint · eval env count · minimum evals per clip",
        "RL eval",
        "dexengine/rl/judge/metrics.py:JUDGE_NE",
        thresholds=(
            ("eval env count", "dexengine/rl/judge/metrics.py:JUDGE_NE"),
            ("minimum evals per clip", "dexengine/rl/judge/metrics.py:MIN_EVALS_PER_CLIP"),
        ),
    ),
    Rule(
        "E-finals",
        "Eval value = mean of the two final evals — mid-run checkpoints are not candidates",
        "RL eval",
        "dexengine/rl/judge/metrics.py:final_iters",
        thresholds=(("final eval points (fraction of all iterations)", "dexengine/rl/judge/metrics.py:FINAL_FRACS"),),
    ),
    Rule(
        "E-pass10",
        "10 % pass clip — a clip completed in at least one of every ten evals",
        "RL eval",
        "dexengine/rl/judge/metrics.py:passes",
    ),
    Rule(
        "E-park",
        "Check of the parked stretch between eval batches — only the two checks that measure a real drop "
        "invalidate an eval",
        "RL eval",
        "dexengine/rl/judge/park.py:park_verdict",
        thresholds=(
            ("lift limit (mm)", "dexengine/rl/judge/park.py:DZ_LIMIT_MM"),
            ("allowed gap above the support (mm)", "dexengine/rl/judge/park.py:REST_GAP_MM"),
            ("cap on the excluded fraction of parked residuals", "dexengine/rl/judge/park.py:EXCL_A_MAX_FRAC"),
            ("tolerance quantile", "dexengine/rl/judge/park.py:PARK_Q"),
            ("sample floor multiplier", "dexengine/rl/judge/park.py:FLOOR_MULT"),
        ),
    ),
    Rule(
        "E-contact-term",
        "RL eval runs without contact termination",
        "RL eval",
        "dexengine/rl/judge/terms.py:CONTACT_TERM",
        thresholds=(("contact termination (0 = off)", "dexengine/rl/judge/terms.py:CONTACT_TERM"),),
    ),
]
