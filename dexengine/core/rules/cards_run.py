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
