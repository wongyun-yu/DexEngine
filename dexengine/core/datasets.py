"""Rule registry: per-dataset pipeline profiles, the identity of the running process (arm, hand, combo), per-hand
output files, final-ledger naming and the rule list with thresholds read from their owning files.

The code lives in `dexengine.core.rules`; this module re-exports every name so `dexengine.core.datasets.<name>`
keeps working, resolves `ACTIVE_HAND` and `ARM` when read, and is the command line
(`python -m dexengine.core.datasets --help`, or `python3 dexengine/core/datasets.py --help`).

Run by file path: shell callers run `python3 dexengine/core/datasets.py …` without an installed package, so the three
lines below put this checkout's root on `sys.path` when the module has no package (the one sys.path edit here).
"""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:  # run by file path (`python3 dexengine/core/<name>.py …`): import this checkout's package
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from dexengine.core.rules.cards import RULES  # noqa: F401
from dexengine.core.rules.cli import _axis, _cli_f0, _parser, main  # noqa: F401
from dexengine.core.rules.identity import (  # noqa: F401
    _ARM_SAID,
    _HAND_SAID,
    ARM_ENV,
    AXIS_HAND,
    HAND_ENV,
    _axis_hand_from_layout,
    active_arm,
    active_hand,
    axis_hand,
    combo,
    hand_dir,
)
from dexengine.core.rules.ledger import (  # noqa: F401
    CHUNK_IDENTITY_SOURCES,
    EXCLUDE_ALL_HANDS,
    LEDGER_PREFIX,
    RETARGET_EXCLUDED_REL,
    TEST_TAG_PREFIXES,
    chunk_combo,
    excluded_from_file,
    excluded_scope,
    ledger_axis,
    ledger_axis_parts,
    ledger_axis_variant,
    ledger_file,
    ledger_files,
    ledger_files_for_hand,
    ledger_key,
    retarget_excluded,
    retarget_excluded_by,
    retarget_excluded_path,
    retarget_excluded_record,
    test_tag_kind,
)
from dexengine.core.rules.outputs import (  # noqa: F401
    ARTIC_SPEC,
    AXIS_OUTPUTS,
    SHARED_OUTPUT_DIR,
    _hand_file,
    _hand_root_mod,
    _WriteCheck,
    audit_out,
    audit_path,
    axis_output,
    fold_owner_file,
    retired_ledger_file,
    scope_arg,
    scope_marker,
    split_purged_file,
    table_out,
    table_path,
    units_replaced_file,
    unproven_retry_file,
)
from dexengine.core.rules.profiles import (  # noqa: F401
    _DESK_OWNER,
    _SCOPE_SAID,
    _TREE_SFX,
    _VARIANT_RE,
    AERIAL_ENV_WIRED,
    ALL,
    ARTIC_ENV_WIRED,
    COMBO_VARIANTS,
    D14_FB_SCOPE,
    DATASET_PROFILE,
    DEAD_AXES,
    SCENES,
    Profile,
    _tree_sfx,
    _with_workspace,
    campaign_excluded,
    datasets_scope,
    f0_axis,
    launch_flags,
)
from dexengine.core.rules.rule import (  # noqa: F401
    PROFILE_SCOPE,
    READER_CALL_SCOPE,
    _blank_reasons,
    applies,
    covers,
    resolve,
    rule_datasets,
    rule_scenes,
    scope_of,
)
from dexengine.core.rules.ruleset import (  # noqa: F401
    EXHAUST_RETRY,
    RETRY_QUEUE_DIRNAME,
    RETRY_RUN_MARKER,
    RULESET_EQUIVALENT,
    RULESET_PREDICATES,
    RULESET_RETIRE,
    RULESET_STEPS,
    RULESET_VERSION,
    RULESET_WHY,
    _miss,
    _pred_judge_supports,
    campaign_dir,
    retry_queue_dir,
    ruleset_chain,
    ruleset_note,
    ruleset_retire_reason,
    ruleset_row_reason,
)
from dexengine.core.rules.source import (  # noqa: F401
    _SELF_FILES,
    REPO,
    SELF_PATH,
    _env_default_expr,
    _flag_default,
    _json_key,
    _num_or_str,
    _paths,
    _source_symbol,
    _sym,
    _var_default,
    ref_env,
)
from dexengine.core.rules.spec import Rule  # noqa: F401
from dexengine.core.rules.thresholds import (  # noqa: F401
    _THRESH_SAID,
    FRAME_MANO,
    FRAME_NONE,
    FRAME_RATIO,
    FRAME_ROBOT,
    HAND_THRESHOLDS,
    HandThreshold,
    _frame_case,
    _frame_mano_always,
    _frame_none,
    _frame_ratio,
    case_source,
    hand_dependent_report,
    hand_thresholds,
)


def __getattr__(name: str):
    """`ACTIVE_HAND` / `ARM` are resolved each time they are read (PEP 562)."""
    if name == "ACTIVE_HAND":
        return active_hand()
    if name == "ARM":
        return active_arm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    raise SystemExit(main(prog="python -m dexengine.core.datasets"))
