"""Command line of the rule registry: `python -m dexengine.data.rules --<query> ...` prints one answer.

Shell stages call it so that they never spell dataset scopes, hand ids, ledger names or exclusion lists themselves.
Exit codes: 0 = answered, 2 = a required companion flag is missing, 3 = unknown dataset or bad ledger key,
4 = bad f0 axis / ledger axis input.
"""

from __future__ import annotations

import argparse
import sys

from dexengine.core.rules.identity import active_hand, axis_hand, combo, hand_dir
from dexengine.core.rules.ledger import (
    TEST_TAG_PREFIXES,
    ledger_axis_parts,
    ledger_axis_variant,
    ledger_file,
    ledger_key,
    retarget_excluded,
    test_tag_kind,
)
from dexengine.core.rules.outputs import units_replaced_file
from dexengine.core.rules.profiles import DATASET_PROFILE, datasets_scope, f0_axis, launch_flags
from dexengine.core.rules.ruleset import campaign_dir, retry_queue_dir
from dexengine.core.rules.source import _paths
from dexengine.core.rules.thresholds import hand_dependent_report


def _cli_f0(axis_ds: str, hand_ds: str, variant: str | None, hand: str) -> int:
    """`--f0-axis DS` prints `scene<TAB>node<TAB>tail` (no `--variant` = the profile's variant; '' = the canonical
    desk root). `--f0-hand-dir DS` prints that axis's f0 folder of one hand and needs both `--variant` and `--hand`.
    rc 4 on missing or unknown input; the two flags together are refused."""
    if axis_ds and hand_ds:
        print("[datasets] --f0-axis and --f0-hand-dir are exclusive", file=sys.stderr)
        return 4
    if hand_ds and (variant is None or not hand):
        print("[datasets] --f0-hand-dir needs --variant ('' for the canonical desk) and --hand", file=sys.stderr)
        return 4
    try:
        if hand_ds:
            P = _paths()
            print(f0_axis(hand_ds, variant, P.hand_spec_dir(P.hand_id(hand)))["hand"])
            return 0
        ax = f0_axis(axis_ds, DATASET_PROFILE[axis_ds].variant if variant is None else variant)
    except (KeyError, ValueError) as e:
        flag = f"--f0-hand-dir {hand_ds!r} hand={hand!r}" if hand_ds else f"--f0-axis {axis_ds!r}"
        print(f"[datasets] {flag} variant={variant!r}: {e}", file=sys.stderr)
        return 4
    print(f"{ax['scene']}\t{ax['node']}\t{ax['tail']}")
    return 0


#: The documented entry point (dexengine.data.rules); the implementation module passes its own name (core.datasets).
PROG = "python -m dexengine.data.rules"


def _parser(prog: str = PROG) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    q = ap.add_argument
    q(
        "--datasets-scope",
        default=None,
        metavar="a,b",
        help="check a dataset list and print it space-separated (empty = all); rc 3 on an unknown name",
    )
    q("--launch-flags", default="", metavar="DS", help="extra launch flags of a dataset")
    q("--f0-axis", default="", metavar="DS", help="f0 shadow axis: 'scene<TAB>node<TAB>tail'")
    q(
        "--f0-hand-dir",
        default="",
        metavar="DS",
        help="f0 shadow folder of one hand on that axis (needs --variant, --hand)",
    )
    q("--variant", default=None, help="scene variant suffix for --f0-axis / --f0-hand-dir")
    q("--hand", default="", help="hand id or spec dir for --f0-hand-dir")
    q("--active-hand", action="store_true", help="hand id of this process (DEXENGINE_HAND)")
    q("--axis-hand", action="store_true", help="hand id that owns the shared outputs (AXIS_HAND)")
    q("--hand-dir", action="store_true", help="retargeting spec dir of the active hand")
    q("--combo", default=None, nargs="?", const="", metavar="VARIANT", help="combo name <arm>_<hand>[<variant>]")
    q("--hand-dependent", action="store_true", help="rule thresholds bound to hand geometry")
    q("--ledger-key", default="", metavar="DS", help="final-ledger axis name of a dataset (needs --emb)")
    q("--emb", default="", help="embodiment for --ledger-key (right|left|bimanual)")
    q("--from-ledger-key", default="", metavar="AXIS", help="ledger axis -> '<dataset> <emb>'")
    q("--axis-launch", default="", metavar="AXIS", help="ledger axis -> 'axis<TAB>dataset<TAB>emb<TAB>variant'")
    q("--ledger-file", default="", metavar="AXIS", help="final-ledger file of that axis for the active hand")
    q("--units-replaced-file", action="store_true", help="unit replacement record of the active hand")
    q("--test-tag-prefix", default="", choices=list(TEST_TAG_PREFIXES), help="tag prefix of a test class")
    q("--test-tag-of", default=None, metavar="TAG", help="test class of a tag, or an empty line")
    q("--retarget-excluded", default="", metavar="DS", help="clips excluded from stage-1 input: 'clip<TAB>reason'")
    q("--retarget-excluded-clips", default="", metavar="DS", help="the same list, clip names only")
    q("--campaign-dir", action="store_true", help="campaign folder (CAMPAIGN_DIR)")
    q("--retry-queue-dir", action="store_true", help="retry dispatch queue folder of the campaign")
    return ap


def _axis(axis: str, fmt) -> int:
    try:
        ds, emb, _v = ledger_axis_parts(axis)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 4
    print(fmt(ds, emb))
    return 0


def _print_datasets_scope(a):
    """--datasets-scope: the datasets of a scope name (rc 3 when unknown)."""
    try:
        print(" ".join(datasets_scope(a.datasets_scope)))
    except KeyError as e:
        print(str(e).strip("'\""), file=sys.stderr)
        return 3
    return 0


def _print_ledger_key(a):
    """--ledger-key: the ledger key of (dataset, --emb)."""
    if not a.emb:
        print("[datasets --ledger-key] --emb is required", file=sys.stderr)
        return 2
    try:
        print(ledger_key(a.ledger_key, a.emb))
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 3
    return 0


def _print_simple_answer(a, ap):
    """The one-value answers (hand, dirs, ledger file, combo, launch flags); no flag prints the help (rc 2)."""
    simple = {
        "active_hand": active_hand,
        "axis_hand": axis_hand,
        "hand_dir": hand_dir,
        "hand_dependent": lambda: "\n".join(hand_dependent_report()),
        "units_replaced_file": units_replaced_file,
        "campaign_dir": campaign_dir,
        "retry_queue_dir": retry_queue_dir,
    }
    for flag, fn in simple.items():
        if getattr(a, flag):
            print(fn())
            return 0
    if a.ledger_file:
        print(ledger_file(a.ledger_file, active_hand()))
        return 0
    if a.combo is not None:
        print(combo(variant=a.combo))
        return 0
    if a.launch_flags:
        print(" ".join(launch_flags(a.launch_flags)))
        return 0
    ap.print_help()
    return 2


def main(argv=None, prog: str = PROG) -> int:
    ap = _parser(prog)
    a = ap.parse_args(argv)
    if a.f0_axis or a.f0_hand_dir:
        return _cli_f0(a.f0_axis, a.f0_hand_dir, a.variant, a.hand)
    if a.axis_launch:
        return _axis(
            a.axis_launch, lambda ds, emb: f"{a.axis_launch}\t{ds}\t{emb}\t{ledger_axis_variant(a.axis_launch)}"
        )
    if a.from_ledger_key:
        return _axis(a.from_ledger_key, lambda ds, emb: f"{ds} {emb}")
    if a.datasets_scope is not None:
        return _print_datasets_scope(a)
    if a.ledger_key:
        return _print_ledger_key(a)
    if a.test_tag_prefix or a.test_tag_of is not None:
        print(TEST_TAG_PREFIXES[a.test_tag_prefix] if a.test_tag_prefix else test_tag_kind(a.test_tag_of))
        return 0
    if a.retarget_excluded or a.retarget_excluded_clips:
        ex = retarget_excluded(a.retarget_excluded or a.retarget_excluded_clips)
        for c in sorted(ex):
            print(f"{c}\t{ex[c]}" if a.retarget_excluded else c)
        return 0
    return _print_simple_answer(a, ap)
