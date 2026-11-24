"""Report of one object-roles run (object_roles.py): where the JSON report goes, the console table, the exit code."""

from __future__ import annotations

import collections
import sys
from typing import TYPE_CHECKING

from dexengine.core.errors import input_missing
from dexengine.data import rules as RR

if TYPE_CHECKING:
    from dexengine.data.object_roles import RunTally


def report_path(dss: list[str], write: bool):
    """(report path, why). The canonical path is only for all-dataset runs: the file rewrites `rows` wholesale, so a
    run over one dataset at a time would leave only the last dataset. The check compares sets
    (not counts), so a list naming one dataset twice does not pass. The report is per hand (roles read that hand's
    base trees), under the hand root; the owner is checked before writing."""
    HID = RR.active_hand()
    if set(dss) == set(RR.ALL):
        out_p, why = RR.audit_out("object_roles_d23.json", HID), "canonical (all datasets, this hand)"
    else:
        out_p = (
            RR._paths().run_path("run_log", HID, tool="object_roles")
            / f"object_roles_d23__{'-'.join(sorted(dss))[:80]}.json"
        )
        RR._hand_root_mod().assert_writable(out_p, HID, by="object_roles")
        why = f"partial scope ({len(dss)}/{len(RR.ALL)}); canonical untouched"
    # A dry run does not overwrite the write run's report (mode suffix, paths.mode_record).
    out_p = RR._paths().mode_record(out_p, None if write else "dryrun")
    return out_p, why


def print_table(rows: list[dict], write: bool) -> None:
    print(
        f"{'dataset/emb':22}{'trees':>6}{'chg':>6}{'main→bg':>8}{'bg→main':>8}{'shared↔1':>9}{'idle':>7}{'no case':>8}"
    )
    for r in rows:
        print(
            f"{r['dataset'] + '/' + r['emb']:22}{r['trees']:>6}{r['changed']:>6}{r['main_to_extra']:>8}"
            f"{r['extra_to_main']:>8}{r['shared_flip']:>9}{r['idle_hand']:>7}{r['no_case']:>8}"
        )
    und = collections.Counter()
    for r in rows:
        und.update(r["support_undecided"])
    print(
        f"[SUPPORT_KIND] support objects {sum(r['support'] for r in rows)} · undecidable (disc path) {dict(und)} · "
        f"unmeasurable {sum(r['support_fail'] for r in rows)} (coords.support_of)",
        flush=True,
    )
    print("mode:", "WRITE (base task_info updated)" if write else "dry-run")
    if write:  # marker: proof that the write entry applied; trees with equal content kept their mtime
        print(
            f"[roles] --write rewrote {sum(r['written'] for r in rows)} · unchanged skipped "
            f"{sum(r['same'] for r in rows)} "
            f"(tree_io.write_json_if_changed)",
            flush=True,
        )


def exit_code(rows: list[dict], tally: RunTally) -> int:
    """2 = stale cases, 3 = trees without a case or no roles built at all, else 0."""
    if tally.stale:
        print(
            f"[D23] stale cases {len(tally.stale)} trees — no roles built. Rebuild the cases first (D35). e.g. "
            f"{tally.stale[0]}",
            file=sys.stderr,
        )
        return 2
    if tally.missing_case:  # trees step 3 did not run; skipping them with rc 0 would let stage 1 report '4 done'
        print(
            f"[D23] ✗ trees without a case {len(tally.missing_case)} — cases/taxonomy.py --write did not write them. "
            f"e.g. {tally.missing_case[0]}",
            file=sys.stderr,
        )
        return 3
    cand, done = sum(r["candidates"] for r in rows), sum(r["trees"] for r in rows)
    # Candidates present but zero roles built = nothing processed, which is not a success.
    if cand and not done:
        print(
            input_missing(
                f"[D23] ✗ of {cand} candidate trees, 0 got roles (no case or unclassifiable "
                f"{sum(r['no_case'] for r in rows)}) — check the errors in case_census: the case layer may "
                "have failed to measure them"
            ),
            file=sys.stderr,
        )
        return 3
    return 0
