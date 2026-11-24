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


def report_usd_missing(tally: RunTally) -> int:
    """Print every main object without a USD file once (dataset, asset, trees, USD path, bake and dump commands) on
    stderr; 4 when there is one, else 0."""
    if not tally.usd_missing:
        return 0
    from dexengine.data.objects import hull as OH  # noqa: PLC0415 — owner of the bake commands

    objs = collections.defaultdict(list)
    for m in tally.usd_missing:
        objs[(m["dataset"], m["asset"], m["articulated"], m["usd"])].append(f"{m['emb']}/{m['clip']}")
    trees_of = {(m["dataset"], m["emb"], m["clip"]) for m in tally.usd_missing}
    print(
        f"[D24] ✗ main objects without a USD: {len(objs)} (trees {len(trees_of)}) — not measured, not "
        "written. Bake each object USD before step 4:",
        file=sys.stderr,
    )
    for (ds, asset, art, usd), trees in sorted(objs.items()):
        bake, dump = OH.bake_commands(ds, asset, art)
        print(f"  {ds} {asset} usd={usd} trees={len(trees)} e.g. {trees[0]}", file=sys.stderr)
        print(f"    bake: {bake}\n    dump: {dump}", file=sys.stderr)
    return 4


def exit_code(rows: list[dict], tally: RunTally) -> int:
    """2 = stale cases, 3 = trees without a case or no roles built at all, 4 = main objects without a USD, else 0.
    Missing USDs are printed whatever the code."""
    usd_rc = report_usd_missing(tally)
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
    cand, done = sum(r["candidates"] for r in rows), sum(r["trees"] + r["usd_missing"] for r in rows)
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
    return usd_rc
