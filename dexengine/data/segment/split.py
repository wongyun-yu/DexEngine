"""D36 long-clip split: hand-independent cut rule on the source (MANO) stage, the same pieces for every hand.

A clip with two tasks (e.g. a handover of one object, then another object handled alone) gets one piece per task.
Cuts are where the hand-to-object assignment changes on the human contact timeline (``case.timeline``, MANO keypoints to
object surfaces, independent of the robot hand). Every parent is cut, whatever its verdict. The pieces are written
once to a hand-independent registry (``data/<ds>/INTAKE/units/<emb>.json``); every hand only applies it. Roles are
re-derived per piece by the role step. Pieces are judged as new candidates; a piece is never cut again (``--nested``).

    python3 dexengine/data/segment/split.py --define --datasets hocap,hot3d --emb bimanual   # write the registry
    DEXENGINE_HAND=<hand> python3 dexengine/data/segment/split.py --apply --datasets ... --emb ...
    python3 dexengine/data/segment/split.py --datasets ... --emb ...                       # dry-run table
    DEXENGINE_HAND=<hand> python3 dexengine/data/segment/split.py --apply --refresh-support [--dry-run] \
        --out-changed <tsv> --datasets ... --emb ...   # re-cut old pieces' rest intervals to the piece window

The rule (``rule``), the piece builder (``build``), the D41 tail cut (``tail``) and the shared names (``common``) are
submodules; this module is the CLI and re-exports their names.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import main_root as _data_root
from dexengine.data.segment.build import (  # noqa: E402,F401  (re-exported)
    _quarantine_one,
    _record_replaced,
    _slice_npz,
    apply_table,
    inherit_piece_exclusions,
    piece_support,
    piece_task_info,
    purge_units,
    refresh_piece_support,
    reslice_stale,
    split_clip_registry,
    sync_piece_exclusions,
    tree_arrays,
    unregistered_units,
)
from dexengine.data.segment.common import (  # noqa: E402,F401  (re-exported)
    COMBO,
    D41_RULE,
    HAND,
    REPO,
    RULE_NAME,
    TAIL_RULE_NAME,
    _base_dir,
    _base_emb,
    _own_registry_path,
    _registry_path,
    _stale_dir,
    _tc,
    split_root,
)
from dexengine.data.segment.rule import (  # noqa: E402,F401  (re-exported)
    CUT_STATS,
    _cut_inputs,
    _fill,
    _label,
    _moved,
    _timeline_segments,
    cut_rule,
    drop_contained,
    grasp_switch_cuts,
    segments_of,
    unit_key,
)
from dexengine.data.segment.split_steps import compute_units, define_step, tail_step, write_d36
from dexengine.data.segment.tail import (  # noqa: E402,F401  (re-exported)
    TAIL_STATS,
    _has_static_artic,
    _parents,
    d36_diff,
    d41_in_scope,
    d41_pending,
    effective_table,
    static_artic_own,
    strip_tail,
    tail_cut,
    tail_diff,
    tail_main,
    tail_overlay,
)

#: Marker line of the unregistered-piece purge and its count field: a stage-1 driver sums the number after
#: QUARANTINED over the log, so the field name is part of the contract.
PURGE_MARK = "[PURGE_UNREGISTERED]"
QUARANTINED = "quarantined="


def purge_line(ds: str, emb: str, hand: str, n_unregistered: int, n_quarantined: int, dry_run: bool) -> str:
    """The [PURGE_UNREGISTERED] line: pieces outside the registry and how many were moved to quarantine."""
    return (
        f"{PURGE_MARK} {ds}/{emb}({hand}): outside registry {n_unregistered} · {QUARANTINED}{n_quarantined}"
        f"{' (dry-run)' if dry_run else ''}"
    )


def _refresh_report(ds: str, a, rs: dict, note: str = "") -> None:
    """Marker of --refresh-support (proof that the refresh really ran: checked and changed counts) and the list of
    changed pieces (--out-changed)."""
    print(
        f"[REFRESH_SUPPORT] {ds}/{a.emb}({HAND}): pieces checked {rs['checked']} · re-sliced to window {rs['changed']}"
        f"{' (dry-run, not written)' if a.dry_run else ''} · no parent {rs['no_parent']} · "
        f"no piece tree {rs['no_piece']}{note}",
        flush=True,
    )
    if a.out_changed:
        with open(a.out_changed, "a") as fh:
            fh.writelines(f"{ds}\t{a.emb}\t{u}\n" for u in rs["changed_units"])


def _add_mode_args(ap: argparse.ArgumentParser) -> None:
    """Registry modes: define the hand-independent registry, verify a hand against it, apply it."""
    ap.add_argument(
        "--define",
        action="store_true",
        help="write the hand-independent registry (reference hand = timeline of DEXENGINE_HAND)",
    )
    ap.add_argument(
        "--verify-hand",
        action="store_true",
        help="recompute on this hand's (DEXENGINE_HAND) timeline and check it equals the registry (rc 2 when not)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="make this hand's (DEXENGINE_HAND) base pieces from the registry (existing ones are compared by "
        "window, rc 2 on a difference)",
    )


def _add_piece_args(ap: argparse.ArgumentParser) -> None:
    """Piece options: minimum length, nested splits, purge of existing or unregistered pieces."""
    ap.add_argument("--min-seconds", type=float, default=1.0, help="segments shorter than this are dropped")
    ap.add_argument(
        "--nested",
        action="store_true",
        help="split clips that are already _tN units again (nested). By default they are skipped and counted, "
        "since a unit's window was already cut once; allowed only when explicit",
    )
    ap.add_argument(
        "--purge-existing",
        action="store_true",
        help="with --apply: remove this hand's existing _tN units first (only when replacing the registry generation)",
    )
    ap.add_argument(
        "--purge-unregistered",
        action="store_true",
        help="with --apply or --refresh-support: move pieces (_tN) present in this hand's base but **missing** "
        "from the registry to the quarantine data/<ds>/base_stale/ (never deleted)"
        " (when the registry rule dropped a piece, e.g. contained-piece removal). Recorded in "
        "units_replaced__<hand>.tsv as new_range=removed. Without it, such pieces give rc 2",
    )


def _add_support_args(ap: argparse.ArgumentParser) -> None:
    """Rest-interval refresh and re-slicing of stale pieces (dry run, changed-piece TSV, quarantine label)."""
    ap.add_argument(
        "--refresh-support",
        action="store_true",
        help="re-slice the rest intervals (support.rest_intervals) of existing pieces from the parent to the "
        "piece window; "
        "alone it only refreshes (stage-1 axis, no new pieces), with --apply it refreshes after applying. "
        "Other piece fields are preserved. Writes the changed pieces to --out-changed (rejudge input)",
    )
    ap.add_argument(
        "--out-changed",
        default="",
        help="TSV of pieces --refresh-support changed (or would change when dry) (ds, emb, piece); appended",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="count only, without writing, for --refresh-support · --reslice-stale"
    )
    ap.add_argument(
        "--reslice-stale",
        action="store_true",
        help="re-slice registered pieces that are not window pieces of the current parent (called after stage 1 "
        "rebuilt the parent). "
        "Only this hand's trees and wrist npz are quarantined (under --label) · units_replaced record · "
        "marker [RESLICE_STALE] · alone only",
    )
    ap.add_argument("--label", default="", help="quarantine subfolder name for --reslice-stale ([A-Za-z0-9_.-])")


def _add_chain_args(ap: argparse.ArgumentParser) -> None:
    """Links of the canonical chain: tail cut (define, verify, step) and the split-rule step."""
    ap.add_argument(
        "--define-tail",
        action="store_true",
        help="D41: keep the registry's D36 statement and add the tail cut of aerial-path parents (reference hand only)",
    )
    ap.add_argument(
        "--verify-tail",
        action="store_true",
        help="D41: re-measure the overlay on this hand's base and compare with the registry (rc 2 on a difference)",
    )
    ap.add_argument(
        "--tail-step",
        action="store_true",
        help="D41 link of the canonical chain (run after --apply): reference hand: overlay → re-slice "
        "(--label) → new pieces · "
        "other hands: compare. Out of scope or no registry: skipped. Marker [D41_STEP]",
    )
    ap.add_argument(
        "--define-step",
        action="store_true",
        help="D36 link of the canonical chain (run before --apply): reference hand: when the current "
        "rule's table differs, "
        "redefine → re-slice (--label) · other hands: compare (rc 2 on a difference). Marker [D36_STEP]",
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", required=True, help="comma-separated list")
    ap.add_argument("--emb", required=True, help="embodiment (right|left|bimanual), no default")
    _add_mode_args(ap)
    _add_piece_args(ap)
    _add_support_args(ap)
    _add_chain_args(ap)
    return ap


def _check_flags(a) -> None:
    """Mode flags are exclusive; label and purge flags only where they mean something."""
    if sum(map(bool, (a.define, a.verify_hand, a.apply, a.define_tail, a.verify_tail, a.tail_step, a.define_step))) > 1:
        raise SystemExit(
            "[split] --define / --verify-hand / --apply / --define-tail / --verify-tail / --tail-step / "
            "--define-step: only one"
        )
    if (a.tail_step or a.define_step) and not re.fullmatch(r"[A-Za-z0-9_.-]+", a.label):
        raise SystemExit(
            "[split] --tail-step · --define-step need --label <[A-Za-z0-9_.-]>"
            " (quarantine subfolder for re-sliced pieces)"
        )
    if a.purge_existing and not a.apply:
        raise InputMissingError("[split] --purge-existing only with --apply (there is no delete-only run)")
    if a.purge_unregistered and not (a.apply or a.refresh_support):
        raise SystemExit("[split] --purge-unregistered only with --apply or --refresh-support")
    if a.reslice_stale and (
        a.apply or a.refresh_support or a.define or a.verify_hand or not re.fullmatch(r"[A-Za-z0-9_.-]+", a.label)
    ):
        raise SystemExit("[split] --reslice-stale only alone, with --label <[A-Za-z0-9_.-]>")


def _reslice_stale_step(ds: str, a, reg: Path) -> None:
    """--reslice-stale: re-slice registered pieces that are no window piece of the current parent."""
    rs = (
        reslice_stale(
            ds,
            a.emb,
            effective_table(ds, a.emb, json.loads(reg.read_text())["parents"]),
            write=not a.dry_run,
            label=a.label,
        )
        if reg.is_file()
        else {"checked": 0, "stale": 0, "resliced": 0, "why": {}, "units": []}
    )
    top = dict(sorted(rs["why"].items(), key=lambda x: -x[1])[:3])
    print(
        f"[RESLICE_STALE] {ds}/{a.emb}({HAND}): pieces checked {rs['checked']} · stale {rs['stale']} · "
        f"re-sliced {rs['resliced']}{' (dry-run)' if a.dry_run else ''} · reasons {top}",
        flush=True,
    )
    if a.out_changed:
        with open(a.out_changed, "a") as fh:
            fh.writelines(f"{ds}\t{a.emb}\t{x}\n" for x in rs["units"])


def _refresh_support_step(ds: str, a, reg: Path) -> bool:
    """--refresh-support alone: re-slice rest intervals of existing pieces. True = rc 2 (unregistered pieces)."""
    bad = False
    # Refresh only (the stage-1 axis calls it before judgement and f0); no new pieces (that is the
    # stage1.split_apply job: making them here would hide 'new pieces' from that step and skip downstream work).
    # A dataset without a registry = no pieces (0, not an error).
    if reg.is_file():
        table = effective_table(ds, a.emb, json.loads(reg.read_text())["parents"])
        rs = refresh_piece_support(ds, a.emb, table, write=not a.dry_run)
        # Pieces outside the registry (old pieces the rule dropped): quarantine them when asked (record
        # units_replaced ·
        # the fold retires the rows), otherwise list them and rc 2 (no silent leftovers). dry-run only counts.
        unreg, purged = unregistered_units(ds, a.emb, table, a.purge_unregistered and not a.dry_run)
        print(purge_line(ds, a.emb, HAND, len(unreg), purged, a.dry_run), flush=True)
        if unreg and not a.purge_unregistered and not a.dry_run:
            bad = True
    else:
        rs = {"checked": 0, "changed": 0, "no_parent": 0, "no_piece": 0, "changed_units": []}
    _refresh_report(ds, a, rs, "" if reg.is_file() else " · no registry (no pieces)")
    return bad


def _apply_step(ds: str, a, reg: Path, per: dict) -> bool:
    """--apply: make this hand's pieces from the registry. True = rc 2."""
    if not reg.is_file():
        print(f"[split] {ds}/{a.emb}: no registry {reg} — run --define first", file=sys.stderr)
        return True
    reg_obj = json.loads(reg.read_text())
    table = effective_table(ds, a.emb, reg_obj["parents"])
    # A registry without the overlay while the D41 scope has aerial-path parents: the reference hand passes (it
    # must make the D36 pieces before measuring; the same split_apply run's tail step overlays next) · other
    # hands
    # fail before making pieces with untrimmed tails.
    if reg_obj.get("reference_hand_dir") != HAND and d41_pending(ds, a.emb, reg_obj):
        print(
            f"[split] ✗ {ds}/{a.emb}: registry without the D41 overlay (reference hand "
            f"{reg_obj.get('reference_hand_dir')}) — "
            f"the reference hand's split_apply (tail step) runs first",
            file=sys.stderr,
        )
        return True
    if a.purge_existing:
        purged = purge_units(ds, a.emb)
        print(
            f"[split] {ds}/{a.emb}: existing _tN units {purged['trees']} removed (wrist npz "
            f"{purged['wrist']} · sidecars {purged['sidecar']}) → {purged['list']}"
        )
    ap = apply_table(ds, a.emb, table, a.purge_unregistered)
    bad = bool(ap["bad"])
    print(
        f"[split] {ds}/{a.emb} applied ({HAND}): parents {len(table)} · new pieces {ap['made']} · existing "
        f"match {ap['kept']} · parent missing {ap['missing']} · window mismatch {ap['bad']} · unregistered "
        f"pieces {ap['unreg']} (quarantined {ap['purged']})"
    )
    if a.refresh_support:
        _refresh_report(ds, a, refresh_piece_support(ds, a.emb, table, write=not a.dry_run))
    per[ds] = [len(table), ap["made"] + ap["kept"]]
    return bad


def _units_step(ds: str, a, reg: Path, per: dict) -> bool:
    """Measure the units on this hand's timeline; --define writes the registry, --verify-hand compares. True = rc 2."""
    flag = False
    table, nested_skipped, contained = compute_units(ds, a.emb, HAND, a.min_seconds, a.nested)
    n_units = sum(len(v) for v in table.values())
    n_cont = sum(len(v) for v in contained.values())
    if a.define:
        # The registry is a hand-independent layer written **only by the reference hand** (the defining hand). Other
        # hands only compare with --verify-hand. Another hand running --define would overwrite the reference hand's
        # registry, so it fails (to redefine, remove the registry explicitly). Equal content is not rewritten
        # (mtime unchanged, tree_io.write_json_if_changed).
        if reg.is_file():
            try:
                prev_ref = json.loads(reg.read_text()).get("reference_hand_dir")
            except ValueError:
                prev_ref = None
            if prev_ref and prev_ref != HAND:
                print(
                    f"[split] ✗ {ds}/{a.emb}: the registry's reference hand is {prev_ref} — hand {HAND} "
                    "does not overwrite it with --define"
                    f" (compare with --verify-hand). To redefine, remove {reg.relative_to(_data_root())} first",
                    file=sys.stderr,
                )
                return True
        wrote = write_d36(reg, table, contained, a.min_seconds, a.nested)
        print(
            f"[split] {ds}/{a.emb} registry → {reg.relative_to(_data_root())}("
            f"{'written' if wrote else 'same content — not rewritten'}): "
            f"parents {len(table)} · pieces {n_units} · contained dropped {n_cont}"
            + (f" · nested skipped {nested_skipped}" if nested_skipped else "")
        )
    elif a.verify_hand:
        if not reg.is_file():
            print(f"[split] {ds}/{a.emb}: no registry — run --define first", file=sys.stderr)
            return True
        r = d36_diff(ds, a.emb, effective_table(ds, a.emb, json.loads(reg.read_text())["parents"]), table)
        bad = r["diff"] + r["only_ref_present"] + r["only_this"]
        print(
            f"[split] {ds}/{a.emb} verify ({HAND} vs registry): shared parents {len(r['common'])} · "
            f"window mismatches {len(r['diff'])} · "
            f"only in registry (parent present here) {len(r['only_ref_present'])} · only in this hand "
            f"{len(r['only_this'])} · "
            f"parents absent here {r['absent']} → {'✗ differs' if bad else 'same'}"
        )
        for c in bad[:5]:
            print(
                f"    {c}: registry {[(u['t0'], u['t1']) for u in r['ref'].get(c, [])]} vs this hand "
                f"{[(u['t0'], u['t1']) for u in table.get(c, [])]}"
            )
        flag = bool(bad)
    else:
        if nested_skipped:
            print(
                f"[split] {ds}/{a.emb}: nested candidates (already _tN, split again) {nested_skipped} "
                "skipped — allowed only with --nested"
            )
    per[ds] = [len(table), n_units]
    return flag


def _print_totals(a, per: dict) -> None:
    print(f"{'dataset':10}{'parents':>8}{'segments':>9}")
    for ds, (nc, nm) in sorted(per.items()):
        print(f"{ds:10}{nc:>8}{nm:>9}")
    tail = "" if (a.define or a.apply) else "  (dry-run — --define writes the registry, --apply builds)"
    print(f"total parents {sum(v[0] for v in per.values())} → segments {sum(v[1] for v in per.values())}{tail}")


def main() -> int:
    a = _parser().parse_args()
    _check_flags(a)
    rc = 0
    per: dict[str, list[int]] = {}
    for ds in a.datasets.split(","):
        print(f"[SPLIT_ROOT] {ds}/{a.emb}({HAND}): {split_root(ds, a.emb)}", flush=True)
        reg = _registry_path(ds, a.emb)
        if a.define_tail or a.verify_tail:
            rc = max(rc, tail_main(ds, a.emb, reg, write=a.define_tail))
            continue
        if a.tail_step:
            rc = max(rc, tail_step(ds, a.emb, a.label))
            continue
        if a.define_step:
            rc = max(rc, define_step(ds, a.emb, a.label, a.min_seconds, a.nested))
            continue
        if a.reslice_stale:
            _reslice_stale_step(ds, a, reg)
            continue
        if a.refresh_support and not a.apply:
            if _refresh_support_step(ds, a, reg):
                rc = 2
            continue
        if a.apply:
            if _apply_step(ds, a, reg, per):
                rc = 2
            continue
        if _units_step(ds, a, reg, per):
            rc = 2
    _print_totals(a, per)
    return rc


if __name__ == "__main__":
    sys.exit(main())
