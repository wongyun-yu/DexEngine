"""Per-hand output files of the data stages (audit JSON, tables, artic specs) and scope markers."""

from __future__ import annotations

import os
import pathlib
import sys

from dexengine.core.errors import InputMissingError
from dexengine.core.rules.identity import hand_dir
from dexengine.core.rules.profiles import datasets_scope
from dexengine.core.rules.source import _paths


#: ── Single owner of the names and places of per-hand outputs (audit JSON, tables) ──
#: Each hand writes into its own folder (`dexengine.core.paths.run_path("exp_audit"|"exp_table", hand)`), so there is
#: nothing to overwrite and no gate restricting writes to the axis hand. Shared pages **read** the axis hand's folder.
#: Writers use `*_out` (write-path check), readers use `*_path` (no I/O).
class _WriteCheck:
    """Write-path check used by the `*_out` helpers. The per-hand root ownership guard of a shared disk is
    operations code outside this package, so the check only normalises the path."""

    @staticmethod
    def assert_writable(path, hand: str, *, combo: str = "", by: str = "", root=None) -> pathlib.Path:
        return pathlib.Path(os.path.abspath(str(path)))


def _hand_root_mod():
    """The write-path check (kept for callers; see `_WriteCheck`)."""
    return _WriteCheck


def _hand_file(layer: str, name: str, hand: str, root=None):
    if not hand:
        raise InputMissingError(f"[{layer}] empty hand — per-hand outputs name the hand explicitly (no default)")
    if not name or "/" in name:
        raise SystemExit(f"[{layer}] file name {name!r} — one path segment only (no subfolder hidden in the name)")
    return _paths().run_path(layer, hand, root=root) / name


def audit_path(name: str, hand: str, root=None, scope=None):
    """Audit output place of hand `hand` (read) — `docs/experiments/hands/<hand>/audit/<name>`. With `scope` (the
    datasets a narrowed run measures — a tuple or a comma list) it is the scoped place `…/audit/scope_<a+b>/<name>`:
    a narrowed run never overwrites the whole-hand record, and its readers (debt gate, --scope-report)
    pass the same scope to read the same place."""
    p = _hand_file("exp_audit", name, hand, root)
    ds = sorted({d.strip() for d in (scope.split(",") if isinstance(scope, str) else scope or ()) if d.strip()})
    return p.parent / ("scope_" + "+".join(ds)) / p.name if ds else p


def audit_out(name: str, hand: str, scope=None):
    """Place to **write** an audit output, returned after the write-path check. The folder is created.
    `scope` means the same as in audit_path (the scoped record place)."""
    p = _hand_root_mod().assert_writable(
        audit_path(name, hand, scope=scope), hand, by=os.path.basename(sys.argv[0] or "?")
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def scope_arg(spec: str | None) -> tuple[str, ...] | None:
    """A tool's `--datasets` argument → the datasets the run measures. Empty = None (not narrowed = everything the
    tool covers). The caller sets the scope (the campaign entry decides once and passes it on); the tool only checks
    the names (datasets_scope): a list with no names or an unknown name raises."""
    if not spec:
        return None
    if not [d for d in spec.split(",") if d.strip()]:
        raise InputMissingError(
            f"[SCOPE] ✗ --datasets {spec!r} holds no dataset name — omit the argument to measure everything"
        )
    try:
        return datasets_scope(spec)
    except KeyError as e:
        raise SystemExit(f"[SCOPE] ✗ --datasets: {str(e).strip(chr(39) + chr(34))}") from e


def scope_marker(tool: str, scope, counts: dict) -> dict:
    """One scope marker line (stderr) `[<tool>] scope=<ds,…|all> checked=<n> skipped_out_of_scope=<ds:n,…|->`
    and the same content for the record (all = `datasets: None`). `counts` = dataset → number of units the tool
    counts (in and out of scope); out of scope is not measured, only counted (no silent skip). The marker form
    lives only here."""
    inside = {d: int(n) for d, n in counts.items() if scope is None or d in scope}
    skipped = {d: int(n) for d, n in sorted(counts.items()) if d not in inside}
    print(
        f"[{tool}] scope={','.join(scope) if scope is not None else 'all'} checked={sum(inside.values())}"
        f" skipped_out_of_scope={','.join(f'{d}:{n}' for d, n in skipped.items()) or '-'}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "datasets": list(scope) if scope is not None else None,
        "checked": sum(inside.values()),
        "skipped_out_of_scope": skipped,
    }


#: Articulated-object asset spec — asset rule `dexengine/data/artic/ranges.py` (asset_spec), stage
#: `dexengine/data/artic/spec_stage.py` (stage-1 step 2b calls it before conversion: spec → USD → cooking dump →
#: surface points → cross-check). OakInk2 joint ranges come from the file's source section (hand-independent, one per
#: name, measured once and fixed); a hand's trees are only checked against it.
ARTIC_SPEC = "artic_ranges.json"
#: **Shared outputs**: committed files that consumers without a running hand (object USD bake, articulated surface
#: points, re-bake, OakInk2 conversion) read → their generator. They live in `configs/datasets/` (SHARED_OUTPUT_DIR),
#: not in a per-hand folder. Consumers read via axis_output(name) (a name off the list raises).
AXIS_OUTPUTS = {ARTIC_SPEC: "dexengine/data/artic/spec_stage.py --dataset <ds> (called by stage-1 step 2b)"}
SHARED_OUTPUT_DIR = ("configs", "datasets")


def axis_output(name: str, must_exist: bool = True, root=None):
    """Place of the committed shared output `name` (`configs/datasets/<name>`; `root` overrides the repository root).
    A name off the list (AXIS_OUTPUTS) or, with must_exist, a missing file raises."""
    if name not in AXIS_OUTPUTS:
        raise InputMissingError(
            f"[AXIS_OUTPUT] ✗ {name!r} is not in the axis-hand shared outputs "
            "(dexengine.core.datasets.AXIS_OUTPUTS) — register it with its generator before reading it"
        )
    from dexengine.core.repo import REPO_ROOT  # noqa: PLC0415

    p = pathlib.Path(root or REPO_ROOT).joinpath(*SHARED_OUTPUT_DIR, name)
    if must_exist and not p.is_file():
        raise InputMissingError(
            f"[AXIS_OUTPUT] ✗ shared output {p} is missing — run the generator "
            f"{AXIS_OUTPUTS[name]} and commit the result"
        )
    return p


def table_path(name: str, hand: str, root=None):
    """Table place of hand `hand` (read) — `docs/experiments/hands/<hand>/tables/<name>`
    (ledger_retired.tsv · unproven_retry.tsv · units_replaced__<spec>.tsv · split_purged_<ds>_<emb>__<spec>.txt)."""
    return _hand_file("exp_table", name, hand, root)


def table_out(name: str, hand: str):
    p = _hand_root_mod().assert_writable(table_path(name, hand), hand, by=os.path.basename(sys.argv[0] or "?"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def retired_ledger_file(hand: str, root=None):
    """Record of the rows the fold retired as outside the current rules — one per hand."""
    return table_path("ledger_retired.tsv", hand, root)


def fold_owner_file(hand: str, root=None):
    """Owner declarations for split fold verdicts (axis, clip → owning campaign, reason) — one per hand. Read by the
    fold's split_verdicts."""
    return table_path("fold_owners.tsv", hand, root)


def unproven_retry_file(hand: str, root=None):
    """D37 retry list — one per hand."""
    return table_path("unproven_retry.tsv", hand, root)


def units_replaced_file(hand: str = ""):
    """**Single owner of the file name** of the segment-window replacement record —
    `docs/experiments/hands/<hand id>/tables/units_replaced__<spec dir>.tsv`.

    The name's axis is the **spec directory** (`hand_dir()`, e.g. `wuji`), because it records events of the
    base tree (data layer, spec axis); the folder is the hand id (docs layer). The writer
    (dexengine/data/segment/split.py _record_replaced) and the reader (status_fold._replaced_units) both use this
    function — separate f-strings would let the axes drift and the fold would miss the record.
    `hand` takes a hand id or a spec directory (empty = the running hand)."""
    P_ = _paths()
    spec = hand_dir(P_.hand_id(hand)) if hand else hand_dir()
    return table_path(f"units_replaced__{spec}.tsv", P_.hand_id(spec))


def split_purged_file(ds: str, emb: str, hand: str = ""):
    """Names of the old segments the split removed — `…/tables/split_purged_<ds>_<emb>__<spec dir>.txt` (written by
    the segment split)."""
    P_ = _paths()
    spec = hand_dir(P_.hand_id(hand)) if hand else hand_dir()
    return table_path(f"split_purged_{ds}_{emb}__{spec}.txt", P_.hand_id(spec))
