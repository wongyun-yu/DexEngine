"""Building piece trees from the registry: slicing arrays, piece task_info and support, quarantine of unregistered
pieces, exclusions inherited from the parent."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from dexengine.core import (
    paths as _P,  # noqa: E402 — owner of layer paths (layer_path); data layer paths are not joined by hand here
)
from dexengine.core.errors import input_missing
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)
from dexengine.data import rules as RR  # noqa: E402
from dexengine.data.artic import (
    convert_version as D1G,  # noqa: E402 — D1g exclusion generator name · piece inheritance rule
)
from dexengine.data.objects import (
    scene_parts as SP,  # noqa: E402 — scene part list check (pieces copy the parent scene; a stale parent fails)
)
from dexengine.data.segment.common import HAND, REPO, _base_dir, _base_emb, _stale_dir  # noqa: F401
from dexengine.data.segment.pieces import (
    PIECE_COPY,
    PIECE_RULES_COPY,
    PIECE_WINDOW_KEY,
    _support_parent_keys,
    _window,
    unknown_support_keys,
    unknown_top_keys,
)


def tree_arrays(tree: Path) -> list[Path]:
    """A base tree's arrays — what a piece slices and the piece invariant compares: ``0/<name>.npz`` whose name has no
    further dot. A dotted name is not the tree's array: ``<file>.<tag>.npz`` is a repair's backup of an array
    (wrist_shift — e.g. the 2e original ``.pre_ws``) and ``<file>.tmp.npz`` an unfinished atomic write. Slicing either
    into a piece would hand it a second, stale copy, and the invariant would read the extra file as a piece defect.
    The rule's one statement is tree_content.tree_arrays (the RL ledger's content identity reads the same set)."""
    from dexengine.data import tree_content as TC  # noqa: PLC0415 — lightweight statement module

    return TC.tree_arrays(tree)


def _slice_npz(src: Path, dst: Path, t0: int, t1: int, T: int) -> None:
    with np.load(src) as z:
        out = {}
        for k in z.files:
            a = z[k]
            out[k] = a[t0:t1] if getattr(a, "ndim", 0) >= 1 and a.shape[:1] == (T,) else a
    np.savez_compressed(dst, **out)


def piece_support(parent_support: dict | None, t0: int, t1: int) -> dict | None:
    """Support of parent base tree → support of piece [t0, t1) = parent-owned fields + rest intervals + window markers.
    Rest intervals are cut to the piece window by support.rest.slice_rest_intervals (the same function as base_adapter
    `_segNN`).
    Window fields are dropped and unknown fields fail. None without a parent."""
    if parent_support is None:
        return None
    from dexengine.data.support.rest import slice_rest_intervals  # noqa: PLC0415 — dependency-free owner

    keep = _support_parent_keys()
    unknown = unknown_support_keys(parent_support)
    if unknown:
        raise ValueError(
            f"[segment_split] unknown fields in the parent support {unknown} — window-independent ones go to "
            "base_layer.common.SUPPORT_COPIED, window-measured ones to their producer's window field declaration "
            "(WINDOW_DECL)"
        )
    sup = json.loads(json.dumps({k: parent_support[k] for k in keep if k in parent_support}))
    sup["rest_intervals"] = slice_rest_intervals(parent_support.get("rest_intervals"), int(t0), int(t1))
    sup[PIECE_WINDOW_KEY] = [int(t0), int(t1)]
    return sup


def piece_task_info(ti: dict, name: str, clip: str, t0: int, t1: int) -> dict:
    """task_info of parent base tree → task_info of piece `name` = [t0, t1) (window rule, see the segment.pieces
    docstring)."""
    unknown = unknown_top_keys(ti)
    if unknown:
        raise ValueError(
            f"[segment_split] {clip}: unknown top-level fields {unknown} — window-independent ones go to "
            "dexengine.data.segment.pieces.PIECE_COPY, "
            "window-measured ones to their producer's window field declaration (piece_schema.WINDOW_DECL) · "
            "fields of an old "
            "generation mean the parent is stale (the stage-1 column 2 rebuilds it; piece fixes run only on "
            "current-generation parents)"
        )
    nt = json.loads(json.dumps({k: ti[k] for k in PIECE_COPY + ("quality", "rules", "segment") if k in ti}))
    qw = _window("quality")
    nt["quality"] = {k: v for k, v in (nt.get("quality") or {}).items() if k not in qw}
    nt["rules"] = {k: v for k, v in (nt.get("rules") or {}).items() if k in PIECE_RULES_COPY}
    nt.update(
        clip_id=name,
        parent_clip=clip,
        segment_range=[t0, t1],
        segment_source="INTAKE/units registry (segment_split.py --apply, D36 v2)",
    )
    sg = nt.setdefault("segment", {})
    sg["start"] = int(sg.get("start", 0) or 0) + t0
    sg["end"] = sg["start"] + (t1 - t0)
    sg["T"] = t1 - t0
    sg["cut_source"] = "segment_split.py(INTAKE/units)"
    if "T_kin" in (sg.get("frame_map") or {}):
        sg["frame_map"]["T_kin"] = t1 - t0
    if "support" in ti:
        nt["support"] = piece_support(ti["support"], t0, t1)
    return nt


def reslice_stale(ds: str, emb: str, table: dict, write: bool, label: str) -> dict:
    """Re-slice registered pieces that are not window pieces of the current parent: only this hand's trees and wrist npz
    move to the quarantine data/<ds>/base_stale/<hand>/<emb>/<label>/<piece> (never deleted; hand-independent sidecars
    are read by other hands too and stay), the piece is rebuilt from the parent with split_clip_registry and recorded in
    units_replaced (the fold retires judgement rows older than that time → rejudge). Missing pieces and pieces without a
    parent are not this job's (new pieces: stage1.split_apply; gaps: recorded only)."""
    from dexengine.data.segment import (
        invariant as PI,  # noqa: PLC0415 — owner of piece invariants (it imports this file; the late import breaks the cycle)
    )

    out = {"checked": 0, "stale": 0, "resliced": 0, "why": {}, "units": []}
    for clip, units in table.items():
        if not (_base_dir(ds, emb, clip) / "task_info.json").is_file():
            continue
        for u in units:
            if not (_base_dir(ds, emb, u["name"]) / "task_info.json").is_file():
                continue
            out["checked"] += 1
            why = PI.piece_stale_reason(ds, emb, clip, u)
            if not why:
                continue
            out["stale"] += 1
            out["why"][why] = out["why"].get(why, 0) + 1
            out["units"].append(u["name"])
            if write:
                cur = json.loads((_base_dir(ds, emb, u["name"]) / "task_info.json").read_text()).get("segment_range")
                _quarantine_one(ds, emb, u["name"], label=label, shared=False)
                split_clip_registry(ds, emb, clip, units, {u["name"]})
                _record_replaced(ds, emb, u["name"], cur, [int(u["t0"]), int(u["t1"])])
                out["resliced"] += 1
    return out


def refresh_piece_support(ds: str, emb: str, table: dict, write: bool) -> dict:
    """Re-slice existing pieces' support from the parent (`--apply --refresh-support`); idempotent, the input is always
    the parent tree. Returns {"checked", "changed", "no_parent", "no_piece", "changed_units": [...]}. Writes go through
    tree_io.write_json_if_changed (equal content keeps mtime, so f0 cache fingerprints break only on pieces that really
    changed)."""
    from dexengine.data import tree_io as _TIO  # noqa: PLC0415

    out = {"checked": 0, "changed": 0, "no_parent": 0, "no_piece": 0, "changed_units": []}
    for clip, units in table.items():
        src = _base_dir(ds, emb, clip)
        if not (src / "task_info.json").is_file():
            out["no_parent"] += 1
            continue
        psup = json.loads((src / "task_info.json").read_text()).get("support")
        for u in units:
            dst = _base_dir(ds, emb, u["name"]) / "task_info.json"
            if not dst.is_file():
                out["no_piece"] += 1
                continue
            ti = json.loads(dst.read_text())
            out["checked"] += 1
            cur = ti.get("support")
            if cur is None or psup is None:
                new = piece_support(psup, int(u["t0"]), int(u["t1"]))
            else:
                # Existing pieces get **only their rest intervals and window markers** re-sliced; other fields belong to
                # the piece (e.g. the D24 input obj_zmin_traj object_roles wrote from the piece's own trajectory).
                # Overwriting with the whole parent support would replace those values with the parent's.
                from dexengine.data.support.rest import slice_rest_intervals  # noqa: PLC0415 — dependency-free owner

                new = dict(cur)
                new["rest_intervals"] = slice_rest_intervals(psup.get("rest_intervals"), int(u["t0"]), int(u["t1"]))
                new[PIECE_WINDOW_KEY] = [int(u["t0"]), int(u["t1"])]
            if cur == new:
                continue
            out["changed"] += 1
            out["changed_units"].append(u["name"])
            if write:
                ti["support"] = new
                _TIO.write_json_if_changed(dst, ti)
    return out


def _quarantine_one(ds: str, emb: str, unit: str, label: str = "", shared: bool = True) -> Path:
    """Move one piece outside the registry, **without deleting it**, to a quarantine outside base: the base tree · this
    hand's wrist npz (per side wrist/wrist_poses_<side>/) · the sidecar of that name.
    Outputs of the canonical axis hand are never deleted by any cleanup; old pieces the registry dropped are only moved.
    Location = data/<ds>/base_stale/<hand spec>/<emb>/<piece> (outside the layer paths, so no layer glob matches it).
    Returns that location.
    label = quarantine generation subfolder (when the same name is moved several times) · shared=False = leave the
    hand-independent sidecar and move only this hand's files (other hands read the sidecars of registered pieces)."""
    dst = _stale_dir(ds, emb) / (label or ".") / unit
    if dst.exists():
        raise SystemExit(f"[split] ✗ quarantine location exists {dst} — not overwriting")
    dst.mkdir(parents=True)
    src = _base_dir(ds, emb, unit)
    if src.is_dir():
        shutil.move(str(src), str(dst / "tree"))
    wb = _P.layer_path("wrist_base", ds, hand=HAND, emb=emb)
    for side in ("right", "left"):
        # both sides' wrist files have the same name (<piece>.npz); the layer's side folders are kept so one never
        # overwrites the other.
        for lbl, f in (
            (f"wrist/wrist_poses_{side}", wb / f"wrist_poses_{side}" / f"{unit}.npz"),
            ("contact_all", _P.sidecar_file("contact_all", ds, unit, side, HAND)),
            ("contact_base", _P.sidecar_file("contact_base", ds, unit, side, HAND)),
        )[: 3 if shared else 1]:
            if f.is_file():
                (dst / lbl).mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst / lbl / f.name))
    return dst


def unregistered_units(ds: str, emb: str, table: dict, purge: bool) -> tuple[list, int]:
    """Pieces (_tN) present in this hand's base but **missing** from the registry (table): leftovers after the registry
    dropped them. With purge, move them to the quarantine (`_quarantine_one`, never deleted) and record them in
    units_replaced__<hand>.tsv as new_range=removed (the fold retires their judgement rows). Returns (those pieces,
    moved count)."""
    reg_names = {u["name"] for us in table.values() for u in us}
    broot = _base_emb(ds, emb)
    here_units = sorted(c for c in os.listdir(broot) if re.fullmatch(r".+_t\d+", c)) if broot.is_dir() else []
    unreg = [c for c in here_units if c not in reg_names]
    purged = 0
    for c in unreg:
        if purge:
            cur = (
                json.loads((broot / c / "task_info.json").read_text()).get("segment_range")
                if (broot / c / "task_info.json").is_file()
                else None
            )
            where = _quarantine_one(ds, emb, c)
            _record_replaced(ds, emb, c, cur, "removed")
            print(
                f"[split] {ds}:{c} outside the registry → quarantined {where.relative_to(_data_root())} (not deleted)",
                flush=True,
            )
            purged += 1
        else:
            print(
                input_missing(
                    f"[split] ✗ {ds}:{c} in base but not in the registry — quarantine it with --purge-unregistered"
                ),
                file=sys.stderr,
            )
    return unreg, purged


def purge_units(ds: str, emb: str) -> dict:
    """Remove the existing _tN units of the target dataset and emb: base trees, this hand's wrist npz, MANO sidecars
    (hand-independent layer). The removed names are written to split_purged_<ds>_<emb>__<spec>.txt in the hand table
    folder (data.rules.split_purged_file). Nothing in this package reads
    that list back."""
    root = _base_emb(ds, emb)
    units = sorted(c for c in os.listdir(root) if re.fullmatch(r".+_t\d+", c)) if root.is_dir() else []
    wb = _P.layer_path("wrist_base", ds, hand=HAND, emb=emb)
    n_w = n_s = 0
    for u in units:
        shutil.rmtree(root / u)
        for side in ("right", "left"):
            f = wb / f"wrist_poses_{side}" / f"{u}.npz"
            if f.is_file():
                f.unlink()
                n_w += 1
            g = _P.sidecar_file("contact_all", ds, u, side, HAND)
            if g.is_file():
                g.unlink()
                n_s += 1
    lst = RR._hand_root_mod().assert_writable(RR.split_purged_file(ds, emb), RR.active_hand(), by="segment_split")
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("\n".join(units) + ("\n" if units else ""))
    return {"trees": len(units), "wrist": n_w, "sidecar": n_s, "list": str(lst.relative_to(REPO))}


def _record_replaced(ds: str, emb: str, unit: str, old: list | None, new: list) -> None:
    """Record a replaced piece with its time, so that judgement rows of lists older than that time can be
    retired."""
    import datetime as _dt  # noqa: PLC0415

    f = RR.units_replaced_file()  # data.rules alone owns the name
    RR._hand_root_mod().assert_writable(f, RR.active_hand(), by="segment_split")  # hand table folder; owner check
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.is_file():
        f.write_text("dataset\temb\tclip\told_range\tnew_range\treplaced_at\n")
    with open(f, "a") as fh:
        fh.write(f"{ds}\t{emb}\t{unit}\t{old}\t{new}\t{_dt.datetime.now().isoformat(timespec='seconds')}\n")


def apply_table(ds: str, emb: str, table: dict, purge_unregistered: bool) -> dict:
    """Build this hand's (HAND) base pieces from the registry table: existing ones are compared by window (a difference
    is bad, never silently overwritten) · existing pieces **missing** from the registry are moved only when quarantine
    was asked, otherwise bad (no silent leftovers; set difference)."""
    made = kept = missing = bad = 0
    for clip, units in table.items():
        if not (_base_dir(ds, emb, clip) / "task_info.json").is_file():
            missing += 1
            continue  # this hand has no parent (retargeting gap): no pieces either, recorded only
        todo = []
        for u in units:
            dst = _base_dir(ds, emb, u["name"])
            if (dst / "task_info.json").is_file():
                cur = json.loads((dst / "task_info.json").read_text()).get("segment_range")
                if cur != [u["t0"], u["t1"]]:
                    print(
                        f"[split] ✗ {ds}:{u['name']} existing window {cur} ≠ registry {[u['t0'], u['t1']]} — "
                        "differs from the registry: "
                        "reslice first (--reslice-stale --label <generation>)",
                        file=sys.stderr,
                    )
                    bad += 1
                else:
                    kept += 1
                continue
            todo.append(u["name"])
        if todo:
            made += len(split_clip_registry(ds, emb, clip, units, set(todo)))
    unreg, purged = unregistered_units(ds, emb, table, purge_unregistered)
    if unreg and not purge_unregistered:
        bad += len(unreg)
    return {"made": made, "kept": kept, "missing": missing, "bad": bad, "unreg": len(unreg), "purged": purged}


def split_clip_registry(ds: str, emb: str, clip: str, units: list[dict], want: set[str]) -> list[str]:
    """Build only the registry's pieces (names and windows fixed): the same output as split_clip (copy · slice · segment
    window), with registry names."""
    src = _base_dir(ds, emb, clip)
    ti = json.loads((src / "task_info.json").read_text())
    T = int((ti.get("case") or {}).get("T") or 0) or int(np.load(src / "0/trajectory_kinematic.npz")["qpos"].shape[0])
    made, written = [], {}
    for u in units:
        name = u["name"]
        if name not in want:
            continue
        t0, t1 = max(0, int(u["t0"])), min(T, int(u["t1"]))
        dst = _base_dir(ds, emb, name)
        (dst / "0").mkdir(parents=True, exist_ok=True)
        SP.verify_dir(
            src
        )  # convex parts of the parent scene = current decomposition, otherwise the parent (base) is stale
        for x in SP.SCENE_XMLS:
            if (src / x).is_file():
                shutil.copy2(src / x, dst / x)
        for f in tree_arrays(src):
            _slice_npz(f, dst / "0" / f.name, t0, t1, T)
        # window rule (piece_task_info): window-measured values are not carried over; cases, roles and quality
        # re-measure on the piece window.
        nt = piece_task_info(ti, name, clip, t0, t1)
        (dst / "task_info.json").write_text(json.dumps(nt, ensure_ascii=False, indent=1))
        made.append(name)
        written[name] = nt
    sync_piece_exclusions(ds, written)
    return made


def sync_piece_exclusions(ds: str, pieces: dict[str, dict]) -> int:
    """**Derive** the generator lines of the exclusion ledger from the piece rule, which owns the statement
    (segment.pieces.PIECE_RULES_COPY: pieces receive the parent's conversion markers when cut).
    Rule: when a piece just built received generator <k>'s conversion record (rules.<k>), that piece's <k> exclusion
    line
    is a stale line judging the old window; it is removed through the ledger writer
    (data.rules.retarget_excluded_record).
    This does at cutting time what that generator would do on its next run (trees with a conversion record have no
    line).
    Lines of pieces without a conversion record, human lines and other generators' lines stay.
    Inheritance: when the parent of a piece without a conversion record is excluded by D1g, the parent's verdict moves
    to
    the piece through convert_version.piece_verdict (single owner of the inheritance rule; the converter uses the same
    function):
    a verdict on the object itself writes the same exclusion line under the piece name (description head 'inherited
    (parent …)'), an interval-dependent verdict writes nothing (re-measured in the piece window, 2b on that piece).
    Returns lines removed + lines inherited · marker [PIECE_EXCLUDE_SYNC]."""
    n = 0
    for src in sorted({k for nt in pieces.values() for k in (nt.get("rules") or {})}):
        listed = RR.retarget_excluded_by(ds, src)
        stale = sorted(p for p, nt in pieces.items() if src in (nt.get("rules") or {}) and p in listed)
        if not stale:
            continue
        r = RR.retarget_excluded_record(ds, src, {}, stale)
        print(
            f"[PIECE_EXCLUDE_SYNC] {ds}: pieces with rules.{src}: stale {src} exclusion lines removed {len(stale)} "
            f"(e.g. {stale[0]}) · ledger {r['path']}",
            flush=True,
        )
        n += len(stale)
    return n + inherit_piece_exclusions(ds, pieces)


def inherit_piece_exclusions(ds: str, pieces: dict[str, dict]) -> int:
    """Piece inheritance: pieces without a conversion record (rules.D1g) whose parent D1g excluded receive the parent's
    verdict (second half of sync_piece_exclusions; rule = artic.convert_version.piece_verdict).
    Returns the number of inherited lines; pieces to re-measure are in the marker."""
    src = D1G.EXCLUDE_SRC
    listed = RR.retarget_excluded_by(ds, src)
    full = RR.retarget_excluded(ds)
    verdicts, remeasure = {}, []
    for p, nt in sorted(pieces.items()):
        par = nt.get("parent_clip")
        if src in (nt.get("rules") or {}) or par not in listed:
            continue
        v = D1G.piece_verdict(par, listed[par], full[par].partition(" — ")[2])
        if v is None:
            remeasure.append(p)
        else:
            verdicts[p] = v
    if not verdicts and not remeasure:
        return 0
    path = RR.retarget_excluded_record(ds, src, verdicts, sorted(verdicts))["path"] if verdicts else "-"
    print(
        f"[PIECE_EXCLUDE_SYNC] {ds}: parent {src} exclusions inherited {len(verdicts)} "
        f"(e.g. {next(iter(verdicts), '-')}) · re-measured in the piece window (2b) {len(remeasure)} "
        f"(e.g. {remeasure[0] if remeasure else '-'}) · ledger {path}",
        flush=True,
    )
    return len(verdicts)
