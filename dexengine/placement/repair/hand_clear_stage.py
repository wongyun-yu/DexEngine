#!/usr/bin/env python3
"""Stage 1 step 2f — S-hand-clear (rule C) over one dataset's base trees of one hand. The rule and its ruler are
data/objects/hand_clear's; this file is the driver: the per-(clip, side) ledger, the hand-scoped exclusion lines and the
step's column (stage-1 table).

Position (stage-1 table, check_inputs): after 5 — it reads the D1-g stamp (2b), the all-object sidecar (2d), the case
stamp (3: the extra classifier's human contact), the main objects (4) and their contact sidecar (5: first touch and
start grip). It writes no tree, so the steps around it read the same trees; its ledger and exclusion lines are read
after stage 1 (candidates). Its input is the tree as 2e left it (3 … 5 write no qpos).
S-hand-clear is measured only and no tree is written: a side whose hand enters a still extra (a row of that side reads
`need`) keeps its reference as recorded and stays in the chain (status KEPT_AS_RECORDED, reason word
hand_clear.KEPT_WORD, the side's max depth in `rows`; the robot-static collision keeps the robot out of the object in
training and the RL judge decides adoption). What each status does to the clip is ACTION, one table. A tree that still
holds a 2f repair output (wrist_shift tag HAND_CLEAR_TAG) is not measured: the run fails (rc 2) and names it, since
that output is not the reference as recorded (base_write.undo_tree puts the input back).
The same step drops a clip whose still rows state a true-shape object overlap (collider_shape.overlap_drop: S-touch-gate
and S-envelope pairs above scene_requirement.TRUE_SHAPE_OVERLAP_M), one ledger row with side OBJECTS (status DROP),
a clip whose free main object would jump more than scene_requirement.FLOAT_SETTLE_MAX_M at the first touch
(S-float-settle, float_settle.stage_row: one row with side float_settle.ROW_SIDE, the statement of every main object),
and a clip S-dyn-pairs drops (dyn_pairs.stage_rows: side dyn_pairs.PAIR_SIDE = a both-dynamic pair whose recorded
visual overlap is above the true-shape overlap, side dyn_pairs.TOGETHER_SIDE = a pair that moves together without an
exact one-body path).
Output: the ledger ``ledger_<ds>.tsv`` in the hand's run log (LEDGER_COLS; one row per (clip, side), status one of
STATUSES; `rows` = the still rows as name:verdict:depth_m of that side's hand plus max_depth_m:<m>, or of the objects
row as name:partner:depth_m; `qpos_sha` = the step's input) and, with --write, one exclusion-ledger line per clip whose
status ACTION excludes, scoped to this hand (rules.retarget_excluded_record, generator GENERATOR, reason word
collider_shape.OVERLAP_DROP_WORD, float_settle.DROP_WORD, dyn_pair_check.DROP_WORD, dyn_pairs.NO_PATH_WORD, then
hand_clear.KEPT_WORD for a hand status ACTION excludes).
Without --write the step measures only (``ledger_<ds>.measure.tsv``, no exclusion line).
With --clips-file the step measures only the listed base trees (the long-clip split's changed pieces): their rows
replace theirs in the ledger and every other tree's lines stay as they are (`merge_ledger`); the exclusion lines of
unlisted clips stay too (rules.retarget_excluded_record replaces only scanned clips). An empty list, a name that is
not one base tree, or an unlisted tree whose verdict is not current (the list is short: its kept line would be stale)
fails (rc 2) before anything is measured — the caller skips the step when it has nothing to measure.
Column: a tree is current when the write ledger holds its clip with the sha of the step's qpos, its input key and the
current rule version, and the tree holds no 2f repair output. The input key (`inputs`, stage1.artifacts.input_key) is
the bytes of every input the step reads: task_info as the steps up to 2f leave it, the contact sidecars of the
artifacts the table says 2f reads (table.step_reads), the tree's scene files, the take's MANO keypoints and every
object's geometry folders; the rule version carries the rule values, the shape rule and the ruler's identity. A verdict
is reused only while all of them are the same.
Worker memory: a pool worker builds the shape of every object its trees hold and keeps it (collider_shape._CACHE),
and glibc keeps the heap a tree frees. Over one worker's trees both grow without bound. ``tree_job`` drops the
shapes and hands the heap back after each tree (capacity.task), so a worker's memory stays bounded; the rows are
byte-identical either way and the CPU cost is about 1.5 %.
Marker: ``[HAND_CLEAR] trees N · excluded k by_reason {...} · kept_as_recorded u (max depth m) · rows CLEAN … DROP …
KEPT_AS_RECORDED …``.
Usage: <retarget py> dexengine/placement/repair/hand_clear_stage.py --dataset <ds> --hand <hand> [--write] [--jobs N]
       [--clips-file <file: one base tree name per line>]
rc 0 · 2 (a tree could not be measured: no verdict is written for any tree) · 3 (no base tree).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from dexengine.core import capacity, stdio
from dexengine.core.errors import input_missing
from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import dyn_pair_check as DC
from dexengine.data.objects import dyn_pairs as DP
from dexengine.data.objects import float_settle as FS
from dexengine.data.objects import hand_clear as HC
from dexengine.data.objects import scene_requirement as SR
from dexengine.data.objects import scene_rules as SC
from dexengine.placement.repair import wrist_shift as WS
from dexengine.placement.repair.wrist_desk_fresh import qpos_sha_file, stage_trees

GENERATOR = "HC"  # exclusion-ledger generator name
OBJECTS = "objects"  # ledger side of the clip's true-shape object overlap row (collider_shape.overlap_drop)
LEDGER_COLS = ("clip", "side", "status", "rows", "qpos_sha", "inputs", "version")
STATUSES = ("CLEAN", "DROP", "KEPT_AS_RECORDED")
CLEAN, DROP, KEPT = STATUSES
#: What a status does to its clip — the one table (a decision to drop the sides kept as recorded is one line):
#: keep = the clip stays in the chain (KEPT_AS_RECORDED with the reference as recorded), exclude = an exclusion-ledger
#: line (hand-scoped) takes the clip out (DROP: the objects row and the float row).
ACTION = {CLEAN: "keep", DROP: "exclude", KEPT: "keep"}
TAG = WS.HAND_CLEAR_TAG  # wrist_shift backup tag of a 2f repair output (the step writes none)
READING_VERSION = "HC-M2"  # the step's rule C reading (measured only); bump when it changes (column stale)
KIN = "0/trajectory_kinematic.npz"
MODULE = "dexengine.placement.repair.hand_clear_stage"  # stage-1 freshness owner name · capacity.NEEDS key


def input_key(tree) -> str:
    """The tree's input key for this step (stage1.artifacts.input_key: its table step, its hand from the tree path)."""
    from dexengine.data.stage1 import artifacts as AF  # noqa: PLC0415

    t = Path(tree)
    return AF.input_key(AF.owner_step(MODULE), t, t.parent.parent.name)


def shape_rule(ds: str) -> str:
    """The shape rule as its owner decides it for dataset `ds` (collider_shape.shape_of of every still kind, pinned and
    not, plus the support) — read through the owner, never its table."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    kinds = sorted({*XS.KINDS, *XS.SCOPED_KINDS, *XS.SPAWNED_ROLES})
    return "+".join(f"{k}{''.join(CS.shape_of(ds, k, p)[0] for p in (True, False))}" for k in kinds)


def rule_version(ds: str) -> str:
    """The version a ledger row of dataset `ds` is current for: the rule ids and their values (the true-shape overlap,
    the target kinds, the reading, what a side kept as recorded does, the float rule, the S-dyn-pairs version), the
    shape rule (which kinds stand as triangles, on for this dataset) and the ruler's identity (sdf_grid.ruler_id)."""
    from dexengine.data.objects.sdf_grid import ruler_id  # noqa: PLC0415 — the ruler's identity owner

    kinds = "+".join(HC.target_kinds(ds))
    return (
        f"{HC.RULE_ID}:{CS.RULE_ID}:overlap={SR.TRUE_SHAPE_OVERLAP_M}:targets={kinds}"
        f":reading={READING_VERSION}:kept={ACTION[KEPT]}"
        f":{FS.RULE_ID}:rest={SR.FLOAT_REST_M}:settle_max={SR.FLOAT_SETTLE_MAX_M}"
        f":{DP.RULE_ID}:{SC.VERSIONS[DP.RULE_ID]}:together={DP.together_values(DP.EXTRA)}/{DP.together_values(DP.MAIN)}"
        f":shapes={shape_rule(ds)}:ruler={ruler_id()[:12]}"
    )


def step_input(t: Path) -> Path:
    """The npz this step reads: the tree's kinematic file, as 2e left it (the step writes none)."""
    return Path(t) / KIN


def holds_repair(t: Path) -> bool:
    """Does the tree still hold a 2f repair output (wrist_shift tag TAG)?"""
    return WS.wrote(step_input(t), TAG)


def side_status(v: dict, side: str) -> dict:
    """{status, why, depth_m} of one side from the row verdicts `v` (hand_clear.row_verdict): KEPT_AS_RECORDED when a
    still row reads `need` for this side's hand, else CLEAN; depth_m = the side's max depth over the rows."""
    d = {n: float(x["sides"].get(side, 0.0)) for n, x in v.items()}
    deep = max(d.values(), default=0.0)
    need = [f"{n} {x} m" for n, x in sorted(d.items()) if HC.judge(x) == "need"]
    if not need:
        return {"status": CLEAN, "why": "", "depth_m": deep}
    return {"status": KEPT, "why": f"measured, kept as recorded: {' · '.join(need)} (max {deep} m)", "depth_m": deep}


def _side_row(clip: str, side: str, v: dict, sha: str, ver: str) -> dict:
    """The ledger row of one hand side (module doc: `rows` = name:verdict:depth_m per still row, max_depth_m)."""
    st = side_status(v, side)
    rows = [f"{n}:{HC.judge(d)}:{d}" for n, d in sorted((n, float(x["sides"].get(side, 0.0))) for n, x in v.items())]
    tail = f" {HC.KEPT_WORD}" if st["status"] == KEPT else ""
    return {
        "clip": clip,
        "side": side,
        "status": st["status"],
        "rows": " ".join([*rows, f"max_depth_m:{st['depth_m']}"]) + tail,
        "qpos_sha": sha,
        "version": ver,
        "why": st["why"],
        "depth_m": st["depth_m"],
    }


def tree_job(tree: str) -> tuple[str, list[dict], str]:
    """tree_rows as a pool task: the object shapes it built are dropped afterwards (module doc, Worker memory)."""
    # collider_shape is not edited for this (its bytes are the 2f ruler's identity, sdf_grid.ruler_id: a changed byte
    # makes every 2f verdict stale), so its cache is named here
    with capacity.task(CS._CACHE):
        return tree_rows(tree)


def tree_rows(tree: str) -> tuple[str, list[dict], str]:
    """(clip, ledger rows, error) of one base tree (module doc). The step writes no tree; a tree that holds a 2f repair
    output is an error (not measured)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    t = Path(tree)
    try:
        if holds_repair(t):
            raise ValueError(f"holds a 2f repair output (wrist_shift tag {TAG}); base_write.undo_tree restores it")
        key = input_key(t)  # before measuring: the inputs this verdict is taken from
        ti = json.loads((t / "task_info.json").read_text(encoding="utf-8"))
        src = step_input(t)
        with np.load(src) as z:
            q = np.asarray(z["qpos"], float)
        with stdio.capture():
            still = XS.clip_rows(str(t), ti, q, strict=False)
            v = HC.clip_verdicts(str(t), ti, q, still)
            fp = CS.statement_fp(str(ti["dataset_name"]), XS.slot_rows_of(str(t), still)[0])  # = env's
    except Exception as e:  # noqa: BLE001 — the run reports every unmeasured tree and writes no verdict
        return t.name, [], f"{type(e).__name__}: {str(e)[:200]}"
    sha, ver = qpos_sha_file(src), rule_version(str(ti["dataset_name"]))
    rows = [_side_row(t.name, str(s), v, sha, ver) for s in (ti.get("active_sides") or ["right", "left"])]
    od = CS.overlap_drop(still)  # a true-shape object overlap drops the clip whatever the hands (one row, side objects)
    rows.append(
        {
            "clip": t.name,
            "side": OBJECTS,
            "status": DROP if od else CLEAN,
            "rows": " ".join(
                [f"statement:{fp}"]
                + [
                    f"{r['name']}:{r['overlap_drop']['partner']}:{r['overlap_drop']['depth_m']}"
                    for r in still
                    if r.get("overlap_drop")
                ]
            ),
            "qpos_sha": sha,
            "version": ver,
            "why": (od or ("", ""))[1],
        }
    )
    rows.append(
        FS.stage_row(str(t), ti, q, still, sha, ver, STATUSES)
    )  # S-float-settle: a float that jumps > cap drops the clip
    try:  # S-dyn-pairs: a visual overlap above the cap, or together without a one-body path, drops
        with stdio.capture():
            rows += DP.stage_rows(str(t), ti, q, still, sha, ver, (CLEAN, DROP))
    except Exception as e:  # noqa: BLE001 — a pair that cannot be read: the tree is unmeasured, as above (no verdict)
        return t.name, [], f"S-dyn-pairs {type(e).__name__}: {str(e)[:200]}"
    for r in rows:
        r["inputs"] = key
    return t.name, rows, ""


def drop_verdicts(rows: list[dict]) -> dict[str, tuple[str, str]]:
    """{clip: (reason word, description)} of the clips with a row whose status ACTION excludes — the exclusion-ledger
    verdicts. The word is the first that applies of: the true-shape overlap's (objects row DROP,
    collider_shape.OVERLAP_DROP_WORD), the float jump's (float row DROP, float_settle.DROP_WORD), the both-dynamic
    visual overlap's (dyn_pairs.PAIR_SIDE DROP, dyn_pair_check.DROP_WORD), the together pair's (dyn_pairs.TOGETHER_SIDE
    DROP, dyn_pairs.NO_PATH_WORD), then for a hand side whose status ACTION excludes: kept as recorded
    (hand_clear.KEPT_WORD). An excluded row with no reason word raises."""
    clip_rows = {
        (OBJECTS, DROP): (0, CS.OVERLAP_DROP_WORD),
        (FS.ROW_SIDE, DROP): (1, FS.DROP_WORD),
        (DP.PAIR_SIDE, DROP): (2, DC.DROP_WORD),
        (DP.TOGETHER_SIDE, DROP): (3, DP.NO_PATH_WORD),
    }
    hand_rows = {KEPT: (4, HC.KEPT_WORD)}
    why: dict[str, list[str]] = {}
    word: dict[str, tuple[int, str]] = {}
    for r in rows:
        if ACTION[r["status"]] != "exclude":
            continue
        hand_side = r["side"] not in (OBJECTS, FS.ROW_SIDE, DP.PAIR_SIDE, DP.TOGETHER_SIDE)
        k = hand_rows.get(r["status"]) if hand_side else clip_rows.get((r["side"], r["status"]))
        if k is None:
            raise ValueError(f"[HAND_CLEAR] {r['clip']} {r['side']} {r['status']}: excluded with no reason word")
        why.setdefault(r["clip"], []).append(f"{r['side']}: {r['why']}")
        word[r["clip"]] = min(word.get(r["clip"], k), k)
    return {c: (word[c][1], " | ".join(w)) for c, w in why.items()}


def ledger_path(ds: str, hand: str, write: bool) -> Path:
    """The write run's ledger ``ledger_<ds>.tsv``; a measuring run's is tagged ``.measure`` (paths.mode_record)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    return P.mode_record(
        P.run_path("run_log", hand, tool="hand_clear") / f"ledger_{ds}.tsv", None if write else "measure"
    )


def _read_ledger(p: Path) -> dict:
    """{clip: (qpos_sha, inputs, version)} of a ledger file ({} when absent; a ledger without `inputs` reads None)."""
    if not p.is_file():
        return {}
    lines = p.read_text(encoding="utf-8").splitlines()
    cols = lines[0].split("\t") if lines else []
    out = {}
    for ln in lines[1:]:
        r = dict(zip(cols, ln.split("\t"), strict=True))
        out[r["clip"]] = (r["qpos_sha"], r.get("inputs"), r["version"])
    return out


def _current(rec, tree: Path, ver: str) -> bool:
    if rec is None or holds_repair(tree):  # a tree holding a 2f repair output is not the step's input
        return False
    if (rec[0], rec[2]) != (qpos_sha_file(step_input(tree)), ver):
        return False
    try:
        return rec[1] == input_key(tree)
    except (OSError, ValueError, KeyError) as e:  # an input that cannot be read: not current (the run reports it)
        print(
            f"[HAND_CLEAR] {Path(tree).name}: input key unreadable ({type(e).__name__}: {str(e)[:120]})",
            file=sys.stderr,
        )
        return False


def tree_fresh(ds: str, spec_dir: str, tree) -> bool:
    """Is this base tree's 2f verdict current (write ledger row with its input's sha and the current rule version;
    no 2f repair output on the tree)?"""
    from dexengine.core import paths as P  # noqa: PLC0415

    rec = _read_ledger(ledger_path(ds, P.hand_id(spec_dir), True)).get(Path(tree).name)
    return _current(rec, Path(tree), rule_version(ds))


def freshness(ds: str, spec_dir: str) -> tuple[int, int]:
    """(base trees whose 2f verdict is current, all base trees) — the stage-1 column."""
    from dexengine.core import paths as P  # noqa: PLC0415

    rec, ver = _read_ledger(ledger_path(ds, P.hand_id(spec_dir), True)), rule_version(ds)
    trees = stage_trees(ds, spec_dir)
    return sum(_current(rec.get(Path(t).name), Path(t), ver) for t in trees), len(trees)


def merge_ledger(old_lines: list[str], rows: list[dict], order: list[str], measured: set) -> list[str]:
    """The ledger lines after a scoped run: the header, then per base tree in `order` the new rows of a measured tree
    and the old lines, verbatim, of any other (lines of a tree no longer in the base go, as a whole run drops them).
    An old ledger with another header fails."""
    head = "\t".join(LEDGER_COLS)
    if old_lines and old_lines[0] != head:
        raise SystemExit(f"[HAND_CLEAR] ✗ the ledger header {old_lines[0]!r} is not {head!r} — rerun without a scope")
    old: dict[str, list[str]] = {}
    for ln in old_lines[1:]:
        old.setdefault(ln.split("\t", 1)[0], []).append(ln)
    new: dict[str, list[str]] = {}
    for r in rows:
        new.setdefault(r["clip"], []).append("\t".join(str(r[k]) for k in LEDGER_COLS))
    return [head] + [ln for c in order for ln in (new.get(c, []) if c in measured else old.get(c, []))]


def scoped(trees: list[str], clips: str, ds: str) -> list[str] | None:
    """The base trees a --clips-file names, in base order (None without a file). An empty list, a name that is not a
    base tree, or base trees that share a name fail with the reason (the caller returns rc 2)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    if not clips:
        return None
    want = set(P.read_clip_list(clips, dataset=ds, quiet=True))
    named = {Path(t).name: t for t in trees}
    if not want:
        raise ValueError(f"--clips-file {clips} names no tree (an empty scope is not 'measure nothing')")
    if len(named) != len(trees):
        raise ValueError(f"{len(trees) - len(named)} base trees share a name — the list cannot be resolved to trees")
    if lost := sorted(want - set(named)):
        raise ValueError(f"{len(lost)} listed names are not base trees, e.g. {lost[:3]}")
    return [t for t in trees if Path(t).name in want]


def run(ds: str, hand: str, write: bool, *, jobs: int | None, clips: str = "") -> int:
    """The step over one dataset of one hand, or over the trees `clips` lists (the CLI is this) -> rc."""
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data import rules as RR  # noqa: PLC0415

    every = [str(t) for t in stage_trees(ds, P.hand_spec_dir(hand))]
    if not every:
        print(f"[HAND_CLEAR] ✗ {ds} hand={hand}: 0 base trees", file=sys.stderr)
        return 3
    try:
        scope = scoped(every, clips, ds)
    except ValueError as e:
        print(input_missing(f"[HAND_CLEAR] ✗ {ds} hand={hand}: {e}"), file=sys.stderr)
        return 2
    trees = every if scope is None else scope
    if scope is not None:  # a scoped run keeps the unlisted trees' lines: each must hold a current verdict
        rec, ver, listed = _read_ledger(ledger_path(ds, hand, write)), rule_version(ds), set(scope)
        stale = [Path(t).name for t in every if t not in listed and not _current(rec.get(Path(t).name), Path(t), ver)]
        if stale:
            why = f"{len(stale)} unlisted trees have no current verdict, e.g. {stale[:3]} — run without --clips-file"
            print(input_missing(f"[HAND_CLEAR] ✗ {ds} hand={hand}: {why}"), file=sys.stderr)
            return 2
    with (
        capacity.sized(MODULE, len(trees), jobs) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        res = list(ex.map(tree_job, trees, chunksize=2))
    if len({c for c, _r, _e in res}) != len(trees):
        raise SystemExit(f"[HAND_CLEAR] ✗ unique results {len({c for c, _r, _e in res})} != base trees {len(trees)}")
    errs = [(c, e) for c, _r, e in res if e]
    if errs:
        print(f"[HAND_CLEAR] ✗ {len(errs)} trees not measured — no verdict written. e.g. {errs[0]}", file=sys.stderr)
        return 2
    led = ledger_path(ds, hand, write)
    led.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for _c, rs, _e in res for r in rs]
    if scope is None:
        body = ["\t".join(LEDGER_COLS)] + ["\t".join(str(r[k]) for k in LEDGER_COLS) for r in rows]
    else:
        old = led.read_text(encoding="utf-8").splitlines() if led.is_file() else []
        body = merge_ledger(old, rows, [Path(t).name for t in every], {Path(t).name for t in trees})
    led.write_text("\n".join(body) + "\n", encoding="utf-8")
    drops = drop_verdicts(rows)
    if write:
        RR.retarget_excluded_record(ds, GENERATOR, drops, [c for c, _r, _e in res], hand=hand)
    by_status = Counter(r["status"] for r in rows)
    kept = {r["clip"] for r in rows if r["status"] == KEPT} - set(drops)
    deep = max((r["depth_m"] for r in rows if r["status"] == KEPT), default=0.0)
    print(
        f"[HAND_CLEAR] trees {len(trees)}{'' if scope is None else f' (scope of {len(every)})'} · "
        f"excluded {len(drops)} by_reason "
        f"{dict(sorted(Counter(w for w, _d in drops.values()).items()))} · "
        f"kept_as_recorded {len(kept)} (max depth {deep} m, measured only) · "
        f"rows {' '.join(f'{k}={by_status[k]}' for k in STATUSES)} · "
        f"dataset={ds} hand={hand} version={rule_version(ds)} · ledger {led}"
        + ("" if write else " (measure only — no exclusion line)"),
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="stage 1 step 2f — S-hand-clear (hand in still extras: measured only)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hand", required=True, help="hand id (no default)")
    ap.add_argument("--write", action="store_true", help="write the ledger and the hand-scoped exclusion lines")
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--clips-file", default="", help="measure only these base trees and merge them into the ledger")
    a = ap.parse_args()
    return run(a.dataset, a.hand, a.write, jobs=a.jobs, clips=a.clips_file)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.placement.repair.hand_clear_stage")
    sys.exit(main())
