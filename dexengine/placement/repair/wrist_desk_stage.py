#!/usr/bin/env python3
"""Stage 1 step 2e — the wrist housing <-> desk reference repair over one dataset's base trees of one hand. The rule,
its constants and the solver are wrist_desk's; this file is the driver: tree I/O, the stamp ``rules.WRIST_SLACK``, the
per-(clip, side) ledger and the step's column (``freshness``, data.stages 2e).

Position: after 2d (the MANO contact sidecar ``contact_all`` it reads — that sidecar's input fingerprint reads object
poses and MANO only, never the hand qpos, so this repair does not stale it) and before 3 (case, roles, quality, the
wrist cache and f0 all read the repaired hand).

Every run computes from the original (the wrist_shift backup ``pre_ws`` while the tree holds this step's output), so a
re-run with a changed rule replaces, never stacks. A tree that needs no repair any more is reverted through
wrist_shift.revert (the one undo path); a clean tree is never written except for its stamp.
A segment piece (a units-registry unit — data.segment.split's statement) is its parent's window, so it is never solved
on its own (smoothing, ramps and the all-or-nothing verdict do not commute with cutting): this step solves the parents,
then the piece owner re-cuts every piece that is no longer its parent's window (split.reslice_stale, ``units_replaced``
-> re-judge) and the piece takes its parent's stamp. Its 2e cell is fresh when the parent's is and the two stamps are
equal.
The arm is the caller's (env DEXENGINE_ARM, datasets.active_arm — no default); the scene is coords.scene_mode (env
DEXENGINE_SCENE from the f0 axis); an aerial tree has no desk -> NA.
Usage: DEXENGINE_ARM=rby1 DEXENGINE_SCENE=<scene> <retarget py> -m dexengine.placement.repair.wrist_desk_stage
       --dataset <ds> --hand <hand> [--write | --need-only] [--clips <file>] [--ledger <tsv>]
Modes: --write solves, writes the trees and the ledger ``ledger_<ds>.tsv`` (the write run's own record); no flag
measures (solves, writes no tree) and --need-only only counts the repair targets — these two write no tree; their
ledger is ``ledger_<ds>.measure.tsv`` / ``.need.tsv`` (paths.mode_record), a --clips run's ``.clips.tsv``
(paths.clips_record). --ledger names the exact file instead. --write with --need-only dies.
rc 0 · 2 (a tree could not be processed, or a piece has no parent tree — nothing is stamped for it, the column stays
short) · 3 (no base tree).

Worker memory: a pool worker compiles one MuJoCo scene per task. Two things grow such a worker without a leak: MuJoCo's
process-global asset cache (it keeps decoded meshes and textures up to its capacity, 500 MB by default — one tree
fills 0.2–0.3 GB of it) and the glibc heap the scene compile frees but keeps. Over distinct trees the cache does not
pay for itself (each tree is built once per run), so ``worker_task`` turns it off and hands the freed heap back after
each task; the pool's end marker (capacity, peak_job) is the run's evidence line. Neither changes a result.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from dexengine.core import capacity
from dexengine.core.errors import input_missing
from dexengine.placement.repair import base_write as BW
from dexengine.placement.repair import wrist_desk as WDS
from dexengine.placement.repair import wrist_shift as WS
from dexengine.placement.repair.wrist_desk_fresh import (
    contact_file,
    follow_pieces,
    fresh_record,
    input_fp,
    pieces,
    qpos_sha,
    read_record,
    stage_trees,
)

LEDGER_COLS = (
    "clip",
    "side",
    "status",
    "reason",
    "need_frames",
    "depth_max_mm",
    "angle_max_deg",
    "lift_max_mm",
    "resid_max_mm",
    "objinc_max_mm",
    "nontip_max_mm",
    "jump_max_mm",
) + WDS.ROOT_KEYS
STEP = "dexengine.placement.repair.wrist_desk_stage"  # capacity.NEEDS key (this module: pool size and memory)


@contextlib.contextmanager
def worker_task():
    """One pool task under the worker memory policy: MuJoCo asset cache capacity 0 (read back — a cache that stays on
    dies here), then the freed heap handed back after the task, whatever it raised (capacity.task)."""
    import mujoco as mj  # noqa: PLC0415

    cache = mj.mj_getCache()
    mj.mj_setCacheCapacity(cache, 0)
    if mj.mj_getCacheCapacity(cache) != 0:
        raise RuntimeError("[WORKER_MEM] ✗ the MuJoCo asset cache capacity did not become 0")
    with capacity.task():
        yield


def freshness(ds: str, spec_dir: str) -> tuple[int, int]:
    """(trees whose 2e stamp is current, all base trees) — the data.stages 2e column. A parent: its stamp matches the
    current inputs and qpos. A piece: its parent's is current and the piece carries the same stamp."""
    from dexengine.core import paths as P  # noqa: PLC0415

    # the arm is compared only when given ("" otherwise)
    hand, arm_env = P.hand_id(spec_dir), os.environ.get("DEXENGINE_ARM", "")
    trees, pcs = stage_trees(ds, spec_dir), pieces(ds, spec_dir)
    ok = {d: fresh_record(ds, d, hand, arm_env) for d in trees if d not in pcs}
    fresh = sum(r is not None for r in ok.values())
    for d in (t for t in trees if t in pcs):
        try:
            fresh += ok.get(pcs[d]) is not None and read_record(d) == ok[pcs[d]]
        except (OSError, ValueError, AttributeError):  # noqa: PERF203 — an unreadable record is not fresh
            continue
    return fresh, len(trees)


def _contact(ds, tree, ti, q, side, spec_dir, tip_names):
    """MANO contact per tip (T, 5) from the 2d sidecar (any object), freshness-checked against this base."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.contact import fingerprint as SF  # noqa: PLC0415

    f = contact_file(ds, ti, tree.name, side, spec_dir)
    if f is None:
        return np.zeros((len(q), len(tip_names)), bool)
    SF.check(f, SF.TreeInputs(P.main_root(), ds, tree.name, ti, q, spec_dir).expected(SF.LAYER_ALL, side), ds=ds)
    with np.load(f, allow_pickle=True) as z:
        c, fing = np.asarray(z["contact"]).any(1), [str(x) for x in z["fingers"]]
    if c.shape[0] != len(q):
        raise ValueError(f"{tree.name}/{side}: contact sidecar length {c.shape[0]} != T {len(q)}")
    return c[:, [fing.index(n) for n in tip_names]]


def undo(tree: Path) -> bool:
    """This step's output on ``tree`` back to its input (base_write.undo_tree, the one undo path). False when the tree
    holds no output of this step. The stamp is the caller's."""
    return BW.undo_tree(tree, WDS.TAG)


def frame_dt(tree: Path) -> float:
    return BW.frame_dt(tree, WDS.TAG)


def tree_job(args) -> tuple:
    """One base tree -> (clip, ledger rows, changed, error), under the worker memory policy (worker_task)."""
    tree, ds, hand, arm, write, need_only = args
    tree = Path(tree)
    try:
        with worker_task():
            return tree.name, *_tree(tree, ds, hand, arm, write, need_only), ""
    except Exception as e:  # noqa: BLE001 — reported and counted; the tree keeps no stamp (column stays short)
        return tree.name, [], False, f"{type(e).__name__}: {e}"[:400]


def load(tree: Path) -> tuple:
    """A base tree as this step reads it → (task_info, original qpos (T, n), qpos layout, MjModel, MjData)."""
    import mujoco as mj  # noqa: PLC0415

    from dexengine.core import qpos_layout as QL  # noqa: PLC0415

    ti = json.loads((tree / "task_info.json").read_text())
    with np.load(WS.original(tree / "0/trajectory_kinematic.npz", WDS.TAG)) as z:
        q0 = np.asarray(z["qpos"], float)
    m = mj.MjModel.from_xml_path(str(tree / "scene.xml"))
    return ti, q0, QL.read(str(tree)), m, mj.MjData(m)


def make_side(tree: Path, loaded: tuple, ds: str, hand: str, arm: str, side: str, **kw) -> WDS.Side:
    """One side's solver on a loaded tree — the one place it is assembled. ``kw`` -> WDS.Side (cap_deg ·
    lift_max_mm)."""
    import mujoco as mj  # noqa: PLC0415

    from dexengine.core import hands as HS  # noqa: PLC0415
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.placement import arm_ik as AIP  # noqa: PLC0415

    ti, q0, lay, m, d = loaded
    H = WDS.Hand(m, lay, ti, side, hand)
    desk_xy, z_off = WDS.desk_of(tree)
    A = WDS.Arm(arm, hand, side, desk_xy)
    spec_h = HS.HAND_REGISTRY[hand]
    palm = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, spec_h.palm_fk_body_right.replace("right_", f"{side}_", 1))
    if palm < 0:
        raise ValueError(f"{side}: palm FK body {spec_h.palm_fk_body_right} is not in the scene")
    return WDS.Side(
        m,
        d,
        q0,
        H,
        A,
        z_off,
        palm,
        AIP.palm_frame_offset(hand),
        _contact(ds, tree, ti, q0, side, P.hand_spec_dir(hand), H.tip_names),
        frame_dt(tree),
        **kw,
    )


def _tree(tree: Path, ds: str, hand: str, arm: str, write: bool, need_only: bool):
    from dexengine.data.tree_io import write_json_if_changed  # noqa: PLC0415 — base task_info writer
    from dexengine.placement.coords import core as VC  # noqa: PLC0415

    loaded = ti, q0, lay, _m, _d = load(tree)
    rows, new, vel = [], q0.copy(), []
    if VC.scene_mode(tree) == "aerial":
        rows = [
            {"clip": tree.name, "side": s, "status": "NA", "reason": "no desk (aerial scene)"} for s in lay.hands
        ]  # data
    else:
        for side in lay.hands:
            S = make_side(tree, loaded, ds, hand, arm, side)
            H = S.H
            r = {
                "clip": tree.name,
                "side": side,
                "need_frames": int(S.need.sum()),
                "depth_max_mm": round(float(S.depth.max(initial=-1e9)) * 1000, 2),
            }
            if need_only:
                rows.append({**r, "status": "NEED" if S.need.any() else "CLEAN", "reason": S.defect()})
                continue
            status, why, sol, st = S.solve()
            if sol and S.verify_rows(sol) > -(WDS.WRIST_DESK_CLEAR_MM - 0.05) / 1000:
                status, why, sol = (
                    "UNRESOLVED",
                    "housing clearance short on the independent re-measure",
                    None,
                )  # ledger reason (data)
            for t, row in (sol or {}).items():
                c0, n = H.cols
                new[t, c0 : c0 + n] = row[c0 : c0 + n]
            if sol:
                vel.append((H.cols[0], H.cols[0], H.cols[1]))
            rows.append(
                {
                    **r,
                    "status": status,
                    "reason": why,
                    **S.root,
                    **{f"{k}_max_{'deg' if k == 'angle' else 'mm'}": v for k, v in st.items()},
                }
            )
    changed = bool(vel)
    if write and not need_only:
        if changed:
            BW.write_tree(tree, q0, new, vel, WDS.TAG)
        else:
            undo(tree)  # this step's old output, no repair now
        ti.setdefault("rules", {})[WDS.RULE_KEY] = {
            "version": WDS.CODE_VERSION,
            "arm": arm,
            "fp": input_fp(ds, tree, ti, arm, hand),
            "qpos_sha": qpos_sha(tree, WDS.TAG),
            "sides": {
                r["side"]: {k: v for k, v in r.items() if k not in ("clip", "side", *WDS.ROOT_KEYS)} for r in rows
            },
        }
        write_json_if_changed(tree / "task_info.json", ti)
    return rows, changed


def ledger_path(ds: str, hand: str, write: bool, need_only: bool, ledger: str = "", scoped: bool = False) -> Path:
    """The run's ledger: --ledger exactly when given; else the hand's run log ``ledger_<ds>.tsv`` for the write run
    and, for a run that writes no tree, the same name tagged with its mode (paths.mode_record — ``.need`` ·
    ``.measure``): the write run's ledger is its own record and a measurement never replaces it."""
    from dexengine.core import paths as P  # noqa: PLC0415

    if write and need_only:
        raise SystemExit("[WRIST_SLACK] ✗ --write and --need-only given together — need-only neither solves nor writes")
    if ledger:
        return Path(ledger)
    led = P.clips_record(P.run_path("run_log", hand, tool="wrist_slack") / f"ledger_{ds}.tsv", scoped)
    return P.mode_record(led, None if write else "need" if need_only else "measure")


def run(
    ds: str, hand: str, write: bool, need_only: bool, clips: str = "", ledger: str = "", *, jobs: int | None = None
) -> int:
    """The step over one dataset of one hand (the CLI is this) -> rc."""
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data import rules as RR  # noqa: PLC0415

    led = ledger_path(ds, hand, write, need_only, ledger, bool(clips))  # bad mode dies first
    arm, spec = RR.active_arm(), P.hand_spec_dir(hand)
    trees, pcs = stage_trees(ds, spec), pieces(ds, spec)
    if clips:  # a listed piece stands for its parent (a piece is never solved on its own)
        want = set(P.read_clip_list(clips, dataset=ds, quiet=True))
        named = {t.name: t for t in trees}
        if len(named) != len(trees):
            print(
                input_missing(
                    f"[WRIST_SLACK] ✗ {len(trees) - len(named)} base trees share a name — the list cannot "
                    "be resolved to trees"
                ),
                file=sys.stderr,
            )
            return 2
        if want - set(named):
            print(
                f"[WRIST_SLACK] ✗ {len(want - set(named))} of the {len(want)} listed clips are not in the base layer",
                file=sys.stderr,
            )
            return 2
        sel = {pcs.get(named[n], named[n]) for n in want}
        trees, pcs = [t for t in trees if t in sel], {d: par for d, par in pcs.items() if par in sel}
    parents = [t for t in trees if t not in pcs]
    if not trees:
        print(f"[WRIST_SLACK] ✗ {ds} hand={hand}: 0 base trees", file=sys.stderr)
        return 3
    with (
        capacity.sized(STEP, len(parents), jobs) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        res = list(ex.map(tree_job, [(str(t), ds, hand, arm, write, need_only) for t in parents], chunksize=2))
    print(WDS.root_mark(r for x in res for r in x[1]))
    names = [r[0] for r in res]
    if len(set(names)) != len(parents):
        raise SystemExit(f"[WRIST_SLACK] ✗ unique results {len(set(names))} != parent trees {len(parents)}")
    led.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(LEDGER_COLS)] + [
        "\t".join(str(r.get(k, "")) for k in LEDGER_COLS) for _n, rs, _c, _e in res for r in rs
    ]
    led.write_text("\n".join(lines) + "\n", encoding="utf-8")
    errs = [(n, e) for n, _r, _c, e in res if e]
    piece_note = f" · pieces {len(pcs)} (not solved — measured only)"
    if write and not need_only:
        bad = {n for n, _e in errs}
        recut, stamped, missed = follow_pieces(ds, hand, pcs, {t: t.name not in bad for t in parents})
        errs += missed
        piece_note = f" · pieces {len(pcs)}: re-cut {recut} · stamped {stamped}/{len(pcs)}"
    cnt = {s: sum(r["status"] == s for _n, rs, _c, _e in res for r in rs) for s in WDS.STATUSES + ("NEED",)}
    print(
        f"[WRIST_SLACK] on clips_changed={sum(c for *_x, c, _e in res)}/{len(parents)} dataset={ds} "
        f"hand={hand} arm={arm} scene={os.environ.get('DEXENGINE_SCENE', 'table')} version={WDS.CODE_VERSION} "
        f"cap={WDS.WRIST_TILT_MAX_DEG:g}° · "
        + " ".join(f"{k}={v}" for k, v in cnt.items() if v)
        + piece_note
        + f" · errors {len(errs)} · ledger {led}"
        + ("" if write else " (count only — no tree written)" if need_only else " (measure only — no tree written)"),
        flush=True,
    )
    if errs:
        print(
            input_missing(
                f"[WRIST_SLACK] ✗ {len(errs)} trees could not be processed — not stamped (the 2e column "
                f"stays short). e.g. {errs[0][0]}: {errs[0][1]}"
            ),
            file=sys.stderr,
        )
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage 1 step 2e — wrist housing <-> desk reference repair (wrist tilt + finger re-solve)"
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hand", required=True, help="hand id (no default)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="solve and write the trees and the ledger ledger_<ds>.tsv (without it: measure only — no "
        "tree written, ledger ledger_<ds>.measure.tsv)",
    )
    mode.add_argument(
        "--need-only",
        action="store_true",
        help="only count the repair targets (no solve · no tree written · ledger ledger_<ds>.need.tsv)",
    )
    ap.add_argument(
        "--clips",
        default="",
        help="clip-name list file (one per line; a piece name stands for its parent) — default: the whole base layer",
    )
    ap.add_argument(
        "--ledger",
        default="",
        help="ledger TSV — used exactly as given (no mode suffix). Default: the mode-specific name in the "
        "hand's run-log folder",
    )
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    a = ap.parse_args()
    return run(a.dataset, a.hand, a.write, a.need_only, a.clips, a.ledger, jobs=a.jobs)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.placement.repair.wrist_desk_stage")
    sys.exit(main())
