#!/usr/bin/env python3
"""Frame-0 hand pose repair: remove hand–[desk+object] penetration at reset.

WHY. Two reset failure modes break otherwise-learnable clips:
  1. Both arms start embedded in the desk: the contact termination fires on almost every
     reset, the PhysX solver goes unstable and completion stays at 0 % on both sides.
  2. The MANO frame-0 hand already grasps the object, and
     retargeting jitter leaves finger meshes INSIDE the object hull — on reset the
     object is ejected ("launched") before learning can start.
Both are frame-0 geometry defects, not learning failures. The fix is a minimal-motion
correction of the hand pose at frame 0 under a hard no-penetration constraint, then a
short soft-start blend back onto the original trajectory.

FORMULATION (Nelder-Mead):
  Find Δq (nd = 6 + n_indep: wrist pos 3 + wrist rot 3 + the independent finger joints) minimising
      f(Δq) = ||W ⊙ Δq||² + λ · n_pen(Δq) + λ_d · d_pen(Δq),   λ = LAMBDA = 1000, λ_d = DEPTH_W = 5e4
  where n_pen counts (a) hand hull vertices below the desk plane or inside the object
  hull and (b) object surface points inside any hand link hull. HARD constraint: the
  accepted solution must reach n_pen = 0, else this script FAILS loudly (no silent
  partial repair). Weights: thumb+index cheapest, other fingers next, wrist dearest —
  the weighted L2 does the "big finger move vs small wrist move" comparison natively.

FRAMES / DATA CONTRACT (all verified against this repo):
  - trajectory_kinematic.npz qpos (T, nq), addressed through qpos_layout (widths from HandSpec):
      per side a 6-column virtual wrist (slide x,y,z + hinge x,y,z; MuJoCo scene.xml axes are
             permuted+negated and the base body carries a rotation). Not hand-derived: the
             translation map cache_pos = A·qpos[0:3] + b takes A from the scene's root axes
             (repair_root.root_axes) and b from the rows; a residual >= WRIST_FIT_MAX_RES
             raises (repair_math.fit_wrist_qpos_map). The writers invert it for new frames
             and set the rotation columns through repair_root.turn_rows.
      then each side's finger columns (HandSpec order) and 7 columns (pos + quat wxyz)
      per object.
  - Wrist cache <wrist root>/wrist_poses_{side}/<seq>.npz  positions (T,3)
    quaternions (T,4) wxyz, trajectory frame. This IS what arm IK targets as
    {side}_hand_base_link (the arm IK stage), so hand-link world placement here
    uses the cache pose as the hand-base pose — consistent with what Isaac reproduces.
  - Desk: trajectory frame desk top = z 0 (objects rest at z≈0.006..0.05; runtime adds
    DESK_HEIGHT). Desk penetration = hull vertex z < desk_z - desk_margin (desk_z: capture-frame desk plane,
    coords.scene_z_offset — TACO staging 0, base layer task_info.support.desk_z).
  - Hand geometry: per-link convex hulls from the collision meshes of the combo's
    merged URDF (HandCfg.urdf — the same URDF the arm IK uses); FK via pinocchio, link
    placement expressed relative to the hand's palm frame so arm joints are irrelevant.
    (Widths and joint order come from HandSpec; a hand with coupled finger joints is
    refused by HandCfg.)

OUTPUT (only when a repair was needed; --check_only never writes):
  <tree root>/<seq><suffix>/                     (--tree_root, else HandCfg.data_root; default suffix _f0)
    0/trajectory_kinematic.npz   P+T frames: frame 0 is the repaired pose, frames
                                 0..P-1 interpolate to ORIGINAL FRAME 0, and frames
                                 P.. are the original trajectory unchanged; the other
                                 hand and the objects hold still at frame 0 in the pad.
                                 P rises from n_blend until the pad respects the finger
                                 cap HC.f_cap_rad_s and the wrist 1 cm/frame cap (at
                                 most 60 frames).  qvel: zeros for new frames.
                                 qpos_rollout: frame-0 repeats (kinematic-only rule).
    0/<sidecars>                 contact_mano_*.npz time-prepended with frame-0 rows;
                                 body_positions.npz copied as-is (superseded by the
                                 directly-written wrist cache; marker records this);
                                 task_info.json / scene.xml / annotations copied.
    F0_REPAIR.json               Δq, n_pen before, per-stage timings, parameters.
  <wrist root>/wrist_poses_{right,left}/<seq><suffix>.npz   blended/held caches (--wrist_root | HandCfg.wrist_root).
  precompute/taco/placement/<seq><suffix>.adopted.json         copied (z_off rides on).
  Arm IK for the new sequence is NOT run here — run the arm IK stage on
  <seq><suffix> afterwards (both sides); the canonical chain is scripts/stage2_ground.sh.

SOLVER: the only engine is the GPU one — `f0/gpu.py` batches this module's objective and
Nelder-Mead across clips (`SolveRequest`), and the only entry that repairs is
`dexengine/retarget/f0/prejudge.py`. This file owns the control flow (generators), the geometry and
the host-side checks. Before the sides, the frame-0 separation (repair_sep, rule 1d) moves a main object out of a
collision-on body; a tree it drops or holds is held (HOLD_OBJ), and its rows are written on every other exit.

Usage:
  # repair = the f0 shadow driver (GPU engine; device from --gpus / DEXENGINE_F0_GPUS)
  $RETARGET_PY dexengine/retarget/f0/prejudge.py --hand <hand> --dataset <ds> --emb <emb> --clips <list>
  # fast screen (no writes) — the only thing a direct run of this file does
  python dexengine/retarget/f0/repair.py --dataset taco --sequence <seq> --check_only
"""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.core import stdio
from dexengine.core.repo import REPO_ROOT as REPO  # noqa: E402, F401  (gpu.py prints it in its worker marker)

# Canonical owner of names and suffixes = dexengine/core/paths.py (no literals)
from dexengine.data import rules as _RR  # noqa: E402  — active hand owner
from dexengine.retarget.f0 import phrases as PH  # noqa: E402  — outcome tokens shadow_tree.finish reads
from dexengine.retarget.f0 import repair_obj as RO  # noqa: E402  — the object repair's switch and policy
from dexengine.retarget.f0 import repair_sep as RSEP  # noqa: E402  — the frame-0 separation (rule 1d)
from dexengine.retarget.f0.repair_args import _build_parser
from dexengine.retarget.f0.repair_cfg import (  # noqa: F401  (HC: main_steps rebinds it)
    HC,
    SIDE_BUDGET_EVALS,
    HandCfg,
    share_hand_cfg,
)
from dexengine.retarget.f0.repair_obj_write import write_obj_window
from dexengine.retarget.f0.repair_side import _side_steps
from dexengine.retarget.f0.repair_write import write_in_place, write_outputs


def _drive_check_only(gen):
    """Run the `main_steps` generator to the end without solving: the driver of the check-only CLI (`--check_only`).
    Any solve request is fatal: the only solver is the GPU engine, entered through `prejudge.py`."""
    try:
        next(gen)
    except StopIteration as e:
        return e.value
    gen.close()
    raise SystemExit(
        "[frame0_repair] a solve request was raised — running this file directly is check-only. "
        "Repairs run only through prejudge.py (GPU engine, gpu workers)"
    )


def obj_acted_part(clip: str, tree_root: Path, wrist_root: Path, dataset: str, sides, hand: str | None = None) -> str:
    """The f0 object repair's part of the f0 cache key (shadow_key.f0_input_key): "" when the rule's gate opens on no
    side of this clip (switch off · gate keeps · not measured · a desk repair), so those keys stay byte-identical to the
    keys before the rule; else one line per opened side — the policy (repair_obj.policy_line) and the cooked hull
    digest, so f0 re-solves the clip when either changes. The downstream keys read the f0 output instead (the rows'
    hash, judge input_clip._f0_obj): a re-solve that writes the same rows, or none, moves nothing downstream. The
    decision is the repair's own gate (check_and_repair_steps without solving; its lines silenced). `hand` (when
    given) must be the module's HC — the keys of another hand would be measured on the wrong geometry."""
    if hand is not None and hand != HC.name:
        raise RuntimeError(f"[F0_OBJ] key part for hand {hand!r} measured with HC {HC.name!r} — set DEXENGINE_HAND")
    if not RO.obj_repair_on():
        return ""
    lines = []
    sep = RSEP.key_line(RSEP.plan(Path(tree_root) / clip))  # "" without a pair: the key stays as it was
    for side in sides:
        if not (Path(wrist_root) / f"wrist_poses_{side}" / f"{clip}.npz").is_file():
            continue
        with stdio.capture():  # per thread (core.stdio): this runs in the prejudge's prep threads
            rep = _drive_check_only(
                check_and_repair_steps(clip, side, Path(tree_root), Path(wrist_root), 0.0, False, dataset=dataset)
            )
        if rep.get("obj_repair"):
            pol = RO.policy_line(bool(rep.get("obj_slots")), bool(rep.get("obj_stills")))
            lines.append(f"obj_repair/{side}|{pol}|{rep.get('obj_cooked_digest')}")
    return "\n".join(lines + ([sep] if sep else []))


def check_and_repair_steps(
    seq: str,
    side: str,
    data_root: Path,
    wrist_root: Path,
    desk_margin: float,
    do_repair: bool,
    seed: int = 0,
    pen_margin: float = 0.0,
    clearance: float = 0.0,
    verify_retries: int = 3,
    side_budget_evals: int = SIDE_BUDGET_EVALS,
    dataset: str = "taco",
):
    """Check and repair one side and return the report dict (the caller writes). A generator: it yields solve requests
    (`SolveRequest`) and receives (x, c). The GPU engine (`gpu`) solves many sides together.

    This wrapper prints the budget marker: every side emits exactly one line
    `[F0_BUDGET] side=… evals=…/… status=OK|EXHAUSTED`, and the driver (`shadow_tree.finish`) checks that the line count
    equals the side count; without the marker the run is invalid."""
    ctr = [0, None]  # [cumulative n_pen calls, cumulative count at optimisation start]: the budget meter
    rep = yield from _side_steps(
        seq,
        side,
        data_root,
        wrist_root,
        desk_margin,
        do_repair,
        seed=seed,
        pen_margin=pen_margin,
        clearance=clearance,
        verify_retries=verify_retries,
        side_budget_evals=side_budget_evals,
        dataset=dataset,
        _ctr=ctr,
    )
    used = 0 if ctr[1] is None else ctr[0] - ctr[1]
    rep["n_evals"] = int(used)
    rep["budget_evals"] = int(side_budget_evals)
    print(
        f"[F0_BUDGET] side={side} evals={used}/{int(side_budget_evals)} "
        f"status={'EXHAUSTED' if rep.get('budget_exhausted') else 'OK'}",
        flush=True,
    )
    return rep


def pathlib_read(path):
    return [ln.rstrip("\n") for ln in open(path)]


def _write_sequence(args, checked_sides, data_root, fails, repaired, seq, sides, wrist_root, moved=None):
    """Write what the checks of one sequence produced: the in-place repair, a clean no-repair ledger, or the side
    outputs. `moved` (repair_sep.moved) = the frame-0 separation's rows, written on every exit (the sides measured
    them)."""
    if (repaired or moved) and not args.check_only:
        obj = {s: r for s, r in repaired.items() if "obj_window" in r}
        if (obj or moved) and not args.in_place:
            raise SystemExit(f"[F0_OBJ] {seq}: the object repair writes in place only (--in_place)")
        if args.in_place:
            if obj or moved:  # first: a desk repair of the other side, written after, prepends its pad to these rows
                write_obj_window(seq, obj, data_root, wrist_root, moved)
                what = sorted(obj) + [f"{RSEP.MARK} {n}" for n in sorted(moved or {})]
                print(f"  -> object window written in place ({', '.join(what)}; length unchanged)")
            desk = {s: r for s, r in repaired.items() if s not in obj}
            mode, err = write_in_place(seq, desk, data_root, wrist_root, args.n_blend) if desk else ("window", None)
            if err:
                fails.append((seq, "write", err))
                print(PH.inplace_failed_line(err))
            else:
                print(f"  -> repaired in place (mode={mode}). next: the bake's arm IK stage (coarse_path.dp) bakes it")
        else:  # a desk repair without --in_place: the suffixed tree (the object rows raised above)
            dst = write_outputs(seq, repaired, data_root, wrist_root, args.n_blend, args.out_suffix)
            print(
                f"  -> {dst.name} written (blend {args.n_blend} frames). "
                f"next: arm IK of {dst.name} (both sides, scripts/stage2_ground.sh)"
            )
    elif (
        args.in_place
        and not args.check_only
        and not repaired
        and checked_sides == sides
        and not any(f[0] == seq for f in fails)
    ):
        # Even when clean (0 penetration, no repair needed) a ledger is left: if
        # "no ledger" meant both "it never ran" and "it was clean so
        # nothing was done", count-based completion would be impossible. n_blend_effective=0, so
        # every consumer (tpad, the re-repair rewrite decision) correctly reads it as no pad.
        fj_clean = data_root / seq / "F0_REPAIR.json"
        if not fj_clean.exists():
            with open(fj_clean, "w") as fh:
                json.dump(
                    {
                        "source": seq,
                        "design": "clean_v6_noop",
                        "n_blend_effective": 0,
                        "pad_new": 0,
                        "pad_baseline": 0,
                        "dt": 0.02,
                        "sides_checked": sides,
                    },
                    fh,
                    indent=1,
                )
            print(f"  -> clean — no-repair ledger written ({fj_clean.name})")
        else:
            print(f"  -> clean — ledger exists ({fj_clean.name}), nothing written")


def _print_side(seq, side, r, args, tsv):
    """The console line of one checked side and its row in the --report_tsv table."""
    tag = (
        f"n_pen={r['n_pen_before']} (desk {r.get('n_desk', '-')}/object {r.get('n_obj', '-')}"
        f"/reverse {r.get('n_rev', '-')} · max depth {r.get('desk_depth_max_mm', '-')}mm)"
        + (
            f" -> {r.get('status', '')}"
            f" (wrist {r.get('wrist_shift_cm', '-')}cm/{r.get('wrist_rot_deg', '-')}deg,"
            f" finger max {r.get('max_finger_rad', '-')}rad,"
            f" opt {r.get('t_opt_s', '-')}s)"
            if r["n_pen_before"]
            else " clean"
        )
    )
    if not (args.all and r["n_pen_before"] == 0):
        print(f"{seq} [{side}] {tag}  (check {r['t_check_s']}s / total {r['t_total_s']}s)")
    if tsv:
        tsv.write(f"{seq}\t{side}\t{r['n_obj']}\t{r['n_rev']}\t{r['n_desk']}\t{r['t_total_s']}\n")


def sequence_steps(seq: str, args, data_root: Path, wrist_root: Path, sides: list, fails: list, st: dict, tsv=None):
    """Check, repair and record one sequence (the body of the clip loop).
    A generator: it passes on the solve requests of `check_and_repair_steps` for each side. The GPU engine (`gpu`, the
    only engine) runs many sequences together."""
    if (
        args.in_place
        and not args.force
        and not args.check_only
        and (data_root / seq / "F0_REPAIR_INPLACE.json").exists()
    ):
        print(PH.already_repaired_line(seq))
        return
    sep = RSEP.plan(data_root / seq)
    print(RSEP.marker(seq, sep), flush=True)
    held = RSEP.held_status(sep)  # a dropped or held tree: its sides are measured, nothing is solved or written
    repaired = {}
    checked_sides = []
    for side in sides:
        if not (wrist_root / f"wrist_poses_{side}" / f"{seq}.npz").exists():
            if args.all:
                st["n_skip"] += 1
            else:
                print(f"{seq} [{side}] wrist cache missing — skipped")
            continue
        checked_sides.append(side)
        r = yield from check_and_repair_steps(
            seq,
            side,
            data_root,
            wrist_root,
            args.desk_margin,
            do_repair=not args.check_only and not held,
            pen_margin=args.pen_margin,
            clearance=args.clearance,
            verify_retries=args.verify_retries,
            side_budget_evals=args.side_budget_evals,
            dataset=args.dataset,
        )
        _print_side(seq, side, r, args, tsv)
        if r.get("status") == "REPAIRED":
            repaired[side] = r
        elif r.get("status", "").startswith(("FAIL", "SKIP")):
            # Batch (in_place) mode records and continues — a failed sequence keeps its hold pad.
            fails.append((seq, side, r["status"]))
            if not args.in_place:
                raise SystemExit(f"{seq} [{side}] {r['status']}")
    if held:
        _hold_line(seq, sep, held, checked_sides, fails, args)
        return
    _write_sequence(args, checked_sides, data_root, fails, repaired, seq, sides, wrist_root, RSEP.moved(sep))


def _hold_line(seq, sep, held, checked_sides, fails, args):
    """The line of a tree the frame-0 separation drops or holds: `<seq> [<side>] [F0_SEP] -> SKIP_OBJ: 1d …`, the
    form shadow_tree.held_row reads (HOLD_OBJ: not judged, the reason in the ledger)."""
    side = RSEP.held_side(sep, checked_sides)
    print(f"{seq} [{side}] {RSEP.MARK} -> {held}", flush=True)
    fails.append((seq, side, held))
    if not args.in_place and not args.check_only:
        raise SystemExit(f"{seq} [{side}] {held}")


def _sequences(args, data_root):
    """The sequences of this run: --seqs_file, every tree with a trajectory (--all), or --sequence."""
    if args.seqs_file:
        seqs = [ln for ln in pathlib_read(args.seqs_file) if ln]
    elif args.all:
        seqs = sorted(p.name for p in data_root.iterdir() if (p / "0/trajectory_kinematic.npz").exists())
    else:
        assert args.sequence, "one of --sequence / --seqs_file / --all"
        seqs = [args.sequence]
    return seqs


def main_steps(argv=None):
    """Generator for one clip's f0 (argument parsing, output text, ledger) that yields solve requests. The GPU worker
    (`gpu --worker`) builds one per clip from the argv given by the prejudge and runs them together. The status
    classification in `shadow_tree.finish` reads this output text (its only input)."""
    ap = _build_parser()
    args = ap.parse_args(argv)
    args.arm = args.arm or _RR.active_arm()

    global HC
    HC = HandCfg(args.hand, args.arm)
    share_hand_cfg(HC)  # the reader modules (repair_cfg.HC_READERS) see the same hand
    data_root, wrist_root = HC.data_root(args.dataset), HC.wrist_root(args.dataset)
    # Tree root override: a tree is not per hand but **per combo**
    # (placement depends on the arm too), so a root derived from the hand name is wrong.
    # When an override arrives, the
    # hand-derived root is dropped.
    if args.tree_root:
        data_root = Path(args.tree_root)
    if args.wrist_root:
        wrist_root = Path(args.wrist_root)
    print(f"[frame0_repair] hand={HC.name} nd={HC.nd} H={HC.H}  tree={data_root}")

    seqs = _sequences(args, data_root)

    sides = ["right", "left"] if args.side == "both" else [args.side]
    fails = []
    tsv = None
    if args.report_tsv:
        # buffering=1 (line buffered): in this file one line is one sequence's result. With the
        # default block buffering, when the process dies to a timeout/kill **the last lines
        # disappear wholesale** (a lost tail turns successful clips into counted failures). A progress
        # file is committed line by line.
        tsv = open(args.report_tsv, "w", buffering=1)  # noqa: SIM115 — written by every sequence step
        tsv.write("sequence\tside\tn_obj\tn_rev\tn_desk_advisory\tt_total_s\n")
    st = {"n_skip": 0}
    for seq in seqs:
        yield from sequence_steps(seq, args, data_root, wrist_root, sides, fails, st, tsv)
    n_skip = st["n_skip"]
    if tsv:
        tsv.close()
        print(f"[report] {args.report_tsv} written; {n_skip} skipped without a wrist cache")
    if fails:
        print(PH.unrepaired_line(len(fails)))
        for sq, sd, st in fails:
            print(f"    {sq} [{sd}] {st}")


def main(argv=None):
    """Direct run = check only (`--check_only`, no writes). Repairs enter only through `prejudge.py` (GPU engine);
    without `--check_only` this raises before any solve."""
    import sys as _s

    args = list(_s.argv[1:] if argv is None else argv)
    # help solves nothing; blocking it would hide this entry from checks that import and inspect the parser via `--help`
    if "--check_only" not in args and not {"-h", "--help"} & set(args):
        raise SystemExit(
            "[frame0_repair] a direct run supports only --check_only (check only); repairs "
            "run only through prejudge.py (GPU engine)"
        )
    return _drive_check_only(main_steps(args))


if __name__ == "__main__":
    main()
