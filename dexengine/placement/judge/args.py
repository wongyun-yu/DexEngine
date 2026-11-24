"""The judge command line: every argument with its default, and the f0-first input roots.

Defaults of the search conditions come from the judge configuration (`judge.config`, `configs/placement/judge.yaml`);
`--config <yaml>` selects another file and single flags still override it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexengine.core.paths import SRC_LAYERS, data_layout, f0_root, hand_spec_dir, layer_path, pool_root
from dexengine.placement import arm_flip as _AF
from dexengine.placement.judge import aug_cells as _AC
from dexengine.placement.judge import config as _CFG
from dexengine.placement.reach_map import clearance as CLR

TRACK_ABLATE_PARTS = ("coarse", "graft", "ball", "stat")
TRACK_ABLATE_TAG = "[TRACK_ABLATE]"


def track_ablate_parts(a) -> frozenset:
    """`--track-ablate` → the set of parts switched off. Unknown / repeated names die; a marker line when on."""
    parts = [p for p in a.track_ablate.split(",") if p]
    bad = sorted(set(parts) - set(TRACK_ABLATE_PARTS))
    if bad or len(parts) != len(set(parts)):
        raise SystemExit(
            f"{TRACK_ABLATE_TAG} ✗ --track-ablate {a.track_ablate!r}: unknown or repeated part "
            f"{bad} (parts: {', '.join(TRACK_ABLATE_PARTS)})"
        )
    if parts:
        print(
            f"{TRACK_ABLATE_TAG} off={','.join(sorted(parts))} track_seeds={a.track_seeds} "
            f"track_reseeds={a.track_reseeds} track_iters={a.track_iters} (default '' = byte-identical)",
            flush=True,
        )
    return frozenset(parts)


def config_path(argv: list[str]) -> str:
    """The `--config` value in argv ('' = the default file), read before the full parser is built."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    return pre.parse_known_args(argv)[0].config


def _build_parser(cfg: _CFG.JudgeConfig | None = None) -> argparse.ArgumentParser:
    """Every judge argument; the order of the helpers below is the `--help` order."""
    cfg = cfg or _CFG.load(config_path(sys.argv[1:]))
    ap = argparse.ArgumentParser(description="Placement-search judge (paper Sec. 3.3).")
    ap.add_argument("--config", default="", help=f"judge configuration YAML (default {_CFG.CONFIG_FILE})")
    _args_grid(ap, cfg)
    _args_path(ap, cfg)
    _args_search(ap, cfg)
    _args_confirm(ap, cfg)
    _args_io(ap)
    return ap


def _args_grid(ap: argparse.ArgumentParser, c) -> None:
    """Combo, IK pool, placement grid (radius, yaw, dz), desk margins and anchors."""
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--clips", default="", help="clip list file (empty = every clip of the combo)")
    ap.add_argument("--pool-root", default=str(pool_root()), help="reachability map (IK pool) root")
    ap.add_argument("--stride", type=int, default=c.stride)
    ap.add_argument("--half-cells", type=int, default=c.half_cells)
    ap.add_argument(
        "--half-cells-x",
        type=int,
        default=c.half_cells_x,
        help="search radius along scene x (desk depth), cells of 3.5 cm",
    )
    ap.add_argument(
        "--half-cells-y",
        type=int,
        default=c.half_cells_y,
        help="search radius along scene y (desk width), cells of 3.5 cm",
    )
    ap.add_argument("--n-yaw", type=int, default=c.n_yaw)
    ap.add_argument("--yaw-lo", type=float, default=c.yaw_lo, help="yaw search lower bound (deg)")
    ap.add_argument("--yaw-hi", type=float, default=c.yaw_hi, help="yaw search upper bound (deg)")
    ap.add_argument(
        "--dz-cells",
        default=c.dz_cells,
        help="robot height candidates (cells of 3.5 cm, comma list or one value), searched by |dz| "
        "ascending with positive first; stage 2 passes the dataset policy (search_policy.py --cells)",
    )
    ap.add_argument("--hand-margin", type=float, default=c.hand_margin)
    ap.add_argument(
        "--arm-margin-cap",
        type=float,
        default=CLR.LOOSE_MARGIN_M,
        help="arm margin (m) of the loose desk-clearance map used to filter cells; must be looser "
        "than the judge's desk threshold",
    )
    ap.add_argument("--anchors", default=c.anchors)
    ap.add_argument("--max-checks", type=int, default=c.max_checks, help="confirm solves per anchor")


def _args_path(ap: argparse.ArgumentParser, c) -> None:
    """Costs and thresholds of the coarse path (DP) and of tracking."""
    ap.add_argument("--dp-stride", type=int, default=c.dp_stride)
    ap.add_argument("--dp-budget-mult", type=float, default=c.dp_budget_mult, help="DP cost threshold (budget factor)")
    ap.add_argument(
        "--dp-max-stretch",
        type=float,
        default=c.dp_max_stretch,
        help="largest total retiming stretch (how many times longer the clip becomes)",
    )
    ap.add_argument(
        "--dp-safety",
        type=float,
        default=c.dp_safety,
        help="retiming safety factor; must equal the bake's for the stretch prediction to hold",
    )
    ap.add_argument("--dp-atoms", type=int, default=c.dp_atoms, help="nearest orientation atoms looked up per frame")
    ap.add_argument(
        "--dp-travel",
        type=float,
        default=c.dp_travel,
        help="weight of the normalised total joint travel on a DP edge (mean |dq| / budget)",
    )
    ap.add_argument(
        "--dp-clr-w",
        type=float,
        default=c.dp_clr_w,
        help="weight of the desk-clearance deficit (m) of a DP node; steers only, the final penetration check decides",
    )
    ap.add_argument(
        "--dp-still-boost",
        type=float,
        default=c.dp_still_boost,
        help="travel cost factor on frames where the reference wrist stands still (<2 mm and <1 deg); 0 = off",
    )
    ap.add_argument(
        "--arm-flip",
        type=int,
        choices=(0, 1),
        default=_AF.JUDGE_GATE_DEFAULT,
        help="arm-flip gate on every confirmed path (arm_flip.fix, as in the bake): 1 = on, "
        "0 = off. The marker [ARM_FLIP_JUDGE] is printed either way",
    )
    ap.add_argument(
        "--jump-max", type=float, default=c.jump_max, help="largest joint jump between adjacent frames (rad, 0 = off)"
    )
    ap.add_argument(
        "--swivel-max-deg",
        type=float,
        default=c.swivel_max_deg,
        help="largest elbow swivel change inside a 0.5 s window (deg); a seed above it is rejected",
    )


def _args_search(ap: argparse.ArgumentParser, c) -> None:
    """Exhaustive search: limits, order, batching and the per-clip time budget."""
    ap.add_argument(
        "--center-scene",
        default="",
        help="search origin in scene coordinates 'x,y' (write '--center-scene=-0.33,-0.15' for negative "
        "values); empty = the dataset policy (search_policy.CENTER_SCENE)",
    )
    ap.add_argument(
        "--gate-batch",
        type=int,
        default=c.gate_batch,
        help="cells per batched exhaustive gate (0 = serial; same verdicts)",
    )
    ap.add_argument(
        "--desk-obj-frame0",
        type=int,
        default=c.desk_obj_frame0,
        help="1 = the desk constraint applies to the objects' first frame only; 0 = every frame",
    )
    ap.add_argument(
        "--order-nat",
        type=int,
        default=c.order_nat,
        help="1 = second sort key of the exhaustive order is the distance to the natural-posture wrist",
    )
    ap.add_argument(
        "--exhaust-cands",
        type=int,
        default=c.exhaust_cands,
        help="placement candidates collected by the exhaustive search (bake retry material)",
    )
    ap.add_argument(
        "--batch-cells",
        type=int,
        default=c.batch_cells,
        help="gate-passing cells confirmed together in one GPU batch (1 = serial)",
    )
    ap.add_argument(
        "--clip-budget-s",
        type=float,
        default=c.clip_budget_s,
        help="wall-clock budget of one clip's exhaustive search (s, 0 = none)",
    )


def _args_confirm(ap: argparse.ArgumentParser, c) -> None:
    """Confirm (FULL track): seeds, tracking, volume threshold, exhaustive switch."""
    ap.add_argument(
        "--track-seeds",
        type=int,
        default=c.track_seeds,
        help="seeds tracked in parallel by the confirm (each tracks one branch of the nullspace circle)",
    )
    ap.add_argument(
        "--track-reseeds", type=int, default=c.track_reseeds, help="largest number of multistart reseeds while tracking"
    )
    ap.add_argument("--track-iters", type=int, default=c.track_iters, help="refinement iterations per frame")
    ap.add_argument(
        "--random-start-mult",
        type=int,
        default=c.random_start_mult,
        help="random IK starts per placement = max(mult * track seeds, --random-start-min)",
    )
    ap.add_argument("--random-start-min", type=int, default=c.random_start_min)
    ap.add_argument(
        "--track-ablate",
        default="",
        help="paper ablation (arm-solver components, counts only): comma list of parts of the batched "
        "confirm (track_full_multi) to switch OFF — coarse = the gate's whole-clip path is not "
        "used (no frame-0 seed from it, no graft), graft = bad frames are not re-solved from the "
        "coarse path, ball = no ball-sample rescue around the previous frame, stat = switch "
        "nothing off, only count how often graft / ball ran. Default '' = byte-identical",
    )
    ap.add_argument(
        "--dp-err-mm", type=float, default=c.dp_err_mm, help="tracking threshold: stretch-weighted p99 wrist error (mm)"
    )
    ap.add_argument(
        "--dp-cov",
        type=float,
        default=c.dp_cov,
        help="smallest fraction of frames with a pose candidate for a DP path to exist",
    )
    ap.add_argument(
        "--vol-frac",
        type=float,
        default=c.vol_frac,
        help="screening coverage: fraction of frames with a pose in the map for a candidate cell",
    )
    ap.add_argument(
        "--exhaustive",
        action=argparse.BooleanOptionalAction,
        default=c.exhaustive,
        help="when every anchor path fails, search the whole volume (a failure is then a certificate "
        "within the grid and thresholds)",
    )
    ap.add_argument("--exhaust-max", type=int, default=c.exhaust_max, help="cells per clip of the exhaustive search")
    ap.add_argument(
        "--exhaust-resume",
        default="",
        help="D37 retry: resume table (judge.retry — clip · start · cands). Each clip's exhaustive search starts at "
        "the cell where the earlier judge stopped (start) and does not look at the cells before it again. A judged "
        "clip missing from the table, or a candidate count that differs from it, dies. Empty (default) = from cell 0",
    )
    ap.add_argument(
        "--n-candidates",
        type=int,
        default=c.n_candidates,
        help="placements collected per clip after the first success (the bake retries the next one)",
    )
    ap.add_argument(
        "--desk-shrink-robot",
        type=float,
        default=c.desk_shrink_robot,
        help="cut this much (m) off the desk's robot-side edge (scene -x); 0 = the full desk",
    )
    ap.add_argument(
        "--prune-radius-mm",
        type=float,
        default=c.prune_radius_mm,
        help="largest wrist displacement inside one cell (mm); a cell whose centre error exceeds the "
        "threshold by more than this is excluded (branch and bound)",
    )
    ap.add_argument("--seed", type=int, default=c.seed, help="base seed of the per-clip random streams")


def _args_outputs(ap):
    """Outputs, active sides and the f0 shadow switches."""
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--end-marks",
        default="",
        help="append `<clip>\\t<seconds>` to this file after each clip's judging ends (the only "
        "completion evidence stage 2 counts)",
    )
    ap.add_argument(
        "--active-sides",
        choices=("both", "auto"),
        default="both",
        help="both = judge both arms; auto = read task_info case.active per clip and judge only the "
        "active arm, the idle arm stands at its rest pose (idle_arm.idle_qpos)",
    )
    ap.add_argument(
        "--place-reselect",
        default="",
        metavar="<record>",
        help="re-selection record (placement.bake.place_check.RESELECT_FILE form): only its clips are "
        "judged with the robot spot disc widened by their margin (placement.judge.reselect · "
        "marker [PLACE_RESELECT]); every record clip must be in the judge list. Not given = no "
        "record (nothing changes)",
    )
    ap.add_argument(
        "--aug-cells-mode",
        choices=_AC.MODES,
        default="off",
        help="augmentation cells: D = dump the volume cells of the canonical dz layer per clip (no "
        "judging), L = judge every cell of a cell list in canonical order; off = normal judging",
    )


def _args_aug_place(ap):
    """Augmentation cells and the placement-strategy plugin options."""
    ap.add_argument("--aug-cells-dir", default="", help="bundle folder of the augmentation cells")
    ap.add_argument("--aug-cells-chunk", default="", help="mode L chunk name (judged/augcells_<combo>__<chunk>.jsonl)")
    ap.add_argument(
        "--place-strategy",
        default="canon",
        help="cell-list strategy; canon = the judge's own search. Any other name is provided by "
        "--place-strategy-module",
    )
    ap.add_argument(
        "--place-strategy-module",
        default="",
        help="importable module with STRATEGIES, check_args(a) and search(judge, ck); required by a "
        "non-canonical --place-strategy",
    )
    ap.add_argument("--place-strategy-cap", type=int, default=4096, help="plugin option: cell cap")
    ap.add_argument(
        "--place-strategy-budget-s",
        type=float,
        default=0.0,
        help="plugin option: per-clip wall budget (0 = --clip-budget-s)",
    )
    ap.add_argument(
        "--place-strategy-adopted-dir",
        default="",
        help="plugin option: folder of placement records whose cell is re-judged",
    )


def _args_io(ap: argparse.ArgumentParser) -> None:
    """Source axis, f0 shadow inputs, outputs, active sides, augmentation cells and the placement-strategy plugin."""
    ap.add_argument("--dataset", required=True, choices=sorted(data_layout.DATASETS))
    ap.add_argument("--emb", required=True, choices=list(data_layout.EMBODIMENTS))
    ap.add_argument(
        "--src-layer",
        default="base",
        choices=SRC_LAYERS,
        help="object tree layer when --traj-root is off (layer_path(src_layer, dataset, hand, emb))",
    )
    ap.add_argument(
        "--wrist-root",
        default="auto",
        help="wrist cache root (<root>/wrist_poses_{side}/): auto = the f0 shadow <F0>/<hand spec dir>/, "
        "off = the source layer order, else a path",
    )
    ap.add_argument(
        "--traj-root",
        default="auto",
        help="object tree root (<root>/<clip>/0/trajectory_kinematic.npz): auto = <F0>/<hand>/<emb>, "
        "off = the --src-layer layer, else a path; must be the same generation as the wrists",
    )
    ap.add_argument(
        "--save-ik", default="", help="save the full joint path of every adopted placement here (the bake reuses it)"
    )
    _args_outputs(ap)
    _args_aug_place(ap)


PLACE_STRATEGY_TAG = "[PLACE_STRATEGY]"


def place_strategy(a):
    """The plugin module of a non-canonical `--place-strategy` (None = canon, the judge's own search).
    Plugin options given without a plugin die, so they are never silently ignored."""
    if a.place_strategy == "canon":
        if (
            a.place_strategy_module
            or a.place_strategy_adopted_dir
            or a.place_strategy_cap != 4096
            or a.place_strategy_budget_s != 0.0
        ):
            raise SystemExit(
                f"{PLACE_STRATEGY_TAG} ✗ --place-strategy-module/-cap/-budget-s/-adopted-dir given "
                "without --place-strategy (canon ignores them)"
            )
        return None
    import importlib  # noqa: PLC0415 — the plugin is imported only for a non-canonical strategy

    name = a.place_strategy_module
    if not name:
        raise SystemExit(
            f"{PLACE_STRATEGY_TAG} ✗ --place-strategy {a.place_strategy!r} needs --place-strategy-module "
            "(the module that provides it)"
        )
    mod = importlib.import_module(name)
    if a.place_strategy not in mod.STRATEGIES:
        raise SystemExit(
            f"{PLACE_STRATEGY_TAG} ✗ --place-strategy {a.place_strategy!r} is not one of {mod.STRATEGIES} ({name})"
        )
    mod.check_args(a)
    return mod


def _resolve_f0_roots(a):
    """Resolve the f0-first defaults (auto) to paths, in place on a.wrist_root and a.traj_root.
    Returns (f0 root, object tree root)."""
    # The canonical order is f0 → judge → bake → retiming. `auto` derives the wrist root and the object tree root
    # together from the same f0 staging, so wrists (with the f0 pad) and objects (without it) cannot mix
    # generations; `off` on one side only would allow that, so it is refused.
    _F0ROOT = f0_root(a.dataset)
    if (a.wrist_root == "off") != (a.traj_root == "off"):
        raise RuntimeError(
            f"[judge] --wrist-root({a.wrist_root}) and --traj-root({a.traj_root}) mix generations — with only one "
            f"side off, wrists (padded) and objects (not padded) are off by the f0 pad. Give off to both or neither."
        )

    def _f0_default(val: str, sub: str, what: str) -> str:
        if val == "off":
            return ""
        if val != "auto":
            return val
        # the f0 shadow's subdirectory is the hand spec directory, not the hand id
        _hd = hand_spec_dir(a.hand)
        p = (_F0ROOT / _hd / sub) if sub else (_F0ROOT / _hd)
        if not p.is_dir():
            raise RuntimeError(
                f"[judge] no f0 shadow staging ({what}): {p}\n"
                f"  The judge reads the trajectory after f0 repair. Build the staging first:\n"
                f"    python -m dexengine.retarget.f0.prejudge --hand {a.hand} --dataset {a.dataset} --emb {a.emb} "
                f"--src-layer {a.src_layer} --clips <list>\n"
                f"  To judge on the source layer instead, pass `--wrist-root off --traj-root off`.\n"
                f"  (The f0 root comes from DEXENGINE_F0_STAGE; now: {_F0ROOT})"
            )
        return str(p)

    a.wrist_root = _f0_default(a.wrist_root, "", "wrist cache")
    a.traj_root = _f0_default(a.traj_root, a.emb, "object trees")
    TRAJ_ROOT = (
        Path(a.traj_root) if a.traj_root else layer_path(a.src_layer, a.dataset, hand=hand_spec_dir(a.hand), emb=a.emb)
    )
    return _F0ROOT, TRAJ_ROOT
