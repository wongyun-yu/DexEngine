"""The f0 repair CLI table (repair.main_steps parses with it): sequence selection, run mode, tree roots and the solve
settings. The hand choices come from the registry."""

from __future__ import annotations

import argparse

from dexengine.data import rules as _RR  # active hand owner
from dexengine.retarget.f0.repair_cfg import SIDE_BUDGET_EVALS


def _add_selection_args(ap):
    """Dataset, hand, sequence-selection and run-mode arguments."""
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequence", default=None)
    # The hand list comes from the **registry** (a lookup table only fits yesterday's data) — a
    # literal list makes a newly registered hand die at the argparse stage while the chain above it
    # reports only "f0 failed", hiding that the cause is the hand list.
    from dexengine.core.paths import HAND_SPEC_DIR as _HSD  # noqa: PLC0415

    ap.add_argument(
        "--hand",
        default=_RR.ACTIVE_HAND,
        choices=tuple(sorted(_HSD)),
        help="which hand's trees to repair: trajectories_gen/<spec_dir>/ + robots/<spec_dir>/ caches "
        "(combo roots: --tree_root/--wrist_root). Default = the running hand (DEXENGINE_HAND).",
    )
    ap.add_argument(
        "--in_place",
        action="store_true",
        help="repair in place instead of writing a suffixed tree (backup *.pre_f0r.npz + "
        "F0_REPAIR_INPLACE.json marker; skipped when the marker exists). generated trees only.",
    )
    ap.add_argument(
        "--seqs_file",
        default=None,
        help="one sequence per line (names may contain spaces), instead of --sequence/--all.",
    )
    ap.add_argument("--force", action="store_true", help="repair again even when the in_place marker exists")
    ap.add_argument("--all", action="store_true", help="check every sequence dir (no writes)")
    ap.add_argument("--side", choices=["right", "left", "both"], default="both")
    ap.add_argument("--check_only", action="store_true")


def _add_solve_args(ap):
    """Tree roots, blending, budget, margins, clearance, report and arm arguments."""
    ap.add_argument(
        "--tree_root",
        default=None,
        help="combo tree root (dexengine.core.layout trees layer, paths.layer_path('trees', …)); "
        "it replaces the root derived from the hand.",
    )
    ap.add_argument(
        "--wrist_root",
        default=None,
        help="wrist cache root (e.g. …/robots/<arm>_<hand>), the parent of wrist_poses_{side}/.",
    )
    ap.add_argument("--n_blend", type=int, default=20, help="minimum pad frames (20 = 0.4 s; the step caps raise it)")
    ap.add_argument(
        "--side_budget_evals",
        type=int,
        default=SIDE_BUDGET_EVALS,
        help="per-side optimisation budget in n_pen evaluations (default owned by the module constant "
        "SIDE_BUDGET_EVALS). Keeps a side without a solution from using up the whole sequence timeout "
        "and losing the other side's repair. Counted in evaluations, not wall-clock seconds, so the "
        "output does not depend on machine or load",
    )
    ap.add_argument("--out_suffix", default="_f0")
    # The constraint is "0 overlap". A 0.003 default would let a 3 mm penetration pass
    # as a 'graze' and give n_pen=0 to sides that actually overlap.
    ap.add_argument(
        "--desk_margin",
        type=float,
        default=0.0,
        help="allowed depth below the desk plane (z=0) (0 = no overlap at all)",
    )
    ap.add_argument(
        "--pen_margin", type=float, default=0.0, help="allowed object penetration depth (0 = no overlap at all)"
    )
    # Default clearance = at least the object surface sample spacing (measured 0.96~3.03 mm).
    # Demanding a value below the resolution of the approximate test leaves the optimiser spinning
    # after a 0 buried in noise (measured 4~6 minutes per sequence).
    ap.add_argument(
        "--clearance",
        type=float,
        default=0.003,
        help="optimisation target clearance (m). Samples are a subset of the surface, so 0 on the "
        "samples does not guarantee a true 0. Solve with this clearance, verify strictly on all "
        "vertices, and retry with a larger clearance on failure.",
    )
    ap.add_argument(
        "--verify_retries",
        type=int,
        default=3,
        help="retries with doubled clearance after a failed strict verification",
    )
    ap.add_argument("--report_tsv", default=None)
    ap.add_argument(
        "--arm",
        default="",
        help="combo URDF selection. Default = DEXENGINE_ARM (data.rules.active_arm, no default). "
        "The hand subtree was measured to be the same for every arm "
        "(max palm-relative FK difference 7.8e-16 m, identical hand mesh md5). "
        "The URDF comes from the resolver every stage reads (sim_collider.resolved_urdf; override DEXENGINE_XR_URDF).",
    )


def _build_parser():
    """The repair CLI (one argparse table; the hand choices come from the registry)."""
    ap = argparse.ArgumentParser()
    _add_selection_args(ap)
    _add_solve_args(ap)
    return ap
