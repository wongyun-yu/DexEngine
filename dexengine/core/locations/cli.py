"""Command line of dexengine.core.paths (`python -m dexengine.core.paths --help`)."""

from __future__ import annotations

import argparse
import os

from dexengine.core.locations.cli_what import WHAT, what_axis, what_combo, what_layout, what_roots
from dexengine.core.locations.cli_what_runs import what_registry, what_run

#: The `--help` description. It is a constant of its own, not `__doc__`, because the two have different readers:
#: the docstring is for whoever edits this file, `--help` for whoever runs it.
HELP = "Canonical path resolver — builds the paths of all 8 output kinds from one clip name."


def _add_identity_args(ap):
    """Clip, combo/hand/arm and dataset/emb axes."""
    ap.add_argument("--clip", default="", help="clip name or tree name (with or without suffix)")
    ap.add_argument(
        "--combo",
        default=os.environ.get("COMBO", ""),
        help="combo id. Empty = derived from --hand (+--arm) (combo_for) — no default",
    )
    ap.add_argument(
        "--hand",
        default=os.environ.get("DEXENGINE_HAND", ""),
        help="hand id or spec directory. No default — only env DEXENGINE_HAND is taken",
    )
    ap.add_argument(
        "--arm",
        default=os.environ.get("DEXENGINE_ARM", ""),
        help="arm id (only to narrow the combo derivation). No default — only env DEXENGINE_ARM is taken",
    )
    # Dataset and emb have no default. Only the branches that **use** that axis (layer · layout · ref-base · src-layer ·
    # canon · meta-dir · counts · f0_root · --clip) require them — branches that need no axis (sfx · hand-spec-dir ·
    # combo-ids) still run without them.
    ap.add_argument(
        "--dataset",
        default=None,
        help="dataset id (dexengine.core.layout.DATASETS). No default — a --what branch that uses this "
        "axis exits rc 2 without it",
    )
    ap.add_argument(
        "--emb",
        default=None,
        help="embodiment: right|left|bimanual. No default — a --what branch that uses this axis exits rc 2 without it",
    )


def _add_query_args(ap):
    """Inputs of the individual --what branches (scope, layer, run path, RL cell, roster …)."""
    ap.add_argument("--scope", default="", help="--what scope-takes: scope list (<ds>\\t<emb>/<clip>)")
    ap.add_argument("--input-dir", default="", help="--what scope-takes --emb: input dir holding the take folders")
    ap.add_argument("--src-layer", default="", help="stage-2 source layer (one base; '' = default_src_layer(dataset))")
    ap.add_argument(
        "--layer",
        default="",
        help="with --what layer: mano|retarget|base|trees|views|"
        "assets_objects|assets_robot|usd|precompute|… "
        "· with --what run-path: run_chunk|run_campaign|run_lists|exp_audit|… "
        "(--what run-layers)",
    )
    ap.add_argument("--tag", default="", help="--what run-path: chunk tag (run_chunk) · --what gate-table")
    ap.add_argument("--root", default="", help="--what gate-table: gate meta root (layer gate_meta + variant)")
    ap.add_argument(
        "--campaign",
        default="",
        help="--what run-path: campaign id = tag prefix (run_campaign) · collect work piece (run_collect_work)",
    )
    ap.add_argument("--tool", default="", help="--what run-path: log tool name (run_log)")
    ap.add_argument(
        "--attempt",
        default="",
        help="--what run-path: attempt id t<epoch> (run_chunk_try — the ledger's attempt column is the only source)",
    )
    ap.add_argument(
        "--bundle",
        default="",
        help="--what run-path|rl-bundle-dir: RL bundle id (run_rl_bundle · mixed · collect work piece "
        "run_collect_work)",
    )
    ap.add_argument("--track", default="", help="--what run-path --layer run_collect_text: text track name")
    ap.add_argument("--rel", action="store_true", help="--what run-path: repo-relative path (for transfer lists)")
    ap.add_argument(
        "--pairs-stdin",
        action="store_true",
        help="--what run-path: 'tag<TAB>path' for each stdin line 'tag<TAB>attempt' (no attempt · path "
        "failure = empty) — all at once, so a dispatcher loop does not start Python per tag",
    )
    ap.add_argument(
        "--path",
        default="",
        help="--what hand-of-path|rl-cell-of: path to classify · --what mode-record: record place of a writing run",
    )
    ap.add_argument(
        "--mode",
        default="",
        help="--what mode-record: mode name of a run that writes nothing (dryrun · need · measure … — form MODE_RE)",
    )
    ap.add_argument("--tree", default="", help="--what rl-cell-of|rl-bundle-dir: repo root (default: this code tree)")
    ap.add_argument("--area", default="", help="--what work-area: work_log | work_ckpt (data_layout.RUN_WORK_AREAS)")
    ap.add_argument("--roster", default="", help="--what roster-root: named roster (NAMED_ROSTERS) · layer = --layer")
    ap.add_argument(
        "--gen",
        default="",
        help="--what compare-root: comparison generation name (dexengine.core.layout.COMPARE_GENERATIONS)",
    )


def build_parser():
    ap = argparse.ArgumentParser(description=HELP)
    _add_identity_args(ap)
    _add_query_args(ap)
    ap.add_argument(
        "--what",
        default="all",
        help=", ".join(sorted(WHAT))
        + " | all | dirs | counts | meta-dir | sfx | placement_sfx | ik_dir_sfx"
        + " | main_root | pool_root | f0_root | owner-root | canon | src-layer | ref-base"
        + " | hand-spec-dir | hand-spec-dirs | combo-ids | combo-dirs | combo-arm-hands | split-combo"
        + " | plate-variant | arm-joint-pd | run-ckpt-root | work-area"
        + " | layout | layer | entry-layers | datasets | retarget-stage | reread-stage"
        + " | mano-model-dir | embodiments | scope-takes"
        + " | gate-table | roster-root"
        + " | taco-source-root | mode-record | compare-root | data-generation",
    )
    return ap


# `--what` branch groups — **this order is the priority** (once a group answers, later ones are not asked; e.g. sfx
# comes first even with --clip).
_WHAT_STEPS = (what_roots, what_axis, what_registry, what_run, what_layout)


def main() -> int:
    ap = build_parser()
    a = ap.parse_args()
    for step in _WHAT_STEPS:
        rc = step(a, ap)
        if rc is not None:
            return rc
    return what_combo(a, ap)
