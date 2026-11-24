"""**Field declarations** of the D36 piece rule: the only home of which fields of a parent base tree go to a piece and
how (window rule).

Why separate: two sides read these declarations: segment.split, which builds pieces (piece_task_info · piece_support),
and base_layer.stale.tree_stale_reason, owner of the stage-1 column 2 (whether a parent is **of a generation the current
piece rule can cut**). segment.split requires an active hand (DEXENGINE_HAND) at import time, so importing it from
base_adapter would break hand-less callers (self-tests, diagnostics). Instead of a copy, the declarations moved to this
dependency-free module. segment.split takes the names from here; references such as `segment.pieces.PIECE_COPY` point
at the same objects.

Building piece task_info, the **window rule**: values measured on a window are re-measured per piece on its window
(no copies of parent values). Every top-level field is one of four kinds, and fields outside them fail (unknown fields
silently carried along or lost would break the rule).
  PIECE_COPY     window-independent statements (take · assets · layout · object structure · source GT gaps); the
                 parent value is received. Roles and sides in objects[] are re-decided per piece by the role step (4)
                 (their markers are rules and are not received). Fields rewritten per tree are declared by
                 segment.invariant.RESTATED_DECL, and the piece invariant (invariant.piece_stale_reason) masks them
  PIECE_DERIVED  built by segment.split on the piece window (name · window · support, piece_support)
  PIECE_PARTIAL  quality = everything except the producers' window fields (per-take retargeting audit) · rules = only
                 base-layer conversion markers (PIECE_RULES_COPY, already applied to the parent arrays); other verdict
                 markers are rewritten onto the piece by their step
  window fields  window values producers declare (WINDOW_DECL); not received. Without re-measuring, downstream fails on
      the missing marker

An unknown field means one of two things: a new field whose declaration was forgotten (add it), or an **old-generation**
field current code no longer writes (e.g. the top-level `object_reassigned` the hand–object assignment standard removed
from base). A parent holding the latter cannot be cut under the current rule, so stage-1 column 2 counts it as stale
(base_layer.stale.tree_stale_reason: a content measurement independent of the version record, an independent oracle).
"""

from __future__ import annotations

from functools import cache

from dexengine.data import (
    rules as RR,  # noqa: E402 — reads top-level constants from the owners' sources (_source_symbol, not executed)
)

PIECE_COPY = (
    "task",
    "dataset_name",
    "robot_type",
    "robot_type_delivered",
    "embodiment_type",
    "data_id",
    "hand",
    "hand_asset_sha256_xml3",
    "hand_qpos_dim",
    "hands",
    "ref_dt",
    "source_ref_dt",
    "frame",
    "provenance",
    "qpos_layout",
    "nq",
    "objects",
    "extra_objects",
    "articulated_object",
    "shared_object",
    "merged_duplicate",
    "supplemented_objects",
    "object_gt_gap",
    "source_frame",
    "mano_model",
    "right_object_slot",
    "left_object_slot",
    "right_object_mesh_dir",
    "left_object_mesh_dir",
    "right_object_convex_dir",
    "left_object_convex_dir",
)
PIECE_DERIVED = ("clip_id", "parent_clip", "segment", "segment_range", "segment_source", "support")
#: rules a piece receives = base-layer conversion markers (already applied to the parent arrays; the piece arrays are
#: their window). The 2e wrist tilt (WRIST_SLACK = dexengine.placement.repair.wrist_desk.RULE_KEY) is also a conversion
#: fixing base qpos; the piece's 2e column is measured as "the same marker as the parent"
#: (dexengine.placement.repair.wrist_desk_stage.freshness), so the piece receives the parent marker when cut. Without
#: it, pieces cut after stage 1 (stage1.split_apply) would stay without the marker and that hand's 2e column would never
#: fill.
PIECE_PARTIAL, PIECE_RULES_COPY = ("quality", "rules"), ("D1g", "D25", "WRIST_SLACK", "SUPP")
#: Producers declaring window fields: (file, top-level tuple constant, slot top|quality|support).
WINDOW_DECL = (
    ("dexengine/data/base_layer/common.py", "QUALITY_WINDOW", "quality"),
    ("dexengine/data/cases/taxonomy.py", "WINDOW_TOP", "top"),
    ("dexengine/data/filtering.py", "WINDOW_TOP", "top"),
    ("dexengine/data/filtering.py", "WINDOW_QUALITY", "quality"),
    ("dexengine/data/desk_penetration.py", "D24_SUPPORT_FIELDS", "support"),
)
#: Marker that the piece's rest intervals were sliced to the piece window (read by the auditor).
PIECE_WINDOW_KEY = "rest_intervals_window"


@cache
def _src_tuple(path: str, name: str) -> tuple:
    """Top-level tuple constant from a producer's source (importing it would pull heavy dependencies). Unreadable fails:
    a silently skipped field check would leave old values or leak window values."""
    ok, v = RR._source_symbol(path, name)
    if not ok or not isinstance(v, tuple) or not v:
        raise SystemExit(f"[piece_schema] cannot read {path}:{name} from the source: {(ok, v)!r}")
    return v


def _support_parent_keys() -> tuple:
    """Support fields a piece must share with its parent: the owner base_adapter's SUPPORT_COPIED + SUPPORT_DERIVED
    (window-independent)."""
    return sum(
        (_src_tuple("dexengine/data/base_layer/common.py", k) for k in ("SUPPORT_COPIED", "SUPPORT_DERIVED")), ()
    )


def _window(slot: str) -> set:
    return {k for path, name, sl in WINDOW_DECL if sl == slot for k in _src_tuple(path, name)}


def unknown_top_keys(ti: dict) -> list[str]:
    """Top-level task_info fields the piece rule (the four kinds, see the module docstring) does not know, sorted. Must
    be empty to build pieces."""
    return sorted(set(ti) - set(PIECE_COPY) - set(PIECE_DERIVED) - set(PIECE_PARTIAL) - _window("top"))


def unknown_support_keys(sup) -> list[str]:
    """Parent support fields the piece rule does not know, sorted (outside parent-owned fields · window fields · rest
    intervals and their window marker). Empty when support is not a dict (missing); the piece is then built without
    support (piece_support → None)."""
    if not isinstance(sup, dict):
        return []
    return sorted(set(sup) - set(_support_parent_keys()) - _window("support") - {"rest_intervals", PIECE_WINDOW_KEY})


def schema_stale_reason(ti: dict) -> str:
    """Why this tree **cannot be cut under the current piece rule** (unknown top-level or support fields), else "";
    stage-1 column 2 (base_layer.stale.tree_stale_reason) and the piece invariant
    (dexengine.data.segment.invariant.piece_stale_reason) use the same function."""
    top, sup = unknown_top_keys(ti), unknown_support_keys(ti.get("support"))
    if not (top or sup):
        return ""
    return f"fields outside the piece rule (old-generation schema) top-level {top} · support {sup}"
