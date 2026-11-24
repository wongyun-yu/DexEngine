"""Shared names of the base-layer adapter: its version, the data root, layer locations and the declared task_info
fields (which fields are copied, derived or per-window)."""

from __future__ import annotations

from pathlib import Path

from dexengine.core import paths as P  # noqa: E402

dl = P.data_layout
#: No ③ reassignment and no ⑤ contact markers: the delivered roles pass through, and the hand–object assignment standard
#: (hand_object_assign) owns who handles what and the contact quality inputs.
# base adapter output version; covers re-derived scene convex parts (scene_parts.rederive), the retarget_input /
# supplement_input fingerprints, support.desk_rule, per-clip hot3d yaw (D5-b), the h2o desk plane rotation (D10-e'') and
# the OakInk2 part-tree groups (D1-d, D1-e)
ADAPTER_VERSION = "2026-10-05a"
SEG_FMT = "{clip}_seg{idx:02d}"
ROOT = P.main_root()
JUMP_RAD = 0.15
#: Fields base measures per window; D36 pieces do not inherit them (segment.pieces.WINDOW_DECL · window rule).
QUALITY_WINDOW = ("finger_jump_frames",)


def _dest(layer: str, ds: str, **axes) -> Path:
    return ROOT / dl.render(layer, ds, **axes)


#: Piece depth limit: following a chain where a piece's parent is again a piece (segment.split --nested) counts as
#: stale, not a crash, when it hits a loop.
_PIECE_DEPTH = 4
#: support fields base copies from INTAKE into the tree. With SUPPORT_DERIVED (fields base builds itself,
#: window-independent)
#: these are the fields a D36 piece (`_tN`) must share with its parent —
#: dexengine.data.segment.invariant.piece_stale_reason reads both tuples from this source (the piece invariant, next to
#: PIECE_COPY · non-window quality · scene · npz), so a parent rebuilt with a new desk re-slices its pieces.
SUPPORT_COPIED = (
    "cls",
    "desk_mode",
    "desk_z",
    "desk_z_est",
    "desk_z_confidence",
    "table_z",
    "support_status",
    "why",
    "hand_zmin",
    "obj_zmin",
    "desk_z_oracle",
    "oracle_diff_m",
    "oracle_source",  # the dexengine/data/support/grab.py (D10-b) fields are carried into the copy too
    # D14': the real desk footprint (capture xy hull) — the placement judge reads it from the tree
    # (desk_footprint_bounds_scene_for)
    "desk_footprint_raw",
    "desk_rule",
    # OakInk2 · HRDexDB: which desk rule version made desk_z (owner = base_layer.stale.DESK_RULE_OWNERS) — a
    # regenerated tree proves its INTAKE generation
)
SUPPORT_DERIVED = ("footprint", "footprint_capture", "source", "frame")
#: rest_intervals is window-dependent (sliced per window) — its piece invariant belongs to segment.split
#: --refresh-support.
#: provenance fields that record the build moment, not the content — a parent rebuilt byte-identical still changes them.
#: The D36 piece invariant (dexengine.data.segment.invariant.RESTATED_DECL) masks them: otherwise every dataset-wide
#: rebuild at the same ADAPTER_VERSION would re-slice (and re-judge) every piece. adapter_version is not here — a new
#: version is a new content claim.
PROVENANCE_VOLATILE = ("time",)
