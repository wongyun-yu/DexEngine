"""Thresholds and stamp vocabulary of the scene-setup rules (every coords submodule reads these).

The rules registry reads several of these values from this file's source, so each one stays a literal top-level
assignment. The start/rest support bands and the pedestal rule id are owned by `dexengine.data.support.bands`.
"""

from __future__ import annotations

from dexengine.data.support.bands import (  # noqa: F401 — re-exported (owner: data.support.bands)
    HIGH_REST_M,
    PEDESTAL_RULE,
    START_EMBED_M,
    SUPPORT_BAND_M,
    SUPPORT_GAP_MAX_M,
    SUPPORT_MIN_VERTS,
    SUPPORT_NEST_MAX_M,
    SUPPORT_POLY_PAD_M,
    SUPPORT_REACH_M,
    SUPPORT_STILL_DEG,
    SUPPORT_STILL_M,
)

#: Desk margin that keeps an object from overhanging the edge (m).
DESK_EDGE_MARGIN = 0.02
#: Scene modes: `table` has a desk, `aerial` has none (objects rest on support discs).
SCENE_MODES = ("table", "aerial")

# ── Support discs and pedestals ────────────────────────────────────────────────────────────────────────────────
#: A support disc collides with **objects only** (its own main object); the robot arm and hand pass through it, so
#: the judge and the gate never see discs. The env builds the same thing from this stamp value.
DISC_COLLIDE = "objects"
#: A support disc is invisible in renders and cameras (physics unchanged — it still carries its object).
DISC_VISIBLE = False
#: In a desk scene the support under an unheld object resting above the desk is a pedestal: a visible column from the
#: desk top up to the object's floor (footprint = the circle around the object's projection — the disc's x·y·r).
#: The judge's arm guard measures every column; aerial scenes keep the invisible objects-only disc.
#: What a column collides with depends on what stands on it, keyed by the stamp's `kind`:
#:   start · rest · end   under where a main object really rests (its start · a rest interval · where the clip
#:                        leaves it) → PEDESTAL_OBJECTS: objects only (the robot passes through)
#:   static               under a fixed object or a moved extra's rest (kinematic bodies) → PEDESTAL_SEEN: seen only
PEDESTAL_OBJECTS = "pedestal_objects"
PEDESTAL_SEEN = "pedestal_seen"
PEDESTAL_COLLIDE = {
    "start": PEDESTAL_OBJECTS,
    "rest": PEDESTAL_OBJECTS,
    "static": PEDESTAL_SEEN,
    "end": PEDESTAL_OBJECTS,
}
PEDESTAL_VISIBLE = True

# ── Start and rest support ─────────────────────────────────────────────────────────────────────────────────────
F0_ARM_OBJ_PEN_M = 0.01  # frame-0 arm↔object penetration: at the spawn pose an arm collider (sim shape, incl. the
#   wrist hull) deeper than this inside any main object excludes the clip — no placement
#   moves the wrist, and PhysX resolves it as a pop the policy never caused
F0_OBJ_OBJ_PEN_M = 0.010  # frame-0 object↔object penetration: two main objects' sim collision parts deeper than this
#   into each other at the spawn pose exclude the clip (PhysX pops them apart)
#: Hand ↔ its own main object at spawn: "report" = measured and reported, never a gate reason. The only mode;
#: f0_pen.set_thresholds raises on any other value.
F0_HAND_OWNOBJ_MODE = "report"
START_HOLD_FRAMES = 15  # the window (base frames) for asking "is it in the hand?" at the start (held → no support)
HELD_CONTACT_FRAC = 0.5  # an object is "held" over a window when its canonical contact sidecar (any finger) is set
#   on at least this fraction of the frames

#: Hidden z of an empty slot (m, env origin) — the training env and the collection form read this one value.
SLOT_HIDDEN_Z = -10.0

#: The fields of a desk-scene support disc row (the stamp).
DESK_DISC_KEYS = ("x", "y", "z", "r", "gap_m", "obj", "side", "start", "end", "kind", "collide")
#: Scene z of the floor — the plane the desk height and the robot base are measured from.
FLOOR_Z = 0.0

# ── End pedestals ──────────────────────────────────────────────────────────────────────────────────────────────
#: A column is not stood where the object's own reference passes: a vertex of that object inside the column (its
#: circle · below its top) by more than this, at any frame before the last, blocks it (`end_path_blocked`).
END_PATH_PEN_M = 0.020
#: Why a main object that ends above the desk got **no** end pedestal — the first that holds, in this order.
END_SKIP_REASONS = ("end_moving", "end_covered", "end_held", "end_off_desk", "end_on_object", "end_path_blocked")

# ── Staleness of a stamped statement ──────────────────────────────────────────────────────────────────────────
SETUP_STALE_TOL_M = 0.001  # a stamped support statement is **current** when every number the env spawns from it
#   (disc x·y·z·r · aerial z_off) is within this of the current rule's recompute on the same
#   tree: 1/10 of the smallest setup threshold (HIGH_REST_M = START_EMBED_M = rest Z_STAB =
#   10 mm), and recompute noise is 0 (JSON float round-trip is exact; the one rounded field
#   `gap_m` is not compared)
#: The base-owned setup statement the setup rules read beyond the object table: `support` (desk_z · rest_intervals ·
#: rest_intervals_window). The object table stays the tree's: its layout addresses the tree's own qpos (a bake may
#: inject the idle hand, so base addresses would read the wrong columns of a baked qpos).
SETUP_STATEMENT_KEYS = ("support",)
