#!/usr/bin/env python3
"""The **single owner** of qpos block addresses — a tree states for itself "which column is what".

Why.
    Consumers (loader, contact, collision, placement, judge, gate) must not restate the qpos layout as arithmetic:
    in a left-only one-hand tree column 0 is the left hand, a shared object has one object block, `<side>_object_slot`
    differs between generations, and npz `n_extra_objects` can disagree with the layout. The tree's
    `task_info.json` is the fact, and only this module reads it.

What it returns — **absolute columns**, not slot numbers.
    read(tree_dir) -> Layout
      .hand_qpos_dim  qpos width H of one hand (measured on the tree)
      .nq             total width
      .hands          {"right": start col, "left": start col}  — a missing hand has no key at all
      .main           {"right": start col, "left": start col}  — a shared object gives both sides the **same column**
      .extras         [start col, …]                           — in position order
      .artic          {"right": {"col": col, "npz_key": …, "axis":…, "range":…}, …}
      .objects        [{name, role, side, obj_id, mesh_dir, start, width, articulation}] (when present)
      .obj_ids        {"right": obj_id, "left": obj_id}  — main object **identity** (obj_id of objects[] role=main;
                       legacy: basename of `<side>_object_mesh_dir`). Shared object → the same id on both sides.
                       A side without one has no key at all
      .mesh_dirs      {"right": mesh_dir, "left": mesh_dir} — mesh directory of that identity (as written in task_info)
      .source         "task_info.layout" | "legacy_slots" | "legacy_scene_xml" | "legacy_default"
      .shared_object  bool
    Callers that still need slot numbers get (right, left) from `.legacy_slots()`.

Legacy trees, byte-identical.
    For trees without a `qpos_layout` key (the frozen V1 generation — the preserved roster root),
    `qpos_legacy.from_legacy()` reproduces the `resolve_object_slots` + `obj_offset = 2H + 7*n_extra` + `7*slot`
    arithmetic **exactly**. The decision rests on a measurement ("the tree has no qpos_layout key"), not a silent
    default, and is kept in `source`. The self-test `python3 -m dexengine.core.qpos_layout --selftest <tree root>`
    (qpos_legacy.selftest) compares old and new results for every tree.

Standard library only, so the Isaac package and the scripts use the same reader.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

#: Home of the side words — the same value is never restated under a second name.
SIDES = ("right", "left")
_CACHE: dict[str, Layout] = {}


#: task_info key of the desk-scene support statement (the bake stamps `coords.compute_desk_support` under it; env,
#: judge, gate and final guard read it). It pairs with `aerial_support` of the aerial scene.
DESK_SUPPORT_KEY = "desk_support"


def desk_z_of(ti: dict, where: str = "") -> tuple[float, str]:
    """Support-surface z (capture coordinates) of a desk-scene tree = (desk_z, source) — **the one rule** read by
    judge and bake (coords.scene_z_offset) and by the training runtime (Layout.world_z_offset → trajectory.loader).
    With `support`: support.desk_z (ValueError if None). Without `support`: 0.0 only for TACO staging (stage 1 —
    z=0 is the desk surface; dataset_name taco and no qpos_layout key); anything else is a ValueError (no silent 0)."""
    sup = ti.get("support")
    if sup is None:
        if str(ti.get("dataset_name", "")).lower() == "taco" and "qpos_layout" not in ti:
            return 0.0, "taco_staging(z0=desk)"
        raise ValueError(f"{where}: task_info has no support.desk_z — a tree outside the base layer cannot be grounded")
    if sup.get("desk_z") is None:
        raise ValueError(
            f"{where}: support.desk_z is None (desk_mode={sup.get('desk_mode')}) — judge a tree with "
            f"an unknown support surface before grounding"
        )
    return float(sup["desk_z"]), f"support.desk_mode={sup.get('desk_mode')}"


def _active_sides(ti: dict) -> tuple:
    """The active sides the tree declares. Both sides when absent (older trees assume two hands)."""
    v = ti.get("active_sides")
    if not v:
        return SIDES
    out = tuple(x for x in v if x in SIDES)
    if not out:
        raise ValueError(f"active_sides={v!r} holds no valid side — it needs at least one of right/left")
    return out


@dataclass
class Layout:
    """The qpos address table of one tree. Every value is an **absolute column index**."""

    tree_dir: str
    nq: int
    hand_qpos_dim: int
    hands: dict = field(default_factory=dict)
    main: dict = field(default_factory=dict)
    extras: list = field(default_factory=list)
    artic: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)
    source: str = "task_info.layout"
    shared_object: bool = False
    n_extra_npz: int | None = None  # extra count the npz states (if any); a check only, never an address
    obj_ids: dict = field(default_factory=dict)  # side -> main object obj_id (owner for spawner·USD·mesh lookup)
    active_sides: tuple = SIDES  # active sides the tree declares; no key = both active
    mesh_dirs: dict = field(default_factory=dict)  # side -> main object mesh_dir (as written in task_info)
    scene: str = "table"  # stamped scene (owner coords.scene_mode): "table" | "aerial"
    aerial: dict | None = None  # aerial scene: task_info.aerial_support (z_off·discs); else None
    desk_support: dict | None = None  # table scene: task_info[DESK_SUPPORT_KEY]; None when not stamped
    desk_z: float | None = None  # table-scene support z (capture coords); None → see desk_z_err
    desk_z_src: str = ""
    desk_z_err: str | None = None
    dataset: str = ""  # dataset the tree states (task_info.dataset_name); "" when absent

    def world_z_offset(self, desk_height: float) -> float:
        """This clip's offset from tree (capture) z to world z — desk scene = desk_height − desk_z, aerial scene =
        aerial_support.z_off (stamped by the bake). ValueError when the statement is missing. The same rule as
        coords.scene_z_offset in judge and bake (only the aerial computation before stamping belongs to coords)."""
        if self.scene == "aerial":
            z_off = (self.aerial or {}).get("z_off")
            if z_off is None:
                raise ValueError(
                    f"{self.tree_dir}: scene=aerial but task_info.aerial_support.z_off is missing — the "
                    f"tree comes from another bake generation"
                )
            return float(z_off)
        if self.desk_z is None:
            raise ValueError(self.desk_z_err or f"{self.tree_dir}: no desk support-surface statement")
        return float(desk_height) - self.desk_z

    def main_articulation(self, side: str) -> dict | None:
        """The joint statement of side `side`'s main object (`objects[].articulation`) — None for a rigid body or no
        main object. Only this is read (never guessed from names or paths). The one source for the asset id
        (`artifact_paths.object_asset_id(main_object_id, main_articulation)`) and the run axis's joint type."""
        oid = self.obj_ids.get(side)
        if oid is None:
            return None
        return next(
            (
                o["articulation"]
                for o in (self.objects or [])
                if o.get("role") == "main" and str(o.get("obj_id")) == str(oid) and o.get("articulation")
            ),
            None,
        )

    # ── Lookups ──────────────────────────────────────────────────────────────
    def hand_slice(self, side: str):
        """(start, end) or None. A missing hand is None — the caller decides zero-fill or mask **explicitly**."""
        s = self.hands.get(side)
        return None if s is None else (s, s + self.hand_qpos_dim)

    def is_active(self, side: str) -> bool:
        """Whether this side is **active**. An inactive side carries the hand at its idle pose and **normally has no
        main object**.

        The only evidence is the `active_sides` the tree writes itself — never guessed from names or datasets.
        An older tree without the key counts as both sides active (byte-identical to legacy).
        """
        return side in self.active_sides

    def idle_sides(self) -> tuple:
        """**Idle sides** = sides the tree declares inactive (outside ``active_sides``) AND that carry no main object.

        The one rule the training entry (dexengine/rl/cli.py --sequence_dir), the env statement (``cfg.idle_side``)
        and the loader cross-check read. The idle hand is baked at its rest pose (fingers at
        hand_surface.rest_finger_q, arm at the idle rest IK); the side has no object to spawn, reward or terminate on.
        A side declared ACTIVE without a main object is not idle — it is a defect, and callers keep dying on it."""
        return tuple(s for s in SIDES if not self.is_active(s) and self.main.get(s) is None)

    def manip_sides(self) -> tuple:
        """**Manipulating hands** = hands present in the tree ∧ with a main object. This is what "loop over sides"
        means for f0, judge and bake — a side with a hand but no main object (an inactive hand) exists only at its idle
        pose and is not processed. Empty → the clip has no manipulation (first-pass exclusion no_main_object).
        This method alone owns the "manipulating hand" / "no main object" decision — candidate selection's
        `dexengine.data.candidates.no_main_object` only asks whether it is empty (no second predicate reading the
        assignment record `rules.ASSIGN`)."""
        return tuple(s for s in ("right", "left") if s in self.hands and s in self.main)

    def main_slice(self, side: str):
        s = self.main.get(side)
        return None if s is None else (s, s + 7)

    def extra_slices(self):
        return [(s, s + 7) for s in self.extras]

    # ── Articulated objects ──────────────────────────────────────────────────────
    def object_starts(self) -> list:
        """Start columns of every 7-wide object block (main ∪ extra), ascending — a shared object once."""
        return sorted(set(self.main.values()) | set(self.extras))

    def artic_cols(self) -> list:
        """Joint columns (absolute), ascending; [] if none. Placement (rigid transform) and resampling must **pass
        them through unchanged**.

        `artic` is keyed by **side**, so it misses the columns of articulated objects demoted to background
        (side=None). The truth for columns is `objects[].articulation` — the union of both is returned (required for
        blocks()/qvel_blocks() to cover nq)."""
        cols = {int(a["col"]) for a in self.artic.values()}
        cols |= {
            int((o.get("articulation") or {})["qpos_index"]) for o in (self.objects or []) if o.get("articulation")
        }
        return sorted(cols)

    def artic_joint_cols(self) -> dict:
        """{scene.xml joint name: qpos column} of every articulated object (objects[].articulation) — the column a
        part's joint FK reads (the cross model's set_pose · the f0 object repair)."""
        return {
            str(o["articulation"]["joint"]): int(o["articulation"]["qpos_index"])
            for o in (self.objects or [])
            if o.get("articulation")
        }

    def artic_records(self) -> list:
        """Every articulated-object record [{name, side, col, npz_key, axis, pos, range, type}] — side-agnostic
        (background articulated objects included)."""
        out = [
            dict(
                name=o.get("name"),
                side=o.get("side"),
                col=int(o["articulation"]["qpos_index"]),
                **{k: o["articulation"].get(k) for k in ("npz_key", "axis", "pos", "range", "type")},
            )
            for o in (self.objects or [])
            if o.get("articulation")
        ]
        if not out:
            out = [
                dict(
                    name=None,
                    side=s,
                    col=int(a["col"]),
                    **{k: a.get(k) for k in ("npz_key", "axis", "pos", "range", "type")},
                )
                for s, a in (self.artic or {}).items()
            ]
        return out

    def artic_npz_keys(self) -> list:
        """The npz keys of every joint-value sidecar (`objects[].articulation.npz_key`, e.g. qpos_obj_arti_right
        (T, 1)) — side-agnostic (background articulated objects included), in statement order, no duplicates, records
        without a key left out. The one owner of which sidecars a time-axis step (the f0 pad · the retiming resample)
        carries along with qpos: picking them from the per-side `artic` misses the background articulated objects."""
        return list(dict.fromkeys(r["npz_key"] for r in self.artic_records() if r.get("npz_key")))

    def blocks(self) -> list:
        """Blocks covering qpos without gaps [(start, kind, width)] — kind ∈ hand|object|joint. Raises when they do
        not cover it (hole · overlap). The only basis for deriving qvel addresses: hand → same width, object (free)
        → 6, joint → same width (1)."""
        bl = [(int(s), "hand", int(self.hand_qpos_dim)) for s in self.hands.values()]
        bl += [(int(s), "object", 7) for s in self.object_starts()]
        bl += [(int(c), "joint", 1) for c in self.artic_cols()]
        bl.sort()
        pos = 0
        for s, k, w in bl:
            if s != pos:
                raise ValueError(
                    f"{self.tree_dir}: qpos blocks break at {pos} (next {k}@{s}) — the layout does not cover qpos"
                )
            pos = s + w
        if pos != self.nq:
            raise ValueError(f"{self.tree_dir}: block sum {pos} != nq {self.nq}")
        return bl

    def qvel_blocks(self) -> list:
        """[(kind, qpos_start, qpos_width, qvel_start, qvel_width)] in blocks() order. A free object has qvel 6
        (3 linear + 3 angular)."""
        out, v = [], 0
        for s, k, w in self.blocks():
            wv = 6 if k == "object" else w
            out.append((k, s, w, v, wv))
            v += wv
        return out

    @property
    def nv(self) -> int:
        return sum(b[4] for b in self.qvel_blocks())

    def main_object_id(self, side: str):
        """The main object obj_id of side `side`, None if absent. Never returns an extra object."""
        return self.obj_ids.get(side)

    @property
    def identity_source(self) -> str:
        """Where the identity came from — "objects[]" (task_info.layout) | "legacy" (`<side>_object_mesh_dir` key)."""
        return "objects[]" if self.source == "task_info.layout" else "legacy"

    def legacy_slots(self) -> tuple[int, int]:
        """The old (right_slot, left_slot) convention — index among main-object blocks. Not for new code."""
        mains = sorted(set(self.main.values()))

        def _idx(side):
            s = self.main.get(side)
            return mains.index(s) if s in mains else 0

        return _idx("right"), _idx("left")

    def check(self) -> None:
        """Measurement ↔ statement oracle (an allowed duplicate). A mismatch never passes silently."""
        for side, s in self.hands.items():
            if s + self.hand_qpos_dim > self.nq:
                raise ValueError(f"{self.tree_dir}: {side} hand block {s}+{self.hand_qpos_dim} > nq {self.nq}")
        for side, s in self.main.items():
            if s + 7 > self.nq:
                raise ValueError(f"{self.tree_dir}: {side} main object block {s}+7 > nq {self.nq}")
        for s in self.extras:
            if s + 7 > self.nq:
                raise ValueError(f"{self.tree_dir}: extra block {s}+7 > nq {self.nq}")
        for side, a in self.artic.items():
            if a["col"] >= self.nq:
                raise ValueError(f"{self.tree_dir}: {side} joint column {a['col']} >= nq {self.nq}")


# ── Reading ──────────────────────────────────────────────────────────────────
def read(
    tree_dir: str,
    *,
    expect_hand_qpos_dim: int | None = None,
    expect_hand: str | None = None,
    nq: int | None = None,
    require_layout: bool = False,
    use_cache: bool = True,
) -> Layout:
    """The address table of one tree. With `expect_*`, the statement (HandSpec) is checked against the tree record
    and a mismatch raises.

    `nq` is the **actual qpos width**. Some legacy trees have a minimal task_info without even `hand_qpos_dim`
    (f0 shadow = retarget staging copy). Defaulting to 18 there makes the hand read another robot's columns — wuji
    (H=26, nq=66) read as 18 puts the objects at columns 36–49. So when the width is known, H is **derived** as
    `H = (nq − 14) / 2` (the same rule retime.py uses; legacy convention = 2 hands + 2 objects).
    """
    key = f"{tree_dir}|{expect_hand_qpos_dim}|{expect_hand}|{require_layout}|{nq}"
    if use_cache and key in _CACHE:
        return _CACHE[key]
    ti_p = os.path.join(tree_dir, "task_info.json")
    if not os.path.isfile(ti_p):
        raise FileNotFoundError(f"task_info.json missing: {tree_dir}")
    with open(ti_p) as f:
        ti = json.load(f)
    lay = layout_of(tree_dir, ti, require_layout=require_layout, nq=nq, expect_H=expect_hand_qpos_dim)
    if expect_hand_qpos_dim is not None and lay.hand_qpos_dim != int(expect_hand_qpos_dim):
        raise ValueError(
            f"{tree_dir}: hand width differs from the spec — tree {lay.hand_qpos_dim} vs spec "
            f"{expect_hand_qpos_dim}. One of them is wrong (the tree is not allowed to win — training "
            "on diverged conditions turns the hand into another robot)."
        )
    if expect_hand and ti.get("hand") and str(ti["hand"]) != str(expect_hand):
        raise ValueError(f"{tree_dir}: hand identity differs — tree {ti['hand']} vs spec {expect_hand}")
    if use_cache:
        _CACHE[key] = lay
    return lay


def layout_of(
    tree_dir: str, ti: dict, *, require_layout: bool = False, nq: int | None = None, expect_H: int | None = None
) -> Layout:
    """Build the address table from an already-read task_info (`ti`) — **the same interpretation** read() applies
    after reading the file (no cache), so callers already holding task_info (candidate selection · setup audit) use
    the same reader without reading the file again."""
    from dexengine.core.qpos_legacy import from_legacy  # imports Layout from here: resolved at call time

    lay = (
        from_layout(tree_dir, ti)
        if ti.get("qpos_layout")
        else from_legacy(tree_dir, ti, require_layout, nq=nq, expect_H=expect_H)
    )
    lay.check()
    lay.dataset = str(ti.get("dataset_name") or "")
    try:  # dies only when world_z_offset needs it; address-only callers are fine
        lay.desk_z, lay.desk_z_src = desk_z_of(ti, tree_dir)
    except ValueError as e:
        lay.desk_z_err = str(e)
    return lay


def _add_object(o, tree_dir, main, obj_ids, mesh_dirs, extras, artic):
    """Record one objects[] entry: per-side main start/id/mesh (contradictions raise), extra starts, articulation."""
    st = int(o["qpos_start"])
    if o.get("role") == "main":
        # identity = obj_id. A mismatch with mesh_dir's basename is a self-contradiction — never pick one silently.
        oid = o.get("obj_id")
        mdir = o.get("mesh_dir")
        if not oid or not mdir:
            raise ValueError(f"{tree_dir}: objects[] main object {o.get('name')!r} has no obj_id/mesh_dir")
        if str(mdir).rstrip("/").split("/")[-1] != str(oid):
            raise ValueError(
                f"{tree_dir}: objects[] main object obj_id={oid!r} differs from the basename of mesh_dir={mdir!r}"
            )
        for side in SIDES if o.get("side") == "both" else [o.get("side")]:
            if side in SIDES:
                if side in main:
                    raise ValueError(f"{tree_dir}: objects[] has two {side} main objects ({obj_ids[side]!r}, {oid!r})")
                main[side] = st
                obj_ids[side] = str(oid)
                mesh_dirs[side] = str(mdir)
    else:
        extras.append(st)
    a = o.get("articulation")
    if a:
        # even for side='both' the npz key carries the original side name — use what the tree wrote, never guess.
        for side in SIDES if o.get("side") == "both" else [o.get("side")]:
            if side in SIDES:
                artic[side] = {
                    "col": int(a["qpos_index"]),
                    "npz_key": a.get("npz_key"),
                    "axis": a.get("axis"),
                    "pos": a.get("pos"),
                    "range": a.get("range"),
                    "type": a.get("type"),
                }


def from_layout(tree_dir: str, ti: dict) -> Layout:
    """Base layer (base_adapter v"c"/"d" and later): qpos_layout + objects[] are the facts."""
    L = ti["qpos_layout"]
    nq = int(ti.get("nq") or sum(int(e["width"]) for e in L))
    if sum(int(e["width"]) for e in L) != nq:
        raise ValueError(f"{tree_dir}: Σ qpos_layout.width != nq({nq})")
    H = int(ti["hand_qpos_dim"])
    hands, artic = {}, {}
    for e in L:
        if e["kind"] != "hand":
            continue
        side = e.get("side")
        if side not in SIDES:
            raise ValueError(f"{tree_dir}: hand block side={side!r}")
        if int(e["width"]) != H:
            raise ValueError(f"{tree_dir}: {side} hand width {e['width']} != hand_qpos_dim {H}")
        hands[side] = int(e["start"])
    objs = ti.get("objects") or []
    main, extras, obj_ids, mesh_dirs = {}, [], {}, {}
    for o in objs:
        _add_object(o, tree_dir, main, obj_ids, mesh_dirs, extras, artic)
    extras.sort()
    return Layout(
        tree_dir=tree_dir,
        nq=nq,
        hand_qpos_dim=H,
        hands=hands,
        main=main,
        extras=extras,
        artic=artic,
        objects=objs,
        source="task_info.layout",
        shared_object=bool(ti.get("shared_object")) or len(set(main.values())) == 1 and len(main) == 2,
        obj_ids=obj_ids,
        mesh_dirs=mesh_dirs,
        active_sides=_active_sides(ti),
        scene=str(ti.get("scene") or "table"),
        aerial=(ti.get("aerial_support") if str(ti.get("scene") or "") == "aerial" else None),
        desk_support=(ti.get(DESK_SUPPORT_KEY) if str(ti.get("scene") or "table") != "aerial" else None),
    )


def with_npz_extras(lay: Layout, n_extra_npz: int) -> Layout:
    """Legacy trees only: shift the main-object columns by npz `n_extra_objects` (the same arithmetic as the loader).

    Not for the base layer — there addresses are already absolute columns and the npz value is only a check (it is
    known to disagree on some hocap trees).
    """
    if lay.source == "task_info.layout":
        lay.n_extra_npz = int(n_extra_npz)
        return lay
    H = lay.hand_qpos_dim
    r, left_slot = lay.legacy_slots()
    off = 2 * H + 7 * int(n_extra_npz)
    lay.main = {"right": off + 7 * r, "left": off + 7 * left_slot}
    lay.extras = [2 * H + 7 * i for i in range(int(n_extra_npz))]
    lay.n_extra_npz = int(n_extra_npz)
    return lay


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", metavar="ROOT", help="compare every <ROOT>/<clip>/task_info.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", metavar="TREE", help="print the address table of one tree")
    a = ap.parse_args()
    if a.show:
        lay = read(a.show)
        print(
            json.dumps(
                {
                    "source": lay.source,
                    "nq": lay.nq,
                    "H": lay.hand_qpos_dim,
                    "hands": lay.hands,
                    "main": lay.main,
                    "extras": lay.extras,
                    "artic": lay.artic,
                    "shared": lay.shared_object,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        sys.exit(0)
    if a.selftest:
        from dexengine.core.qpos_legacy import selftest

        sys.exit(selftest(a.selftest, a.limit))
    ap.print_help()
