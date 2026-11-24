"""f0 repair of hand–object penetration (default on): when a side's fingers start deep inside its own main object,
the hand is moved out to a small residual depth over the whole spawn-hold window, keeping every link that touches the
object.

WHY. Measured on the final data (pen audit, wuji2 4,000 episodes): a deep start overlap kicks the object or never
resolves (> 10 mm: 26 % stay inside to the end). Measured in the GPU replay (contact on, zero-residual PD,
22 clips × 4): a deep in-grasp penetration persists only when the hand–object overlap is ≥ 5 mm at the step where the
spawn hold (configs/rl/recipe.yaml DEXENGINE_OBJ_SPAWN_HOLD) releases the object — 62 of 62 persistent replicas,
0 below.
The hold re-pins the object to its reference every step, so a hand that returns to a penetrating reference during the
hold recreates the overlap: a frame-0-only repair with a pad left it at 7/88 (original 6/88) once the hold covered the
pad, and dropped the object (released mid-pad) when it did not. Repairing every frame of the hold window gave 0/88
without losing more objects (74 vs 72 of 88 off the reference by > 50 mm).

THE MEASURE (what opens the repair and what accepts it) is the overlap PhysX sees: SAT depth between each hand
collision mesh (the strict model's convex pieces, one per <collision>) and each cooked convex piece of the object
(data.objects.hull.collision_parts — the shape the simulation collides). Face-normal axes only, as the pen audit's
SAT (against PhysX contact separation: correlation 0.996, median |difference| 0.38 mm). The solve (repair_obj_solve)
projects on the same cooked hull.

THE RULE
  open      no desk repair (desk depth ≤ the shallow threshold) · switch on · main object with a cooked hull ·
            frame-0 SAT depth > OBJ_REPAIR_MM; a desk-deep side opens only when a pinned still is over its open
            threshold (desk_start: frame 0 then lifts the hand off the desk by the desk repair's own clearance and
            margin, the desk repair does not run)
  window    frames 0 … window_frames()−1 (the spawn hold + OBJ_WINDOW_MARGIN) are each solved, warm-started from the
            previous frame; the last change then fades to 0 over OBJ_FADE_FRAMES frames. The clip's length and
            timeline are unchanged (no pad)
  accept    per frame: SAT depth ≤ OBJ_RESIDUAL_MM (the target) · desk no deeper than its start by more than
            DESK_DEEPER_MM (every vertex) · every link touching at that frame
            (within OBJ_TOUCH_MM on the visual mesh, or inside) still within the contact band of the cooked hull;
            the window: SAT at the release frame (release_frame) < OBJ_REPAIR_MM
  keep      frame 0 only: when no attempt passes, the attempt with the least SAT that passes the same checks at the
            open thresholds (main ≤ OBJ_REPAIR_MM, each other body ≤ its open value: what the gate would not open
            again) is kept and marked in the report (`obj_kept`); later frames keep carrying the previous change
  desk      a desk-repaired side is measured again at the desk repair's result: SAT > OBJ_REPAIR_MM refuses it
  refuse    frame 0, the release check or the desk check not accepted: SKIP_OBJ — the side is not written; the
            prejudge holds the clip (HOLD_OBJ — shadow_status.HELD: not judged, the reason — the failed check, its
            value and threshold — in the ledger; a rule rejection, the trajectory is dropped). A later frame not
            accepted keeps
            the previous frame's change (recorded per frame)
THE BODIES. A side measures its own main object, the other side's main object, every hand-on kinematic slot of the
clip (repair_bodies: SLOT rows whose training preset keeps the hand colliding) and every hand-on pinned still
(STATIC_SPAWN rows, the same hand cell; on triangles through repair_still: exact measure, grid solve), each with its own
thresholds (THRESHOLDS_MM): a main object opens above OBJ_REPAIR_MM and aims at OBJ_RESIDUAL_MM; a slot or a still does
not move, so it opens above SLOT_REPAIR_MM and aims at SLOT_RESIDUAL_MM (the same value: frame 0's keep frees nothing).
Every check above applies per body with that body's values.
The report's object values (obj_sat_mm …) stay the side's own main object; the other bodies are listed beside them.
A side without other bodies measures exactly its main object (same pieces, digest and key).
An articulated main object is measured on every link (ArticObject: the union of the links' cooked pieces, the parts
posed by the joint FK at each frame's joint values — the links the simulation collides); the repair moves the hand
only, the joint values stay the data's. Objects without a cooked hull are not measured here (marker `n/a`).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from dexengine.retarget.f0 import phrases as PH

#: SAT depth (mm) above which a side's frame-0 object penetration is repaired
OBJ_REPAIR_MM = 5.0
#: SAT depth (mm) the repair aims for — the solve's target and the first pass of every attempt (a touch, not an
#: overlap: 2–5 mm resolves without a kick in 69 %). A frame-0 side that no attempt brings there keeps its least-SAT
#: attempt that the gate would not open again (every body at or under its open threshold: failed_checks k=0) — a
#: refused repair no longer holds a clip whose fingers end shallower than an unrepaired side may start
OBJ_RESIDUAL_MM = 2.0
#: Desk deepening (mm) beyond the frame's start depth that a repaired frame may show and still count as 'no deeper'
DESK_DEEPER_MM = 1.0
#: Distance (mm) within which a link counts as touching the object at frame 0 (with CONTACT_MIN_PTS points, or points
#: inside). The repair must keep each such link within the contact band (repair_cfg.CONTACT_BAND, 5 mm). The band alone
#: also counts near-misses: on the example (oakink2 pour, wuji2 right) index/middle proximal and the thumb tip sit
#: 2.7–3.6 mm off the mug, and every feasible repair "dropped" them while opening the fingers 0.1–0.2 rad.
OBJ_TOUCH_MM = 1.0
#: Frames after the spawn hold's release that the window still covers (the release step is inside the window)
OBJ_WINDOW_MARGIN = 3
#: Frames over which the window's last change fades back to the data pose
OBJ_FADE_FRAMES = 10
#: SAT depth (mm) above which a hand-on kinematic slot's frame-0 overlap is repaired (a slot does not move)
SLOT_REPAIR_MM = 1.0
#: SAT depth (mm) a repaired side may keep against a hand-on kinematic slot
SLOT_RESIDUAL_MM = 1.0
#: (open, residual) mm per body class: "main" = a main object (the side's own or the other side's), "slot" = a hand-on
#: kinematic slot, "still" = a hand-on pinned still (repair_still; it does not move either: the slot's values)
THRESHOLDS_MM = {
    "main": (OBJ_REPAIR_MM, OBJ_RESIDUAL_MM),
    "slot": (SLOT_REPAIR_MM, SLOT_RESIDUAL_MM),
    "still": (SLOT_REPAIR_MM, SLOT_RESIDUAL_MM),
}
MARK = "[F0_OBJ]"


def window_frames() -> int:
    """Frames the object repair solves: the longest spawn hold of the RL recipe (configs/rl/recipe.yaml, read through
    its owner rl.launch.recipe; the tree clock is the policy step) + OBJ_WINDOW_MARGIN."""
    from dexengine.rl.launch import recipe as RR  # noqa: PLC0415 — the hold's owner; light (yaml)

    holds = [int(RR.defaults(p).get("DEXENGINE_OBJ_SPAWN_HOLD") or 0) for p in RR.PATHS]
    return max(holds) + OBJ_WINDOW_MARGIN


def obj_repair_on() -> bool:
    """The switch DEXENGINE_F0_OBJ_REPAIR (default on): "1" = on, "0" = off (desk-only, the rule
    before); any other value is an unknown input and dies. Every side prints the applied value (gate)."""
    v = os.environ.get("DEXENGINE_F0_OBJ_REPAIR", "1")
    if v not in ("0", "1"):
        raise SystemExit(f"{MARK} DEXENGINE_F0_OBJ_REPAIR={v!r} — only 0 (off) or 1 (on, default)")
    return v == "1"


def _planes(v: np.ndarray) -> np.ndarray | None:
    """Half-space equations (n, d), n·x + d ≤ 0 inside, of the convex hull of `v`; None for a flat set."""
    try:
        return ConvexHull(v).equations
    except Exception:  # noqa: BLE001 — qhull refuses flat / degenerate sets: no volume, no overlap
        return None


def policy_line(slots: bool = False, stills: bool = False) -> str:
    """The values that decide the object repair's outcome, as the f0 cache key carries them (shadow_key); `slots` = the
    side measures bodies besides its main object (their thresholds join the line; a side without keeps its line);
    `stills` = it measures pinned stills (their thresholds, the solve grid and the ruler's identity join the line)."""
    line = (
        f"open>{OBJ_REPAIR_MM!r}|residual≤{OBJ_RESIDUAL_MM!r}|touch≤{OBJ_TOUCH_MM!r}|window={window_frames()}"
        f"|fade={OBJ_FADE_FRAMES}|f0 keep≤open"
    )
    line += f"|slot open>{SLOT_REPAIR_MM!r}|slot residual≤{SLOT_RESIDUAL_MM!r}" if slots else ""
    if stills:
        from dexengine.data.objects.sdf_grid import ruler_id  # noqa: PLC0415 — the triangle ruler's identity owner
        from dexengine.retarget.f0 import repair_still as RST  # noqa: PLC0415

        o, r = THRESHOLDS_MM["still"]
        line += f"|still open>{o!r}|still residual≤{r!r}|grid={RST.GRID_STEP_M!r}+{RST.GRID_MARGIN_M!r}"
        line += f"|ruler={ruler_id()[:12]}|desk-deep start=object branch, floor=desk+clearance"
    return line


class CookedObject:
    """The main object's cooked convex pieces (object frame) with their planes and boxes; `digest` = the dump's.
    `bodies` = [(name, class)] of the measured bodies and `body_of` = each piece's body index (None: one body, the main
    object — repair_bodies builds the others)."""

    articulation = None  # a rigid object: its pieces do not depend on the qpos row
    bodies = (("main", "main"),)
    body_of = None
    stills = ()  # pinned stills on triangles (repair_bodies.BodySet sets them)

    def __init__(self, pieces, digest: str = ""):
        self.digest = digest
        self.pieces = []
        for v in pieces:
            e = _planes(np.asarray(v, float))
            if e is not None:
                v = np.asarray(v, float)
                self.pieces.append((v, e, v.min(0), v.max(0)))
        if not self.pieces:
            raise ValueError("cooked hull has no solid piece")

    def at(self, _qrow) -> CookedObject:
        """The pieces at a qpos row — a rigid object's are the same at every row."""
        return self


def _moved(piece: tuple, R: np.ndarray, p: np.ndarray) -> tuple:
    """A prepared piece (vertices, planes, box) under x → R x + p; the planes turn with it (no new hull)."""
    v, e, _lo, _hi = piece
    w = v @ R.T + p
    n = e[:, :3] @ R.T
    return (w, np.column_stack([n, e[:, 3] - n @ p]), w.min(0), w.max(0))


class ArticObject:
    """An articulated main object as the simulation collides it: the cooked pieces of every link (root link in the
    object frame, each part in its own link frame), and each part's joint FK spec and qpos column. `at(row)` = one
    CookedObject of all links with the parts posed by the joint FK (link_shapes.artic_part_pose — the cross model's
    owner) at that row's joint values."""

    stills = ()  # an articulated main object alone measures no still (repair_bodies.BodySet carries them)

    def __init__(self, root: CookedObject, parts: list, articulation: dict, digest: str):
        self.root, self.parts, self.articulation, self.digest = root, parts, articulation, digest

    def links_at(self, qrow) -> list:
        """[(CookedObject, R, p)] of every link at a qpos row, in the object frame: the root as is, each part posed by
        the joint FK at that row's joint value."""
        from dexengine.placement.collision.link_shapes import artic_part_pose  # noqa: PLC0415

        out = [(self.root, np.eye(3), np.zeros(3))]
        for spec, col, cobj in self.parts:
            p, R = artic_part_pose(np.zeros(3), np.eye(3), spec, float(qrow[col]))
            out.append((cobj, R, p))
        return out

    def at(self, qrow) -> CookedObject:
        out = CookedObject.__new__(CookedObject)
        out.digest, out.pieces = self.digest, list(self.root.pieces)
        for cobj, R, p in self.links_at(qrow)[1:]:
            out.pieces += [_moved(pc, R, p) for pc in cobj.pieces]
        return out


def scene_file(tree: Path) -> Path:
    """The tree's scene.xml, which an articulated main object's links are read from (`_artic_links`)."""
    return Path(tree) / "scene.xml"


def _artic_links(o: dict, dataset: str, tree: Path) -> tuple:
    """(root body name, [(link body name, [mesh paths])], welded, [part specs]) of the tree's scene.xml main object
    body — the readers the cross model uses (link_shapes: scene_meshes · xml_links · geom_files · part_spec); a relative
    meshdir uses the dataset's assets layer (the stage's --assets-root)."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    from dexengine.core.locations.layers import layer_path  # noqa: PLC0415
    from dexengine.placement.collision import link_shapes as LS  # noqa: PLC0415

    sx = ET.parse(str(scene_file(tree))).getroot()
    mesh_root, mfile = LS.scene_meshes(sx, layer_path("assets_objects", dataset).parent)
    body = next((b for b in sx.find("worldbody").findall("body") if b.get("name") == o["name"]), None)
    if body is None:
        raise ValueError(f"{tree.name}: main object body {o['name']!r} is not in scene.xml")
    links = LS.xml_links(body)

    def paths(el):
        return [Path(mesh_root) / f for f in dict.fromkeys(LS.geom_files(el, mfile, mesh_root))]

    files = [(el.get("name"), paths(el)) for el, _w in links]
    welded = {el.get("name"): [(paths(w), p, R) for w, p, R in ws] for el, ws in links if ws}
    specs = [LS.part_spec(child, body.get("name"), tree.name) for child, _w in links[1:]]
    return body.get("name"), files, welded, specs


def _artic_cooked(o: dict, dataset: str, tree: Path) -> tuple:
    """(ArticObject, "") of a tree's articulated main object — every link's cooked pieces (objects.hull.body_parts: the
    USD links matched to the scene.xml bodies) and each part's joint column (qpos_layout.artic_joint_cols, unknown
    dies); (None, why) without a USD or collision mesh (HullMissing), as for a rigid object. Only the geometry is
    cached (per dataset, asset and links); the joint columns and the articulation entry are this tree's."""
    from dexengine.core.paths import qpos_layout as QL  # noqa: PLC0415
    from dexengine.data.objects import hull as OH  # noqa: PLC0415

    root, files, welded, specs = _artic_links(o, dataset, tree)
    aid = OH.asset_id(o)
    key = (dataset, aid, repr(files), repr(sorted((k, repr(v)) for k, v in welded.items())), repr(specs))
    if key not in _COOKED:
        try:
            parts = OH.body_parts(dataset, aid, True, files, welded=welded)
            digest = OH.collision_parts(dataset, aid, True).digest
        except OH.HullMissing as e:
            _COOKED[key] = (None, f"no cooked hull ({e})"[:160])
            return _COOKED[key]
        _COOKED[key] = ({n: CookedObject(parts[n], digest) for n in [root] + [s["name"] for s in specs]}, digest)
    links, digest = _COOKED[key]
    if links is None:
        return None, digest
    cols = QL.read(str(tree), nq=None).artic_joint_cols()
    lost = [s["joint"] for s in specs if s["joint"] not in cols]
    if lost:
        raise ValueError(f"{tree.name}: part joints {lost} are not in qpos_layout.artic — their columns are unknown")
    moving = [(s, cols[s["joint"]], links[s["name"]]) for s in specs]
    return ArticObject(links[root], moving, o["articulation"], digest), ""


_COOKED: dict = {}


def cooked(o: dict | None, dataset: str, tree: Path | None = None):
    """(CookedObject | ArticObject | None, why) — cached per (dataset, asset). `o` = the side's task_info objects[] main
    entry (None = legacy tree); an articulated one is built from the tree's scene.xml (`tree` required). None with a
    reason when the object is not measured here: legacy tree · no USD or collision mesh (HullMissing). A stale dump of
    a USD that has collision meshes stays fatal (hull.CookedHullInvalid)."""
    if o is None:
        return None, "legacy tree (no objects[])"
    if o.get("articulation"):
        if tree is None:
            raise ValueError(f"{MARK} articulated main object {o.get('name')!r}: the tree (scene.xml) is required")
        return _artic_cooked(o, dataset, Path(tree))
    from dexengine.data.objects import hull as OH  # noqa: PLC0415

    key = (dataset, OH.asset_id(o))
    if key not in _COOKED:
        try:
            cp = OH.collision_parts(dataset, key[1], False)
            _COOKED[key] = (CookedObject([p for ps in cp.links.values() for p in ps], cp.digest), "")
        except OH.HullMissing as e:
            _COOKED[key] = (None, f"no cooked hull ({e})"[:160])
    return _COOKED[key]


class Sat(tuple):
    """(depth m, link) of the side's own main object — the value the report carries — with `others` = [(name, class,
    depth m, link)] of the other measured bodies."""

    def __new__(cls, rows: list):
        out = super().__new__(cls, (rows[0][2], rows[0][3]))
        out.others = rows[1:]
        return out


def _body_depths(hand_parts, cobj: CookedObject, oR, opos) -> list:
    """[(max overlap depth [m], deepest link)] per body of `cobj` between the hand pieces and the body's pieces.
    hand_parts = [(link, [world pts, …])]; each piece pair is overlapped along the face normals of both hulls (the
    smallest overlap is the depth; ≤ 0 = apart)."""
    best = [[0.0, ""] for _ in cobj.bodies]
    for ln, parts in hand_parts:
        for w in parts:
            loc = (w - opos) @ oR
            lo, hi = loc.min(0), loc.max(0)
            ea = None
            for k, (v, e, vlo, vhi) in enumerate(cobj.pieces):
                if (lo > vhi).any() or (hi < vlo).any():
                    continue
                ea = _planes(loc) if ea is None else ea
                if ea is None:
                    break
                o1 = -(loc @ e[:, :3].T + e[:, 3]).min(0)
                o2 = -(v @ ea[:, :3].T + ea[:, 3]).min(0)
                d = float(min(o1.min(), o2.min()))
                b = best[0 if cobj.body_of is None else cobj.body_of[k]]
                if d > b[0]:
                    b[0], b[1] = d, ln
    return best


def sat_depth(hand_parts, cobj: CookedObject, oR, opos) -> tuple[float, str]:
    """(max overlap depth [m], deepest link) between the hand pieces and every piece of `cobj`."""
    best = _body_depths(hand_parts, cobj, oR, opos)
    return tuple(max(best, key=lambda b: b[0])) if len(best) > 1 else tuple(best[0])


def sat_bodies(hand_parts, cobj: CookedObject, oR, opos) -> Sat:
    """The per-body SAT of the hand pieces (Sat: the main object's (depth, link) and the other bodies')."""
    best = _body_depths(hand_parts, cobj, oR, opos)
    return Sat([(n, c, d, ln) for (n, c), (d, ln) in zip(cobj.bodies, best, strict=True)])


def _over(measured, k: int, strict: bool = True) -> list:
    """The other bodies of `measured` (a Sat; a plain (depth, link) has none) above their threshold k (0 = open,
    1 = residual; strict = '>', else '≥'): [text]."""
    out = []
    for n, c, d, ln in getattr(measured, "others", ()):
        thr, mm = THRESHOLDS_MM[c][k], d * 1000.0
        if mm > thr or (not strict and mm >= thr):
            out.append(f"{c} {n} SAT {mm:.2f}mm {'>' if strict else '≥'} {thr:g}mm ({ln or '-'})")
    return out


def hand_parts_world(strict, wpos, wR, f12) -> list:
    """The strict model's collision pieces in world at (wrist pose, fingers): [(link, [pts, …])]."""
    out = []
    for ln, _pts, Rw, tw, _dl in strict.link_points_world(wpos, wR, f12):
        out.append((ln, [p @ Rw.T + tw for p in strict.parts.get(ln, ())]))
    return out


def desk_max_depth(strict, wpos, wR, f12, desk_z: float) -> float:
    """Deepest strict-model vertex below the desk plane (m, 0 = none)."""
    z = np.concatenate([p[:, 2] for _, p, _, _, _ in strict.link_points_world(wpos, wR, f12)])
    return float(np.clip(desk_z - z, 0.0, None).max()) if z.size else 0.0


def gate(rep: dict, measured: tuple | None, why: str, desk_deep: bool) -> bool:
    """Open the object repair for this side? measured = the frame-0 (SAT depth [m], link), None = not measured (`why`).
    A desk-deep side opens only for a still over its open threshold (`obj_desk_start`, desk_start). Records the measure
    in the report and prints the one marker line (the applied switch and threshold are visible on every side)."""
    on = obj_repair_on()
    if measured is None:
        rep["obj_sat_mm"] = None
        print(f"{MARK} side={rep['side']} switch={'on' if on else 'off'} n/a: {why}", flush=True)
        return False
    d, ln = measured
    rep["obj_sat_mm"] = round(d * 1000.0, 2)
    rep["obj_sat_link"] = ln
    others = getattr(measured, "others", ())
    if others:
        rep["obj_bodies"] = [
            {"name": n, "class": c, "sat_mm": round(x * 1000.0, 2), "link": b} for n, c, x, b in others
        ]
    still_open = any(c == "still" and x * 1000.0 > THRESHOLDS_MM[c][0] for _n, c, x, _b in others)
    opened = on and (not desk_deep or still_open) and (d * 1000.0 > OBJ_REPAIR_MM or bool(_over(measured, 0)))
    rep["obj_repair"] = bool(opened)
    rep.update({"obj_desk_start": True} if opened and desk_deep else {})
    extra = "".join(f" {c}:{n}={x * 1000.0:.2f}mm(open>{THRESHOLDS_MM[c][0]:g})" for n, c, x, _b in others)
    print(
        f"{MARK} side={rep['side']} switch={'on' if on else 'off'} sat={d * 1000.0:.2f}mm link={ln or '-'}{extra} "
        f"open>{OBJ_REPAIR_MM:g}mm residual≤{OBJ_RESIDUAL_MM:g}mm desk_repair={int(desk_deep)} → "
        f"{'REPAIR' if opened else 'keep'}{' (desk-deep start: a still opens)' if opened and desk_deep else ''}",
        flush=True,
    )
    return opened


def failed_checks(
    sat_m: float, link: str, desk_m: float, desk0_m: float, measured=None, desk_cap=None, k: int = 1
) -> str:
    """The acceptance checks a frame's result fails, each with its value and threshold ("" = accepted): SAT ≤
    OBJ_RESIDUAL_MM · every other body of `measured` (a Sat) within its residual · desk no deeper than the frame's
    start by more than DESK_DEEPER_MM, or, with `desk_cap` (frame 0 of a desk-deep start), the desk repair's own rule:
    no deeper than desk_cap (its desk margin). `k` = which THRESHOLDS_MM column bounds the bodies: 1 = the residual
    (the target, the pass), 0 = the open threshold (frame 0's keep: what the gate would not open again)."""
    out = []
    thr = THRESHOLDS_MM["main"][k]
    if sat_m * 1000.0 > thr:
        out.append(f"SAT {sat_m * 1000:.2f}mm > {thr:g}mm ({link or '-'})")
    out += _over(measured, k)
    if desk_cap is not None:
        out += [f"desk {desk_m * 1000:.3f}mm > {desk_cap * 1000:g}mm (desk-deep start)"] if desk_m > desk_cap else []
    elif (desk_m - desk0_m) * 1000.0 > DESK_DEEPER_MM:
        out.append(f"desk {desk_m * 1000:.2f}mm > start {desk0_m * 1000:.2f}mm + {DESK_DEEPER_MM:g}mm (deeper)")
    return " · ".join(out)


def desk_start(rep: dict, desk_z: float, desk_margin: float, clearance: float):
    """(floor, cap) of frame 0 on a desk-deep side the gate opened (`obj_desk_start`: a still opens there, so the object
    branch, not the desk repair, lifts the hand): the desk repair's own rule — solve with every vertex above the desk
    plane by its clearance, accept no deeper than its desk margin. None on any other side."""
    return (desk_z - desk_margin + clearance, desk_margin) if rep.get("obj_desk_start") else None


def release_frame() -> int:
    """The window frame the spawn hold releases the object at (the hold's length: window_frames − OBJ_WINDOW_MARGIN)."""
    return window_frames() - OBJ_WINDOW_MARGIN


def release_failed(sats_mm: list, measured: list | None = None) -> str:
    """The release check of a solved window ("" = passes): the SAT at the last pinned frame (release_frame − 1, the
    replay's release depth) and at the release frame (the state free physics starts from) must both be below
    OBJ_REPAIR_MM (the depth at release from which the replay's overlap persists), and every other body of the
    frames' Sat (`measured`, per frame) below its own open threshold. A shorter clip uses its last frame."""
    frames = [min(g, len(sats_mm) - 1) for g in (release_frame() - 1, release_frame())]
    f = max(frames, key=lambda g: sats_mm[g])
    if sats_mm[f] >= OBJ_REPAIR_MM:
        return f"release frame {f} SAT {sats_mm[f]:.2f}mm ≥ {OBJ_REPAIR_MM:g}mm"
    for g in dict.fromkeys(frames):
        late = _over(measured[g], 0, strict=False) if measured else []
        if late:
            return f"release frame {g} " + " · ".join(late)
    return ""


def after_desk(rep: dict, sat_fn) -> dict:
    """The object rule on a desk-repaired side: the object is measured again at the desk repair's result; a SAT above
    OBJ_REPAIR_MM refuses the side (the object repair does not run on a desk-repaired side). sat_fn = None (object not
    measured: `n/a` already printed) leaves the report as it is."""
    if rep.get("status") != "REPAIRED" or sat_fn is None:
        return rep
    sat = sat_fn(np.asarray(rep["_internal"][0], float))
    d, ln = sat
    rep["obj_sat_after_desk_mm"] = round(d * 1000.0, 2)
    print(f"{MARK} side={rep['side']} after the desk repair sat={d * 1000.0:.2f}mm link={ln or '-'}", flush=True)
    if d * 1000.0 > OBJ_REPAIR_MM:
        return refuse(rep, f"object SAT after the desk repair {d * 1000:.2f}mm > {OBJ_REPAIR_MM:g}mm ({ln or '-'})")
    late = _over(sat, 0)
    if late:
        return refuse(rep, "after the desk repair " + " · ".join(late))
    return rep


def refuse(rep: dict, why: str) -> dict:
    """The status of a refused object repair: a SKIP that leaves the side as it was (no pad)."""
    rep["status"] = f"{PH.SKIP_OBJ}: object repair refused — {why} (side kept as is)"
    rep["obj_refused"] = why
    return rep
