#!/usr/bin/env python3
"""The arm colliders the sim spawns — the one generator of their shapes, and a digest of those shapes.

Rule (owners read, not restated): every arm link (ArmSpec bodies index_base..ee_body_index, each side) collides with its
URDF ``<collision>`` capsules (what the USD bake converts), except the links the collider revision
``core.arms.ARM_SIM_COLLIDER_REV[arm]`` names: ``replace`` drops the vendor collider, ``replace`` and ``add`` get the
convex hull of the link's ``<visual>`` meshes reduced to ``hull_vertex_cap`` vertices (``robots.wrist_hull.link_hull``).
A revised link's URDF joint parent must be a link of the same arm (``_check_rev``). An arm with no revision spawns its
body-generation colliders. The torso (links joined to the URDF root by fixed joints — one static root body in the USD)
follows the same rule by link name (``torso_replace`` · ``torso_add``), in root coordinates (``torso_bodies``), cut into
parts where the decomposition rule says so (torso_split.torso_parts — measured, never listed); every vendor collider on
that body must be named (``_check_torso``).

Readers: the USD generator (dexengine/robots/usd/collider_rev.py) writes the hull vertices of ``bodies`` · the
collision sets (cross_sets.load_urdf_geometry) measure arm and torso links with ``shape_dicts`` · the judge ruler
(arm_surface) measures ``bodies`` and ``torso_bodies`` · the env's spawn check checks the same revision on the stage.
``digest`` hashes the shapes themselves (capsule radius and axis end points, hull vertices) — a measurement, so a new
revision or a changed mesh cannot pass under an old name. Marker: ``[XR_SIM_COLLIDER]`` (every shape_dicts call).
``measured_digest`` is how a reader in any interpreter gets it (the judge input fingerprint · the IK pool clearance
sidecar): one child process of the retargeting python (``paths.require_retarget_py`` — the interpreter with the mesh
stack) runs this module with ``--digest <arm>``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dexengine.core import arms as AR
from dexengine.core import paths as P
from dexengine.core.repo import FULL_ROBOT_URDF_DIR, REPO_ROOT
from dexengine.robots import wrist_hull as WG


@dataclass(frozen=True, eq=False)
class Capsule:
    """A vendor capsule in its link frame: radius ``r``, cylinder length ``L`` along the local z of (xyz, R)."""

    link: str
    r: float
    L: float
    xyz: np.ndarray
    R: np.ndarray

    def ends(self) -> tuple[np.ndarray, np.ndarray]:
        """The two end points of the capsule axis (link frame)."""
        a = self.R @ np.array([0.0, 0.0, self.L / 2.0])
        return self.xyz - a, self.xyz + a


@dataclass(frozen=True, eq=False)
class Hull:
    """A revision hull: the reduced convex hull vertices of a link's visual meshes (an arm link: link frame) or of one
    part of them (a torso link: root coordinates, ``part`` = its index under the decomposition rule torso_split)."""

    link: str
    verts: np.ndarray
    mode: str  # "replace" | "add" (the revision field that named the link)
    stats: dict = field(default_factory=dict)
    meshes: list = field(default_factory=list)  # the visual mesh files it was built from (file · sha256)
    part: int | None = None  # torso part index (0.. — one part = the link keeps one hull)

    @property
    def name(self) -> str:
        """The body's name: the link, or ``<link>#<part>`` for a torso part (the judge keys · digest rows)."""
        return self.link if self.part is None else f"{self.link}#{self.part}"


def combo_urdf(arm: str, hand: str, body_variant: str | None = None) -> Path:
    """The merged URDF of (arm, hand) for a registered variant (default: the arm's canon one) — the URDF its body was
    baked from (core.arms.urdf_variant: a derivative reads its source body's, a material-only body its source's)."""
    v = AR.canon_variant(arm) if body_variant is None else AR.urdf_variant(arm, body_variant)
    d = P.combo_id(arm, hand, v)
    return FULL_ROBOT_URDF_DIR / d / f"{d}.urdf"


def arm_link_names(arm: str, side: str) -> list[str]:
    """The arm link names of one side, shoulder to EE (ArmSpec bodies index_base..ee_body_index)."""
    spec = AR.ARM_REGISTRY[arm]
    return [spec.body_name(side, k) for k in range(spec.index_base, spec.ee_body_index + 1)]


def _link_el(root, link: str):
    el = next((x for x in root.findall("link") if x.get("name") == link), None)
    if el is None:
        raise ValueError(f"arm link {link} is not in the URDF")
    return el


def vendor_capsules(root, link: str) -> list[Capsule]:
    """The URDF ``<collision>`` bodies of an arm link — capsules only; any other shape dies (it cannot be stated)."""
    out = []
    for el in _link_el(root, link).findall("collision"):
        g = el.find("geometry")
        kids = [c.tag for c in g] if g is not None else []
        if kids != ["capsule"]:
            raise ValueError(f"{link} <collision> shape {kids} — an arm link can only state capsules")
        c, o = g.find("capsule"), el.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        R = WG.rpy_matrix((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split())
        out.append(Capsule(link, float(c.get("radius")), float(c.get("length")), xyz, R))
    return out


def _check_rev(spec, rev, root, side: str) -> None:
    """The revision's link numbers are arm links, ``replace`` and ``add`` do not overlap, and every revised link's URDF
    joint parent is a link of the same arm side. The joint-neighbour pair is filtered only inside one articulation;
    the first arm link's parent is the torso, which the split (two-articulation) USD does not filter, so a hull there
    pushes the shoulder joint. Known limitation: until the env filters the first-arm-link ↔ torso pair in the split
    articulation (owner: the env self-collision filter), that link cannot join the revision."""
    ks = range(spec.index_base, spec.ee_body_index + 1)
    bad = [k for k in tuple(rev.replace) + tuple(rev.add) if k not in ks]
    if bad or set(rev.replace) & set(rev.add):
        raise ValueError(
            f"revision {rev.suffix}: link numbers {bad} lie outside the arm links {list(ks)} or replace · add overlap"
        )
    arm = {spec.body_name(side, k) for k in ks}
    parent = {j.find("child").get("link"): j.find("parent").get("link") for j in root.findall("joint")}
    out = [
        (ln, parent.get(ln))
        for ln in (spec.body_name(side, k) for k in tuple(rev.replace) + tuple(rev.add))
        if parent.get(ln) not in arm
    ]
    if out:
        raise ValueError(
            f"revision {rev.suffix}: the joint parent of a link lies outside the arm {out} — the split "
            "articulation does not filter that joint-neighbour pair, so a hull there pushes the joint. "
            "Add it to the revision only after the env filters the pair (env self-collision filter)"
        )


def bodies(arm: str, side: str, urdf) -> list:
    """Every collider the sim spawns on one arm side, link by link: [Capsule | Hull] (link frame)."""
    spec, rev = AR.ARM_REGISTRY[arm], AR.ARM_SIM_COLLIDER_REV.get(arm)
    urdf = Path(urdf)
    root = ET.parse(str(urdf)).getroot()
    if rev is not None:
        _check_rev(spec, rev, root, side)
    out: list = []
    for k in range(spec.index_base, spec.ee_body_index + 1):
        ln = spec.body_name(side, k)
        mode = None if rev is None else ("replace" if k in rev.replace else "add" if k in rev.add else None)
        caps = vendor_capsules(root, ln)  # stated even when replaced (the generator removes them)
        if mode != "replace":
            out += caps
        if mode is not None:
            verts, stats, meshes = WG.link_hull(urdf, ln, rev.hull_vertex_cap)
            out.append(Hull(ln, np.asarray(verts, float), mode, stats, meshes))
    return out


def origin_matrix(el) -> np.ndarray:
    """4×4 of a URDF element's ``<origin>`` (a joint · a visual · a collision); identity when it has none."""
    o, A = el.find("origin"), np.eye(4)
    A[:3, :3] = WG.rpy_matrix((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split())
    A[:3, 3] = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    return A


def fixed_frame(root, link: str, base: str | None = None) -> np.ndarray:
    """4×4 of a link in ``base``'s frame (None = the URDF root), reached through fixed joints only. A moving joint on
    the way dies — a torso hull sits on the static body the USD merges into ``world``, a palm on the last arm link."""
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    T, cur = np.eye(4), link
    while cur in by_child and cur != base:
        j = by_child[cur]
        if j.get("type") != "fixed":
            raise ValueError(
                f"{link}: joint {j.get('name')}({j.get('type')}) on the way to {base or 'the root'} is "
                "not fixed — cannot merge into one rigid body"
            )
        T, cur = origin_matrix(j) @ T, j.find("parent").get("link")
    if base is not None and cur != base:
        raise ValueError(f"{link}: not a descendant of {base}")
    return T


def fixed_tree(root, top: str, skip=frozenset()) -> list[str]:
    """``top`` and every link reached from it through fixed joints, never entering ``skip``."""
    kids: dict[str, list] = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    out, todo = [], [top]
    while todo:
        ln = todo.pop()
        out.append(ln)
        todo += [
            j.find("child").get("link")
            for j in kids.get(ln, [])
            if j.get("type") == "fixed" and j.find("child").get("link") not in skip
        ]
    return out


def _static_links(root) -> list[str]:
    """The static root body: the URDF root and every link reached from it through fixed joints, never entering an arm
    (a registered ArmSpec body moves with its arm whatever joint type a file declares)."""
    child = {j.find("child").get("link") for j in root.findall("joint")}
    tops = [el.get("name") for el in root.findall("link") if el.get("name") not in child]
    if len(tops) != 1:
        raise ValueError(f"the URDF does not have exactly one root link: {tops}")
    arms = {ln for a, sp in AR.ARM_REGISTRY.items() if sp is not None for s in sp.sides for ln in arm_link_names(a, s)}
    return fixed_tree(root, tops[0], arms)


def _check_torso(rev, root) -> None:
    """``torso_replace`` links carry a vendor collider, ``torso_add`` links carry none, and every vendor collider on the
    static root body is named in ``torso_replace`` (an unnamed one would be a torso shape nobody states)."""
    named = tuple(rev.torso_replace) + tuple(rev.torso_add)
    static = _static_links(root)
    has = {ln: bool(_link_el(root, ln).findall("collision")) for ln in static}
    bad = (
        [ln for ln in named if ln not in has]
        + [ln for ln in rev.torso_replace if not has.get(ln)]
        + [ln for ln in rev.torso_add if has.get(ln)]
        + [ln for ln, h in has.items() if h and ln not in rev.torso_replace]
    )
    if bad or len(set(named)) != len(named):
        raise ValueError(
            f"revision {rev.suffix}: the torso link statement disagrees with the static root body {bad} "
            f"— torso_replace = every static link with a vendor collider · torso_add = static links "
            f"without one (static links {static})"
        )


def torso_bodies(arm: str, urdf) -> list:
    """The torso colliders the sim spawns on the static root body: [Hull] in **root coordinates** (the USD ``world``
    body frame — the arm FK frame of the judge), one per part of every torso link under the decomposition rule
    (torso_split.torso_parts — part 0 alone = the link keeps one hull). [] for an arm with no revision; a revision is
    checked against the static body even when it names no torso link (a vendor torso collider it leaves unstated
    dies)."""
    rev = AR.ARM_SIM_COLLIDER_REV.get(arm)
    if rev is None:
        return []
    from dexengine.placement.collision import torso_split as TS

    urdf = Path(urdf)
    _check_torso(rev, ET.parse(str(urdf)).getroot())
    links = tuple(rev.torso_replace) + tuple(rev.torso_add)
    parts = TS.torso_parts(arm, urdf, links, rev.hull_vertex_cap)
    return [
        Hull(ln, np.asarray(v, float), "replace" if ln in rev.torso_replace else "add", st, ms, part=i)
        for ln in links
        for i, (v, st, ms) in enumerate(parts[ln])
    ]


def arm_of(urdf) -> str | None:
    """The registered arm this URDF carries — every body of exactly one arm present. None = no arm link at all (a
    floating hand). Some but not all bodies of an arm, or two arms, die: the arm shapes would be partly unstated."""
    names = {x.get("name") for x in ET.parse(str(urdf)).getroot().findall("link")}
    hit, part = [], []
    for arm, spec in AR.ARM_REGISTRY.items():
        if spec is None:
            continue
        want = {ln for s in spec.sides for ln in arm_link_names(arm, s)}
        (hit if want <= names else part if want & names else []).append(arm)
    if part or len(hit) > 1:
        raise ValueError(
            f"{Path(urdf).name}: cannot tell the arm — arms with every link {hit} · arms with some links {part}"
        )
    return hit[0] if hit else None


def _hull_stl(verts: np.ndarray, tag: str, mesh_dir: Path) -> str:
    """The hull as an STL in the mesh cache (keyed by its vertices). Its vertices are all extreme points
    (robots.wrist_hull.reduce_hull), so the hull MuJoCo takes of this mesh is exactly that vertex set."""
    import trimesh
    from scipy.spatial import ConvexHull

    v = np.ascontiguousarray(verts, dtype=np.float64)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    dst = mesh_dir / f"simhull_{tag}_{zlib.crc32(v.tobytes()) & 0xFFFFFFFF:08x}.stl"
    if not dst.is_file():
        m = trimesh.Trimesh(vertices=v, faces=ConvexHull(v).simplices)
        tmp = dst.with_suffix(f".{os.getpid()}.tmp.stl")
        m.export(str(tmp), file_type="stl")
        os.replace(tmp, dst)
    return str(dst)


def shape_dicts(urdf, mesh_dir) -> dict[str, list[dict]]:
    """Arm and torso link → its sim shapes in cross_sets' shape form (capsule: r · L · xyz · R; hull: an STL
    mesh + its vertices, link frame). Every arm link of the URDF's arm and every revision torso link is a key (an
    empty list = no collider); {} for a URDF with no arm."""
    urdf, mesh_dir = Path(urdf), Path(mesh_dir)
    arm = arm_of(urdf)
    if arm is None:
        return {}
    rev = AR.ARM_SIM_COLLIDER_REV.get(arm)
    out: dict[str, list[dict]] = {}
    said = []
    for side in AR.ARM_REGISTRY[arm].sides:
        out.update({ln: [] for ln in arm_link_names(arm, side)})
        for b in bodies(arm, side, urdf):
            if isinstance(b, Capsule):
                out[b.link].append({"kind": "capsule", "r": b.r, "L": b.L, "xyz": b.xyz, "R": b.R})
                continue
            out[b.link].append(
                {
                    "kind": "mesh",
                    "file": _hull_stl(b.verts, b.link, mesh_dir),
                    "xyz": np.zeros(3),
                    "R": np.eye(3),
                    "scale": [1, 1, 1],
                    "verts": b.verts,
                }
            )
            said.append(f"{b.link}:{b.mode}:{b.stats['vertices']}v:{b.stats['max_inward_mm']}mm")
    root = ET.parse(str(urdf)).getroot()
    for b in torso_bodies(arm, urdf):
        T = fixed_frame(root, b.link)  # cross_sets places links by FK: link-frame vertices
        v = (b.verts - T[:3, 3]) @ T[:3, :3]
        out.setdefault(b.link, []).append(
            {
                "kind": "mesh",
                "file": _hull_stl(v, b.link, mesh_dir),
                "xyz": np.zeros(3),
                "R": np.eye(3),
                "scale": [1, 1, 1],
                "verts": v,
            }
        )
        said.append(f"{b.name}:torso-{b.mode}:{b.stats['vertices']}v:{b.stats['max_inward_mm']}mm")
    head = f"rev={rev.suffix} cap={rev.hull_vertex_cap}" if rev else "rev=none"
    print(
        f"[XR_SIM_COLLIDER] arm={arm} {head} urdf={urdf.name} " + (" ".join(said) or "(URDF <collision>)"), flush=True
    )
    return out


def _digest_rows(arm: str, urdf) -> list[str]:
    """One text row per spawned shape (both sides) — the digest input. Hull vertices are sorted (order-free)."""
    rows = []
    for side in AR.ARM_REGISTRY[arm].sides:
        for b in bodies(arm, side, urdf):
            if isinstance(b, Capsule):
                p0, p1 = b.ends()
                nums = np.concatenate([[b.r], p0, p1])
                rows.append(f"{b.link}|capsule|" + ",".join(f"{x:.9f}" for x in nums))
            else:
                rows.append(_hull_row(b))
    return rows + [_hull_row(b) for b in torso_bodies(arm, urdf)]


def _hull_row(b) -> str:
    v = np.round(b.verts, 9)
    v = v[np.lexsort(v.T[::-1])]
    return f"{b.name}|hull|" + ",".join(f"{x:.9f}" for x in v.reshape(-1))


def registered_hands(arm: str) -> list[str]:
    """The hands registered with ``arm`` (paths.combo_ids — the mount registry), sorted."""
    return [h for a, h, _v in (P.split_combo(c, P.ARM_IDS, P.HAND_IDS) for c in P.combo_ids()) if a == arm]


_HANDS_SAID: set = set()


def arm_hands(arm: str) -> list[str]:
    """The hands of ``arm`` that every-hand code reads (digest, capsule fit, torso reach oracle, derived USD jobs, run
    inputs): the registered ones whose vendor source is set (``vendor.unavailable_hands``) and whose merged URDF is
    built (``combo_urdf``). A run needs only its own hand: a hand left out is never opened, and the reach statement it
    sets is used as stated (torso_split.reach_stated). A run whose own hand is not built still dies where it reads that
    hand's URDF (resolved_urdf). Marker ``[HANDS]`` once per arm and process, naming each hand left out and why."""
    from dexengine.robots import vendor  # noqa: PLC0415

    why = dict(vendor.unavailable_hands())
    reg = registered_hands(arm)
    for h in reg:
        if h not in why and not combo_urdf(arm, h).is_file():
            why[h] = f"NOT BUILT ({combo_urdf(arm, h).name} missing — vendor build --hands {h})"
    sel = [h for h in reg if h not in why]
    if arm not in _HANDS_SAID:
        _HANDS_SAID.add(arm)
        left = "".join(f" · not selected {h}: {why[h]}" for h in reg if h in why)
        print(f"[HANDS] arm={arm} selected {sel}{left}", flush=True)
    return sel


def digest(arm: str, urdfs=None) -> str:
    """sha256 (16 hex) of the arm shapes the sim spawns. ``urdfs`` default: every registered combo of the arm (canon
    body variant) — they must all state the same arm shapes, or this dies (the arm would differ by hand)."""
    if urdfs is None:
        urdfs = [combo_urdf(arm, h) for h in arm_hands(arm)]
    if not urdfs:
        raise ValueError(f"arm {arm}: no URDF to read the shapes from (0 registered combos)")
    seen: dict[str, list[str]] = {}
    for u in urdfs:
        h = hashlib.sha256("\n".join(_digest_rows(arm, u)).encode()).hexdigest()[:16]
        seen.setdefault(h, []).append(Path(u).name)
    if len(seen) != 1:
        raise ValueError(f"arm {arm}: the arm shapes differ by hand — {seen}")
    return next(iter(seen))


_MEASURED: dict = {}


def measured_digest(arm: str) -> str:
    """``digest(arm)`` measured by the retargeting python in a child process — the same answer in every interpreter
    (the reader's own may lack the mesh libraries). Readers: the IK pool sidecar head · the judge input fingerprint.
    The child is this module's CLI — the one 'digest now' entry (derived_usd.collider_digest_now runs the same CLI for
    one URDF), run with the checkout root as cwd so it imports this tree; its [COLLIDER_DIGEST] tree marker is passed on
    to stderr. Once per process per arm. A failed or malformed answer dies."""
    if arm not in _MEASURED:
        r = subprocess.run(
            [P.require_retarget_py(), "-m", "dexengine.placement.collision.sim_collider", "--digest", arm],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        last = (r.stdout.strip().splitlines() or [""])[-1]
        mark = [ln for ln in r.stderr.splitlines() if ln.startswith(_mark())]
        if r.returncode != 0 or not re.fullmatch(r"[0-9a-f]{16}", last) or not mark:
            raise RuntimeError(
                f"[sim_collider] digest({arm}) measurement failed rc={r.returncode} output={last!r} "
                f"markers {len(mark)} {(r.stderr.strip().splitlines() or [''])[-1]}"
            )  # last = the error
        print(mark[-1], file=sys.stderr, flush=True)
        _MEASURED[arm] = last
    return _MEASURED[arm]


def _mark() -> str:
    """The 'digest now' marker — owner robots.derived_usd.DIGEST_MARK (the name the spawn check reads)."""
    from dexengine.robots import derived_usd

    return derived_usd.DIGEST_MARK


def main(argv=None) -> int:
    """The one 'digest now' entry: stdout = the digest (last line) · stderr = ``[COLLIDER_DIGEST] <key> = <digest> ·
    tree=<code tree> commit=<HEAD>``."""
    ap = argparse.ArgumentParser(
        description="arm collider shape digest (every registered combo must state the same shapes)"
    )
    ap.add_argument("--digest", required=True, metavar="ARM", help="arm id — prints the 16-hex digest as the last line")
    ap.add_argument(
        "--urdf-of",
        nargs=2,
        metavar=("HAND", "BODY_VARIANT"),
        help="only the arm shapes of that combo URDF (derived USD check) — default: every registered combo of the arm",
    )
    a = ap.parse_args(argv)
    d = digest(a.digest, [combo_urdf(a.digest, *a.urdf_of)] if a.urdf_of else None)
    g = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    print(
        f"{_mark()} {'_'.join([a.digest, *(a.urdf_of or [])])} = {d} · tree={REPO_ROOT} "
        f"commit={g.stdout.strip() if g.returncode == 0 else '?(not git)'}",
        file=sys.stderr,
        flush=True,
    )
    print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
