#!/usr/bin/env python3
"""Torso decomposition rule — the one gate that picks, by measurement, which torso links leave one hull and into which
parts. sim_collider.torso_bodies builds the torso from ``torso_parts``; the judge (collision.torso), the collision sets,
the digest and the USD generator read those bodies, so every reader sees the same decomposition.

Ruler (the only one). A torso hull (a whole link or one part) reads the p90 of the nearest distance
to the torso visual surface (every torso link — the union) over its surface samples that
  1. face the arm side: outward normal mostly +x of the root;
  2. lie within the arm's reach (``arm_reach``) — a face no arm body can come to blocks no pose. The reach radius and
     the hand that sets it are a statement (``ARM_REACH``): the largest reach over every registered hand of the arm,
     so a run needs no vendor mesh of a hand it does not use; the oracle (``reach_oracle``) measures it again over the
     hands whose URDF meshes are present and dies when the statement is not what they give;
  3. lie outside the visual body of every torso link (a 6-view depth carve per link) — an excess inside a neighbour
     link's body blocks no pose the visual torso allows.
No such sample = 0. Fixed sampling seeds — the reading is deterministic.
Rule: a link whose one hull reads over TORSO_SPLIT_FRONT_P90_MM is decomposed — its visual mesh (link frame) is cut by
the plane through the midpoint of its longest extent, the worst part again, until every part reads within the rule.
A part = the hull of its piece of the mesh (cut triangles clipped: the parts cover the surface, and each lies inside
the link's one full hull) reduced to the vertex cap. Still over at TORSO_SPLIT_MAX_PARTS parts = death.
Marker ``[TORSO_SPLIT]`` once per input: the ruler as built (``_Ruler.desc`` — reach radius and the hand that sets it,
union link count) and every link's one-hull reading → its part count; ``[TORSO_REACH]`` once per arm: the statement and
what the oracle measured.
Gate ``torso_split_check``: the USD generator reads the bodies it is about to write and dies on one over the rule.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexengine.core import arms as AR
from dexengine.core import capacity
from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.placement.collision import geom_cache as GC
from dexengine.placement.collision import sim_collider as SC
from dexengine.robots import wrist_hull as WG

#: Torso decomposition rule: a torso hull may reach at most this far (mm, the p90 of the ruler above)
#: beyond the torso visual body on the arm-facing face within the arm's reach; a link over it is decomposed.
TORSO_SPLIT_FRONT_P90_MM = 20.0
TORSO_SPLIT_MAX_PARTS = 32  # a link still over the rule at this many parts dies
_SPLIT_PIX_M, _SPLIT_DENSITY = 0.002, 2.5e6  # carve pixel (m) · surface samples per m² (≈ 10 per pixel)
_PARTS: dict = {}  # input digest → {link: [(verts root, stats)]}
#: The arm reach statement (its one owner): arm → (reach radius m, the registered hand that sets it) = the largest
#: ``_chain_reach`` over every registered hand of the arm (rby1: dg5f_s 0.8957 · sharpa_wave 0.9148 · wuji2 0.8733 ·
#: xhand_pro 0.9155 m). The torso decomposition reads it; ``reach_oracle`` checks it wherever hand meshes are present.
ARM_REACH = {"rby1": (0.9154693820216969, "xhand_pro")}
REACH_ORACLE_TOL_M = 1e-9  # the oracle's agreement (m): the measurement is deterministic; this absorbs float rounding
_REACH: dict = {}  # arm → (reach radius m, the registered hand that sets it), as stated and checked
_SERVER_KEEP = ("_PARTS", "_REACH")  # clip-independent caches: whole input · arm (bake.server)


def _surface_samples(V, F, density: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform random points on triangles (V, F) and the triangle normals (outward for an oriented mesh)."""
    A = V[F]
    cr = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
    ti = np.repeat(np.arange(len(F)), rng.poisson(0.5 * np.linalg.norm(cr, axis=1) * density))
    u, w = rng.random(len(ti)), rng.random(len(ti))
    s = u + w > 1
    u[s], w[s] = 1 - u[s], 1 - w[s]
    X = A[ti, 0] + u[:, None] * (A[ti, 1] - A[ti, 0]) + w[:, None] * (A[ti, 2] - A[ti, 0])
    return X, cr[ti] / (np.linalg.norm(cr[ti], axis=1, keepdims=True) + 1e-18)


class _Carve:
    """6-view orthographic depth carve of surface samples Pv: each view keeps its first hit per pixel; a point is
    inside when no view sees it in front of that hit (+1 mm). Needs no closed mesh (hollow housings)."""

    def __init__(self, Pv: np.ndarray, pix: float):
        self.lo, self.pix, self.maps = Pv.min(0) - 2 * pix, pix, []
        for ax in range(3):
            o = [i for i in range(3) if i != ax]
            iv = tuple(((Pv[:, i] - self.lo[i]) / pix).astype(int) for i in o)
            for s in (1.0, -1.0):
                D = np.full(tuple(int(x.max()) + 2 for x in iv), -np.inf)
                np.maximum.at(D, iv, s * Pv[:, ax])
                self.maps.append((ax, o, s, D))

    def inside(self, Q: np.ndarray) -> np.ndarray:
        """Points Q (n, 3) inside the carved body (no view sees them in front of its first hit)."""
        ins = np.ones(len(Q), bool)
        for ax, o, s, D in self.maps:
            i, j = (np.floor((Q[:, k] - self.lo[k]) / self.pix).astype(int) for k in o)
            ok = (i >= 0) & (i < D.shape[0]) & (j >= 0) & (j < D.shape[1])
            d = np.full(len(Q), -np.inf)
            d[ok] = D[i[ok], j[ok]]
            ins &= s * Q[:, ax] <= d + 0.001  # -inf (no hit in that pixel) = outside
        return ins


class MeshAbsent(InputMissingError):
    """A URDF mesh the reach measurement cannot read (one line: the hand, the path, where the vendor stage goes).
    ``fn`` comes from the one URDF mesh resolver (P.urdf_mesh): an untracked hand stage resolves in the data root
    (P.main_root — env DEXENGINE_MAIN_ROOT in a worktree). ``reach_oracle`` counts such a hand as not measured; any
    other reader dies on it."""

    def __init__(self, hand: str, urdf: Path, fn: Path):
        try:
            root = P.main_root()
        except SystemExit as e:  # no data root — the text keeps its own reason
            root = f"(unreadable: {str(e)[:80]})"
        self.hand, self.path = hand, fn
        super().__init__(
            f"[TORSO_SPLIT] reach radius: hand {hand} mesh file missing: {fn} ({urdf.name}) · the vendor stage of hand"
            f" {hand} (HandSpec vendor_meshes) goes in the data root {root}"
        )


def _geom_radius(urdf: Path, el, cache: dict, hand: str) -> float:
    """Farthest point (m) from the link origin of one <visual>/<collision> element; a shape it cannot bound dies, a
    mesh file that is not there raises ``MeshAbsent`` naming the hand."""
    A, k = SC.origin_matrix(el), (list(el.find("geometry")) or [None])[0]
    tag = getattr(k, "tag", None)
    if tag == "mesh":
        fn, sc = P.urdf_mesh(urdf, k.get("filename")), k.get("scale", "1 1 1")
        if not fn.is_file():
            raise MeshAbsent(hand, urdf, fn)
        if (fn, sc) not in cache:
            cache[(fn, sc)] = GC.mesh_arrays(fn)[0] * [float(v) for v in sc.split()]
        return float(np.linalg.norm(cache[(fn, sc)] @ A[:3, :3].T + A[:3, 3], axis=1).max(initial=0.0))
    f = {
        "sphere": lambda: float(k.get("radius")),
        "capsule": lambda: 0.5 * float(k.get("length")) + float(k.get("radius")),
        "cylinder": lambda: float(np.hypot(0.5 * float(k.get("length")), float(k.get("radius")))),
        "box": lambda: 0.5 * float(np.linalg.norm([float(v) for v in k.get("size").split()])),
    }.get(tag)
    if f is None:
        raise InputMissingError(f"[TORSO_SPLIT] {urdf.name}: shape {tag} — cannot bound its reach")
    return float(np.linalg.norm(A[:3, 3])) + f()


def _chain_reach(arm: str, hand: str) -> float:
    """Upper bound (m) of how far any body below a shoulder of the (arm, hand) URDF reaches from it: max over the links
    below the first arm link of (Σ |joint origin offset| on the path + prismatic travel + the link's farthest
    <visual>/<collision> point). A revolute joint turns its child about an axis through the child origin, so this
    bounds every pose."""
    spec, urdf = AR.ARM_REGISTRY[arm], SC.combo_urdf(arm, hand)
    if not urdf.is_file():
        raise InputMissingError(f"[TORSO_SPLIT] {urdf} missing — build it (python -m dexengine.robots.vendor build)")
    root = ET.parse(str(urdf)).getroot()
    kids: dict[str, list] = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    els, cache, best = {x.get("name"): x for x in root.findall("link")}, {}, 0.0
    todo = [(spec.body_name(side, spec.index_base), 0.0) for side in spec.sides]
    while todo:
        ln, L = todo.pop()
        best = max(
            [best] + [L + _geom_radius(urdf, e, cache, hand) for e in els[ln] if e.tag in ("visual", "collision")]
        )
        for j in kids.get(ln, []):
            if j.get("type") not in ("fixed", "revolute", "continuous", "prismatic"):
                raise InputMissingError(
                    f"[TORSO_SPLIT] {urdf.name}: joint {j.get('name')}({j.get('type')}) — no reach bound"
                )
            lim, move = j.find("limit"), 0.0
            if j.get("type") == "prismatic":
                move = max(abs(float(lim.get("lower"))), abs(float(lim.get("upper"))))
            todo.append((j.find("child").get("link"), L + float(np.linalg.norm(SC.origin_matrix(j)[:3, 3])) + move))
    return best


def arm_reach(arm: str, urdf) -> tuple[np.ndarray, float, str]:
    """(shoulders (sides, 3) in this URDF's root coordinates, reach radius m, the hand that sets it). Shoulder = the
    origin of a side's first arm link (fixed whatever the arm joints do). Radius = the statement ``ARM_REACH`` (the
    largest ``_chain_reach`` over every registered hand of the arm — the torso is one shape for all hands,
    sim_collider.digest, so it is cut for the farthest of them; a hand that reaches farther changes the torso and its
    digest), checked by ``reach_stated``."""
    spec, root = AR.ARM_REGISTRY[arm], ET.parse(str(urdf)).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    S = [
        (SC.fixed_frame(root, by_child[a0].find("parent").get("link")) @ SC.origin_matrix(by_child[a0]))[:3, 3]
        for a0 in (spec.body_name(side, spec.index_base) for side in spec.sides)
    ]
    if arm not in _REACH:
        _REACH[arm] = reach_stated(arm)
    return np.asarray(S, float), *_REACH[arm]


def reach_oracle(arm: str, hands: list) -> tuple[dict, dict]:
    """({hand: ``_chain_reach`` m} of the hands whose URDF meshes are all present, {hand: the first missing mesh} of
    the others)."""
    read = {h: _reach_or_absent(arm, h) for h in hands}
    return {h: v for h, v in read.items() if isinstance(v, float)}, {
        h: v for h, v in read.items() if isinstance(v, str)
    }


def _reach_or_absent(arm: str, hand: str) -> float | str:
    """``_chain_reach`` of one hand (m), or the path of its first missing mesh."""
    try:
        return _chain_reach(arm, hand)
    except MeshAbsent as e:
        return str(e.path)


def reach_stated(arm: str) -> tuple[float, str]:
    """(reach radius m, the hand that sets it) of ``arm``: the statement ``ARM_REACH``, checked by the oracle — no
    measured hand reaches farther, and the stated hand, when measured, reaches exactly that far (REACH_ORACLE_TOL_M);
    either failing dies (the statement is stale: measure every hand and state the largest). An arm without a statement,
    or a stated hand that is not registered, dies. The oracle measures the selected hands (``SC.arm_hands``: a hand
    whose vendor source is not set is never opened); when the stated hand is such a hand, the statement is used as
    stated and the marker says it could not be verified. Marker ``[TORSO_REACH]``."""
    hands, registered = SC.arm_hands(arm), SC.registered_hands(arm)
    if not hands:
        raise InputMissingError(f"[TORSO_SPLIT] arm {arm}: 0 registered combos — no hand to measure the reach with")
    if arm not in ARM_REACH:
        raise InputMissingError(
            f"[TORSO_SPLIT] arm {arm}: no reach statement (torso_split.ARM_REACH) — state the largest reach_oracle"
            f" value over its registered hands {hands}"
        )
    r, h = ARM_REACH[arm]
    if h not in registered:
        raise ValueError(f"[TORSO_SPLIT] arm {arm}: the stated reach hand {h} is not registered ({registered})")
    got, absent = reach_oracle(arm, hands)
    over = {k: v for k, v in got.items() if v > r + REACH_ORACLE_TOL_M}
    if over or (h in got and abs(got[h] - r) > REACH_ORACLE_TOL_M):
        raise ValueError(
            f"[TORSO_SPLIT] arm {arm}: reach statement {r!r} m ({h}) is not what the hands measure {got} — the"
            " statement is stale: state the largest reach over every registered hand (torso_split.ARM_REACH)"
        )
    print(
        f"[TORSO_REACH] {arm}: stated {r * 1e3:.1f} mm ({h}) · oracle measured {len(got)}/{len(hands)} hands"
        + "".join(f" · {k} {v * 1e3:.1f} mm" for k, v in sorted(got.items()))
        + (f" · not measured (meshes absent) {sorted(absent)}" if absent else "")
        + (
            f" · stated only, not verifiable: {h} is not selected (source not set or not built — [HANDS])"
            if h not in hands
            else ""
        ),
        flush=True,
    )
    return r, h


class _Ruler:
    """The decomposition ruler over the torso links of one URDF (module doc). ``read(verts)`` → (p90 mm, counted
    samples)."""

    def __init__(self, arm: str, urdf: Path, links):
        from scipy.spatial import cKDTree

        root = ET.parse(str(urdf)).getroot()
        self.S, self.R, self.hand = arm_reach(arm, urdf)
        Pv = []
        for ln in links:
            T, (V, F, _) = SC.fixed_frame(root, ln), WG.visual_mesh(urdf, ln)
            Pv.append(_surface_samples(V @ T[:3, :3].T + T[:3, 3], F, _SPLIT_DENSITY, np.random.default_rng(0))[0])
        self.carves, self.kd = [_Carve(p, _SPLIT_PIX_M) for p in Pv], cKDTree(np.concatenate(Pv))

    @property
    def desc(self) -> str:
        """The ruler as it measures now (the markers print this, not a fixed phrase)."""
        return f"front +x · reach {self.R * 1e3:.0f} mm({self.hand}) · union {len(self.carves)} links"

    def samples(self, verts: np.ndarray) -> tuple[np.ndarray, dict]:
        """The counted samples of a hull (root coordinates) and how many pass each filter in turn."""
        Ph, Nh = _surface_samples(verts, WG.hull_triangles(verts), _SPLIT_DENSITY, np.random.default_rng(0))
        Q = Ph[(np.argmax(np.abs(Nh), 1) == 0) & (Nh[:, 0] > 0)]
        n = {"front": len(Q)}
        Q = Q[np.linalg.norm(Q[:, None, :] - self.S[None], axis=2).min(1) <= self.R]
        n["reach"] = len(Q)
        for c in self.carves:
            Q = Q[~c.inside(Q)]
        n["outside"] = len(Q)
        return Q, n

    def read(self, verts: np.ndarray) -> tuple[float, int]:
        """(p90 mm of the counted samples' distance to the visual torso, their count); (0, 0) with none."""
        Q, _n = self.samples(verts)
        if not len(Q):
            return 0.0, 0
        return round(float(np.percentile(self.kd.query(Q, workers=capacity.threads_here())[0], 90)) * 1e3, 1), len(Q)


def _pieces(V: np.ndarray, F: np.ndarray, ax: int, c: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """The mesh cut by the plane x[ax] = c: its two sides, cut triangles clipped. Arrays only (``slice_faces_plane``) —
    the digest is measured in the spawn interpreter too (derived_usd), which has trimesh but not its path/polygon
    dependencies (shapely)."""
    from trimesh.intersections import slice_faces_plane

    return [
        tuple(
            np.asarray(a, t)
            for a, t in zip(
                slice_faces_plane(V, F, s * np.eye(3)[ax], np.eye(3)[ax] * c)[:2], (float, np.int64), strict=True
            )
        )
        for s in (-1.0, 1.0)
    ]


def _decompose(ruler: _Ruler, V: np.ndarray, F: np.ndarray, T: np.ndarray, cap: int, link: str) -> tuple[list, float]:
    """([(hull verts in root coordinates, stats)], one-hull reading mm) of one link (V, F in its frame, T link →
    root): the one hull when it reads within the rule, else the parts (module doc)."""

    def hull(pts):
        v, st = WG.reduce_hull(pts, cap)
        v = v @ T[:3, :3].T + T[:3, 3]
        p90, n = ruler.read(v)
        return v, dict(st, split_p90_mm=p90, split_samples=n)

    whole = hull(V)
    parts = [(V, F, *whole)]
    while over := [i for i, p in enumerate(parts) if p[3]["split_p90_mm"] > TORSO_SPLIT_FRONT_P90_MM]:
        if len(parts) >= TORSO_SPLIT_MAX_PARTS:
            raise SystemExit(
                f"[TORSO_SPLIT] {link}: still over the rule {TORSO_SPLIT_FRONT_P90_MM} mm at "
                f"{len(parts)} parts {[parts[i][3]['split_p90_mm'] for i in over]}"
            )
        i = max(over, key=lambda k: parts[k][3]["split_p90_mm"])
        pts = parts[i][0][np.unique(parts[i][1])]
        ax = int(np.argmax(pts.max(0) - pts.min(0)))
        cut = _pieces(parts[i][0], parts[i][1], ax, 0.5 * (pts[:, ax].min() + pts[:, ax].max()))
        parts[i : i + 1] = [(v, f, *hull(v[np.unique(f)])) for v, f in cut]
    return [(v, st) for _V, _F, v, st in parts], whole[1]["split_p90_mm"]


def torso_parts(arm: str, urdf, links, cap: int) -> dict[str, list]:
    """Torso link → its colliders [(hull verts in root coordinates, stats, meshes)] under the rule (one entry = the
    link keeps one hull). Memoised by the whole input (torso visual meshes and frames, shoulders, reach and the hand
    that sets it, cap) — every hand's URDF carries the same torso, so a process cuts it once — and kept on disk under
    the same input key + this code (geom_cache), so a new process reads the cut instead of re-running the ruler.
    Marker ``[TORSO_SPLIT]`` once per input (the same line on a disk read)."""
    urdf, root = Path(urdf), ET.parse(str(urdf)).getroot()
    S, R, _hand = arm_reach(arm, urdf)
    mesh = {ln: (WG.visual_mesh(urdf, ln), SC.fixed_frame(root, ln)) for ln in links}
    h = hashlib.sha256(repr((tuple(links), cap, TORSO_SPLIT_FRONT_P90_MM, TORSO_SPLIT_MAX_PARTS, R)).encode())
    for (V, F, _m), T in mesh.values():
        for x in (V, F, T, S):
            h.update(np.ascontiguousarray(x).tobytes())
    key = h.hexdigest()
    if key not in _PARTS:

        def _cut():
            ruler, out, whole = _Ruler(arm, urdf, links), {}, {}
            for ln, ((V, F, _m), T) in mesh.items():
                out[ln], whole[ln] = _decompose(ruler, V, F, T, cap, ln)
            return out, (
                f"[TORSO_SPLIT] rule p90 ≤ {TORSO_SPLIT_FRONT_P90_MM} mm · {ruler.desc} · "
                + " ".join(f"{ln.removeprefix('link_')}={whole[ln]}→{len(out[ln])}" for ln in links)
                + f" · input {key[:12]}"
            )

        _PARTS[key], line = GC.cached("torso_parts", GC.code_rev((__file__, WG.__file__)), (key, _hand), _cut)
        print(line, flush=True)
    return {ln: [(v, st, mesh[ln][0][2]) for v, st in _PARTS[key][ln]] for ln in links}


def torso_split_check(arm: str, urdf, bodies) -> dict[str, float]:
    """The gate on what ships: every torso body (sim_collider.Hull — a whole link or a part) reads within the rule.
    Marker ``[TORSO_SPLIT] gate`` with every reading; dies when one is over (that link must be decomposed)."""
    links = list(dict.fromkeys(b.link for b in bodies))
    ruler = _Ruler(arm, Path(urdf), links) if links else None
    m = {b.name: ruler.read(np.asarray(b.verts, float))[0] for b in bodies}
    over = {k: v for k, v in m.items() if v > TORSO_SPLIT_FRONT_P90_MM}
    print(
        f"[TORSO_SPLIT] gate ≤ {TORSO_SPLIT_FRONT_P90_MM} mm · {ruler.desc if ruler else 'no ruler'} · "
        + (" ".join(f"{k}={v}" for k, v in m.items()) or "no torso colliders (0 torso links in the revision)"),
        flush=True,
    )
    if over:
        raise ValueError(
            f"[TORSO_SPLIT] torso colliders over the rule {TORSO_SPLIT_FRONT_P90_MM} mm {over} — that "
            "link is decomposed, not kept as one hull (torso_parts)"
        )
    return m
