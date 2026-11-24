#!/usr/bin/env python3
"""Collision proxy geometry of the **search filters** — single owner of which shape the torso · desk filters (the
judge's pruning) measure for the arm revision hulls and the hand bodies (read by arm_surface · torso).

COLL_PROXY = "capsule" is the filter proxy only; every ruler that decides adoption or repairs geometry is the original
mesh (GUARD_KIND: collision.core.side_guard · cross_guard = judge = bake = gate), run inside ``mesh_ruler()``. Marker
``[COLL_PROXY] filter=capsule-q40 guard=mesh``.
  capsule  every arm revision hull and finger body as a capsule from TABLE, the palm as surface points:
             desk   exact depth below the desk top of the lowest point of (capsule ∩ footprint prism);
             torso  the torso's signed-distance grid on the centre line − r (palm points − 0), KD shell GRID_SHELL_M.
  mesh     the original rulers on the sim collider hulls.
Explicit, no fallback: an unknown kind dies at import; in capsule mode a body missing from the table or whose hull
vertices differ from the fitted ones dies (rebake: ``capsule_fit --write``) — never the mesh in its place.

Fit (capsule_fit, the generator): axis candidates = the URDF child-joint direction (offset > JOINT_AXIS_MIN_M) + the 3
principal axes of the hull's surface samples; the line passes through the samples' centroid; per candidate the two ends
(FIT_GRID grid over the vertices' axial extent) and the radius (median of the support gap) minimise the mean
|support-function error| over FIT_DIRS directions ("l1" — bias 0); the candidate with the least error wins. The radius
is then set to the RADIUS_QUANTILE quantile of that capsule's support gap (RADIUS_RULE "q40": the capsule lies inside
the hull in 60 % of the directions). Why q40, not l1 (q50): l1 leaned strict against the mesh on the judge and on the
IK pool (torso and fingers); q40 balances torso and fingers and eases the desk, so the inflated
capsules do not reject more than the mesh would. Rejected: "max" (strict only) · "mean" (lenient).

Table (``TABLE``, generated — never edited by hand): one entry per body, keyed ``arm/<arm>/<link>`` ·
``hand/<hand>/<side>/<group root>``, with the digest of the hull vertices it was fitted to (``src``). A hand whose
bodies are not in the base table lives in one append file ``<table stem>_<hand>.json`` (same schema, same fit);
``entry`` picks the base table or the append file by the key's hand (a key in both dies). Identity (``identity()``) =
kind + radius rule + the table file's hash — the judge input fingerprint's collider field and the desk ruler identity
(collision.core.ruler_sources → the IK-pool clearance sidecar head) carry it, so a re-fitted table is a different ruler;
an append hand adds ``+<hand>-<append hash>`` to the judge identity only (``_filter_id``). Marker: ``[COLL_PROXY]
filter=<kind> guard=<kind> id=<identity> ruler=<kind> …`` (``marker``) once per body set.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.repo import ROBOT_ASSETS_DIR

#: The search-filter proxy geometry kind (decision above) — "capsule" | "mesh".
COLL_PROXY = "capsule"
#: The deciding / repairing rulers' geometry (decision above) — never the proxy.
GUARD_KIND = "mesh"
KINDS = ("capsule", "mesh")
if COLL_PROXY not in KINDS:
    raise InputMissingError(f"[COLL_PROXY] ✗ COLL_PROXY={COLL_PROXY!r} — known kinds are {KINDS} (no fallback)")
#: Radius rule of a fitted capsule (module doc): the radius = this quantile of the support gap (ends by l1).
RADIUS_QUANTILE = 0.40
RADIUS_RULE = f"q{round(RADIUS_QUANTILE * 100)}"
FIT_DIRS = 2000  # fibonacci directions of the support-function error
FIT_GRID = 41  # end-parameter grid per axis candidate (over the vertices' axial extent)
FIT_SAMPLE_M = 0.004  # hull surface-sample spacing: centroid · principal axes
JOINT_AXIS_MIN_M = 0.05  # a child joint this far from the link origin makes its direction an axis candidate
PALM_SAMPLE_M = 0.012  # palm surface-sample spacing (capsule mode: the palm is its points on the torso grid)
#: Torso grid in capsule mode: KD-distance shell and box margin around the torso hulls — the centre line reads the grid
#: at up to radius (≤ 0.07 m) + the filter's reach (torso.SKIP_M), so the exact-distance shell must reach past it.
GRID_SHELL_M = 0.11
GRID_MARGIN_M = 0.12
#: Capsule mode torch torso filter (torso_torch.torso_filter_sd_t): poses per block of a capsule / point body =
#: PTS_BLOCK points ÷ the body's points, at most POSE_BLOCK_MAX (the batch its cuBLAS products were measured bit-stable
#: at), at least the caller's CHUNK (no poses × torso vertices array); mesh hulls keep CHUNK.
PTS_BLOCK = 1 << 21
POSE_BLOCK_MAX = 16384


def pose_block(npts: int, floor: int) -> int:
    """Poses per block of a capsule / point body with npts points (constants above)."""
    return max(floor, min(POSE_BLOCK_MAX, PTS_BLOCK // max(npts, 1)))


CHECK_TOL_M = 1e-6  # --check: largest allowed difference of a regenerated number (BLAS rounding across machines)
TABLE = ROBOT_ASSETS_DIR / "collision/coll_proxy_capsules.json"
SCHEMA = "coll_proxy_capsules/1"
#: How to rebake the table (the die messages print it): the retargeting python (paths.require_retarget_py — the mesh
#: stack) runs the generator from the checkout root.
REBAKE = "<retarget python> -m dexengine.placement.collision.capsule_fit --write"
_STATE: dict = {}
_SERVER_KEEP = ("_STATE",)  # the loaded table · its hash · the said markers (clip-independent)


def kind() -> str:
    """Read at call time: GUARD_KIND inside mesh_ruler(), else COLL_PROXY."""
    if COLL_PROXY not in KINDS:
        raise InputMissingError(f"[COLL_PROXY] ✗ COLL_PROXY={COLL_PROXY!r} — known kinds are {KINDS} (no fallback)")
    return GUARD_KIND if _STATE.get("guard", 0) else COLL_PROXY


@contextlib.contextmanager
def mesh_ruler():
    """A deciding / repairing ruler's scope (process-wide: the judge is one thread)."""
    _STATE["guard"] = _STATE.get("guard", 0) + 1
    try:
        yield
    finally:
        _STATE["guard"] -= 1


def guarding() -> bool:
    """Inside mesh_ruler()."""
    return bool(_STATE.get("guard", 0))


def capsule() -> bool:
    """The current ruler is the capsule proxy (kind() == "capsule")."""
    return kind() == "capsule"


def table_hash() -> str:
    """sha256 (16 hex) of the table file — its identity (the committed bytes, not a regeneration)."""
    if ("hash", str(TABLE)) not in _STATE:
        if not TABLE.is_file():
            raise InputMissingError(f"[COLL_PROXY] ✗ capsule table missing: {TABLE} — bake it: {REBAKE} (no fallback)")
        _STATE[("hash", str(TABLE))] = hashlib.sha256(TABLE.read_bytes()).hexdigest()[:16]
    return _STATE[("hash", str(TABLE))]


def identity() -> str:
    """'mesh' | 'capsule-<rule>-<table hash>' — the filter's (the desk ruler identity of the IK-pool sidecar)."""
    return "mesh" if not capsule() else f"capsule-{RADIUS_RULE}-{table_hash()}"


def _filter_id() -> str:
    return "mesh" if COLL_PROXY == "mesh" else f"capsule-{RADIUS_RULE}-{table_hash()}{hand_suffix()}"


def judge_identity() -> str:
    """The judge input fingerprint's collider suffix '<filter id>+guard-<GUARD_KIND>' (both rulers shape it)."""
    return f"{_filter_id()}+guard-{GUARD_KIND}"


def marker() -> str:
    """The [COLL_PROXY] marker head (filter · guard · identity · the current ruler)."""
    f = "mesh" if COLL_PROXY == "mesh" else f"capsule-{RADIUS_RULE}"
    return f"[COLL_PROXY] filter={f} guard={GUARD_KIND} id={_filter_id()} ruler={kind()}"


def ruler_sources() -> list[str]:
    """What this module adds to the desk ruler identity (collision.core.ruler_sources): the kind, and in capsule mode
    the table's hash, this file and the generator that fits the table (capsule_fit)."""
    if not capsule():
        return ["COLL_PROXY = 'mesh'"]
    from dexengine.placement.collision import capsule_fit as CF

    return [
        f"COLL_PROXY = {identity()!r}",
        Path(__file__).read_text(encoding="utf-8"),
        Path(CF.__file__).read_text(encoding="utf-8"),
    ]


# ── the table: base file + one append file per hand ──


def append_table(table: Path, hand: str) -> Path:
    """Where one hand's append table lives (used only when it exists)."""
    return table.with_name(f"{table.stem}_{hand}.json")


def file_hash(p: Path, cache: dict) -> str:
    """sha256 (16 hex) of a table file — the committed bytes are its identity (not a rebake)."""
    if ("hash", str(p)) not in cache:
        cache[("hash", str(p))] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return cache[("hash", str(p))]


def hand_suffix() -> str:
    """The append field of the judge identity — `+<hand>-<hash>` when this process's hand (env DEXENGINE_HAND) has an
    append table, else '' (a base-table hand · a process without a hand keep the plain identity)."""
    hand = os.environ.get("DEXENGINE_HAND", "").strip()
    p = append_table(TABLE, hand) if hand else None
    return f"+{hand}-{file_hash(p, _STATE)}" if p is not None and p.is_file() else ""


def _load(p: Path) -> dict:
    """One table file (cached); a header (schema, radius_rule) other than this module's dies."""
    key = ("table", str(p))
    if key not in _STATE:
        t = json.loads(p.read_text(encoding="utf-8"))
        if (t.get("schema"), t.get("radius_rule")) != (SCHEMA, RADIUS_RULE):
            raise SystemExit(
                f"[COLL_PROXY] ✗ capsule table {p.name}: schema {t.get('schema')!r} · radius rule "
                f"{t.get('radius_rule')!r} ≠ {SCHEMA!r} · {RADIUS_RULE!r} — rebake: {REBAKE}"
            )
        _STATE[key] = t
    return _STATE[key]


def _table() -> dict:
    table_hash()
    return _load(TABLE)


def entry(key: str) -> dict | None:
    """The one read: a body key's entry — a hand key from the base table or that hand's append table; in both dies."""
    e = _table()["bodies"].get(key)
    parts = key.split("/")
    if parts[0] == "hand" and len(parts) > 1 and append_table(TABLE, parts[1]).is_file():
        p = append_table(TABLE, parts[1])
        a = _load(p)["bodies"].get(key)
        if e is not None and a is not None:
            raise SystemExit(f"[COLL_PROXY] ✗ {key}: in both the base table and the append table {p.name}")
        e = a if e is None else e
    return e


def say(key, msg: str) -> None:
    """Print msg once per key and process."""
    if ("said", key) not in _STATE:
        _STATE[("said", key)] = True
        print(msg, flush=True)


def src_digest(v) -> str:
    """Digest of a hull's vertices (order-free, 9 decimals — sim_collider's hull row convention)."""
    import numpy as np

    v = np.round(np.asarray(v, float), 9)
    v = v[np.lexsort(v.T[::-1])]
    return hashlib.sha256(",".join(f"{x:.9f}" for x in v.reshape(-1)).encode()).hexdigest()[:16]


def frame(d):
    """Rotation whose z column is the unit axis d (the capsule dict's R — capsule_endpoints reads R·ez)."""
    import numpy as np

    d = np.asarray(d, float) / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, d)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(d, x), d], 1)


def lookup(key: str, v) -> dict:
    """The capsule of body ``key`` fitted to hull vertices v: {p, R, r, L, axis} (link / group frame). Missing or stale
    (the hull's vertices are not the ones the table was fitted to) dies — no mesh in its place."""
    import numpy as np

    e = entry(key)
    got = src_digest(v)
    if e is None or e["src"] != got:
        why = "not in the table" if e is None else f"vertex digest {got} ≠ table {e['src']} (the shape changed)"
        raise SystemExit(
            f"[COLL_PROXY] ✗ {key}: {why} — rebake the capsule table: {REBAKE} (table {TABLE.name} · "
            "never replaced by the original mesh)"
        )
    return {
        "p": np.asarray(e["p"], float),
        "R": frame(e["d"]),
        "r": float(e["r"]),
        "L": float(e["L"]),
        "axis": e["axis"],
    }


def finger_cap(hand: str, side: str, group: dict) -> dict:
    """The table capsule of one finger group of hand_surface.hand_tree (link = its root)."""
    return dict(lookup(hand_key(hand, side, group["root"]), group["verts"]), link=group["root"])


def arm_key(arm: str, link: str) -> str:
    """Table key of an arm revision hull."""
    return f"arm/{arm}/{link}"


def hand_key(hand: str, side: str, root: str) -> str:
    """Table key of a finger group (its root link)."""
    return f"hand/{hand}/{side}/{root}"


def grid_shell(mesh_shell_m: float, mesh_margin_m: float) -> tuple[float, float]:
    """(KD shell, box margin) of the torso grid — the mesh values, or GRID_SHELL_M · GRID_MARGIN_M for capsules."""
    return (GRID_SHELL_M, GRID_MARGIN_M) if capsule() else (mesh_shell_m, mesh_margin_m)


def palm_body(link: str, k: int, v, mask: tuple) -> dict:
    """Capsule mode: the palm hull as a torso body of its surface points (vertices + PALM_SAMPLE_M samples,
    read on the torso grid, r 0; the bounding sphere = the hull's)."""
    import numpy as np

    from dexengine.placement.collision.arm_surface import hull_samples
    from dexengine.robots import wrist_hull as G

    v = np.asarray(v, float)
    c = v.mean(0)
    return {
        "kind": "pts",
        "link": link,
        "k": k,
        "mask": mask,
        "pts": hull_samples(v, G.hull_triangles(v), PALM_SAMPLE_M),
        "r": 0.0,
        "ctr": c,
        "rad": float(np.linalg.norm(v - c, axis=1).max()),
    }


def mark_torso(where: str, bodies: list, fingers: list) -> None:
    """The torso half of the marker (torso.build_torso) — once per combo/side."""
    if capsule():
        rs = [b["r"] * 1e3 for b in fingers] or [0.0]
        what = (
            f"palm points {sum(len(b['pts']) for b in bodies if b['kind'] == 'pts')} · fingers {len(fingers)} → "
            f"capsules (r {min(rs):.1f}–{max(rs):.1f} mm) · search filter = grid (shell {GRID_SHELL_M * 1e3:.0f} mm"
            f" · margin {GRID_MARGIN_M * 1e3:.0f} mm) · guard = original mesh"
        )
    else:
        what = "palm · finger hulls as they are (original mesh) · guard = exact KD"
    say(("torso", where, kind()), f"{marker()} · {where} torso ruler: " + what)


# ── desk: exact capsule ∩ footprint prism ──


def _sort(xp, x):
    s = xp.sort(x, -1)
    return getattr(s, "values", s)


def _roots(xp, al, be, ga, one):
    """Both roots of al·s² + be·s + ga = 0 (numerically stable form; a negative discriminant reads as 0 = the vertex;
    al ≈ 0 → the linear root twice; no root → NaN). Extra or inexact roots are harmless to the caller (candidates)."""
    zero = one * 0.0
    quad = xp.abs(al) > 1e-18
    lin_ok = xp.abs(be) > 1e-18
    lin = xp.where(lin_ok, -ga / xp.where(lin_ok, be, one), zero * float("nan"))
    sq = xp.sqrt(xp.maximum(be * be - 4.0 * al * ga, zero))
    qq = -0.5 * (be + xp.where(be >= 0, sq, -sq))
    q_ok = xp.abs(qq) > 1e-30
    r1 = xp.where(quad, qq / xp.where(quad, al, one), lin)
    r2 = xp.where(quad & q_ok, ga / xp.where(q_ok, qq, one), r1)
    return r1, r2


def capsule_desk_pen(xp, a, b, r, ztop: float, box):
    """Depth (m, positive = below the desk top) of the lowest point of (capsule ∩ footprint prism), exact — xp = numpy |
    torch (one formula for both twins). a, b (…, 3) world axis ends · r broadcastable to (…) · box ((x0, x1), (y0, y1)).
    −1e9 where the capsule misses the prism.

    With p(s) = a + s(b − a), s ∈ [0, 1], and d(s) the xy distance of p(s) to the footprint, the lowest capsule point
    over the prism at s is p_z(s) − √(r² − d(s)²) (where d ≤ r) — a convex function of s on the interval where d ≤ r.
    Its minimum sits at a piece end (the axis crossing a footprint wall line, or s = 0, 1), where d(s) = r, or where the
    derivative vanishes; on each piece d² is a quadratic in s, so all three are roots of quadratics — every candidate
    is evaluated with the true distance, an infeasible one (d > r) never counts."""
    (x0, x1), (y0, y1) = box
    v = b - a
    zero = a[..., 0] * 0.0
    one = zero + 1.0
    r2 = (zero + r) ** 2

    def wall(p0, dv, val):
        ok = xp.abs(dv) > 1e-15
        s = (val - p0) / xp.where(ok, dv, one)
        return xp.where(ok, xp.minimum(xp.maximum(s, zero), one), zero)

    bp = _sort(
        xp,
        xp.stack(
            [
                zero,
                one,
                wall(a[..., 0], v[..., 0], x0),
                wall(a[..., 0], v[..., 0], x1),
                wall(a[..., 1], v[..., 1], y0),
                wall(a[..., 1], v[..., 1], y1),
            ],
            -1,
        ),
    )
    lo, hi = bp[..., :-1], bp[..., 1:]  # (…, 5) pieces
    z5, o5 = lo * 0.0, lo * 0.0 + 1.0
    ax, ay = a[..., 0:1], a[..., 1:2]
    vx, vy, vz = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    m = 0.5 * (lo + hi)
    px, py = ax + m * vx, ay + m * vy
    cx0 = xp.where(px < x0, x0 - ax + z5, xp.where(px > x1, ax - x1 + z5, z5))  # d_x(s) = cx0 + cx1·s on the piece
    cx1 = xp.where(px < x0, -vx + z5, xp.where(px > x1, vx + z5, z5))
    cy0 = xp.where(py < y0, y0 - ay + z5, xp.where(py > y1, ay - y1 + z5, z5))
    cy1 = xp.where(py < y0, -vy + z5, xp.where(py > y1, vy + z5, z5))
    A, B, C = cx1 * cx1 + cy1 * cy1, 2.0 * (cx0 * cx1 + cy0 * cy1), cx0 * cx0 + cy0 * cy0
    R2, w = r2[..., None] + z5, vz * vz + z5
    s1, s2 = _roots(xp, A, B, C - R2, o5)  # d(s) = r
    s3, s4 = _roots(xp, A * (A + w), B * (A + w), 0.25 * B * B - w * (R2 - C), o5)  # stationary (squared)
    cand = xp.stack([lo, hi, s1, s2, s3, s4], -1)  # (…, 5, 6)
    cand = xp.where(xp.isnan(cand), lo[..., None], cand)
    cand = xp.minimum(xp.maximum(cand, lo[..., None]), hi[..., None])
    cx, cy = a[..., None, None, 0] + cand * v[..., None, None, 0], a[..., None, None, 1] + cand * v[..., None, None, 1]
    dx = xp.maximum(xp.maximum(x0 - cx, cx - x1), cand * 0.0)
    dy = xp.maximum(xp.maximum(y0 - cy, cy - y1), cand * 0.0)
    q = dx * dx + dy * dy
    rr = r2[..., None, None] + cand * 0.0
    f = a[..., None, None, 2] + cand * v[..., None, None, 2] - xp.sqrt(xp.maximum(rr - q, cand * 0.0))
    f = xp.where(q <= rr * (1.0 + 1e-12) + 1e-18, f, cand * 0.0 + 1e9)
    zmin = xp.amin(xp.amin(f, -1), -1)
    return xp.where(zmin < 1e8, ztop - zmin, zero - 1e9)
