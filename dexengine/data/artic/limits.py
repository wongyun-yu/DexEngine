#!/usr/bin/env python3
"""Physical joint range: the joint-value interval in which the two link meshes do not penetrate each other, the scan
error band, and clamping of joint values outside it. Single owner.

Why: the spec window (dexengine.data.artic.ranges lo · hi, read by the bake limits and the converter qpos) is the union
of the data joint intervals. Frames where the source tracking fits a one-axis joint but passes the closing limit
(scissors: at the moment the hand grasps, both blades overlap down to the handles and close 57° further, while the
source video shows them closed throughout) are invisible to the one-axis check and to the FK oracle, because the limits
come from that data. Rule (no per-object table; computed from the spec statement and the link meshes):
  measure      from the scan pose (relative transform of the two link meshes = identity, the pose the object really had)
               step the joint value by STEP_DEG each way and measure the penetration depth (the KTH largest depth of the
               N_PTS surface samples inside the other mesh, discarding noise from one or two points). Inside/outside =
               generalised winding number (robust on open scan meshes and links joined from several pieces, unlike ray
               parity; computed on a copy reduced to at most FACES_MAX faces). Limit = the last value before the depth
               exceeds the scan pose's by PEN_DELTA_M.
  decidable    per side (lo · hi), measured when: a hinge (slides touch along their rail all the time and are not
               measured) · the scan pose lies on the joint (scan_off_joint) · the scan pose is not already in contact on
               that side (scan_contact: for a side scanned in contact the mesh cannot tell play beyond it (a stapler
               press) from a stop) · a limit was found (no_limit: thin parts, cranks) · beyond the limit the penetration
               neither resolves nor shrinks over MONO_DEG (not_monotonic: a ridge brushed past is not a stop).
  band         scan mesh thickness error: effective limit = limit ± SCAN_BAND_DEG (a laptop's closed pose exceeds the
      mesh limit by 3.4°).
  clamp        joint values beyond the effective limit of a decidable side are clamped to that limit regardless of
      length
               (clamp; done by the converter after the FK oracle, on the joint-value column only; source pose files stay
               as they are). Record = rules.D1g.range_clamp (frame count · max excess · intervals).
  undecidable  counted with a reason and passed through; that side's limit is the data window. Marker [ARTIC_RANGE] n=
      clamp= unmeasurable=.
The spec entry's phys = output of measure (version VERSION); the spec stage (dexengine.data.artic.spec_stage) measures
it once, stores it and narrows the window (narrow).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from dexengine.data.artic.fk import artic_moving_vertices

#: Version of the physical range rule; bump when measurement, decidability or band change (a spec phys of another
#: version is remeasured)
VERSION = "phys-1"
MARK = "[ARTIC_RANGE]"
#: Scan error band (degrees): effective limit = mesh limit ± this value. Chosen so the closed poses of 72 laptop clips
#: (mesh limit +2.8–3.4°) do not trigger
SCAN_BAND_DEG = 4.0
PEN_DELTA_M = 0.003  # limit = last value before the penetration depth exceeds the scan pose's by this much
# 'scanned in contact' when the scan pose penetrates this much deeper than the swept minimum and grows on that side
SCAN_CONTACT_M = 0.00015
STEP_DEG, COARSE, SWEEP_DEG, MONO_DEG = 1.0, 4, 180.0, 5.0
MONO_TOL_M = (
    0.001  # numerical slack of the monotonicity check (depth saturates at part thickness and sample noise remains)
)
N_PTS, KTH = 1500, 8
FACES_MAX, CHUNK = 2000, 256  # face limit for the winding-number computation · points per batch
SCAN_ROT_DEG, SCAN_POS_M = (
    2.0,
    0.01,
)  # slack for treating the scan pose (relative transform = identity) as on the joint (microwave door 1.4° · 5 mm)
_FRAME = np.vstack([np.zeros(3), np.eye(3)])


def _pose(spec: dict, theta: float) -> tuple[np.ndarray, np.ndarray]:
    v = artic_moving_vertices(spec, _FRAME, float(theta))
    return (v[1:] - v[0]).T, v[0]


def _solid(mesh):
    """Copy for penetration checks: vertices merged and faces reduced to at most FACES_MAX (fails if it cannot be
    reduced)."""
    import trimesh  # noqa: PLC0415 — only the measuring side (spec stage) loads mesh tools

    m = trimesh.Trimesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces), process=True)
    return m if len(m.faces) <= FACES_MAX else m.simplify_quadric_decimation(face_count=FACES_MAX)


def _inside(tri: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Whether points are inside the mesh: generalised winding number |Σ solid angle| / 4π > 0.5 (triangles (F,3,3);
    points outside the mesh box are not measured)."""
    out = np.zeros(len(pts), bool)
    v = tri.reshape(-1, 3)
    idx = np.flatnonzero(((pts >= v.min(0)) & (pts <= v.max(0))).all(1))
    for s in range(0, len(idx), CHUNK):
        i = idx[s : s + CHUNK]
        a, b, c = (tri[None, :, k, :] - pts[i, None, :] for k in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        num = np.einsum("nfi,nfi->nf", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("nfi,nfi->nf", a, b) * lc
            + np.einsum("nfi,nfi->nf", b, c) * la
            + np.einsum("nfi,nfi->nf", c, a) * lb
        )
        out[i] = np.abs(np.arctan2(num, den).sum(1)) > math.pi  # Σ Ω/2 > π ⇔ winding number > 0.5
    return out


def _depth(mesh, tri: np.ndarray, pts: np.ndarray) -> float:
    import trimesh  # noqa: PLC0415

    ins = _inside(tri, pts)
    if int(ins.sum()) < KTH:
        return 0.0
    return float(np.sort(trimesh.proximity.closest_point(mesh, pts[ins])[1])[-KTH])


def unmeasurable(why: str) -> dict:
    """phys with both sides undecidable (one reason)."""
    return {"version": VERSION, "scan": None, "lo": None, "hi": None, "why": {"lo": why, "hi": why}}


def scan_theta(spec: dict) -> tuple[float, float, float]:
    """Joint value ∈ [−π, π] closest to the scan pose (relative transform = identity) · rotation difference there
    (degrees) · origin difference (m)."""
    best = None
    for span, step in ((math.pi, math.radians(1.0)), (math.radians(1.0), math.radians(0.02))):
        c = 0.0 if best is None else best[0]
        for th in np.arange(c - span, c + span + step / 2, step):
            R, p = _pose(spec, th)
            a = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))
            if best is None or (a, float(np.linalg.norm(p))) < best[1:]:
                best = (float(th), a, float(np.linalg.norm(p)))
    return best


def measure(spec: dict, mesh_base, mesh_moving) -> dict:
    """Measure the physical range from the spec statement `spec` (type · axis · pos · child_pose) and the two link
    meshes
    (base · moving frame) → phys {version, scan, lo, hi (canonical joint coordinates; undecidable side = None), why{lo,
    hi}, pen_scan_mm, band_deg}."""
    if str(spec.get("type")) != "hinge":
        return unmeasurable("slide")
    th0, a0, p0 = scan_theta(spec)
    if a0 > SCAN_ROT_DEG or p0 > SCAN_POS_M:
        return unmeasurable("scan_off_joint")
    rel = _sweep(spec, th0, mesh_base, mesh_moving)
    return {
        "version": VERSION,
        "scan": round(th0, 6),
        "band_deg": SCAN_BAND_DEG,
        "pen_scan_mm": rel["pen_scan_mm"],
        "why": rel["why"],
        **{s: None if rel[s] is None else round(th0 + rel[s], 6) for s in ("lo", "hi")},
    }


def _sweep(spec: dict, th0: float, mesh_base, mesh_moving) -> dict:
    """Result of sweeping both ways from scan joint value th0: {lo, hi (limits relative to the scan, rad; undecidable =
    None), why, pen_scan_mm}."""
    import trimesh  # noqa: PLC0415

    A, B = _solid(mesh_base), _solid(mesh_moving)
    TA, TB = np.asarray(A.triangles, float), np.asarray(B.triangles, float)
    PA, PB = (np.asarray(trimesh.sample.sample_surface(m, N_PTS, seed=0)[0]) for m in (A, B))
    step, n_max, n_mono = math.radians(STEP_DEG), int(round(SWEEP_DEG / STEP_DEG)), int(round(MONO_DEG / STEP_DEG))

    def pen(th: float) -> float:
        R, p = _pose(spec, th)
        return max(_depth(A, TA, PB @ R.T + p), _depth(B, TB, (PA - p) @ R))

    pen0 = pen(th0)
    out = {"lo": None, "hi": None, "why": {}}
    seen, first = [pen0], {}
    for side, sg in (("lo", -1.0), ("hi", 1.0)):
        got: dict[int, float] = {}

        def at(k: int, got=got, sg=sg) -> float:
            if k not in got:
                got[k] = pen(th0 + sg * k * step)
            return got[k]

        at(1)  # whether the scan pose already touches on that side (the adjacent value)
        kc = next((k for k in range(COARSE, n_max + 1, COARSE) if at(k) > pen0 + PEN_DELTA_M), None)
        j = None if kc is None else next(k for k in range(kc - COARSE + 1, kc + 1) if at(k) > pen0 + PEN_DELTA_M)
        mono = j is None or all(
            at(i) > pen0 + PEN_DELTA_M and at(i) >= at(j) - MONO_TOL_M for i in range(j + 1, min(j + n_mono, n_max) + 1)
        )
        first[side] = (j, mono, got[1])
        seen += list(got.values())
    floor = min(seen)
    out["pen_scan_mm"] = round((pen0 - floor) * 1e3, 3)
    for side, sg in (("lo", -1.0), ("hi", 1.0)):
        j, mono, p1 = first[side]
        if pen0 - floor > SCAN_CONTACT_M and p1 > pen0:
            out["why"][side] = "scan_contact"
        elif j is None:
            out["why"][side] = "no_limit"
        elif not mono:
            out["why"][side] = "not_monotonic"
        else:
            out[side], out["why"][side] = sg * (j - 1) * step, "measured"
    return out


def eff(phys: dict | None) -> tuple[float, float]:
    """Effective limits (lo, hi) rad: decidable side = limit ± band · undecidable side = ∓∞ (that side is the data
    window).
    A different version fails. Its ValueError text reaches the exclusion ledger (through reexpress) and stays as
    written."""
    if not phys:
        return -math.inf, math.inf
    if phys.get("version") != VERSION:
        raise ValueError(
            f"{MARK} spec physical-range version {phys.get('version')!r} ≠ {VERSION!r} — the spec stage"
            " (dexengine.data.artic.spec_stage) measures it again"
        )
    b = math.radians(SCAN_BAND_DEG)
    return (
        -math.inf if phys.get("lo") is None else phys["lo"] - b,
        math.inf if phys.get("hi") is None else phys["hi"] + b,
    )


def clip_iv(iv: tuple[float, float], phys: dict | None) -> tuple[float, float]:
    """Joint interval clipped into the effective limits: the interval after the converter clamps (window check; the spec
    window reads this interval)."""
    lo, hi = eff(phys)
    return min(max(float(iv[0]), lo), hi), min(max(float(iv[1]), lo), hi)


def narrow(entry: dict, phys: dict) -> dict:
    """Put phys on the spec entry and narrow the window: (lo, hi) = data window ∩ effective limits. phys is moved to the
    window's 2π representative (scan joint value closest to the window midpoint); calling again gives the same entry."""
    ph = dict(phys)
    if ph.get("scan") is not None:
        k = 2.0 * math.pi * round(((float(entry["lo"]) + float(entry["hi"])) / 2.0 - ph["scan"]) / (2.0 * math.pi))
        if k:
            ph.update({key: round(ph[key] + k, 6) for key in ("scan", "lo", "hi") if ph.get(key) is not None})
    lo, hi = clip_iv((entry["lo"], entry["hi"]), ph)
    return {**entry, "lo": lo, "hi": hi, "span_max": hi - lo, "phys": ph}


def verdict(phys: dict) -> str:
    """Verdict line 'lo:<reason> hi:<reason>' (decidable when any side is measured)."""
    return " ".join(f"{s}:{phys['why'][s]}" for s in ("lo", "hi"))


def clamp(cj: dict, theta: np.ndarray, res: dict) -> None:
    """Converter part (after the FK oracle): clamp the canonical joint-value column θ **in place** to the spec asset's
    effective limits and record into res: range_phys = verdict line · range_clamp = {frames, max_excess_deg, intervals
    (frame intervals, inclusive), limit_deg, band_deg} (only when frames were clamped). --per-tree (bootstrap, no spec)
    does not measure. A spec asset without phys fails (stale spec)."""
    rx = cj.get("rx")
    if rx is None:
        res["range_phys"] = "per_tree"
        return
    ph = rx.get("phys")
    if not ph:
        raise ValueError(
            f"{MARK} spec asset {cj.get('asset')} has no physical range (phys) — stale spec: the spec stage "
            "(dexengine.data.artic.spec_stage) measures it first"
        )
    lo, hi = eff(ph)
    res["range_phys"] = verdict(ph)
    ex = np.maximum(lo - theta, theta - hi)
    out = ex > 0.0
    if not out.any():
        return
    idx = np.flatnonzero(out)
    cuts = np.flatnonzero(np.diff(idx) > 1)
    res["range_clamp"] = {
        "frames": int(out.sum()),
        "max_excess_deg": round(math.degrees(float(ex.max())), 2),
        "intervals": [
            [int(a), int(b)] for a, b in zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]], strict=True)
        ],
        "limit_deg": [None if math.isinf(x) else round(math.degrees(x), 2) for x in (lo, hi)],
        "band_deg": SCAN_BAND_DEG,
    }
    np.clip(theta, lo, hi, out=theta)


def mark(rows: list[dict]) -> None:
    """Conversion summary marker: trees with joint values n · clamped trees · undecidable (both sides) trees and their
    reasons."""
    js = [r for r in rows if r.get("range_phys") not in (None, "per_tree")]
    cl = [r for r in js if r.get("range_clamp")]
    un = Counter(r["range_phys"] for r in js if "measured" not in r["range_phys"])
    print(
        f"{MARK} n={len(js)} clamp={len(cl)} unmeasurable={sum(un.values())} "
        f"frames={sum(r['range_clamp']['frames'] for r in cl)} band={SCAN_BAND_DEG}° phys={VERSION}"
        + (f" why={dict(un)}" if un else ""),
        flush=True,
    )
