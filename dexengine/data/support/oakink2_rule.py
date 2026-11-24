#!/usr/bin/env python3
"""**Single owner** of the OakInk2 desk rule: constants (thresholds, window, policy) and pure functions (testable
without sources or USD).

The rule text (①–④), reading and writing are described in the docstring of dexengine/data/support/oakink2.py; only
values and computation live here.
ClipError texts raised while fitting a clip (fixed_rx_t) are stored in INTAKE desk_rule.error and stay as written.
"""

from __future__ import annotations

import collections
import re

import numpy as np

RULE_VERSION = "2026-10-01"
#: Desk policy: **session_w** (one of sequence · session · session_w · global); this constant owns the value.
#:   session = median of the session's (scene+date) sequence G values · session_w = duration-weighted median over all of
#:   the session's in-window rest levels (the same function seq_desk as G, at session size, so sequences with little
#:   evidence do not get a full vote) · sequence = the sequence's G (else the session value, source session_fill) ·
#:   global = one median of G over all sequences
DESK_POLICY = "session_w"
POLICIES = ("session", "session_w", "sequence", "global")

MOCAP_HZ = 120.0
STRIDE = 4  # rest samples = 120/4 = 30 Hz (the window constants were derived at this resolution)
REST_HALF_WIN = 4  # speed window ±4 samples (±0.13 s)
V_REST_MPS = 0.03  # rest: centre speed < 3 cm/s
W_REST_DPS = 20.0  #        angular speed < 20 °/s
MIN_REST_S = 0.5  # minimum rest interval length
DESK_WINDOW_M = (
    -0.012,
    0.004,
)  # window of desk rest levels (rotated raw z, hull bottom; peak −5…−3 mm, cut at −10 below, above +10 = parts)
T_XY_MAX_M = 0.002  # transform oracle: x·y of the translation (an exact transform is z only; measured ≤ 0.14 mm)
RES_P95_MAX_M = 0.010  #   origin residual p95 (measured max 5.4 mm)
ANG_MED_MAX_DEG = 5.0  #   median pose difference (measured max 2.9°)
MIN_OVERLAP = 10  # minimum number of compared frames
HAND_ORACLE_M = (
    0.02  # lowest hand < desk_z − 2 cm → hand_below marker (keypoints are joint centres and must be above the desk)
)
SESSION_DEV_FLAG_M = 0.004  # |G − session| > 4 mm → seq_dev_flag
RX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # raw y-up → z-up (+90° about x)
USD_PREFIXES = (
    "raw",
    "left",
    "right",
    # hull of a raw rid = the first existing USD copy with these prefixes (same mesh; bottom difference ≤ 0.013 mm,
    # measured)
)
SEQ_RE = re.compile(r"^(scene_\d+)__\w+\+\+seq__([0-9a-f]+)__(\d{4}-\d\d-\d\d)-")
CLIP_RE = re.compile(r"oakink2_.+__scene_\d+_([0-9a-f]{12})_s(\d+)$")
OLD_FIELDS = ("raw_table_zup", "raw_t", "raw_fit_residual")  # fields of the old version, removed on write


class ClipError(ValueError):
    """The rule cannot be applied to this clip (no source · nothing to compare · transform oracle violated)."""


class SessionError(RuntimeError):
    """The session desk cannot be determined (unreadable name · no in-window rest anywhere in the session)."""


def session_of(seq: str) -> tuple[str, str]:
    m = SEQ_RE.match(seq)
    if not m:
        raise SessionError(f"cannot read (scene, date) from the sequence name: {seq!r}")
    return m.group(1), m.group(3)


def rest_mask(pos: np.ndarray, rot: np.ndarray, hz: float) -> np.ndarray:
    """(n,3) positions · (n,3,3) rotations (STRIDE samples) → rest sample mask (centre and angular speed, ±REST_HALF_WIN
    window)."""
    n = len(pos)
    a = np.clip(np.arange(n) - REST_HALF_WIN, 0, n - 1)
    b = np.clip(np.arange(n) + REST_HALF_WIN, 0, n - 1)
    span = np.maximum((b - a) / hz, 1e-9)
    sp = np.linalg.norm(pos[b] - pos[a], axis=1) / span
    tr = np.einsum("nij,nij->n", rot[a], rot[b])  # trace(Ra^T Rb)
    w = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))) / span
    return (sp < V_REST_MPS) & (w < W_REST_DPS)


def rest_intervals(rest: np.ndarray, level: np.ndarray, hz: float) -> list[dict]:
    """Rest mask → intervals (at least MIN_REST_S) · interval level = median of level (bottom)."""
    out, n, i, m = [], len(rest), 0, int(round(MIN_REST_S * hz))
    while i < n:
        if not rest[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and rest[j + 1]:
            j += 1
        if j - i + 1 >= m:
            out.append({"i0": i, "i1": j, "dur_s": (j - i + 1) / hz, "level": float(np.median(level[i : j + 1]))})
        i = j + 1
    return out


def weighted_median(v, w) -> float:
    o = np.argsort(v)
    v, w = np.asarray(v, float)[o], np.asarray(w, float)[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


def seq_desk(levels: list[tuple[float, float]]) -> float | None:
    """[(level, duration s)] → duration-weighted median within the window (G); None when nothing falls inside."""
    win = [(z, d) for z, d in levels if DESK_WINDOW_M[0] <= z <= DESK_WINDOW_M[1]]
    return None if not win else weighted_median([z for z, _ in win], [d for _, d in win])


def raw_desks(
    seq_G: dict[str, float | None], policy: str, seq_levels: dict[str, list[tuple[float, float]]] | None = None
) -> dict[str, tuple[float, str]]:
    """Sequence → (source-world desk, source). policy ∈ POLICIES. A session without any G raises SessionError.
    session_w needs [(level, duration s)] per sequence (seq_levels, the samples that made G); ValueError without it."""
    if policy not in POLICIES:
        raise ValueError(f"unknown desk policy {policy!r} (known: {POLICIES})")
    if policy == "session_w" and seq_levels is None:
        raise ValueError("the session_w policy needs the sequence rest levels (seq_levels)")
    sess = collections.defaultdict(list)
    for s, g in seq_G.items():
        sess[session_of(s)].append(g)
    smed = {}
    for k, gs in sess.items():
        gs = [g for g in gs if g is not None]
        if not gs:
            raise SessionError(
                f"session {k}: none of its {len(sess[k])} sequences has a rest inside the desk window "
                f"{DESK_WINDOW_M} — "
                "no evidence for the desk (never falls back to a global value)"
            )
        smed[k] = float(np.median(gs))
    if policy == "session_w":  # all in-window rests of the session; the same function as G (seq_desk)
        pooled = collections.defaultdict(list)
        for s in seq_G:
            pooled[session_of(s)] += list(seq_levels.get(s) or [])
        smed = {k: seq_desk(pooled[k]) for k in smed}
    allg = [g for g in seq_G.values() if g is not None]
    if not allg:
        raise SessionError("no sequence has a rest inside the desk window")
    glob = float(np.median(allg))
    out = {}
    for s, g in seq_G.items():
        k = session_of(s)
        if policy in ("session", "session_w"):
            out[s] = (smed[k], policy)
        elif policy == "sequence":
            out[s] = (g, "sequence") if g is not None else (smed[k], "session_fill")
        else:
            out[s] = (glob, "global")
    return out


def fixed_rx_t(
    raw_t: np.ndarray,
    raw_pos: np.ndarray,
    raw_rot: np.ndarray,
    ours_t: np.ndarray,
    ours_pos: np.ndarray,
    ours_rot: np.ndarray,
) -> dict:
    """Rotation-fixed (Rx) translation t = median(ours − Rx·raw) + oracle. Times (s) share the origin (clip interval
    start)."""
    ok = (ours_t >= raw_t[0]) & (ours_t <= raw_t[-1]) & np.isfinite(ours_pos).all(1)
    if ok.sum() < MIN_OVERLAP:
        raise ClipError(f"compared frames {int(ok.sum())} < {MIN_OVERLAP}")
    rp = np.stack([np.interp(ours_t[ok], raw_t, raw_pos[:, k]) for k in range(3)], 1)
    d = ours_pos[ok] - rp @ RX.T
    t = np.median(d, 0)
    res = np.linalg.norm(d - t, axis=1)
    j = np.clip(np.searchsorted(raw_t, ours_t[ok]), 0, len(raw_t) - 1)
    rr = RX @ raw_rot[j]
    ang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", ours_rot[ok], rr) - 1) / 2, -1, 1)))
    rec = {
        "t": [float(x) for x in t],
        "res_p95_m": float(np.percentile(res, 95)),
        "ang_med_deg": float(np.median(ang)),
        "n": int(ok.sum()),
    }
    why = []
    if max(abs(t[0]), abs(t[1])) > T_XY_MAX_M:
        why.append(f"|t_xy| {max(abs(t[0]), abs(t[1])) * 1000:.2f} mm > {T_XY_MAX_M * 1000:.1f}")
    if rec["res_p95_m"] > RES_P95_MAX_M:
        why.append(f"residual p95 {rec['res_p95_m'] * 1000:.2f} mm > {RES_P95_MAX_M * 1000:.1f}")
    if rec["ang_med_deg"] > ANG_MED_MAX_DEG:
        why.append(f"pose difference {rec['ang_med_deg']:.2f}° > {ANG_MED_MAX_DEG}")
    if why:
        raise ClipError(
            "transform oracle: " + " · ".join(why) + " — the collaborator transform is not Rx + translation"
        )
    return rec
