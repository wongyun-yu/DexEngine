"""v4 approach schedule — timing of σ along the route (hand-set speed profile, end-speed join, length K).

The values (TIME_*, JOIN_*, SPEED_*) are owned by rules.py — this module holds the functions only (split for size).
Readers: plan.py (schedule, length) and scene.py (first frame of the straight entry).
"""

from __future__ import annotations

import math

import numpy as np

from experiments.real_kit.v4 import rules as V4


def path_time(q: np.ndarray, budget: np.ndarray, P: np.ndarray, v_tgt: float, dt: float) -> np.ndarray:
    """Route parameter σ (M,) = the cumulative sum (in frames) of max(joint time, wrist time |ΔP| / (v_tgt · dt)) per
    sample. Joint time alone crowds the wrist speed into short shoulder-moving stretches, and that one point slowed the
    whole approach; with the wrist time the wrist speed stays evenly below v_tgt along the route."""
    dj = (np.abs(np.diff(q, axis=0)) / np.asarray(budget)[None]).max(axis=1)
    dh = np.linalg.norm(np.diff(P, axis=0), axis=1) / (float(v_tgt) * float(dt))
    return np.r_[0.0, np.cumsum(np.maximum(np.maximum(dj, dh), 1e-9))]


def speed_profile(K: int, sig_end: float, e: float) -> tuple[np.ndarray, dict]:
    """σ(k) (K+1,) — integral of the accelerate · cruise · join speed profile. σ(0) = 0, σ(K) = sig_end, end speed
    e
    (σ/frame), end acceleration 0. Short routes shrink the accelerate and join stretches into K; when the cruise gets
    slower than the end speed, the end speed is lowered to the cruise (recorded e_used). → (σ, {c cruise, e_used, Ta,
    Tc})."""
    from dexengine.placement.retime import stretch as VST

    Ta, Tc = V4.TIME_ACCEL_S / VST.FRAME_DT, V4.TIME_JOIN_S / VST.FRAME_DT
    if Ta + Tc > K:
        Ta, Tc = Ta * K / (Ta + Tc), Tc * K / (Ta + Tc)
    Tb = K - Ta - Tc
    e = max(0.0, float(e))
    c = (sig_end - e * Tc / 2) / (Ta / 2 + Tb + Tc / 2)
    if c < e:  # length left even when cruising at the end speed — cruise = end speed
        c = e = sig_end / (Ta / 2 + Tb + Tc)
    n = 32
    t = np.linspace(0.0, K, K * n + 1)
    v = np.where(
        t < Ta,
        c * V4.minjerk(t / max(Ta, 1e-9)),
        np.where(t < Ta + Tb, c, c + (e - c) * V4.minjerk((t - Ta - Tb) / max(Tc, 1e-9))),
    )
    s_ = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * (t[1] - t[0]))]
    s_ = s_[::n] * (sig_end / s_[-1])
    return s_, {"c": float(c), "e_used": float(e), "Ta": float(Ta), "Tc": float(Tc)}


def resample(K: int, sig: np.ndarray, *cols: np.ndarray, e: float = 0.0) -> list[np.ndarray]:
    """Schedule over K steps (K+1 frames) (speed_profile on σ) → every column (values over route samples) interpolated
    to
    frames; last frame = route end; end speed e (σ/frame)."""
    sk, _ = speed_profile(K, float(sig[-1]), e)
    return [np.stack([np.interp(sk, sig, c[:, j]) for j in range(c.shape[1])], 1) for c in cols]


def join_speed(sig: np.ndarray, pos: np.ndarray, v_ref: float, dt: float, safety: float) -> float:
    """End speed (σ/frame) = the reference frame-0 wrist speed v_ref (m/s) divided by the route end tangent
    (|ΔP|/Δσ),
    capped at 1/safety (σ speed 1 = the joint budget or the wrist v_tgt)."""
    dsig = float(sig[-1] - sig[-2])
    dp = float(np.linalg.norm(pos[-1] - pos[-2]))
    if dp <= 1e-9 or dsig <= 0.0:
        return 0.0
    return min(float(v_ref) * float(dt) / (dp / dsig), 1.0 / float(safety))


def choose_frames(
    sig: np.ndarray,
    q: np.ndarray,
    pos: np.ndarray,
    v_target: float,
    dt: float,
    budget: np.ndarray,
    safety: float,
    extra=None,
    e: float = 0.0,
) -> tuple[int, dict]:
    """Approach length (frames K) — start from cruise σ speed = 1/safety (length including accelerate and join) →
    stretch until the joint and finger speed ratio (stretch.frame_ratio × safety) ≤ 1 (extra(K) → extra columns
    (K+1, m), budget (m,) — fingers) → stretch while the wrist peak speed is above v_target (why = speed_target); the
    end speed e
    (σ/frame) stays."""
    from dexengine.placement.retime import stretch as VST

    def ratio(K):
        qk, pk = resample(K, sig, q, pos, e=e)
        r = VST.frame_ratio(qk, budget).max() * safety
        if extra is not None:
            qe, be = extra(K)
            r = max(r, VST.frame_ratio(qe, be).max() * safety)
        v = np.linalg.norm(np.diff(pk, axis=0), axis=1).max() / dt
        return r, v

    c0 = 1.0 / safety
    Ta, Tc = V4.TIME_ACCEL_S / dt, V4.TIME_JOIN_S / dt
    Tb = (float(sig[-1]) - c0 * Ta / 2 - (c0 + e) * Tc / 2) / c0
    K = max(10, int(math.ceil(Ta + Tc + max(0.0, Tb))))
    r, v = ratio(K)
    while r > 1.0 and K < 4000:
        K = int(math.ceil(K * max(1.01, r)))
        r, v = ratio(K)
    why = "joint_limit"
    if v > v_target:
        K = int(math.ceil(K * v / v_target))
        r, v = ratio(K)
        why = "speed_target"
    return K, {"K": K, "T_s": K * dt, "why": why, "ratio_max": float(r), "v_peak": float(v)}


def join_floor(
    K: int, sig: np.ndarray, q: np.ndarray, e: float, dt: float, safety: float, floor: float | None = None
) -> tuple[float, dict]:
    """Join floor (rules.JOIN_FLOOR, 0 = off) — when the end value of the active arm joint speed norm on the schedule
    (K, end speed e) is below JOIN_LOW_FRAC × peak × JOIN_FLOOR, raise e proportionally and measure again (at most 6
    times, cap 1/safety). floor None = rules.JOIN_FLOOR · 0 = off (re-plan, rules.REPLAN_FLOOR_OFF) → (e, record {on,
    e0, e1,
    end0, end1, tgt})."""

    def speeds(e_):
        (qk,) = resample(K, sig, q, e=e_)
        w = np.linalg.norm(np.diff(qk, axis=0), axis=1) / dt
        return float(w[-1]), float(w.max())

    fl = V4.JOIN_FLOOR if floor is None else float(floor)
    end0, peak = speeds(e)
    tgt = fl * V4.JOIN_LOW_FRAC * peak
    rec = {
        "on": False,
        "e0": round(float(e), 4),
        "e1": round(float(e), 4),
        "end0": round(end0, 4),
        "end1": round(end0, 4),
        "tgt": round(tgt, 4),
    }
    if fl <= 0 or end0 >= tgt:
        return float(e), rec
    e1, end1 = float(e), end0
    for _ in range(6):
        e1 = min(e1 * tgt / max(end1, 1e-9) * 1.02, 1.0 / float(safety))
        end1, peak = speeds(e1)
        tgt = fl * V4.JOIN_LOW_FRAC * peak
        if end1 >= tgt or e1 >= 1.0 / float(safety):
            break
    rec.update(on=True, e1=round(e1, 4), end1=round(end1, 4), tgt=round(tgt, 4))
    return e1, rec
