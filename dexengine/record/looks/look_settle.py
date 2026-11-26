"""Lighting settle of look D before the first frame (look_d module doc): stage 1 (loading) renders until consecutive
captures stop changing, stage 2 (light-transport drift) is off unless DRIFT_MAX > 0. Callers: looks.rev (record render,
the camera reference render), looks.augvis.render, the E5 closed-loop sessions; store.store keeps settle_row in the
episode metadata. render_meta.json ``look_settle`` = settle(). Marker SETTLE_MARK (look_d.log_problems requires it).
"""

from __future__ import annotations

import os

from dexengine.record.looks.look_assets import MARK

SETTLE_TOL, SETTLE_MAX = 0.6, 60
# Settle stage 1 (loading): mean pixel |delta| (8 bit) between consecutive captures (4 renders each) must drop below
# SETTLE_TOL within SETTLE_MAX rounds, otherwise the run dies.
# Plateau (two-eye settle only): converged also when, from round FLAT_N on, the last FLAT_K deltas have a relative
# half-window difference < FLAT_EPS, a linear trend (change over the window / mean) of magnitude < FLAT_EPS, and a
# mean < FLAT_CAP (a large desk in both eyes keeps the noise floor above SETTLE_TOL after loading has finished).
SETTLE_FLAT_N, SETTLE_FLAT_K, SETTLE_FLAT_EPS = 56, 16, 0.05
SETTLE_FLAT_CAP = 2 * SETTLE_TOL
# Settle stage 2 (light-transport drift): channel-mean difference of two captures DRIFT_GAP renders apart, after
# DRIFT_EXTRA renders, until < DRIFT_TOL (8 bit), at most DRIFT_MAX times; hitting the cap is recorded in
# look_settle.drift_cap. DRIFT_MAX 0 = stage 2 off (with reflections, GI and AO off nothing drifts).
DRIFT_EXTRA, DRIFT_GAP, DRIFT_TOL, DRIFT_MAX = 32, 32, 0.12, 0
SETTLE_EYES_ENV = "DEXENGINE_COLLECT_SETTLE_EYES"  # "2" = settle on both eyes (egoL, egoR) when exo is off
SETTLE_EYES_MARK = f"{MARK} settle two eyes"  # applied marker; log_problems requires it when the env is "2"


def settle_cams() -> tuple:
    """Cameras the settle measures: default (empty) = left eye + exo (exo on) or left eye only (exo off); "2" = both
    eyes with exo off. "2" with exo on, or any other value, is an error."""
    from dexengine.record.render import cameras as CAMS

    v = os.environ.get(SETTLE_EYES_ENV, "")
    if v not in ("", "2"):
        raise ValueError(f"{SETTLE_EYES_ENV}={v!r} — empty or 2 (two eyes with exo off) only")
    exo = CAMS.exo_on()
    if v == "2":
        if exo:
            raise ValueError(f"{SETTLE_EYES_ENV}=2 requires exo off; with exo on the settle uses left eye + exo")
        return ("egoL", "egoR")
    return ("egoL", "exo") if exo else ("egoL",)  # exo off: left eye only (exo is not rendered)


SETTLE_CAMS = settle_cams()
SETTLE_MARK = f"{MARK} settle drift"  # stage 2 ran (log_problems requires it)


def settle_flat(trace: list) -> dict | None:
    """Plateau test (SETTLE_FLAT_* comment): {floor, rel, trend} when flat, else None."""
    import numpy as np

    K = SETTLE_FLAT_K
    if len(trace) < K:
        return None
    w = np.asarray(trace[-K:], float)
    m = float(w.mean())
    rel = abs(float(w[K // 2 :].mean()) - float(w[: K // 2].mean())) / m
    trend = float(np.polyfit(np.arange(K), w, 1)[0]) * (K - 1) / m
    if rel < SETTLE_FLAT_EPS and abs(trend) < SETTLE_FLAT_EPS and m < SETTLE_FLAT_CAP:
        return {"floor": round(m, 3), "rel": round(rel, 4), "trend": round(trend, 4)}
    return None


def settle_load(rig, info: dict | None = None) -> int:
    """Settle stage 1 (MDL compile, texture load): until the mean pixel |delta| of consecutive captures (4 renders) is
    below SETTLE_TOL; returns the number of rounds. Two-eye settles also converge on a plateau (settle_flat, from
    round SETTLE_FLAT_N; recorded in info). Dies when it never converges."""
    import numpy as np

    prev, trace = None, []
    flat_on = SETTLE_CAMS == ("egoL", "egoR")
    for r in range(SETTLE_MAX):
        cap = rig.capture(4)
        cur = np.concatenate([cap[c].reshape(-1) for c in SETTLE_CAMS]).astype(np.int16)
        if prev is not None:
            d = float(np.abs(cur - prev).mean())
            trace.append(round(d, 3))
            if d < SETTLE_TOL:
                print(f"{MARK} settle {r + 1} rounds · |Δ| {d:.3f} < {SETTLE_TOL}", flush=True)
                if SETTLE_CAMS == ("egoL", "egoR"):
                    print(f"{SETTLE_EYES_MARK} {SETTLE_CAMS} · {SETTLE_EYES_ENV}=2", flush=True)
                return r + 1
            f = settle_flat(trace) if flat_on and r + 1 >= SETTLE_FLAT_N else None
            if f:
                print(
                    f"{MARK} settle {r + 1} rounds · plateau floor {f['floor']} · rel {f['rel']} · "
                    f"trend {f['trend']} "
                    f"(above {SETTLE_TOL}; flat = loading done)",
                    flush=True,
                )
                print(f"{SETTLE_EYES_MARK} {SETTLE_CAMS} · {SETTLE_EYES_ENV}=2", flush=True)
                if info is not None:
                    info.update(plateau=True, rounds=r + 1, **f)
                return r + 1
        prev = cur
    print(f"{MARK} settle not converged, |Δ| trace (cameras {SETTLE_CAMS} · tol {SETTLE_TOL}): {trace}", flush=True)
    raise SystemExit(f"{MARK} settle not converged in {SETTLE_MAX} rounds — check material and texture loading")


def chan_means(cap) -> dict:
    """Capture -> per-camera channel means (8-bit floats), the unit of the stage-2 drift measurement."""
    import numpy as np

    return {c: cap[c].reshape(-1, 3).astype(np.float64).mean(0) for c in SETTLE_CAMS}


def settle_drift(rig) -> dict:
    """Settle stage 2: light-transport drift (DRIFT_* comment). drift_cap = True at the cap (not an error)."""
    import numpy as np

    hist = []
    for _ in range(DRIFT_MAX):
        rig.render(DRIFT_EXTRA)
        a = chan_means(rig.capture(4))
        rig.render(DRIFT_GAP)
        b = chan_means(rig.capture(4))
        d = max(float(np.abs(b[c] - a[c]).max()) for c in SETTLE_CAMS)
        hist.append(round(d, 3))
        if d < DRIFT_TOL:
            break
    return {"drift": hist, "drift_tol": DRIFT_TOL, "drift_cap": bool(hist) and hist[-1] >= DRIFT_TOL}


def settle_row(look_settle) -> dict:
    """look_settle -> the episode line's render_settle field (an int = stage 1 only -> drift None)."""
    if isinstance(look_settle, int):
        return {"load_rounds": look_settle, "drift": None, "drift_cap": None}
    return {k: look_settle[k] for k in ("load_rounds", "drift", "drift_cap")}


def settle(rig) -> dict:
    """Convergence before the first frame = stage 1 (settle_load) + stage 2 (settle_drift); stored as render_meta
    look_settle. Marker SETTLE_MARK; '[COLLECT_LOOK] settle cap' when stage 2 hits its cap."""
    import time

    t0 = time.time()
    pl: dict = {}
    n = settle_load(rig, pl)
    dr = settle_drift(rig)
    renders = n * 5 + len(dr["drift"]) * (DRIFT_EXTRA + DRIFT_GAP + 12)
    out = dict(load_rounds=n, **dr, drift_rounds=len(dr["drift"]), renders=renders, s=round(time.time() - t0, 1))
    if pl:  # only plateau-converged settles get this field
        out["load_plateau"] = pl
    tail = (
        f"cap {DRIFT_MAX} rounds — recorded (not a failure, look_settle.drift_cap)"
        if dr["drift_cap"]
        else f"converged in {len(dr['drift'])} rounds"
        if DRIFT_MAX
        else "stage 2 off (cap 0)"
    )
    print(f"{SETTLE_MARK} {dr['drift']} (tol {DRIFT_TOL}) · {tail} · renders {renders} · {out['s']} s", flush=True)
    if dr["drift_cap"]:
        print(f"{MARK} settle cap · last difference {dr['drift'][-1]} >= {DRIFT_TOL}", flush=True)
    return out
