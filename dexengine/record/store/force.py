"""Contact-force quality marks and cleaning rules (CPU). The store writes the marks; cleaning adds a copy channel
(the raw forces are never changed).

Spikes are contact-solver (PhysX) artifacts, not recording defects: on spike steps the separately recorded object
state jumps too (measured on 646 XHand Pro episodes: link |F| p99 23.8 N, p99.9 78.5 N, p99.99 577 N, max 6,038 N).
Most first spikes happen right after the reset teleport (steps 1-2, frame-0 hand-object overlap pushed apart) or
during SPAWN_HOLD (the object is rewritten to its reference pose each step while fingers press on it); after the hold
half of the spikes are two links pushing in opposite directions (the object is wedged in the hand).
Threshold: the per-run maximum of post-hold (episode, link) runs falls geometrically from 30 to 141 N and stays far
above that trend from 171 N on, so F_SPIKE_N = 170 N.
Wedge: two links of the same hand pushing against each other (cos <= WEDGE_COS) with >= WEDGE_N each, i.e. a finger
segment stuck ~1 cm in a thin part of the object that PhysX penetration recovery cannot resolve. It is detected by
shape, not size (wedges peaking at 172-240 N escaped the F_SPIKE_N run rule).
Rules (episodes are never dropped; raw forces stay, marks and a cleaned channel are added; keys and meanings =
QUALITY_KEYS):
  marks (quality.* [T]): pinned = the sample's policy step <= pin_steps (reset + object hold, record meta
    spawn_hold_steps); spike = any link |F| > F_SPIKE_N; wedge = wedge sample; flag = post-hold (spike ± PAD_SAMPLES)
    | wedge; filled = flagged samples filled in the cleaned channel.
  cleaned channel (<stem>.force_n_clean, per force kind): FILL_RULE. Flag runs of JAM_SECONDS or longer (persistent,
    a wedge usually lasts the episode) are not filled and keep the raw value: the neighbouring normal samples are too
    far away, filling would invent forces.
  episode summary (episodes.jsonl force_quality, quality()): unreliable = a post-hold spike run >= JAM_SECONDS or a
    post-hold peak > JAM_PEAK_N; force_wedge = a post-hold wedge run >= JAM_SECONDS. Both are labels, not exclusions.
  inputs: the rules use the total and object forces only (QUALITY_KINDS). Partner-split kinds are per-class components
    that can cancel each other and exceed the total, so they are not thresholded (the cleaned channel covers every
    force kind).
FILL_RULE and QUALITY_KEYS are stored in info.json (merge compares info.collect between a piece and the final).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError
from dexengine.record.rollout import recorder as REC

F_SPIKE_N = 170.0
PAD_SAMPLES = 1
JAM_SECONDS = 0.5
JAM_PEAK_N = 1000.0
WEDGE_N = 100.0  # min magnitude of each link of a wedge pair (>= 5x the 15-19.4 N fingertip ratings)
WEDGE_COS = -0.8  # wedge pair direction (opposite)
CLEAN_SUFFIX = ".force_n_clean"  # cleaned channel key = <stem> + this (next to the raw <stem>.force_n)
FILL_RULE = (
    "fill only flag runs shorter than jam_seconds — linear interpolation in time between the normal samples "
    "(outside the flag · after the hold) just before and just after, same key and same column · keep the "
    "one value when only one side exists · raw value when neither exists or the run is jam_seconds or longer"
)
RULE = {
    "f_spike_n": F_SPIKE_N,
    "pad_samples": PAD_SAMPLES,
    "jam_seconds": JAM_SECONDS,
    "jam_peak_n": JAM_PEAK_N,
    "wedge_n": WEDGE_N,
    "wedge_cos": WEDGE_COS,
    "fill": FILL_RULE,
    "owner": "scripts/collect/collect_force.py",
}  # format tag in info.json (kept)
# Keys added to the force file and their meaning (stored as info.json collect.force_quality_keys).
QUALITY_KEYS = {
    "quality.pinned": "hold-window samples (policy step ≤ spawn_hold_steps — right after reset + the window that "
    "rewrites "
    "the object to its reference pose)",
    "quality.spike": "samples where any link has |F| > f_spike_n (hold window included)",
    "quality.wedge": "wedge samples — two links of one hand (one total-force sensor) push in opposite directions "
    "(cos ≤ wedge_cos), each ≥ wedge_n (hold window included)",
    "quality.flag": "spike flag — after the hold, (spike ± pad_samples samples) ∪ wedge",
    "quality.filled": "flagged samples filled in the cleaned channel (fill rule)",
    "<stem>" + CLEAN_SUFFIX: "cleaned force — same shape as the raw <stem>.force_n · only filled samples are "
    "filled by the fill rule · the rest = raw values",
}
QUALITY_KINDS = (REC.KIND_TOTAL, REC.KIND_OBJECT)
_NEW_KINDS = tuple(k for k in REC.CONTACT_KINDS if k not in QUALITY_KINDS)
MARK = "[COLLECT_FORCE]"


def sample_steps(t: np.ndarray, step_dt: float) -> np.ndarray:
    """Sample times -> the policy step each sample belongs to (physics samples of step k are in ((k-1)dt, k dt])."""
    return np.ceil(np.asarray(t, float) / step_dt - 1e-6).astype(np.int64)


def link_mag(forces: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """{name: (force_n [T, B, 3], body names)} -> (max |F| per sample [T], per-link |F| [T, sum B], column names)."""
    cols, names = [], []
    for k, (f, nm) in forces.items():
        cols.append(np.linalg.norm(np.asarray(f, np.float64), axis=-1))
        names += [f"{k}:{n}" for n in nm]
    m = np.concatenate(cols, axis=1)
    return m.max(axis=1), m, names


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True runs [(start, end inclusive)]."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def _head(stem: str) -> str:
    return stem.split(".", 1)[0]


def quality_inputs(forces: dict) -> dict:
    """Inputs of the quality rules: drop partner-split kinds (sensor keys without a kind prefix stay)."""
    return {k: v for k, v in forces.items() if _head(k) not in _NEW_KINDS}


def _total_like(stem: str) -> bool:
    """A total force of one hand = one sensor: force_total.<sensor> (or a <side>_hand_contact sensor key)."""
    return _head(stem) == REC.KIND_TOTAL or stem.endswith("_hand_contact")


def wedge_mask(forces: dict) -> tuple[np.ndarray, dict]:
    """Wedge samples [T] and {sensor: [[link a, link b], ...]}: per hand (total-force sensor) link pairs pushing in
    opposite directions (cos <= WEDGE_COS) with both >= WEDGE_N. No finger tables (works for any hand)."""
    T = len(next(iter(forces.values()))[0])
    w = np.zeros(T, bool)
    pairs = {}
    for k, (f, nm) in forces.items():
        if not _total_like(k):
            continue
        f = np.asarray(f, np.float64)
        m = np.linalg.norm(f, axis=-1)
        hit = set()
        for a, b in itertools.combinations(range(f.shape[1]), 2):
            ok = (m[:, a] >= WEDGE_N) & (m[:, b] >= WEDGE_N)
            if not ok.any():
                continue
            cos = (f[:, a] * f[:, b]).sum(-1) / np.maximum(m[:, a] * m[:, b], 1e-9)
            hw = ok & (cos <= WEDGE_COS)
            if hw.any():
                w |= hw
                hit.add((str(nm[a]), str(nm[b])))
        if hit:
            pairs[k] = sorted(list(p) for p in hit)
    return w, pairs


def _grow(m: np.ndarray, pad: int) -> np.ndarray:
    g = m.copy()
    for s in range(1, pad + 1):
        g[s:] |= m[:-s]
        g[:-s] |= m[s:]
    return g


def quality(forces: dict, t: np.ndarray, step_dt: float, pin_steps: int) -> tuple[dict, dict]:
    """Sample masks {pinned, spike, wedge, flag, filled} and the episode force_quality summary (rule RULE)."""
    if len(t) == 0:
        raise ValueError(f"{MARK} no force samples")
    forces = quality_inputs(forces)
    mx, per, names = link_mag(forces)
    pinned = sample_steps(t, step_dt) <= int(pin_steps)
    spike = mx > F_SPIKE_N
    free = spike & ~pinned
    dt_s = float(np.median(np.diff(t))) if len(t) > 1 else step_dt
    runs = _runs(free)
    longest = max((b - a + 1 for a, b in runs), default=0)
    peak_free = float(mx[~pinned].max()) if (~pinned).any() else 0.0
    wedge, wpairs = wedge_mask(forces)
    wlong = max((b - a + 1 for a, b in _runs(wedge & ~pinned)), default=0)
    flag = (_grow(free, PAD_SAMPLES) | wedge) & ~pinned
    filled = np.zeros_like(flag)
    for a, b in fill_runs(flag, pinned, dt_s):
        filled[a : b + 1] = True
    flong = max((b - a + 1 for a, b in _runs(flag)), default=0)
    q = {
        "threshold_n": F_SPIKE_N,
        "pin_steps": int(pin_steps),
        "max_link_n": round(float(mx.max()), 3),
        "max_link_n_after_pin": round(peak_free, 3),
        "spike_samples": int(spike.sum()),
        "spike_samples_after_pin": int(free.sum()),
        "first_spike_step_after_pin": int(sample_steps(t, step_dt)[free][0]) if free.any() else None,
        "spike_links_after_pin": sorted({names[j] for j in np.flatnonzero((per[free] > F_SPIKE_N).any(axis=0))}),
        "longest_run_s": round(longest * dt_s, 4),
        "unreliable": bool(longest * dt_s >= JAM_SECONDS or peak_free > JAM_PEAK_N),
        "rule": RULE,
        "wedge_samples_after_pin": int((wedge & ~pinned).sum()),
        "wedge_longest_s": round(wlong * dt_s, 4),
        "wedge_pairs": wpairs,
        "pinned_wedge": bool((wedge & pinned).any()),
        "force_wedge": bool(wlong * dt_s >= JAM_SECONDS),
        "flag_samples": int(flag.sum()),
        "filled_samples": int(filled.sum()),
        "flag_unfilled_samples": int((flag & ~filled).sum()),
        "longest_flag_s": round(flong * dt_s, 4),
    }
    return {"pinned": pinned, "spike": spike, "wedge": wedge, "flag": flag, "filled": filled}, q


def fill_runs(flag: np.ndarray, pinned: np.ndarray, dt_s: float) -> list[tuple[int, int]]:
    """Flag runs to fill [(start, end inclusive)]: shorter than JAM_SECONDS with a normal sample before (not pinned)
    or after. Flag runs are maximal, so the samples right before and after are not flagged."""
    T = len(flag)
    return [
        (a, b)
        for a, b in _runs(flag)
        if (b - a + 1) * dt_s < JAM_SECONDS and ((a > 0 and not pinned[a - 1]) or b + 1 < T)
    ]


def clean_force(arr: np.ndarray, t: np.ndarray, masks: dict) -> np.ndarray:
    """Cleaned force [T, ...]: only `filled` samples are filled per FILL_RULE, the rest are raw (same dtype)."""
    out = np.array(arr, copy=True)
    T = len(t)
    filled, pinned = masks["filled"], masks["pinned"]
    src = np.asarray(arr, np.float64)
    for a, b in _runs(filled):
        lo = a - 1 if a > 0 and not pinned[a - 1] else None
        hi = b + 1 if b + 1 < T else None
        if lo is not None and hi is not None:
            w = ((np.asarray(t[a : b + 1], np.float64) - t[lo]) / (t[hi] - t[lo])).reshape(-1, *[1] * (src.ndim - 1))
            out[a : b + 1] = ((1.0 - w) * src[lo] + w * src[hi]).astype(out.dtype)
        else:
            out[a : b + 1] = arr[lo if lo is not None else hi]
    return out


# ---- store time axis and hold window -------------------------------------------------------------------------
def force_timeline(forces: dict, step_dt: float, physics_dt: float, L: int):
    """History columns (newest -> oldest) -> physics-step time order [T, B, 3] and t.
    Step 0 is right after reset (its history is the reset state), so unrolling starts at step 1."""
    out, t = {}, None
    if len({arr.shape[1] for arr, _n in forces.values()}) > 1:
        raise SystemExit("[COLLECT_STORE] force channels have different history lengths — no single time axis")
    for k, (arr, names) in forces.items():
        h = arr.shape[1]
        # history of step k [h-1 ... 0] = time order (force [.., 3], separation [..])
        seq = arr[1:, ::-1].reshape(-1, *arr.shape[2:])
        tt = (np.arange(1, L)[:, None] * step_dt - (np.arange(h)[::-1][None, :]) * physics_dt).reshape(-1)
        out[k] = {"force_n": seq.astype(np.float32), "body_names": names}
        t = tt
    return out, t


# Name keys of the force npz (body = receiving side, link = applying side, class = partner) and value keys
# (force = force_n, separation = sep_m).
NAME_KEY = {
    REC.KIND_TOTAL: "body_names",
    REC.KIND_OBJECT: "link_names",
    REC.KIND_PARTNER: "body_names",
    REC.KIND_OBJ_PARTNER: "partner_names",
    REC.KIND_SEP: "link_names",
}
VALUE_KEY = {**dict.fromkeys(REC.FORCE_KINDS, "force_n"), REC.KIND_SEP: "sep_m"}


def value_key(stem: str) -> str:
    """Stored key stem (<kind>.<name>) -> value key (<stem>.force_n | .sep_m)."""
    return f"{stem}.{VALUE_KEY[_head(stem)]}"


def clean_keys(stems) -> list[str]:
    """Cleaned channel keys, one per force stem (separation is not a force)."""
    return sorted(f"{k}{CLEAN_SUFFIX}" for k in stems if _head(k) != REC.KIND_SEP)


def write(fp: Path, fo: dict, t, masks: dict) -> None:
    """Write one episode force file: t, `<kind>.<name>.force_n`, name keys (NAME_KEY), quality.<mask> and the cleaned
    channels `<kind>.<name>.force_n_clean` (QUALITY_KEYS)."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fp,
        t=t,
        **{value_key(k): v["force_n"] for k, v in fo.items()},
        **{f"{k}.{NAME_KEY[k.split('.', 1)[0]]}": np.array(v["body_names"]) for k, v in fo.items()},
        **{f"quality.{m}": v for m, v in masks.items()},
        **{
            f"{k}{CLEAN_SUFFIX}": clean_force(v["force_n"], t, masks) for k, v in fo.items() if _head(k) != REC.KIND_SEP
        },
    )


def info_fields(fo: dict, pins: set, frates: set) -> dict:
    """Force fields of info.json collect: value keys, signs, hold windows, rules, rate, cleaned keys, key meanings."""
    return dict(
        force_keys=sorted({value_key(k) for k in fo}),
        force_sign=REC.FORCE_SIGN,
        **({"partner_sign": REC.PARTNER_SIGN} if any(_head(k) in REC.PARTNER_SIGN for k in fo) else {}),
        force_component="normal",
        spawn_hold_steps_set=sorted(pins),
        force_quality_rule=RULE,
        force_rate_hz=force_rate(frates),
        force_clean_keys=clean_keys(fo),
        force_quality_keys=QUALITY_KEYS,
    )


def force_rate(rates: set) -> float | None:
    """Per-episode rates measured on force_timeline's t -> one value (several is an error: one dataset, one rate)."""
    if len(rates) > 1:
        raise SystemExit(f"[COLLECT_STORE] force rate differs between episodes {sorted(rates)}")
    return next(iter(rates), None)


def pin_steps(meta_hold, traj: Path, clip: str) -> int:
    """Hold window of the quality marks: the record meta spawn_hold_steps (required)."""
    if meta_hold is None:
        raise InputMissingError(f"{MARK} {clip}: record meta has no spawn_hold_steps ({traj})")
    return int(meta_hold)


def load_forces(z) -> tuple[dict, dict]:  # noqa: V103 — used by experiments/sim_eval/judges/judge_io.py
    """Force npz -> ({stem: (force_n, body names)}, raw key -> stem). Keys `<kind>.<name>.force_n` (or sensor-only
    `<sensor>.force_n`)."""
    out, keys = {}, {}
    for k in z.files:
        if not k.endswith(".force_n"):
            continue
        stem = k[: -len(".force_n")]
        nk = next(
            (
                f"{stem}.{s}"
                for s in ((NAME_KEY[_head(stem)],) if _head(stem) in NAME_KEY else ("body_names", "link_names"))
                if f"{stem}.{s}" in z.files
            ),
            None,
        )
        if nk is None:
            raise InputMissingError(f"{MARK} {k} has no name key")
        out[stem] = (z[k], [str(x) for x in z[nk]])
        keys[k] = stem
    if not out:
        raise InputMissingError(f"{MARK} no force keys: {z.files[:6]}")
    return out, keys
