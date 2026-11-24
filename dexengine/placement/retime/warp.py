"""Time-warp resampling helpers shared by every retiming path (library, no CLI).

Arm actuators are velocity-limited, and retargeted references can demand more than that limit on some frames; the
simulator then pins those joints at the limit, so the tracking error there is physically mandated. The limit is the
real hardware value, so the fix is to slow the reference down where it is infeasible.

The whole clip shares ONE time axis, so retiming the source qpos keeps the object, finger and contact channels
synchronized by construction. Only the intervals that exceed the limit are stretched:

    r_t  = max_j |q_ref[t+1,j] - q_ref[t,j]| / dt / vlim_j     (both arms + every finger, retime.py)
    s_t  = max(1, r_t * safety)                                 (stretch factor)
    tau  = cumsum(s_t * dt)                                     (new time of frame t)

then every channel is resampled onto a uniform dt grid over [0, tau_end]. Feasible segments keep s_t = 1, i.e. they
are untouched in shape and speed.

Quaternions use sign-aligned normalized lerp — adjacent frames are < 1 deg apart at 50 Hz, where nlerp and slerp agree
well below the retarget noise floor.

The warp itself is computed by retime.py (after IK, from the combo's arm caps and the hand's applied finger limits);
this module holds the resampling helpers it and the bake import (``DT``, ``_quat_slots``, ``_euler_slots``,
``_resample``) and the hand-root spelling rule.
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.retime import stretch as _VST

DT = _VST.FRAME_DT  # the resample grid = the tree frame interval (owner: stretch.FRAME_DT)
HAND_DIM = 18


def _quat_slots(D: int, hand_qpos_dim: int = HAND_DIM) -> list[tuple[int, int]]:
    """(start, end) of every object quaternion block in a qpos row (two hand blocks of ``hand_qpos_dim``, then 7-wide
    object blocks: right object, left object, extra objects)."""
    h2 = 2 * hand_qpos_dim
    slots = []
    if h2 + 7 <= D:
        slots.append((h2 + 3, h2 + 7))  # right object
    if h2 + 14 <= D:
        slots.append((h2 + 10, h2 + 14))  # left object
    k = 0
    while h2 + 14 + 7 * k + 7 <= D:  # extra objects
        slots.append((h2 + 17 + 7 * k, h2 + 21 + 7 * k))
        k += 1
    return slots


def _euler_slots(D: int, hand_qpos_dim: int = HAND_DIM) -> list[tuple[int, int]]:
    """(start, end) of every Euler wrist-rotation block in a qpos row.

    Each hand block = pos(3) + rot_x/y/z(3) + finger joints. The rot channels are Euler angles wrapped to (-pi, pi]
    and can cross the boundary (a genuine +-360 deg jump in one step); linear interpolation across such a jump would
    sweep the wrist through the whole range, so callers unwrap these columns before resampling and re-wrap after.
    """
    return [(base + 3, base + 6) for base in (0, hand_qpos_dim) if base + 6 <= D]


def _resample(arr: np.ndarray, tau: np.ndarray, new_t: np.ndarray, quat_slots: list[tuple[int, int]]) -> np.ndarray:
    """Linear resample, with sign-aligned normalized lerp on quaternion blocks."""
    out = np.empty((len(new_t), arr.shape[1]), dtype=arr.dtype)
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(new_t, tau, arr[:, c])
    for lo, hi in quat_slots:
        q = arr[:, lo:hi].copy()
        # make the quaternion path continuous before interpolating
        for t in range(1, len(q)):
            if float(np.dot(q[t], q[t - 1])) < 0.0:
                q[t] = -q[t]
        qi = np.stack([np.interp(new_t, tau, q[:, j]) for j in range(hi - lo)], axis=1)
        n = np.linalg.norm(qi, axis=1, keepdims=True)
        out[:, lo:hi] = qi / np.maximum(n, 1e-12)
    return out


# ── hand root: one rotation has two Euler spellings ─────────────────────────────────────────────────────────────
# The three root rotation columns of a hand block are Euler angles (three hinge joints in series), and the SAME
# rotation has a second spelling (the convention owner's ``alt_euler``). The placement bake re-extracts the canonical
# spelling every frame, so where the middle angle passes 90 deg two neighbouring frames can be spelled on different
# branches: the rotation moves a few degrees while two columns jump by about pi. ``np.unwrap`` only removes multiples
# of 2*pi per column, so the per-column interpolation can sweep an inserted frame far from the motion it stands for.
# The rule below keeps the per-column interpolation and, only for an inserted frame that is off by more than
# ROOT_BRANCH_KEEP_DEG, takes whichever of the two spellings of the interval's end frame puts that frame nearer the
# true rotation. Every other output value keeps its bytes.

ROOT_BRANCH_KEEP_DEG = (
    5.0  # an inserted frame keeps its interpolated value while its rotation is within this of the true one
)
# (the geodesic interpolation of its two source frames). Measured over every baked hand
# side of the nine datasets x four hands: all but 9 sides stay under it untouched.
ROOT_TURN_MAX_DEG = 20.0  # reference mark: a retimed frame that sits further than this (as a ROTATION) from the
# original motion at the same instant is counted and the clip is marked — it is not killed.
# Same census: worst 5.3 deg once the rule above ran.


def _wrap(v):
    return (np.asarray(v, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


class RootSpelling:
    """The two Euler spellings of one root rotation, for a convention owner that knows its chain signs (``self.sign`` —
    retime_variants.BaseMap fits them to the model and checks ``alt_euler`` against the model's own FK)."""

    def alt_euler(self, q3):
        """The OTHER spelling of the same rotation (chain angles = column x sign: first + pi, pi − middle, third +
        pi)."""
        m = np.asarray(q3, dtype=np.float64) * self.sign
        return np.stack([m[..., 0] + np.pi, np.pi - m[..., 1], m[..., 2] + np.pi], axis=-1) * self.sign

    def canon_euler(self, q3):
        """The stored spelling (what the owner's from_world writes): columns in [−pi, pi), middle angle within 90
        deg."""
        w = _wrap(q3)
        bad = np.abs(w[..., 1]) > np.pi / 2
        return np.where(bad[..., None], _wrap(self.alt_euler(w)), w)


def _inside(tau: np.ndarray, new_t: np.ndarray):
    """(k, i, f): the output frames that fall STRICTLY inside a source interval, that interval, the fraction along
    it."""
    T = len(tau)
    if T < 2 or len(new_t) == 0:
        z = np.zeros(0, dtype=int)
        return z, z, np.zeros(0)
    i = np.clip(np.searchsorted(tau, new_t, side="right") - 1, 0, T - 2)
    k = np.flatnonzero((new_t > tau[i]) & (new_t < tau[i + 1]))
    i = i[k]
    return k, i, (new_t[k] - tau[i]) / (tau[i + 1] - tau[i])


def _truth(e: np.ndarray, i: np.ndarray, f: np.ndarray, rot):
    """The true rotation of an inserted frame: the geodesic interpolation of its two source frames."""
    R = rot(e)
    step = (R[i].inv() * R[i + 1]).as_rotvec()
    return R[i] * type(R).from_rotvec(step * np.asarray(f)[:, None])


def root_turn_bound_deg(e: np.ndarray, out_e: np.ndarray, tau: np.ndarray, new_t: np.ndarray) -> float:
    """Upper bound (deg) of how far any inserted frame's root ROTATION is from the true one — from the angle values
    alone, no model. Three hinges in series: changing the columns by d turns the rotation by at most sum|d|
    (bi-invariant metric, one triangle step per hinge). So for an output frame at fraction f of interval i,
    dist(out, truth) <= L1(out − e_i) + f*G and <= L1(out − e_{i+1}) + (1−f)*G, G = L1(e_{i+1} − e_i) >= the true step
    (all differences modulo 2*pi). A bound under a threshold proves it without reading the convention."""
    e = np.asarray(e, dtype=np.float64)
    k, i, f = _inside(tau, new_t)
    if len(k) == 0:
        return 0.0
    o = np.asarray(out_e, dtype=np.float64)[k]
    G = np.abs(_wrap(e[i + 1] - e[i])).sum(axis=1)
    b0 = np.abs(_wrap(o - e[i])).sum(axis=1) + f * G
    b1 = np.abs(_wrap(o - e[i + 1])).sum(axis=1) + (1.0 - f) * G
    return float(np.rad2deg(np.minimum(b0, b1).max()))


def root_branch_patch(out_e: np.ndarray, e: np.ndarray, tau: np.ndarray, new_t: np.ndarray, owner) -> tuple[int, int]:
    """Rewrite IN PLACE the inserted frames whose root rotation is more than ROOT_BRANCH_KEEP_DEG off the true one AND
    that the other spelling serves better: the per-column interpolation from frame i to the OTHER spelling of frame i+1
    (2*pi-aligned to frame i), stored back in the canonical form. Returns (intervals changed, frames changed).

    ``owner`` = the convention owner of this hand root (retime_variants.BaseMap — read from the model): ``rot``
    (columns -> rotation), ``alt_euler`` (the other spelling), ``canon_euler`` (the stored form). Nothing here knows an
    axis order or a sign. A frame the rule does not pick is not written — 0 frames = the array keeps its bytes."""
    e = np.asarray(e, dtype=np.float64)
    k, i, f = _inside(tau, new_t)
    if len(k) == 0:
        return 0, 0
    truth = _truth(e, i, f, owner.rot)
    d_cur = np.rad2deg((truth.inv() * owner.rot(np.asarray(out_e, dtype=np.float64)[k])).magnitude())
    m = d_cur > ROOT_BRANCH_KEEP_DEG
    if not m.any():
        return 0, 0
    k, i, f, d_cur, truth = k[m], i[m], f[m], d_cur[m], truth[np.flatnonzero(m)]
    a = e[i]
    oth = a + _wrap(np.asarray(owner.alt_euler(e[i + 1]), dtype=np.float64) - a)
    cand = np.asarray(owner.canon_euler(a + f[:, None] * (oth - a)), dtype=np.float64)
    w = np.rad2deg((truth.inv() * owner.rot(cand)).magnitude()) < d_cur
    if not w.any():
        return 0, 0
    out_e[k[w]] = cand[w]
    return int(len(np.unique(i[w]))), int(w.sum())


def root_turn_dev(e: np.ndarray, out_e: np.ndarray, tau: np.ndarray, new_t: np.ndarray, rot) -> np.ndarray:
    """The mark's exact measure, per retimed frame (deg): the rotation between that frame and the original motion at the
    same instant. ``rot`` = the convention owner's map from the three columns to a rotation — the angle VALUES are
    never differenced."""
    e = np.asarray(e, dtype=np.float64)
    T = len(e)
    if T < 2 or len(new_t) == 0:
        return np.zeros(len(new_t))
    src = np.interp(new_t, tau, np.arange(T, dtype=np.float64))
    i0 = np.clip(np.floor(src).astype(int), 0, T - 2)
    truth = _truth(e, i0, src - i0, rot)
    return np.rad2deg((truth.inv() * rot(np.asarray(out_e, dtype=np.float64))).magnitude())


def root_branch_apply(
    qpos: np.ndarray, out_qpos: np.ndarray, tau: np.ndarray, new_t: np.ndarray, slots, root_maps
) -> tuple[int, int]:
    """The resampler's hook: apply ``root_branch_patch`` to every hand-root slot of a resampled qpos (in place) and
    print the marker. ``root_maps`` = zero-argument callable -> the convention owners in slot order; it is called only
    when a slot is not already proven within ROOT_BRANCH_KEEP_DEG from the angle values alone, so a clip that needs
    nothing never loads the model and keeps its bytes. No default owner — a caller that cannot read one dies."""
    n_iv = n_fr = 0
    owners = None
    for j, (lo, hi) in enumerate(slots):
        if root_turn_bound_deg(qpos[:, lo:hi], out_qpos[:, lo:hi], tau, new_t) <= ROOT_BRANCH_KEEP_DEG:
            continue
        owners = root_maps() if owners is None else owners
        a, b = root_branch_patch(out_qpos[:, lo:hi], qpos[:, lo:hi], tau, new_t, owners[j])
        n_iv += a
        n_fr += b
    print(f"[ROOT_BRANCH] changed intervals {n_iv} · frames {n_fr}", flush=True)
    return n_iv, n_fr


def root_turn_mark(qpos, out_qpos, tau, hand_qpos_dim: int, *, root_maps) -> tuple[float, int, int]:
    """Reference mark of a retimed tree: (worst deg, frames over ROOT_TURN_MAX_DEG, hand blocks measured) — how far a
    retimed frame's hand-root ROTATION is from the original motion at the same instant, over every hand block. A block
    whose model-free upper bound is already under ROOT_TURN_MAX_DEG has no such frame and contributes that bound (so
    the worst is an "at most"); otherwise the convention owner is read and the rotation itself is measured. The caller
    marks the clip and reports the count — one clip never stops a bundle."""
    qpos = np.asarray(qpos, dtype=np.float64)
    out_qpos = np.asarray(out_qpos, dtype=np.float64)
    new_t = np.arange(len(out_qpos)) * DT
    worst, over, exact, owners = 0.0, 0, 0, None
    for j, (lo, hi) in enumerate(_euler_slots(qpos.shape[1], hand_qpos_dim)):
        v = root_turn_bound_deg(qpos[:, lo:hi], out_qpos[:, lo:hi], tau, new_t)
        if v > ROOT_TURN_MAX_DEG:
            owners = root_maps() if owners is None else owners
            d = root_turn_dev(qpos[:, lo:hi], out_qpos[:, lo:hi], tau, new_t, owners[j].rot)
            v = float(d.max()) if len(d) else 0.0
            over += int((d > ROOT_TURN_MAX_DEG).sum())
            exact += 1
        worst = max(worst, v)
    return worst, over, exact
