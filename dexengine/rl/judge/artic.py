"""Articulation θ judge: the one owner of the failure predicate that says whether an articulated clip opened or
closed its object.

Failure = the joint error |θ − θ_ref| (in the joint type's unit) stays above the threshold for the hold time
**consecutively**. The training termination term and the fresh-process judge's termination term are the same: the
judge builds the same env as training, and that env's termination term (`ObjArtiThetaJudge` in
dexengine.rl.tasks.mdp.rewards.task, registered by env setup) calls only this module. The judge's success is
"no early termination", so an episode this term ends is a failure.

Units (`err_units`; the reward `RewardObjArtic` reads this function too):
  hinge = rad as is · slide = |Δ| / range (the joint limit range of the joint spawned in that env, m/m).
  The same k then means "per hinge rad" and "per slide range": the median hinge range is 0.99 rad, so both are on the
  same scale. Measured in m, a slide wrong over its whole range (median 0.26 m) would still earn a reward of 0.46 (no
  signal).

Thresholds (`THETA_JUDGE`): hinge 0.5 rad · slide 0.5 of the range · 0.5 s consecutive. Envs parked by the judge
(PARK) are not counted: the PARK owner hands the park mask to the term's `set_parked` (`parked` of
`ThetaOutCounter.update`).
Marker: the env prints `[JUDGE_TERM] artic_theta=…` (judge/terms.py artic_marker · threshold text `describe()`).

Depends on torch only (testable without Isaac).
"""

from __future__ import annotations

import math
from typing import NamedTuple


class ThetaJudge(NamedTuple):
    """θ judge thresholds: hinge rad · slide fraction of the range · consecutive hold time (s). All finite positive."""

    tol_hinge_rad: float
    tol_slide_frac: float
    hold_s: float


#: hinge 0.5 rad · slide 0.5 of its range · 0.5 s consecutive.
THETA_JUDGE = ThetaJudge(tol_hinge_rad=0.5, tol_slide_frac=0.5, hold_s=0.5)


def _check(j: ThetaJudge | None) -> ThetaJudge:
    """None = `THETA_JUDGE` (read at call time, not bound to a default argument). Thresholds must be finite and
    positive: 0, negative, NaN or inf would silently end every env or none."""
    j = THETA_JUDGE if j is None else j
    for name, v in j._asdict().items():
        if not (v > 0) or not math.isfinite(v):  # NaN fails here too
            raise ValueError(f"[ARTIC][theta-judge] {name}={v!r} — only a finite positive value")
    return j


def describe(j: ThetaJudge | None = None) -> str:
    """The threshold text carried by the marker."""
    j = _check(j)
    return (
        f"hinge |Δθ|>{j.tol_hinge_rad:g} rad · slide |Δ|/range>{j.tol_slide_frac:g} · "
        f"failure after {j.hold_s:g} s in a row · parked judge envs are not counted"
    )


def err_units(theta, ref, is_slide, span):
    """Joint error in the joint type's unit: hinge rad · slide fraction of the range. All (N,) tensors (is_slide
    bool · span > 0). Hinge rows are exactly `(ref − θ).abs()` (torch.where selects them)."""
    import torch

    err = (ref - theta).abs()
    return torch.where(is_slide, err / span, err)


def hold_steps(step_dt: float, j: ThetaJudge | None = None) -> float:
    """The hold time in policy steps (at least 1)."""
    j = _check(j)
    if not step_dt > 0:
        raise ValueError(f"[ARTIC][theta-judge] step_dt={step_dt!r} — must be positive")
    return float(max(1, math.ceil(j.hold_s / step_dt - 1e-9)))


class ThetaOutCounter:
    """Per env, the number of consecutive steps outside the threshold; update returns the failure mask.
    Reset envs and envs parked by the judge count again from 0."""

    def __init__(self, num_envs: int, device, step_dt: float, j: ThetaJudge | None = None):
        import torch

        j = _check(j)
        self.judge = j
        self.hold = hold_steps(step_dt, j)
        self.count = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._tol = torch.tensor([j.tol_hinge_rad, j.tol_slide_frac], dtype=torch.float32, device=device)

    def update(self, err_u, is_slide, parked=None):
        """err_u (N,) error in type units · is_slide (N,) bool · parked (N,) bool | None (nothing parked) → failure
        (N,) bool. Parked envs are not counted even outside the threshold (reset to 0), so θ drift while parked
        cannot cause a reset while parked."""
        import torch

        tol = self._tol[is_slide.long()]
        out = err_u > tol
        if parked is not None:
            out = out & ~parked
        self.count = torch.where(out, self.count + 1.0, torch.zeros_like(self.count))
        return self.count >= self.hold

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.count.zero_()
        else:
            self.count[env_ids] = 0.0
