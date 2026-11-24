"""PARK: park finished envs between judge batches (DEXENGINE_EVAL_PARK) — the rule, its values and its verdict.

PARK is part of the judge standard. The fresh-process judge (FROZEN_REPLAY) turns it on by default; the one owner of
that default is judge/launch.py `resolve_park` (an explicit DEXENGINE_EVAL_PARK=0 turns it off and leaves a marker).
Every other context (in-training eval and the like) has it off, and this module only reads the variable (unset = 0).
The per-batch state and loop hooks are `park_batch.Park`; the E-clip report is `park_e_oracle`.

Why: a batch cut (two full resets) teleports all envs in one frame. An object held in the air off the desk then misses
its desk-contact generation, SPAWN_HOLD re-pins that overlapping pose every step, and the object free-falls. With PARK
on, an env that finished its first episode stands at its frame-0 pose until the cut: zero action, episode_length_buf 0,
objects pinned to their frame-0 reference pose, wrist reference at frame 0 (the park mask goes to the action terms'
`set_parked`: no training wrist lookahead, zero arm feed-forward velocity). The cut of every real batch therefore
happens under the warm-up's conditions (objects on the desk), with the parked hand and the pinned object at the same
frame 0.

PARK gates (checked at every batch cut; this module owns the rules and the values; the decision is the pure function
`park_verdict`). Only gates C and D, which measure real falls, make a judge invalid; either one firing is enough.
Gates A and B are notes (reported numbers, never invalid): gate A is an early sign of loss (moving off the pinned pose
at the cut) and also reads the one-step push of frame-0 overlaps (object<->object, robot links, hinges) as loss.
The score does not depend on parking (the cut comes after the episode ends).
Symbols: N = cut samples (finished env × object side) · E = parked-hand-touch samples · q = PARK_Q · n = batch envs.
- |dz| = object z at the cut − z of the frame-0 pinned pose (the pose parking actually used; owner
  `object_pin.pin_pose`: the reference pose raised by its penetration into the static support). The lift (data
  embedded into the desk or a disc) is printed on the same line per source (`object_pin.LIFT_SOURCES`). The data
  contract owns the allowed embedding (20 mm); DZ_LIMIT_MM (10 mm) owns the allowed parked residual. Grounds: loss is
  triggered by an object lifted ≈1 cm or more off the desk at the cut; the parked residual |dz| is the response of one
  physics step after pinning (p99 ≈3.1 mm on TACO, ≈6.5 mm on overfit clips; 185–237 mm without parking).
- Parked-hand touch (E): among the envs parked at the cut step (the batch's last step; the last park mask sent), the
  (env, side) whose hand links (both hands) push on that side's object with a PhysX filtered contact force above
  TOUCH_F_MIN. Measurement owner `hand_object_contact` (env method `hand_object_contact_force`), read once at the cut.
  Objects are re-pinned every step, so |dz| at the cut is the response of the last step, and a touch in that step is
  its cause (a park-start window would drop more samples and could hide a real loss after the hand let go). An env
  that finished on the last step and never parked is not E. Why exclude: a (clip, side) whose hand touches or
  penetrates its object at frame 0 has the parked robot push the pinned object every step, so |dz| reads tens of mm,
  which is not loss (free fall above a support). Liveness: a parked env whose own fingertip sensor touches must also
  read as touching here, otherwise the judge dies.
- Gate-A exclusion: when a majority of a (clip, side)'s samples exceed DZ_LIMIT_MM in gate A's quantity (|dz| with E
  set to 0; i.e. that (clip, side)'s median E-adjusted |dz| is above the threshold), all of its samples leave this
  batch's gate checks (A–C) only. They still count in the score (batch totals, completion rate): the cut follows the
  episode, so those completions are not affected by parking, and the first-frame push is the same as in training.
  Why exclude from the gates: a clip pushed with the same sign in every env and every cut is frame-0 geometry (e.g.
  object<->object hull overlap), not loss from the previous cut, and with ≈1.6 % of N per clip one clip would make a
  whole bundle invalid. The majority rule keeps scattered, env-dependent exceedances (real loss) in gate A. Nothing is
  hidden: every excluded (clip, side) is printed with name, side and median |dz| in [EVAL_PARK_EXCL_A] and kept in the
  judge record. Cap: excluded (clip, side) above EXCL_A_MAX_FRAC of the batch's (clip, side) give a gate-A note —
  first-frame pushes (data) are 5–10 % of a bundle, a broken park device pushes most (clip, side) together.
- Gate A (note): in the samples left after the exclusion, E samples count |dz| = 0, and p99 > DZ_LIMIT_MM gives a note
  (with E = ∅ and no exclusion this is the plain |dz| p99).
- Gate B (floor, note): fewer remaining non-E samples than `non_e_floor(N)` (N = all cut samples; FLOOR_MULT times
  gate A's allowed exceedances (1 − q)·N) gives a note. It guards against "everything excluded = free pass" (by E or by
  exclusion) and against a measurement that reads everything as touching. The E fraction has no cap: it is a property
  of the dataset (pieces that start grasped), not a failure signal.
- Gate C (over-lift): among the lifted samples (lift > 0 states "rests on a support"), those that at the cut fell one
  step freely from the pinned pose (`pin_fall_mask`: velocity signature `free_fall_mask` and displacement
  dz = z − pinned z ≤ −(drop − DROP_TOL_MM)). More than 1 − q of N is invalid; every sample counts, E included (a
  one-step fall is invisible in |dz|). drop = the distance fallen from rest in n physics substeps of semi-implicit
  Euler, g·h²·n(n+1)/2 (`pin_drop_mm`; h = physics_dt, n = step_dt/h, read from the env). Why the displacement too:
  an object pushed up off its support by a frame-0 overlap (object<->object, accepted as data) has the same velocity,
  but its displacement differs from drop (pushed up, it falls less). An over-lifted object falls from the pinned pose,
  so its displacement is drop; an over-lift smaller than drop touches within the step and leaves no velocity signature,
  so the displacement clause loses no over-lift. One-sided: samples that went further down (pushed down) still count.
  Self-proof: when the same cut has at least AIR_MIN airborne samples (`air_drop`: lift 0, velocity signature, not E,
  inside the gravity reach `gravity_reach_mask`, not an articulated part `artic_asset`), their median displacement
  must lie within −drop ± DROP_TOL_MM. Outside it, the reference formula disagrees with this physics setup and gate C
  cannot be measured: a note when the real falls and gate D are 0, invalid otherwise (judge/park_recount.py). With
  fewer samples the marker says there is no check. Why the last two clauses: position correction (overlaps, hands;
  it leaves no velocity) also gives pushed objects the velocity signature, while an object accelerated only downward
  from rest cannot travel farther than |v_z|·T (whatever the substep count and integrator, so samples that break the
  formula remain). Articulated parts hang on their hinge and do not fall freely. The dropped candidates are printed
  per (clip, side) in [EVAL_PARK_AIR].
- Gate D (GHOST): in the two steps G1 → G2 after the object hold window (SPAWN_HOLD, `d_steps`), count per side the
  objects whose pinned pose rests on a support (|gap| of `pin_support_gap` ≤ REST_GAP_MM: the desk, or a disc the
  object touches in physics), with zero hand-link contact force in both steps and the accumulated free-fall signature
  (`fall_accum_mask`: one-step Δv_z = −g·dt ±2 %, v_z ≤ −2·g·dt, no horizontal motion). A count above (1 − q)·n is
  invalid (every batch, warm-up included); it is 0 in a sound environment. Held objects (SPAWN_HOLD, PARK) are
  rewritten to zero velocity every step and cannot show the accumulated signature. A batch that ended before G2 could
  not be measured and is invalid.
  Blind spot: an E sample pushed by the hand shows no free-fall signature and is invisible to gate D. Its harm is
  measured by an independent oracle, the warm-up <-> real-batch completion of E clips (`park_e_oracle`, a report, not
  a gate); `park_batch.Park` hands it the E (clip, side) counts of every batch (`e_rows`).

Markers. The token "INVALID(" is printed only when a batch is invalid: the judge sidecar reads a row as invalid by
that token.
- `[EVAL_PARK_CUT] batch k: parked-hand touch (at the cut) R e/n_env(…) · L … · hand links … · fingertip sensor
  check …` — parked-hand
  touch at the cut, one line per cut (the sidecar checks the count = warm-up + real batches).
- `[PARK] batch k: lookahead=0 parked=m · action terms t · parked wrist ↔ ref frame 0 …` — proof that every action
  term used the
  last step's park mask (otherwise the judge dies) and the parked-wrist distance to the frame-0 reference. A probe adds
  the reset-wrist-motion column (a pilot gate that drops no clip); each (clip, hand) over its limit is an
  [EVAL_PARK_CLIPS] row.
- `[EVAL_PARK] batch k: parked … · |dz| p99 · … · gate D free fall …[ — INVALID(gate C|D — …)]
  [ · NOTE(gate A|B|C — …)]` — the cut verdict. An invalid reason always has the form `INVALID(gate X — …)` (the
  sidecar's park_gates reads the gate from it); note segments (`NOTE(…)`) never contain the token INVALID.
- `[EVAL_PARK_EXCL_A] batch k: c<n> <clip> <side> <median |dz|> mm · …` — one line per gate-A-excluded (clip, side),
  plus a count line (also when 0).
- `[EVAL_PARK_EXCL] batch k: excluded (clip, side) …` — the excluded E samples per (clip, side), always printed.
- `[EVAL_GHOST] batch k: gate D count R n / L n (…)` — gate D's count over G1 → G2 after the hold window, with the
  count after
  each filter (on a support, no hand touch, accumulated signature).
"""

from __future__ import annotations

import math
import os

import torch

from dexengine.rl.judge.park_recount import proof_reference, proof_text

ENV_VAR = "DEXENGINE_EVAL_PARK"
# Policy-free PARK probe (pre-flight check): value = the cut steps "N[,N2…]" (unset or 0 = off). See probe_steps.
PROBE_VAR = "DEXENGINE_PARK_PROBE"
# Object hold window (steps after a reset); the window's owner is tasks/scene/spawn_hold (same variable name)
HOLD_VAR = "DEXENGINE_OBJ_SPAWN_HOLD"
# Gate D's first measurement = hold window + this many steps (kept apart from the one-step fall of the last held step);
# the second is the step after
D_AFTER_HOLD = 2
# Gate D "on a support": allowed |gap| between the pinned pose's lowest vertex and the support top (equal to
# DZ_LIMIT_MM by value only)
REST_GAP_MM = 10.0
FF_TOL = 0.02  # free-fall signature: relative tolerance on v_z (±2 %)
FF_VXY = 0.005  # free-fall signature: horizontal speed limit [m/s]
# Gate C displacement tolerance [mm]. Measured on airborne, unpushed (clip, side): the one-step fall from the pinned
# pose is −2.91…−3.10 mm (reference −2.94); samples pushed off their support by overlaps give −0.59…+7.4 mm. The
# velocity tolerance (FF_TOL, 2 %) used on the displacement would be ±0.06 mm and miss real falls.
DROP_TOL_MM = 0.5
AIR_MIN = 20  # gate C self-proof: airborne samples needed before their median displacement is checked
# Numeric slack of the gravity reach: float32 position (ulp 1.2e-4 mm at 1 m), not a physics tolerance
REACH_EPS_MM = 0.01
DZ_LIMIT_MM = 10.0
# Reset-wrist tolerance (probe gate): same value as DZ_LIMIT_MM, different meaning. The cut is step 3 because the desk
# push happens in the first physics step.
RESET_WRIST_MM = 10.0
RESET_WRIST_CUT = 3
EXCL_A_MAX_FRAC = 0.25  # gate A cap: a larger fraction of gate-A-excluded (clip, side) is a device failure, not data
# The one owner of the allowed quantile: gate A (p99) · gate B floor · gate C (1 − q of samples) · gate D (1 − q of
# envs)
PARK_Q = 0.99
FLOOR_MULT = 10  # gate B floor = this many times gate A's allowed exceedances, in non-E samples
TOUCH_F_MIN = 0.0  # N — above it is a touch (PhysX reports exactly 0 without an active contact)


def _flag(name: str) -> bool:
    v = (os.environ.get(name, "0") or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"[EVAL_PARK] {name}={v!r} — only 0 or 1")
    return v == "1"


def enabled() -> bool:
    """Whether PARK is on in this process (DEXENGINE_EVAL_PARK = 0 | 1, unset = 0)."""
    return _flag(ENV_VAR)


def probe_steps() -> tuple[int, ...]:
    """The cut steps of the policy-free PARK probe (empty tuple = off).

    When on, every env of the batch parks right after the reset (what before_step does for finished envs), and at each
    cut at_cut measures the same gates (A–D) as a judge. All actions are zero, so only the data, the robot assets and
    the env's frame-0 physics are tested (pre-flight check). Several cuts measure the sensitivity to the parking length
    (the batch ends at the last cut). Accepts a comma list of positive integers only, and dies when PARK is off (there
    would be nothing to measure)."""
    v = (os.environ.get(PROBE_VAR, "") or "0").strip()
    if v == "0":
        return ()
    parts = v.split(",")
    if not all(x.isdigit() and int(x) > 0 for x in parts):
        raise ValueError(f"[PARK_PROBE] {PROBE_VAR}={v!r} — only a comma list of positive integers (cut steps)")
    if not enabled():
        raise RuntimeError(f"[PARK_PROBE] {PROBE_VAR}={v} but {ENV_VAR} is off — nothing to measure without parking")
    return tuple(sorted({int(x) for x in parts}))


def d_steps() -> tuple[int, int]:
    """The two loop steps (G1, G2) that measure gate D: hold window (HOLD_VAR, unset = 0) + D_AFTER_HOLD, and the next.
    Loop step s starts at episode length s − 1 after a reset and objects are held while length < window, so with a
    window N steps 1…N are held and steps from N + 1 are free."""
    from dexengine.rl.tasks import switches  # stdlib only

    v = (switches.text(HOLD_VAR, "0") or "0").strip()
    if not v.isdigit():
        raise ValueError(f"[EVAL_GHOST] {HOLD_VAR}={v!r} — only an integer >= 0")
    g1 = int(v) + D_AFTER_HOLD
    return g1, g1 + 1


def free_fall_mask(v: torch.Tensor, gdt: float) -> torch.Tensor:
    """(n,) — whether the linear velocity v (n,3) is the one-step free-fall signature: v_z = −g·dt ±2 % and
    |v_xy| < 5 mm/s (nothing touched it and it fell for one step). Gate C and gate D's accumulated signature use it."""
    return ((v[:, 2] + gdt).abs() <= FF_TOL * gdt) & (v[:, :2].norm(dim=-1) < FF_VXY)


def fall_accum_mask(v1: torch.Tensor, v2: torch.Tensor, gdt: float) -> torch.Tensor:
    """(n,) — whether the velocities of two consecutive steps v1 → v2 (n,3) are an accumulated free fall (gate D): the
    one-step change is the free-fall signature (`free_fall_mask(v2 − v1)`), v2_z ≤ −2·g·dt(1 − FF_TOL) (falling for more
    than two steps; a held object, rewritten to zero velocity every step, cannot show it) and |v2_xy| < FF_VXY."""
    return free_fall_mask(v2 - v1, gdt) & (v2[:, 2] <= -2.0 * gdt * (1.0 - FF_TOL)) & (v2[:, :2].norm(dim=-1) < FF_VXY)


def pin_drop_mm(g: float, physics_dt: float, step_dt: float) -> float:
    """The z distance [mm] an object falls in one env step from rest (the pinned pose) without touching anything:
    n = step_dt / physics_dt physics substeps (a non-integer n dies) of semi-implicit Euler (velocity first, then the
    position with the new velocity): |g|·h²·n(n+1)/2."""
    n = step_dt / physics_dt
    k = round(n)
    if k < 1 or abs(n - k) > 1e-6:
        raise ValueError(
            f"[EVAL_PARK] step_dt {step_dt} / physics_dt {physics_dt} = {n} — not a whole number of substeps"
        )
    return abs(g) * physics_dt**2 * k * (k + 1) / 2.0 * 1000.0


def pin_fall_mask(v: torch.Tensor, dz_mm: torch.Tensor, gdt: float, drop_mm: float) -> torch.Tensor:
    """(n,) — gate C's samples: fell one step freely from the pinned pose at the cut — velocity signature
    (`free_fall_mask`) and signed displacement dz_mm (z − pinned z) ≤ −(drop_mm − DROP_TOL_MM) (a sample pushed up that
    fell less is not one; one-sided)."""
    return free_fall_mask(v, gdt) & (dz_mm <= -(drop_mm - DROP_TOL_MM))


def gravity_reach_mask(v_z: torch.Tensor, dz_mm: torch.Tensor, step_dt: float) -> torch.Tensor:
    """(n,) — whether the displacement dz_mm (z − pinned z) lies in the one-step reach of an object accelerated only
    downward from rest, −(|v_z|·T + ε) ≤ dz ≤ ε. Outside = a non-gravity displacement (position-correction push) was
    added, and the sample says nothing about the reference formula. ε = REACH_EPS_MM."""
    lim = v_z.abs() * step_dt * 1000.0 + REACH_EPS_MM
    return (dz_mm >= -lim) & (dz_mm <= REACH_EPS_MM)


def artic_asset(obj) -> bool:
    """Whether the asset is articulated: only an Isaac Lab Articulation has `num_joints` (a RigidObject has none; 0 =
    rigid)."""
    return int(getattr(obj, "num_joints", 0) or 0) > 0


def air_drop(
    dz_mm: torch.Tensor,
    lift_mm: torch.Tensor,
    ff_v: torch.Tensor,
    touch: torch.Tensor,
    reach: torch.Tensor,
    artic: torch.Tensor,
) -> tuple[int, float | None]:
    """Gate C self-proof measurement: the number of airborne samples (lift 0, velocity signature ff_v, no hand touch,
    inside the reach, not an articulated part) and their median displacement dz_mm (None without samples)."""
    m = (lift_mm <= 0) & ff_v.to(torch.bool) & ~touch.to(torch.bool) & reach.to(torch.bool) & ~artic.to(torch.bool)
    k = int(m.sum())
    return k, (float(dz_mm[m].median()) if k else None)


def excl_a_mask(dz_mm: torch.Tensor, excl: torch.Tensor, group: torch.Tensor) -> tuple[torch.Tensor, list]:
    """Gate-A exclusion (module doc), a pure function. dz_mm (N,) · excl (N,) bool (E) · group (N,) long ((clip, side)
    index). Returns (samples to drop (N,) bool, [(group, exceedances, non-E count, sample count, non-E median |dz|
    mm)]).
    Rule: a majority of the (clip, side)'s samples exceed DZ_LIMIT_MM in gate A's quantity (E = 0), so the one or two
    non-E samples of a mostly-E (clip, side) stay in gate A (same quantity, same denominator as gate A)."""
    ne = ~excl.to(torch.bool)
    over = (dz_mm > DZ_LIMIT_MM) & ne
    out = torch.zeros_like(ne)
    rows = []
    for g in torch.unique(group).tolist():
        m = group == g
        n, o = int(m.sum()), int((m & over).sum())
        if 2 * o > n:
            out |= m
            rows.append((int(g), o, int((m & ne).sum()), n, float(dz_mm[m & ne].median())))
    return out, rows


def non_e_floor(n: int) -> int:
    """Gate B floor: the minimum non-E sample count ⌈FLOOR_MULT·(1 − q)·n⌉, rounded to 6 decimals before the ceiling so
    the float noise of (1 − 0.99) cannot raise the bound by one."""
    return math.ceil(round(FLOOR_MULT * (1.0 - PARK_Q) * n, 6))


def park_verdict(
    dz_mm: torch.Tensor,
    excl: torch.Tensor,
    lift_mm: torch.Tensor,
    ff: torch.Tensor,
    d_count: dict,
    n_env: int,
    excl_a: torch.Tensor | None = None,
    d_pending: bool = False,
    n_grp: int = 0,
    n_grp_a: int = 0,
    drop: tuple | None = None,
) -> dict:
    """Gates A–D (module doc), a pure function. dz_mm · lift_mm (N,) mm · excl · ff · excl_a (N,) bool, all in the same
    sample order (ff = gate C's samples, `pin_fall_mask`; excl_a = gate-A-excluded samples, None = none).
    drop = gate C self-proof (reference drop mm, airborne sample count, airborne median displacement mm | None)
    (`pin_drop_mm` · `air_drop`; at_cut always passes it, None = no proof, for tests of the pure decision only).
    d_count = side letter → gate D count (None = not measured) · n_env = envs gate D counted over ·
    d_pending = a policy-free probe cut comes before gate D's measuring step G2 (not yet, rather than not measured;
    only true in a probe, a judge batch never ends before G2) · n_grp = the batch's (clip, side) count ·
    n_grp_a = how many of them are gate-A-excluded (cap EXCL_A_MAX_FRAC).
    Returns the numbers, the invalid reasons `why` (gates C and D only; empty = valid; each
    `INVALID(gate X — …)`) and the notes `note` (gates A and B and a failed gate-C self-proof without real falls;
    reports that never invalidate; each `NOTE(gate X — …)`, never containing the token INVALID)."""
    n = int(dz_mm.numel())
    excl = excl.to(torch.bool)
    keep = torch.ones_like(excl) if excl_a is None else ~excl_a.to(torch.bool)
    eff = torch.where(excl, torch.zeros_like(dz_mm), dz_mm)
    over = dz_mm > DZ_LIMIT_MM
    nk = int(keep.sum())
    v = {
        "n": n,
        "p99": float(torch.quantile(dz_mm, PARK_Q)),
        "over": int(over.sum()),
        "n_a": n - nk,
        "n_excl": int((excl & keep).sum()),
        "excl_over": int((over & excl & keep).sum()),
        "p99_eff": float(torch.quantile(eff[keep], PARK_Q)) if nk else 0.0,
        "rest_over": int((over & ~excl & keep).sum()),
        "n_ff": int(((lift_mm > 0) & ff & keep).sum()),
        "floor": non_e_floor(n),
        "ghost_lim": (1.0 - PARK_Q) * n_env,
        "ghost": dict(d_count),
        "n_grp": n_grp,
        "n_grp_a": n_grp_a,
    }
    v["n_ne"] = nk - v["n_excl"]
    why, note = [], []
    if v["p99_eff"] > DZ_LIMIT_MM:
        note.append(f"NOTE(gate A — p99 after exclusion > {DZ_LIMIT_MM:.0f} mm)")
    if n_grp_a > EXCL_A_MAX_FRAC * n_grp:
        note.append(f"NOTE(gate A — gate-A excluded (clip, side) {n_grp_a}/{n_grp} > {EXCL_A_MAX_FRAC:.0%})")
    if v["n_ne"] < v["floor"]:
        note.append(f"NOTE(gate B — non-E samples {v['n_ne']}/{n} < floor {v['floor']})")
    if v["n_ff"] > (1.0 - PARK_Q) * max(nk, 1):
        why.append(
            f"INVALID(gate C — {v['n_ff']}/{nk} lifted-object samples fall freely above the support — "
            "fixed-pose lift too large · stated > actual penetration)"
        )
    if drop is not None:
        d_mm, k_air, med = drop
        v.update(drop_mm=d_mm, n_air=k_air, air_med=med)
        if k_air >= AIR_MIN and abs(med + d_mm) > DROP_TOL_MM:
            ref = proof_reference(v["n_ff"], d_count)
            (note if ref else why).append(proof_text(ref, d_mm, k_air, med))
    if not d_count and not d_pending:
        why.append("INVALID(gate D — no free-fall count in this batch — cannot measure after the hold window)")
    for s, c in d_count.items():
        if c is None:
            why.append(f"INVALID(gate D — {s} count missing — cannot measure)")
        elif c > v["ghost_lim"]:
            why.append(
                f"INVALID(gate D — free fall {s} {c} > {v['ghost_lim']:.2f} — "
                "untouched objects on a support fall after the hold window)"
            )
    v["why"], v["note"] = why, note
    return v
