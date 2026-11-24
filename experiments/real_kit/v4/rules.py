"""Real fine-tuning data v4 — a fixed start pose + a scripted approach.

Why: in the v2 sets the frame-0 arm pose differs per placement variant (wrist 5–14 cm, joints 3–4.4 rad across IK
branches), so the policy could tell the placement from the start state alone. v4: every episode starts from **one fixed
whole-robot pose** (BASE_POSE); a scripted approach (no RL) brings the hand to the variant's frame 0, and from frame 0
the same per-clip RL policy as v2 takes over.

This module is the **single owner** of the combo, the fixed pose candidates and the chosen name, the relaxed fingers,
the approach rule values, the approach route and the schedule. Readers: stage_dp (--v4-base, clip IK seed), the bake
(passes --v4-base only with AUG_V4=1), plan.py (approach schedule — functions in timing.py) and the augment session
(AUG_V4_APPROACH — approach physics, phys.py). Off (no AUG_V4) = every caller unchanged.

Coordinates: routes and IK are in stage_dp's IK frame (robot body — the frame of ArmIK.fk, palm frame). Palm frame
axes:
  −z = finger direction · +x = palm normal.
"""

from __future__ import annotations

import math
import os

import numpy as np

MARK = "[AUG_V4]"
ENV = "AUG_V4"  # bake switch — "" off · "1" on · anything else dies
ENV_BASE = "AUG_V4_BASE"  # explicit candidate name (only for pilots of other candidates) — default BASE_POSE_NAME
COMBO = ("rby1", "wuji2")  # (arm, hand) of the v4 sets — read by plan, stitch, run and render

# ── fixed pose (both arms, rby1 joint order arm_0..arm_6)
# ─────────────────────────────────────────────────
# rule: one for the whole robot · a natural ready pose (hands above and in front of the desk) · not copied from any
# clip's
# frame 0 · clear of every variant's object place and the desk. The values come from IK on palm targets (scene frame);
# left and right are mirrors (left arm = the mirrored solution of the right arm, FK mirror error 0).
# palm (scene; robot body x=−0.70, desk front edge x=−0.35, top z=0.76) · finger direction F · palm normal N (right
# hand).
BASE_POSE_CANDIDATES: dict[str, dict[str, tuple[float, ...]]] = {
    # P2 raised ready — palm (−0.34, ∓0.22, 1.05) · F forward-up (0.51, 0.26, 0.82) · N inward (0.05, 0.94,
    # −0.33) ·
    # elbow −2.11
    "P2_hands_up_palms_in": {
        "right": (-0.039, -0.24, -0.002, -2.109, 0.045, -0.46, 0.052),
        "left": (-0.039, 0.24, 0.002, -2.109, -0.045, -0.46, -0.052),
    },
    # P3 low ready — palm (−0.38, ∓0.26, 0.97) · F forward (0.96, 0.27, 0.11) · N down (0.03, 0.30, −0.96) ·
    # elbow −2.52
    "P3_low_palms_down": {
        "right": (0.358, -0.838, 0.048, -2.517, 0.36, 0.358, 0.183),
        "left": (0.358, 0.838, -0.048, -2.517, -0.36, 0.358, -0.183),
    },
    # P4 ready — palm (−0.36, ∓0.24, 1.00) · F forward (0.92, 0.31, 0.26) · N inward (−0.20, 0.91, −0.36) ·
    # elbow −2.47
    "P4_ready_palms_in": {
        "right": (0.21, -0.565, -0.001, -2.473, -0.16, 0.314, -0.072),
        "left": (0.21, 0.565, 0.001, -2.473, 0.16, 0.314, 0.072),
    },
}
BASE_POSE_NAME = "P4_ready_palms_in"

# relaxed fingers (wuji2, by joint-name suffix rule — unseen names die): slightly bent, not straight.
RELAXED_RULES = (
    ("thumb_cmc_flex", 0.25),
    ("thumb_cmc_abd", -0.20),
    ("thumb_mcp", 0.20),
    ("thumb_ip", 0.20),
    ("mcp_flex", 0.30),
    ("mcp_abd", 0.0),
    ("pip", 0.40),
    ("dip", 0.25),
)

# ── approach rule values
# ──────────────────────────────────────────────────────────────────────────────────
BACKOFF_STEPS = (0.08, 0.06)  # backoff distances tried when the entry IK is blocked (within the 6–10 cm design range)
BACKOFF_RING_EL = (
    0,
    30,
    60,
)  # direction ring tried when all three rules are blocked (elevation °) · azimuth step BACKOFF_RING_AZ °
BACKOFF_RING_AZ = 45
# the last fraction of the joint-space share left out of the margin check (around the backoff point the entry decides)
ROUTE_TAIL_FREE = 0.2
FIRST_MOVE_M = 0.01  # first motion direction = direction of the first frame where the wrist leaves frame 0 by this much
FIRST_MOVE_MAX_S = 1.5  # not leaving within it = no direction (→ palm-normal rule)
# cap on the downward component of the backoff direction (coming from further below = under the desk or object — next
# rule)
DOWN_MAX = 0.25
PATH_STEP_M = 0.004  # straight entry sample spacing (m)
SPEED_RATIO = 1.0  # approach wrist peak speed target = clip wrist speed p90 × this value
# schedule = a hand-set speed profile (the route stays; only the timing on σ): accelerate (TIME_ACCEL_S, 0 → cruise,
# min-jerk form) → cruise (σ speed = limit ÷ SAFETY) → join (TIME_JOIN_S, cruise → end speed, end acceleration 0).
# End
# speed = the bake reference's frame-0 wrist speed (mean of the first JOIN_REF_FRAMES) turned into σ speed along the
# route
# end tangent — the approach does not stop at frame 0 but joins at the reference speed (an end speed of 0 left the arm
# nearly still before the join).
TIME_ACCEL_S = 0.5
TIME_JOIN_S = 0.4
JOIN_REF_FRAMES = 2
JOIN_LOW_FRAC = 0.1  # join stop gate: active arm joint speed < approach peak × this value = slow
# within the join ± this time, the slow run touching the join ≤ the run touching the v2 policy's first step +
# tolerance
JOIN_LOW_WIN_S = 0.5
JOIN_LOW_TOL_STEPS = 1  # tolerance (control steps)
# join floor: when the planned end joint speed (active arm norm) joining at the reference speed is below JOIN_LOW_FRAC
# ×
# approach peak, the end speed is raised to that threshold × JOIN_FLOOR (other variants keep the reference; 0 = off).
# Right
# after the join the v2 policy speeds up at once to 1.6–3.0 rad/s
# — variants with a much slower reference sagged below the threshold at the approach end only.
JOIN_FLOOR = 1.2
JOIN_FLOOR_TAG = "[AUG_V4_JOIN_FLOOR]"
# floor of the wrist peak speed target (m/s) — target = max(p90 × SPEED_RATIO, this). Joint-space approaches move the
# hand
# 30–70 cm, so slow clips (p90 ≈ 8 cm/s) at the p90 cap took 4–11 s; the floor 0.30 m/s shortens those approaches.
V_TARGET_MIN = 0.30
FINGER_START = (
    0.35  # fraction of the time when the fingers start closing from relaxed to the frame-0 shape (min-jerk after it)
)
# fraction when the fingers reach the frame-0 shape — the last 15 % only the arm moves in (time for the finger PD Kd/Kp
# 0.25 s to settle)
FINGER_END = 0.85
IK_ITERS = 8  # iterations per sample of the sequential DLS (stage_dp --final-iters default)
# null-space posture bias of the sequential DLS (towards the joint range centre, ArmIK._dls null_alpha) — 0 = none
IK_NULL = 0.0

# ── idle arm (the other arm of one-hand scenes) = park — stays at park over the whole window (start to join)
# whatever
# the
# base pose; the base pose (BASE_POSE) is for the active arm only. Park value = the one the env's v2 RSI reset uses, as
# is
# (one per env, not merged):
# bimanual env (emb bimanual), idle side = the bake's IK cache of that side (solver collision.IDLE_SOLVER_TAG,
# value collision.IDLE_ARM_QPOS; the reset writes state + target)
# one-hand env (emb right|left), other arm = the articulation default (dexengine.rl.tasks.scene.robot._joint_pos_map
# catch-all; the reset writes the target only). Its owner cannot be read without omni, so PARK_UNI_ENV_DEFAULT states
# it. Both statements are checked by the session (phys.py) against the RSI reset target with park_check; beyond PARK_TOL
# it dies (oracle) · marker PARK_TAG. The active arm route's guard measures the park arm as a static body (plan.py —
# collision.pen_cross_guard).
PARK_UNI_ENV_DEFAULT = 0.0
PARK_TOL = 1e-4  # rad — planned park ↔ RSI reset target
PARK_TAG = "[V4_IDLE_PARK]"
# active arm approach — clip IK keeps the v2 branch · fixed pose → joint-space route → backoff point → straight
# entry:
# ── scene clearance — during the approach the hand (palm, finger shells) does not touch the desk, objects, supports
# or
# static bodies. Before the straight entry (route share) every body keeps ≥ SCENE_CLEAR_M; on the straight entry
# (backoff
# point → frame 0) ≥ min(SCENE_CLEAR_M, frame-0 clearance − SCENE_TOL_M) — clips already touching at frame 0 (early
# contact) are allowed that depth on the last entry only.
# Geometry = the judge guard's (objects = cooked convex pieces dexengine.data.objects.hull, supports =
# support_surface.scene_supports, hand = the guard's hand tree, desk = hand_desk_m) + static bodies (extras.decide rows
# — receiving containers, render-only
# objects: physics disables robot ↔ static body collision and the force sensors cannot see them, so they are measured
# geometrically with the same threshold). This rule owns the route's scene clearance — ACTIVE_CLEAR and ROUTE_TAIL_FREE
# only filter candidates.
# (Without it fingers brushed supports and lost the frame-0 shape, or hands hit cans and the desk.)
SCENE_CLEAR_M = 0.005
SCENE_TOL_M = 0.001
# (backoff rule × route) candidates are tried up to this many — beyond it the variant is recorded as failed (kept)
SCENE_MAX_OPTS = 48
# backoff point block (scene.bite_blocked — a route-independent place): if even the most generous finger value the
# schedule
# can have at the backoff point (relaxed → f0 segment, this many points) is below the route threshold − SCENE_TOL_M,
# the
# rule's other routes are skipped
SCENE_BITE_FINGER_SAMPLES = 21
# cap on sliding inside the desk (the allowance to touch an embedded frame 0 stays): wrist travel of the run with hand
# ↔
# desk clearance < 0 right before frame 0 (scene.desk_slide_m) — beyond it the entry is violated
SCENE_DESK_SLIDE_M = 0.030
SCENE_ROUTES_PER_BACKOFF = 6  # route candidates tried per backoff rule (best first)
SCENE_TAG = "[AUG_V4_SCENE]"
ACTIVE_GRID = (
    (-0.8, -0.4, 0.0, 0.4, 0.8),
) * 4  # bias of control point c1 on joints 0–3 (rad, from the straight line's 1/3 point)
ACTIVE_SAMPLES = 64
# minimum (desk, torso) margin of the joint-space share (m) — when the straight entry is closer, that value + 2 mm
ACTIVE_CLEAR = (-0.03, -0.01)
ENTRY_MAX_ERR_M = 0.002  # cap on the straight entry IK position error (beyond it: failure)
# hand-off and retries (the home of these two values is here):
# canonical hand-off = no teleport (session mode every_pin — the policy takes over from the post-approach state,
# objects
# pinned to the RSI pose during the whole approach: zero object motion during the approach; unpinned objects settled 3
# mm/10°).
# 'all succeed' = at least one success passing the post gate per variant — else only that variant rolls again with a
# new
# seed (RETRY_SEED0 + round) · attempt (episode) cap RETRY_MAX_ATTEMPTS · 0 within the cap is reported as failed, not
# dropped.
HANDOFF_MODE = "every_pin"
RETRY_SEED0 = 42
RETRY_MAX_ATTEMPTS = 64
RETRY_TAG = "[AUG_V4_RETRY]"
# variants with 0 successes at the attempt cap → re-plan with the join floor off (join at the reference speed) and roll
# again up to the cap (once per variant).
REPLAN_FLOOR_OFF = True
REPLAN_TAG = "[AUG_V4_REPLAN]"
# plan pieces: variants split over PLAN_JOBS processes (torch threads PLAN_THREADS each; size both to the host's CPU
# cores) ·
# piece time cap = PLAN_BASE_S + variants × PLAN_S_PER_VARIANT (stall guard) · worst variant = all SCENE_MAX_OPTS
# candidates
# → about 1.7 × the measured 15.6–18 s per candidate = PLAN_S_PER_CAND.
PLAN_JOBS = 6
PLAN_THREADS = 2
PLAN_BASE_S = 600
PLAN_S_PER_CAND = 30
PLAN_S_PER_VARIANT = PLAN_S_PER_CAND * SCENE_MAX_OPTS


def ok_rank(kept: list, name: str) -> int | None:
    """Variant trajectory = the first post-gate-passing rank of the passing round (post.json kept) — stitch variant
    row,
    retry.json ok_rank, render.
    """
    nm = name.removesuffix("_pl_rt")
    rk = [int(k["rank"]) for k in kept if k["post_ok"] and str(k["name"]).removesuffix("_pl_rt") == nm]
    return min(rk) if rk else None


def park_q(side: str, emb: str, idle_cache) -> tuple[np.ndarray, str]:
    """Idle-arm park (7,) and the source of the statement — the two env rules of the 'idle arm' comment above. In
    bimanual
    envs that side's IK cache was baked with idle_rest and all frames must be equal (else death). Unknown embs die."""
    from pathlib import Path

    if emb in ("right", "left"):
        if side == emb:
            raise SystemExit(f"{MARK} park_q: in the one-hand env {emb}, {side} is the active side")
        return np.full(
            7, PARK_UNI_ENV_DEFAULT, float
        ), "one-hand env articulated-arm default (PARK_UNI_ENV_DEFAULT · reset = target only)"
    if emb != "bimanual":
        raise SystemExit(f"{MARK} park_q: unknown emb {emb!r}")
    from dexengine.placement.collision import idle_arm as IA

    z = np.load(Path(idle_cache), allow_pickle=True)
    if str(z["solver"]) != IA.IDLE_SOLVER_TAG:
        raise SystemExit(
            f"{MARK} park_q: {idle_cache} solver {str(z['solver'])!r} ≠ {IA.IDLE_SOLVER_TAG!r} — "
            f"{side} was not baked as the idle side"
        )
    q = np.asarray(z["arm_joint_positions_all"], float)
    if float(np.ptp(q, axis=0).max()) > 0.0:
        raise SystemExit(f"{MARK} park_q: {idle_cache} idle-arm IK differs between frames")
    return q[
        0
    ].copy(), "two-hand env resting-side IK cache (idle_rest = collision.IDLE_ARM_QPOS · reset = state + target)"


def park_check(plan_q: np.ndarray, rsi_target: np.ndarray, what: str) -> float:
    """Max difference planned park ↔ RSI reset target (rad) — beyond PARK_TOL it dies (session oracle)."""
    d = float(np.abs(np.asarray(plan_q, float) - np.asarray(rsi_target, float)).max())
    if d > PARK_TOL:
        raise SystemExit(
            f"{PARK_TAG} {what}: planned park ↔ RSI reset target {d:.4g} rad > {PARK_TOL} — the statement and the env "
            "disagree"
        )
    return d


def enabled() -> bool:
    v = os.environ.get(ENV, "")
    if v not in ("", "1"):
        raise SystemExit(f"{MARK} {ENV}={v!r} — empty or 1 only")
    return v == "1"


def base_name() -> str:
    n = os.environ.get(ENV_BASE, "") or BASE_POSE_NAME
    if n not in BASE_POSE_CANDIDATES:
        raise SystemExit(f"{MARK} {ENV_BASE}={n!r} — not among the candidates {sorted(BASE_POSE_CANDIDATES)}")
    return n


def base_q(side: str, name: str | None = None) -> np.ndarray:
    return np.asarray(BASE_POSE_CANDIDATES[name or base_name()][side], dtype=np.float64)


def relaxed_finger(name: str) -> float:
    for tail, v in RELAXED_RULES:
        if name.endswith(tail):
            return float(v)
    raise KeyError(f"{MARK} joint not in the relaxed-finger rule {name!r}")


def relaxed_fingers(names) -> np.ndarray:
    return np.array([relaxed_finger(n) for n in names], dtype=np.float64)


# ── route
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def first_motion_dir(P: np.ndarray, dt: float) -> tuple[np.ndarray | None, int]:
    """Reference wrist positions (T,3) → (unit direction of the first frame leaving frame 0 by FIRST_MOVE_M, that
    frame)
    · none = (None, −1).
    """
    d = np.linalg.norm(P - P[0], axis=1)
    kmax = min(len(P) - 1, int(round(FIRST_MOVE_MAX_S / dt)))
    hit = np.nonzero(d[: kmax + 1] > FIRST_MOVE_M)[0]
    if hit.size == 0:
        return None, -1
    k = int(hit[0])
    return (P[k] - P[0]) / d[k], k


def seq_ik(
    ik, Ts: np.ndarray, q0: np.ndarray, torch, DT, DEV, iters: int | None = None, null: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential DLS — every sample starts from the previous sample's solution (branch continuity, null-space bias
    IK_NULL). → (q (M,7), position error m (M,)).
    """
    iters = IK_ITERS if iters is None else iters
    null = IK_NULL if null is None else null
    q = torch.tensor(np.asarray(q0, np.float64)[None], dtype=DT, device=DEV)
    out, err = np.empty((len(Ts), len(q0))), np.empty(len(Ts))
    Tt = torch.tensor(Ts, dtype=DT, device=DEV)
    for i in range(len(Ts)):
        q, pe, _re, _m = ik._dls(Tt[i : i + 1], iters, 1e-2, null, q)
        out[i] = q[0].detach().cpu().numpy()
        err[i] = float(pe[0])
    return out, err


# ── schedule
# ──────────────────────────────────────────────────────────────────────────────────────────────
def minjerk(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t**3 * (10 - 15 * t + 6 * t**2)


def joint_time(q: np.ndarray, budget: np.ndarray) -> np.ndarray:
    """Route parameter σ (M,) = cumulative joint time — the sum over samples of max_j |Δq_j| / budget_j (the joint
    motion
    allowed in one frame). Min-jerk on this σ raises the joint speed ratio evenly along the route — the shortest
    within
    the limits (timing by wrist arc length crowded the joints in one place on rotation-heavy approaches, 2–3×
    longer)."""
    d = (np.abs(np.diff(q, axis=0)) / np.asarray(budget)[None]).max(axis=1)
    return np.r_[0.0, np.cumsum(np.maximum(d, 1e-9))]


def backoff_cands(T0: np.ndarray, P: np.ndarray, dt: float) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """All backoff candidates: rule order (tangent → palm → above) × backoff distances BACKOFF_STEPS (earlier first)
    —
    when
    the straight entry's IK from the v2 solution is blocked by joint limits (v2 frame 0 at a joint limit), the next one
    is tried."""
    p0 = T0[:3, 3]
    u, _k = first_motion_dir(P, dt)
    dirs = ([("tangent", -u)] if u is not None else []) + [("palm", -T0[:3, 0]), ("up", np.array([0.0, 0.0, 1.0]))]
    for el in (
        BACKOFF_RING_EL
    ):  # all three rules blocked (a branch with v2 frame 0 at a joint limit) → ring directions of the upper hemisphere
        for az in range(0, 360, BACKOFF_RING_AZ):
            a_, e_ = math.radians(az), math.radians(el)
            dirs.append(
                (f"ring{az}_{el}", np.array([math.cos(e_) * math.cos(a_), math.cos(e_) * math.sin(a_), math.sin(e_)]))
            )
    out = []
    for dist in BACKOFF_STEPS:
        for rule, r in dirs:
            r = r / np.linalg.norm(r)
            if r[2] >= -DOWN_MAX:
                out.append((p0 + dist * r, -r, f"{rule}{int(round(dist * 100))}"))
    return out


def entry_poses(T0: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Straight entry samples from the backoff point w to the frame-0 palm T0 (IK frame 4×4, orientation fixed at frame
    0, PATH_STEP_M spacing).
    """
    p0 = T0[:3, 3]
    n = max(4, int(math.ceil(float(np.linalg.norm(p0 - w)) / PATH_STEP_M)))
    Ts = np.tile(T0, (n + 1, 1, 1))
    Ts[:, :3, 3] = w[None] + np.linspace(0.0, 1.0, n + 1)[:, None] * (p0 - w)[None]
    return Ts


def active_routes(
    q_base: np.ndarray,
    q_w: np.ndarray,
    t_w: np.ndarray,
    comp,
    budget: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    limit: tuple[float, float],
    plen=None,
) -> list[dict]:
    """Joint-space share candidates of the active arm approach (best first) — a cubic Bézier (ACTIVE_SAMPLES samples)
    from
    the fixed pose to the backoff point's joints q_w (v2 IK branch, first solution of the straight entry). End tangent =
    the straight entry's joint tangent t_w (unit) — C1 into the entry, no seam. Control point c1 = the straight line's
    1/3
    point + ACTIVE_GRID (bias on joints 0–3; wrist joints 4–6 stay on the line and turn evenly along the route) ·
    c2 = q_w − (|q_w − q_base| / 3)·t_w. Filter = comp(q) → (desk m, torso m) within limit · order = shorter hand
    path
    (plen(q) → m; else less deviation from the line's 1/3 point)."""
    import itertools

    qb, qw = np.asarray(q_base, float), np.asarray(q_w, float)
    t = np.asarray(t_w, float) / max(float(np.linalg.norm(t_w)), 1e-9)
    c2 = np.clip(qw - (float(np.linalg.norm(qw - qb)) / 3.0) * t, lo, hi)
    s = np.linspace(0.0, 1.0, ACTIVE_SAMPLES)[:, None]
    one3 = qb + (qw - qb) / 3.0
    out = []
    for a in itertools.product(*ACTIVE_GRID):
        c1 = one3.copy()
        c1[:4] += np.asarray(a)
        c1 = np.clip(c1, lo, hi)
        q = (1 - s) ** 3 * qb + 3 * (1 - s) ** 2 * s * c1 + 3 * (1 - s) * s**2 * c2 + s**3 * qw
        d, tor = comp(q[: int(round(len(q) * (1.0 - ROUTE_TAIL_FREE)))])
        out.append(
            {
                "q": q,
                "c1": c1,
                "c2": c2,
                "desk_m": float(d),
                "torso_m": float(tor if tor is not None else -1e9),
                "sig_end": float(joint_time(q, budget)[-1]),
                "dev": float(np.abs(c1 - one3).sum()),
            }
        )
    ok = [r for r in out if r["desk_m"] <= limit[0] and r["torso_m"] <= limit[1]]
    if not ok:
        return []
    for r in ok:
        r["hand_len_m"] = float(plen(r["q"])) if plen is not None else r["dev"]
    return sorted(ok, key=lambda r: (r["hand_len_m"], r["sig_end"]))


def finger_schedule(K: int, f_relax: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """(K + 1, n) — min-jerk from FINGER_START to FINGER_END over K steps (relaxed → f0). The plan (plan.py — time,
    guard)
    and the session (phys.py — end = RSI measured) both use this one function."""
    tau = np.arange(K + 1) / K
    g = minjerk((tau - FINGER_START) / (FINGER_END - FINGER_START))
    f_relax = np.asarray(f_relax, float)
    return f_relax[None] + g[:, None] * (np.asarray(f0, float) - f_relax)[None]
