"""Arm-flip gate + re-solve — the one implementation of the rule and its values.

Callers: the bake (every save point of `coarse_path.dp` — reuse of the judge trajectory · its own DP · field mode) and
the judge (`--arm-flip`, default `JUDGE_GATE_DEFAULT` = on, on both confirm paths; the only way off is an explicit
`--arm-flip 0`). A trajectory the judge confirmed passes this gate again at the bake's reuse point. Callers pass their
own thresholds (wrist error · stretch) and guards as functions — this module does not restate those values.

What it stops: a trajectory where the hand barely moves but one arm joint turns more than half a revolution in a short
time, so the whole arm switches to another posture (another IK branch). The one-step jump gate (1.0 rad) does not see a
switch spread over 14–83 frames, and retiming stretches such a switch into the speed limits and lets it through.

Rule (values = the constants below, one place):
  gate     within `WINDOW_S` seconds of the original time axis one joint turns more than `TRAVEL_MAX_RAD` → re-solve.
           A trajectory the gate does not hit is **returned as the same object** (numpy only · no GPU call → output
           bytes unchanged).
  re-solve at the start frame of the hit span, pick the arm posture again (the branch before the switch · the branch
           after it · other branches found by random multi-start) → track forward and backward from that posture,
           warm-started from the previous frame → the caller's polish.
  replace  only when a candidate (1) does not hit the gate, (2) is within the caller's wrist-error · stretch thresholds,
           (3) has wrist-orientation p99 ≤ `ROT_P99_MAX_DEG`, (4) moves the arm over the hit span (sum of 7 joints) at
           least `EXCESS_MIN_RAD` less than now and (5) passes the caller's guards (penetration · cross). Among several,
           the one with the smallest arm travel.
  no fix   the bake does not drop the clip — the trajectory is returned unchanged with the marker
           `[ARM_FLIP] … → kept (reason)` only.
  judge    `unresolved` = the gate was hit and fewer than `FEASIBLE_MIN` candidates continue within the thresholds →
           that placement candidate is dropped. Candidates within the thresholds whose arm travel did not shrink (a
           rotation that follows the hand) are not dropped.
Verdict vocabulary of `fix`'s info (compared by `unresolved` and set by judge.confirm): VERDICT_REPLACED,
VERDICT_KEPT. The reason strings WHY_* are the keys of the rejection counts (`feasible_count` reads WHY_EXCESS).
"""

from __future__ import annotations

import math

import numpy as np

# isort: split
from dexengine.core import stdio
from dexengine.placement.retime.stretch import FRAME_DT

WINDOW_S = 2.0  # time window of the gate (original time axis, s)
TRAVEL_MAX_RAD = math.pi  # most one joint may turn inside that window (more = gate hit)
EXCESS_MIN_RAD = 2.0  # arm travel over the hit span must shrink by at least this much to replace (a rotation
#                              that follows the hand does not shrink when re-solved)
ROT_P99_MAX_DEG = 8.0  # wrist-orientation error p99 cap of a replacing trajectory (deg)
WIN_MARGIN = 10  # frames before and after the hit span included in its arm travel
N_RAND = 256  # random multi-start count
N_SEEDS = 12  # cap on seeds to track (incl. the before · after branches)
SEED_POS_M = 0.005  # a random seed counts as converged to that frame's target within this position (m) ·
SEED_ROT_DEG = 3.0  # orientation (deg)
SEED_DEDUP_RAD = 0.5  # joint distance (max joint difference) under which two seeds are the same branch
SEED_ITERS = 40
TRACK_ITERS = 4  # tracking iterations per frame (same value as the judge's --track-iters)
RNG_SEED = 42  # re-seeded on every call (result independent of clip order)
FEASIBLE_MIN = 1  # at least this many candidates must continue within the thresholds, else the flip is
#                              unresolvable
JUDGE_GATE_DEFAULT = 1  # whether the judge (selection) applies this rule — read as the default of the judge's
#                              --arm-flip; the only way off is an explicit `--arm-flip 0`
#: Bake-side defaults — read as the defaults of coarse_path.dp's --arm-flip-fail · --dp-pin-rad (explicit off = off ·
#: 0).
#: BAKE_GATE_DEFAULT: "dp" = a trajectory the bake solved itself (own DP · field mode) is not saved when it is
#:   unresolved while the judge trajectory is clean (rc 3 → the bake driver moves to the next placement candidate).
#:   Trajectories handed over by the judge are not checked here — that is the judge-stage gate's job.
#: BAKE_PIN_RAD: band (max joint difference, rad) around the judge trajectory for one more solve of an own DP that hit
#:   that gate. The one-step cap of the pinned solve is the judge's --jump-max (judge_jump_max).
BAKE_GATE_DEFAULT = "dp"
BAKE_PIN_RAD = 1.0
#: Why a candidate is rejected — in the order `accept` checks. Keys of the rejection counts in info["reason"] and in the
#: marker; `feasible_count` reads WHY_EXCESS.
WHY_FLIP, WHY_ERR, WHY_ROT = "candidate flips too", "wrist error over threshold", "wrist orientation over threshold"
WHY_STRETCH, WHY_EXCESS, WHY_GUARD = (
    "stretch over cap",
    "arm travel did not shrink (rotation following the hand)",
    "penetration/cross guard",
)
#: Verdict of `fix`'s info (in-memory only: read by `unresolved`, also set by judge.confirm).
VERDICT_REPLACED, VERDICT_KEPT = "replaced", "kept"


def window_frames() -> int:
    """Gate window in frames (WINDOW_S at FRAME_DT)."""
    return int(round(WINDOW_S / FRAME_DT))


def window_travel(q):
    """Joint trajectory q (T,n) → (largest in-window travel rad, that joint, span start frame, span end frame). numpy
    only."""
    q = np.asarray(q, dtype=np.float64)
    T = len(q)
    if T < 2:
        return 0.0, 0, 0, 0
    w = min(window_frames() + 1, T)
    v = np.lib.stride_tricks.sliding_window_view(q, w, axis=0)  # (T-w+1, n, w)
    r = v.max(-1) - v.min(-1)
    c, j = np.unravel_index(int(np.argmax(r)), r.shape)
    seg = q[c : c + w, j]
    i0, i1 = sorted((int(seg.argmin()), int(seg.argmax())))
    return float(r[c, j]), int(j), int(c + i0), int(c + i1)


def triggered(q) -> bool:
    """Whether q hits the gate (in-window travel > TRAVEL_MAX_RAD)."""
    return window_travel(q)[0] > TRAVEL_MAX_RAD


def span_path(q, fa: int, fb: int) -> float:
    """Arm travel over the hit span (with WIN_MARGIN frames on each side) — sum of |Δq| over the 7 joints (rad)."""
    a, b = max(0, fa - WIN_MARGIN), min(len(q) - 1, fb + WIN_MARGIN)
    return float(np.abs(np.diff(np.asarray(q, dtype=np.float64)[a : b + 1], axis=0)).sum())


def accept(m: dict, err_mm_max: float, stretch_max: float):
    """Measurements m of one candidate → (may replace, else the reason). The only place of the replace conditions.
    m = {travel (rad) · err_mm · rot_deg · stretch · excess (rad)}"""
    if m["travel"] > TRAVEL_MAX_RAD:
        return False, WHY_FLIP
    if m["err_mm"] > err_mm_max:
        return False, WHY_ERR
    if m["rot_deg"] > ROT_P99_MAX_DEG:
        return False, WHY_ROT
    if m["stretch"] > stretch_max:
        return False, WHY_STRETCH
    if m["excess"] < EXCESS_MIN_RAD:
        return False, WHY_EXCESS
    return True, ""


def feasible_count(n_accepted: int, why: dict) -> int:
    """Candidates that continue within the thresholds = replaceable candidates + candidates whose arm travel alone did
    not shrink (both passed the gate · error · orientation · stretch)."""
    return int(n_accepted) + int(why.get(WHY_EXCESS, 0))


def unresolved(info) -> bool:
    """`fix`'s info → whether the flip is unresolvable (the only place of the judge's drop condition). Not when the
    gate was not hit (None), when the trajectory was replaced, or when at least FEASIBLE_MIN candidates continue within
    the thresholds (a rotation that follows the hand)."""
    return info is not None and info["verdict"] != VERDICT_REPLACED and info["feasible"] < FEASIBLE_MIN


def _side_stream(ik, device, torch):
    """The second CUDA stream of `_track` (one per IK, kept so the IK's graph cache sees one stable stream) · None off
    the GPU."""
    if device.type != "cuda":
        return None
    s = ik.__dict__.get("_track_side_stream")
    if s is None:
        s = ik.__dict__["_track_side_stream"] = torch.cuda.Stream(device=device)
    return s


def _track(ik, Tt, Q0, f0, torch):
    """Track seeds Q0 (M,n) forward and backward from frame f0 → (M,T,n).

    The forward chain (f0 → T−1) and the backward chain (f0−1 → 0) share only frame f0: the backward chain starts from
    the forward chain's solution there. After that frame they are independent, so on the GPU the backward chain runs
    on a second stream while the forward chain runs on the current one, and their serial one-row-bucket solves overlap.
    Each chain makes the same `ik._dls` calls as before (same targets, seeds, row counts and order); only the overlap
    changes. `ArmIK._dls_fast` keys its CUDA graphs and graph pools by stream, so the two chains never share a graph's
    buffers. Order of events: frame f0 on the current stream → the side stream waits for it → both chains → the
    current stream waits for the side stream (join) → QQ is read. Off the GPU the same interleaved order runs
    serially."""
    import contextlib  # noqa: PLC0415

    T, M = Tt.shape[0], Q0.shape[0]
    QQ = torch.zeros((T, M, ik.n), dtype=Q0.dtype, device=Q0.device)

    def step(f, Q):
        Q, _, _, _ = ik._dls(Tt[f : f + 1].expand(M, -1, -1).contiguous(), TRACK_ITERS, 1e-2, 0.0, Q)
        QQ[f] = Q
        return Q

    qf = step(f0, Q0.clone())
    side = _side_stream(ik, Q0.device, torch)
    if side is not None:
        side.wait_stream(torch.cuda.current_stream())  # QQ (zero fill + frame f0), Tt and Q0 are ready before it reads
    on_side = contextlib.nullcontext() if side is None else torch.cuda.stream(side)
    with on_side:
        qb = QQ[f0].clone()
    n_fwd, n_bwd = T - f0 - 1, f0
    for k in range(max(n_fwd, n_bwd)):
        if k < n_fwd:
            qf = step(f0 + 1 + k, qf)
        if k < n_bwd:
            with on_side:
                qb = step(f0 - 1 - k, qb)
    if side is not None:
        torch.cuda.current_stream().wait_stream(side)  # join: every backward write to QQ lands before QQ is read
    return QQ.permute(1, 0, 2).contiguous()


def _flip_seeds(q64, fa, _t, fb, ik, Tt, torch, dev, dt_):
    """IK seeds of the window: the before-window pose, a backward-tracked chain from the after-window pose, and the
    random restarts that pass the seed gate."""
    seeds = [q64[fa]]
    Q = _t(q64[fb : fb + 1])
    for f in range(fb - 1, fa - 1, -1):
        Q, _, _, _ = ik._dls(Tt[f : f + 1], TRACK_ITERS, 1e-2, 0.0, Q)
    seeds.append(Q[0].cpu().numpy().astype(np.float64))
    g = torch.Generator(device=dev)
    g.manual_seed(RNG_SEED)
    rq = ik.lo + (ik.hi - ik.lo) * torch.rand((N_RAND, ik.n), dtype=dt_, device=dev, generator=g)
    rq, rpe, rre, _ = ik._dls(Tt[fa : fa + 1].expand(N_RAND, -1, -1).contiguous(), SEED_ITERS, 1e-2, 0.0, rq)
    okr = ((rpe < SEED_POS_M) & (rre < SEED_ROT_DEG)).cpu().numpy()
    for qq in rq.cpu().numpy().astype(np.float64)[okr]:
        if len(seeds) >= N_SEEDS:
            break
        if all(np.abs(qq - s).max() > SEED_DEDUP_RAD for s in seeds):
            seeds.append(qq)
    return seeds


def _feasible_candidates(M, qp, err_mm_fn, pe, re, stretch_fn, p_cur, fa, fb, err_mm_max, stretch_max, cands, why):
    """Score each polished candidate into cands (accepted: score, index, metrics) and why (rejection counts)."""
    for i in range(M):
        m = {
            "travel": window_travel(qp[i])[0],
            "err_mm": float(err_mm_fn(pe[i] * 1000.0, qp[i])),
            "rot_deg": float(np.percentile(re[i], 99)),
            "stretch": float(stretch_fn(qp[i])),
            "excess": p_cur - span_path(qp[i], fa, fb),
        }
        ok, r = accept(m, err_mm_max, stretch_max)
        if ok:
            cands.append((float(np.abs(np.diff(qp[i], axis=0)).sum()), i, m))
        else:
            why[r] = why.get(r, 0) + 1


def fix(ik, Tt, q, *, err_mm_fn, err_mm_max, stretch_fn, stretch_max, polish_iters, tag, torch, guard_fn=None):
    """Gate → re-solve → replace / keep. Returns (q, pe, re, info).

    q **the same object as given** means not replaced (pe · re = None) — the caller continues with its own values.
    When replaced, q is a new array (T,n float64) and pe (m) · re (deg) are re-measured after polishing that trajectory
    with `polish_iters`. info = None when the gate is not hit.
      ik            the arm IK solver (DLS clipped at the limits — the same solver the caller uses)
      Tt            (T,4,4) wrist targets (base frame) · same time axis as q
      err_mm_fn     (pe_mm, q) → the wrist-error value the caller compares with its threshold (weighted p99 after
                    retiming)
      stretch_fn    q → the stretch the caller compares with its cap
      guard_fn      q → bool (the caller's penetration · cross guards — only passing candidates replace). None = no
                    guard
    """
    travel, jn, fa, fb = window_travel(q)
    if travel <= TRAVEL_MAX_RAD:
        return q, None, None, None
    q64 = np.asarray(q, dtype=np.float64)
    T = len(q64)
    dt_, dev = Tt.dtype, Tt.device
    info = {
        "travel": travel,
        "joint": jn,
        "fa": fa,
        "fb": fb,
        "verdict": VERDICT_KEPT,
        "reason": "",
        "excess": float("nan"),
        "n_seeds": 0,
        "feasible": 0,
    }

    def _t(x):
        return torch.tensor(np.asarray(x), dtype=dt_, device=dev)

    # seeds: the branch before the switch · the branch after it (tracked backwards from the span end to the start
    # frame) · other branches at that frame
    seeds = _flip_seeds(q64, fa, _t, fb, ik, Tt, torch, dev, dt_)
    M = len(seeds)
    info["n_seeds"] = M
    QQ = _track(ik, Tt, _t(np.stack(seeds)), fa, torch)  # (M,T,n)
    qp, pe, re, _ = ik._dls(Tt.repeat(M, 1, 1), polish_iters, 1e-2, 0.0, QQ.reshape(M * T, ik.n))
    qp = qp.reshape(M, T, ik.n).cpu().numpy().astype(np.float64)
    pe = pe.reshape(M, T).cpu().numpy()
    re = re.reshape(M, T).cpu().numpy()
    p_cur = span_path(q64, fa, fb)
    cands, why = [], {}
    _feasible_candidates(M, qp, err_mm_fn, pe, re, stretch_fn, p_cur, fa, fb, err_mm_max, stretch_max, cands, why)
    info["feasible"] = feasible_count(len(cands), why)
    head = (
        f"[ARM_FLIP] {tag}: {WINDOW_S:g} s window {math.degrees(travel):.0f}° (joint {jn} · frames {fa}–{fb}) · "
        f"seeds {M}"
    )
    for _, i, m in sorted(cands):
        if guard_fn is not None and not guard_fn(qp[i]):
            why[WHY_GUARD] = why.get(WHY_GUARD, 0) + 1
            continue
        info.update(verdict=VERDICT_REPLACED, excess=m["excess"], err_mm=m["err_mm"], rot_deg=m["rot_deg"])
        print(
            f"{head} → replaced (span arm travel {p_cur:.1f}→{p_cur - m['excess']:.1f} rad · "
            f"error {m['err_mm']:.1f} mm · orientation p99 {m['rot_deg']:.1f}° · stretch {m['stretch']:.2f})",
            flush=True,
        )
        return qp[i], pe[i], re[i], info
    info["reason"] = " · ".join(f"{k} {v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1]))
    print(f"{head} → kept (no continuous solution: {info['reason']})", flush=True)
    return q, None, None, info


def bake_gate(fail: str, site: str, info, ref) -> bool:
    """Bake end gate — whether not to save this trajectory. True at a save point where the bake **solved itself** (site
    "dp" = own DP · field mode) when the gate was hit and no solution continues within the thresholds (unresolved — the
    same condition the judge drops a candidate on). False when the trajectory handed over by the judge (ref) already
    hits the gate (not a flip the bake made — the judge-stage gate's job) and at the reuse site (site "reuse").
    Always False for fail "off" (explicit off). On True coarse_path.dp solves once more pinned around the judge
    trajectory (BAKE_PIN_RAD), and if that fails too, rc 3 (the bake driver moves to the next placement candidate)."""
    return fail == "dp" and site == "dp" and unresolved(info) and not (ref is not None and triggered(ref))


def bake_fix(a, ik, Tt, qf, pef, redf, torch, VST, BUDGET, FF, err_mm_fn, guards, site="dp", ref=None):
    """Arm-flip gate + re-solve (fix) + bake end gate (bake_gate) at a bake save point. When the gate is not hit or the
    flip cannot be fixed, the three inputs are returned **as the same objects** (output bytes unchanged — only the
    `[ARM_FLIP]` marker). A replacing candidate has already passed that save point's thresholds (wrist error · stretch)
    and guards (guards = penetration · cross) with the same functions. A candidate's guard output is relabelled
    `[ARM_FLIP] candidate guard` and its failure word becomes `— candidate dropped`: a line with the failure word is the
    clip's failure line (the bake driver copies `[stage_dp]` lines containing that word into the ledger), so it is not
    used for a candidate rejection. 4th value = tail of the output npz `solver` ("+flipfix" only when replaced) · 5th
    value = whether the bake end gate (bake_gate) blocks the save."""
    import io  # noqa: PLC0415

    from dexengine.placement.coarse_path import guards as GD  # noqa: PLC0415 — owner of the failure word

    def _guard(q):
        buf = io.StringIO()
        with stdio.capture(buf):
            ok = guards(q)
        for ln in buf.getvalue().splitlines():
            print(
                f"[ARM_FLIP] candidate guard: {ln.replace(GD.FAIL_MARK, '— candidate dropped')}"
                if GD.FAIL_MARK in ln
                else ln,
                flush=True,
            )
        return ok

    q2, pe2, re2, info = fix(
        ik,
        Tt,
        qf,
        err_mm_fn=err_mm_fn,
        err_mm_max=a.max_err_mm,
        stretch_fn=lambda q: VST.total_stretch(q, BUDGET, a.safety, floor=FF),
        stretch_max=a.max_stretch,
        polish_iters=a.final_iters,
        tag=f"{a.sequence}/{a.side}",
        torch=torch,
        guard_fn=_guard,
    )
    bad = bake_gate(a.arm_flip_fail, site, info, ref)
    if bad:
        print(
            f"[ARM_FLIP_FAIL] {a.sequence}/{a.side}: unresolvable arm flip {np.degrees(info['travel']):.0f}°",
            flush=True,
        )
    return (qf, pef, redf, "", bad) if q2 is qf else (q2, pe2, re2, "+flipfix", False)


def judge_jump_max() -> float:
    """One-step cap of a re-solved trajectory = the judge's --jump-max default (7-joint L2, rad), read from the judge
    configuration (configs/placement/judge.yaml)."""
    from dexengine.placement.judge import config as _JC  # noqa: PLC0415

    return float(_JC.load().jump_max)
