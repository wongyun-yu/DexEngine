"""Campaign planning model: training / judge time estimates, the sidecar parameters the runner hands the judge sidecar,
and the list schedule of trainings and judges with its disk and latency figures (estimates, no I/O). The plan CLI and
the launch loop that read it live in dexengine.rl.campaign."""

from __future__ import annotations

import heapq

from dexengine.core import paths as P
from dexengine.rl.cell_state import JUDGE_FINALS

# ── planning model (estimates, no I/O) ──────────────────────────────────────────────────────────────────────────────
#   training  list-scheduled on n_train slots in queue order: a free slot takes the next bundle at once
#   judge     at a training's end its JUDGE_FINALS final judges (then its mid judges, mids(b)) are released to the
#             n_judge judge GPUs, FIFO by release time, one judge per GPU at a time (the host gate's one claim per
#             GPU), judge_h each
#   overflow  the wait = the judge GPUs' backlog past the training end (Σ busy time left, other runs only) / n_judge;
#             when it exceeds the threshold the finals run on that run's own training GPU right after it (serially,
#             JUDGE_FINALS × judge_h) and the slot takes its next bundle after them. None = off
#   deadline  the sidecar dies INCOMPLETE SIDECAR_DEADLINE_H after its start (= the training start): a split whose
#             worst (judge end − training start) exceeds it is not feasible (split_table marks it)
#   disk      a run holds all its checkpoints (per_run_gb) from its training start until its judge end; after that it
#             keeps kept_gb when pruned, per_run_gb when not

#: Training estimates in hours at MI 1600 (measured median 3.82 h + 0.1 h boot; one-hand runs 0.64×). Articulated =
#: the rigid row × ARTIC_X, the largest measured artic/rigid s/iter ratio on one card (1.31–1.58), so a plan never runs
#: short. One judge = its GPU time at the judging shape (NE 1024, 3–4 real batches at K 43–64): 0.095 h.
ARTIC_X = 1.58


TRAIN_H = {"_": 3.92, "uni": 3.92 * 0.64, "artic": 3.92 * ARTIC_X}


TRAIN_H_MI = 1600  #: the MI the TRAIN_H estimates are for (a run at another MI scales linearly, train_hours)


JUDGE_FINAL_H = 0.095


#: What the runner hands the judge sidecar (RLJUDGE_*), stated once here; the model plans with the same values. No
#: quick judges (a bundle is judged at its finals only, judge.metrics.final_iters), the sidecar deadline from its
#: start, and concurrent judges = every GPU of the host split (the gate's per-GPU claim is the real limit; overflow
#: judges run on training GPUs).
SIDECAR_QUICK_EVERY = 0


SIDECAR_DEADLINE_H = 24.0


#: Hands whose campaign sidecars record the final judges' trajectories (RLJUDGE_TRAJ=1; the replay render reads
#: <RUN>_jf<iter>_traj.npz from the cell). Empty = off for every hand. Registered hand ids only (traj_record_hands
#: checks them at every read).
TRAJ_RECORD_HANDS: frozenset[str] = frozenset()


#: A training's timeout = TRAIN_TIMEOUT_X × its class estimate × MI/1600 (a stuck Isaac never holds a slot).
TRAIN_TIMEOUT_X = 3.0


def train_hours(klass: str, mi: int) -> float:
    """A bundle's training estimate [h]: its class row of TRAIN_H (default "_") scaled by MI / TRAIN_H_MI."""
    return TRAIN_H.get(klass, TRAIN_H["_"]) * mi / TRAIN_H_MI


def traj_record_hands() -> frozenset[str]:
    """TRAJ_RECORD_HANDS checked against the hand registry (paths.HAND_IDS). An unknown id (a typo, an alias, a
    retired hand, a bare string iterated as letters) would match no bundle and record nothing; it raises instead,
    before any judge command is built."""
    unknown = sorted(set(TRAJ_RECORD_HANDS) - set(P.HAND_IDS))
    if unknown:
        raise ValueError(
            f"[TRAJ_RECORD_HANDS] hands not in the hand registry {unknown} — registered: {' '.join(P.HAND_IDS)}"
        )
    return TRAJ_RECORD_HANDS


def sidecar_env(max_conc: int, deadline_h: float | None = None, *, hand: str | None) -> list[str]:
    """RLJUDGE_* of a campaign sidecar: max_conc = concurrent judges; deadline_h = a host's own budget where its
    trainings outlast SIDECAR_DEADLINE_H (default the constant); hand = the bundle's hand, RLJUDGE_TRAJ=1 when it is in
    traj_record_hands() (keyword without default: a caller that forgets it fails instead of silently judging without
    the hand's record switch)."""
    dl = SIDECAR_DEADLINE_H if deadline_h is None else deadline_h
    return [
        f"RLJUDGE_QUICK_EVERY={SIDECAR_QUICK_EVERY}",
        f"RLJUDGE_DEADLINE_S={int(dl * 3600)}",
        f"RLJUDGE_MAX_CONC={max_conc}",
    ] + (["RLJUDGE_TRAJ=1"] if hand in traj_record_hands() else [])


def mid_judges(mi: int) -> int:
    """The mid judges a campaign run asks for at MI `mi`: none (quick judges are off and no special checkpoint tag
    exists; a run is judged at its finals only)."""
    return 0


def overflow_due(wait_h: float, threshold_h: float | None) -> bool:
    """The overflow rule (the runner at a training's end and the model use this one comparison): the finals go to the
    run's own training GPU when the expected wait for a judge GPU exceeds the threshold. None = off."""
    return threshold_h is not None and wait_h > threshold_h


def live_wait_h(backlog_h: float, n_judge: int) -> float:
    """The expected wait for a judge GPU at a training's end: the judge GPUs' backlog (hours of final judges running or
    queued ahead, other runs only) spread over the judge GPUs."""
    if n_judge < 1:
        raise ValueError("0 judge GPUs")
    return max(backlog_h, 0.0) / n_judge


def schedule(
    queue: list[dict],
    n_train: int,
    n_judge: int,
    train_h,
    *,
    mids,
    overflow_h: float | None = None,
    judge_h: float = JUDGE_FINAL_H,
) -> tuple[list[dict], float]:
    """(rows, makespan h). Each row = the bundle + slot · start_h · train_end_h · judge_end_h · overflow (planning model
    above). mids(b) = the run's mid judges (mid_judges over its MI; no default). n_judge 0 is refused (the sidecar's
    finals would never find a GPU: every run INCOMPLETE)."""
    if n_train < 1 or n_judge < 1:
        raise ValueError(
            f"training GPUs {n_train} · judge GPUs {n_judge} — both must be >= 1"
            " (0 judge GPUs = final judges INCOMPLETE)"
        )
    rows: list[dict] = []
    ev: list[tuple[float, int, int]] = []  # (training end h, row index, slot)
    nxt = 0

    def start(slot: int, t: float) -> None:
        nonlocal nxt
        if nxt < len(queue):
            b = queue[nxt]
            rows.append(dict(b, slot=slot, start_h=round(t, 3), train_end_h=round(t + train_h(b), 3)))
            heapq.heappush(ev, (t + train_h(b), len(rows) - 1, slot))
            nxt += 1

    for s in range(n_train):
        start(s, 0.0)
    jfree = [0.0] * n_judge
    while ev:
        t, k, slot = heapq.heappop(ev)
        wait = live_wait_h(sum(max(j - t, 0.0) for j in jfree), n_judge)
        ov = overflow_due(wait, overflow_h)
        end = t + JUDGE_FINALS * judge_h if ov else 0.0
        for _ in range(mids(rows[k]) + (0 if ov else JUDGE_FINALS)):
            j = min(range(n_judge), key=lambda x: (max(jfree[x], t), x))
            jfree[j] = max(jfree[j], t) + judge_h
            end = max(end, jfree[j])
        rows[k].update(judge_end_h=round(end, 3), overflow=ov)
        start(slot, t + JUDGE_FINALS * judge_h if ov else t)
    return rows, max((r["judge_end_h"] for r in rows), default=0.0)


def peak_disk_gb(rows: list[dict], per_run_gb: float, kept_gb: float) -> float:
    """Peak checkpoint disk of a schedule: Σ per_run_gb over runs between start and judge end, plus kept_gb over runs
    whose judge ended (prune); pass kept_gb = per_run_gb for no prune."""
    ev = sorted(
        [(r["start_h"], 0, per_run_gb) for r in rows] + [(r["judge_end_h"], 1, kept_gb - per_run_gb) for r in rows]
    )
    cur = peak = 0.0
    for _t, _o, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def worst_latency_h(rows: list[dict]) -> float:
    """The longest (judge end − training start) of a schedule, against SIDECAR_DEADLINE_H (the sidecar starts with its
    training)."""
    return max((r["judge_end_h"] - r["start_h"] for r in rows), default=0.0)


def split_table(queue: list[dict], n_gpus: int, th, per_run_gb: float, kept_gb: float, mids) -> list[str]:
    """The planning model over the host's splits (judge 1–3 GPUs) × overflow (off · 0 · 0.5 · 1 · 2 h): makespan,
    overflowed runs, the worst judge latency (a row past SIDECAR_DEADLINE_H is marked infeasible: its sidecars would
    die INCOMPLETE) and the checkpoint disk peak with and without prune; same queue, same estimates."""
    out = []
    for nj in (1, 2, 3):
        for oh in (None, 0.0, 0.5, 1.0, 2.0):
            if n_gpus - nj < 1:
                continue
            rows, end = schedule(queue, n_gpus - nj, nj, th, mids=mids, overflow_h=oh)
            lat = worst_latency_h(rows)
            ok = (
                ""
                if lat <= SIDECAR_DEADLINE_H
                else f" · ⛔ infeasible (worst latency > deadline {SIDECAR_DEADLINE_H:g} h)"
            )
            out.append(
                f"  train {n_gpus - nj} + judge {nj} · overflow {'off' if oh is None else f'>{oh:g} h'} = "
                f"{end:.0f} h ({end / 24:.1f} d) · overflowed runs {sum(r['overflow'] for r in rows)} · worst "
                f"latency {lat:.1f} h · disk peak pruned {peak_disk_gb(rows, per_run_gb, kept_gb) / 1e3:.2f} TB "
                f"/ unpruned {peak_disk_gb(rows, per_run_gb, per_run_gb) / 1e3:.2f} TB{ok}"
            )
    return out
