"""The overflow step of the RL campaign loop (scheduler.step_overflow).

At a training's end the expected wait for a judge GPU (judge_backlog_h spread over the judge GPUs) is
compared with the threshold; when due, the run's own training GPU is held for its final
judges (host gate kind `hold`, bound to the run's sidecar, so the hold ends with the sidecar)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

#: The overflow step's three answers (overflow_at_train_end): held · no (recorded once, overflow.txt) · wait (not
#: decided — the next tick asks again: the training still lives, or the gate refused the hold for now).
OVERFLOW = ("held", "no", "wait")


def hold(gpu: str, sidecar_pid: int, run: str) -> tuple[int, str]:
    """The overflow hold (host gate kind `hold`): after its training ended, the run's own training GPU is held for its
    final judges — the gate refuses training claims on it and lets only this run's judge claim it. Its pid is the run's
    sidecar, so the hold dies with the sidecar (no stale hold). (gate rc, why): 0 held · 3 no live sidecar · 4 the GPU
    holds a live claim (for now) · else a bad call."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    r = CH.gate("hold", gpu, str(sidecar_pid), run)
    return r.returncode, (r.stderr.strip() or r.stdout.strip())


def _finished_finals(run: str, sdir: Path) -> set[str]:
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    tsv = CH.judge_files(run, sdir)[1]
    rows = [ln.split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines()[1:]] if tsv.is_file() else []
    return {r[2] for r in rows if len(r) > 8 and r[1] == "f" and r[8] in ("OK", "INVALID")}


def judge_backlog_h(
    cells: list[Path], own_run: str, judge_gpus: list[str], claims: list[dict] | None = None, now: float | None = None
) -> float:
    """Hours of final-judge work ahead of `own_run` on the judge GPUs — the quantity the planning model's overflow
    rule reads (Σ over judge GPUs of their busy time past now), measured from the host:
      running  every live judge claim (gate kind gpu) on a judge GPU, other runs only: max(JUDGE_FINAL_H − age, 0)
      queued   every other cell whose training ended (train.rc 0) with a live sidecar, no judge rc yet and no
               overflow hold (its finals go to its own GPU — the model keeps them off the judge GPUs too): its finals
               neither finished (f rows OK/INVALID) nor running (a live claim of its final alias anywhere) ×
               JUDGE_FINAL_H
      blocked  a live training or data claim on a judge GPU (the judge-only reservation keeps them off while the loop
               lives — one there means it was lost): JUDGE_FINAL_H each (estimate: its end is unknown; a lower
               bound, not 0)
    Excluded: this run itself · held runs · cells whose sidecar is dead (the loop relaunches it; state trained)."""
    from dexengine.rl import campaign_model as CPLM  # noqa: PLC0415
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    claims = CH.gate_claims() if claims is None else claims
    now = time.time() if now is None else now
    held = {c["run"] for c in claims if c["kind"] == "hold"}
    finals_running: dict[str, int] = {}
    backlog = 0.0
    for c in claims:
        if c["kind"] != "gpu":
            continue
        if re.fullmatch(re.escape(c["run"]) + r"_jf\d{4,}", c["alias"]):
            finals_running[c["run"]] = finals_running.get(c["run"], 0) + 1
        if c["gpu"] in judge_gpus and c["run"] != own_run:  # a claim whose start is unreadable counts in full
            age = (now - int(c["t0"])) / 3600 if c.get("t0", "").isdigit() else 0.0
            backlog += max(CPLM.JUDGE_FINAL_H - age, 0.0)
    backlog += CPLM.JUDGE_FINAL_H * sum(c["kind"] in ("train", "data") and c["gpu"] in judge_gpus for c in claims)
    for c in cells:
        rec = CH.cell_file(c, "launch")
        if not rec.is_file():
            continue
        run = json.loads(rec.read_text(encoding="utf-8"))["run"]
        if run == own_run or run in held or CH.judge_verdict(run, c) is not None:
            continue
        trc, jp = CH.cell_file(c, "train_rc"), CH.pid_of(CH.cell_file(c, "judge_pid"))
        if not (trc.is_file() and trc.read_text().strip() == "0" and jp is not None and CH.alive(jp, run)):
            continue
        left = CH.JUDGE_FINALS - len(_finished_finals(run, c)) - finals_running.get(run, 0)
        backlog += max(left, 0) * CPLM.JUDGE_FINAL_H
    return backlog


def overflow_at_train_end(
    run: str,
    sdir: Path,
    gpu: str,
    cells: list[Path],
    judge_gpus: list[str],
    threshold_h: float | None,
    claims: list[dict] | None = None,
) -> tuple[str, str]:
    """The overflow step of the runner at a training's end, before `gpu` is refilled: the expected judge wait
    (judge_backlog_h — the model's quantity spread over the judge GPUs) against the threshold
    (`threshold_h`); due → hold `gpu` for this run's finals under its live
    sidecar. (decision ∈ OVERFLOW, why) — decided by structure, never by the words of a reason:
      wait  the training process still lives (its claim would refuse the hold) · the gate refused the hold for
            now (rc 3 the sidecar is between starts · rc 4 a live claim on the GPU)
      no    not due (or overflow off) · no live sidecar to bind the hold to
      held  the gate holds `gpu` under the sidecar's pid
    Any other gate answer raises (a bad call is loud, not a silent no-hold)."""
    from dexengine.rl import campaign_model as CPLM  # noqa: PLC0415
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    tp = CH.pid_of(CH.cell_file(sdir, "train_pid"))
    if tp is not None and CH.alive(tp, run):
        return "wait", f"{run}: the training process still lives — next tick"
    w = CPLM.live_wait_h(judge_backlog_h(cells, run, judge_gpus, claims), len(judge_gpus))
    if not CPLM.overflow_due(w, threshold_h):
        return "no", f"judge wait {w:.2f} h — at or below the threshold {threshold_h} (or off)"
    jp = CH.pid_of(CH.cell_file(sdir, "judge_pid"))
    if jp is None or not CH.alive(jp, run):
        return "no", f"{run}: no live sidecar — no hold (a hold lives as long as the sidecar)"
    rc, why = hold(gpu, jp, run)
    if rc == 0:
        return "held", f"judge wait {w:.2f} h > {threshold_h} h → GPU {gpu} held for the finals of {run}"
    if rc in (3, 4):
        return "wait", f"judge wait {w:.2f} h > {threshold_h} h but the gate refuses the hold for now (rc {rc}): {why}"
    raise RuntimeError(f"{run}: host gate hold rc {rc} — {why}")
