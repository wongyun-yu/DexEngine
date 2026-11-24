"""The RL campaign launch loop around the ticks (dexengine/cluster/scheduler.py — one tick; dexengine/rl/campaign.py
--launch runs this):

  start     one loop per campaign (flock on loop.lock — a second loop is refused, rc 2) · the code tree fingerprint
            (ctx.code) · the campaign's settings frozen at the first start (params_of → params.json; a
            restart with other settings is refused unless --change-params names why — params_history.tsv) · the
            judge GPUs reserved judge-only (host gate kind rsv under the loop's pid; a live training/data claim on one
            refuses the start) · pidfile loop.pid "pid time host campaign" (removed at the end)
  ticks     a tick every tick_s; its status → status.json (atomic) · states.tsv · events.tsv (appended) in the loop dir
            (paths layer run_rl_loops / <campaign>) — the dashboard and the operator read these
  errors    a tick that raises is recorded (status loop=error, the error, events) and the next tick tries again;
            TICK_ERRORS_MAX in a row (or one under --once / while stopping) end the loop: loop=crashed, rc RC_CRASHED
  SIGTERM   no launch from then on; the loop ends after the current tick, or within a second while it sleeps
            (loop=stopped). Children (trainings, sidecars) run on in their own sessions
  drain     a drain request (loop dir drain.json): no launch; once nothing is in flight the loop ends (loop=drained,
            rc 0) — the code swap's ff comes after. A restart with other code (the resume) goes only through
            loop_files.check_swap (nothing in flight · a training-path change named) and clears the request
  refusal   a done bundle whose finalize is refused stays live (the loop does not end on it) and is tried again each
            tick: each tick prints `[TICK] ⚠ FINALIZE_REFUSED <hand/bundle> — <why>` and status.json carries
            `finalize_refused` {bundle: why} — a watcher alerts on it (EV_FINALIZE_REFUSED),
            since `loop` stays running with a fresh `time`
status.json `loop` = running · stopped · drained · ended · error · crashed. A watcher reads `loop`, the age of `time`
against `tick_s` and `pid` (checked by its command line) — a SIGKILL leaves `running` with an ageing `time`.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import time
from pathlib import Path

from dexengine.cluster import loop_files as LF
from dexengine.cluster import scheduler as CT
from dexengine.cluster.loop_files import PARAMS, PARAMS_LOG
from dexengine.cluster.scheduler import Ctx

#: Tick errors in a row before the loop gives up (status loop=crashed, rc RC_CRASHED) — one error (a gate call that
#: failed once, a file moved under a read) is recorded (loop=error, events) and the next tick tries again.
TICK_ERRORS_MAX = 3
RC_CRASHED = 4


#: The loop dir's frozen settings and their change ledger (freeze_params).
def write_status(ldir: Path, st: dict, hand: str, root=None) -> None:
    """status.json (atomic) · states.tsv (generated each tick) · events.tsv (appended) in the loop dir."""
    ldir.mkdir(parents=True, exist_ok=True)
    tmp = ldir / ".status.json.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, ldir / "status.json")
    tmp = ldir / ".states.tsv.tmp"
    tmp.write_text(
        "hand\tbundle\tstate\n"
        + "".join(f"{k.split('/')[0]}\t{k.split('/', 1)[1]}\t{v}\n" for k, v in st.get("states", {}).items()),
        encoding="utf-8",
    )
    os.replace(tmp, ldir / "states.tsv")
    with open(ldir / "events.tsv", "a", encoding="utf-8") as f:
        f.writelines("\t".join(map(str, e)) + "\n" for e in st.get("events", []))


def params_of(ctx: Ctx) -> dict:
    """What a campaign runs under (frozen at its first start — freeze_params): the host and its GPU split (on the
    explicit path also the declared resources — ctx.resources; absent on the host path), overflow,
    prune, the disk plan, the ordered queue, the blocked set, the code fingerprint and the launch identity
    (loop_files.IDENT — training path · commands · inherited env) with the first start's (ident_first — the
    CANARY_VALIDATED exemption holds while they are equal). --tick-s is not in it."""
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    return dict(
        host=ctx.host,
        train_gpus=list(ctx.train_gpus),
        judge_gpus=list(ctx.judge_gpus),
        overflow_h=ctx.overflow_h,
        prune=ctx.prune,
        per_run_gb=round(ctx.per_run_gb, 6),
        queue=[f"{h}/{i}" for h, i in map(bkey, ctx.bundles) if (h, i) not in ctx.blocked],
        blocked=sorted(f"{h}/{i}" for h, i in ctx.blocked),
        code=ctx.code0,
        train_fp=ctx.train0,
        **ctx.ident,
        ident_first=ctx.ident_first,
        **({"resources": ctx.resources} if ctx.resources else {}),
    )


def _short(v) -> str:
    return f"[{len(v)} items]" if isinstance(v, list) and len(v) > 4 else json.dumps(v, ensure_ascii=False)


def freeze_params(ldir: Path, ctx: Ctx, change_why: str = "", allow_train: str = "") -> tuple[bool, str]:
    """params.json (first start: written) against this start's params_of: equal → go · different → refused unless
    change_why names why (then rewritten, the diff and the why appended to params_history.tsv: a restart that
    drops a flag never changes the campaign silently). A different code fingerprint is a code swap: it also has to
    pass loop_files.check_swap (nothing in flight · a training-path change only with allow_train) and ledgers a
    `swap` row (commit · both fingerprints · both reasons)."""
    new, f = params_of(ctx), ldir / PARAMS
    commit = str(ctx.git().get("commit", "?"))

    def write(kind: str, show: str, why: str) -> None:
        tmp = ldir / f".{PARAMS}.tmp"
        tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, f)
        LF.ledger_row(ldir, kind, show, why)

    old = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
    if allow_train.strip() and (old is None or not LF.ident_moved(old, new)):
        return False, (
            "[TICK] ✗ --allow-train-path-change was given but the launch identity (training path · "
            "commands · launch env) equals the frozen one (or this is the first start) — an input with no "
            "consumer"
        )
    if old is None:
        write("first", f"commit {commit[:12]}", change_why or "first start")
        return True, f"[TICK] campaign settings frozen → {f}"
    diff = {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
    if not diff:
        return True, f"[TICK] campaign settings = frozen ({f.name})"
    show = " · ".join(f"{k}: {_short(a)} → {_short(b)}" for k, (a, b) in diff.items())
    if not change_why.strip():
        return False, (
            f"[TICK] ✗ campaign settings differ from the frozen ones ({f}) — {show}. To change them pass "
            f'--change-params "<why>" (ledger {PARAMS_LOG}) · otherwise relaunch with the missing flags'
        )
    live = LF.in_flight_keys(ctx.bundles, CT.read_all(ctx), ctx.cell_of)
    ok, kind = LF.check_swap(old, new, live, allow_train)
    if not ok:
        return False, f"[TICK] ✗ {kind}"
    if kind == "swap":
        allow = f" · launch identity change allowed: {allow_train}" if LF.ident_moved(old, new) else ""
        write("swap", LF.swap_show(old, new, commit) + " · " + show, change_why + allow)
    else:
        write("change", show, change_why)
    return True, (
        f"[TICK] ⚠ campaign settings changed (explicit{', code swap' if kind else ''}) — {show} · why: {change_why}"
    )


def run_loop(ctx: Ctx, tick_s: float, once: bool = False, change_why: str = "", allow_train: str = "") -> int:
    """The loop (module doc): rc 0 = complete (done set = queue set), stopped, drained or --once · 2 = refused at the
    start (a second loop · settings ≠ the frozen ones without change_why · a code swap check_swap refuses · a judge
    GPU not reservable) · 3 = ended incomplete
    (failed bundles — the operator diagnoses, the relaunch tool clears) · 4 = crashed (status loop=crashed)."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    ldir = LF.loop_dir(ctx.campaign, ctx.root)
    hand = ctx.bundles[0]["hand"]
    ldir.mkdir(parents=True, exist_ok=True)
    with open(ldir / "loop.lock", "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            other = (ldir / "loop.pid").read_text().strip() if (ldir / "loop.pid").is_file() else "?"
            print(f"[TICK] ✗ the loop of campaign {ctx.campaign} is already running ({other}) — no second loop")
            return 2
        ctx.code0, ctx.train0 = ctx.code(), ctx.train()
        old = json.loads((ldir / PARAMS).read_text(encoding="utf-8")) if (ldir / PARAMS).is_file() else None
        ctx.ident_first = (old or {}).get("ident_first") or LF.ident_of(old or dict(train_fp=ctx.train0, **ctx.ident))
        # the judge reservation before any settings or resume row: a refused reservation (rc 2) leaves no swap row, and
        # the drain request stays
        ok, why = CH.reserve_judges(ctx.judge_gpus, os.getpid(), ctx.campaign)
        if not ok:
            print(f"[TICK] ✗ could not reserve judge GPUs {ctx.judge_gpus} for judging only — {why}", flush=True)
            return 2
        try:
            ok, msg = freeze_params(ldir, ctx, change_why, allow_train)
            print(msg, flush=True)
            if not ok:
                return 2
            ctx.exempt = LF.ident_of(params_of(ctx)) == ctx.ident_first
            if not ctx.exempt:
                print(
                    "[TICK] the launch identity differs from the first start — even validated PARK keys "
                    "(CANARY_VALIDATED) need a canary .ok to launch",
                    flush=True,
                )
            if (
                change_why.strip()
                and LF.read_request(ldir)
                and not LF.in_flight_keys(ctx.bundles, CT.read_all(ctx), ctx.cell_of)
            ):
                LF.clear(ldir, "drain_cleared", f"resume: {change_why}")
                print("[TICK] drain request withdrawn (resume — --change-params)", flush=True)
            print(
                f"[TICK] judge GPUs {','.join(ctx.judge_gpus)} reserved for judging only (host gate rsv) · code "
                f"{ctx.code0}",
                flush=True,
            )
            return _loop(ctx, ldir, hand, tick_s, once)
        finally:
            with contextlib.suppress(Exception):
                CH.unreserve_judges(ctx.judge_gpus, os.getpid())


def _mark(ldir: Path, ctx: Ctx, hand: str, last: dict, n: int, tick_s: float, word: str, why: str, errs: int) -> None:
    """The status of a tick that did not finish or of the loop's end outside a tick: the last good status (this loop's,
    else the one on disk — every field a reader knows stays) with the loop word, the time, the error and the events not
    yet written (a crashed tick's partial events + one marker row)."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not last:
        with contextlib.suppress(OSError, ValueError):
            last = json.loads((ldir / "status.json").read_text(encoding="utf-8"))
    st = dict(last or {"campaign": ctx.campaign, "host": ctx.host, "states": {}})
    st.update(
        time=now,
        pid=os.getpid(),
        tick=n,
        tick_s=tick_s,
        loop=word,
        errors_in_row=errs,
        tick_error=why or None,
        events=[list(e) for e in ctx.events] + [[now, f"loop_{word}", "-", "-", " ".join(why.split())[:500]]],
    )
    ctx.events = []
    try:
        write_status(ldir, st, hand, root=ctx.root)
    except Exception as e:  # noqa: BLE001 — the marker is best effort; the log line below still says it
        print(f"[TICK] ✗ could not write the status file either ({type(e).__name__}: {e})", flush=True)


def _sleep(ctx: Ctx, s: float) -> None:
    """Sleep up to s seconds in slices — a SIGTERM (ctx.stop) ends it within a second (time.sleep alone sleeps on after
    the handler, PEP 475)."""
    end = time.monotonic() + s
    while not ctx.stop and (left := end - time.monotonic()) > 0:
        time.sleep(min(left, 1.0))


def _tick_once(ctx, n, tick_s, once, ldir, hand):
    """One tick: run it, mark a drained loop, write the status files; returns the tick status."""
    st = CT.tick(ctx)
    drained = bool(st["drain"]) and not st["in_flight"] and not st["ended"]
    if drained:
        st["events"].append([st["time"], "drained", "-", "-", f"drain requested {st['drain'].get('why')}"])
    st.update(
        tick=n,
        tick_s=tick_s,
        errors_in_row=0,
        loop="ended" if st["ended"] else "drained" if drained else "stopped" if once or ctx.stop else "running",
    )
    write_status(ldir, st, hand, root=ctx.root)
    ctx.events = []
    return st


def _print_tick(st, n):
    """The tick's console summary and its idle-GPU / finalize-refused warnings."""
    c = st["counts"]
    print(
        f"[TICK {n} {st['time']}] queue {st['queue']} · done {st['done_finalized']} · {dict(c)} · idle GPUs "
        f"{ {g: v['idle'] for g, v in st['gpus'].items() if v.get('idle')} } · events {len(st['events'])}",
        flush=True,
    )
    for g in st["idle_bug"]:
        print(f"[TICK] ⚠ IDLE_BUG GPU {g} — idle for no reason while the queue is not empty", flush=True)
    for k, why in st["finalize_refused"].items():
        print(
            f"[TICK] ⚠ FINALIZE_REFUSED {k} — finalize refused (retried next tick · the loop does not end "
            f"until it clears): {why[:200]}",
            flush=True,
        )


def _loop(ctx: Ctx, ldir: Path, hand: str, tick_s: float, once: bool) -> int:
    pidf = ldir / "loop.pid"
    me = f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')} {ctx.host} {ctx.campaign}\n"
    pidf.write_text(me)
    old = signal.signal(signal.SIGTERM, lambda *_a: ctx.stop.append(1))
    n, errs, last, done = 0, 0, {}, False
    try:
        while True:
            if n and ctx.stop:  # SIGTERM while it slept — no further tick
                _mark(ldir, ctx, hand, last, n, tick_s, "stopped", "SIGTERM (between steps) — no next step", errs)
                print(
                    "[TICK] SIGTERM — stopping between ticks (no further tick · no launch · children keep running "
                    "in their own sessions)",
                    flush=True,
                )
                done = True
                return 0
            n += 1
            try:
                st = _tick_once(ctx, n, tick_s, once, ldir, hand)
            except Exception as e:  # noqa: BLE001 — every tick error is recorded; TICK_ERRORS_MAX in a row end the loop
                errs += 1
                final = errs >= TICK_ERRORS_MAX or once or bool(ctx.stop)
                why = f"{type(e).__name__}: {e}"
                _mark(ldir, ctx, hand, last, n, tick_s, "crashed" if final else "error", why, errs)
                print(f"[TICK {n}] ✗ tick error {errs}/{TICK_ERRORS_MAX} — {why[:300]}", flush=True)
                if final:
                    print(
                        f"[TICK] CRASHED — stopping the loop (rc {RC_CRASHED} · status.json loop=crashed · "
                        "children keep running in their own sessions)",
                        flush=True,
                    )
                    done = True
                    return RC_CRASHED
                _sleep(ctx, tick_s)
                continue
            errs, last = 0, st
            _print_tick(st, n)
            if st["ended"]:
                word = "COMPLETE" if st["complete"] else "INCOMPLETE"
                print(
                    f"[TICK] {word} done {st['done_finalized']}/{st['queue']} · failed {len(st['failed'])} "
                    f"{st['failed'][:5]} · missing {st['missing'][:5]} · manifest {st['manifest']} · excluded "
                    f"{st['blocked']} {st['blocked_why']}",
                    flush=True,
                )
                done = True
                return 0 if st["complete"] else 3
            if st["loop"] == "drained":
                print(
                    f"[TICK] DRAINED — 0 bundles in flight · drain request ({st['drain'].get('why')}) — ending the "
                    "loop. Resume after the ff with --change-params (loop_files.check_swap)",
                    flush=True,
                )
                done = True
                return 0
            if once or ctx.stop:
                print(
                    f"[TICK] {'--once' if once else 'SIGTERM'} — stopping the loop (children keep running in "
                    f"their own sessions)",
                    flush=True,
                )
                done = True
                return 0
            _sleep(ctx, tick_s)
    except BaseException as e:  # KeyboardInterrupt · SystemExit — say so before leaving
        if not done:
            _mark(
                ldir, ctx, hand, last, n, tick_s, "crashed", f"exception escaped the loop {type(e).__name__}: {e}", errs
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        with contextlib.suppress(OSError):
            if pidf.read_text() == me:
                pidf.unlink()
