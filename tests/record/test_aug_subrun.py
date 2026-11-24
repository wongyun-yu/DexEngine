"""Augment child runs (record.augment.subrun) and the failure cause field (core.errors), with the
child rc cases. CPU only.

  bound    cap formula: floor before the work line · work = unit x checks x T / workers x margin · a smaller deadline
           side wins · no deadline
  genbound no bundle deadline env = stated in the marker · not a number dies · decided again on the first work line
  run      done line -> return · rc 0 without it = no_done · rc != 0 = crash · signal = 128 + n · past the cap =
           timeout (rc 124, grandchild reaped) · the work line re-decides the cap · deadline side < MIN_LEFT_S = not
           launched
  pool     results in input order · one progress line per item
  cause    unknown values die · input missing = the marker (never cause=input_missing) · bundle_rc: the cluster timeout
           sentence is AUG_FAILED, the marker INPUT_MISSING · AUG_FAIL line starts with the cause field
  wiring   generation cap GenBound · run_logged = subrun.run · work line before the bake · collision gate pool_map ·
           driver session env deadline · AUG_FAIL cause field · no old fixed generation cap; a copy without a call is
           caught
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from dexengine.core import errors as ERR
from dexengine.core.repo import REPO_ROOT
from dexengine.record import bundle as C
from dexengine.record import driver_spec as DS
from dexengine.record.augment import stage as AS
from dexengine.record.augment import subrun as SR

WORK = {"checks": 69, "T": 331, "jobs": 6}
WANT = SR.UNIT_S * 69 * 331 / 6 * SR.MARGIN


def _py(code: str) -> list:
    return [sys.executable, "-c", code]


def _exit_text(fn) -> str | None:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except SystemExit as e:
        return str(e)
    return None


def _slow_echo(x: tuple) -> dict:
    """pool_map item: earlier items finish later (finish order != input order)."""
    i, n = x
    time.sleep(0.05 * (n - i))
    return {"name": f"v{i}", "ok": True, "sec": 0.0, "hh_sec": 0.0}


# ---- bound ----------------------------------------------------------------------------------------------------------
def test_bound_formula():
    assert SR.bound_s(None, None) == (SR.FLOOR_S, None)
    lim, w = SR.bound_s(WORK, None)
    assert abs(w - WANT) < 1e-6 and lim == w > SR.FLOOR_S
    assert SR.bound_s({"checks": 6, "T": 100, "jobs": 2}, None)[0] == SR.FLOOR_S  # never below the old fixed cap
    assert SR.bound_s(WORK, 1000.0)[0] == 1000.0 and SR.bound_s(None, 900.0)[0] == 900.0


def test_genbound_without_deadline_says_so():
    gb = SR.GenBound(1200, environ={})
    assert gb.deadline is None and gb.cap is None and "no deadline" in gb.line()


def test_genbound_deadline_not_a_number_dies():
    with pytest.raises(ValueError):
        SR.GenBound(1200, environ={SR.DEADLINE_ENV: "soon"})


def test_genbound_redecides_on_the_first_work_line_only():
    t0 = 1_000_000.0
    gb = SR.GenBound(1200, environ={SR.DEADLINE_ENV: str(t0 + 9885)}, now=t0)
    assert gb.cap == 9885 - 1200 and gb.limit == SR.FLOOR_S and not gb.update("[AUG_GEN] bake\n")
    txt = "x\n" + SR.work_line(69, 331, 6) + "\n"
    assert gb.update(txt) and gb.limit == min(8685.0, WANT) and not gb.update(txt) and "checks 69" in gb.line()


def test_deadline_env_is_start_plus_cap():
    assert SR.deadline_env(1000.4, 600) == {SR.DEADLINE_ENV: "1600"}


# ---- run ------------------------------------------------------------------------------------------------------------
@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(SR, "POLL_S", 0.1)
    monkeypatch.setattr(SR, "KILL_AFTER_S", 2.0)


def test_run_outcomes(tmp_path, fast):
    env, lg = dict(os.environ), tmp_path / "r.log"
    assert _exit_text(lambda: SR.run(_py("print('[X] DONE')"), lg, env, "[X] DONE", 30, "[T]")) is None
    t = _exit_text(lambda: SR.run(_py("print('half')"), lg, env, "[X] DONE", 30, "[T]")) or ""
    assert "cause=no_done" in t and "rc 0" in t and t.endswith(f"— {lg}")
    t = _exit_text(lambda: SR.run(_py("import sys; sys.exit(3)"), lg, env, "[X] DONE", 30, "[T]")) or ""
    assert "cause=crash" in t and "-c rc 3" in t
    code = "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
    t = _exit_text(lambda: SR.run(_py(code), lg, env, "[X] DONE", 30, "[T]")) or ""
    assert "cause=crash" in t and "rc 139" in t


def test_run_timeout_reaps_the_grandchild(tmp_path, fast):
    pid_f, lg = tmp_path / "grand.pid", tmp_path / "r.log"
    code = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f" open({str(pid_f)!r}, 'w').write(str(p.pid)); time.sleep(60)"
    )
    t0 = time.monotonic()
    t = _exit_text(lambda: SR.run(_py(code), lg, dict(os.environ), "[X] DONE", 1, "[T]")) or ""
    el = time.monotonic() - t0
    gone = False
    for _ in range(50):
        try:
            os.kill(int(pid_f.read_text()), 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.1)
    assert "cause=timeout" in t and f"rc {SR.RC_TIMEOUT}" in t and el < 20 and gone


def test_run_work_line_redecides_the_cap(tmp_path, fast, monkeypatch):
    monkeypatch.setattr(SR, "MIN_LEFT_S", 0)
    now = time.time()
    gb = SR.GenBound(10, environ={SR.DEADLINE_ENV: str(now + 10 + 3)}, now=now)
    out = io.StringIO()
    code = f"print({SR.work_line(69, 331, 6)!r}, flush=True); import time; time.sleep(60)"
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as e:
        SR.run(_py(code), tmp_path / "r.log", dict(os.environ), "[X] DONE", gb, "[T]")
    marks = [ln for ln in out.getvalue().splitlines() if SR.BOUND_MARK in ln]
    assert "cause=timeout" in str(e.value) and len(marks) == 2 and gb.work == WORK and "deadline side" in marks[1]


def test_run_not_launched_below_min_left(tmp_path, fast):
    lg = tmp_path / "never.log"
    gb = SR.GenBound(10, environ={SR.DEADLINE_ENV: str(time.time() + 10 + 5)})
    t = _exit_text(lambda: SR.run(_py("print('[X] DONE')"), lg, dict(os.environ), "[X] DONE", gb, "[T]")) or ""
    assert "cause=timeout" in t and "not launched" in t and not lg.exists()


# ---- pool -----------------------------------------------------------------------------------------------------------
def test_pool_map_keeps_input_order():
    items = [(i, 6) for i in range(6)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out), ProcessPoolExecutor(3) as ex:
        res = SR.pool_map(ex, _slow_echo, items, "[T]")
    prog = [ln for ln in out.getvalue().splitlines() if SR.PROGRESS in ln]
    assert [r["name"] for r in res] == [f"v{i}" for i in range(6)] and len(prog) == 6 and "6/6" in prog[-1]


# ---- cause ----------------------------------------------------------------------------------------------------------
def test_unknown_cause_dies():
    for fn in (lambda: ERR.cause("slow"), lambda: ERR.cause_of("x cause=slow")):
        with pytest.raises(ValueError):
            fn()
    with pytest.raises(ValueError):  # input missing has one statement: the marker
        ERR.cause_of("x cause=input_missing")


def test_cause_field_word_boundary_and_marker():
    assert ERR.cause_of("because=timeout") is None and ERR.cause_of("a · cause=timeout · b") == "timeout"
    assert ERR.cause("input_missing") == ERR.INPUT_MISSING
    assert ERR.cause_of(ERR.input_missing("[X] y missing")) == ERR.CAUSE_INPUT_MISSING
    assert ERR.cause_of("plain text") is None


def test_bundle_rc_reads_the_rc_and_the_marker():
    """A bundle whose child timed out (rc 124 child, rc 3 bundle) is AUG_FAILED, never INPUT_MISSING."""
    line = (
        "rc 3 · [AUG_COLLECT] FAIL cause=timeout · SystemExit: [AUG_COLLECT] cause=timeout · place_gen.py rc 124 · "
        "[AUG_GEN] DONE missing · 1800 s / cap 1800 s — /w/gen.log"
    )
    aug = {"state": "FAILED", "rc": C.RC_AUG_FAILED, "reason": line}
    assert DS.bundle_rc(aug) == DS.CODE["AUG_FAILED"]
    assert DS.bundle_rc(dict(aug, reason="rc 3 · SystemExit: g.py rc 124 · DONE missing")) == DS.CODE["AUG_FAILED"]
    im = f"rc 3 · [AUG_COLLECT] FAIL {ERR.cause('input_missing')} · InputMissingError: [AUG_COLLECT] x missing"
    assert DS.bundle_rc(dict(aug, reason=im)) == DS.CODE["INPUT_MISSING"]
    assert DS.bundle_rc({"state": "FAILED", "rc": 1, "reason": "rc 1 · cause=crash · x DONE missing"}) == DS.BROKEN


def test_judged_bundle_missing_carries_the_marker(tmp_path):
    a = types.SimpleNamespace(hand="xhand_pro", main=tmp_path / "main")
    t = _exit_text(lambda: AS.judged_bundle(a, {"campaign": "c", "bundle": "b"})) or ""
    assert ERR.cause_of(t) == ERR.CAUSE_INPUT_MISSING


# ---- wiring ---------------------------------------------------------------------------------------------------------
FILES = {
    "stage": "dexengine/record/augment/stage.py",
    "place_gen": "dexengine/record/augment/place_gen.py",
    "driver_bundle": "dexengine/record/driver_bundle.py",
    "chain": "dexengine/record/chain.py",
}


def wiring_problems(src: dict) -> list[str]:
    """Wiring check (calls and order, not words). src = {name: source text}."""
    bad = []
    st, pg, db, ch = src["stage"], src["place_gen"], src["driver_bundle"], src["chain"]
    if "SR.GenBound(TIMEOUT_PER_CLIP_S)" not in st or "SR.run(cmd, log, env, done, limit, MARK)" not in st:
        bad.append("stage: generation cap GenBound · run_logged = SR.run")
    if "GEN_TIMEOUT_S" in st or '"timeout", "-k"' in st:
        bad.append("stage: old fixed generation cap")
    i_w, i_b = pg.find("_work_line(a, src, len(todo) + 2)"), pg.find("bake = bake_step(a, src, recs)")
    if not 0 <= i_w < i_b or "SR.pool_map(ex, ACK.pre_one, gate, MARK)" not in pg:
        bad.append("place_gen: work line before the bake · collision gate pool_map")
    if "SR.deadline_env(time.time(), timeout)" not in db or "stderr=subprocess.STDOUT, env=env" not in db:
        bad.append("driver_bundle: bundle session env deadline")
    if "{AUG_FAIL} {ERR.cause(why)} ·" not in ch:
        bad.append("chain: AUG_FAIL cause field first")
    return bad


def test_wiring_and_its_negatives():
    src = {k: (REPO_ROOT / v).read_text() for k, v in FILES.items()}
    assert wiring_problems(src) == []
    muts = [
        ("stage", "SR.GenBound(TIMEOUT_PER_CLIP_S)", "1800"),
        ("place_gen", "_work_line(a, src, len(todo) + 2)", "len(todo)"),
        ("place_gen", "SR.pool_map(ex, ACK.pre_one, gate, MARK)", "list(ex.map(ACK.pre_one, gate))"),
        ("driver_bundle", "SR.deadline_env(time.time(), timeout)", "{}"),
        ("chain", "{AUG_FAIL} {ERR.cause(why)} ·", "{AUG_FAIL}"),
    ]
    for k, a, b in muts:
        assert src[k].count(a) == 1, (k, a)
        assert wiring_problems(dict(src, **{k: src[k].replace(a, b)})), (k, a)


def test_work_line_form_is_what_genbound_reads():
    assert SR.WORK_RE.search(SR.work_line(3, 10, 2)) and Path(SR.__file__).is_file()
