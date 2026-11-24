"""The dry run of a training or judging launch (DRYRUN=1 · train, smoke, judge) starts every resolved trainer process
once in its preflight mode (`process.preflight`): the same argv and environment with DEXENGINE_PREFLIGHT_ONLY=1 in that
process only, on the CPU, output beside the dump; the dry run fails when the preflight fails, ends without its PASS
line or runs past its time, and the preflight process dies with the launcher. Render and replay dry runs and the dumps
data recording replays (DUMP_ONLY=1) start nothing. A real launch never carries the flag; a caller cannot pass it."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dexengine.rl import launch
from dexengine.rl.cli_entry import preflight as PF
from dexengine.rl.launch import process, request
from dexengine.rl.launch.recipe import RecipeError
from dexengine.rl.tasks import switches
from tests.rl.test_launch import CASES, parse_case

REPO = Path(__file__).resolve().parents[2]
ARGV = ["dexengine/rl/cli.py", "--task", "train", "--sequence_dir", "/t", "--resume_from", "/r/checkpoints/a.pt"]


def _launch(tmp_path: Path, preflight: bool = True) -> process.Launch:
    log = tmp_path / "logs" / "R_frozenreplay_train.log"
    env = {"DEXENGINE_ARM": "rby1", switches.SWITCHES_ENV: str(process.switches_path(log))}
    pe = {"CUDA_VISIBLE_DEVICES": "3"}
    return process.Launch(argv=list(ARGV), env=env, process_env=pe, log=log, preflight=preflight)


class _Popen:
    """Stand-in child: records its command and environment, writes `text` into its output, ends with `rc` (or never
    ends when `rc` is None)."""

    calls: list = []

    def __init__(self, text: str, rc: int | None):
        self.text, self.rc, self.killed = text, rc, False

    def __call__(self, cmd, cwd, env, stdout, stderr, start_new_session, preexec_fn):
        assert stderr == subprocess.STDOUT and start_new_session and callable(preexec_fn)
        stdout.write(self.text.encode())
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        self.pid = 4242
        return self

    def wait(self, timeout=None):
        if self.rc is None and not self.killed:
            raise subprocess.TimeoutExpired("python", timeout)
        return -9 if self.rc is None else self.rc

    def poll(self):
        return None if self.rc is None and not self.killed else self.rc


@pytest.fixture
def child(monkeypatch, tmp_path):
    monkeypatch.setenv("ISAACLAB_DIR", str(tmp_path / "isaac"))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    _Popen.calls = []
    killed = []

    def use(text: str, rc: int | None) -> _Popen:
        fake = _Popen(text, rc)
        monkeypatch.setattr(process.subprocess, "Popen", fake)

        def killpg(pid, sig):
            killed.append(pid)
            fake.killed = True

        monkeypatch.setattr(process.os, "killpg", killpg)
        return fake

    use.killed = killed
    return use


def test_a_dry_run_starts_the_trainer_once_in_preflight_mode(child, tmp_path):
    child(f"[PRE_APP] import order ok\n{PF.PASS} · extra slots on · clips 26\n", 0)
    p = _launch(tmp_path)
    assert process.start(p, tmp_path, dry_run=True) == 0
    (call,) = _Popen.calls
    venv_python = str(tmp_path / "isaac" / "env_isaaclab" / "bin" / "python")  # what isaaclab.sh -p runs
    assert call["cmd"] == [venv_python, *ARGV] and call["cwd"] == tmp_path
    env = call["env"]
    assert env[PF.ENV] == "1" and env["CUDA_VISIBLE_DEVICES"] == "" and env["OMP_NUM_THREADS"] == "1"
    assert env["ISAACLAB_PATH"] == str(tmp_path / "isaac")
    assert env["DEXENGINE_ARM"] == "rby1" and env[switches.SWITCHES_ENV] == p.env[switches.SWITCHES_ENV]
    assert p.log.read_text() == p.dump()  # the dump stays the replayable argv + env
    assert PF.ENV not in p.dump()
    assert PF.PASS in process.preflight_log(p.log).read_text()


@pytest.mark.parametrize(
    "text,rc,want",
    [
        ("ValueError: [EXTRA_SLOTS] PART_UNMERGED c1: main right_object(o1) …\n", 1, 1),
        ("[EXTRA_SLOTS] preflight train: spawn lists of 1/1 clips differ from training\n", 1, 1),
        ("[PRE_APP] import order ok\n", 0, 1),  # rc 0 without the PASS line is no pass
    ],
)
def test_a_failing_preflight_fails_the_dry_run(child, tmp_path, capsys, text, rc, want):
    child(text, rc)
    assert process.start(_launch(tmp_path), tmp_path, dry_run=True) == want
    out = capsys.readouterr().out
    assert f"{PF.MARK} FAIL rc={rc}" in out and text.strip()[:40] in out


def test_a_preflight_past_its_time_is_killed_and_fails(child, tmp_path, capsys):
    child("[PRE_APP] import order ok\n", None)
    assert process.start(_launch(tmp_path), tmp_path, dry_run=True) == process.PREFLIGHT_TIMEOUT_RC
    assert child.killed == [4242]
    assert f"no end within {process.PREFLIGHT_TIMEOUT_S} s" in capsys.readouterr().out


def test_a_launch_without_preflight_starts_nothing(child, tmp_path, capsys):
    """(a) The dry run of a render, a replay or a DUMP_ONLY launch writes its dump and starts no process."""
    child(f"{PF.PASS}\n", 0)
    p = _launch(tmp_path, preflight=False)
    assert process.start(p, tmp_path, dry_run=True) == 0
    assert _Popen.calls == [] and not process.preflight_log(p.log).exists()
    assert PF.MARK not in capsys.readouterr().out and p.log.read_text() == p.dump()


def test_a_real_launch_carries_no_preflight_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ISAACLAB_DIR", str(tmp_path / "isaac"))
    seen = {}

    def run(cmd, cwd, env, stdout, stderr):
        seen.update(env)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(process.subprocess, "run", run)
    monkeypatch.setattr(process, "preflight", lambda launch, root: pytest.fail("a real launch ran the preflight"))
    assert process.start(_launch(tmp_path), tmp_path, dry_run=False) == 0
    assert PF.ENV not in seen and seen["CUDA_VISIBLE_DEVICES"] == "3"


def test_a_caller_cannot_pass_the_preflight_flag():
    with pytest.raises(RecipeError, match=PF.ENV):
        launch.from_env("R", {"GPU": "0", "SIDE": "both", PF.ENV: "1"})


# ── which dry runs run the preflight ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", request.MODES)
def test_dry_inputs_by_mode(mode):
    assert request._dry_inputs({"DRYRUN": "1"}, mode) == (True, mode in ("train", "smoke", "judge"))
    assert request._dry_inputs({"DUMP_ONLY": "1"}, mode) == (True, False)
    assert request._dry_inputs({}, mode) == (False, False)
    with pytest.raises(RecipeError, match="give one"):
        request._dry_inputs({"DRYRUN": "1", "DUMP_ONLY": "1"}, mode)
    with pytest.raises(RecipeError, match="only 0|1"):
        request._dry_inputs({"DUMP_ONLY": "yes"}, mode)


def test_recorded_launches_preflight_only_at_training_and_judging_entries():
    """(a) · (c) Over every recorded launch: DRYRUN=1 asks for the preflight at train · smoke · judge, not at render ·
    replay; the same launch with DUMP_ONLY=1 (data recording) asks for none."""
    seen = {}
    for case in CASES:
        env, _ = parse_case(case)
        req = launch.from_env("R", env)
        assert req.dry_run and req.preflight == (req.mode in request.PREFLIGHT_MODES), case.stem
        dump = {k: v for k, v in env.items() if k != "DRYRUN"} | {"DUMP_ONLY": "1"}
        r2 = launch.from_env("R", dump)
        assert r2.dry_run and not r2.preflight, case.stem
        seen[req.mode] = req.preflight
    assert seen == {"train": True, "smoke": True, "judge": True, "render": False, "replay": False}


def test_data_recording_asks_for_dumps_only(monkeypatch, tmp_path):
    from dexengine.record import bundle

    seen = {}

    def run(cmd, cwd, stdout, stderr, timeout):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(bundle.subprocess, "run", run)
    bundle.dryrun(tmp_path, ["SIDE=both"], "a1", tmp_path / "d.out")
    assert "DUMP_ONLY=1" in seen["cmd"] and not any(t.startswith("DRYRUN=") for t in seen["cmd"])


# ── (b) the preflight process never outlives its launcher ─────────────────────────────────────────────────────────

_PARENT = """
import sys
from pathlib import Path
from dexengine.rl.launch import process
if sys.argv[2] == "off":  # the negative: no parent-death signal
    process._die_with_parent = lambda parent: (lambda: None)
root = Path(sys.argv[1])
log = root / "logs" / "R_train.log"
env = {"DEXENGINE_RL_SWITCHES": str(process.switches_path(log))}
L = process.Launch(argv=["dexengine/rl/cli.py"], env=env, process_env={}, log=log, preflight=True)
process.start(L, root, True)
"""


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _child_after_launcher_killed(tmp_path: Path, arm: str) -> int:
    """Start a launcher whose preflight process (a stand-in interpreter that sleeps 60 s) outlives the launcher's
    own life, kill the launcher with SIGKILL (its time limit), and return the preflight process's pid."""
    py = tmp_path / "isaac" / "env_isaaclab" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text('#!/bin/sh\necho $$ > "$PIDFILE"\nexec sleep 60\n')
    py.chmod(0o755)
    pidfile = tmp_path / "pid"
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin", "ISAACLAB_DIR": str(tmp_path / "isaac")}
    env |= {"PIDFILE": str(pidfile), "CUDA_VISIBLE_DEVICES": ""}
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, str(tmp_path), arm], cwd=REPO, env=env)
    end = time.time() + 60
    while not (pidfile.is_file() and pidfile.read_text().strip()) and time.time() < end and parent.poll() is None:
        time.sleep(0.1)
    pid = int(pidfile.read_text())
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait()
    end = time.time() + 10
    while _alive(pid) and time.time() < end:
        time.sleep(0.1)
    return pid


def test_the_preflight_process_dies_with_its_launcher(tmp_path):
    pid = _child_after_launcher_killed(tmp_path, "on")
    assert not _alive(pid)


def test_without_the_parent_death_signal_an_orphan_would_remain(tmp_path):
    """Negative of the test above: the same launcher without `_die_with_parent` leaves its preflight running."""
    pid = _child_after_launcher_killed(tmp_path, "off")
    try:
        assert _alive(pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
