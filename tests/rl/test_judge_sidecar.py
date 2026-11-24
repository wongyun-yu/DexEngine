"""The judge sidecar (scripts/rl/judge_sidecar.sh) on a fake repository: real sidecar, host gate and judge-metric
modules, a stand-in judge launcher (sidecar_fake_judge.py), fake GPUs and host memory. No Isaac Lab, no GPU.

Expected value of a finished 64-clip run at MI 1600: final judges 1400 and 1600 complete 64 % and 66 %, mean 65.00,
62 of 64 clips pass the 10 % rule."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from dexengine.core.layout.runs import RUN_CKPT_ROOT
from dexengine.core.repo import REPO_ROOT

pytestmark = pytest.mark.skipif(
    shutil.which("flock") is None or shutil.which("setsid") is None, reason="needs flock and setsid"
)
RUN = "R1"
REFBASE = "overfit/logs/refbase"
CKR = RUN_CKPT_ROOT


def add_ckpt(t: Path, roll: int, it: int) -> None:
    (t / CKR / RUN / "checkpoints" / f"agent_{it * roll}.pt").touch()
    with open(t / REFBASE / f"{RUN}_train.log", "a") as fh:
        fh.write(
            f"\r  3%|3   | {it * roll - 1}/51200 [40:10<20:01:00,  1.45s/it]\n"
            f"[CKPT_SAVED] agent_{it * roll}.pt iter={it}\n"
        )


def mk_repo(t: Path, upto: int = 1600, roll: int = 32, side: str = "both", k: int = 64, mem_gb: int = 400) -> Path:
    """A fake repository with a training run R1 whose checkpoints exist up to iteration `upto`."""
    for d in ("scripts/rl", REFBASE, f"{CKR}/{RUN}/checkpoints", "bin", "smi", "lock"):
        (t / d).mkdir(parents=True, exist_ok=True)
    for f in ("judge_sidecar.sh", "host_gate.sh"):
        shutil.copy(REPO_ROOT / "scripts/rl" / f, t / "scripts/rl" / f)
    shutil.copy(Path(__file__).with_name("sidecar_fake_judge.py"), t / "scripts/stage3_rl.py")
    for d in ("dexengine", "configs"):
        (t / d).symlink_to(REPO_ROOT / d)
    smi = t / "bin/nvidia-smi"
    smi.write_text(f'#!/bin/bash\ng=${{2:-0}}; cat "{t}/smi/$g" 2>/dev/null || echo 0\n')
    smi.chmod(0o755)
    (t / "meminfo").write_text(f"MemTotal: 1000000000 kB\nMemAvailable: {mem_gb * 1048576} kB\n")
    man = [
        "date=x git=abc1234 dirty=0 code=c0de00000001 codedirty=0 MODE=train OUT=R1",
        f"RUN=R1 {'' if side == 'both' else f'SIDE={side} '}SEQ=dir:s TOOL=0 LOBJ=0 NE=4096 MI=1600 MB=16 "
        "TIGHTEN=0.5 NOCOL=0 ROBOT_Z=0",
    ]
    man += [] if side == "both" else ["TRAIN_EXTRA="]
    man += [
        "COLL_PRESET=canon_r2r3off",
        "RESET_SCHEME=or PSA_FLAGS=x",
        "EXTRA=--arm rby1 --hand h LGATE_RAMP=0.10,180",
        "DEXENGINE_FINGER_EFFORT_SCALE=0.603",
    ] + ([] if roll == 32 else [f"DEXENGINE_ROLLOUTS={roll}"])
    (t / REFBASE / f"{RUN}_cmd.txt").write_text("\n".join(man) + "\n")
    (t / CKR / RUN / "coll_preset.txt").write_text("canon_r2r3off\n")
    from dexengine.rl.tasks import switches

    switches.write(t / CKR / RUN / "switches.json", switches.build({"arm_ff": "0"}), {"arm_ff"})
    (t / REFBASE / f"{RUN}_train.log").write_text(
        "[ROSTER] hand=h axis=a source=ledger:x universe=64 trees=64 mode=list:x → "
        f"train {k} (trees left out of the folder 0) sha256=0\n"
        "[INTRAIN_EVAL] off — no in-training eval\n"
    )
    for it in range(50, upto + 1, 50):
        add_ckpt(t, roll, it)
    return t


def sidecar(t: Path, *, side: str = "both", deadline: int = 90, **env) -> tuple[int, str]:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("DEXENGINE_", "RLJUDGE_", "FAKE_"))}
    e.update(
        PATH=f"{t}/bin:{e['PATH']}",
        DEXENGINE_DATASET="taco",
        DEXENGINE_EMBODIMENT="bimanual",
        DEXENGINE_ARM="rby1",
        DEXENGINE_HAND="h",
        SIDE=side,
        SEQ_DIR="/nonexistent",
        RLJUDGE_GPUS="0 1",
        RLJUDGE_POLL_S="0.2",
        RLJUDGE_LOCK_DIR=str(t / "lock"),
        RLJUDGE_MEMINFO=str(t / "meminfo"),
        RLJUDGE_DEADLINE_S=str(deadline),
        CUDA_VISIBLE_DEVICES="",
    )
    e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(
        ["bash", "scripts/rl/judge_sidecar.sh", RUN],
        cwd=t,
        env=e,
        capture_output=True,
        text=True,
        timeout=deadline + 60,
    )
    return r.returncode, r.stdout + r.stderr


def table(t: Path) -> list[list[str]]:
    f = t / REFBASE / f"{RUN}_judge.tsv"
    return [ln.split("\t") for ln in f.read_text().splitlines()] if f.exists() else []


def rc_line(t: Path) -> str:
    f = t / REFBASE / f"{RUN}_judge.rc"
    return f.read_text().strip() if f.exists() else ""


def test_finished_run_is_judged_on_the_final_checkpoints(tmp_path):
    t = mk_repo(tmp_path)
    rc, out = sidecar(t)
    assert rc == 0, out[-800:]
    assert rc_line(t) == "DONE 65.00 pass10=62/64"
    rows = table(t)
    assert rows[0] == [
        "time",
        "kind",
        "iter",
        "ckpt",
        "gpu",
        "agg_pct",
        "batch_pct",
        "n_real",
        "status",
        "log_or_reason",
        "pass10",
    ]
    assert all(len(r) == 11 for r in rows)
    assert sorted(r[2] for r in rows if r[1] == "f" and r[8] == "OK") == ["1400", "1600"]
    assert len([r for r in rows if r[1] == "q" and r[8] == "SKIP"]) == 6
    judge = [r for r in rows if r[1] == "JUDGE"]
    assert len(judge) == 1 and judge[0][5] == "65.00" and judge[0][8] == "park_last3" and judge[0][10] == "62/64"
    assert [r[1] for r in rows if r[1] in ("best", "last")] == ["best", "last"]
    assert (t / REFBASE / f"{RUN}_judge_extras.json").is_file()
    assert not list((t / "lock").glob("gpu*.claim")), "claims must be released"


def test_every_judge_reads_the_training_record(tmp_path):
    """Each judge launch gets `--switches-from <run folder>/switches.json`; no switch crosses as environment."""
    t = mk_repo(tmp_path)
    rc, out = sidecar(t)
    assert rc == 0, out[-800:]
    outs = sorted((t / REFBASE).glob("*_judge_launch.out"))
    assert len(outs) >= 2
    assert all(f"[FAKE_JUDGE] switches_from={CKR}/{RUN}/switches.json" in f.read_text() for f in outs)


def test_a_run_with_checkpoints_and_no_record_fails(tmp_path):
    t = mk_repo(tmp_path)
    (t / CKR / RUN / "switches.json").unlink()
    rc, out = sidecar(t)
    assert rc == 1 and "no switches record" in rc_line(t), out[-800:]


def test_quick_judges_during_training_are_written_once_each(tmp_path):
    t = mk_repo(tmp_path, upto=400)

    def grow():
        for it in range(450, 1601, 50):
            time.sleep(0.15)
            add_ckpt(t, 32, it)

    th = threading.Thread(target=grow)
    th.start()
    rc, out = sidecar(t)
    th.join()
    assert rc == 0, out[-800:]
    q = [r[2] for r in table(t) if r[1] == "q"]
    assert sorted(set(q), key=int) == ["200", "400", "600", "800", "1000", "1200"] and len(q) == 6
    assert rc_line(t) == "DONE 65.00 pass10=62/64"


@pytest.mark.parametrize(
    "env, reason",
    [
        ({"FAKE_BAD_PARK": "1"}, "invalid batch"),
        ({"FAKE_PARK": "0(given)"}, "PARK-on marker missing"),
        ({"FAKE_WARMUP": "0"}, "warm-up 0 ≠ 1"),
        ({"FAKE_CODE": "c0de00000002"}, "code tree training c0de00000001 ≠ judge c0de00000002"),
        ({"FAKE_TOT": "32"}, "evaluations per clip 32 < 64"),
        ({"FAKE_EXTRA": "--arm rby1 --hand other"}, "EXTRA differs from training"),
        ({"FAKE_SCALE": "1.0"}, "DEXENGINE_FINGER_EFFORT_SCALE training 0.603 ≠ judge 1.0"),
        ({"FAKE_CT": "1"}, "judge log contact termination contact_term=1 ≠ judge standard 0"),
        ({"FAKE_ROSTER_SHA": "ab"}, "clip roster training sha256=0 ≠ judge sha256=ab"),
        ({"FAKE_FINGER_DROP": "left"}, "path=frozen_replay lines 1 ≠ sides 2"),
    ],
)
def test_invalid_judge_rows_fail_the_run(tmp_path, env, reason):
    t = mk_repo(tmp_path)
    rc, out = sidecar(t, **env)
    rows = table(t)
    assert rc == 1, out[-800:]
    assert any(r[1] == "f" and r[8] == "INVALID" and reason in r[9] for r in rows), [r[9] for r in rows if r[1] == "f"]
    assert rc_line(t).startswith("FAIL final judge f:")


def test_tree_change_is_allowed_only_explicitly_and_recorded(tmp_path):
    t = mk_repo(tmp_path)
    rc, out = sidecar(t, FAKE_CODE="c0de00000002", RLJUDGE_ALLOW_TREE_CHANGE=1)
    assert rc == 0, out[-800:]
    assert all("explicitly allowed" in r[9] for r in table(t) if r[1] == "f")


def test_one_hand_run_and_trajectory_record(tmp_path):
    t = mk_repo(tmp_path, side="right")
    rc, out = sidecar(t, side="right", RLJUDGE_TRAJ=1)
    assert rc == 0, out[-800:]
    assert rc_line(t) == "DONE 65.00 pass10=62/64"
    assert (t / REFBASE / f"{RUN}_jf1600_traj.npz").is_file() and (t / REFBASE / f"{RUN}_jf1400_traj.npz").is_file()
    t2 = mk_repo(tmp_path / "b", side="right")
    rc, out = sidecar(t2, side="right", RLJUDGE_TRAJ=1, FAKE_TRAJ_SKIP=1)
    assert rc == 1 and any("[TRAJ_REC] DONE marker missing" in r[9] for r in table(t2)), out[-800:]


def test_gate_closed_host_memory_is_incomplete_and_second_sidecar_refused(tmp_path):
    t = mk_repo(tmp_path, mem_gb=60)
    rc, out = sidecar(t, deadline=3)
    assert rc == 7 and rc_line(t).startswith("INCOMPLETE"), out[-800:]
    t2 = mk_repo(tmp_path / "b")
    (t2 / REFBASE / f"{RUN}_judge.pid").write_text(f"{os.getpid()}\n")
    rc, out = sidecar(t2)
    assert rc == 3, out[-800:]


@pytest.mark.parametrize(
    "env",
    [
        {"RLJUDGE_SHADOW": "1"},
        {"RLJUDGE_SPECIAL": "1"},
        {"RLJUDGE_ITERS": "1600"},
        {"RLJUDGE_RECOUNT": "1"},
        {"RLJUDGE_TRAJ": "2"},
        {"RLJUDGE_LAUNCH_ID": "x"},
    ],
)
def test_removed_or_inconsistent_inputs_exit_2(tmp_path, env):
    t = mk_repo(tmp_path)
    rc, out = sidecar(t, **env)
    assert rc == 2, out[-400:]
    assert rc_line(t) == ""


def test_event_wake_does_not_wait_out_the_poll(tmp_path):
    """POLL 20 s, NAP 0.2 s: the final checkpoints arrive 1 s in, the judges end within the next seconds — every wait
    ends at its event, so the run is judged long before one POLL (a plain `sleep POLL` takes >= 2 POLLs here)."""
    t = mk_repo(tmp_path, upto=400)

    def finish_training():
        time.sleep(1.0)
        for it in range(450, 1601, 50):
            add_ckpt(t, 32, it)

    th = threading.Thread(target=finish_training)
    t0 = time.monotonic()
    th.start()
    rc, out = sidecar(t, RLJUDGE_POLL_S="20", RLJUDGE_NAP_S="0.2", RLJUDGE_QUICK_EVERY="0")
    th.join()
    took = time.monotonic() - t0
    assert rc == 0 and rc_line(t) == "DONE 65.00 pass10=62/64", out[-800:]
    assert took < 15, f"{took:.1f} s: a wait slept out the poll"


def test_a_late_saved_line_wakes_the_loop(tmp_path):
    """The [CKPT_SAVED] lines arrive 1.5 s after the checkpoint files (more than the one extra NAP): wake_sig counts
    those lines, so their arrival wakes the loop instead of the next POLL (20 s here)."""
    t = mk_repo(tmp_path, upto=400)

    def finish_training():
        time.sleep(1.0)
        for it in range(450, 1601, 50):
            (t / CKR / RUN / "checkpoints" / f"agent_{it * 32}.pt").touch()
        time.sleep(1.5)
        with open(t / REFBASE / f"{RUN}_train.log", "a") as fh:
            fh.writelines(f"[CKPT_SAVED] agent_{it * 32}.pt iter={it}\n" for it in range(450, 1601, 50))

    th = threading.Thread(target=finish_training)
    t0 = time.monotonic()
    th.start()
    rc, out = sidecar(t, RLJUDGE_POLL_S="20", RLJUDGE_NAP_S="0.2", RLJUDGE_QUICK_EVERY="0")
    th.join()
    took = time.monotonic() - t0
    assert rc == 0 and rc_line(t) == "DONE 65.00 pass10=62/64", out[-800:]
    assert took < 15, f"{took:.1f} s: the late [CKPT_SAVED] lines waited out the poll"
