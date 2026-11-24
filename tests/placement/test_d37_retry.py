"""Rule D37 — the stage-2 budget retry (owner dexengine.placement.judge.retry · round scripts/stage2/_ground_lib.sh).

A  classify: le2x · assembly are retried; gt2x · big_budget · unknown are not
B  plan on a chunk: targets present (marker n=2 · resume rows · retry tag) · none (n=0) · a retry chunk whose clip is
   still UNPROVEN at 2048 (kind retried · n=0 — terminal)
C  retry tag rule (prefix + r · the regression retry prefix of TEST_TAG_PREFIXES) · resume table round trip and dies
D  judge: the start cell of a resumed clip · a changed candidate count dies · the flush cut counts from the start cell
E  stage-2 round: n=0 launches nothing · n>0 relaunches stage 2 with the same inputs minus the chunk identity, plus
   --retry-budget --exhaust-resume · its rc is the round's rc · the retry inputs come as a pair and reach run.env
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexengine.core import datasets as RR
from dexengine.placement.judge import retry as R
from tests.imports import own_module

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "scripts/stage2/_ground_lib.sh"
TAG = "G1rfourbr_taco_71"
REC = R.default()


def _chunk(d: Path, rows, budget_lines, anchor_checks=None, retry_budget=""):
    """A chunk folder as stage 2 leaves it: status ledger · judge.log · p0/noconfirm · run.env."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "status_ledger.tsv").write_text("clip\tstatus\n" + "".join(f"{c}\t{s}\n" for c, s in rows))
    (d / "judge.log").write_text(
        "".join(f"[JUDGE_BUDGET] {c} cell:r0 checked {n}/{m} · adopted 0/3 · elapsed 1s\n" for c, n, m in budget_lines)
    )
    (d / "p0").mkdir(exist_ok=True)
    (d / "p0" / "noconfirm_combo.tsv").write_text(
        "".join(f"{c}\tanchor fail\tchecks {k} · confirm none\n" for c, k in (anchor_checks or {}).items())
    )
    (d / "run.env").write_text(f"tag={TAG}\n" + (f"retry_budget={retry_budget}\n" if retry_budget else ""))
    return d


def test_classify_kinds():
    assert R.classify(1024, 2048, 0, None, REC) == "le2x"
    assert R.classify(1024, 3252, 5, None, REC) == "assembly"
    assert R.classify(1024, 3252, 0, None, REC) == "gt2x"
    assert R.classify(None, 3252, 5, None, REC) == "unknown"
    assert R.classify(1024, 2048, 0, 8192, REC) == "big_budget"
    assert set(R.RETRY_KINDS) == {"le2x", "assembly"}
    assert R.budget(REC) == RR.EXHAUST_RETRY["from"] * RR.EXHAUST_RETRY["factor"] == 2048


def test_plan_writes_targets_table_and_marker(tmp_path, capsys):
    ck = _chunk(
        tmp_path / TAG,
        [("a", "UNPROVEN"), ("b", "UNPROVEN"), ("c", "UNPROVEN"), ("d", "ADOPTED")],
        [("a", 1024, 3252), ("b", 1024, 1500), ("c", 1024, 3000)],
        {"a": 3},
    )
    assert R.plan(ck, TAG, tmp_path / "d37") == 2
    out = capsys.readouterr().out
    assert f"{RR.RETRY_RUN_MARKER} r1 exhaust_max=2048 n=2 " in out
    assert "{'assembly': 1, 'gt2x': 1, 'le2x': 1}" in out
    assert (tmp_path / "d37/targets.txt").read_text() == "a\nb\n"
    assert R.read_resume(tmp_path / "d37/resume.tsv") == {"a": (1024, 3252), "b": (1024, 1500)}
    assert (tmp_path / "d37/tag.txt").read_text().strip() == "G1rfourbrr_taco_71"
    assert R.load(tmp_path / "d37" / R.FILE) == REC


def test_plan_without_targets_prints_n0(tmp_path, capsys):
    ck = _chunk(tmp_path / TAG, [("d", "ADOPTED"), ("e", "EXHAUSTED")], [])
    assert R.plan(ck, TAG, tmp_path / "d37") == 0
    assert f"{RR.RETRY_RUN_MARKER} r1 exhaust_max=2048 n=0 " in capsys.readouterr().out
    assert (tmp_path / "d37/targets.txt").read_text() == ""


def test_still_unproven_at_2048_is_terminal(tmp_path, capsys):
    """The retry chunk's own round: the clip is UNPROVEN again at 2048 → kind retried, nothing is launched."""
    ck = _chunk(tmp_path / "G1rfourbrr_taco_71", [("a", "UNPROVEN")], [("a", 2048, 3252)], {"a": 3}, "2048")
    assert R.plan(ck, "G1rfourbrr_taco_71", tmp_path / "d37") == 0
    out = capsys.readouterr().out
    assert "n=0 " in out and "{'retried': 1}" in out


def test_retry_tag_and_resume_table(tmp_path):
    assert R.retry_tag(TAG) == "G1rfourbrr_taco_71"
    assert R.retry_tag("G1regr_taco_03").startswith(RR.TEST_TAG_PREFIXES["regress_retry"])
    with pytest.raises(SystemExit, match="no prefix"):
        R.retry_tag("notag")
    f = tmp_path / "t.tsv"
    assert R.write_resume(f, [("a", 3, 9)]) == 1
    assert R.read_resume(f) == {"a": (3, 9)}
    with pytest.raises(SystemExit, match="range"):
        R.write_resume(f, [("a", 10, 9)])
    with pytest.raises(SystemExit, match="twice"):
        R.write_resume(f, [("a", 1, 9), ("a", 2, 9)])


def test_judge_start_cell_and_flush_cut(capsys):
    search = own_module("dexengine.placement.judge.search")
    exhaust = own_module("dexengine.placement.judge.exhaust")
    J = SimpleNamespace(_EXH_RESUME={"a": (1024, 3252)}, a=SimpleNamespace(exhaust_max=2048))
    assert search.SearchMixin.exh_resume_start(J, "a", 3252) == 1024
    assert "[EXH_RESUME] a from cell 1024/3252" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="input changed"):
        search.SearchMixin.exh_resume_start(J, "a", 3000)
    assert search.SearchMixin.exh_resume_start(SimpleNamespace(_EXH_RESUME=None), "a", 3252) == 0
    # cut kind of a resumed search: n_ex counts list positions from the start cell
    ex = exhaust._Exhaustion.__new__(exhaust._Exhaustion)
    ex.J = SimpleNamespace(a=SimpleNamespace(exhaust_max=2048))
    ex.cands, ex._start, ex._cl = list(range(3252)), 1024, list(range(1024, 2048))
    ex.n_found, ex._ncex, ex._cut_wall, ex.n_pruned = 0, 3, False, 0
    ex.n_ex = 2048
    assert ex._budget_cut() == "cell"  # the whole retry list was seen; the ranked list was cut at 2048
    ex.n_ex = 1524
    assert ex._budget_cut() == "flush"  # stopped inside the retry list


def _fn(name: str) -> str:
    return re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", LIB.read_text(), re.S | re.M).group(0)


def _round(tmp_path: Path, n_targets: int, rc: int = 0) -> tuple[subprocess.CompletedProcess, Path]:
    rows = [(f"c{i}", "UNPROVEN") for i in range(n_targets)] + [("z", "ADOPTED")]
    ck = _chunk(tmp_path / "chunk", rows, [(f"c{i}", 1024, 1500) for i in range(n_targets)])
    fake = tmp_path / "repo/scripts/stage2_ground.sh"
    fake.parent.mkdir(parents=True)
    fake.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{tmp_path}/argv"\nexit {rc}\n')
    argv = "--arm rby1 --hand sharpa_wave --clips L --tag T --out O --fold --judge-extra 'A B' --skip-judge --gpu 0"
    script = (
        _fn("ground_d37_round") + f"OUT={ck}; TAG={TAG}; _S2_REPO={tmp_path}/repo; _S2_ARGV=({argv})\n"
        "ground_d37_round; echo rc=$?\n"
    )
    env = dict(os.environ, PYTHONPATH=f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120), ck


def test_round_without_targets_launches_nothing(tmp_path):
    r, ck = _round(tmp_path, 0)
    assert "rc=0" in r.stdout and "n=0 " in r.stdout, r.stderr
    assert not (tmp_path / "argv").exists()
    assert "[RETRY_BUDGET] r1 exhaust_max=2048 n=0" in (ck / "summary.txt").read_text()


def test_round_relaunches_stage2_for_the_targets(tmp_path):
    r, ck = _round(tmp_path, 2)
    assert "rc=0" in r.stdout and "n=2 " in r.stdout, r.stderr
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[:6] == ["--arm", "rby1", "--hand", "sharpa_wave", "--judge-extra", "A B"]
    assert "--fold" not in argv and "--skip-judge" not in argv and "O" not in argv and "L" not in argv
    assert argv[argv.index("--clips") + 1] == f"{ck}/d37/targets.txt"
    assert argv[argv.index("--tag") + 1] == "G1rfourbrr_taco_71"
    assert argv[argv.index("--retry-budget") + 1] == f"{ck}/d37/retry.json"
    assert argv[argv.index("--exhaust-resume") + 1] == f"{ck}/d37/resume.tsv"


def test_round_fails_with_the_retry_chunk(tmp_path):
    r, _ = _round(tmp_path, 1, rc=3)
    assert "rc=3" in r.stdout and "retry chunk G1rfourbrr_taco_71 rc=3" in r.stderr


def _inputs(tmp_path: Path, rec: str, table: str) -> subprocess.CompletedProcess:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / "run.env").write_text("tag=x\n")
    script = (
        _fn("ground_retry_inputs")
        + _fn("ground_retry_run_env")
        + f"OUT={out}; RETRY_REC={rec}; EXH_RESUME={table}; JUDGE_EXTRA='--x 1'; _KEEP_RUN_ENV=0\n"
        'ground_retry_inputs || exit $?; ground_retry_run_env; echo "JX=$JUDGE_EXTRA"\n'
    )
    env = dict(os.environ, PYTHONPATH=f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)


def test_retry_inputs_pair_and_run_env(tmp_path):
    rec = tmp_path / R.FILE
    R.ensure(rec)
    table = tmp_path / "resume.tsv"
    R.write_resume(table, [("a", 1024, 3252)])
    r = _inputs(tmp_path, str(rec), "")
    assert r.returncode == 2 and "go together" in r.stderr
    r = _inputs(tmp_path, str(rec), str(table))
    assert r.returncode == 0, r.stderr
    assert f"{RR.RETRY_RUN_MARKER} --exhaust-max 2048" in r.stdout
    assert f"JX=--x 1 --exhaust-max 2048 --exhaust-resume {table}" in r.stdout
    assert (tmp_path / "out/run.env").read_text().endswith("retry_budget=2048\nexh_resume=1\n")
