"""INTAKE derivations for every driver (dexengine.data.read_raw.derive) and the hoi driver's manifest: a missing
input or an unwritten output stops the run with the command that makes it; written outputs pass."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.data import intake as IN
from dexengine.data import readers as R
from dexengine.data.handoff.checks import Report
from dexengine.data.read_raw import derive as D
from dexengine.data.read_raw import hoi

REPO = Path(__file__).resolve().parents[2]


class Stop(Exception):
    pass


def _fail(msg: str):
    raise Stop(msg)


def _cli(*args: str, root: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, DEXENGINE_MAIN_ROOT=str(root), RETARGET_PY=sys.executable, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def _writer(tmp_path: Path, error_only: bool) -> str:
    """A stand-in derivation that writes both base-layer INTAKE files (argv: the INTAKE folder)."""
    script = tmp_path / "write_audits.py"
    cls = "error" if error_only else "table"
    script.write_text(
        "import json, pathlib, sys\n"
        "d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)\n"
        f"(d / 'support_audit.json').write_text(json.dumps({{'clips': {{'c1': {{'cls': '{cls}'}}}}}}))\n"
        "(d / 'segment_audit.json').write_text(json.dumps({'clips': {'c1': {'long': False}}}))\n"
    )
    return f"{script} {P.layer_path('intake', 'hocap')}"


def _run(rows, tmp_path: Path) -> int:
    return D.run(rows, "hocap", tmp_path / "out", tmp_path / "logs", sys.executable, _fail)


def test_derive_cli_dies_on_an_empty_mano_layer(data_root):
    r = _cli("-m", "dexengine.data.read_raw", "hocap", "derive", root=data_root)
    assert r.returncode == 2 and "canonical MANO layer has no clips" in r.stderr, r.stderr


def test_a_row_without_steps_dies(tmp_path):
    with pytest.raises(Stop, match="no derive steps"):
        _run(R.spec("h2o").derive, tmp_path)


def test_a_step_that_writes_nothing_dies(tmp_path):
    with pytest.raises(Stop, match="support_audit.json was not written"):
        _run((("support_audit", "-c pass", ""),), tmp_path)


def test_a_failing_step_dies_with_its_last_log_line(tmp_path):
    script = tmp_path / "boom.py"
    script.write_text("import sys\nprint('boom')\nsys.exit(3)\n")
    with pytest.raises(Stop, match=r"derivation support_audit rc=3 \(boom\)"):
        _run((("support_audit", str(script), ""),), tmp_path)


def test_an_audit_with_every_clip_an_error_dies(tmp_path):
    with pytest.raises(Stop, match="every clip"):
        _run((("audits", _writer(tmp_path, error_only=True), ""),), tmp_path)


def test_an_old_audit_left_in_place_dies(tmp_path):
    rows = (("audits", _writer(tmp_path, error_only=False), ""),)
    assert _run(rows, tmp_path) == 1
    for name in D.BASE_INPUTS:  # as if an earlier run wrote them and this run's step skipped them
        os.utime(P.layer_path("intake", "hocap") / name, (1, 1))
    with pytest.raises(Stop, match="support_audit.json was not rewritten"):
        _run((("support_only", "-c pass", ""),), tmp_path)


def test_written_audits_pass(tmp_path):
    assert _run((("audits", _writer(tmp_path, error_only=False), ""),), tmp_path) == 1
    assert D.proof("hocap", 0) == ""


def test_every_derive_row_makes_support_before_segment():
    for ds, s in R.READERS.items():
        names = [r[0] for r in s.derive]
        if names:
            assert names.index("support_audit") < names.index("segment_audit"), ds  # cuts.py reads the support audit


def test_base_refusal_names_the_derive_command(data_root):
    r = _cli("dexengine/data/base_adapter.py", "--dataset", "hocap", "--hand", "sharpa_wave", root=data_root)
    assert r.returncode == 2 and "`python -m dexengine.data.read_raw hocap derive`" in r.stderr, r.stderr


def _stage(tmp_path: Path) -> Path:
    st = tmp_path / "stage"
    for rel in ("hocap/mano/bimanual/t1__a/0/trajectory_keypoints.npz", "hocap/sharpa_wave/bimanual/t1__a/0/q.npz"):
        f = st / "processed" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(rel.encode())
    return st


def test_hoi_run_with_a_failure_writes_no_manifest(tmp_path):
    st = _stage(tmp_path)
    with pytest.raises(SystemExit, match="manifest not written"):
        hoi.finish(st, "sharpa_wave", [("hocap", "t1")], 1, tmp_path / "log.jsonl")
    rep = Report()
    assert IN.verify_manifest(st / "processed", "hocap", rep) == {"present": False}
    assert any("MANIFEST.sha256 missing" in x for x in rep.ds_reject["hocap"])


def test_hoi_run_without_failure_writes_the_manifest_the_intake_accepts(tmp_path):
    st = _stage(tmp_path)
    assert hoi.finish(st, "sharpa_wave", [("hocap", "t1")], 0, tmp_path / "log.jsonl") == 0
    rep = Report()
    got = IN.verify_manifest(st / "processed", "hocap", rep)
    assert got == {"present": True, "listed": 2, "bad": 0, "missing": 0, "unlisted": 0} and not rep.ds_reject


def test_desk_recovery_runs_between_the_support_and_segment_audits():
    """Datasets whose support audit carries a desk recovery (hot3d, hrdexdb, oakink2) run it before the segment
    audit, which reads the recovered desk."""
    for ds, step in (
        ("hot3d", "hot3d_desk_recover"),
        ("hrdexdb", "hrdexdb_desk_recover"),
        ("oakink2", "oakink2_desk_recover"),
    ):
        names = [r[0] for r in R.spec(ds).derive]
        assert names == ["support_audit", step, "segment_audit"], (ds, names)
