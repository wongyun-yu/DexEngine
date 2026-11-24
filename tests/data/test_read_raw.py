"""The raw re-read driver's gates on a scratch data root (dry runs only; nothing is read or written) and the per-row
logs of its decompose and contact children (a stub interpreter in a scratch folder)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexengine.data.read_raw import stage as S

REPO = Path(__file__).resolve().parents[2]


def _root(tmp_path: Path) -> Path:
    ds = tmp_path / "data/hot3d/raw/dataset"
    for name, flag, pose in (("P0001_a", True, b"x\n"), ("P0002_b", False, b"")):
        (ds / name).mkdir(parents=True)
        (ds / name / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": flag}))
        (ds / name / "mano_hand_pose_trajectory.jsonl").write_bytes(pose)
    return tmp_path


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(
        os.environ,
        DEXENGINE_MAIN_ROOT=str(root),
        RETARGET_PY=sys.executable,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(REPO),
    )
    return subprocess.run(
        [sys.executable, "-m", "dexengine.data.read_raw", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_dry_read_counts_recordings(tmp_path):
    r = _run(_root(tmp_path), "hot3d", "read", "--dry")
    assert r.returncode == 0, r.stderr
    assert "read recordings 1 · already read 0 · to read 1" in r.stdout


def test_dry_read_refuses_old_generation(tmp_path):
    root = _root(tmp_path)
    (root / "data/hot3d/mano/bimanual/old_clip").mkdir(parents=True)
    r = _run(root, "hot3d", "read", "--dry")
    assert r.returncode == 3 and "✗(dry)" in r.stdout


def test_unknown_dataset_and_missing_hand(tmp_path):
    root = _root(tmp_path)
    assert _run(root, "show3d", "read", "--dry").returncode != 0
    r = _run(root, "hot3d", "intake")
    assert r.returncode == 2 and "first hand" in r.stderr


def _stub_python(tmp_path: Path, rc: int) -> Path:
    """An interpreter stand-in that prints its arguments to stderr and exits with `rc`."""
    f = tmp_path / "stub_py"
    f.write_text(f'#!/bin/sh\necho "stub stderr: $*" >&2\necho "stub stdout"\nexit {rc}\n')
    f.chmod(0o755)
    return f


def test_decompose_contact_failure_names_a_log_with_the_child_output(tmp_path):
    """A failed decompose or contact row leaves the child's stdout and stderr in a per-row log, and its FAIL line ends
    with that log's path."""
    lg, pend_f = tmp_path / "_logs", tmp_path / "pending.tsv"
    lg.mkdir()
    pend_f.write_text("bimanual\tP0001 two words__0\n")
    run = SimpleNamespace(pys=str(_stub_python(tmp_path, 1)), st=tmp_path / "stage", ds="hot3d", tag="[T]")
    with pytest.raises(S.Refused):
        S._decompose_contact(run, ["bimanual\tP0001 two words__0"], lg, pend_f)
    for name, mod in (("decomp", "dexengine.retarget.decompose"), ("contact", "dexengine.retarget.contact")):
        (line,) = (lg / f"{name}_fail.txt").read_text().splitlines()
        log = lg / f"{name}_bimanual_P0001 two words__0.txt"
        assert line == f"FAIL bimanual P0001 two words__0 log={log}"
        text = log.read_text()
        assert f"stub stderr: -m {mod} --dataset-dir" in text and "--task P0001 two words__0" in text, text
        assert "stub stdout" in text
    assert pend_f.read_text(), "a failed run keeps the pending list"


def test_decompose_contact_success_clears_pending(tmp_path):
    lg, pend_f = tmp_path / "_logs", tmp_path / "pending.tsv"
    lg.mkdir()
    pend_f.write_text("right\tP0002__1\n")
    run = SimpleNamespace(pys=str(_stub_python(tmp_path, 0)), st=tmp_path / "stage", ds="hot3d", tag="[T]")
    S._decompose_contact(run, ["right\tP0002__1"], lg, pend_f)
    assert pend_f.read_text() == ""
    assert (lg / "decomp_fail.txt").read_text() == "" and (lg / "contact_fail.txt").read_text() == ""
    assert "stub stderr" in (lg / "decomp_right_P0002__1.txt").read_text()


def test_object_assets_link_resolves_on_a_fresh_data_root(tmp_path, monkeypatch):
    """A data root with no object-asset layer yet: the link must resolve, so a reader's mkdir under it works."""
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(tmp_path))
    st = tmp_path / "stage"
    (st / "processed/hot3d/assets").mkdir(parents=True)
    link = S.link_object_assets(st, "hot3d")
    assert link.is_symlink() and link.resolve().is_dir()
    (link / "obj_a").mkdir(parents=True, exist_ok=True)
    assert (link.resolve() / "obj_a").is_dir()
