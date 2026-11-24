"""stage2_ground.sh --fold: the fold step at the end of a chunk (function ground_fold of scripts/stage2/_ground_lib.sh,
run here on its own with a stand-in python3) and the options it refuses (dry runs, nothing written)."""

import os
import re
import subprocess
from pathlib import Path

import pytest

from dexengine.core.child_output import child_failure

REPO = Path(__file__).resolve().parents[2]
STAGE2 = REPO / "scripts/stage2_ground.sh"
LIB = REPO / "scripts/stage2/_ground_lib.sh"  # ground_fold lives here


def _fold(tmp_path: Path, fold: str, rc: int) -> subprocess.CompletedProcess:
    body = re.search(r"^ground_fold\(\) \{\n.*?^\}\n", LIB.read_text(), re.S | re.M).group(0)
    stub = tmp_path / "bin" / "python3"
    stub.parent.mkdir()
    stub.write_text(f'#!/bin/bash\necho "$DEXENGINE_HAND $*" > "{tmp_path}/called"\necho "last line"\nexit {rc}\n')
    stub.chmod(0o755)
    out = tmp_path / "chunk"
    out.mkdir()
    setup = f"FOLD={fold}; OUT={out}; DATASET=hocap; HAND=sharpa_wave; export DEXENGINE_HAND=$HAND\n"
    script = f"{body}{setup}ground_fold\n"
    env = dict(os.environ, PATH=f"{stub.parent}:{os.environ['PATH']}")
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)


def test_fold_off_does_and_prints_nothing(tmp_path):
    r = _fold(tmp_path, "0", 0)
    assert (r.returncode, r.stdout, r.stderr) == (0, "", "") and not (tmp_path / "called").exists()


def test_fold_runs_status_fold_for_the_dataset_and_marks_it(tmp_path):
    r = _fold(tmp_path, "1", 0)
    assert r.returncode == 0 and r.stdout.startswith("[FOLD] hocap folded for hand sharpa_wave"), r.stdout
    assert (tmp_path / "called").read_text().split() == [
        "sharpa_wave",
        "dexengine/placement/status_fold.py",
        "--datasets",
        "hocap",
    ]


def test_a_failed_fold_stops_with_rc_8_and_the_command(tmp_path):
    r = _fold(tmp_path, "1", 2)
    assert r.returncode == 8 and "fold rc=2 (last line)" in r.stderr
    assert "DEXENGINE_HAND=sharpa_wave python3 dexengine/placement/status_fold.py --datasets hocap" in r.stderr


def _isaac_python() -> bool:
    """Does this environment find the Isaac Lab python (owner scripts/_isaac_py.sh, isaac_py)?"""
    r = subprocess.run(
        ["bash", "-c", '. scripts/_isaac_py.sh && p=$(isaac_py) && [ -x "$p" ]'], cwd=REPO, capture_output=True
    )
    return r.returncode == 0


@pytest.mark.skipif(
    not _isaac_python(),
    reason="needs Isaac Lab (scripts/_isaac_py.sh isaac_py): stage2_ground.sh resolves the Isaac python before it "
    "reads --fold, so without it the run stops earlier and the refusal is not reached",
)
@pytest.mark.parametrize("extra", [["--variant", "_x"], ["--allow-tag-mismatch"], ["--f0-only"], ["--judge-only"]])
def test_fold_refuses_runs_it_cannot_take(tmp_path, extra):
    """Passes on the refusal line only (rc 2 alone is also the rc of a run that stops before the option check)."""
    args = ["--arm", "rby1", "--hand", "sharpa_wave", "--dataset", "hocap", "--emb", "bimanual"]
    r = subprocess.run(
        ["bash", str(STAGE2), *args, "--data-root", str(tmp_path), "--fold", *extra, "--dry-run"],
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 2 and f"--fold cannot go with {extra[0]}" in r.stderr, child_failure(r.stderr, 400)
