"""The collision-preset record of a run (CPU): a judge or render of a checkpoint whose run folder has no record is
refused and names the file — a missing record never reads as "trained with none"; the first training launch alone may
start without it."""

from __future__ import annotations

from pathlib import Path

import pytest

from dexengine.rl.launch import records as R


def test_a_missing_record_is_refused_and_named(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        R.check_coll_preset(Path("."), "run", "canon_r2r3off", "judge", render_rundir=str(tmp_path))
    assert e.value.code == 7
    out = capsys.readouterr().out
    assert "no collision-preset record" in out and str(tmp_path / "coll_preset.txt") in out


def test_the_first_training_launch_may_start_without_it(tmp_path):
    R.check_coll_preset(Path("."), "run", "canon_r2r3off", "train", fresh=True, render_rundir=str(tmp_path))


def test_a_record_is_compared(tmp_path):
    (tmp_path / "coll_preset.txt").write_text("canon_r2r3off\n# note\n", encoding="utf-8")
    R.check_coll_preset(Path("."), "run", "canon_r2r3off", "judge", render_rundir=str(tmp_path))
    with pytest.raises(SystemExit):
        R.check_coll_preset(Path("."), "run", "other", "judge", render_rundir=str(tmp_path))
