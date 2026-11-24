"""The intake run state (--dest, --archive) lives in intake_ingest; the CLI sets it there and both modules read it."""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from dexengine.data import intake as IN
from dexengine.data import intake_ingest as II


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(II, "DEST", None)
    monkeypatch.setattr(II, "ARCHIVE", None)
    return II


def test_dest_root_follows_the_owner_state(state, tmp_path):
    assert state.dest_root() == state.P.main_root()
    state.DEST = tmp_path
    assert state.dest_root() == tmp_path
    assert state._dest("intake", "taco") == tmp_path / "data" / "taco" / "INTAKE"


def test_verify_manifest_reads_the_archive_state(state, tmp_path):
    rep = SimpleNamespace(warnings=[], ds_reject=defaultdict(list))
    assert IN.verify_manifest(tmp_path, "taco", rep) == {"present": False}
    assert rep.ds_reject["taco"] and not rep.warnings
    state.ARCHIVE = {"sha256": "ab" * 32, "bytes": 1}
    rep = SimpleNamespace(warnings=[], ds_reject=defaultdict(list))
    assert IN.verify_manifest(tmp_path, "taco", rep) == {"present": False, "archive": state.ARCHIVE}
    assert rep.warnings and not rep.ds_reject["taco"]


def test_cli_sets_the_owner_state(state, tmp_path, monkeypatch):
    monkeypatch.setattr(IN.sys, "argv", ["intake.py", str(tmp_path), "--dest", str(tmp_path / "d")])
    IN.main()  # no dataset directories: it reports and returns
    assert (tmp_path / "d").resolve() == II.DEST
