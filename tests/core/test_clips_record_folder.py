"""paths.clips_record returns a write place whose folder exists, so a record written into a run output folder that a
fresh checkout does not have (run_path layer exp_shared = docs/experiments) succeeds — the articulation surface ledger
(dexengine/data/artic/surface.py) died there with FileNotFoundError."""

from pathlib import Path

import pytest

from dexengine.core import paths as P


@pytest.mark.parametrize("scoped", [False, True])
def test_record_in_missing_folder_is_writable(tmp_path: Path, scoped: bool):
    target = tmp_path / "fresh" / "docs" / "experiments" / "artic_surface_ledger.json"
    assert not target.parent.exists()
    rec = P.clips_record(target, scoped)
    assert rec.name == ("artic_surface_ledger.clips.json" if scoped else "artic_surface_ledger.json")
    rec.write_text("{}\n", encoding="utf-8")
    assert rec.read_text(encoding="utf-8") == "{}\n"


def test_run_path_layer_record_in_fresh_tree(tmp_path: Path):
    """The surface ledger's own path: run_path('exp_shared') under an empty output root, then clips_record."""
    led = P.run_path("exp_shared", root=tmp_path) / "artic_surface_ledger.json"
    assert not led.parent.exists()
    P.clips_record(led, True).write_text("{}\n", encoding="utf-8")
    assert (led.parent / "artic_surface_ledger.clips.json").is_file()
