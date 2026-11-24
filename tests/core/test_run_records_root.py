"""Run records (the exp_* run layers) live under `records/` of the run root, never under the repository's `docs/`,
and a checkout used as its own data root does not show them as untracked files (.gitignore). Two plain data roots
that share one code checkout keep separate records (roots.run_root_of: a folder data root is its own run root).
Known limit (docs/configuration.md, Data layout): two *git checkouts* used as data roots with one code checkout
share the code checkout's records."""

import shutil
import subprocess
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.core.layout import runs as L
from dexengine.core.locations import roots as RO

REPO = Path(__file__).resolve().parents[2]
EXP = [k for k in L.RUN_LAYERS if k.startswith("exp_")] + [k for k in L.RUN_SHARED if k.startswith("exp_")]


def test_exp_layers_live_under_records():
    assert EXP, "no exp_* run layers found"
    for k in EXP:
        rel = L.render_run(k, "wuji") if k in L.RUN_LAYERS else L.render_run(k)
        assert rel.split("/")[0] == "records", f"{k} → {rel} (run records must not land in the repository's docs/)"


def test_two_folder_roots_one_checkout_keep_records_apart(tmp_path: Path):
    a, b = tmp_path / "root_a", tmp_path / "root_b"
    a.mkdir()
    b.mkdir()
    ra, rb = RO.run_root_of(a, REPO), RO.run_root_of(b, REPO)
    la, lb = P.run_path("exp_shared", root=ra), P.run_path("exp_shared", root=rb)
    assert la != lb
    assert la.is_relative_to(a) and lb.is_relative_to(b)
    assert not la.is_relative_to(REPO) and not lb.is_relative_to(REPO)


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs a git checkout")
def test_records_are_ignored_when_the_checkout_is_the_data_root():
    rel = L.render_run("exp_shared") + "/artic_surface_ledger.json"
    r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", rel], check=False)
    assert r.returncode == 0, f"{rel} is not ignored — a run in this checkout leaves untracked files"
