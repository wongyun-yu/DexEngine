"""Every file the augment cache fingerprint reads exists in the tree (record.augment.stage: record CODE_FILES +
AUG_CODE_FILES + augment.bake.bake_code_files). A moved file made every aug-round bundle die with FileNotFoundError at
the fingerprint (bake_code_files named scripts/placement/stage_*.py after they moved to dexengine/placement/bake)."""

from __future__ import annotations

from dexengine.core.repo import REPO_ROOT
from dexengine.record import cache as C
from dexengine.record.augment import bake as AB
from dexengine.record.augment import stage as AS


def test_augment_fingerprint_files_exist():
    files = C.CODE_FILES + C.RENDER_CODE_FILES + AS.AUG_CODE_FILES + AB.bake_code_files()
    missing = [f for f in files if not (REPO_ROOT / f).is_file()]
    assert not missing, missing
