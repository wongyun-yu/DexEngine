"""Static checks of the Isaac-side visual desk (the spawner needs Kit, so it is compiled and read, not run)."""

from __future__ import annotations

import ast
import py_compile
import subprocess
from pathlib import Path

from tests.core.references_paths import SITE

REPO = Path(__file__).resolve().parents[2]
SPAWN_FILES = (
    "dexengine/rl/tasks/scene/desk_visual.py",
    "dexengine/rl/tasks/cfg/scene.py",
)


def test_spawner_files_compile(tmp_path):
    for rel in SPAWN_FILES:
        py_compile.compile(str(REPO / rel), cfile=str(tmp_path / (Path(rel).stem + ".pyc")), doraise=True)


def test_the_scene_desk_comes_from_the_one_chooser():
    tree = ast.parse((REPO / "dexengine/rl/tasks/cfg/scene.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TacoEvalSceneCfg")
    desk = [n for n in cls.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "desk"]
    assert len(desk) == 1 and ast.unparse(desk[0].value) == "_desk_cfg(DESK_OFFSET_X)"


def test_the_willow_desk_is_stated_once():
    """The Willow desk's location and bytes live in look_assets only (WILLOW · WILLOW_SHA256 — NVIDIA's S3 into the
    local mirror); no tree carries a copy and no code names another desk file."""
    out = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "grep",
            "-n",
            "-i",
            "-e",
            "desk_willow",
            "-e",
            "Tables/Willow",
            "--",
            "dexengine",
            "scripts",
            "experiments",
            SITE,
            "configs",
            "assets",
        ],
        capture_output=True,
        text=True,
    ).stdout
    hits = [ln for ln in out.splitlines() if not ln.startswith("dexengine/record/looks/look_assets.py:")]
    assert hits == [], hits
    from dexengine.record.looks import look_assets as LA

    assert LA.WILLOW in LA.WILLOW_SHA256 and len(LA.WILLOW_SHA256) == 8
    assert not (REPO / "assets/scene/desk_willow").exists()
