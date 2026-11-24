"""The stage-2 render fingerprint covers the files that decide what is drawn: the session layout (record.render.clip),
the episode/look plan (record.episodes) and the Kit arguments (record.session.kit). They were missing, so a change
there reused renders made by the old code."""

from __future__ import annotations

import shutil

import pytest

from dexengine.core.repo import REPO_ROOT
from dexengine.record import cache as C

DRAW_DECIDERS = (
    "dexengine/record/render/clip.py",
    "dexengine/record/render/clip_all.py",
    "dexengine/record/render/episode_frames.py",
    "dexengine/record/episodes.py",
    "dexengine/record/session/kit.py",
)


@pytest.mark.parametrize("rel", DRAW_DECIDERS)
def test_a_change_in_a_draw_decider_misses_the_render_cache(tmp_path, rel):
    for f in C.RENDER_CODE_FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / f, tmp_path / f)
    before = C.code_fingerprint(tmp_path, C.RENDER_CODE_FILES)
    with open(tmp_path / rel, "a") as fh:
        fh.write("\n# edited\n")
    assert C.code_fingerprint(tmp_path, C.RENDER_CODE_FILES) != before


AUGVIS = sorted(p.name for p in (REPO_ROOT / "dexengine/record/looks/augvis").glob("*.py"))


@pytest.mark.parametrize("name", AUGVIS)
def test_every_augvis_module_is_in_the_render_fingerprint(name):
    assert f"dexengine/record/looks/augvis/{name}" in C.RENDER_CODE_FILES


def test_marker_tag_covers_every_marker_module(monkeypatch, tmp_path):
    from dexengine.record.render import markers as PM

    names = {p.name for p in PM.code_files()}
    assert names == {p.name for p in (REPO_ROOT / "dexengine/record/render").glob("markers*.py")} and len(names) > 1
    ref = tmp_path / "e5_place_ref.py"
    ref.write_text("# ref\n")
    monkeypatch.setattr(PM, "REF_DIR", str(tmp_path))
    monkeypatch.setattr(PM, "env_style", lambda: "s")
    before = PM.tag()
    real = PM.code_files()
    edited = tmp_path / "markers_geom.py"
    edited.write_bytes(next(p for p in real if p.name == "markers_geom.py").read_bytes() + b"\n# edited\n")
    monkeypatch.setattr(PM, "code_files", lambda: [edited if p.name == "markers_geom.py" else p for p in real])
    assert PM.tag() != before
