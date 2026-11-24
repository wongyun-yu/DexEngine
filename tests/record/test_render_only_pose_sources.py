"""The raw -> base transform owners of the render-only object list (render_only_build.POSE_SOURCES) resolve through
pose_source: every entry imports and has render_only_tracks; a name from before the package layout or a module without
the function dies naming the entry (the oakink2 entry was once the flat script name)."""

from __future__ import annotations

import pytest

from dexengine.record.scene import render_only_build as ROB


def test_every_pose_source_resolves():
    assert ROB.POSE_SOURCES
    for ds, name in ROB.POSE_SOURCES.items():
        fn = ROB.pose_source(ds)
        assert callable(fn) and fn.__module__ == name and fn.__name__ == "render_only_tracks"


def test_a_flat_script_name_dies(monkeypatch):
    monkeypatch.setitem(ROB.POSE_SOURCES, "oakink2", "oakink2_raw_objects")
    with pytest.raises(ROB.RenderOnlyError, match="does not import"):
        ROB.pose_source("oakink2")


def test_a_module_without_the_function_dies(monkeypatch):
    monkeypatch.setitem(ROB.POSE_SOURCES, "oakink2", "json")
    with pytest.raises(ROB.RenderOnlyError, match="has no render_only_tracks"):
        ROB.pose_source("oakink2")
