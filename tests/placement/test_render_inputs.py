"""The render's object inputs (CPU): an overlay states what it opens (its sublayer and textures, resolved next to it),
an overlay that is not a text layer cannot be stated and dies, and the render statement maps exactly the spawned
object USDs (main objects, extra slots) to their overlays, never the supports."""

from __future__ import annotations

import pytest

from dexengine.core import artifact_paths as TP
from dexengine.placement.bundle import run_inputs as RI


def test_an_overlay_states_its_sublayer_and_textures(tmp_path):
    o = tmp_path / "plate" / "object_render.usd"
    o.parent.mkdir()
    o.write_text("#usda 1.0\n(\n subLayers = [\n  @./object.usd@\n ]\n)\nasset inputs:file = @./render_tex/a.png@\n")
    assert TP.overlay_assets(str(o)) == [str(o.parent / "object.usd"), str(o.parent / "render_tex" / "a.png")]


def test_an_overlay_that_is_not_a_text_layer_dies(tmp_path):
    o = tmp_path / "object_render.usd"
    o.write_bytes(b"PXR-USDC\x00\x01")
    with pytest.raises((ValueError, UnicodeDecodeError)):
        TP.overlay_assets(str(o))


def test_the_render_states_the_spawned_objects_not_the_supports():
    """The readers whose USD the render swaps for an overlay (env_setup objects · extra_slots): main objects and extra
    slots; the supports spawn their training USD. An articulated USD keeps its own overlay name."""
    assert set(RI.RENDER_SPAWNED) == {"entry:preflight", "kit:extra"} and "kit:support" not in RI.RENDER_SPAWNED
    assert {"render:object", "render:look"} <= set(RI.RENDER_READERS) <= set(RI.READERS)
    assert TP.render_usd_file("/d/b/object_artic.usd") == "/d/b/object_artic_render.usd"
    with pytest.raises(ValueError):
        TP.render_usd_file("/d/e/mesh.obj")


def test_render_only_inputs_read_through_the_index(tmp_path, monkeypatch):
    """A clip with no render-only object states the index alone; a missing index dies (no silent empty list)."""
    from dexengine.record.scene import render_only as RO

    monkeypatch.setattr(RO, "list_dir", lambda ds, root=None: tmp_path)
    ti = {"dataset_name": "taco", "clip_id": "c0"}
    with pytest.raises(RO.RenderOnlyError):
        RO.render_inputs(ti)
    (tmp_path / RO.INDEX).write_text(f'{{"version": {RO.VERSION}, "clips": {{"c0": 0}}}}', encoding="utf-8")
    assert RO.render_inputs(ti) == ([tmp_path / RO.INDEX], [])
