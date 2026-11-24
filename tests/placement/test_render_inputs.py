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


def test_render_only_inputs_state_each_objects_surface_samples(tmp_path, monkeypatch):
    """scene.extras places a render-only object on the desk through static_desk.surface: the statement names that file
    for every listed object, and the packer refuses a data root where one of them is missing."""
    from dexengine.core import paths as P
    from dexengine.data.contact import fingerprint as SF
    from dexengine.placement.bundle import rl_bundle as RB
    from dexengine.record.scene import render_only as RO

    objs = [{"name": "render_only_a", "obj_id": "left_a"}, {"name": "render_only_b", "obj_id": "b"}]
    monkeypatch.setattr(RO, "list_dir", lambda ds, root=None: tmp_path)
    monkeypatch.setattr(RO, "read_clip", lambda ds, clip, root=None: {"objects": objs})
    monkeypatch.setattr(RO, "frame_files", lambda ti, rec: [])
    files, usds = RO.render_inputs({"dataset_name": "oakink2", "clip_id": "c0"})
    want = [SF.surface_path(P.main_root(), "oakink2", o) for o in objs]
    assert all(w in files for w in want) and len(usds) == 2
    assert all(w.name == "surface_points.npz" and w.parent.name == o["obj_id"] for w, o in zip(want, objs, strict=True))
    root = tmp_path / "root"
    rels = [f"meshes/{o['obj_id']}/surface_points.npz" for o in objs]
    for r in rels:
        (root / r).parent.mkdir(parents=True)
        (root / r).write_bytes(b"x")
    assert set(RB.stat_all(root, rels)) == set(rels)
    (root / rels[0]).unlink()  # negative: one stated surface file removed → the packer refuses
    with pytest.raises(SystemExit):
        RB.stat_all(root, rels)
