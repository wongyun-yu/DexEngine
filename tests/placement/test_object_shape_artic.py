"""The object key says whether a main object is articulated, and the shape family reads that member's surface points
through artifact_paths.surface_dir: an articulated member from the usd_artic generation folder (the only place v0.1
writes them), a rigid member from the meshes layer as before."""

import json
from pathlib import Path

import numpy as np
import pytest

from dexengine.core import artifact_paths as AP
from dexengine.core import layout as L
from dexengine.core.errors import InputMissingError
from dexengine.core.paths import layer_path
from dexengine.placement.train_set import object_shape as OS

HAND, EMB = "sharpa_wave", "bimanual"
BOX = {"type": "hinge", "parts": {"base": "bottom", "moving": "top"}}  # moving part inside the object folder (arctic)
PAIR = {"type": "hinge", "parts": {"moving_id": "right_o02_0001_00002"}}  # moving part = another object (OakInk2)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(OS, "_OBJ_CACHE", {})
    monkeypatch.setattr(OS, "_EXT_CACHE", {})
    AP.get.cache_clear()
    yield tmp_path
    AP.get.cache_clear()


def _clip(ds: str, clip: str, objects: list[dict]) -> str:
    f = layer_path("base", ds, hand=HAND, emb=EMB) / clip / "task_info.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"objects": objects}))
    return clip


def _cloud(d: Path, scale) -> None:
    d.mkdir(parents=True, exist_ok=True)
    pts = np.random.RandomState(0).uniform(-1, 1, (500, 3)) * np.asarray(scale, float)
    np.savez(d / "surface_points.npz", points=pts)


def _main(obj_id: str, ds: str, art=None) -> dict:
    return {
        "name": "right_object",
        "role": "main",
        "obj_id": obj_id,
        "mesh_dir": f"data/{ds}/assets/objects/{obj_id}",
        "articulation": art,
    }


def test_articulated_member_reads_the_usd_artic_folder(root):
    c = _clip("arctic", "c0", [_main("box", "arctic", BOX)])
    key = OS.target_object(c, "arctic", EMB, HAND)
    assert key == "box" + OS.ARTIC_MEMBER
    gen = root / f"data/arctic/usd_artic/{L.ARTIC_BAKE_GEN}/box"
    assert OS.shape_files("arctic", key) == [gen / "surface_points.npz"]
    _cloud(gen, (0.30, 0.02, 0.02))  # v0.1 cloud: a rod
    _cloud(root / "data/arctic/meshes/box", (0.1, 0.1, 0.1))  # the older tree's cloud: a cube — must not be read
    ext = OS.object_extents("arctic", key)
    assert ext[0] > 10 * ext[2]  # measured the usd_artic cloud
    # a paired articulation is keyed by its asset, not by its base part's rigid folder
    c2 = _clip("oakink2", "c1", [_main("left_o02_0001_00001", "oakink2", PAIR)])
    key2 = OS.target_object(c2, "oakink2", EMB, HAND)
    assert key2 == "left_o02_0001_00001__right_o02_0001_00002__hinge" + OS.ARTIC_MEMBER
    assert OS.shape_files("oakink2", key2)[0].parent.parent == root / f"data/oakink2/usd_artic/{L.ARTIC_BAKE_GEN}"


def test_rigid_member_is_unchanged(root):
    c = _clip("arctic", "c0", [_main("bowl", "arctic"), _main("box", "arctic", BOX)])
    key = OS.target_object(c, "arctic", EMB, HAND)
    assert key == "bowl+box" + OS.ARTIC_MEMBER  # sorted, rigid member = its mesh folder name as before
    rigid = OS.shape_files("arctic", key)[0]
    assert rigid == layer_path("meshes", "arctic") / "bowl" / "surface_points.npz"  # the old formula


def test_half_moved_and_malformed_members_die(root):
    """Only the older tree's file exists → missing (no fallback to the meshes layer); a malformed member dies."""
    _cloud(root / "data/arctic/meshes/box", (0.1, 0.1, 0.1))
    with pytest.raises(InputMissingError, match="usd_artic"):
        OS.object_extents("arctic", "box" + OS.ARTIC_MEMBER)
    for bad in (OS.ARTIC_MEMBER, "a" + OS.ARTIC_MEMBER + "b" + OS.ARTIC_MEMBER):
        with pytest.raises(ValueError, match="key member"):
            OS.member_surface("arctic", bad)
