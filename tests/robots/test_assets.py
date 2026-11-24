"""Hand rigs and merged robot URDFs: the resolvers point at the shipped assets and the assets are the registered
ones."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from dexengine.core import layout, paths
from dexengine.core.repo import FULL_ROBOT_URDF_DIR, HAND_ASSETS_DIR
from dexengine.robots import hand_rig

RIGS = ("sharpa_wave", "dg5f_s", "wuji2", "xhand_pro")


@pytest.mark.parametrize("spec", RIGS)
def test_rig_resolves_and_matches_registry(spec):
    """paths.hand_rig_dir is the one rig resolver, and the rig bytes are the identity the registry locks."""
    rig = paths.hand_rig_dir(spec)
    assert rig == HAND_ASSETS_DIR / spec / "mjcf"
    assert (rig / "retarget_prior.yaml").is_file()
    assert layout.asset_hash(str(rig)) == layout.RETARGET_HAND_ASSETS[spec].sha256_xml3


def test_output_places(tmp_path):
    """The rig builder writes the canonical rig unless told otherwise; the render URDF sits beside the merged one."""
    from dexengine.robots.urdf import merge

    assert hand_rig.rig_out_dir(None, "dg5f_s") == paths.hand_rig_dir("dg5f_s")
    assert hand_rig.rig_out_dir(tmp_path, "dg5f_s") == tmp_path / "dg5f_s"
    assert merge.render_urdf_path(tmp_path / "rby1_x.urdf") == tmp_path / "rby1_x_render.urdf"


def _mesh_refs(urdf):
    return sorted(set(re.findall(r'filename="([^"]+)"', urdf.read_text())))


@pytest.mark.parametrize("urdf", sorted(FULL_ROBOT_URDF_DIR.glob("*/*.urdf")), ids=lambda p: p.name)
def test_merged_urdf_tracked_meshes_resolve(urdf):
    """Every mesh of a merged URDF that lives in the tracked assets resolves inside this tree; data-layer meshes
    (data/_stage/...) are checked only when the data root has them."""
    ET.parse(urdf)
    data_stage = paths.main_root() / "data" / "_stage"
    missing = []
    for fn in _mesh_refs(urdf):
        if "data/_stage/" in fn and not data_stage.is_dir():
            continue
        if not paths.urdf_mesh(urdf, fn).is_file():
            missing.append(fn)
    assert not missing, missing[:3]
