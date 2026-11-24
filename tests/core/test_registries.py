"""Registries: datasets, retargeting hand assets, hands, arms, mounts and the names built from them."""

from __future__ import annotations

import os

import pytest

from dexengine.core import layout as L
from dexengine.core import paths as P
from dexengine.core.arms import ARM_REGISTRY
from dexengine.core.hands import HAND_REGISTRY
from dexengine.core.mounts import MOUNT_REGISTRY

CANON_HANDS = ("sharpa_wave", "wuji2", "xhand_pro", "dg5f_s")
PAPER_DATASETS = {"arctic", "dexcanvas", "grab", "h2o", "hocap", "hot3d", "hrdexdb", "oakink2", "taco"}


def test_datasets():
    assert set(L.DATASETS) >= PAPER_DATASETS
    for name, spec in L.DATASETS.items():
        assert spec.name == name
        assert spec.workspace in L.WORKSPACES
        assert set(spec.embodiments) <= set(L.EMBODIMENTS)
        assert spec.source_fps > 0
        assert L.dataset(name) is spec
    with pytest.raises(KeyError):
        L.dataset("nope")


@pytest.mark.parametrize("hand", CANON_HANDS)
def test_hand_rig_matches_its_registered_hash(hand):
    spec_dir = P.hand_spec_dir(hand)
    asset = L.RETARGET_HAND_ASSETS[spec_dir]
    assert asset.status == "canon" and asset.isaac_hand_id == hand
    assert L.asset_hash(str(P.hand_rig_dir(spec_dir))) == asset.sha256_xml3
    assert L.identify_asset(str(P.hand_rig_dir(spec_dir))) is asset


@pytest.mark.parametrize(
    ("hand", "spec_dir"),
    [("sharpa_wave", "sharpa_wave"), ("wuji2", "wuji2"), ("xhand_pro", "xhand_pro"), ("dg5f_s", "dg5f_s")],
)
def test_combo_names(hand, spec_dir):
    assert P.hand_spec_dir(hand) == spec_dir
    assert P.hand_id(hand) == hand
    combo = P.combo_id("rby1", hand)
    assert combo == f"rby1_{hand}"
    assert combo in P.combo_ids()
    assert P.split_combo(combo, P.ARM_IDS, P.HAND_SPEC_DIR) == ("rby1", hand, "")
    assert P.combo_dirname(combo) == f"rby1_{hand}"
    assert P.split_combo(combo + "_artic", P.ARM_IDS, P.HAND_SPEC_DIR) == ("rby1", hand, "_artic")


@pytest.mark.parametrize("hand", CANON_HANDS)
def test_hand_specs(hand):
    spec = HAND_REGISTRY[hand]
    assert (ARM_REGISTRY["rby1"].name, hand) in {(a, h) for a, h in MOUNT_REGISTRY}
    assert spec.hand_qpos_dim > 0
    assert spec.vendor_urdf


def test_repo_vendor_urdf_exists():
    spec = HAND_REGISTRY["sharpa_wave"]
    for side in ("right", "left"):
        rel = spec.vendor_urdf.format(side=side)
        assert os.path.isfile(L.repo_place(rel, str(P.REPO_ROOT))), rel
        assert os.path.isdir(L.repo_place(spec.vendor_meshes.format(side=side), str(P.REPO_ROOT)))


def test_mounts_reference_registered_arms_and_hands():
    for arm, hand in MOUNT_REGISTRY:
        assert arm in ARM_REGISTRY and hand in HAND_REGISTRY


@pytest.mark.parametrize(
    ("name", "tree", "clip"),
    [
        (
            "brush_brush_bowl__20230919_036_pl_rt",
            "brush_brush_bowl__20230919_036_pl_rt",
            "brush_brush_bowl__20230919_036",
        ),
        ("brush_brush_bowl__20230919_036", "brush_brush_bowl__20230919_036_pl_rt", "brush_brush_bowl__20230919_036"),
        (
            "p0002_45904c71__s00617_e01084_seg01_t0",
            "p0002_45904c71__s00617_e01084_seg01_t0_pl_rt",
            "p0002_45904c71__s00617_e01084_seg01_t0",
        ),
    ],
)
def test_clip_names(name, tree, clip):
    assert P.tree_name(name) == tree
    assert P.canonical_clip(name) == clip
    assert P.variant_name(name, 3) == name + "_pv03"
    assert P.ikc_name(name, "a0", "right") == f"{clip}_aa0_right.npz"
    assert L.variant_source(name + "_pv03") == name


def test_clip_name_errors():
    with pytest.raises(ValueError):
        P.canonical_clip("Brush Brush-Bowl (2023.09.19) 036")
    assert P.clip_id_from_raw("(brush, brush, bowl)", "20230919_036", "taco") == "brush_brush_bowl__20230919_036"
