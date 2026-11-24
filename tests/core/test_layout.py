"""dexengine.core.layout: layer templates, object references, run layers and the data root."""

from __future__ import annotations

import os

import pytest

from dexengine.core import layout as L
from dexengine.core import paths as P
from dexengine.core.layout import roots


@pytest.mark.parametrize(
    ("layer", "ds", "axes", "want"),
    [
        (
            "trees",
            "taco",
            {"combo": "rby1_sharpa_wave", "emb": "bimanual"},
            "data/taco/trees/rby1_sharpa_wave/bimanual",
        ),
        (
            "trees",
            "hot3d",
            {"combo": "rby1_wuji2", "emb": "right", "tree": "c_pl_rt"},
            "data/hot3d/trees/rby1_wuji2/right/c_pl_rt",
        ),
        (
            "retarget",
            "taco",
            {"hand": "sharpa_wave", "emb": "bimanual"},
            "data/taco_retarget/processed/taco/sharpa_wave/bimanual",
        ),
        ("base", "grab", {"hand": "xhand_pro", "emb": "bimanual", "clip": "c"}, "data/grab/base/xhand_pro/bimanual/c"),
        ("precompute", "arctic", {"combo": "rby1_dg5f_s"}, "precompute/arctic/robots/rby1_dg5f_s"),
        ("usd", "taco", {}, "data/usd_cache/objects_taco"),
        ("collect", "hocap", {"combo": "rby1_sharpa_wave"}, "data/hocap/collect/rby1_sharpa_wave"),
    ],
)
def test_render(layer, ds, axes, want):
    assert L.render(layer, ds, **axes) == want


@pytest.mark.parametrize(
    ("layer", "ds", "axes", "exc"),
    [
        ("nope", "taco", {}, KeyError),  # unknown layer
        ("trees", "nope", {"combo": "x", "emb": "right"}, KeyError),  # unknown dataset
        ("trees", "taco", {"emb": "bimanual"}, KeyError),  # missing required axis
        ("trees", "taco", {"combo": "x", "emb": "both"}, ValueError),  # unknown embodiment
        ("usd", "taco", {"campaign": "c"}, KeyError),  # axis of no data layer
    ],
)
def test_render_refuses(layer, ds, axes, exc):
    with pytest.raises(exc):
        L.render(layer, ds, **axes)


def test_object_references():
    assert L.object_ref("taco", "116") == "data/taco/assets/objects/116"
    assert (
        L.object_path("taco", "data/taco/assets/objects/116/convex")
        == "data/taco_retarget/processed/taco/assets/objects/116/convex"
    )
    assert L.object_path("hocap", "data/hocap/assets/objects/G07_4") == "data/hocap/assets/objects/G07_4"
    assert (
        L.object_part_ref("oakink2", "data/oakink2/assets/objects/left_a", "../right_b/visual.obj")
        == "data/oakink2/assets/objects/right_b/visual.obj"
    )
    assert (
        L.object_part_ref("arctic", "data/arctic/assets/objects/box", "top.obj")
        == "data/arctic/assets/objects/box/top.obj"
    )
    assert (
        L.cooked_hull_file("data/oakink2/usd/a__b__hinge/object_artic.usd")
        == "data/oakink2/usd/a__b__hinge/object_artic.cooked.npz"
    )
    for bad in (
        lambda: L.object_path("taco", "data/hocap/assets/objects/1"),
        lambda: L.object_ref("taco", "../x"),
        lambda: L.object_part_ref("arctic", "data/arctic/assets/objects/a/b", "../c.obj"),
        lambda: L.cooked_hull_file("x.obj"),
    ):
        with pytest.raises(ValueError):
            bad()


def test_run_layers():
    assert L.render_run("run_chunk", "wuji2", tag="G1v_taco_00") == "overfit/hands/wuji2/chunks/G1v_taco_00"
    assert L.render_run("exp_shared") == "records"
    assert (
        L.render_run("run_rl_bundle", "sharpa_wave", campaign="c1", dataset="taco", bundle="b")
        == "overfit/hands/sharpa_wave/rl_campaigns/c1/taco/b"
    )
    with pytest.raises(KeyError):
        L.render_run("run_chunk", "wuji2")  # missing axis
    with pytest.raises(KeyError):
        L.render_run("run_chunk")  # hand layer without a hand
    with pytest.raises(ValueError):
        L.render_run("exp_shared", "wuji2")  # hand-free layer with a hand
    with pytest.raises(ValueError):
        L.render_run("run_chunk", "wuji2", tag="a/b")  # one axis = one path part
    with pytest.raises(ValueError):
        L.render_run("run_collect", "wuji2", tag="work")  # the fixed folder of another layer


def test_generations_and_rosters():
    assert L.generation_rel("data/taco/trajectories", "v1_o64") == "data/taco/_generations/v1_o64/trajectories"
    assert L.roster_root("v1_o64", "trees") == "data/taco/_generations/v1_o64/trajectories_gen_v2/rby1_sharpa_wave"
    assert L.taco_source_root() == "data/taco/_generations/v1_o64/trajectories"
    assert L.stage_kind("reread_hot3d", {"wuji2"}, {"hot3d"}) == "reread_stage"
    assert L.stage_kind("reretarget_nope", {"wuji2"}, {"hot3d"}) is None
    with pytest.raises(ValueError):
        L.generation_rel("data/taco", "v1")


def test_data_root_precedence(tmp_path, monkeypatch, data_root):
    assert L.data_root("/default") == str(data_root)  # env wins
    monkeypatch.delenv("DEXENGINE_MAIN_ROOT")
    assert L.data_root("/default") == "/default"  # nothing configured: the caller's default
    cfg_root = tmp_path / "cfg_root"
    cfg_root.mkdir()
    monkeypatch.setattr(roots, "_LAYOUT_CFG", [roots.LayoutConfig(data_root=str(cfg_root))])
    assert L.data_root("/default") == str(cfg_root)  # configs/layout.yaml
    assert P.main_root() == cfg_root


def test_layer_paths_sit_under_the_data_root(data_root):
    p = P.layer_path("trees", "hot3d", combo="rby1_sharpa_wave", emb="bimanual")
    assert p == data_root / "data/hot3d/trees/rby1_sharpa_wave/bimanual"


def test_repo_place_and_urdf_mesh(tmp_path, data_root):
    code = str(tmp_path / "code")
    assert L.repo_place("assets/robots/x.urdf", code) == os.path.join(code, "assets/robots/x.urdf")
    assert L.repo_place("data/_stage/x/y.urdf", code) == os.path.join(str(data_root), "data/_stage/x/y.urdf")
    with pytest.raises(ValueError):
        L.repo_place("../outside", code)
    urdf = os.path.join(code, "assets/robots/full_robot_gen/c/c.urdf")
    assert L.urdf_mesh(urdf, "meshes/a.stl", code) == os.path.join(code, "assets/robots/full_robot_gen/c/meshes/a.stl")
    assert L.urdf_mesh(urdf, "../../../../data/_stage/h/m.stl", code) == os.path.join(
        str(data_root), "data/_stage/h/m.stl"
    )


def test_record_storage_roots_default_and_site_file(tmp_path, monkeypatch, data_root):
    # release default: every record root inside the data root
    assert L.collect_round_roots() == {"orig": None, "aug": None}
    assert L.collect_job_archive_roots("aug") == (str(data_root / "archive" / "collect_traj"),)
    assert L.collect_final("taco", "rby1_x", "aug", str(data_root)) == str(data_root / "data/taco/collect/rby1_x")
    # a site layout file named by DEXENGINE_LAYOUT_CONFIG replaces configs/layout.yaml in this and every child process
    site = tmp_path / "site_layout.yaml"
    site.write_text(
        "collect_aug_root: /site/aug\ncollect_traj_archive: /site/traj\ncollect_archive_old_aug: [/site/old]\n"
    )
    monkeypatch.setenv(roots.LAYOUT_CONFIG_ENV, str(site))
    monkeypatch.setattr(roots, "_LAYOUT_CFG", [])
    assert L.collect_round_roots()["aug"] == "/site/aug"
    assert L.collect_job_archive_roots("aug") == ("/site/traj", "/site/old")
    assert L.collect_job_archive_roots("orig") == ("/site/traj",)
    assert L.collect_stage1_archive("w", "taco", "k") == "/site/traj/stage1/w/taco/k"
    with pytest.raises(KeyError):
        L.collect_job_archive_root("third")
