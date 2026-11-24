"""Step 4 (object roles, D24 input) needs every main object's USD: a USD file that does not exist fails loudly and
names its bake; `no_usd` is left only for a USD that exists without a rigid body or collision mesh."""

from __future__ import annotations

import collections
import json

import numpy as np
import pytest

from dexengine.data import desk_penetration as D24
from dexengine.data.objects import hull as OH

DS = "hrdexdb"


@pytest.fixture
def usd_dir(tmp_path, monkeypatch):
    """Object USD paths under a scratch folder (only the path rule is replaced; existence is the test's choice)."""
    d = tmp_path / "usd"
    monkeypatch.setattr(OH, "usd_file", lambda ds, mesh_id, articulated: d / str(mesh_id) / "object.usd")
    return d


def _tree(tmp_path, name="pan"):
    tree = tmp_path / "tree"
    (tree / "0").mkdir(parents=True)
    q = np.zeros((4, 7))
    q[:, 2], q[:, 3] = 0.8, 1.0  # z 0.8 m, identity rotation (wxyz)
    np.savez(tree / "0" / "trajectory_kinematic.npz", qpos=q)
    ti = {
        "dataset_name": DS,
        "clip_id": "pan__0",
        "objects": [
            {"name": name, "obj_id": name, "qpos_start": 0, "slot": 0, "spider_mesh_dir": "m", "spider_convex_dir": "c"}
        ],
    }
    return tree, ti


def _make_usd(usd_dir, name="pan"):
    (usd_dir / name).mkdir(parents=True)
    (usd_dir / name / "object.usd").write_text("#usda 1.0\n")


def test_measure_refuses_a_main_object_without_a_usd_file(tmp_path, usd_dir):
    tree, ti = _tree(tmp_path)
    with pytest.raises(D24.ObjectUsdMissing) as e:
        D24.measure(ti, str(tree), {"pan"})
    assert e.value.missing == [
        {"name": "pan", "asset": "pan", "articulated": False, "usd": str(usd_dir / "pan" / "object.usd")}
    ]
    assert not isinstance(e.value, OH.HullMissing)


def test_no_usd_is_a_present_usd_without_a_hull(tmp_path, usd_dir, monkeypatch):
    tree, ti = _tree(tmp_path)
    _make_usd(usd_dir)

    def no_hull(ds, mesh_id, articulated=False):
        raise OH.HullMissing("no rigid body")

    monkeypatch.setattr(OH, "hull_points", no_hull)
    assert D24.measure(ti, str(tree), {"pan"}) == "no_usd"
    monkeypatch.setattr(OH, "hull_points", lambda ds, mesh_id, articulated=False: np.array([[0.0, 0.0, -0.05]]))
    assert D24.measure(ti, str(tree), {"pan"}) == "hull"
    assert ti["support"]["obj_zmin_hull_traj"] == 0.75


def test_hull_reader_raises_usd_missing_for_an_absent_file(tmp_path, usd_dir):
    with pytest.raises(OH.UsdMissing):
        OH.hull_points(DS, "pan", False)


@pytest.fixture
def roles(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    from dexengine.data import object_roles as O

    dec = {
        "roles": {"pan": ("main", "right")},
        "tags": {},
        "main": {"right": "pan"},
        "active": ["right"],
        "support": {},
        "assign": "test",
    }
    monkeypatch.setattr(O, "check_case_input", lambda ds, tree_dir, ti: None)
    monkeypatch.setattr(O, "from_assign", lambda ti: dict(dec))
    monkeypatch.setattr(O, "support_roles", lambda ti, d, tree_dir: d)
    return O


def _write_tree(tmp_path):
    tree, ti = _tree(tmp_path)
    ti["case"] = {"case": "UNI_R"}
    p = tree / "task_info.json"
    p.write_text(json.dumps(ti))
    return p


def test_roles_step_fails_and_names_the_bake_for_a_missing_usd(tmp_path, usd_dir, roles, capsys):
    p = _write_tree(tmp_path)
    before = p.read_bytes()
    c, tally = collections.Counter(), roles.RunTally()
    roles._role_tree(DS, "right", str(p), True, c, tally)
    assert p.read_bytes() == before, "a tree whose main object has no USD must not be written"
    assert c["usd_missing"] == 1 and c["trees"] == 0
    assert [(m["asset"], m["clip"], m["usd"]) for m in tally.usd_missing] == [
        ("pan", "tree", str(usd_dir / "pan" / "object.usd"))
    ]
    rows = [{"candidates": 1, "trees": 0, "usd_missing": 1, "no_case": 0}]
    assert roles.exit_code(rows, tally) == 4
    err = capsys.readouterr().err
    bake, dump = OH.bake_commands(DS, "pan", False)
    assert str(usd_dir / "pan" / "object.usd") in err and bake in err and dump in err
    assert f"{OH.BAKE_GEN} --batch --dataset {DS} --ids pan" in bake


def test_roles_step_measures_when_the_usd_exists(tmp_path, usd_dir, roles, monkeypatch):
    p = _write_tree(tmp_path)
    _make_usd(usd_dir)
    monkeypatch.setattr(OH, "hull_points", lambda ds, mesh_id, articulated=False: np.array([[0.0, 0.0, -0.05]]))
    c, tally = collections.Counter(), roles.RunTally()
    roles._role_tree(DS, "right", str(p), True, c, tally)
    assert not tally.usd_missing and c["trees"] == 1 and c["written"] == 1
    assert json.loads(p.read_text())["rules"]["D24"]["basis"] == "hull"
    assert roles.exit_code([{"candidates": 1, "trees": 1, "usd_missing": 0, "no_case": 0}], tally) == 0


def test_articulated_bake_command():
    bake, dump = OH.bake_commands(DS, "lid_box", True)
    assert " --articulated " in bake and "--ids lid_box" in bake and OH.COOKED_GEN in dump
