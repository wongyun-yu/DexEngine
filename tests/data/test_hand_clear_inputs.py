"""2f reuses a tree's verdict only while every input it read is the same (hand_clear_stage.input_key ·
stage1.artifacts.input_key): before, the key was the qpos sha and the rule version only, so a rerun after a change of
the contacts, roles, case stamp, shapes or ruler kept the old verdict and exclusion lines (the stage-1 column read the
tree as current and the driver skipped the step)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dexengine.core import paths as P
from dexengine.data.stage1 import artifacts as AF
from dexengine.placement.repair import hand_clear_stage as HCS

CLIP = "p0001_abcd__s00010_e00090"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """One base tree with every input 2f reads: task_info (case, roles, rules incl. the later 5b stamp), qpos, a scene
    naming a mesh, a MANO take, two sidecar layers per side and one object folder."""
    root = tmp_path / "root"
    t = root / "data/hot3d/base/sharpa_wave/bimanual" / CLIP
    (t / "0").mkdir(parents=True)
    np.savez(t / "0/trajectory_kinematic.npz", qpos=np.arange(12.0).reshape(3, 4))
    ti = {
        "dataset_name": "hot3d",
        "case": {"contact_frac": 0.4},
        "objects": [{"name": "mug", "obj_id": "mug", "role": "main", "side": "right", "tag": None}],
        "qpos_layout": [{"kind": "object", "name": "mug", "role": "main", "side": "right"}],
        "shared_object": False,
        "rules": {"D1g": {"version": 1}, "ASSIGN": {"v": 2}, "D23": {"version": 3}, "C640": {"gap": 0.01}},
        "provenance": {"mano_dir": "data/hot3d/mano/bimanual/p0001_abcd__s00010_e00090"},
    }
    (t / "task_info.json").write_text(json.dumps(ti))
    (root / "data/hot3d/assets/hand").mkdir(parents=True)
    (root / "data/hot3d/assets/hand/palm.stl").write_bytes(b"palm-v1")
    (t / "scene.xml").write_text(
        '<mujoco><compiler meshdir="../../../../assets/"/><mesh file="hand/palm.stl"/></mujoco>'
    )
    mano = root / ti["provenance"]["mano_dir"] / "0"
    mano.mkdir(parents=True)
    (mano / "trajectory_keypoints.npz").write_bytes(b"mano-v1")
    side = {}
    for layer in ("contact_all", "contact_base"):
        for s in ("right", "left"):
            f = root / "precompute" / layer / f"{CLIP}_{s}.npz"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(f"{layer}-{s}-v1".encode())
            side[(layer, s)] = f
    obj = root / "data/hot3d/assets/objects/mug"
    obj.mkdir(parents=True)
    (obj / "visual.obj").write_bytes(b"v 0 0 0")
    monkeypatch.setattr(P, "main_root", lambda: str(root))
    monkeypatch.setattr(P, "sidecar_file", lambda layer, ds, clip, s, hand: side[(layer, s)])
    monkeypatch.setattr(AF, "_object_dirs", lambda root_, ds, o: [obj])
    monkeypatch.setattr(AF, "_MEMO", {})
    from dexengine.retarget.f0 import shadow_key as SK

    monkeypatch.setattr(SK, "_GEOM_MEMO", {})
    return {"t": t, "ti": ti, "side": side, "obj": obj, "root": root, "mano": mano / "trajectory_keypoints.npz"}


def _key(tree) -> str:
    AF._MEMO.clear()
    from dexengine.retarget.f0 import shadow_key as SK

    SK._GEOM_MEMO.clear()
    return HCS.input_key(tree["t"])


def _edit_ti(tree, fn) -> None:
    ti = json.loads((tree["t"] / "task_info.json").read_text())
    fn(ti)
    (tree["t"] / "task_info.json").write_text(json.dumps(ti))


def test_the_key_follows_every_input_2f_reads(tree):
    k0 = _key(tree)
    assert _key(tree) == k0  # the same inputs, the same key
    changes = {
        "case stamp (3)": lambda: _edit_ti(tree, lambda ti: ti["case"].update(contact_frac=0.5)),
        "roles (4)": lambda: _edit_ti(tree, lambda ti: ti["objects"][0].update(role="extra")),
        "D1-g stamp (2b)": lambda: _edit_ti(tree, lambda ti: ti["rules"]["D1g"].update(version=2)),
        "all-object sidecar (2d)": lambda: tree["side"][("contact_all", "left")].write_bytes(b"x"),
        "main-object sidecar (5)": lambda: tree["side"][("contact_base", "right")].write_bytes(b"y"),
        "scene mesh": lambda: (tree["root"] / "data/hot3d/assets/hand/palm.stl").write_bytes(b"palm-v2"),
        "MANO take": lambda: tree["mano"].write_bytes(b"mano-v2"),
        "object geometry": lambda: (tree["obj"] / "visual.obj").write_bytes(b"v 0 0 1"),
        "qpos": lambda: np.savez(tree["t"] / "0/trajectory_kinematic.npz", qpos=np.ones((3, 4))),
    }
    seen = {k0}
    for name, change in changes.items():
        change()
        k = _key(tree)
        assert k not in seen, name  # every input moves the key
        seen.add(k)


def test_a_later_step_stamp_keeps_the_key(tree):
    """5b writes its stamp after 2f: the key strips it (table.written_after), so a whole run leaves 2f current."""
    assert "support_gap" in AF.written_after(AF.owner_step(HCS.MODULE))
    k0 = _key(tree)
    _edit_ti(tree, lambda ti: ti["rules"]["C640"].update(gap=0.02))
    assert _key(tree) == k0


def test_the_sidecars_in_the_key_are_the_table_reads(tree, monkeypatch):
    """Negative: were contact_base not a declared 2f read, its change would not move the key — the reads come from the
    table (step_reads), not from a list in the step."""
    reads = tuple(a for a in AF.step_reads("2f") if a != "contact_base")
    monkeypatch.setattr(AF, "step_reads", lambda step, stages=None: reads)
    k0 = _key(tree)
    tree["side"][("contact_base", "right")].write_bytes(b"z")
    assert _key(tree) == k0


def test_current_needs_the_same_inputs(tree, monkeypatch):
    """The column (tree_fresh): a ledger row with the tree's qpos sha, key and rule version is current; a case change
    makes it stale (it read current before this fix: the key was qpos + version only)."""
    from dexengine.placement.repair.wrist_desk_fresh import qpos_sha_file

    t = tree["t"]
    monkeypatch.setattr(HCS, "holds_repair", lambda tr: False)
    rec = (qpos_sha_file(HCS.step_input(t)), _key(tree), "v")
    assert HCS._current(rec, t, "v")
    _edit_ti(tree, lambda ti: ti["case"].update(contact_frac=0.9))
    AF._MEMO.clear()
    assert not HCS._current(rec, t, "v")
    assert not HCS._current((rec[0], None, "v"), t, "v")  # a ledger written before the key (no inputs column)


def test_the_rule_version_carries_the_ruler_and_the_shape_rule(monkeypatch):
    from dexengine.data.objects import sdf_grid

    v0 = HCS.rule_version("hot3d")
    assert ":ruler=" in v0 and ":shapes=" in v0
    monkeypatch.setattr(sdf_grid, "ruler_id", lambda: "0" * 40)
    assert HCS.rule_version("hot3d") != v0


def test_every_artifact_has_a_place(monkeypatch):
    AF._check_places()
    monkeypatch.setattr(AF, "PLACES", {k: v for k, v in AF.PLACES.items() if k != "support_gap"})
    with pytest.raises(SystemExit, match="artifact places"):
        AF._check_places()
