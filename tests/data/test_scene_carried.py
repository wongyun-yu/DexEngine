"""Rule S-carried on synthetic scenes (CPU): carry relation, every carried row attached to its carrier (kind
carried_attached, no replay decision), the collision tables, and a dataset that opts out keeping the rows before."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from dexengine.core import datasets as RR
from dexengine.data.objects import extra_slots as XS
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import scene_rules as SC
from tests.imports import own_module

T = 100


def _col(pos, rotvec=None):
    """(T,7) pose column xyz · wxyz."""
    q = R.from_rotvec(np.zeros((len(pos), 3)) if rotvec is None else rotvec).as_quat()
    return np.c_[pos, q[:, 3], q[:, :3]]


def _carried_scene(noise=0.001, put_down=False, seed=0):
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, T)
    tray = np.c_[0.3 * s, 0.1 * np.sin(np.pi * s), 0.8 + 0.05 * np.sin(np.pi * s)]
    rot = np.c_[np.zeros(T), np.zeros(T), 0.2 * s]
    Tt = XG.pose_mats(_col(tray, rot), 0)
    rel = np.eye(4)
    rel[:3, 3] = (0.03, -0.01, 0.025)
    To = Tt @ rel
    To[:, :3, 3] += rng.normal(0, noise, (T, 3))
    if put_down:  # the carrier leaves after frame 80; the carried object stays where it was put
        To[81:] = To[80]
        Tt[81:, :3, 3] += np.linspace(0, 0.2, T - 81)[:, None] * np.array([1.0, 0, 0])
    decoy = np.c_[-0.2 * s, 0.3 * s**2, 0.8 * np.ones(T)]
    q = np.concatenate(
        [
            _col(Tt[:, :3, 3], R.from_matrix(Tt[:, :3, :3]).as_rotvec()),
            _col(To[:, :3, 3], R.from_matrix(To[:, :3, :3]).as_rotvec()),
            _col(decoy),
        ],
        axis=1,
    )
    return q, rel


def test_carry_relation_finds_the_carrier_interval_and_relative_pose():
    q, rel = _carried_scene()
    got = XG.carry_relation(q, 7, {"tray": 0, "decoy": 14})
    assert [g["carrier"] for g in got] == ["tray"]
    g = got[0]
    assert (g["start"], g["end"]) == (0, T - 1)
    assert np.allclose(g["rel"][:3], rel[:3, 3], atol=0.002) and abs(abs(g["rel"][3]) - 1.0) < 1e-3
    assert g["dev_m"] < XG.CARRY_REL_M


def test_carry_relation_put_down_ends_the_interval():
    q, _ = _carried_scene(put_down=True)
    g = XG.carry_relation(q, 7, {"tray": 0})[0]
    assert g["start"] == 0 and 79 <= g["end"] <= 82


def test_carry_relation_rejects_motion_no_carrier_explains():
    q, _ = _carried_scene()
    q[:, 7:10] += np.linspace(0, 0.1, T)[:, None] * np.array([0, 0, 1.0])  # it also climbs on its own
    assert XG.carry_relation(q, 7, {"tray": 0, "decoy": 14}) == []


def _box_pts(half, n=7):
    """Surface samples of a box (a grid on each face) — the dense vertex set `placed` returns for a mesh."""
    g = np.linspace(-1, 1, n)
    a, b = np.meshgrid(g, g)
    faces = [
        np.c_[a.ravel(), b.ravel(), np.full(a.size, s)][:, perm]
        for s in (-1, 1)
        for perm in ([0, 1, 2], [0, 2, 1], [2, 0, 1])
    ]
    return np.unique(np.concatenate(faces), axis=0) * np.asarray(half, float)


def _tree(monkeypatch, tmp_path):
    q, _ = _carried_scene()
    q = q[:, :14]
    ti = {
        "dataset_name": "hot3d",
        "idle_sides": [],
        "case": {"contact_frac": {"bread": {}}, "grasp_s": {}},
        "objects": [
            {
                "name": "tray",
                "obj_id": "tray_id",
                "role": "main",
                "side": "right",
                "qpos_start": 0,
                "mesh_dir": "m/tray",
            },
            {"name": "bread", "obj_id": "bread_id", "role": "extra", "qpos_start": 7, "mesh_dir": "m/bread"},
        ],
    }
    shape = {"tray": _box_pts((0.12, 0.08, 0.01)), "bread": _box_pts((0.03, 0.03, 0.015))}

    def geom(tree_dir, ti_, q_):
        def placed(o, f):
            T_ = XG.pose_mats(q_[f : f + 1], int(o["qpos_start"]))[0]
            return shape[o["name"]] @ T_[:3, :3].T + T_[:3, 3]

        return placed, lambda Vu, Vl: False

    monkeypatch.setattr(XG, "geom", geom)
    monkeypatch.setattr(XG, "below_work_surface", lambda *a: False)
    seen = []
    carry_of = XG.carry_of

    def spy(tree_dir, ti_, q_, name, cands):
        seen.append((name, tuple(cands)))
        return carry_of(tree_dir, ti_, q_, name, cands)

    monkeypatch.setattr(XG, "carry_of", spy)
    return str(tmp_path), ti, q, seen


def _opt_out(monkeypatch, *rules):
    p = RR.DATASET_PROFILE["hot3d"]
    off = dict.fromkeys(rules, "test: the rows before the rule")
    monkeypatch.setitem(RR.DATASET_PROFILE, "hot3d", dataclasses.replace(p, scene_rules_off=off))


def test_carried_row_is_attached_by_default_and_moving_when_opted_out(monkeypatch, tmp_path, capsys):
    tree, ti, q, seen = _tree(monkeypatch, tmp_path)
    on = XS.clip_rows(tree, ti, q)
    r = on[0]
    assert r["cls"] == "SLOT" and r["kind"] == XS.KIND_ATTACHED and seen == [("bread", ("tray",))]
    assert r["carry"]["carrier"] == "tray" and r["carry"]["carrier_role"] == "main"
    assert r["carry"]["carrier_side"] == "right" and not {"stand", "why", "sensitive"} & set(r["carry"])
    assert "[CARRY_ATTACHED]" in capsys.readouterr().out
    assert XS.kinds_present(on) == XS.KINDS + (XS.KIND_ATTACHED,)
    assert XS.rule_version(["hot3d"]) == "+".join([XS.RULE, *SC.VERSIONS.values()])
    _opt_out(monkeypatch, *SC.VERSIONS)
    off = XS.clip_rows(tree, ti, q)
    assert [(r["name"], r["cls"], r["why"], r["kind"]) for r in off] == [("bread", "SLOT", "carried", "moving")]
    assert "carry" not in off[0] and len(seen) == 1
    assert XS.rule_version(["hot3d"]) == XS.RULE and XS.kinds_present(off) == XS.KINDS
    assert XS.spawn_key(on) == XS.spawn_key(off) == "1/0/0"  # counted in the moving bucket


def test_a_main_object_of_an_idle_side_is_not_a_carrier(monkeypatch, tmp_path):
    """The idle side has no robot hand: its main object cannot carry (R1) — the row keeps kind moving."""
    tree, ti, q, seen = _tree(monkeypatch, tmp_path)
    ti = dict(ti, idle_sides=["right"])
    r = XS.clip_rows(tree, ti, q)[0]
    assert r["kind"] == "moving" and "carry" not in r and seen == [("bread", ())]
    assert r["carry_miss"] == "unexplained" and r["carry_miss"] in XG.CARRY_MISS  # no active carrier left


def test_carried_collision_tables(run_dataset):
    cp = own_module("dexengine.rl.tasks.scene.collision_presets")
    run_dataset()
    base = cp.extra_collide_text()
    assert cp.extra_collide("carried_attached") == {"hand": "ON", "arm": "ON", "objects": "OFF"}
    assert cp.extra_has_collider("carried_attached")
    with pytest.raises(KeyError):
        cp.extra_collide("carried")  # the dynamic kind is gone
    assert "carried" not in base and cp.extra_kinds_in_play([["moving", None], [None]]) == tuple(cp.EXTRA_KIND_COLLIDE)
    assert cp.extra_kinds_in_play([["moving", "carried_attached"]])[-1] == "carried_attached"
    assert cp.ALL_OBJ_KEEP_OFF_KINDS == ("touched",) and set(cp.EXTRA_KIND_COLLIDE) == set(XS.KINDS)
    assert set(cp.EXTRA_SCOPED_KIND_COLLIDE) == set(XS.SCOPED_KINDS)
    assert cp.extra_collide("touched")["objects"] == "OFF"  # a touched static that fails the gate (or opted out)
    with pytest.raises(KeyError):
        cp.extra_collide("no_such_kind")
