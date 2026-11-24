"""The canonical cell pv00 of the placement augment is the canonical clip as it is:
augment.place_src canon_variant copies the canonical tree arrays, arm IK and wrist caches byte for byte instead of
baking them again from the judge seed, augment.bake_post canon_check rejects a pv00 that is not that copy, and a clip
with only pv00 left still gives the place gate its bake env (augment.bake_run bake_all)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from dexengine.core import paths as P
from dexengine.record.augment import bake as AB
from dexengine.record.augment import bake_post as ABP
from dexengine.record.augment import bake_run as ABR
from dexengine.record.augment import place_src as APS
from dexengine.record.augment import scene as SC

H, DS, EMB, ARM, CLIP = "xhand_pro", "grab", "bimanual", "rby1", "s5__cube_lift_seg02"
T, T_PRE = 12, 10


def _npz(p: Path, **arrays) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **arrays)
    return p


def _canon(main: Path) -> types.SimpleNamespace:
    """A canonical clip in the data root: tree (task_info · scene · F0_REPAIR · 0/ arrays, sidecars, contacts) + arm IK
    and wrist caches of both sides with sidecars + placement, and another clip next to it."""
    combo = SC.tree_combo(ARM, H, DS, EMB)
    trees_rel, wr, ik_rel, pl_rel = APS.rel_layers(DS, combo, EMB)
    t = P.tree_name(CLIP)
    rng = np.random.default_rng(7)
    td = main / trees_rel / t
    ti = {"dataset_name": DS, "hand": H, "embodiment_type": EMB, "scene": "desk"}
    (td / "0").mkdir(parents=True)
    (td / "task_info.json").write_text(json.dumps(ti))
    (td / "scene.xml").write_text("<mujoco/>\n")
    (td / "F0_REPAIR.json").write_text("{}")
    _npz(td / "0/trajectory_kinematic.npz", qpos=rng.normal(size=(T, 9)), retime_tau=np.linspace(0, 1, T_PRE))
    _npz(td / "0/trajectory_kinematic.pre_rt.npz", qpos=rng.normal(size=(T_PRE, 9)))
    _npz(td / "0/trajectory_ikrollout.npz", qpos=rng.normal(size=(T, 9)))
    _npz(td / "0/body_positions.npz", pos=rng.normal(size=(T, 3, 3)))
    (td / "0/xr_report.json").write_text('{"ok": true}')
    _npz(td / "0/contact_mano_right.npz", contact=np.zeros(T, bool))
    sc = P.sidecar_file("contact_base", DS, CLIP, "right", P.hand_spec_dir(H))  # the base the variant name aliases
    _npz(sc, contact=np.zeros(T_PRE, bool))
    for s in ("right", "left"):
        _npz(main / ik_rel[s] / f"{t}.npz", arm_joint_positions_all=rng.normal(size=(T, 7)), solver="pool_dp_v5_reuse")
        _npz(main / ik_rel[s] / f"{t}.pre_rt.npz", arm_joint_positions_all=rng.normal(size=(T_PRE, 7)))
        _npz(main / wr[s] / f"{t}.npz", pos=rng.normal(size=(T, 3)))
        _npz(main / ik_rel[s] / "other_clip_pl_rt.npz", arm_joint_positions_all=np.zeros((3, 7)))
        _npz(main / wr[s] / "other_clip_pl_rt.npz", pos=np.zeros((3, 3)))
    adopted = {"sequence": t, "anchor": "a0", "robot_z_off": 0.01, "idle_sides": []}
    (main / pl_rel).mkdir(parents=True, exist_ok=True)
    (main / pl_rel / f"{t}{P.PLACEMENT_SFX}").write_text(json.dumps(adopted))
    return types.SimpleNamespace(
        main=main,
        hand=H,
        arm=ARM,
        ds=DS,
        tree=td,
        tree_name=t,
        trees_rel=trees_rel,
        ik_rel=ik_rel,
        wrist_rel=wr,
        pl_rel=pl_rel,
        ti=ti,
        adopted=adopted,
        idle=[],
        active=["right", "left"],
        object_xy_frame0=lambda: np.array([0.1, -0.2, 0.0]),
    )


def _canonical_files(src) -> dict[str, Path]:
    """pv00 path (relative to the shadow) -> canonical file, listed here independently of canon_pairs."""
    name = P.tree_name(P.variant_name(CLIP, 0))
    out = {
        f"{src.trees_rel}/{name}/0/{f.name}": f
        for f in (src.tree / "0").iterdir()
        if not f.name.startswith("contact_mano_")
    }
    for s in ("right", "left"):
        for rel in (src.ik_rel[s], src.wrist_rel[s]):
            for f in (src.main / rel).glob(f"{src.tree_name}.*"):
                out[f"{rel}/{name}{f.name[len(src.tree_name) :]}"] = f
    return out


@pytest.fixture
def made(data_root, tmp_path):
    src = _canon(data_root)
    shadow = tmp_path / "shadow"
    SC.init_shadow(shadow, data_root)
    return src, shadow, APS.canon_variant(src, shadow)


def test_pv00_is_the_canonical_clip_byte_for_byte(made):
    src, shadow, rec = made
    name = P.tree_name(P.variant_name(CLIP, 0))
    assert rec["status"] == "baked" and rec["name"] == name and rec["k"] == 0
    want = _canonical_files(src)
    assert len(want) == 5 + 6  # tree 0/ arrays and sidecars · arm IK (+ .pre_rt) and wrist, both sides
    for rel, f in want.items():
        got = shadow / rel
        assert got.is_file() and not got.is_symlink(), rel
        assert got.read_bytes() == f.read_bytes(), rel
    # the arm solution pv00 carries = the canonical one
    for s in ("right", "left"):
        a = np.load(shadow / src.ik_rel[s] / f"{name}.npz")["arm_joint_positions_all"]
        b = np.load(src.main / src.ik_rel[s] / f"{src.tree_name}.npz")["arm_joint_positions_all"]
        assert np.array_equal(a, b)
    assert rec["bake"] == {
        "copy": "canonical",
        "files": len(want),
        "T": T,
        "T_pre": T_PRE,
        "stretch": round(T / T_PRE, 3),
        "solver": {"right": "pool_dp_v5_reuse", "left": "pool_dp_v5_reuse"},
        "xr": True,
    }
    # contacts come from after_bake (remap_contacts), other clips are not touched
    assert not (shadow / src.trees_rel / name / "0/contact_mano_right.npz").exists()
    assert not any((shadow / src.ik_rel["right"]).glob("other_clip_pl_rt_pv*"))
    ti = json.loads((shadow / src.trees_rel / name / "task_info.json").read_text())
    ap = ti["aug_place"]
    assert (ap["dx_scene"], ap["dy_scene"], ap["yaw_deg"], ap["fn"]) == (0.0, 0.0, 0.0, APS.CANON_COPY_FN)
    tf = SC.record_tf(ti)
    q = np.array([0.3, 0.4, 0.9])
    assert np.allclose(tf.p(q), q)  # identity
    pl = json.loads((shadow / src.pl_rel / f"{name}{P.PLACEMENT_SFX}").read_text())
    assert pl["sequence"] == name and pl["robot_z_off"] == src.adopted["robot_z_off"]
    # the base alias of the variant name (the place gate worker finds the base layer by tree name)
    al = [shadow / a for a in rec["alias"]]
    assert (
        len(al) == 1
        and al[0].is_symlink()
        and al[0].resolve() == P.sidecar_file("contact_base", DS, CLIP, "right", P.hand_spec_dir(H)).resolve()
    )


def test_the_copy_check_passes_the_copy_and_catches_a_changed_one(made):
    src, shadow, rec = made
    rec = dict(rec, status="ok")
    ABP.canon_check(src, shadow, rec)
    assert rec["status"] == "ok" and rec["canon_copy"]["ok"] and rec["canon_copy"]["same"] == rec["bake"]["files"]
    # negative: one arm joint changed in the pv00 arm IK -> pv00 is not the canonical clip
    f = shadow / src.ik_rel["left"] / f"{rec['name']}.npz"
    z = dict(np.load(f))
    z["arm_joint_positions_all"][3, 2] += 1e-3
    np.savez(f, **z)
    bad = dict(rec)
    ABP.canon_check(src, shadow, bad)
    assert not bad["canon_copy"]["ok"] and (bad["status"], bad["why"]) == ("reject", "bake")
    assert f.name in bad["detail"]
    # negative: a copied file gone
    f.unlink()
    gone = dict(rec)
    ABP.canon_check(src, shadow, gone)
    assert gone["status"] == "reject" and "missing" in gone["detail"]


@pytest.mark.parametrize("gone", ["wrist_left", *APS.CANON_TREE_MUST])
def test_a_missing_canonical_file_rejects_pv00_without_dying(data_root, tmp_path, gone):
    src = _canon(data_root)
    f = src.main / src.wrist_rel["left"] / f"{src.tree_name}.npz" if gone == "wrist_left" else src.tree / "0" / gone
    f.unlink()
    shadow = tmp_path / "shadow"
    SC.init_shadow(shadow, data_root)
    rec = APS.canon_variant(src, shadow)
    assert (rec["status"], rec["why"]) == ("reject", "bake")
    assert rec["detail"] == f"canonical file missing — {f}"


def test_a_clip_with_only_pv00_gives_the_place_gate_its_bake_env(made, monkeypatch, tmp_path):
    src, shadow, rec = made
    monkeypatch.setattr(ABR, "clip_ctx", lambda s: {"tag": "T1", "active": "both"})
    bake = ABR.bake_all(src, shadow, [rec], tmp_path / "bundle", None, tmp_path / "log")
    assert bake["n"] == 0 and bake["active"] == "both" and bake["log_dir"] == str(tmp_path / "log")
    seen = {}
    monkeypatch.setattr(ABP, "bake_env", lambda s, active, sh: seen.setdefault("active", active) and {})
    troot = Path(shadow) / src.trees_rel
    monkeypatch.setattr(
        ABP, "worker", lambda env, cmd, args, log_dir: [{"tree": str(troot / rec["name"]), "why": "", "short_m": None}]
    )
    ok = dict(rec, status="ok")
    ABP.place_gate(src, shadow, [ok], bake)
    assert seen["active"] == "both" and ok["status"] == "ok" and ok["place"] == {"why": "", "short_mm": None}


def test_the_canonical_cell_has_no_bake_seed(made, tmp_path):
    src, _shadow, _rec = made
    with pytest.raises(SystemExit, match="no bake seed"):
        AB.seeds(src, 0, tmp_path / "bundle")


def test_after_bake_runs_the_copy_check_on_pv00(made, monkeypatch):
    from dexengine.record.augment import place_gen as G

    src, shadow, rec = made
    monkeypatch.setattr(G.ABP, "remap_contacts", lambda s, tree: [])
    monkeypatch.setattr(G.SPG, "speed", lambda s, tree, qs: (True, "", {}))
    good = dict(rec)
    G.after_bake(src, shadow, [good])
    assert good["status"] == "ok" and good["canon_copy"]["ok"]
    f = shadow / src.trees_rel / rec["name"] / "0/body_positions.npz"
    f.write_bytes(b"X" + f.read_bytes()[1:])  # first byte of the zip changed
    bad = dict(rec)
    G.after_bake(src, shadow, [bad])
    assert (bad["status"], bad["why"]) == ("reject", "bake") and not bad["canon_copy"]["ok"]
