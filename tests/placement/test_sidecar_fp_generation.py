"""The setup input statement (coords.setup.sidecar_fp_inputs) carries what the sidecar fingerprint check opens.

Measured (strace of f0 on an aerial arctic tree): the check `_contact_sidecars → fingerprint.expected` reads the
dataset's generation declaration `data/<ds>/GENERATION` (ds_ref: the reference direction the base must carry), and the
statement did not list it, so neither did the aerial f0 key (scene_z_offset_inputs → setup_inputs) nor the judge and
dexengine.placement.train_ready keys. The statement now appends the check's own inputs from their owner
(fingerprint.expected_inputs, via contacts.checked_sidecars) after its entries, only where they are not already stated:
- the aerial f0 key moves when that dataset's declaration changes, and before the fix it did not (negative);
- the key of a tree of another dataset keeps its bytes when this dataset's declaration changes, and so does a
  desk (table) tree's key: its z offset reads task_info only, never the sidecars;
- a tree whose check opens nothing new (a dataset without a declaration file · a tree without a sidecar) keeps its
  statement entry for entry (the statement before the fix is kept here as the reference, `_old_sidecar_fp_inputs`).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from dexengine.core import paths as P
from dexengine.data import align as RD
from dexengine.data.contact import fingerprint as SF
from dexengine.placement.coords import contacts as CC
from dexengine.placement.coords import setup as CS
from dexengine.retarget.f0 import shadow_key as SK

H, EMB, CLIP = "xhand_pro", "bimanual", "s5__cube_lift_seg02"


def _w(p: Path, body="x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, dict):
        p.write_text(json.dumps(body))
    elif p.suffix == ".npz":
        np.savez(p, qpos=np.zeros((3, 4)))
    else:
        p.write_text(body)
    return p


def _tree(main: Path, ds: str, decl: str | None = "119.79", sidecar: bool = True) -> tuple[Path, dict]:
    """A base tree of `ds` whose right hand holds its main object, with the files the fingerprint check opens: the
    contact_base sidecar of the right side (optional), the MANO keypoints, the surface sample and the generation
    declaration (optional, with that reference value)."""
    obj = {
        "name": "cube",
        "obj_id": "cubesmall",
        "mesh_dir": f"data/{ds}/assets/objects/cubesmall",
        "role": "main",
        "side": "right",
    }
    ti = {
        "dataset_name": ds,
        "hand": H,
        "embodiment_type": EMB,
        "hands": ["right"],
        "objects": [obj],
        "provenance": {"mano_dir": f"data/{ds}/mano/{EMB}/s5_seq"},
    }
    td = P.ref_base_dir(ds, H, EMB) / CLIP
    _w(td / "task_info.json", ti)
    _w(td / "0/trajectory_kinematic.npz")
    _w(main / ti["provenance"]["mano_dir"] / "0/trajectory_keypoints.npz")
    _w(SF.surface_path(main, ds, obj))
    if sidecar:
        _w(P.sidecar_file("contact_base", ds, CLIP, "right", P.hand_spec_dir(H)))
    if decl is not None:
        _w(RD.decl_path(P.layer_path("intake", ds)), f"generation=g1\n{RD.KEY}={decl}\n")
    return td, ti


def _old_sidecar_fp_inputs(tree_dir, ti: dict, stated=()) -> list:
    """The statement before the fix (base tree · surface samples · h2o plane), the reference for 'bytes unchanged'."""
    tree_dir = Path(tree_dir)
    ds = str(ti.get("dataset_name"))
    out = []
    if ti.get("hand") and ti.get("embodiment_type"):
        bt = CC.sidecar_base_tree(tree_dir, ti)
        out += [bt / "task_info.json", bt / "0/trajectory_kinematic.npz"]
    out += [SF.surface_path(P.main_root(), ds, o) for o in ti.get("objects") or [] if o.get("obj_id")]
    if ds == "h2o":
        from dexengine.data.readers import h2o_frame as _HF  # noqa: PLC0415

        out.append(_HF.plane_path())
    return out


def _key(td: Path, ds: str, monkeypatch, scene: str = "aerial") -> str:
    """The f0 input key of a tree in `scene`, with the hand and object parts out of the way (no manipulating side, no
    object repair): what moves it here is the zin part, the files scene_z_offset_inputs states."""
    monkeypatch.setattr(SK, "clip_sides", lambda *a: ())
    monkeypatch.setattr(SK, "_f0_mod", lambda: types.SimpleNamespace(HC=types.SimpleNamespace(f_cap_rad_s=1.0)))
    monkeypatch.setattr(SK, "_feed_obj_acted", lambda *a: None)
    monkeypatch.setattr(P.qpos_layout, "read", lambda *a, **k: None)
    monkeypatch.setattr(CS, "scene_mode", lambda tree_dir: scene)
    return SK.f0_input_key(td.name, td.parent, {}, ds, "base")


def test_the_declaration_is_stated_after_the_old_entries(data_root, monkeypatch):
    td, ti = _tree(data_root, "grab")
    decl = RD.decl_path(P.layer_path("intake", "grab"))
    new, stmt = CS.setup_inputs(td, ti), CS.sidecar_fp_inputs(td, ti)
    old_stmt = _old_sidecar_fp_inputs(td, ti)
    # standalone (the render lane): the old entries in their order, then the check's MANO keypoints and declaration
    assert stmt[: len(old_stmt)] == old_stmt and decl in stmt[len(old_stmt) :]
    monkeypatch.setattr(CS, "sidecar_fp_inputs", _old_sidecar_fp_inputs)
    # in setup_inputs the MANO keypoints are already stated: the declaration is the one entry added, at the end
    assert [str(x) for x in new] == [str(x) for x in CS.setup_inputs(td, ti)] + [str(decl)]


@pytest.mark.parametrize("case", ["no_declaration", "no_sidecar"])
def test_a_tree_whose_check_opens_nothing_new_keeps_its_statement(data_root, monkeypatch, case):
    td, ti = _tree(data_root, "grab", decl=None if case == "no_declaration" else "119.79", sidecar=case != "no_sidecar")
    new = CS.setup_inputs(td, ti)
    monkeypatch.setattr(CS, "sidecar_fp_inputs", _old_sidecar_fp_inputs)
    assert [str(x) for x in new] == [str(x) for x in CS.setup_inputs(td, ti)]


def test_the_aerial_f0_key_moves_with_the_declaration(data_root, monkeypatch):
    td, _ti = _tree(data_root, "grab", decl="119.79")
    k0 = _key(td, "grab", monkeypatch)
    _tree(data_root, "grab", decl="119.91")
    assert _key(td, "grab", monkeypatch) != k0
    _tree(data_root, "grab", decl="119.79")
    assert _key(td, "grab", monkeypatch) == k0


def test_the_key_before_the_fix_missed_the_declaration(data_root, monkeypatch):
    """Negative: with the statement before the fix, a changed declaration leaves the aerial key as it was."""
    td, _ti = _tree(data_root, "grab", decl="119.79")
    monkeypatch.setattr(CS, "sidecar_fp_inputs", _old_sidecar_fp_inputs)
    k0 = _key(td, "grab", monkeypatch)
    _tree(data_root, "grab", decl="119.91")
    assert _key(td, "grab", monkeypatch) == k0


def test_another_datasets_key_keeps_its_bytes(data_root, monkeypatch):
    td_g, _ = _tree(data_root, "grab", decl="119.79")
    td_t, _ = _tree(data_root, "taco", decl="119.79")
    kg, kt = _key(td_g, "grab", monkeypatch), _key(td_t, "taco", monkeypatch)
    _tree(data_root, "grab", decl="119.91")
    assert _key(td_t, "taco", monkeypatch) == kt
    assert _key(td_g, "grab", monkeypatch) != kg


def test_a_desk_trees_key_does_not_read_the_declaration(data_root, monkeypatch):
    td, _ = _tree(data_root, "grab", decl="119.79")
    k0 = _key(td, "grab", monkeypatch, scene="table")
    _tree(data_root, "grab", decl="119.91")
    assert _key(td, "grab", monkeypatch, scene="table") == k0
