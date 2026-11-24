"""Joint-value sidecars stay on qpos's time axis through both steps that change it — the f0 pad (prepend P frames) and
the retiming resample — for every stated sidecar (sided and background articulated objects), and a stated sidecar
that is missing or off the axis dies. Shapes: sided key · background key (side None) · no articulated object."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dexengine.core import qpos_layout as QL

H = 8  # hand block: pos 3 · rot 3 · 2 fingers
NQ = 2 * H + 7 + 7 + 1 + 1  # two hands · main object · background object · two joint columns
COL_R, COL_BG = 2 * H + 14, 2 * H + 15
KEY_R, KEY_BG = "qpos_obj_arti_right", "qpos_obj_arti_supp_0"


def _objects(sided=True, background=True):
    obj = [
        {
            "name": "right_object",
            "role": "main",
            "side": "right",
            "obj_id": "box",
            "mesh_dir": "objects/box",
            "qpos_start": 2 * H,
            "articulation": {"qpos_index": COL_R, "npz_key": KEY_R} if sided else None,
        },
        {
            "name": "supp_0",
            "role": "extra",
            "side": None,
            "qpos_start": 2 * H + 7,
            "articulation": {"qpos_index": COL_BG, "npz_key": KEY_BG} if background else None,
        },
    ]
    return obj


def _task_info(sided=True, background=True) -> dict:
    lay = [
        {"kind": "hand", "side": "right", "start": 0, "width": H},
        {"kind": "hand", "side": "left", "start": H, "width": H},
        {"kind": "object", "name": "right_object", "start": 2 * H, "width": 7},
        {"kind": "object", "name": "supp_0", "start": 2 * H + 7, "width": 7},
        {"kind": "joint", "object": "right_object", "start": COL_R, "width": 1},
        {"kind": "joint", "object": "supp_0", "start": COL_BG, "width": 1},
    ]
    return {"qpos_layout": lay, "nq": NQ, "hand_qpos_dim": H, "objects": _objects(sided, background)}


def _tree(tmp_path, name, **kw):
    d = tmp_path / name
    d.mkdir()
    (d / "task_info.json").write_text(json.dumps(_task_info(**kw)))
    return d


def _qpos(T, rng):
    q = np.zeros((T, NQ))
    q[:, :3] = rng.normal(0, 0.05, (T, 3))
    q[:, H : H + 3] = rng.normal(0, 0.05, (T, 3))
    for s in (2 * H, 2 * H + 7):
        q[:, s : s + 3] = rng.normal(0, 0.05, (T, 3))
        q[:, s + 3] = 1.0  # identity quaternion (wxyz)
    q[:, COL_R] = np.linspace(0.0, 1.2, T)
    q[:, COL_BG] = np.sin(np.linspace(0.0, 3.0, T))
    return q


def _npz(q, keys=(KEY_R, KEY_BG)):
    z = {"qpos": q, "qvel": np.zeros_like(q)}
    for k, c in ((KEY_R, COL_R), (KEY_BG, COL_BG)):
        if k in keys:
            z[k] = q[:, c : c + 1].copy()  # the sidecar = a copy of its joint column
    return z


@pytest.fixture
def W(monkeypatch):
    """The f0 writer module (its import reads the run's hand and arm)."""
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    from dexengine.retarget.f0 import repair_write

    return repair_write


def _retime(z, lay, stretch=1.2):
    from dexengine.placement import retime_variants as MCV

    T = len(z["qpos"])
    s = np.full(T - 1, stretch)
    tau = np.concatenate([[0.0], np.cumsum(s * MCV.DT)])
    return MCV.resample_all(z, tau, s, H, layout=lay, root_maps=lambda: pytest.fail("root maps read"))


def test_artic_npz_keys_is_every_stated_key_in_order(tmp_path):
    """The owner lists sided and background keys, statement order, once; a record without a key is left out."""
    ti = _task_info()
    ti["objects"].append(dict(ti["objects"][1], name="supp_1", articulation={"qpos_index": COL_BG, "npz_key": None}))
    ti["objects"].append(dict(ti["objects"][1], name="supp_2", articulation={"qpos_index": COL_BG, "npz_key": KEY_BG}))
    (tmp_path / "t").mkdir()
    (tmp_path / "t/task_info.json").write_text(json.dumps(ti))
    lay = QL.read(str(tmp_path / "t"), nq=NQ, use_cache=False)
    assert lay.artic_npz_keys() == [KEY_R, KEY_BG]
    assert set(lay.artic) == {"right"}  # the per-side table alone would miss the background key


@pytest.mark.parametrize(
    ("sided", "background"), [(True, True), (True, False), (False, True)], ids=["both", "sided", "background"]
)
def test_resample_moves_every_sidecar_with_qpos(tmp_path, sided, background):
    """Sided and background sidecars come out on the new time axis, equal to the resampled joint column."""
    lay = QL.read(str(_tree(tmp_path, "t", sided=sided, background=background)), nq=NQ, use_cache=False)
    keys = tuple(k for k, on in ((KEY_R, sided), (KEY_BG, background)) if on)
    out = _retime(_npz(_qpos(40, np.random.default_rng(0)), keys), lay)
    T_new = len(out["qpos"])
    assert T_new > 40
    for k, c in ((KEY_R, COL_R), (KEY_BG, COL_BG)):
        if k in keys:
            assert out[k].shape == (T_new, 1)
            assert np.array_equal(out[k][:, 0], out["qpos"][:, c])


def test_resample_without_articulated_objects_passes_other_keys(tmp_path):
    """No articulated object: nothing is required, other arrays pass through untouched."""
    lay = QL.read(str(_tree(tmp_path, "t", sided=False, background=False)), nq=NQ, use_cache=False)
    z = _npz(_qpos(30, np.random.default_rng(1)), keys=())
    z["extra_note"] = np.arange(3)
    out = _retime(z, lay)
    assert lay.artic_npz_keys() == []
    assert np.array_equal(out["extra_note"], np.arange(3))


@pytest.mark.parametrize("defect", ["missing", "short", "flat"])
def test_resample_dies_on_a_stated_sidecar_off_the_axis(tmp_path, defect):
    """Negative: a stated sidecar that is missing, shorter than qpos (an unpadded f0 source) or 1-D dies — before
    the fix the per-side loop skipped the background key and every length mismatch silently."""
    lay = QL.read(str(_tree(tmp_path, "t")), nq=NQ, use_cache=False)
    z = _npz(_qpos(40, np.random.default_rng(2)))
    if defect == "missing":
        del z[KEY_BG]
    elif defect == "short":
        z[KEY_BG] = z[KEY_BG][20:]
    else:
        z[KEY_BG] = z[KEY_BG][:, 0]
    with pytest.raises(ValueError, match=r"\[ARTIC\]"):
        _retime(z, lay)


def test_f0_pad_extends_every_sidecar_with_frame_zero(tmp_path, W, capsys):
    """The f0 pad holds every stated sidecar at frame 0 for P frames, like the qpos pad rows (blend = qpos[0]), and
    prints its marker."""
    d = _tree(tmp_path, "t")
    q = _qpos(30, np.random.default_rng(3))
    out = _npz(q)
    W._pad_artic_sidecars(out, d, 20, 30)
    assert f"[ARTIC_PAD] t: joint sidecars {[KEY_R, KEY_BG]} +20 frames" in capsys.readouterr().out
    padded = np.vstack([np.repeat(q[0:1], 20, axis=0), q])
    for k, c in ((KEY_R, COL_R), (KEY_BG, COL_BG)):
        assert out[k].shape == (50, 1) and np.array_equal(out[k][:, 0], padded[:, c])


@pytest.mark.parametrize("defect", ["missing", "short"])
def test_f0_pad_dies_on_a_stated_sidecar_off_the_axis(tmp_path, W, defect):
    d = _tree(tmp_path, "t")
    out = _npz(_qpos(30, np.random.default_rng(4)))
    if defect == "missing":
        del out[KEY_R]
    else:
        out[KEY_R] = out[KEY_R][5:]
    with pytest.raises(ValueError, match=r"\[ARTIC\]"):
        W._pad_artic_sidecars(out, d, 20, 30)


def test_f0_then_retime_keeps_the_sidecar_on_the_joint_column(tmp_path, W):
    """The chain: f0 pad then retiming — the sidecar equals its joint column at every output frame."""
    d = _tree(tmp_path, "t")
    q = _qpos(30, np.random.default_rng(5))
    z = _npz(q)
    z["qpos"] = np.vstack([np.repeat(q[0:1], 20, axis=0), q])
    z["qvel"] = np.zeros_like(z["qpos"])
    W._pad_artic_sidecars(z, d, 20, 30)
    out = _retime(z, QL.read(str(d), nq=NQ, use_cache=False))
    for k, c in ((KEY_R, COL_R), (KEY_BG, COL_BG)):
        assert np.array_equal(out[k][:, 0], out["qpos"][:, c])


def test_every_f0_prepend_writer_pads_the_sidecars():
    """Structure: both writers that prepend the f0 pad to the kinematic npz (in place · suffixed tree) call the one
    sidecar pad — a writer that drops the call leaves the sidecars P frames short."""
    import ast
    import inspect

    from dexengine.retarget.f0 import repair_write

    tree = ast.parse(inspect.getsource(repair_write))
    calls = {
        fn.name: {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef)
    }
    assert {n for n, c in calls.items() if "_pad_artic_sidecars" in c} == {"_write_trees", "write_outputs"}
