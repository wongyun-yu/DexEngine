"""Reach map inputs (store module doc 'Inputs'): the builder records the wrist-cache set's content digest, every
reader recomputes it and dies on a difference or no record; the only exception is a map named as a locked input
(DEXENGINE_POOL_LOCK), which prints `[POOL] locked=<id> stale_inputs=…`. The clearance sidecar is tied to its map and
shard the same way, the judge writes the map ids into its plan lines and the tree's record is compared back."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dexengine.placement.reach_map import build as RB
from dexengine.placement.reach_map import clearance as CLR
from dexengine.placement.reach_map import store as PM

ARM, HAND, SIDE = "rby1", "sharpa_wave", "right"
BASE, PALM = (0.1, 0.0, 0.8), (0.0, 0.01, 0.02)
IDENT = {"ruler": "r1", "collider_digest": "c1"}


def _wrists(d, n=4):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.savez(d / f"clip{i:02d}.npz", positions=np.full((5, 3), float(i)), quaternions=np.ones((5, 4)))
    return d


def _pool(tmp_path, wrist_dir, *, record=True, nz=2):
    pdir = tmp_path / "pool" / f"{ARM}_{HAND}_{SIDE}"
    pdir.mkdir(parents=True)
    extra = {}
    if record:
        dig, n = PM.wrist_inputs(wrist_dir)
        extra = {PM.INPUTS_KEY: np.str_(dig), PM.INPUTS_N_KEY: np.int64(n)}
    np.savez(
        pdir / "meta.npz",
        axes_x=np.arange(3.0),
        axes_y=np.arange(3.0),
        axes_z=np.arange(float(nz)),
        atoms_xyzw=np.ones((2, 4)),
        pos_step=0.035,
        base_xyz=np.asarray(BASE),
        desk_height=0.7,
        palm_off=np.asarray(PALM),
        arm=ARM,
        hand=HAND,
        side=SIDE,
        **extra,
    )
    for k in range(nz):
        np.savez(pdir / f"shard_z{k:03d}.npz", vq=np.full((9, 2, 1, 7), float(k), np.float16))
    return pdir


def _verify(pdir, wrist_dir):
    return PM.verify(pdir, arm=ARM, hand=HAND, side=SIDE, base_xyz=BASE, palm_off=PALM, wrist_dir=wrist_dir, who="t")


def _lock(monkeypatch, pdir, sha=None):
    pid = PM.pool_id(pdir)
    monkeypatch.setenv(PM.LOCK_ENV, pid if sha is None else f"{pdir.name}@{sha}")
    return pid


def test_the_lock_setting_is_a_registered_process_variable():
    from dexengine.core import config

    assert PM.LOCK_ENV in config.ENV_VARS


def test_a_map_built_from_the_current_wrist_set_passes(tmp_path, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    _verify(pdir, wd)
    out = capsys.readouterr().out
    assert f"inputs={PM.wrist_inputs(wd)[0][:12]} id={PM.pool_id(pdir)} OK" in out
    assert "[POOL] locked" not in out


def test_one_changed_wrist_cache_kills_every_reader(tmp_path):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    np.savez(wd / "clip01.npz", positions=np.full((5, 3), 9.0), quaternions=np.ones((5, 4)))
    with pytest.raises(PM.PoolIdentityError, match="reach map inputs .* != now"):
        _verify(pdir, wd)


def test_an_added_or_removed_wrist_cache_also_kills(tmp_path):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    (wd / "clip03.npz").unlink()
    with pytest.raises(PM.PoolIdentityError):
        _verify(pdir, wd)


def test_only_the_files_the_builder_reads_are_fingerprinted_with_every_name(tmp_path):
    wd = _wrists(tmp_path / "wrist", n=2 * PM.WRIST_SAMPLES + 1)  # stride 2: every second sorted file is read
    files, picked = PM.wrist_files(wd)
    assert [f.name for f in picked] == [f.name for f in files[::2]]
    pdir = _pool(tmp_path, wd)
    np.savez(wd / "clip01.npz", positions=np.full((5, 3), 9.0), quaternions=np.ones((5, 4)))  # not read
    _verify(pdir, wd)
    np.savez(wd / "clip02.npz", positions=np.full((5, 3), 9.0), quaternions=np.ones((5, 4)))  # read
    with pytest.raises(PM.PoolIdentityError, match="!= now"):
        _verify(pdir, wd)


def test_the_builder_samples_through_the_same_rule(tmp_path):
    import inspect

    assert "_PM.wrist_files(wrist_dir)[1]" in inspect.getsource(RB._sample_wrists)


def test_the_same_arrays_saved_again_are_the_same_inputs(tmp_path):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    np.savez(wd / "clip01.npz", positions=np.full((5, 3), 1.0), quaternions=np.ones((5, 4)))
    _verify(pdir, wd)


def test_a_locked_map_passes_with_stale_inputs_and_says_so(tmp_path, monkeypatch, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    np.savez(wd / "clip01.npz", positions=np.full((5, 3), 9.0), quaternions=np.ones((5, 4)))
    pid = _lock(monkeypatch, pdir)
    _verify(pdir, wd)
    cap = capsys.readouterr()
    assert f"[POOL] locked={pid} stale_inputs=yes" in cap.out
    assert f"[POOL] locked={pid} stale_inputs=yes" in cap.err
    assert f"id={pid} LOCKED" in cap.out


def test_a_map_without_an_input_record_dies_unless_locked(tmp_path, monkeypatch, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd, record=False)
    with pytest.raises(PM.PoolIdentityError, match="unrecorded"):
        _verify(pdir, wd)
    pid = _lock(monkeypatch, pdir)
    _verify(pdir, wd)
    assert f"[POOL] locked={pid} stale_inputs=yes map_inputs=unrecorded" in capsys.readouterr().out


def test_a_locked_map_passes_without_a_wrist_set_as_unknown(tmp_path, monkeypatch, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    empty = tmp_path / "no_wrist"
    empty.mkdir()
    with pytest.raises(PM.WristSetMissing, match="input set is empty"):  # no lock, no set: dies
        _verify(pdir, empty)
    pid = _lock(monkeypatch, pdir)
    _verify(pdir, empty)
    assert f"[POOL] locked={pid} stale_inputs=unknown (no wrist set)" in capsys.readouterr().out


def test_a_lock_naming_the_dir_with_another_meta_dies(tmp_path, monkeypatch):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd, record=False)
    _lock(monkeypatch, pdir, sha="0" * 12)
    with pytest.raises(PM.PoolIdentityError, match="names another map"):
        _verify(pdir, wd)


@pytest.mark.parametrize(
    "bad", ["rby1_sharpa_wave_right", "rby1_sharpa_wave_right@xyz", "a@0123456789ab,a@ba9876543210"]
)
def test_a_malformed_lock_dies(monkeypatch, bad):
    monkeypatch.setenv(PM.LOCK_ENV, bad)
    with pytest.raises(PM.PoolIdentityError):
        PM.locked_ids()


def test_the_tree_record_must_name_the_verified_map(tmp_path, monkeypatch, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    PM.check_recorded(pdir, PM.pool_id(pdir), who="t")
    with pytest.raises(PM.PoolIdentityError, match="another map"):
        PM.check_recorded(pdir, f"{pdir.name}@{'0' * 12}", who="t")
    with pytest.raises(PM.PoolIdentityError, match="records no reach map id"):
        PM.check_recorded(pdir, None, who="t")
    pid = _lock(monkeypatch, pdir)
    PM.check_recorded(pdir, None, who="t")
    assert f"[POOL] locked={pid} tree_record=unrecorded" in capsys.readouterr().out


def test_the_builder_does_not_resume_over_shards_of_another_input_set(tmp_path):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    RB._inputs_guard(pdir, PM.wrist_inputs(wd))  # same inputs: resume
    np.savez(wd / "clip02.npz", positions=np.full((5, 3), 7.0), quaternions=np.ones((5, 4)))
    with pytest.raises(SystemExit, match="another wrist-cache set"):
        RB._inputs_guard(pdir, PM.wrist_inputs(wd))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    RB._inputs_guard(fresh, PM.wrist_inputs(wd))


def _sidecars(pdir, nz=2, source=True):
    pid = PM.pool_id(pdir)
    for k in range(nz):
        src = CLR.layer_source(pdir, k, pid) if source else {}
        np.savez(CLR.sidecar_path(pdir, k), vclr_arm_cap=np.zeros((9, 2, 1), np.float16), **IDENT, **src)


@pytest.fixture
def ident(monkeypatch):
    monkeypatch.setattr(CLR, "sidecar_identity", lambda arm: dict(IDENT))


def test_a_sidecar_measured_on_this_map_and_shard_is_usable(tmp_path, ident, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    _sidecars(pdir)
    assert CLR.sidecar_problem(pdir, ARM) is None
    CLR.check_sidecar(pdir, ARM)
    assert f"pool={PM.pool_id(pdir)}" in capsys.readouterr().out


def test_an_old_sidecar_never_attaches_to_a_rebuilt_shard(tmp_path, ident):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    _sidecars(pdir)
    np.savez(pdir / "shard_z001.npz", vq=np.full((9, 2, 1, 7), 5.0, np.float16))  # the shard rebuilt in place
    assert "measured on another map or shard 1" in CLR.sidecar_problem(pdir, ARM)
    with pytest.raises(SystemExit, match="CLRCAP_ID"):
        CLR.check_sidecar(pdir, ARM)


def test_an_old_sidecar_never_attaches_to_a_rebuilt_map(tmp_path, ident):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    _sidecars(pdir)
    np.savez(wd / "clip00.npz", positions=np.full((5, 3), 3.0), quaternions=np.ones((5, 4)))
    old = {k: CLR.sidecar_path(pdir, k).read_bytes() for k in range(2)}
    (pdir / "meta.npz").unlink()
    for f in pdir.glob("*.npz"):
        f.unlink()
    pdir.rmdir()
    pdir = _pool(tmp_path, wd)  # rebuilt from the new set: the same shard bytes, another meta
    for k, b in old.items():
        CLR.sidecar_path(pdir, k).write_bytes(b)
    assert "measured on another map or shard 2" in CLR.sidecar_problem(pdir, ARM)


def test_a_sidecar_without_a_source_is_read_only_on_a_locked_map(tmp_path, ident, monkeypatch, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd, record=False)
    _sidecars(pdir, source=False)
    assert "without a source on a map that is not a locked input 2" in CLR.sidecar_problem(pdir, ARM)
    _lock(monkeypatch, pdir)
    assert CLR.sidecar_problem(pdir, ARM) is None
    CLR.check_sidecar(pdir, ARM)
    assert "source=unrecorded on 2 layers (locked input)" in capsys.readouterr().out


def test_the_judge_plan_line_carries_the_verified_map_ids():
    import io
    import types

    from dexengine.placement.judge.common import _PlanWriter

    fh = io.StringIO()
    ids = {"right": "m_right@0123456789ab", "left": "m_left@ba9876543210"}
    w = _PlanWriter(fh, types.SimpleNamespace(exhaust_max=3, active_sides="both"), {}, ids)
    w.write(json.dumps({"clip": "c", "anchor": 0}))
    assert json.loads(fh.getvalue())["pool"] == ids


def test_the_tree_record_round_trips_through_one_owner():
    ti = {"provenance": {"mano_dir": "m"}}
    PM.record_into(ti, None)
    assert ti == {"provenance": {"mano_dir": "m"}} and PM.recorded(ti, "right") is None
    PM.record_into(ti, {"right": "a@0123456789ab", "left": "b@ba9876543210"})
    assert ti["provenance"]["mano_dir"] == "m"
    assert PM.recorded(ti, "right") == "a@0123456789ab" and PM.recorded(ti, "left") == "b@ba9876543210"


def test_the_bake_and_the_coarse_path_use_the_record_owner():
    import inspect

    from dexengine.placement.bake import combo
    from dexengine.placement.coarse_path import dp

    assert '_PM.record_into(tio, ad.get("pool"))' in inspect.getsource(combo.cmd_bake)
    assert "_PM.check_recorded(pd, _PM.recorded(_ti, a.side)" in inspect.getsource(dp._reach_map)


def _coarse(tmp_path, pdir, wd, ids):
    import types

    from dexengine.placement.coarse_path import dp

    tree = tmp_path / "tree"
    tree.mkdir(exist_ok=True)
    ti = {"provenance": {}}
    PM.record_into(ti, ids)
    (tree / "task_info.json").write_text(json.dumps(ti))
    a = types.SimpleNamespace(pool_root=str(pdir.parent), arm=ARM, hand=HAND, side=SIDE, tree=str(tree), sequence="c")
    return dp._reach_map(a, f"{ARM}_{HAND}", BASE, PALM, wd, "table", 0.7, np.asarray(BASE))


def test_the_coarse_path_dies_on_another_or_no_recorded_map(tmp_path, monkeypatch, capsys):
    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd)
    pid = PM.pool_id(pdir)
    assert _coarse(tmp_path, pdir, wd, {SIDE: pid})[0] == pdir
    with pytest.raises(PM.PoolIdentityError, match="another map"):
        _coarse(tmp_path, pdir, wd, {SIDE: f"{pdir.name}@{'0' * 12}"})
    with pytest.raises(PM.PoolIdentityError, match="records no reach map id"):
        _coarse(tmp_path, pdir, wd, None)
    _lock(monkeypatch, pdir)
    _coarse(tmp_path, pdir, wd, None)
    assert f"[POOL] locked={pid} tree_record=unrecorded who=stage_dp" in capsys.readouterr().out


def test_the_stage1_status_note_says_how_to_lock_a_map_that_is_not_ready(tmp_path, monkeypatch):
    from dexengine.data.stage1 import status as ST

    wd = _wrists(tmp_path / "wrist")
    pdir = _pool(tmp_path, wd, record=False)
    monkeypatch.setattr(ST.RR, "active_arm", lambda: ARM)
    monkeypatch.setattr(CLR, "sidecar_problem", lambda d, arm: "sidecar layers 0/2")
    monkeypatch.setattr(ST, "POOL_NOTES", {})
    assert ST._pool_ready(pdir.parent, f"{ARM}_{HAND}", SIDE) is False
    assert (
        ST.POOL_NOTES[pdir.name]
        == f"sidecar layers 0/2 — to use it as is: {PM.LOCK_ENV}={PM.pool_id(pdir)} (else rebuild)"
    )
    monkeypatch.setattr(CLR, "sidecar_problem", lambda d, arm: None)
    assert ST._pool_ready(pdir.parent, f"{ARM}_{HAND}", SIDE) is True
    assert ST._pool_ready(pdir.parent, f"{ARM}_{HAND}", "left") is False  # no map: no note
    assert list(ST.POOL_NOTES) == [pdir.name]
