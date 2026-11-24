"""Camera-only load of the original (clip_phases.orig_camera -> episode.load_traj camera_only): the idle hand is not
measured, the marker line says so (session log stays valid), nothing is cached, and such a context cannot be drawn.
A drawing load still measures once per trajectory and reuses the cache. The Isaac side is stubbed: the test checks
the control flow of load_traj, not the scene.
"""

from __future__ import annotations

import sys
import types

import pytest

from dexengine.record.render import clip_phases as RCP
from dexengine.record.render import clip_session as RCL
from dexengine.record.render import episode as CR
from dexengine.record.render import framing as CF
from dexengine.record.render import idle as IDLE
from dexengine.record.render import markers as PM
from dexengine.record.scene import extras_apply as EXV
from dexengine.record.scene import supports as SUP


class _Rig:
    def desk_local(self):
        return {}

    def calibrate(self, groups):
        return {"cal": 1}

    def pose(self, name):
        return ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


@pytest.fixture
def stubbed(monkeypatch):
    calls = []

    def fake_bind(S, k, clip, path, tm, hv):
        return {
            "clip": clip,
            "traj": path,
            "write": lambda t: None,
            "steps": [0, 5, 10],
            "body": {},
            "binds": [],
            "tf": None,
            "boxes": {},
            "groups": {},
            "per": {},
            "P": [[0.0]],
        }

    def fake_measure(raw_env, binds, tf, desk, steps, write, boxes):
        calls.append(list(steps))
        print(f"{IDLE.MARK} idle hand ['left'] · measured", flush=True)
        return {"sides": ["left"], "n": len(calls)}

    monkeypatch.setattr(CR, "_bind", fake_bind)
    monkeypatch.setattr(CR, "_tree_dir", lambda args_cli, clip: "/x")
    monkeypatch.setattr(IDLE, "measure", fake_measure)
    monkeypatch.setattr(RCL, "prims", lambda: {"base"})
    monkeypatch.setattr(RCL, "cams_off", lambda rig: [])
    monkeypatch.setattr(RCL, "check_view", lambda rig, groups: {"ok": True})
    monkeypatch.setattr(SUP, "show", lambda *a: {"visible": 0})
    monkeypatch.setattr(PM, "show", lambda *a: None)
    monkeypatch.setattr(EXV, "apply", lambda *a: {"placed": []})
    monkeypatch.setattr(CF, "exo_cam", lambda cal: ((0.0, 0.0, 1.0),))
    aoc = types.ModuleType("dexengine.rl.tasks.scene.all_obj_collision")
    aoc.place_all_obj_bodies = lambda *a: None
    aoc.assert_all_obj_ready = lambda *a: None
    monkeypatch.setitem(sys.modules, aoc.__name__, aoc)
    S = types.SimpleNamespace(
        env=None, rig=_Rig(), idle={}, prims0=None, cal=None, off=None, look={}, args_cli=None, k=0
    )
    return S, calls


def _tm():
    return dict.fromkeys(CR.TM_KEYS, 0.0)


def test_camera_only_skips_and_says_so(stubbed, capsys):
    S, calls = stubbed
    T = CR.load_traj(S, 0, "orig", "/t.npz", _tm(), camera_only=True)
    out = capsys.readouterr().out
    assert calls == [] and S.idle == {} and "idle" not in T
    assert IDLE.log_problems(out) == []  # the session log keeps the marker
    assert "not measured for orig" in out and "loaded only to fix the camera" in out
    assert S.cal == {"cal": 1}  # the camera is still fixed by this load


def test_drawing_load_measures_once_and_caches(stubbed, capsys):
    S, calls = stubbed
    CR.load_traj(S, 0, "orig", "/t.npz", _tm(), camera_only=True)
    T1 = CR.load_traj(S, 0, "orig", "/t.npz", _tm())  # a later drawing load of the same trajectory measures
    T2 = CR.load_traj(S, 0, "orig", "/t.npz", _tm())  # reload: cached
    assert calls == [[0, 5]]
    assert T1["idle"] == T2["idle"] == {"sides": ["left"], "n": 1}
    assert list(T1)[-5:] == ["idle", "look", "supports", "extras", "view"]  # key order as before


def test_camera_only_context_is_not_drawable(stubbed):
    S, _ = stubbed
    T = CR.load_traj(S, 0, "orig", "/t.npz", _tm(), camera_only=True)
    with pytest.raises(RuntimeError, match="camera-only"):
        CR.render_episode(S, T, None, _tm(), None, 0.0, 0.0)


def test_orig_camera_loads_camera_only(monkeypatch):
    seen = {}

    def fake_load(S, k, clip, path, tm, camera_only=False):
        seen.update(clip=clip, camera_only=camera_only)
        return {"groups": {}, "grasp": None}

    monkeypatch.setattr(CR, "load_traj", fake_load)
    monkeypatch.setattr(CR, "ego_view_meta", lambda cal, view, grasp: {"ev": 1})
    monkeypatch.setattr(RCP, "check_view", lambda rig, groups: None)
    monkeypatch.setattr(RCP, "restore", lambda raw_env, T, prims0: {"oracle": 1})
    S = types.SimpleNamespace(rig=None, cal=None, prims0=None)
    plan = {"trajs": [{"name": "o", "traj": "/o.npz"}], "random_per_traj": 0, "out": "/x"}
    summ = {"prim_oracle": []}
    ev, LR = RCP.orig_camera(S, plan, ["o"], summ, None, measure=False)
    assert seen == {"clip": "o", "camera_only": True} and ev == {"ev": 1} and LR is None
