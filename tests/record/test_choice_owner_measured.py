"""Trajectory choice record under an off-screen owner (render.fov): the record keeps the original's own measured
value even after variants were probed (render.clip_phases.choose_phase), and the pilot check (record.driver_check.
choice_problems) compares the original's row measurement to measurement (render.fov.choice_expect),
plus the record fix behind it."""

from __future__ import annotations

import inspect
import types

from dexengine.record import driver_check as DC
from dexengine.record import episodes as EPI
from dexengine.record.render import clip_phases as RP
from dexengine.record.render import episode as CR
from dexengine.record.render import fov as FOW

ns = types.SimpleNamespace
LADDER_ALL = {"pass": {lv["name"]: True for lv in FOW.VIS_LADDER}, "score": 1.0}


def _o(gv: bool) -> dict:
    return {"name": "o", "role": "orig", "rank": 0, "grasp_visible": gv, "vis_strict": True, "vis_in_frame": 1.0}


def _v(name: str, rank: int) -> dict:
    return {
        "name": name,
        "role": "aug",
        "rank": rank,
        "grasp_visible": True,
        "vis_strict": True,
        "vis_in_frame": 1.0,
        "vis_ladder": LADDER_ALL,
    }


def _tc(measured, owner, cands: list[dict]) -> dict:
    _r0, _ch, mt = EPI.choose_rows(cands, 3)
    tc = dict(mt, unprobed=[])
    if owner is not None:
        tc["orig_fov"] = dict(tc["orig_fov"], owner=owner, measured=measured)
    return tc


def _ep(name: str, gv, vs=True) -> dict:
    return {"clip": name, "ego_view": {"grasp_visible": gv, "vis_strict": vs}}


# ---- check side -----------------------------------------------------------------------------------------------------
def test_owner_original_row_is_compared_to_its_measurement():
    """A measured hot3d clip: owner off screen, the session measured visible, 0 variants (row 0 = original)."""
    tc = _tc(True, False, [_o(FOW.orig_fov_decide(True, False))])
    eps = [_ep("o", True)] * 5
    assert tc["cands"][0]["grasp_visible"] is False and tc["row0"] == "o"
    assert DC.choice_problems("c", tc, ["o"], eps) == []


def test_render_differs_from_the_measurement():
    tc = _tc(True, False, [_o(False)])
    assert DC.choice_problems("c", tc, ["o"], [_ep("o", False)] + [_ep("o", True)] * 4)


def test_render_vis_strict_differs():
    tc = _tc(True, False, [_o(False)])
    assert DC.choice_problems("c", tc, ["o"], [_ep("o", True, False)] + [_ep("o", True)] * 4)


def test_judgment_differs_from_the_owner_rule():
    tc = _tc(True, False, [_o(False)])
    bad = dict(tc, cands=[dict(tc["cands"][0], grasp_visible=True)])
    assert any("owner rule" in x for x in FOW.choice_expect("c", bad, {}))


def test_owner_record_without_a_measured_bool():
    tc = _tc(True, False, [_o(False)])
    no_meas = dict(tc, orig_fov=dict(tc["orig_fov"], measured=None))
    assert any("cannot be recomputed" in x for x in FOW.choice_expect("c", no_meas, {}))


def test_owner_record_without_an_original_candidate():
    tc = _tc(True, False, [_o(False)])
    assert FOW.choice_expect("c", dict(tc, cands=[]), {}) == ["c: owner record without an original candidate"]


def test_variants_probed_the_original_expectation_is_its_measurement():
    """With variants probed the record's measured is still the original's (record fix), so the expectation is set."""
    tv = _tc(True, False, [_o(False), _v("v1", 1)])
    vis = {}
    assert tv["row0"] == "v1" and FOW.choice_expect("c", tv, vis) == [] and vis == {"o": (True, True)}


def test_no_owner_keeps_the_old_path():
    old = _tc(False, None, [_o(False)])
    assert FOW.choice_expect("c", old, {}) == []
    assert DC.choice_problems("c", old, ["o"], [_ep("o", False)]) == []
    assert DC.choice_problems("c", old, ["o"], [_ep("o", True)] * 5)


def test_choice_problems_swaps_before_comparing():
    src = inspect.getsource(DC.choice_problems)
    assert "FOW.choice_expect(" in src and src.index("FOW.choice_expect(") < src.index("off = ")


# ---- record side ----------------------------------------------------------------------------------------------------
def _choose(monkeypatch, orig_visible: bool, variants: list[bool]) -> dict:
    """choose_phase with the camera head and the variant probe faked: the original measures orig_visible, variant i
    measures variants[i]. Returns summ['choice']."""
    ev0 = {"grasp_visible": orig_visible, "vis_strict": orig_visible, "vis_in_frame": 1.0}
    monkeypatch.setattr(RP, "orig_camera", lambda *a, **k: (dict(ev0), None))
    probe = {f"v{i + 1}": gv for i, gv in enumerate(variants)}

    def fake_probe(_S, _i, name, _traj):
        gv = probe[name]
        ladder = {"pass": {lv["name"]: gv for lv in FOW.VIS_LADDER}, "score": 1.0 if gv else 0.0}
        return {"grasp_visible": gv, "vis_strict": gv, "vis_in_frame": 1.0, "vis_ladder": ladder}

    monkeypatch.setattr(CR, "probe_view", fake_probe)
    trajs = [{"name": "o", "traj": "o.npz"}] + [{"name": n, "traj": f"{n}.npz"} for n in probe]
    plan = {"trajs": trajs, "choose": {"n": 3, "orig_fov_owner": False}, "deck": {"n_rows": 4}}
    summ: dict = {}
    RP.choose_phase(ns(), plan, [t["name"] for t in trajs], summ, None)
    return summ["choice"]


def test_record_keeps_the_original_measurement_after_probing_variants(monkeypatch):
    """The original measured visible; the last probed variant measured not visible. The old code wrote the last
    variant's value into orig_fov.measured."""
    ch = _choose(monkeypatch, True, [True, True, True, False])
    assert [c["name"] for c in ch["cands"]] == ["o", "v1", "v2", "v3", "v4"]
    assert ch["orig_fov"]["owner"] is False and ch["orig_fov"]["measured"] is True
    assert ch["cands"][0]["grasp_visible"] is False  # the owner's judgment stays in the candidate row


def test_record_measured_not_visible_original(monkeypatch):
    ch = _choose(monkeypatch, False, [True, True, True, True])
    assert ch["orig_fov"]["measured"] is False
