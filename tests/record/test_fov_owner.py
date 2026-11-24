"""Off-screen ownership of the original placement (render.fov):
the canonical-cell stop rule, the row-0 choice, the plan field and the owner list."""

from __future__ import annotations

import contextlib
import io
import random
import types

import pytest

from dexengine.record import episodes as EPI
from dexengine.record.augment import policy as AP
from dexengine.record.augment import stage as AST
from dexengine.record.render import fov as FOW

ns = types.SimpleNamespace


def _o(gv: bool) -> dict:
    return {"name": "o", "role": "orig", "rank": 0, "grasp_visible": gv, "vis_strict": True, "vis_in_frame": 1.0}


def _m(*st) -> dict:
    drop = {"k": 0, "status": "reject", "why": "judge", "detail": "full"}
    return {"variants": [drop] + [{"k": i + 1, "status": x} for i, x in enumerate(st)]}


def _stop(m, nm, fov):
    return FOW.canon_stop(m, AST.canon_problem(m), nm, fov)


V1 = {
    "name": "v1",
    "role": "aug",
    "rank": 1,
    "grasp_visible": True,
    "vis_strict": True,
    "vis_in_frame": 1.0,
    "vis_ladder": {"pass": {lv["name"]: True for lv in FOW.VIS_LADDER}, "score": 1.0},
}


def test_off_screen_owner_with_a_passing_variant_does_not_stop():
    st, fo = _stop(_m("ok", "reject"), "c", frozenset({"c"}))
    assert st is None and fo


def test_off_screen_owner_without_passing_variant_stops_and_keeps_orig_row0():
    st, fo = _stop(_m("reject", "spare"), "c", frozenset({"c"}))
    r0, ch, mt = EPI.choose_rows([_o(FOW.orig_fov_decide(True, False))], 3)
    assert st and not fo and r0 == "o" and ch == []
    assert mt["orig_fov"]["row0_source"] == "orig" and mt["orig_fov"]["orig_kept"] is False
    assert EPI.choose_rows([_o(FOW.orig_fov_decide(True, False)), V1], 3)[0] == "v1"


def test_visible_clip_keeps_the_old_stop_rule():
    st, fo = _stop(_m("ok"), "c", frozenset({"other"}))
    assert st and not fo and FOW.orig_fov_decide(True, None) is True
    assert EPI.choose_rows([_o(FOW.orig_fov_decide(True, None)), V1], 3)[0] is None
    st, fo = _stop(_m("ok"), "c", frozenset())
    assert st and not fo


def test_owner_decision_rejects_bad_inputs():
    with pytest.raises(ValueError):
        FOW.orig_fov_decide(True, True)
    with pytest.raises(ValueError):
        FOW.orig_fov_decide(None, None)


def test_empty_owner_list_equals_the_old_rule():
    rng = random.Random(42)
    for _ in range(400):
        v = [{"k": i, "status": rng.choice(("ok", "reject", "spare"))} for i in range(rng.randint(0, 4))]
        m = {"variants": v}
        old = AST.canon_problem(m) or not any(x["status"] == "ok" for x in v)
        assert _stop(m, "c", frozenset()) == ((AST.canon_problem(m) or "") if old else None, False)


def test_plan_field_only_when_owned():
    tr = [{"name": "o_pl_rt", "traj": "/t/o", "role": "orig"}, {"name": "o_pv01_pl_rt", "traj": "/t/v", "role": "aug"}]
    deck = EPI.deck_args(AP.PER_CLIP)
    p0 = EPI.plan("o_pl_rt", "/out", "/s", "/l", tr, "aug", deck, n_choose=AP.PER_CLIP)
    p1 = EPI.plan("o_pl_rt", "/out", "/s", "/l", tr, "aug", deck, n_choose=AP.PER_CLIP, orig_fov_owner=None)
    p2 = EPI.plan("o_pl_rt", "/out", "/s", "/l", tr, "aug", deck, n_choose=AP.PER_CLIP, orig_fov_owner=False)
    assert p0 == p1 and "orig_fov_owner" not in p0["choose"]
    assert p2["choose"]["orig_fov_owner"] is False and p2["plan_sha"] != p0["plan_sha"]
    with pytest.raises(ValueError):
        EPI.plan("o_pl_rt", "/out", "/s", "/l", tr[:1], "orig", deck, orig_fov_owner=False)


def test_owner_set_marker_once_and_inputs_checked():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a1 = ns(orig_fov_out="a_pl_rt", round="aug", clips="a_pl_rt,b")
        got, again = FOW.owner_set(a1), FOW.owner_set(a1)
        none = FOW.owner_set(ns(orig_fov_out="", round="orig", clips="a"))
    assert got == {"a"} and again is got and none == frozenset()
    assert buf.getvalue().count(FOW.ORIG_FOV_OWNER_MARK) == 1
    with pytest.raises(SystemExit):
        FOW.owner_set(ns(orig_fov_out="z", round="aug", clips="a"))
    with pytest.raises(SystemExit):
        FOW.owner_set(ns(orig_fov_out="a", round="orig", clips="a"))


def test_gate_records_the_canonical_problem():
    c = {"pre": 2}
    with contextlib.redirect_stdout(io.StringIO()):
        st, fo = FOW.gate(
            ns(orig_fov_out="o_pl_rt", round="aug", clips="o_pl_rt"),
            {"variants": [{"k": 0, "status": "reject"}, {"k": 1, "status": "ok"}]},
            c,
            "o_pl_rt",
            "x",
        )
    assert st is None and fo and c["canon_fovout"] == "x"


def test_row0_owner_list_outside_spec_mode_is_empty():
    from dexengine.record import driver_spec as DS

    row = {"round": "orig", "expect": [["a_pl_rt", "<aug 0>", "D"]]}
    assert DS.row0_of(ns(spec_row=row), ["a_pl_rt"]) == [] and DS.row0_of(ns(), ["a_pl_rt"]) == []
