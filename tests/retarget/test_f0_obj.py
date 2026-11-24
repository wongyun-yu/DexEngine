"""The f0 object repair (repair_obj): its switch, measure, gate, solution pick, status line and the keys it moves."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0 import repair_obj as RO
from tests.imports import own_module


def _cube(c, h):
    """The 8 corners of an axis-aligned cube (centre c, half size h)."""
    return np.array(
        [[c[0] + sx * h, c[1] + sy * h, c[2] + sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )


@pytest.mark.parametrize("v,on", [(None, True), ("1", True), ("0", False)])
def test_switch_default_on_and_off(monkeypatch, v, on):
    if v is None:
        monkeypatch.delenv("DEXENGINE_F0_OBJ_REPAIR", raising=False)
    else:
        monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", v)
    assert RO.obj_repair_on() is on


def test_unknown_switch_value_dies(monkeypatch):
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "yes")
    with pytest.raises(SystemExit):
        RO.obj_repair_on()


def test_sat_depth_of_two_cubes():
    obj = RO.CookedObject([_cube((0.0, 0.0, 0.0), 0.05)])
    hand = [("tip", [_cube((0.095, 0.0, 0.0), 0.05)])]  # 5 mm into the object along x
    d, ln = RO.sat_depth(hand, obj, np.eye(3), np.zeros(3))
    assert ln == "tip" and d == pytest.approx(0.005, abs=1e-9)
    apart = [("tip", [_cube((0.2, 0.0, 0.0), 0.05)])]
    assert RO.sat_depth(apart, obj, np.eye(3), np.zeros(3)) == (0.0, "")


@pytest.mark.parametrize(
    "sat_mm,desk_deep,switch,opened",
    [(8.0, False, "1", True), (4.0, False, "1", False), (8.0, True, "1", False), (8.0, False, "0", False)],
)
def test_gate(monkeypatch, capsys, sat_mm, desk_deep, switch, opened):
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", switch)
    rep = {"side": "right"}
    assert RO.gate(rep, (sat_mm / 1000.0, "right_tip"), "", desk_deep) is opened
    assert rep["obj_repair"] is opened and rep["obj_sat_mm"] == sat_mm
    line = capsys.readouterr().out
    assert line.startswith(RO.MARK) and ("REPAIR" if opened else "keep") in line


def test_gate_not_measured_says_why(monkeypatch, capsys):
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "1")
    rep = {"side": "left"}
    assert RO.gate(rep, None, "no cooked hull (no USD)", False) is False
    assert rep["obj_sat_mm"] is None and "n/a: no cooked hull (no USD)" in capsys.readouterr().out


def test_failed_checks_name_the_criterion_value_and_threshold():
    assert RO.failed_checks(0.0015, "tip", 0.0, 0.0) == ""
    assert RO.failed_checks(0.00839, "tip", 0.0, 0.0) == "SAT 8.39mm > 2mm (tip)"
    assert RO.failed_checks(0.00018, "tip", 0.0009, 0.0) == ""  # sub-mm deepening is within DESK_DEEPER_MM
    assert RO.failed_checks(0.00018, "tip", 0.0012, 0.0) == "desk 1.20mm > start 0.00mm + 1mm (deeper)"
    assert " · " in RO.failed_checks(0.003, "tip", 0.004, 0.001)


def test_release_check_is_the_hold_frame(monkeypatch):
    monkeypatch.setattr(RO, "window_frames", lambda: 13)
    assert RO.release_frame() == 13 - RO.OBJ_WINDOW_MARGIN
    sats = [1.0] * 13
    assert RO.release_failed(sats) == ""
    sats[10] = 5.52
    assert RO.release_failed(sats) == "release frame 10 SAT 5.52mm ≥ 5mm"
    assert RO.release_failed([1.0, 1.0, 6.0]) == "release frame 2 SAT 6.00mm ≥ 5mm"  # a clip shorter than the hold


@pytest.mark.parametrize("frame", [9, 10])
def test_release_check_holds_a_deep_last_pinned_or_first_free_frame(monkeypatch, frame):
    """A face ≥ 5 mm at only the last pinned frame (9) or only the first free frame (10) fails the check; the frames
    around them do not count."""
    monkeypatch.setattr(RO, "window_frames", lambda: 13)
    sats = [1.0] * 13
    sats[frame] = 5.2
    assert RO.release_failed(sats) == f"release frame {frame} SAT 5.20mm ≥ 5mm"
    for other in (8, 11):
        quiet = [1.0] * 13
        quiet[other] = 9.0
        assert RO.release_failed(quiet) == ""


@pytest.mark.parametrize("release_mm,repaired", [(4.9, True), (5.52, False)])
def test_a_window_repaired_only_below_the_release_depth(monkeypatch, release_mm, repaired):
    """REPAIRED needs the release frame's SAT below OBJ_REPAIR_MM; otherwise the side is refused (held) with that
    reason, without more tries."""
    S = _solve_mod(monkeypatch)
    monkeypatch.setattr(RO, "window_frames", lambda: 13)
    sats = [1.0] * 10 + [release_mm, 1.0, 1.0]
    monkeypatch.setattr(S, "solve_frame", lambda prob, warm, vr: (np.zeros(S.HC.nd), "", prob["s"], None))
    proj = types.SimpleNamespace(n_eval=0, lost_touch=lambda dq: [])
    probs = ({"s": s, "touch": [], "proj": proj} for s in sats)
    pose0 = (np.zeros(3), np.array([1.0, 0, 0, 0]), np.eye(3), np.zeros(S.HC.nd - 6), None, 0, slice(0, 7))
    rep = S.obj_solve_window({"side": "right"}, pose0, probs, 3, [0, 0], 0.0)
    assert (rep["status"] == "REPAIRED") is repaired
    if not repaired:
        assert rep["status"].startswith(PH.SKIP_OBJ) and "release frame 10 SAT 5.52mm ≥ 5mm" in rep["status"]


@pytest.mark.parametrize("after_mm,held", [(3.0, False), (7.2, True)])
def test_a_desk_repaired_side_goes_through_the_object_rule(capsys, after_mm, held):
    """After the desk repair the object is measured again at its result: above OBJ_REPAIR_MM the side is held with the
    value in the reason, else it stays REPAIRED; an unmeasured object leaves the report as it is."""
    rep = {"side": "left", "status": "REPAIRED", "_internal": (np.zeros(4),)}
    out = RO.after_desk(dict(rep), lambda dq: (after_mm / 1000.0, "left_thumb_tip"))
    assert out["obj_sat_after_desk_mm"] == after_mm and "after the desk repair" in capsys.readouterr().out
    assert out["status"].startswith(PH.SKIP_OBJ) is held
    if held:
        assert "object SAT after the desk repair 7.20mm > 5mm (left_thumb_tip)" in out["status"]
    assert RO.after_desk(dict(rep), None)["status"] == "REPAIRED"


def _hinged_box():
    """A 4 cm root cube at the origin and a 4 cm lid cube on a z hinge 10 cm out in x (the lid's centre 5 cm along the
    lid's x): at θ = 0 the lid sits at x 0.13–0.17, at θ = π/2 at y 0.03–0.07."""
    spec = {
        "name": "lid",
        "parent": "box",
        "joint": "box_hinge",
        "type": "hinge",
        "axis": np.array([0.0, 0.0, 1.0]),
        "anchor": np.zeros(3),
        "b_pos": np.array([0.1, 0.0, 0.0]),
        "b_R": np.eye(3),
    }
    root = RO.CookedObject([_cube((0, 0, 0), 0.02)], "d")
    return RO.ArticObject(root, [(spec, 3, RO.CookedObject([_cube((0.05, 0, 0), 0.02)], "d"))], {"joint": "x"}, "d")


def test_articulated_object_is_every_link_at_the_row():
    """An articulated object is measured on all links with the part at the row's joint value: a finger 6 mm inside the
    open lid reads 6 mm (the root alone reads 0), the same lid closed is apart; the moved planes bound the moved
    vertices."""
    art = _hinged_box()
    hand = [("right_tip", [_cube((0.1, 0.074, 0.0), 0.01)])]
    opened, closed = art.at([0, 0, 0, np.pi / 2]), art.at([0, 0, 0, 0.0])
    assert RO.sat_depth(hand, opened, np.eye(3), np.zeros(3)) == (pytest.approx(0.006), "right_tip")
    assert RO.sat_depth(hand, closed, np.eye(3), np.zeros(3)) == (0.0, "")
    assert RO.sat_depth(hand, art.root, np.eye(3), np.zeros(3)) == (0.0, "")
    for v, e, lo, hi in opened.pieces:
        side = v @ e[:, :3].T + e[:, 3]
        assert side.max() < 1e-12 and np.allclose(side.max(0), 0.0) and np.allclose([lo, hi], [v.min(0), v.max(0)])


def test_articulated_penetration_opens_the_repair_and_is_solved(monkeypatch, capsys):
    """A ≥ 5 mm finger penetration of an articulated object is measured, opens the repair and the projection moves the
    link to the surface — never passed as not measured."""
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "1")
    S = _solve_mod(monkeypatch)
    cobj = _hinged_box().at([0, 0, 0, np.pi / 2])
    hand = [("right_tip", [_cube((0.1, 0.074, 0.0), 0.01)])]
    rep = {"side": "right"}
    assert RO.gate(rep, RO.sat_depth(hand, cobj, np.eye(3), np.zeros(3)), "", False) is True
    assert "REPAIR" in capsys.readouterr().out
    nf = S.HC.nd - 6
    strict = types.SimpleNamespace(links=[("right_tip", 0, None, None)], parts={"right_tip": [_cube((0, 0, 0), 0.01)]})
    box = (np.full(nf, -9.0), np.full(nf, 9.0))
    P = S.Projection(strict, cobj, np.eye(3), np.zeros(3), (np.zeros(3), np.eye(3), np.zeros(nf)), -1.0, box, [])
    P.link_T = lambda dq: [(np.eye(3), np.array([0.1, 0.074, 0.0]) + np.asarray(dq[:3], float))]
    dq = P.solve(0.001)
    moved = [("right_tip", [_cube(np.array([0.1, 0.074, 0.0]) + dq[:3], 0.01)])]
    assert 0.0 <= RO.sat_depth(moved, cobj, np.eye(3), np.zeros(3))[0] <= 0.0011


def test_window_frames_measure_the_object_at_their_joint_values(monkeypatch):
    """A window frame's projection, SAT and touch see the articulated object at that frame's joint value (the lid
    open at π/2), not the unposed object or frame 0's geometry."""
    S = _solve_mod(monkeypatch)
    RS = own_module("dexengine.retarget.f0.repair_side")
    seen = {}
    monkeypatch.setattr(RS, "_strict_model", lambda side: types.SimpleNamespace(fingers12=lambda x: x))
    monkeypatch.setattr(RS.RO, "desk_max_depth", lambda *a: 0.0)
    monkeypatch.setattr(RS, "_finger_box", lambda hand, indep: (None, None))
    monkeypatch.setattr(RS, "_geom_at", lambda scene, art, qrow: ("geom at", float(qrow[3])))
    monkeypatch.setattr(RS, "_contact_links_fn", lambda hand, obj, *a: (seen.update(touch=obj), lambda dq: [])[1])
    monkeypatch.setattr(RS, "Projection", lambda strict, cobj, *a: seen.update(proj=cobj))
    nf = S.HC.nd - 6
    scene = (None, "frame-0 geom", ("mesh", "sp", "artic", 0.0), np.eye(3), np.zeros(3), 0.0)
    fpose = (np.zeros(3), np.eye(3), np.zeros(nf), np.eye(3), np.zeros(3), np.array([0, 0, 0, np.pi / 2]))
    RS._frame_problem("right", _hinged_box(), scene, fpose)
    hand = [("right_tip", [_cube((0.1, 0.074, 0.0), 0.01)])]
    assert isinstance(seen["proj"], RO.CookedObject) and seen["touch"] == ("geom at", np.pi / 2)
    assert RO.sat_depth(hand, seen["proj"], np.eye(3), np.zeros(3))[0] == pytest.approx(0.006)


def test_articulated_geometry_is_shared_but_joint_columns_are_the_trees(monkeypatch, tmp_path):
    """Two trees of one articulated asset share the cooked geometry (built once) and each reads its own joint column —
    a cached column would read θ from another tree's layout."""
    import importlib

    OH = importlib.import_module("dexengine.data.objects.hull")
    QL = importlib.import_module("dexengine.core.paths").qpos_layout
    spec = _hinged_box().parts[0][0]
    built = []
    shapes = {"box": [_cube((0, 0, 0), 0.02)], "lid": [_cube((0.05, 0, 0), 0.02)]}
    monkeypatch.setattr(RO, "_COOKED", {})
    monkeypatch.setattr(RO, "_artic_links", lambda o, ds, tree: ("box", [], {}, [spec]))
    monkeypatch.setattr(OH, "asset_id", lambda o: "asset")
    monkeypatch.setattr(OH, "body_parts", lambda *a, **k: built.append(1) or shapes)
    monkeypatch.setattr(OH, "collision_parts", lambda *a: types.SimpleNamespace(digest="d"))
    cols = {"t1": 3, "t2": 5}
    layout = lambda tree, nq=None: types.SimpleNamespace(artic_joint_cols=lambda: {"box_hinge": cols[Path(tree).name]})  # noqa: E731
    monkeypatch.setattr(QL, "read", layout)
    o = {"name": "box", "articulation": {"joint": "box_hinge"}}
    a1, a2 = RO.cooked(o, "ds", tmp_path / "t1")[0], RO.cooked(o, "ds", tmp_path / "t2")[0]
    assert len(built) == 1 and a1.root is a2.root
    assert (a1.parts[0][1], a2.parts[0][1]) == (3, 5)


def test_an_articulated_main_object_is_built_from_its_tree(monkeypatch, tmp_path):
    """cooked() measures an articulated main object (the tree's links), never `n/a`; without the tree it dies."""
    o = {"name": "box", "articulation": {"joint": "box_hinge", "qpos_index": 3}}
    monkeypatch.setattr(RO, "_artic_cooked", lambda oo, ds, tree: ("ARTIC", str(tree)))
    assert RO.cooked(o, "ds", tmp_path) == ("ARTIC", str(tmp_path))
    with pytest.raises(ValueError, match="tree"):
        RO.cooked(o, "ds")


def test_pick_solution_has_no_object_keep(monkeypatch):
    """The object repair no longer rides the Nelder-Mead engine: the selection rule is the desk repair's alone."""
    import inspect

    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")  # repair_cfg resolves the active hand and arm at import
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    C = own_module("dexengine.retarget.f0.repair_cfg")
    assert list(inspect.signature(C.pick_solution).parameters) == ["dqs", "oks", "cost"]
    assert "accept" not in C.SolveRequest.__slots__


def test_window_follows_the_recipe_hold(monkeypatch):
    """The window is the recipe's longest spawn hold + the margin; a longer hold widens it (negative plant)."""
    from dexengine.rl.launch import recipe as RR

    assert RO.window_frames() == 10 + RO.OBJ_WINDOW_MARGIN
    real = RR.defaults
    monkeypatch.setattr(RR, "defaults", lambda p: {**real(p), "DEXENGINE_OBJ_SPAWN_HOLD": "20"})
    assert RO.window_frames() == 20 + RO.OBJ_WINDOW_MARGIN
    assert f"window={20 + RO.OBJ_WINDOW_MARGIN}" in RO.policy_line()


def _solve_mod(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair_obj_solve")


def _one_cube_problem(S, overlap_m):
    """A one-link hand (a 2 cm cube that translates with dq[:3]) under a 4 cm object cube, `overlap_m` deep."""
    nf = S.HC.nd - 6
    strict = types.SimpleNamespace(links=[("right_tip", 0, None, None)], parts={"right_tip": [_cube((0, 0, 0), 0.01)]})
    cobj = RO.CookedObject([_cube((0, 0, 0.03 - overlap_m), 0.02)])
    P = S.Projection(
        strict,
        cobj,
        np.eye(3),
        np.zeros(3),
        (np.zeros(3), np.eye(3), np.zeros(nf)),
        -1.0,
        (np.full(nf, -9.0), np.full(nf, 9.0)),
        ["right_tip"],
    )
    P.link_T = lambda dq: [(np.eye(3), np.asarray(dq[:3], float))]
    return P, cobj


def test_projection_moves_a_link_to_the_surface(monkeypatch):
    """5 mm overlap → the projection ends on the surface (≤ the 1 mm target), not past the touch distance."""
    S = _solve_mod(monkeypatch)
    P, cobj = _one_cube_problem(S, 0.005)
    hand = [("right_tip", [_cube((0, 0, 0), 0.01)])]
    assert RO.sat_depth(hand, cobj, np.eye(3), np.zeros(3))[0] == pytest.approx(0.005)
    dq = P.solve(0.001)
    moved = [("right_tip", [_cube(dq[:3], 0.01)])]
    d = RO.sat_depth(moved, cobj, np.eye(3), np.zeros(3))[0]
    assert 0.0 <= d <= 0.0011 and not P.lost_touch(dq)
    assert P.solve(0.001, dq0=dq) is not None and np.allclose(P.solve(0.001, dq0=dq), dq)  # warm start = no move


def test_projection_without_overlap_does_nothing(monkeypatch):
    S = _solve_mod(monkeypatch)
    P, _cobj = _one_cube_problem(S, -0.0005)  # 0.5 mm apart: touching, not overlapping
    assert np.allclose(P.solve(0.001), 0.0)


def test_f0_and_cross_repair_read_the_same_urdf(monkeypatch):
    """One body per (arm, hand): the f0 repair (HandCfg.urdf) and the cross repair (HandLayout → sim_collider) resolve
    the same URDF for every registered hand, and an operator override moves both (negative plant: a wrong path dies)."""
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    C = own_module("dexengine.retarget.f0.repair_cfg")
    from dexengine.core import paths as P
    from dexengine.placement.collision import sim_collider as SC

    monkeypatch.delenv("DEXENGINE_XR_URDF", raising=False)
    from dexengine.robots import vendor

    for hand in [h for h in P.HAND_IDS if h not in vendor.unavailable_hands()]:
        u = C.HandCfg(hand, "rby1").urdf("right")
        assert u == SC.resolved_urdf("rby1", hand) and u.is_file(), hand
    ov = SC.resolved_urdf("rby1", "wuji2")
    monkeypatch.setenv("DEXENGINE_XR_URDF", str(ov))
    assert C.HandCfg("sharpa_wave", "rby1").urdf("left") == ov == SC.resolved_urdf("rby1", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_XR_URDF", "/nonexistent/x.urdf")
    with pytest.raises(FileNotFoundError):
        C.HandCfg("sharpa_wave", "rby1").urdf("right")
    monkeypatch.delenv("DEXENGINE_XR_URDF")
    monkeypatch.setenv("DEXENGINE_URDF_OVERRIDE", str(ov))  # the retired name dies instead of being ignored
    with pytest.raises(SystemExit, match="retired"):
        C.HandCfg("sharpa_wave", "rby1").urdf("right")


def _turned_problem(S, Q):
    """One hand link (a 2 cm cube 3 mm off the wrist) 4 mm into an object cube, the whole scene turned by Q."""
    nf = S.HC.nd - 6
    strict = types.SimpleNamespace(links=[("right_tip", 0, None, None)], parts={"right_tip": [_cube((0, 0, 0), 0.01)]})
    cobj = RO.CookedObject([_cube((0.004, 0.002, 0.026), 0.02)])
    base = Q @ np.array([0.1, -0.2, 0.3])
    P = S.Projection(
        strict, cobj, Q, base, (base, Q, np.zeros(nf)), -10.0, (np.full(nf, -9.0), np.full(nf, 9.0)), ["right_tip"]
    )
    off = np.array([0.0, 0.003, 0.0])

    def link_T(dq):
        R = S._aa_to_R(dq[3:6]) @ Q
        return [(R, R @ off + base + dq[:3])]

    P.link_T = link_T
    return P


@pytest.mark.parametrize("iters", [1, 3, None])
def test_projection_turns_with_the_scene(monkeypatch, iters):
    """The same problem turned rigidly solves to the same answer turned (wrist move and rotation by Q, SAT equal) —
    the clip's placement on the desk must not change the repair. The old damping (diag(A), plant) fails this."""
    S = _solve_mod(monkeypatch)
    if iters:
        monkeypatch.setattr(S, "ITERS", iters)
    ax = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    Q = S._aa_to_R(ax * np.radians(40.0))

    def gap():
        a, b = _turned_problem(S, np.eye(3)).solve(0.001), _turned_problem(S, Q).solve(0.001)
        return max(np.abs(b[:3] - Q @ a[:3]).max(), np.abs(b[3:6] - Q @ a[3:6]).max()), a

    err, a = gap()
    assert np.linalg.norm(a[:6]) > 1e-3 and err < 1e-6
    if iters:  # negative plant: the axis-dependent scale turns a different step
        monkeypatch.setattr(S, "damping_scale", lambda A: np.diag(A) + 1e-12)
        assert gap()[0] > 1e-5


def test_failed_checks_at_the_open_threshold():
    """k=0 bounds the bodies by their open threshold (what the gate would not open again): the 2–5 mm band passes,
    above 5 mm fails with the open value in the reason."""
    assert RO.failed_checks(0.00295, "left_thumb_PP", 0.0, 0.0) == "SAT 2.95mm > 2mm (left_thumb_PP)"
    assert RO.failed_checks(0.00295, "left_thumb_PP", 0.0, 0.0, k=0) == ""
    assert RO.failed_checks(0.0051, "tip", 0.0, 0.0, k=0) == "SAT 5.10mm > 5mm (tip)"
    assert RO.failed_checks(0.003, "tip", 0.0012, 0.0, k=0) == "desk 1.20mm > start 0.00mm + 1mm (deeper)"


def _attempts_problem(sats_mm, lost_at=None, keep=True):
    """A solve_frame problem whose attempt i returns SAT sats_mm[i] (main object, no other body, desk 0); attempt
    `lost_at` loses a contact link."""
    calls = {"n": 0}

    def solve(r_t, warm):
        calls["n"] += 1
        return np.full(3, float(calls["n"] - 1))

    proj = types.SimpleNamespace(
        res_mm=np.array([RO.OBJ_RESIDUAL_MM]),
        solve=solve,
        lost_touch=lambda dq: ["hand_right_tip"] if lost_at is not None and int(dq[0]) == lost_at else [],
    )
    prob = {
        "proj": proj,
        "sat_fn": lambda dq: (sats_mm[int(dq[0])] / 1000.0, "tip"),
        "desk_fn": lambda dq: 0.0,
        "d0": 0.0,
    }
    if keep:
        prob["keep_open"] = True
    return prob


def test_frame0_keeps_the_least_sat_attempt_at_or_under_the_open_depth(monkeypatch):
    """No attempt reaches the 2 mm target: frame 0 keeps the shallowest attempt ≤ 5 mm, marked in prob['kept']."""
    S = _solve_mod(monkeypatch)
    prob = _attempts_problem([4.1, 3.4, 2.95, 3.2])
    dq, why, s, _sat = S.solve_frame(prob, None, 3)
    assert why == "" and s == 2.95 and int(dq[0]) == 2
    assert prob["kept"] == "kept SAT 2.95mm ≤ 5mm (target 2mm not reached, attempt 3 of 4)"


def test_a_passing_attempt_returns_as_before(monkeypatch):
    """An attempt that reaches the target returns at once (same attempt as without the keep), nothing marked."""
    S = _solve_mod(monkeypatch)
    prob = _attempts_problem([3.0, 1.5, 0.5, 0.2])
    dq, why, s, _sat = S.solve_frame(prob, None, 3)
    assert why == "" and s == 1.5 and int(dq[0]) == 1 and "kept" not in prob


def test_frame0_above_the_open_depth_stays_refused(monkeypatch):
    """Every attempt deeper than 5 mm (best 5.1): refused as before, no keep."""
    S = _solve_mod(monkeypatch)
    prob = _attempts_problem([7.0, 6.2, 5.1, 5.3])
    dq, why, s, _sat = S.solve_frame(prob, None, 3)
    assert dq is None and why.startswith("check failed after 4 tries: SAT 5.30mm > 2mm") and "kept" not in prob


def test_later_frames_do_not_keep(monkeypatch):
    """Without keep_open (a later window frame) the 2–5 mm attempt is not kept: the window carries the previous
    change."""
    S = _solve_mod(monkeypatch)
    prob = _attempts_problem([4.1, 3.4, 2.95, 3.2], keep=False)
    dq, why, _s, _sat = S.solve_frame(prob, None, 3)
    assert dq is None and why.startswith("check failed after 4 tries") and "kept" not in prob


def test_a_lost_contact_after_a_keepable_attempt_keeps_it(monkeypatch):
    """A later attempt that loses a contact link ends the retries; the keepable attempt before it is kept."""
    S = _solve_mod(monkeypatch)
    prob = _attempts_problem([3.4, 9.0, 1.0], lost_at=1)
    dq, why, s, _sat = S.solve_frame(prob, None, 3)
    assert why == "" and s == 3.4 and int(dq[0]) == 0 and "attempt 1 of 2" in prob["kept"]
    first = _attempts_problem([3.4], lost_at=0)
    dq, why, _s, _sat = S.solve_frame(first, None, 3)
    assert dq is None and why.startswith("no result keeps every contact link")


def test_policy_line_carries_the_keep():
    """The f0 cache key carries the keep rule: a clip held under the old rule is solved again."""
    assert "|f0 keep≤open" in RO.policy_line()


def test_keep_needs_every_other_body_within_its_open_value():
    """A main object at 3 mm is keepable, but a hand-on slot above its open value (1 mm) blocks the keep."""
    ok = RO.Sat([("main", "main", 0.003, "tip"), ("lid", "slot", 0.0008, "t")])
    bad = RO.Sat([("main", "main", 0.003, "tip"), ("lid", "slot", 0.0012, "t")])
    assert RO.failed_checks(0.003, "tip", 0.0, 0.0, ok, k=0) == ""
    assert RO.failed_checks(0.003, "tip", 0.0, 0.0, bad, k=0) == "slot lid SAT 1.20mm > 1mm (t)"


def test_keep_boundary_and_tie(monkeypatch):
    """Exactly the open depth (5.00 mm) is kept (the gate opens only above it); equal SATs keep the earlier attempt."""
    S = _solve_mod(monkeypatch)
    prob = _attempts_problem([6.0, 5.0, 5.5, 7.0])
    dq, why, s, _sat = S.solve_frame(prob, None, 3)
    assert why == "" and s == 5.0 and int(dq[0]) == 1
    tie = _attempts_problem([3.0, 3.0, 4.0, 4.0])
    dq, _why, s, _sat = S.solve_frame(tie, None, 3)
    assert s == 3.0 and int(dq[0]) == 0 and "attempt 1 of 4" in tie["kept"]


def _window(monkeypatch, sats, kept_text):
    """obj_solve_window over `sats` (one per frame) with solve_frame faked: frame 0 reports `kept_text` as kept."""
    S = _solve_mod(monkeypatch)
    monkeypatch.setattr(RO, "window_frames", lambda: len(sats))
    seen = []

    def fake(prob, warm, vr):
        seen.append(bool(prob.get("keep_open")))
        if prob.get("keep_open") and kept_text:
            prob["kept"] = kept_text
        return np.zeros(S.HC.nd), "", prob["s"], None

    monkeypatch.setattr(S, "solve_frame", fake)
    proj = types.SimpleNamespace(n_eval=0, lost_touch=lambda dq: [])
    probs = ({"s": s, "touch": [], "proj": proj} for s in sats)
    pose0 = (np.zeros(3), np.array([1.0, 0, 0, 0]), np.eye(3), np.zeros(S.HC.nd - 6), None, 0, slice(0, 7))
    return S.obj_solve_window({"side": "left"}, pose0, probs, 3, [0, 0], 0.0), seen


def test_window_carries_the_keep_mark_only_when_accepted(monkeypatch, capsys):
    """keep_open reaches frame 0 only; an accepted window reports obj_kept in the report, the verify notes and the
    [F0_OBJ] marker (with the side); a window the release check refuses reports its own reason, not the keep."""
    text = "kept SAT 2.22mm ≤ 5mm (target 2mm not reached, attempt 1 of 4)"
    rep, seen = _window(monkeypatch, [2.22] + [1.0] * 12, text)
    assert seen == [True] + [False] * 12
    assert rep["status"] == "REPAIRED" and rep["obj_kept"] == text and text in rep["verify"]
    assert f"side=left frame 0 {text}" in capsys.readouterr().out
    rep, _seen = _window(monkeypatch, [2.22] + [1.0] * 9 + [5.52, 1.0, 1.0], text)
    assert rep["status"].startswith(PH.SKIP_OBJ) and "obj_kept" not in rep and text not in rep.get("verify", "")
