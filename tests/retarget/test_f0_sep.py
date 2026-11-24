"""Rule 1d, the f0 frame-0 separation (repair_sep): the move, its verdicts, the one qpos source every f0 reader sees,
the writer, the hold path and the keys. The main pairs (two main objects) are in test_f0_sep_main.py."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import types
from pathlib import Path

import numpy as np
import pytest

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import touch_gate as TG
from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0 import repair_obj as RO
from dexengine.retarget.f0 import repair_sep as S
from tests.imports import own_module

#: (i, j, k) of the 12 outward-wound triangles of a box whose corner v = 4·x + 2·y + z (0 = low, 1 = high)
BOX_F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1)]
BOX_F += [(2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _corners(lo, hi):
    return np.array([[(lo, hi)[x][0], (lo, hi)[y][1], (lo, hi)[z][2]] for x in (0, 1) for y in (0, 1) for z in (0, 1)])


def _cube(c, h):
    return _corners(np.asarray(c) - h, np.asarray(c) + h)


def _still(name, lo, hi, qs, kind="receiver"):
    """A pinned still standing as its triangles (a closed box), at the identity pose."""
    tri = CS.Triangles(_corners(np.asarray(lo, float), np.asarray(hi, float)), np.array(BOX_F))
    return S.Item(name, "still", kind, qs, tri, True, tri.digest)


def _main(name, qs, h=0.02, side="right"):
    """A dynamic main object: a cube of half size h (its cooked part) at its qpos pose."""
    return S.Item(name, "main", "main", qs, TG.Body([CS.Parts([_cube((0, 0, 0), h)])]), False, name, side)


def _rows(*centres, T=40, leave=True):
    """qpos rows (T, 7·n): each body at its centre (identity rotation) on every frame; with `leave` the first body (the
    main object) slides 6 → 30 mm along +x over the last 5 frames, so it does not move together with any other body
    (a pair that moves together is S-dyn-pairs', not 1d's)."""
    q = np.tile(np.concatenate([np.r_[np.asarray(c, float), 1.0, 0.0, 0.0, 0.0] for c in centres]), (T, 1))
    if leave:
        q[-5:, 0] += np.linspace(0.006, 0.03, 5)
    return q


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """A tree directory whose bodies are the test's (repair_sep.bodies stubbed); the desk far below by default."""
    from dexengine.placement.coords import core as C

    monkeypatch.setattr(S, "_MEMO", {})
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "1")
    monkeypatch.setattr(C, "desk_height", lambda: -1.0)
    monkeypatch.setattr(C, "scene_z_offset", lambda tree: 0.0)
    t = tmp_path / "c0"
    t.mkdir()

    def put(items, q):
        objs = [{"name": it.name, "role": "main" if it.cls == "main" else "extra", "qpos_start": it.qs} for it in items]
        for o in objs:
            if o["role"] == "main":
                o.update(mesh_dir=o["name"], obj_id=o["name"])
        (t / "task_info.json").write_text(json.dumps({"dataset_name": "taco", "objects": objs}))
        by = {it.name: it for it in items}
        monkeypatch.setattr(S, "bodies", lambda tr, ti, qq, ds: (items, []))
        monkeypatch.setattr(TG, "body_of", lambda ds, o, tree: (by[o["name"]].ruler, ""))  # the main pair generator's
        return S.plan(t, q)

    return types.SimpleNamespace(path=t, put=put)


def test_a_shallow_pair_moves_only_the_dynamic_object(tree):
    """A cube 3.2 mm into a receiver's top: the least move frees it (2.5 mm up, true-shape depth after ≤ CLEAR_TOL),
    held over the window and faded; the receiver's rows and every other column stay as they are."""
    cube, box = _main("cube", 0), _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, 0.0), 7)
    q = _rows((0, 0, 0.02 - 0.0032), (0, 0, 0))
    pl = tree.put([cube, box], q)
    assert pl["verdict"] == S.SEPARATE and set(pl["moves"]) == {"cube"}
    p = pl["pairs"][0]
    assert p["d0_mm"] == pytest.approx(3.2) and p["move_mm"] == pytest.approx(2.5) and p["dir"] == 13
    q2 = S.applied(tree.path, q)
    n, B = RO.window_frames(), RO.OBJ_FADE_FRAMES
    assert np.allclose(q2[:n, 2] - q[:n, 2], 0.0025) and np.allclose(q2[n + B :], q[n + B :])
    assert np.all(np.diff(q2[n - 1 : n + B + 1, 2] - q[n - 1 : n + B + 1, 2]) < 0)
    assert np.array_equal(q2[:, 3:], q[:, 3:]) and np.array_equal(q2[:, :2], q[:, :2])
    poses = {"cube": S.XG.pose_mats(q2, 0), "box": S.XG.pose_mats(q2, 7)}
    assert (
        S.depth(cube, box, q2, poses, 0) <= S.TOL_M < S.depth(cube, box, q, {**poses, "cube": S.XG.pose_mats(q, 0)}, 0)
    )


def test_the_move_search_reading_is_the_full_one_where_it_is_read():
    """least_move's per-direction reading (`_depth_within`) is `depth` bit for bit at or below TOL_M and None above
    it, on every step and direction of a cube 3.2 mm into a receiver (both branches occur)."""
    cube, box = _main("cube", 0), _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, 0.0), 7)
    q = _rows((0, 0, 0.02 - 0.0032), (0, 0, 0))
    poses = {"cube": S.XG.pose_mats(q, 0), "box": S.XG.pose_mats(q, 7)}
    seen = set()
    for s in S.STEPS:
        for u in S.HCL.DIRS:
            d = S.depth(cube, box, q, poses, 0, ma=s * u)
            w = S._depth_within(cube, box, q, poses, 0, S.TOL_M, ma=s * u)
            assert (w is None) == (d > S.TOL_M) and (w is None or w == d)
            seen.add(w is None)
    assert seen == {True, False}


@pytest.mark.parametrize("mm", [6.0, 5.5])
def test_a_pair_deeper_than_the_cap_drops(tree, mm):
    """A planted pair deeper than TRUE_SHAPE_OVERLAP_M drops the tree (even where a ≤ 5 mm move would bring it under
    CLEAR_TOL): nothing moves, and the hold status names it."""
    cube, box = _main("cube", 0), _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, 0.0), 7)
    q = _rows((0, 0, 0.02 - mm / 1000.0), (0, 0, 0))
    pl = tree.put([cube, box], q)
    assert pl["verdict"] == S.DROP and S.moved(pl) is None and S.applied(tree.path, q) is q
    assert S.held_status(pl).startswith(f"{PH.SKIP_OBJ}: 1d drop: cubexbox(receiver) {mm}mm")


def test_a_kinematic_partner_is_never_moved(tree):
    """A moving slot (kinematic, its pose per frame) 2.7 mm into the cube: the cube moves; the slot's rows stay."""
    cube = _main("cube", 0)
    slot = S.Item("lid", "slot", "moving", 7, CS.Parts([_cube((0, 0, 0), 0.01)]), False, "s")
    q = _rows((0, 0, 0), (0.0, 0.0, 0.02 + 0.01 - 0.0027))
    q[:, 7] = np.linspace(0.0, 0.001, len(q))  # it moves: read at each frame
    pl = tree.put([cube, slot], q)
    assert pl["verdict"] == S.SEPARATE and set(pl["moves"]) == {"cube"}
    assert np.array_equal(S.applied(tree.path, q)[:, 7:], q[:, 7:])


def _hold_case(case):
    cube = _main("cube", 0)
    if case == "articulated mover":
        cube.art = object()
        return [cube, _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, -0.02 + 0.002), 7)], _rows((0, 0, 0), (0, 0, 0))
    if case == "several pairs":
        floor = _still("floor", (-0.05, -0.05, -0.04), (0.05, 0.05, -0.02 + 0.002), 7)
        wall = _still("wall", (0.02 - 0.002, -0.05, -0.04), (0.06, 0.05, 0.05), 14)
        return [cube, floor, wall], _rows((0, 0, 0), (0, 0, 0), (0, 0, 0))
    if case == "new overlap":
        floor = _still("floor", (-0.05, -0.05, -0.04), (0.05, 0.05, -0.02 + 0.0032), 7)
        lid = _still("lid", (-0.05, -0.05, 0.02 + 0.001), (0.05, 0.05, 0.05), 14)
        return [cube, floor, lid], _rows((0, 0, 0), (0, 0, 0), (0, 0, 0))
    raise KeyError(case)


@pytest.mark.parametrize("case", ["articulated mover", "several pairs", "new overlap"])
def test_cases_the_rule_does_not_decide_hold(tree, case):
    """An articulated mover · one mover in two pairs · a move that makes a new overlap: the tree holds with that
    reason, nothing moves."""
    items, q = _hold_case(case)
    pl = tree.put(items, q)
    assert pl["verdict"] == S.HOLD and S.applied(tree.path, q) is q
    assert S.held_status(pl).startswith(f"{PH.SKIP_OBJ}: 1d hold({case})")


def test_a_move_into_the_desk_holds(tree, monkeypatch):
    """The least move out of a body above goes down, deeper into f0's desk plane than before: hold (new overlap)."""
    from dexengine.placement.coords import core as C

    monkeypatch.setattr(C, "desk_height", lambda: -0.02 + 0.0005)  # the cube already 0.5 mm in (within the tolerance)
    cube, top = _main("cube", 0), _still("top", (-0.05, -0.05, 0.02 - 0.0032), (0.05, 0.05, 0.06), 7)
    pl = tree.put([cube, top], _rows((0, 0, 0), (0, 0, 0)))
    assert pl["verdict"] == S.HOLD and "cubexdesk" in pl["new"][0]


def test_nothing_to_separate_is_the_identity(tree):
    """No pair deeper than CLEAR_TOL: the rows every reader sees are the tree's own (the same object), no key line,
    nothing to write, no hold."""
    cube, box = _main("cube", 0), _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, 0.0), 7)
    q = _rows((0, 0, 0.02 - 0.0008), (0, 0, 0))  # 0.8 mm: a touch
    pl = tree.put([cube, box], q)
    assert pl["verdict"] == S.CLEAN and S.applied(tree.path, q) is q
    assert S.key_line(pl) == "" and S.moved(pl) is None and S.held_status(pl) is None


def test_switch_off_is_the_rule_before(tree, monkeypatch):
    cube, box = _main("cube", 0), _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, 0.0), 7)
    q = _rows((0, 0, 0.02 - 0.0032), (0, 0, 0))
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "0")
    pl = tree.put([cube, box], q)
    assert pl["verdict"] == S.CLEAN and S.applied(tree.path, q) is q and S.key_line(pl) == ""


def _hand_env(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair_side")


@pytest.mark.parametrize("hook", [True, False])
def test_the_hand_measures_the_moved_object(monkeypatch, tmp_path, hook):
    """The one qpos source (repair_side._qpos → repair_sep.applied): the frame-0 load and the object gate see the
    moved object. A finger 2 mm into the object reads 6 mm after a 4 mm move toward it and opens the repair; with the
    hook removed (rows as on disk) the same planted overlap reads 2 mm and goes unrepaired."""
    RS = _hand_env(monkeypatch)
    H = RS.HC.H
    T, d = 30, tmp_path / "data" / "c0"
    (d / "0").mkdir(parents=True)
    q = np.zeros((T, H + 7))
    q[:, H + 3] = 1.0
    np.savez(d / "0/trajectory_kinematic.npz", qpos=q)
    (d / "task_info.json").write_text("{}")
    (tmp_path / "w/wrist_poses_right").mkdir(parents=True)
    np.savez(
        tmp_path / "w/wrist_poses_right/c0.npz", positions=np.zeros((T, 3)), quaternions=np.tile([1.0, 0, 0, 0], (T, 1))
    )
    rows = np.tile([0.004, 0.0, 0.0], (len(S.weights(T)), 1)) * S.weights(T)[:, None]
    fake = {"verdict": S.SEPARATE, "moves": {"cube": {"qs": H, "rows": rows, "partner": "box"}}}
    monkeypatch.setattr(S, "plan", lambda tree, qq=None: fake)
    if not hook:
        monkeypatch.setattr(RS.RSEP, "applied", lambda tree, qq: qq)
    lay = types.SimpleNamespace(hand_qpos_dim=H, hand_slice=lambda s: (0, H), main_slice=lambda s: (H, H + 7))
    monkeypatch.setattr(RS.QL, "read", lambda *a, **k: lay)
    monkeypatch.setattr(RS, "main_object_dirs", lambda *a: ("m", "s", None))
    monkeypatch.setattr(RS, "desk_height", lambda: -1.0)
    monkeypatch.setattr(RS, "scene_z_offset", lambda tree: 0.0)
    monkeypatch.setitem(RS._HAND_CACHE, (RS.HC.name, "right"), types.SimpleNamespace(pen_margin=0.0))
    monkeypatch.setitem(RS._OBJ_CACHE, ("m", "s"), "geom")
    _t0, _tl, _art, _th, pose0, scene = RS._load_side("c0", "right", tmp_path / "data", tmp_path / "w", 0.0, "taco")
    assert scene[4][0] == pytest.approx(0.004 if hook else 0.0)
    cobj = RO.CookedObject([_cube((0, 0, 0), 0.02)], "c")
    monkeypatch.setattr(RS, "_cooked_main_object", lambda *a: (cobj, ""))
    monkeypatch.setattr(RS, "_strict_model", lambda side: types.SimpleNamespace(fingers12=lambda x: x))
    tip = [("right_tip", [_cube((0.004 + 0.02 + 0.01 - 0.006, 0, 0), 0.01)])]
    monkeypatch.setattr(RS.RO, "hand_parts_world", lambda *a: tip)
    rep = {"side": "right"}
    _fn, _c, opened = RS._obj_gate(rep, "c0", "right", tmp_path / "data", "taco", pose0, scene, False)
    assert opened is hook and rep["obj_sat_mm"] == pytest.approx(6.0 if hook else 2.0)


def test_the_moves_are_written_and_keyed(monkeypatch, tmp_path):
    """write_obj_window with the separation only: the mover's position columns move by its rows, nothing else
    changes, the markers carry the rows and their hash, and the judge's base field moves with them."""
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    W = own_module("dexengine.retarget.f0.repair_obj_write")
    from dexengine.placement.judge.f0_key import OBJ_WINDOW_MARKER

    T, rng = 40, np.random.default_rng(1)
    q = rng.normal(0, 0.1, (T, 21))
    (tmp_path / "data/c0/0").mkdir(parents=True)
    np.savez(tmp_path / "data/c0/0/trajectory_kinematic.npz", qpos=q, qvel=np.zeros_like(q))
    rows = S.weights(T)[:, None] * np.array([0.0, 0.0, 0.002])
    sep = {"cup": {"qs": 14, "rows": rows, "partner": "bowl"}}
    W.write_obj_window("c0", {}, tmp_path / "data", tmp_path / "w", sep)
    q2 = np.load(tmp_path / "data/c0/0/trajectory_kinematic.npz")["qpos"]
    F = len(rows)
    assert np.allclose(q2[:F, 14:17] - q[:F, 14:17], rows) and np.array_equal(q2[F:], q[F:])
    assert np.array_equal(q2[:, :14], q[:, :14]) and np.array_equal(q2[:, 17:], q[:, 17:])
    mk = json.loads((tmp_path / "data/c0" / OBJ_WINDOW_MARKER).read_text())
    assert mk["rows_sha"] == W.obj_rows_sha({f"{S.SEP_FIELD}/cup": rows}) and mk[S.SEP_FIELD]["cup"]["move_mm"] == 2.0
    inplace = json.loads((tmp_path / "data/c0/F0_REPAIR_INPLACE.json").read_text())
    assert inplace["mode"] == "window" and inplace[S.SEP_FIELD]["cup"]["partner"] == "bowl"
    assert S._written(tmp_path / "data/c0")  # the reader takes these rows as they are from now on


#: the window writer's outputs on the inputs of test_f0_obj_outputs.test_window_writer_rewrites_rows_in_place, measured
#: on the code before rule 1d: JSON bytes and array contents (sha256, 16 hex)
BEFORE_1D = {
    "data/c0/F0_OBJ_WINDOW.json": "32f1519816e33b7d",
    "data/c0/F0_REPAIR_INPLACE.json": "39b6f9bb9e3f520a",
    # the root's axes from the scene, not a row fit (repair_root.root_axes): the 8 window rows' root translation moved
    # by <= 1.3e-15 m against the hash taken before rule 1d (fe7f4a2a06952a36, which the old row fit still gives)
    "data/c0/0/trajectory_kinematic.npz": "b3139b59ce95ccc0",
    "data/c0/0/trajectory_kinematic.pre_f0r.npz": "3f8f3e70c938f324",
    "w/wrist_poses_right/c0.npz": "10589960471a18ce",
    "w/wrist_poses_right/c0.pre_f0r.npz": "7f8c7a134fc8aee6",
}


def _content(f: Path) -> str:
    if f.suffix != ".npz":
        return hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    z, h = np.load(f), hashlib.sha256()
    for k in sorted(z.files):
        h.update(k.encode() + np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()[:16]


def test_a_tree_without_a_move_is_written_as_before(monkeypatch, tmp_path, retarget_venv):
    """Identity: the hand window written without a separation gives the bytes of the code before rule 1d."""
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    W = own_module("dexengine.retarget.f0.repair_obj_write")
    from dexengine.retarget.f0 import repair_root as RT

    # no scene in this tree: its wrist cache is qpos[:3] + a constant, i.e. the root's axes are the identity
    monkeypatch.setattr(RT, "root_axes", lambda tree, off: np.eye(3))
    H, T, rng = W.HC.H, 40, np.random.default_rng(0)
    qpos = rng.normal(0, 0.1, (T, H + 7))
    (tmp_path / "data/c0/0").mkdir(parents=True)
    np.savez(tmp_path / "data/c0/0/trajectory_kinematic.npz", qpos=qpos, qvel=np.zeros_like(qpos))
    (tmp_path / "w/wrist_poses_right").mkdir(parents=True)
    pos, quat = qpos[:, :3] + np.array([0.1, 0.2, 0.3]), np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    np.savez(tmp_path / "w/wrist_poses_right/c0.npz", positions=pos, quaternions=quat)
    rows = np.zeros((8, W.HC.nd))
    rows[:, 0], rows[:, 6:] = 0.01, 0.05
    r = {"_internal": (None,) * 6 + (0, None), "obj_window": rows, "obj_window_n": 5, "obj_window_sat_mm": [1.0] * 5}
    W.write_obj_window("c0", {"right": {**r, "obj_window_carried": []}}, tmp_path / "data", tmp_path / "w", None)
    got = {str(f.relative_to(tmp_path)): _content(f) for f in tmp_path.rglob("*") if f.is_file()}
    assert got == BEFORE_1D


BUDGET = "[F0_BUDGET] side={} evals=0/10 status=OK"


@pytest.mark.parametrize("verdict", ["drop", "hold"])
def test_a_dropped_or_held_tree_is_never_adopted(monkeypatch, tmp_path, verdict):
    """The hold line (repair._hold_line) through the f0 classification: HOLD_OBJ (not judgeable, no marker, the
    placement ledger's HELD ∈ REJECTED), the ledger note '1d drop' or '1d hold(<why>)'."""
    _hand_env(monkeypatch)
    R = own_module("dexengine.retarget.f0.repair")
    from dexengine.retarget.f0 import shadow_status as SS
    from dexengine.retarget.f0 import shadow_tree as ST

    monkeypatch.setattr(ST, "applied_scene", lambda: "test")
    pair = {
        "mover": "cup",
        "partner": "bowl",
        "kind": "receiver",
        "side": "left",
        "d0_mm": 6.2 if verdict == "drop" else 2.0,
    }
    pair.update(verdict=verdict, reason="cap" if verdict == "drop" else "several pairs", why="w")
    pl = {"verdict": verdict, "pairs": [pair], "new": []}
    fails, a = [], types.SimpleNamespace(in_place=True, check_only=False)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        R._hold_line("c0", pl, S.held_status(pl), ["right", "left"], fails, a)
    out = "\n".join([BUDGET.format("right"), BUDGET.format("left"), buf.getvalue()])
    d = tmp_path / "shadow"
    d.mkdir()
    job = {"clip": "c0", "t0": 0.0, "d": d, "key_now": "k", "stale": "", "sides": ["right", "left"]}
    res = ST.finish(job, 0, out, types.SimpleNamespace(hand="sharpa_wave", dataset="taco"), Path("/s"), {})
    assert res["status"] == SS.HELD and not SS.is_judgeable(res["status"]) and not (d / ST.MARKER).exists()
    assert res["log"].startswith("left: 1d drop" if verdict == "drop" else "left: 1d hold(several pairs)")
    assert fails == [("c0", "left", S.held_status(pl))]


@pytest.mark.parametrize("exit_", ["clean", "shallow", "desk"])
def test_the_moves_are_written_on_every_judgeable_exit(monkeypatch, tmp_path, exit_):
    """_write_sequence writes the separation's rows (before any desk pad) whether the sides were clean, shallow or
    desk-repaired; without a move it does what it did before."""
    _hand_env(monkeypatch)
    R = own_module("dexengine.retarget.f0.repair")
    calls = []
    monkeypatch.setattr(R, "write_obj_window", lambda seq, obj, dr, wr, sep: calls.append(("window", sorted(obj), sep)))
    monkeypatch.setattr(R, "write_in_place", lambda seq, desk, *a: calls.append(("desk", sorted(desk))) or ("x", None))
    (tmp_path / "c0").mkdir()
    a = types.SimpleNamespace(check_only=False, in_place=True, n_blend=20, out_suffix="_f0")
    repaired = {"right": {"status": "REPAIRED"}} if exit_ == "desk" else {}
    fails = [("c0", "right", "SKIP: shallow")] if exit_ == "shallow" else []
    sep = {"cup": {"qs": 0, "rows": np.zeros((3, 3)), "partner": "bowl"}}
    R._write_sequence(a, ["right"], tmp_path, fails, repaired, "c0", ["right"], tmp_path, sep)
    assert calls[0] == ("window", [], sep) and (calls[1:] == [("desk", ["right"])]) is (exit_ == "desk")
    calls.clear()
    R._write_sequence(a, ["right"], tmp_path, fails, repaired, "c0", ["right"], tmp_path, None)
    assert all(c[0] != "window" for c in calls)


def test_the_f0_key_moves_only_for_a_tree_with_a_pair(monkeypatch, tmp_path):
    """repair.obj_acted_part: a tree without a pair keeps its part ("" with the gate closed); a pair adds the
    separation's line (policy with the ruler identity, bodies, digests, verdict, move)."""
    _hand_env(monkeypatch)
    R = own_module("dexengine.retarget.f0.repair")

    def steps(clip, side, tree_root, wrist_root, desk_margin, do_repair, dataset="taco"):
        yield from ()
        return {"side": side, "obj_repair": False}

    monkeypatch.setattr(R, "check_and_repair_steps", steps)
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "1")
    (tmp_path / "wrist_poses_right").mkdir()
    (tmp_path / "wrist_poses_right/c0.npz").write_bytes(b"")
    clean = {"verdict": S.CLEAN, "pairs": [], "new": []}
    monkeypatch.setattr(R.RSEP, "plan", lambda tree, q=None: clean)
    assert R.obj_acted_part("c0", tmp_path, tmp_path, "taco", ["right"]) == ""
    pair = {"mover": "cup", "partner": "bowl", "kind": "receiver", "digests": ["a", "b"], "verdict": S.SEPARATE}
    monkeypatch.setattr(
        R.RSEP,
        "plan",
        lambda tree, q=None: {**clean, "verdict": S.SEPARATE, "pairs": [{**pair, "move_m": 0.002, "dir": 13}]},
    )
    part = R.obj_acted_part("c0", tmp_path, tmp_path, "taco", ["right"])
    assert (
        part.startswith(f"sep|{S.RULE_ID}|") and "ruler=" in part and "cup>bowl:receiver:a/b:separate:0.002:13" in part
    )
