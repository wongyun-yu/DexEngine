"""Rule S-dyn-pairs on synthetic scenes (CPU): a both-dynamic frame-0 pair that the F0-1d separation does not free has
its collision off and no object moves (the `scene` fixture states no separation; the separation itself is
tests/retarget/test_f0_sep.py's), a pair it frees keeps its collision and has no visual reading, the recorded visual
overlap and the delivered output check drop above the true-shape overlap (6 mm drops, 4 mm does not), two main objects
that move together (2 mm / 2°) have no one-body path and drop, a separating pair is not together, and removing the
pair-off fails the pair-off check."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import dyn_pair_check as DC
from dexengine.data.objects import dyn_pairs as DP
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import scene_requirement as SR
from dexengine.data.objects import scene_rules as SC
from dexengine.data.objects import touch_gate as TG
from dexengine.rl.tasks.scene import collision_presets as CP

T = 50
HA, HB = 0.05, 0.02  # half sizes of the two boxes (m)
RES = 2e-4  # reading error allowed on flat faces (m): vertices and face samples read the planted depth exactly


def _box_mesh(half):
    import trimesh  # noqa: PLC0415

    m = trimesh.creation.box(extents=2 * np.full(3, half))
    return np.asarray(m.vertices, float), np.asarray(m.faces)


def _corners(half):
    return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * half


def _col(xyz):
    """(T,7) pose column xyz · wxyz (identity rotation)."""
    return np.c_[xyz, np.ones(len(xyz)), np.zeros((len(xyz), 3))]


def _ti(art=False):
    return {
        "dataset_name": "hot3d",
        "objects": [
            {"name": "right_object", "role": "main", "side": "right", "qpos_start": 0, "mesh_dir": "a", "obj_id": "A"},
            {
                "name": "left_object",
                "role": "main",
                "side": "left",
                "qpos_start": 7,
                "mesh_dir": "b",
                "obj_id": "B",
                **({"articulation": {"joints": 1}} if art else {}),
            },
            {"name": "extra_object_0", "role": "extra", "qpos_start": 14, "mesh_dir": "c", "obj_id": "C"},
        ],
    }


def _q(pen0, pen_mid=None, ride=False):
    """Box A (right_object) still at the origin; box B (left_object) enters A's +x face by pen0 at frame 0 and leaves
    from frame 10 (pen_mid: B goes back in that deep at frame 30). ride: A and B travel together the whole clip."""
    xa = np.zeros((T, 3))
    xb = np.zeros((T, 3))
    xb[:, 0] = HA + HB - pen0 + 0.1 * np.clip(np.arange(T) - 10, 0, None) / T
    if pen_mid is not None:
        xb[30, 0] = HA + HB - pen_mid
    if ride:
        path = np.c_[np.linspace(0, 0.3, T), np.zeros(T), 0.1 * np.sin(np.linspace(0, np.pi, T))]
        xa, xb = path, path + [HA + HB - pen0, 0.0, 0.0]
    xc = np.tile([1.0, 1.0, 0.0], (T, 1))  # a far extra, never a pair (not dynamic)
    return np.concatenate([_col(xa), _col(xb), _col(xc)], axis=1)


@pytest.fixture
def scene(monkeypatch):
    """Rulers of the synthetic boxes: collision = one convex part, visual = the box triangles; the rule on."""
    parts = {"A": TG.Body([CS.Parts([_corners(HA)])]), "B": TG.Body([CS.Parts([_corners(HB)])])}
    tris = {"A": CS.Triangles(*_box_mesh(HA)), "B": CS.Triangles(*_box_mesh(HB))}
    monkeypatch.setattr(SC, "on", lambda rid, ds: rid == DP.RULE_ID)
    monkeypatch.setattr(TG, "body_of", lambda ds, o, tree: (parts[o["obj_id"]], ""))
    monkeypatch.setattr(DC, "ruler", lambda ds, o, tree=None: DC.Visual([tris[o["obj_id"]]]))
    monkeypatch.setattr(CS, "triangles", lambda ds, oid: tris[oid])
    monkeypatch.setattr(DP, "separated", lambda tree, q: set())  # the F0-1d separation frees nothing here
    return tris


def _st(tree, ti, q):
    """The statement of a synthetic tree (no extra rows)."""
    return DP.dyn_pairs(tree, ti, q, rows=[])


def _hash(q):
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()


def _cells(st):
    """The collision cells of a statement's pair_off pairs."""
    return CP.dyn_pair_cells([p for p in st["pairs"] if p["verdict"] == CP.DYN_PAIR_OFF])


def _assert_pair_off(st):
    """The planted pair is off in training and delivered, and nothing else is."""
    cells = _cells(st)
    assert [(c["a"], c["b"]) for c in cells] == [("left_object", "right_object")]
    assert cells[0]["training"] == "OFF" and cells[0]["delivered"] == "OFF"


def test_dynamic_is_the_main_role():
    assert DP.dynamic({"role": "main"}) and not any(DP.dynamic({"role": r}) for r in ("extra", "support", None))


def test_both_dynamic_pair_is_off_and_no_object_moves(scene, capsys):
    q = _q(0.003)
    before = _hash(q)
    st = _st("/x/clip_pl_rt", _ti(), q)
    assert _hash(q) == before  # the statement never moves an object
    (p,) = st["pairs"]
    assert p["verdict"] == CP.DYN_PAIR_OFF and st["drop"] is None and st["no_path"] is None
    assert abs(p["f0_m"] - 0.003) < RES and p["visual_m"] <= SR.TRUE_SHAPE_OVERLAP_M and not DC.exact(p["visual_m"])
    _assert_pair_off(st)
    assert "[DYN_PAIRS] clip_pl_rt: left_object×right_object f0 3.0 mm" in capsys.readouterr().out


def test_removing_the_pair_off_fails_the_check(scene, monkeypatch):
    st = _st("/x/clip_pl_rt", _ti(), _q(0.003))
    monkeypatch.setattr(CP, "DYN_PAIR_COLLIDE", {"training": "ON", "delivered": "ON"})  # the pair-off removed
    with pytest.raises(AssertionError):
        _assert_pair_off(st)


def test_free_pair_is_not_stated(scene):
    st = _st("/x/clip_pl_rt", _ti(), _q(-0.002))  # 2 mm apart at frame 0
    assert st["pairs"] == [] and _cells(st) == []


@pytest.mark.parametrize(("pen", "verdict"), [(0.006, DP.DROP), (0.004, CP.DYN_PAIR_OFF)])
def test_recorded_visual_overlap_drops_above_the_true_shape_overlap(scene, pen, verdict):
    assert 0.004 < SR.TRUE_SHAPE_OVERLAP_M < 0.006
    st = _st("/x/clip_pl_rt", _ti(), _q(pen))
    (p,) = st["pairs"]
    assert p["verdict"] == verdict and DC.exact(p["visual_m"]) == (pen > SR.TRUE_SHAPE_OVERLAP_M)
    assert abs(p["visual_m"] - pen) < RES if DC.exact(p["visual_m"]) else p["visual_m"] <= SR.TRUE_SHAPE_OVERLAP_M
    assert (st["drop"] is not None) == (verdict == DP.DROP)
    if st["drop"]:
        assert st["drop"][0] == DC.DROP_WORD and _cells(st) == []


def test_recorded_overlap_reads_the_whole_clip(scene):
    st = _st("/x/clip_pl_rt", _ti(), _q(0.003, pen_mid=0.006))
    (p,) = st["pairs"]
    assert not DC.exact(p["visual_f0_m"]) and abs(p["visual_m"] - 0.006) < RES and p["visual_frame"] == 30
    assert p["verdict"] == DP.DROP


def test_reading_stops_at_the_first_frame_above_the_cap(scene, monkeypatch):
    """Frames are read deepest first on the collision shapes; the first one above the cap ends the reading."""
    read = []
    real = DC.screened
    monkeypatch.setattr(
        DC, "screened", lambda ra, rb, M, cap=SR.TRUE_SHAPE_OVERLAP_M: read.append(M) or real(ra, rb, M)
    )
    st = _st("/x/clip_pl_rt", _ti(), _q(0.003, pen_mid=0.006))
    (p,) = st["pairs"]
    assert p["visual_frame"] == 30 and p["verdict"] == DP.DROP
    assert len(read) == 2  # frame 0 alone (visual_f0_m), then frame 30 first in the deepest-first order


def test_main_pair_that_moves_together_has_no_path_and_drops(scene):
    q = _q(0.003, ride=True)
    rel = DP.together(q, 0, 7, DP.MAIN)
    assert rel is not None and rel["dev_m"] <= DP.MAIN_TOGETHER_POS_M and rel["dev_deg"] <= DP.MAIN_TOGETHER_ROT_DEG
    M = np.eye(4)
    M[:3, :3] = XG.pose_mats(np.array([[0, 0, 0, *rel["rel"][3:]]], float), 0)[0, :3, :3]
    M[:3, 3] = rel["rel"][:3]
    dev_m, dev_deg = XG.pose_dev(XG.pose_mats(q, 0) @ M, XG.pose_mats(q, 7))  # carrier × relative pose = the rider
    assert dev_m.max() < 1e-6 and dev_deg.max() < 1e-3
    st = _st("/x/clip_pl_rt", _ti(), q)
    (p,) = st["pairs"]
    assert p["verdict"] == DP.NO_PATH_V and p["why"] == DP.NO_PATH["main"] and st["no_path"][0] == DP.NO_PATH_WORD
    assert _cells(st) == []


def test_separating_pair_is_not_one_body(scene):
    q = _q(0.003)
    assert DP.together(q, 0, 7, DP.MAIN) is None
    assert _st("/x/clip_pl_rt", _ti(), q)["pairs"][0]["verdict"] == CP.DYN_PAIR_OFF


def test_the_two_pair_classes_have_their_own_band():
    """Two boxes whose relative pose drifts 3 mm over the clip: together as an extra pair (5 mm), not as a main pair
    (2 mm); an unknown pair class raises."""
    q = _q(0.003, ride=True)
    q[:, 8] += np.linspace(0.0, 0.003, T)  # left_object creeps 3 mm along +y relative to right_object
    assert DP.together(q, 0, 7, DP.MAIN) is None and DP.together(q, 0, 7, DP.EXTRA) is not None
    assert DP.together_values(DP.MAIN) == (0.002, 2.0) and DP.together_values(DP.EXTRA) == (0.005, 5.0)
    with pytest.raises(ValueError, match="unknown pair class"):
        DP.together(q, 0, 7, "both")


def _freed(monkeypatch):
    monkeypatch.setattr(DP, "separated", lambda tree, q: {frozenset(("left_object", "right_object"))})


def test_a_pair_the_separation_frees_keeps_its_collision(scene, monkeypatch):
    """The F0-1d separation frees the pair (dyn_pairs.separated): verdict separate, no visual reading (a 6 mm recorded
    overlap at frame 30 drops nothing), no pair-off cell; the scene reads the rows f0 writes: with the move written
    (left_object 3.5 mm out at frame 0) it finds no pair to turn off, and rows without the move keep it off."""
    _freed(monkeypatch)
    q = _q(0.003, pen_mid=0.006)
    st = _st("/x/clip_pl_rt", _ti(), q)
    (p,) = st["pairs"]
    assert p["verdict"] == DP.SEPARATE_V and "visual_m" not in p and st["drop"] is None and _cells(st) == []
    moved = q.copy()
    moved[:10, 7] += 0.0035
    assert DP.off_pairs("/x/clip_pl_rt", _ti(), moved) == [] and len(DP.off_pairs("/x/clip_pl_rt", _ti(), q)) == 1


def test_without_the_separation_the_same_pair_is_off_and_drops(scene):
    """Negative of the test above: the same rows with no separation are pair_off in the scene and drop on the visual
    overlap."""
    q = _q(0.003, pen_mid=0.006)
    (p,) = _st("/x/clip_pl_rt", _ti(), q)["pairs"]
    assert p["verdict"] == DP.DROP and [x["a"] for x in DP.off_pairs("/x/clip_pl_rt", _ti(), q)] == ["left_object"]


def test_the_scene_reads_the_frame0_pairs_that_are_not_together(scene):
    """off_pairs (the scene statement, env_setup.dyn_pairs) = the frame-0 pairs of the statement without its visual
    reading: a 6 mm pair is off in the scene (its drop is the exclusion step's), a together or free pair is not."""

    def names(q):
        return [(p["a"], p["b"]) for p in DP.off_pairs("/x/clip_pl_rt", _ti(), q)]

    pair = [("left_object", "right_object")]
    assert names(_q(0.003)) == pair and names(_q(0.006)) == pair
    assert DP.off_pairs("/x/clip_pl_rt", _ti(), _q(0.003))[0]["f0_m"] == pytest.approx(0.003, abs=2e-4)
    assert DP.off_pairs("/x/clip_pl_rt", _ti(), _q(0.003, ride=True)) == []
    assert DP.off_pairs("/x/clip_pl_rt", _ti(), _q(-0.002)) == []


def test_opted_out_dataset_states_no_pair(scene, monkeypatch):
    monkeypatch.setattr(SC, "on", lambda rid, ds: False)
    assert _st("/x/clip_pl_rt", _ti(), _q(0.006))["pairs"] == []
    assert DP.off_pairs("/x/clip_pl_rt", _ti(), _q(0.006)) == []


@pytest.mark.parametrize(("pen", "verdict"), [(0.006, DC.DROP), (0.004, DC.KEEP)])
def test_output_check_drops_a_delivered_penetration_above_the_true_shape_overlap(scene, pen, verdict):
    objs = {o["name"]: o for o in _ti()["objects"]}
    q = _q(-0.002)  # the reference is free; the delivered output goes in at frame 20
    poses = {"right_object": XG.pose_mats(q, 0), "left_object": XG.pose_mats(q, 7)}
    poses["left_object"][20, 0, 3] = HA + HB - pen
    res = DC.output_check("hot3d", objs, [("left_object", "right_object")], poses)
    (p,) = res["pairs"]
    assert p["verdict"] == verdict
    assert abs(p["depth_m"] - pen) < RES and p["frame"] == 20 if verdict == DC.DROP else not DC.exact(p["depth_m"])
    assert (res["drop"] is not None) == (verdict == DC.DROP)


def test_output_check_needs_the_pair_poses(scene):
    with pytest.raises(KeyError):
        DC.output_check("hot3d", {}, [("left_object", "right_object")], {"left_object": np.eye(4)[None]})


def test_screen_never_changes_the_threshold_answer(scene):
    """The screened reading equals collider_shape.pair_depth wherever that is above the cap, and stays at or below the
    cap wherever it is not (planted depths around the cap, rotated)."""
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    ra, rb = scene["A"], scene["B"]
    rng = np.random.default_rng(0)
    seen = {True: 0, False: 0}
    for pen in (0.001, 0.0045, 0.0049, 0.0051, 0.0055, 0.008, 0.015):
        for _ in range(3):
            M = np.eye(4)
            M[:3, :3] = R.from_rotvec(rng.normal(0, 0.05, 3)).as_matrix()
            M[:3, 3] = (HA + HB - pen, rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01))
            full, got = CS.pair_depth(ra, rb, M), DC.screened(ra, rb, M)
            above = full > SR.TRUE_SHAPE_OVERLAP_M
            seen[above] += 1
            assert DC.exact(got) == above
            assert got == pytest.approx(full, abs=1e-12) if above else got <= SR.TRUE_SHAPE_OVERLAP_M
    assert seen[True] and seen[False]


def test_the_scene_pair_is_one_right_and_one_left_main_object(monkeypatch):
    """dyn_pairs.scene_pair (read by the scene statement and the delivered-output statement): one pair, its sides;
    no pair -> None; two pairs, or a pair on one side, raise."""
    ti = {"objects": [{"name": "r", "side": "right"}, {"name": "l", "side": "left"}, {"name": "s", "side": "right"}]}
    pr = {"a": "l", "b": "r", "f0_m": 0.003}
    monkeypatch.setattr(DP, "off_pairs", lambda *a: [pr])
    assert DP.scene_pair("/x/c", ti, None) == {**pr, "sides": {"l": "left", "r": "right"}}
    monkeypatch.setattr(DP, "off_pairs", lambda *a: [])
    assert DP.scene_pair("/x/c", ti, None) is None
    monkeypatch.setattr(DP, "off_pairs", lambda *a: [pr, {"a": "l", "b": "s", "f0_m": 0.003}])
    with pytest.raises(ValueError, match="one per clip"):
        DP.scene_pair("/x/c", ti, None)
    monkeypatch.setattr(DP, "off_pairs", lambda *a: [{"a": "r", "b": "s", "f0_m": 0.003}])
    with pytest.raises(ValueError, match="cuts the right object against the left one"):
        DP.scene_pair("/x/c", ti, None)


# ── After cross repair (dyn_pairs.after_cross_repair · gate J-gate-dynpairs-xr) ──────────────────────────────────────
# q_pre / q_post = the rows before / after cross repair on one time axis; the final rows are q_post here (no retiming).


def _xr(q_pre, q_post):
    return DP.after_cross_repair("/x/clip_pl_rt", _ti(), q_pre, q_post, q_post)


def _count_reads(monkeypatch):
    read = []
    real = DC.depth_over
    monkeypatch.setattr(DC, "depth_over", lambda *a, **k: read.append(1) or real(*a, **k))
    return read


def test_xr_a_pair_cross_repair_turns_off_is_read_on_the_final_rows(scene, capsys):
    """0.8 mm before cross repair (free: its collision was on) and 1.2 mm after: the deeper-than-the-rule branch —
    read on the final rows (pair_off, no drop at 1.2 mm), never separated again (the rows are not touched)."""
    q_post = _q(0.0012)
    before = _hash(q_post)
    st = _xr(_q(0.0008), q_post)
    (p,) = st["read"]
    assert p["why"] == DP.XR_TURNED_OFF and p["verdict"] == CP.DYN_PAIR_OFF and st["drop"] is None
    assert p["f0_m"] == pytest.approx(0.0012, abs=RES) and p["f0_pre_m"] == pytest.approx(0.0008, abs=RES)
    assert "visual_m" in p and not DC.exact(p["visual_m"]) and _hash(q_post) == before and st["freed"] == []
    assert "[DYN_PAIRS_XR] clip_pl_rt: left_object×right_object f0 0.8 → 1.2 mm (turned off by cross repair)" in (
        capsys.readouterr().out
    )


def test_xr_a_pair_below_the_tolerance_after_cross_repair_keeps_its_collision(scene, monkeypatch, capsys):
    """0.8 mm after cross repair is no pair: nothing is read, nothing is printed."""
    read = _count_reads(monkeypatch)
    st = _xr(_q(-0.002), _q(0.0008))
    assert (
        st == {"read": [], "freed": [], "drop": None} and read == [] and "[DYN_PAIRS_XR]" not in capsys.readouterr().out
    )


def test_xr_an_off_pair_cross_repair_did_not_move_is_not_read_again(scene, monkeypatch):
    """Off before and after, rows unchanged: the exclusion step's reading stands (no visual reading here)."""
    read = _count_reads(monkeypatch)
    q = _q(0.003, pen_mid=0.006)
    assert _xr(q, q.copy()) == {"read": [], "freed": [], "drop": None} and read == []


def test_xr_an_off_pair_cross_repair_moved_but_did_not_deepen_is_not_read_again(scene, monkeypatch):
    """Off before and after, cross repair moved the objects but the deepest depth fell (4 → 3.5 mm): not read again."""
    read = _count_reads(monkeypatch)
    assert _xr(_q(0.003, pen_mid=0.004), _q(0.003, pen_mid=0.0035)) == {"read": [], "freed": [], "drop": None}
    assert read == []


def test_xr_an_off_pair_cross_repair_deepens_from_4_to_6_mm_drops(scene):
    """Off before and after; cross repair deepens its recorded overlap at frame 30 from 4 to 6 mm: read again on the
    final rows and dropped (reason dyn_pair_check.DROP_WORD)."""
    st = _xr(_q(0.003, pen_mid=0.004), _q(0.003, pen_mid=0.006))
    (p,) = st["read"]
    assert p["why"] == DP.XR_DEEPENED and p["verdict"] == DP.DROP and p["visual_frame"] == 30
    assert abs(p["visual_m"] - 0.006) < RES and st["drop"][0] == DC.DROP_WORD


def test_xr_without_the_reread_the_deepened_pair_would_pass(scene, monkeypatch):
    """Negative of the test above: with 'deepened' never true, the 6 mm overlap the cross repair made passes."""
    monkeypatch.setattr(DP, "_deepened", lambda *a: False)
    assert _xr(_q(0.003, pen_mid=0.004), _q(0.003, pen_mid=0.006))["drop"] is None


def test_xr_a_pair_cross_repair_frees_is_counted(scene, capsys):
    """Off before (3 mm), free after (1 mm apart): its collision is on (the scene's statement); counted, not read."""
    st = _xr(_q(0.003), _q(-0.001))
    assert st["read"] == [] and st["drop"] is None
    assert [(p["a"], p["b"]) for p in st["freed"]] == [("left_object", "right_object")]
    assert "left_object×right_object f0 3.0 mm → free (freed by cross repair)" in capsys.readouterr().out


def test_xr_rows_of_two_lengths_raise(scene):
    with pytest.raises(ValueError, match="before / after cross repair"):
        _xr(_q(0.003)[:-1], _q(0.003))


def test_xr_opted_out_dataset_states_nothing(scene, monkeypatch):
    monkeypatch.setattr(SC, "on", lambda rid, ds: False)
    assert _xr(_q(0.0008), _q(0.0012)) == {"read": [], "freed": [], "drop": None}


def _tree(tmp_path, q_pre=None, q_post=None, q_final=None, report=True, retimed=False):
    """A baked tree on disk with the files the gate reads (cross-repair report · input copy · copy before retiming)."""
    from dexengine.placement.repair import cross_repair as XR  # noqa: PLC0415
    from dexengine.placement.retime import retime as RT  # noqa: PLC0415

    t = tmp_path / "clip_pl_rt"
    (t / "0").mkdir(parents=True)
    (t / "task_info.json").write_text(__import__("json").dumps(_ti()))
    final = {"qpos": q_final if q_final is not None else q_post}
    if retimed:
        final[RT.TAU_KEY] = np.arange(T, dtype=float)
        if q_post is not None:
            np.savez(t / "0" / "trajectory_kinematic.pre_rt.npz", qpos=q_post)
    np.savez(t / "0" / "trajectory_kinematic.npz", **final)
    if report:
        (t / XR.REPORT_REL).write_text("{}")
    if q_pre is not None:
        np.savez(t / "0" / f"trajectory_kinematic.{XR.XR_TAG}.npz", qpos=q_pre)
    return t


def _n():
    from dexengine.placement.gate import dyn_xr as DXR  # noqa: PLC0415

    return dict.fromkeys(DXR.COUNTS, 0)


def test_gate_reads_nothing_without_cross_repair(scene, tmp_path):
    from dexengine.placement.gate import dyn_xr as DXR  # noqa: PLC0415

    n = _n()
    assert DXR.reason(_tree(tmp_path, q_post=_q(0.006), report=False), n) is None
    assert n["trees"] == 1 and n["no_xr"] == 1 and n["read"] == 0


def test_gate_dies_on_a_report_without_its_input_copy(scene, tmp_path):
    from dexengine.placement.gate import dyn_xr as DXR  # noqa: PLC0415

    with pytest.raises(FileNotFoundError, match="without its input copy"):
        DXR.reason(_tree(tmp_path, q_post=_q(0.003)), _n())


def test_gate_dies_on_a_retimed_tree_without_the_copy_before_retiming(scene, tmp_path):
    from dexengine.placement.gate import dyn_xr as DXR  # noqa: PLC0415

    t = _tree(tmp_path, q_pre=_q(0.003), q_final=_q(0.003), retimed=True)
    with pytest.raises(FileNotFoundError, match="without the copy before retiming"):
        DXR.reason(t, _n())


def test_gate_fails_a_clip_cross_repair_deepens_past_the_true_shape_overlap(scene, tmp_path):
    """The gate reason of the 4 → 6 mm case on disk (retimed tree: the copy before retiming is the rows after cross
    repair) and its counts; the same tree with unchanged rows passes."""
    from dexengine.placement.gate import dyn_xr as DXR  # noqa: PLC0415

    n = _n()
    q4, q6 = _q(0.003, pen_mid=0.004), _q(0.003, pen_mid=0.006)
    why = DXR.reason(_tree(tmp_path / "a", q_pre=q4, q_post=q6, q_final=q6, retimed=True), n)
    assert why == "dyn_pair_xr:left_objectxright_object(6.0mm@30)"
    assert (n["read"], n["deepened"], n["turned_off"], n["drop"]) == (1, 1, 0, 1)
    assert DXR.reason(_tree(tmp_path / "b", q_pre=q4, q_post=q4.copy(), q_final=q4.copy(), retimed=True), _n()) is None
