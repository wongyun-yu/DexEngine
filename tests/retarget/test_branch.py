"""Finger branch windows (ik/branch.py): the prior rule that declares them and the re-solve that applies them."""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from dexengine.core import paths  # noqa: E402
from dexengine.retarget.ik import branch, prior  # noqa: E402

RIGS = ("sharpa_wave", "dg5f_s", "wuji2", "xhand_pro")


def _conf(spec, sides=("right", "left")):
    return prior.build_prior_conf(str(paths.hand_rig_dir(spec) / "retarget_prior.yaml"), spec, sides)[0]


@pytest.mark.parametrize("spec", RIGS)
def test_only_declared_hands_get_windows(spec):
    """A hand without `branch_window_deg` has no `branch_windows` key (its conf and outputs stay as they were)."""
    conf = _conf(spec)
    if spec == "dg5f_s":
        got = {w["joint"]: (math.degrees(w["lo"]), math.degrees(w["hi"])) for w in conf["branch_windows"]}
        assert got == {"right_joint_1_2": (-90.0, 0.0), "left_joint_1_2": (0.0, 90.0)}
        assert {w["finger"] for w in conf["branch_windows"]} == {"right_thumb", "left_thumb"}
    else:
        assert conf is None or "branch_windows" not in conf


def test_one_hand_scene_keeps_its_side():
    conf = _conf("dg5f_s", ("left",))
    assert [w["joint"] for w in conf["branch_windows"]] == ["left_joint_1_2"]


def test_window_rules_die(tmp_path):
    src = (paths.hand_rig_dir("dg5f_s") / "retarget_prior.yaml").read_text()
    line = "  branch_window_deg: {right_joint_1_2: [-90, 0], left_joint_1_2: [0, 90]}\n"
    assert line in src
    for bad, msg in (
        ("  branch_window_deg: {right_joint_9_9: [-90, 0]}\n", "not a joint of a declared"),
        ("  branch_window_deg: {right_joint_1_2: [0, -90]}\n", "lo < hi"),
        ("  branch_window_deg: {right_joint_1_2: [-90, 0], right_joint_1_3: [0, 45]}\n", "one per finger"),
    ):
        p = tmp_path / "retarget_prior.yaml"
        p.write_text(src.replace(line, bad))
        with pytest.raises(Exception, match=msg):
            prior.build_prior_conf(str(p), "dg5f_s", ("right", "left"))


def test_resolve_moves_a_z_thumb_to_the_window_and_keeps_its_tip():
    """A mirrored ("Z") thumb measured on the canonical TACO set (right hand: proximal phalanx across the palm, distal
    joints folded back) is re-solved inside the window with the same tip; every other coordinate is untouched."""
    m = mujoco.MjModel.from_xml_path(str(paths.hand_rig_dir("dg5f_s") / "right.xml"))
    conf = _conf("dg5f_s", ("right",))
    adr = {n: int(m.jnt_qposadr[m.joint(f"right_joint_1_{n}").id]) for n in (1, 2, 3, 4)}
    q = np.zeros((6, m.nq))
    rng = np.random.default_rng(0)
    other = [i for i in range(m.nq) if i not in adr.values()]
    q[:, other] = rng.uniform(-0.05, 0.05, size=(6, len(other)))
    z = np.radians([-56.4, -152.0, 63.7, 20.5])
    for t in range(6):
        q[t, [adr[1], adr[2], adr[3], adr[4]]] = z + np.radians(2.0 * t) * np.array([1, 0, -1, 1])
    d = mujoco.MjData(m)
    site = m.site("right_thumb_tip").id

    def tips(qq):
        out = []
        for row in qq:
            d.qpos[:] = row
            mujoco.mj_kinematics(m, d)
            out.append(d.site_xpos[site].copy())
        return np.array(out)

    before, keep = tips(q), q[:, other].copy()
    reports = branch.resolve_branches(m, q, conf)
    assert len(reports) == 1 and reports[0].outside_raw == 6
    assert np.all(np.abs(q[:, adr[2]]) <= math.radians(90.0) + 1e-9)
    assert np.abs(tips(q) - before).max() < 1e-3
    assert q[:, other].tobytes() == keep.tobytes()
    assert reports[0].marker().startswith("[BRANCH] right_thumb right_joint_1_2 window=[-90.0,0.0]deg")


def test_no_window_is_a_no_op():
    m = mujoco.MjModel.from_xml_path(str(paths.hand_rig_dir("wuji2") / "right.xml"))
    q = np.random.default_rng(1).uniform(-0.1, 0.1, size=(4, m.nq))
    keep = q.copy()
    assert branch.resolve_branches(m, q, _conf("wuji2", ("right",))) == []
    assert q.tobytes() == keep.tobytes()


def _canon_window(hand, monkeypatch):
    """The window run as truncate.py and wrist_shift.finger_ranges run it, on the hand's committed full-robot URDF."""
    import sys
    import xml.etree.ElementTree as ET

    from dexengine.robots.urdf import truncate as T

    monkeypatch.setenv("RETARGET_PY", sys.executable)
    cdir = paths.combo_dirname(paths.combo_for(hand))
    lim = T._limits(ET.parse(T.FULL_ROBOT_URDF_DIR / cdir / f"{cdir}.urdf").getroot())
    return lim, T._window(paths.hand_spec_dir(hand), lim)


@pytest.mark.parametrize("hand", ["dg5f_s", "wuji2"])
def test_window_exports_branch_windows_apart_from_rows(hand, monkeypatch):
    """The windows leave the window under their own key; `rows` (what the sim asset cuts) keep the hyperextension cut
    alone — on DG-5F-S joint_1_2 is a clip target whose cut leaves the full 153 deg."""
    _, win = _canon_window(hand, monkeypatch)
    got = {
        w["joint"]: (round(math.degrees(w["lo"]), 6), round(math.degrees(w["hi"]), 6)) for w in win["branch_windows"]
    }
    want = {"right_joint_1_2": (-90.0, 0.0), "left_joint_1_2": (0.0, 90.0)} if hand == "dg5f_s" else {}
    assert got == want
    rows = {r["joint"]: round(math.degrees(r["new_hi"] - r["new_lo"]), 6) for r in win["rows"] if r["joint"] in want}
    assert rows == ({"right_joint_1_2": 153.0, "left_joint_1_2": 153.0} if want else {})


def test_finger_ranges_apply_the_window(monkeypatch):
    """The reference finger range = the same range with the branch step left out, narrowed on the window joints only
    (measured against finger_ranges itself, so the IK cut it applies — not the sim asset's — is the baseline)."""
    from dexengine.placement.repair import wrist_shift as WS

    lim, _ = _canon_window("dg5f_s", monkeypatch)
    vendor = {k: v for k, v in lim.items() if "joint_" in k and k.split("_", 1)[0] in ("right", "left")}
    monkeypatch.setattr(WS, "_RANGES", {})
    ref = WS.finger_ranges("dg5f_s", vendor, "bimanual")
    monkeypatch.setattr(WS, "_RANGES", {})
    monkeypatch.setattr(WS, "branch_cut", lambda ranges, windows, what: "")
    plain = WS.finger_ranges("dg5f_s", vendor, "bimanual")
    assert ref["right_joint_1_2"] == (
        max(plain["right_joint_1_2"][0], -math.pi / 2),
        min(plain["right_joint_1_2"][1], 0.0),
    )
    assert ref["left_joint_1_2"] == (max(plain["left_joint_1_2"][0], 0.0), min(plain["left_joint_1_2"][1], math.pi / 2))
    assert ref["right_joint_1_2"] != plain["right_joint_1_2"]
    assert {k: v for k, v in ref.items() if not k.endswith("joint_1_2")} == {
        k: v for k, v in plain.items() if not k.endswith("joint_1_2")
    }


def test_branch_cut_rules():
    from dexengine.placement.repair.wrist_shift import branch_cut

    r = {"a": (-2.0, 2.0), "b": (0.0, 1.0)}
    assert branch_cut(r, [], "x") == "" and r == {"a": (-2.0, 2.0), "b": (0.0, 1.0)}
    assert branch_cut(r, [{"joint": "a", "lo": -1.0, "hi": 3.0}], "x").startswith("a=[")
    assert r == {"a": (-1.0, 2.0), "b": (0.0, 1.0)}
    with pytest.raises(RuntimeError, match="not in the asset ranges"):
        branch_cut(r, [{"joint": "c", "lo": 0.0, "hi": 1.0}], "x")
    with pytest.raises(RuntimeError, match="misses the range"):
        branch_cut(r, [{"joint": "b", "lo": 2.0, "hi": 3.0}], "x")
