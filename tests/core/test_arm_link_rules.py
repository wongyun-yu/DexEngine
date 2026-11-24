"""A second arm loads from its spec file: its body generations and placement link rules follow its file, not RB-Y1's.

The arm spec owns the link-name facts placement reads: the arm and EE links (body_name_fmt · ee_body_index), the torso
group (skeleton root · torso_link_prefixes), the idle arm's rest pose (idle_qpos) and the reach-map desk classes
(reach_clearance_links). A renamed copy of the rby1 file with another link spelling, another torso and another rest
pose resolves through the loader, and group_of · side_of_link · is_hand_link · idle_arm · reach_map._link_groups ·
the hand look's EE token give that arm's answers (an RB-Y1 link name means nothing to it). Its own body generations
(generations · collider_rev — none here) give its base (base_pos: canon_generation) and its generation table. A
malformed rest pose, class table or generations section dies with the file. The arm's defaults the judge and the RL
robot read (drives, head, limb joints, ported-placement base, generation pitch joint) come from the same file: an arm
without a head states head_drive null, an arm without ported placements port_base_pos null.
"""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import pytest
import yaml

from dexengine.core import embodiment as E

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "configs" / "robots"
ARM = "testarm"
IDLE = {"right": [0.1, -0.2, 0.0, -0.4, 0.0, 0.0], "left": [0.1, 0.2, 0.0, -0.4, 0.0, 0.0]}
GEN = {"variant": "_t1", "base_pos": [0.0, 0.0, 0.5], "torso3_pitch_rad": -0.3, "chest_pitch_deg": 0.0}


def _arm_doc(edit=None) -> dict:
    """rby1's arm file renamed, with 6 joints spelled ``<side>_j<k>`` / ``<side>_seg<k>`` (1-based), a chest torso, its
    own rest pose and reach classes, its own body generations (base from the canonical one) and no collider revision
    (no torso)."""
    d = yaml.safe_load((SPECS / "arms" / "rby1.yaml").read_text(encoding="utf-8"))
    d.update(
        name=ARM,
        joint_prefix=ARM,
        n_joints=6,
        joint_classes=[{"suffix": "[1-6]", "indices": [1, 2, 3, 4, 5, 6], "velocity_limit": 2.0, "effort_limit": 20.0}],
        ee_body="seg6",
        base_pos="canon_generation",
        head_joint_names=[],
        joint_name_fmt="{side}_j{index}",
        index_base=1,
        body_name_fmt="{side}_seg{index}",
        ee_body_index=6,
        torso_ruler=False,
        torso_link_prefixes=["chest_"],
        idle_qpos={s: list(q) for s, q in IDLE.items()},
        reach_clearance_links={"arm": [3, 4], "hand": [5, 6]},
        generations={
            "canon": {**GEN, "why": "test body"},
            "render": {"variant": "_t1_mi", "material_of": "_t1", "why": "test plate"},
        },
        collider_rev=None,
        arm_drive={"kp": 300.0, "kd": 30.0},
        head_drive=None,
        limb_joints={"shoulder": 1, "elbow": 3, "wrist": 5},
        port_base_pos=None,
    )
    d["skeleton"] = dict(d["skeleton"], root_link="pedestal", posture={"waist": {"generation": "torso3_pitch_rad"}})
    (edit or (lambda x: None))(d)
    return d


def _root(tmp: Path, edit=None) -> Path:
    root = tmp / "specs"
    for kind in E.KINDS:
        (root / kind).mkdir(parents=True)
    shutil.copy(SPECS / "arms" / "rby1.yaml", root / "arms" / "rby1.yaml")
    shutil.copy(SPECS / "hands" / "xhand_pro.yaml", root / "hands" / "xhand_pro.yaml")
    shutil.copy(SPECS / "mounts" / "rby1+xhand_pro.yaml", root / "mounts" / "rby1+xhand_pro.yaml")
    (root / "arms" / f"{ARM}.yaml").write_text(yaml.safe_dump(_arm_doc(edit), sort_keys=False), encoding="utf-8")
    return root


@pytest.fixture()
def arms(tmp_path, monkeypatch):
    """(second arm, rby1) loaded from one spec root; the second arm joins this process's registry for the readers."""
    from dexengine.core import arms as AS

    cat = E.catalog(_root(tmp_path))
    monkeypatch.setitem(AS.ARM_REGISTRY, ARM, cat.arms[ARM])
    return cat.arms[ARM], cat.arms["rby1"]


def test_link_groups_follow_the_arm_spec(arms):
    from dexengine.placement.collision.link_shapes import group_of
    from dexengine.placement.repair.cross_util import is_hand_link, side_of_link

    arm, rby1 = arms
    want = {
        "right_seg2": ("RA", "right", False),
        "right_seg6": ("RH", "right", True),  # the EE link sits with the hand
        "left_seg1": ("LA", "left", False),
        "left_thumb_tip": ("LH", "left", True),  # a merged hand link: its side prefix
        "pedestal": ("TORSO", None, False),  # the skeleton root
        "chest_2": ("TORSO", None, False),
        "link_torso_0": (None, None, False),  # RB-Y1's torso is not this arm's
        "link_right_arm_3": (None, None, False),
        "world": (None, None, False),
    }
    got = {n: (group_of(n, arm), side_of_link(n, arm), is_hand_link(n, arm)) for n in want}
    assert got == want
    assert (group_of("link_torso_0", rby1), group_of("link_right_arm_6", rby1)) == ("TORSO", "RH")
    assert group_of("right_seg6", None) == "RH"  # no arm: hand prefix only


def test_idle_pose_and_reach_classes_follow_the_arm_spec(arms):
    import numpy as np

    from dexengine.placement.collision import idle_arm as IA
    from dexengine.placement.reach_map import build as RB

    arm, _ = arms
    q = IA.idle_arm_q(ARM, "left", 3)
    assert q.shape == (3, 6) and np.array_equal(q[2], np.array(IDLE["left"]))
    chain = ["world", "pedestal", "chest_1", *[f"right_seg{k}" for k in range(1, 7)], "right_hand_base"]
    sa, sh = RB._link_groups(types.SimpleNamespace(arm=ARM, side="right"), chain)
    assert [chain[i] for i in sa] == ["right_seg3", "right_seg4"]
    assert [chain[i] for i in sh] == ["right_seg5", "right_seg6", "right_hand_base"]
    with pytest.raises(SystemExit, match="reach_clearance_links names none"):
        RB._link_groups(types.SimpleNamespace(arm=ARM, side="right"), ["world", "link_right_arm_3"])


def test_hand_look_reaches_the_arms_ee_link(arms):
    import re

    pytest.importorskip("pxr")
    from dexengine.robots.usd import hand_look as HLK

    ee = {a.name: HLK.look_for(a.name, "wuji2").rules[1][0] for a in arms}  # the glove rule on the EE link
    assert re.fullmatch(ee[ARM], "{side}_seg6") and not re.fullmatch(ee[ARM], "link_{side}_arm_6")
    assert ee["rby1"] == r"link_\{side\}_arm_6"
    assert (
        HLK._in_scope("{side}_seg6", HLK.ee_link(ARM))[1]
        and not HLK._in_scope("link_{side}_arm_6", HLK.ee_link(ARM))[1]
    )


def test_judge_and_rl_defaults_follow_the_arm_spec(arms):
    arm, rby1 = arms
    assert (arm.arm_drive.kp, arm.head_drive, arm.port_base_pos) == (300.0, None, None)
    assert arm.generation_pitch_joint == "waist" and rby1.generation_pitch_joint == "torso_3"
    assert (rby1.head_drive.pitch_deg, rby1.limb_joints) == (30.0, {"shoulder": 2, "elbow": 3, "wrist": 4})


def test_the_generation_pitch_joint_is_read_by_name(tmp_path):
    from dexengine.placement import arm_ik as AIP

    u = tmp_path / "a.urdf"
    u.write_text('<robot><joint name="waist" type="fixed"><origin rpy="0 -0.3 0" xyz="0 0 0"/></joint></robot>')
    assert AIP.urdf_torso3_pitch(u, "waist") == -0.3
    assert AIP.urdf_torso3_pitch(u, "torso_3") is None and AIP.urdf_torso3_pitch(u, None) is None
    with pytest.raises(SystemExit, match="no origin rpy on the generation pitch joint 'torso_3'"):
        AIP._verify_urdf_posture(types.SimpleNamespace(ARM_REGISTRY=E.default_catalog().arms), "rby1", u, "")


def test_body_generations_follow_the_arm_spec(arms):
    from dexengine.core import arm_generations as AG

    arm, rby1 = arms
    assert arm.base_pos == (0.0, 0.0, 0.5) and rby1.base_pos == AG.ARM_CANON_GEN["rby1"].base_pos
    gens, rev = AG.body_sections(_arm_doc(), "testarm.yaml")
    table = AG.generation_table(*gens, rev)
    assert rev is None and list(table) == ["_t1", "_t1_mi", "_t1_truncated", "_t1_truncated_mi"]
    assert table["_t1_mi"].material_of == "_t1" and table["_t1_mi"].base_pos == (0.0, 0.0, 0.5)


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda d: d["idle_qpos"]["left"].pop(), r"idle_qpos \{'left': 5\}"),
        (lambda d: d.update(reach_clearance_links={"arm": [3]}), r"reach_clearance_links keys \['arm'\]"),
        (lambda d: d.pop("collider_rev"), r"missing key \['collider_rev'\]"),
        (lambda d: d["generations"]["canon"].pop("why"), r"generations.canon: keys .* exactly"),
        (lambda d: d["generations"]["render"].update(material_of="_t2"), "material_of '_t2'"),
        (lambda d: d.pop("port_base_pos"), r"missing key\(s\) \['port_base_pos'\]"),
        (lambda d: d.update(head_drive=None, head_joint_names=["neck"]), "arm_drive / head_drive are stated exactly"),
        (lambda d: d.update(limb_joints={"shoulder": 0, "elbow": 3, "wrist": 5}), r"limb_joints .* each a joint 1..6"),
        (lambda d: d.update(generation_pitch_joint="waist"), "generation_pitch_joint is computed"),
    ],
)
def test_a_bad_link_rule_dies(tmp_path, edit, match):
    with pytest.raises(E.SpecError, match=match):
        E.catalog(_root(tmp_path, edit)).arms  # noqa: B018 — the load is the test
