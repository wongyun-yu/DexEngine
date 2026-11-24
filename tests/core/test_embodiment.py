"""Embodiment spec files and their loader (dexengine.core.embodiment).

- The registries (HAND_REGISTRY · ARM_REGISTRY · MOUNT_REGISTRY · HAND_RL_RECIPES) are the spec files' objects.
- A new hand is added by config alone: a spec root holding the rby1 arm file, a renamed copy of the xhand_pro file
  (pointing at a tiny tracked vendor URDF) and its mount file resolves through the loader, and the merge CLI builds
  its robot URDF with no code edit; the mount distance and the adapter thickness move the mount joints by exactly
  their sum along the flange approach and change nothing else.
- Unknown keys, missing keys, wrong types and inconsistent files die with the file and the key.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from dexengine.core import embodiment as E

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "configs" / "robots"
FIXTURE_URDF = "tests/core/fixtures/embodiment/testhand_{side}.urdf"
NAME = "testhand"


def _spec_root(tmp: Path, adapter: float, face: float, hand_edit=None, mount_edit=None) -> Path:
    root = tmp / f"specs_{adapter}_{face}"
    for kind in E.KINDS:
        (root / kind).mkdir(parents=True)
    shutil.copy(SPECS / "arms" / "rby1.yaml", root / "arms" / "rby1.yaml")
    hand = yaml.safe_load((SPECS / "hands" / "xhand_pro.yaml").read_text(encoding="utf-8"))
    hand.update(
        name=NAME,
        display_name="Test hand",
        vendor_urdf=FIXTURE_URDF,
        vendor_meshes="",
        urdf_subtree_root="{side}_hand_link",
        mount_face_to_root=face,
        retarget_dirname=NAME,
    )
    (hand_edit or (lambda d: None))(hand)
    (root / "hands" / f"{NAME}.yaml").write_text(yaml.safe_dump(hand, sort_keys=False), encoding="utf-8")
    mount = yaml.safe_load((SPECS / "mounts" / "rby1+xhand_pro.yaml").read_text(encoding="utf-8"))
    mount.update(hand=NAME, translation={"stack": {"adapter_thickness": adapter}})
    (mount_edit or (lambda d: None))(mount)
    (root / "mounts" / f"rby1+{NAME}.yaml").write_text(yaml.safe_dump(mount, sort_keys=False), encoding="utf-8")
    return root


def test_registries_are_the_spec_files():
    from dexengine.core import arm_generations as AG
    from dexengine.core.arms import ARM_REGISTRY
    from dexengine.core.hands import HAND_REGISTRY, HAND_RL_RECIPES
    from dexengine.core.mounts import MOUNT_REGISTRY

    cat = E.default_catalog()
    assert sorted(HAND_REGISTRY) == sorted(p.stem for p in (SPECS / "hands").glob("*.yaml"))
    assert all(HAND_REGISTRY[h] is cat.hands[h] for h in HAND_REGISTRY)
    assert set(HAND_RL_RECIPES) == set(HAND_REGISTRY)
    assert all(ARM_REGISTRY[a] is s for a, s in cat.arms.items())
    assert all(MOUNT_REGISTRY[k] is m for k, m in cat.mounts.items())
    assert {f"{a}+{h}" for a, h in cat.mounts} == {p.stem for p in (SPECS / "mounts").glob("*.yaml")}
    gen = AG.ARM_CANON_GEN["rby1"]
    if not os.environ.get(E.base_pos_env_name("rby1")):
        assert ARM_REGISTRY["rby1"].base_pos == gen.base_pos  # base_pos: canon_generation
    assert ARM_REGISTRY["rby1"].skeleton.posture["torso_3"] == gen.torso3_pitch_rad  # {generation: torso3_pitch_rad}


def test_a_new_hand_resolves_from_config_alone(tmp_path, capsys):
    from dexengine.core.hands import HAND_REGISTRY

    root = _spec_root(tmp_path, adapter=0.004, face=0.002)
    emb = E.resolve("rby1", NAME, root)
    assert emb.hand.name == NAME and emb.mount.hand == NAME and emb.arm.name == "rby1"
    fl = emb.arm.flange
    want = tuple(p + a * (0.004 + 0.002) for p, a in zip(fl.xyz, fl.approach, strict=True))
    assert emb.mount.ee_to_wrist_offset_for("right") == want == emb.mount.ee_to_wrist_offset_for("left")
    assert want[2] == pytest.approx(-0.1261 - 0.006, abs=1e-15)
    err = capsys.readouterr().err
    assert re.search(rf"\[EMBODIMENT\] hand={NAME} arm=rby1 mount=rby1\+{NAME} spec_sha=[0-9a-f]{{16}} root=", err)
    assert NAME not in HAND_REGISTRY  # the tree's registry is untouched


def _merge(root: Path, out: Path) -> str:
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexengine.robots.urdf.merge",
            f"rby1+{NAME}",
            "--spec-root",
            str(root),
            "--out-root",
            str(out),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert f"[EMBODIMENT] hand={NAME} arm=rby1" in r.stderr
    from dexengine.core.arms import canon_variant

    d = f"rby1_{NAME}{canon_variant('rby1')}"
    return (out / d / f"{d}.urdf").read_text(encoding="utf-8")


_MOUNT = re.compile(r'(<joint name="(right|left)_mount_joint" type="fixed">\s*<origin xyz=")([^"]+)(")')


def test_merge_builds_a_new_hand_and_the_mount_moves_it(tmp_path):
    a = _merge(_spec_root(tmp_path, adapter=0.004, face=0.002), tmp_path / "out_a")
    b = _merge(_spec_root(tmp_path, adapter=0.010, face=0.005), tmp_path / "out_b")
    xyz = {}
    for tag, text in (("a", a), ("b", b)):
        got = {m.group(2): tuple(map(float, m.group(3).split())) for m in _MOUNT.finditer(text)}
        assert set(got) == {"right", "left"}, tag
        xyz[tag] = got
        assert '<link name="right_hand_index_link">' in text and '<link name="link_right_arm_6">' in text
    for side in ("right", "left"):
        (xa, ya, za), (xb, yb, zb) = xyz["a"][side], xyz["b"][side]
        assert (xa, ya) == (xb, yb) == (0.0, 0.0)
        assert za == pytest.approx(-0.1261 - 0.006, abs=1e-12)
        assert za - zb == pytest.approx(0.009, abs=1e-12)  # (0.010 + 0.005) − (0.004 + 0.002) along approach −z
    blank = lambda t: _MOUNT.sub(lambda m: m.group(1) + "·" + m.group(4), t)  # noqa: E731
    assert blank(a) == blank(b)  # nothing but the two mount origins differs


def _dies(root: Path, kind: str, match: str):
    with pytest.raises(E.SpecError, match=match):
        getattr(E.Catalog(root), kind)


def _set(key, value):
    return lambda d: d.__setitem__(key, value)


@pytest.mark.parametrize(
    "hand_edit, mount_edit, kind, match",
    [
        (_set("finger_stiffness", 1.0), None, "hands", r"unknown key\(s\) \['finger_stiffness'\]"),
        (lambda d: d.pop("palm_site_offset"), None, "hands", r"missing key\(s\) \['palm_site_offset'\]"),
        (_set("hand_qpos_dim", "27"), None, "hands", r"hand_qpos_dim: expected int"),
        (_set("name", "other"), None, "hands", "≠ the file name"),
        (lambda d: d.pop("rl_recipe"), None, "hands", r"missing key \['rl_recipe'\]"),
        (_set("finger_gains", "from_vibes"), None, "hands", "not a known rule"),
        (None, _set("adapter", 0.01), "mounts", r"unknown key\(s\) \['adapter'\]"),
        (None, lambda d: d.pop("provisional"), "mounts", r"missing key\(s\) \['provisional'\]"),
        (
            None,
            _set("rotation", {"right": [1.0, 0.0, 0.0, 0.0], "left": [1.0, 0.0, 0.0, 0.0]}),
            "mounts",
            "maps the hand's \\+z",
        ),
        (None, _set("translation", {"stack": {"adapter_thickness": 0.0}, "measured": {}}), "mounts", "exactly one"),
        (None, _set("ee_to_wrist_offset", [0.0, 0.0, 0.0]), "mounts", "computed from translation"),
    ],
)
def test_a_bad_spec_dies(tmp_path, hand_edit, mount_edit, kind, match):
    _dies(_spec_root(tmp_path, 0.004, 0.002, hand_edit, mount_edit), kind, match)


def _no_torso(arm):
    arm["torso_ruler"] = False


def _bad_root(arm):
    arm["skeleton"]["root"] = "base"


@pytest.mark.parametrize(
    "edit, match",
    [
        (_bad_root, r"skeleton: unknown key\(s\) \['root'\]"),
        (_no_torso, "torso_ruler False disagrees with the collider revision"),
    ],
)
def test_a_bad_arm_spec_dies(tmp_path, edit, match):
    root = _spec_root(tmp_path, 0.004, 0.002)
    arm = yaml.safe_load((root / "arms" / "rby1.yaml").read_text(encoding="utf-8"))
    edit(arm)
    (root / "arms" / "rby1.yaml").write_text(yaml.safe_dump(arm, sort_keys=False), encoding="utf-8")
    _dies(root, "arms", match)


def test_a_hand_without_a_mount_dies(tmp_path):
    root = _spec_root(tmp_path, 0.004, 0.002)
    (root / "mounts" / f"rby1+{NAME}.yaml").unlink()
    _dies(root, "mounts", rf"hand\(s\) \['{NAME}'\] have no mount file")
