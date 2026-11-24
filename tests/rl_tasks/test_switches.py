"""RLSwitches: defaults, the switches file, the trainer-side checks and the text form readers get."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.rl.tasks import env_flags, switches

ROOT = Path(__file__).resolve().parents[2]

#: Independent statement of each switch's unset text (what a reader gets when nothing sets the switch); the defaults
#: of RLSwitches are checked against it.
UNSET_TEXT = {
    "DEXENGINE_PLUGINS": "",
    "DEXENGINE_FH_NAN_RESET": "",
    "DEXENGINE_FH_PALM_POINT": "",
    "DEXENGINE_RIGHT_PARK": "0",
    "DEXENGINE_ARM_KP": None,
    "DEXENGINE_ARM_KD": None,
    "DEXENGINE_LEFT_ARM_KP_SCALE": "",
    "DEXENGINE_ARM_FF": "1",
    "DEXENGINE_ARM_FF_MODE": "lead",
    "DEXENGINE_ARM_FF_VEL_SCALE": "1.0",
    "DEXENGINE_ARM_JOINT_RESIDUAL": "0",
    "DEXENGINE_ARM_JOINT_RESIDUAL_RAD": "0.15",
    "DEXENGINE_ACTION_ABS": "0",
    "DEXENGINE_ZERO_RESIDUAL": "0",
    "DEXENGINE_ARM_IK_RESET_SUFFIX": "",
    "DEXENGINE_ENVZ": "1",
    "DEXENGINE_ROBOT_Z_OFF": "0",
    "DEXENGINE_MOUNT_TOL": "0.0025",
    "DEXENGINE_COLL_PRESET": "",
    "DEXENGINE_BI_SPLIT_ARTIC": "0",
    "DEXENGINE_BI_SPLIT_OFF": "",
    "DEXENGINE_BI_SELFCOLL": "0",
    "DEXENGINE_DESK_ROBOT_NOCOLL": "0",
    "DEXENGINE_ARM_OBJ_NOCOLL": "0",
    "DEXENGINE_EVAL_CROSS_OFF": "0",
    "DEXENGINE_COLLCURR_CATS": "hh,ho,oo",
    "DEXENGINE_ALL_OBJ_COLLISION": "",
    "DEXENGINE_OBJ_SPAWN_HOLD": "0",
    "DEXENGINE_OBJ_REF_VEL_INIT": "0",
    "DEXENGINE_PHYSX_LITE": "0",
    "DEXENGINE_PHYSX_CAP_AUTO": "0",
    "DEXENGINE_PHYSX_CONTACT_CAP_LOG2": "",
    "DEXENGINE_PHYSX_PATCH_CAP_LOG2": "",
    "DEXENGINE_DISABLE_ROBOT_GRAVITY": None,
    "DEXENGINE_LEFT_GATE_RAMP": "",
    "DEXENGINE_OBJTRACK_DEBUG": None,
    "DEXENGINE_CAMERA_RES": "1920",
    "DEXENGINE_CAMERA_FOCAL_MM": "8.0",
    "DEXENGINE_KEY_LIGHT": "",
}
#: Old names kept on the environment boundary (dexengine.core.config.ENV_VARS: core, record and placement read them).
KEPT_ON_ENV = ("DEXENGINE_SEQ_LIST", "DEXENGINE_ROSTER", "DEXENGINE_ARM_IK_DIR_SUFFIX")


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(switches, "_STATE", {"path": None, "sw": None, "given": frozenset(), "env_reads": {}})
    for name in (*switches.FIELD_OF, switches.SWITCHES_ENV):
        monkeypatch.delenv(name, raising=False)


def test_defaults_equal_the_old_unset_values():
    assert set(switches.FIELD_OF) == set(UNSET_TEXT)
    assert {n: switches.unset_text(f) for n, f in switches.FIELD_OF.items()} == UNSET_TEXT
    sw = switches.RLSwitches()
    for name, text in UNSET_TEXT.items():  # the typed default reads as the unset text
        assert sw == switches.build({} if text is None else {switches.FIELD_OF[name]: text}), name
    for name, text in UNSET_TEXT.items():  # a process without a file reads the same texts
        assert env_flags.get(name) == text, name


def test_kept_names_are_process_boundary_variables():
    from dexengine.core import config

    assert all(n in config.ENV_VARS and n not in switches.FIELD_OF for n in KEPT_ON_ENV)


def test_unknown_switch_names_fail_loudly(tmp_path):
    with pytest.raises(switches.SwitchError, match="unknown switch"):
        switches.build({"arm_fff": "0"})
    with pytest.raises(switches.SwitchError, match="not switches"):
        switches.from_env_names({"DEXENGINE_ARM_FFF": "0"})
    with pytest.raises(switches.SwitchError, match="not a switch"):
        switches.text("DEXENGINE_ARM_FFF")
    p = switches.write(tmp_path / "s.json", switches.RLSwitches())
    body = json.loads(p.read_text())
    body["values"]["arm_fff"] = 0
    p.write_text(json.dumps(body))
    with pytest.raises(switches.SwitchError, match="unknown"):
        switches.read(p)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/stage3_rl.py"), "r", "--set", "arm_fff=0"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert r.returncode == 2 and "not a switch" in r.stderr


@pytest.mark.parametrize(
    "field,bad", [("arm_ff", "2"), ("camera_res", "x"), ("obj_spawn_hold", "-1"), ("arm_ff_vel_scale", None)]
)
def test_bad_values_fail(field, bad):
    with pytest.raises(switches.SwitchError):
        switches.build({field: bad})


def test_file_round_trip_and_derive(tmp_path):
    sw, given = switches.from_env_names(
        {
            "DEXENGINE_OBJ_SPAWN_HOLD": "10",
            "DEXENGINE_LEFT_ARM_KP_SCALE": "1.0",
            "DEXENGINE_ROBOT_Z_OFF": "0",
            "DEXENGINE_KEY_LIGHT": "3200",
        }
    )
    p = switches.write(tmp_path / "a.json", sw, given)
    assert switches.read(p) == (sw, given)
    assert sw.obj_spawn_hold == 10 and sw.left_arm_kp_scale == 1.0 and sw.key_light == "3200"
    d = switches.derive(p, {"arm_ff_vel_scale": 0.0, "all_obj_collision": "1"}, tmp_path / "b.json")
    sw2, given2 = switches.read(d)
    assert (sw2.arm_ff_vel_scale, sw2.all_obj_collision, sw2.obj_spawn_hold) == (0.0, True, 10)
    assert given2 == given | {"arm_ff_vel_scale", "all_obj_collision"}
    body = json.loads(Path(d).read_text())
    body["values"]["arm_ff"] = "1"  # text is not a typed value
    Path(d).write_text(json.dumps(body))
    with pytest.raises(switches.SwitchError, match="typed"):
        switches.read(d)


def test_trainer_reads_the_file_with_the_old_text_semantics(tmp_path, monkeypatch):
    sw, given = switches.from_env_names({"DEXENGINE_OBJ_SPAWN_HOLD": "10", "DEXENGINE_BI_SPLIT_ARTIC": "1"})
    p = switches.write(tmp_path / "s.json", sw, given)
    monkeypatch.setenv(switches.SWITCHES_ENV, str(p))
    monkeypatch.setenv("DEXENGINE_BI_SPLIT_ARTIC", "1")  # an inherited copy that agrees is harmless
    assert "given 2" in switches.require_file()
    assert env_flags.get("DEXENGINE_OBJ_SPAWN_HOLD") == "10"
    assert env_flags.get("DEXENGINE_ARM_FF") == "1"
    assert env_flags.get("DEXENGINE_ARM_FF", "(unset)") == "(unset)"  # not given: the caller's default
    assert env_flags.get("DEXENGINE_ARM_KP") is None
    monkeypatch.setenv("DEXENGINE_CAMERA_RES", "64")  # environment no longer reaches a reader
    assert env_flags.get("DEXENGINE_CAMERA_RES") == "1920"
    switches.override(right_park=True)
    assert env_flags.get("DEXENGINE_RIGHT_PARK") == "1" and switches.current().right_park
    assert switches.legacy_environ()["DEXENGINE_RIGHT_PARK"] == "1"
    assert "DEXENGINE_CAMERA_RES" not in switches.legacy_environ()


def test_trainer_stops_without_a_good_file(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="is not set"):
        switches.require_file()
    p = switches.write(tmp_path / "s.json", switches.RLSwitches())
    monkeypatch.setenv(switches.SWITCHES_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="not found"):
        switches.require_file()
    monkeypatch.setenv(switches.SWITCHES_ENV, str(p))
    monkeypatch.setenv("DEXENGINE_ARM_FF", "0")  # a value the file contradicts
    with pytest.raises(SystemExit, match="DEXENGINE_ARM_FF='0'"):
        switches.require_file()
    monkeypatch.delenv("DEXENGINE_ARM_FF")
    monkeypatch.delenv(switches.SWITCHES_ENV)
    monkeypatch.setenv("DEXENGINE_CAMERA_RES", "64")
    assert env_flags.get("DEXENGINE_CAMERA_RES") == "64"  # read before the variable was set
    monkeypatch.delenv("DEXENGINE_CAMERA_RES")
    monkeypatch.setenv(switches.SWITCHES_ENV, str(p))
    with pytest.raises(SystemExit, match="read before it was loaded"):
        switches.require_file()


def test_a_set_variable_loads_the_file_on_the_first_read(tmp_path, monkeypatch):
    """No read in a process that has DEXENGINE_RL_SWITCHES falls back to a default, even before require_file."""
    sw, given = switches.from_env_names({"DEXENGINE_BI_SPLIT_ARTIC": "1"})
    p = switches.write(tmp_path / "s.json", sw, given)
    monkeypatch.setenv(switches.SWITCHES_ENV, str(p))
    assert env_flags.get("DEXENGINE_BI_SPLIT_ARTIC") == "1" and switches.loaded()
    same = switches.write(tmp_path / "same.json", sw, given)  # another dump of a persistent session
    monkeypatch.setenv(switches.SWITCHES_ENV, str(same))
    assert switches.current().bi_split_artic
    other = switches.write(tmp_path / "other.json", switches.RLSwitches())
    monkeypatch.setenv(switches.SWITCHES_ENV, str(other))
    with pytest.raises(SystemExit, match="other switches"):
        env_flags.get("DEXENGINE_BI_SPLIT_ARTIC")


def test_a_file_with_an_unstated_change_is_refused(tmp_path):
    p = switches.write(tmp_path / "s.json", switches.build({"envz": "0"}))  # changed but not given
    with pytest.raises(switches.SwitchError, match="not given"):
        switches.read(p)


def test_trainer_entry_loads_the_file_before_any_switch_reader():
    """dexengine/rl/cli.py calls require_file() before it imports cli_entry or any RL task module but switches."""
    import ast

    tree = ast.parse((ROOT / "dexengine/rl/cli.py").read_text(encoding="utf-8"))
    req = imp = None
    for node in tree.body:
        line = node.lineno
        if req is None and "require_file" in ast.unparse(node):
            req = line
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or node.names[0].name
            names = [a.name for a in node.names]
            if mod.startswith(("dexengine.rl.cli_entry", "dexengine.rl.tasks", "dexengine.rl.env_setup")) and names != [
                "switches"
            ]:
                imp = line if imp is None else imp
    assert req is not None and imp is not None and req < imp


def test_override_needs_the_file():
    with pytest.raises(switches.SwitchError, match="before the switches file"):
        switches.override(right_park=True)


def test_dump_env_view_restores_the_old_names(tmp_path):
    sw, given = switches.from_env_names({"DEXENGINE_PHYSX_LITE": "1", "DEXENGINE_KEY_LIGHT": "3200"})
    p = switches.write(tmp_path / "s.json", sw, given)
    env = {"DEXENGINE_ARM": "rby1", switches.SWITCHES_ENV: str(p)}
    assert switches.dump_env_view(env) == {
        "DEXENGINE_ARM": "rby1",
        "DEXENGINE_PHYSX_LITE": "1",
        "DEXENGINE_KEY_LIGHT": "3200",
    }


def test_cli_writes_a_file_for_a_manual_run(tmp_path):
    out = tmp_path / "m.json"
    assert switches.main(["--write", str(out), "--set", "physx_lite=1", "--set", "camera_res=480"]) == 0
    sw, given = switches.read(out)
    assert (sw.physx_lite, sw.camera_res, given) == (True, 480, {"physx_lite", "camera_res"})
    assert switches.main(["--write", str(out), "--set", "nope=1"]) == 2
