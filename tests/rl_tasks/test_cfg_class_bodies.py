"""The RL cfg-class builders produce the stored class bodies, built on CPU with recording stand-ins for Isaac Lab.

Isaac Lab cannot be imported without the Kit app, so every module under the Isaac roots below is replaced, for this
test only, by a module whose names are recorder classes: calling one keeps its arguments. The builders then run as
plain Python and each built class is dumped field by field in declaration order (the order Isaac Lab reads), with
every recorded call, keyword order included. `python tests/rl_tasks/test_cfg_class_bodies.py --write` rewrites the
golden.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import io
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

GOLDEN = Path(__file__).with_name("golden") / "cfg_class_bodies.json"
FAKE_ROOTS = ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaaclab_assets", "isaacsim", "omni", "pxr", "carb")


class _Recorder(type):
    """Metaclass of the stand-ins: an unknown class attribute is a nested stand-in (RigidObjectCfg.InitialStateCfg)."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _stand_in(f"{cls.__name__}.{name}")
        setattr(cls, name, sub)
        return sub


def _stand_in(name: str) -> type:
    def __init__(self, *args, **kwargs):
        self.rec_args, self.rec_kwargs = args, kwargs

    return _Recorder(name, (), {"__init__": __init__, "rec_name": name})


class _FakeModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = (lambda cls: cls) if name == "configclass" else _stand_in(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _FakeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FAKE_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _FakeModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        return None


@contextlib.contextmanager
def _fake_isaac():
    """Install the stand-in modules; afterwards put back sys.modules under the Isaac roots and dexengine (the modules
    first imported here hold stand-ins, and the builders' robot module is replaced)."""
    saved = dict(sys.modules)
    finder = _FakeFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in [n for n in sys.modules if n.split(".")[0] in (*FAKE_ROOTS, "dexengine")]:
            if name not in saved:
                del sys.modules[name]
            elif sys.modules[name] is not saved[name]:
                sys.modules[name] = saved[name]


class _Mdp:
    """The mdp namespace the builders receive: each name is a stand-in class."""

    def __getattr__(self, name):
        value = _stand_in(f"mdp.{name}")
        setattr(self, name, value)
        return value


def _dump(v):
    if isinstance(type(v), _Recorder):
        return {
            "call": type(v).rec_name,
            "args": [_dump(x) for x in v.rec_args],
            "kwargs": [[k, _dump(x)] for k, x in v.rec_kwargs.items()],
        }
    if isinstance(v, _Recorder):
        return {"class": v.rec_name}
    if isinstance(v, type):
        return {"class": v.__qualname__, "module": v.__module__}
    if isinstance(v, types.FunctionType):
        return {"func": f"{v.__module__}.{v.__qualname__}"}
    if isinstance(v, dict):
        return {"dict": [[k, _dump(x)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        return {type(v).__name__: [_dump(x) for x in v]}
    return repr(v)


def _dump_class(cls) -> dict:
    """Fields in declaration order (nested classes recursively), annotations in order, and the base names."""
    fields = []
    for name, value in cls.__dict__.items():
        if name.startswith("__"):
            continue
        if isinstance(value, type) and value.__qualname__.startswith(cls.__qualname__ + "."):
            fields.append([name, _dump_class(value)])
        else:
            fields.append([name, _dump(value)])
    return {
        "name": cls.__qualname__,
        "bases": [b.__qualname__ for b in cls.__bases__],
        "annotations": [[k, str(a)] for k, a in cls.__dict__.get("__annotations__", {}).items()],
        "fields": fields,
        "post_init": "__post_init__" in cls.__dict__,
    }


class _Base:
    pass


class _Spec(types.SimpleNamespace):
    pass


_HAND = _Spec(
    name="hand_x",
    contact_body_regex_right="right_hand/(right_palm|right_thumb_tip).*",
    tip_body_names_right=("right_thumb_tip", "right_index_tip", "right_middle_tip", "right_ring_tip", "right_little"),
)


def _residual_cases() -> dict:
    from dexengine.rl.tasks.cfg import residual_scene as RS

    out = {}
    for bimanual in (False, True):
        cls, left_present, mass = RS._residual_terminations_cfg(
            bimanual=bimanual, left_uni=False, mdp=_Mdp(), object_mass_props="MASS", task_terminations_cfg_cls=_Base
        )
        out[f"terminations bimanual={bimanual}"] = [_dump_class(cls), left_present, mass]
    with pytest.raises(ValueError):
        RS._residual_terminations_cfg(
            bimanual=True, left_uni=True, mdp=_Mdp(), object_mass_props=None, task_terminations_cfg_cls=_Base
        )
    for left_present, left_uni in ((False, False), (True, False), (False, True)):
        cls, arm, to_left = RS._residual_scene_cfg(
            arm_spec=_Spec(name="arm_x"),
            eval_scene_cfg_cls=_Base,
            ground_friction=0.7,
            hand_spec=_HAND,
            left_present=left_present,
            left_uni=left_uni,
            max_depenetration_velocity=3.0,
            object_mass_props="MASS",
        )
        out[f"scene left_present={left_present} left_uni={left_uni}"] = [_dump_class(cls), arm.name, to_left("right_a")]
    return out


def _task_cases() -> dict:
    from dexengine.rl.tasks.cfg import task as T

    out = {}
    for bimanual, left_uni in ((False, False), (True, False), (False, True)):
        built = T.build_task_mdp_cfg_classes(
            mdp=_Mdp(),
            rewards_cfg_parent_cls=_Base,
            terminations_cfg_parent_cls=_Base,
            event_cfg_parent_cls=_Base,
            curriculum_cfg_parent_cls=_Base,
            bimanual=bimanual,
            left_uni=left_uni,
        )
        out[f"task bimanual={bimanual} left_uni={left_uni}"] = {k: _dump_class(c) for k, c in built.items()}
    return out


def _arm_env_case(monkeypatch_robot) -> dict:
    from dexengine.rl.tasks.cfg import arm_ik_env as AE

    gains = {
        "rigid_linear_damping": 0.1,
        "rigid_angular_damping": 0.2,
        "max_depenetration_velocity": 3.0,
        "finger_armature": 0.01,
        "mimic_kp": 20.0,
        "mimic_kd": 2.0,
        "mimic_armature": 0.02,
    }
    out = {}
    for relpath in ("link_head/zed", ""):
        arm = _Spec(name="arm_x", head_camera_prim_relpath=relpath)
        cls = AE._build_train_arm_env_cfg(
            _stand_in("TrajectoryManagerCfg"),
            _stand_in("SceneCfg"),
            _stand_in("ObservationsCfg"),
            _stand_in("RewardsCfg"),
            _stand_in("TerminationsCfg"),
            "/r/dir",
            "/arm/ik",
            "/wrist/poses",
            0.76,
            gains["rigid_linear_damping"],
            gains["rigid_angular_damping"],
            gains["max_depenetration_velocity"],
            gains["finger_armature"],
            gains["mimic_kp"],
            gains["mimic_kd"],
            gains["mimic_armature"],
            _HAND,
            _Spec(name="arm_spec_x"),
            _stand_in("ArmCurriculumCfg"),
            _stand_in("ArmIKActionsCfg"),
            _stand_in("ArmTrainEventCfg"),
            arm,
            lambda spec: {"hand": spec.name, "urdf": "h.urdf"},
        )
        robot = _Spec(prim_path="{ENV_REGEX_NS}/Robot", init_state=_Spec(pos=[0.0, 0.0, 0.1], rot=[1.0, 0.0, 0.0, 0.0]))
        cam = _Spec(prim_path="{ENV_REGEX_NS}/Robot/HeadCamera")
        env = _Spec(scene=_Spec(robot=robot, head_camera=cam), sim=_Spec(), terminations=_Spec())
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            cls.__post_init__(env)
        out[f"arm_env relpath={relpath!r}"] = {
            "class": _dump_class(cls),
            "robot_call": monkeypatch_robot[-1],
            "env": {
                "decimation": env.decimation,
                "episode_length_s": env.episode_length_s,
                "dt": repr(env.sim.dt),
                "render_interval": env.sim.render_interval,
                "head_camera": None if env.scene.head_camera is None else env.scene.head_camera.prim_path,
            },
            "stdout": printed.getvalue(),
        }
    return out


def _build_all() -> dict:
    calls = []

    def make_combo_articulation_cfg(arm_spec, hand_spec, **kwargs):
        calls.append([arm_spec.name, hand_spec.name, [[k, repr(v)] for k, v in kwargs.items()]])
        return _Spec(spawn=_Spec(usd_path="/usd/combo.usd"))

    robot_module = types.ModuleType("dexengine.rl.tasks.scene.robot")
    robot_module.make_combo_articulation_cfg = make_combo_articulation_cfg
    with _fake_isaac():
        sys.modules["dexengine.rl.tasks.scene.robot"] = robot_module
        out = {**_residual_cases(), **_task_cases(), **_arm_env_case(calls)}
    return json.loads(json.dumps(out))


@pytest.fixture
def no_flags(monkeypatch):
    for name in ("DEXENGINE_ARM_JOINT_RESIDUAL", "DEXENGINE_BI_SPLIT_ARTIC", "DEXENGINE_FH_NAN_RESET"):
        monkeypatch.delenv(name, raising=False)


def test_cfg_class_bodies_match_the_golden(no_flags):
    got = _build_all()
    want = json.loads(GOLDEN.read_text())
    assert sorted(got) == sorted(want)
    for case, body in want.items():
        assert got[case] == body, case


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(_build_all(), indent=1) + "\n")
    print(f"wrote {GOLDEN}")
