"""Head-pitch hold in the arm-IK finger write (dexengine/rl/tasks/mdp/actions/arm_ik_kinematics.py `_apply_fingers`).

The joint id and the 0.8 rad target are built once and reused. Before/after proof: the method (extracted from the
source, its module imports Isaac Lab) runs on a fake asset next to the pre-change head block (kept here verbatim as the
oracle); over repeated calls both make the same set_joint_position_target calls with byte-equal targets and ids, with
and without a head joint."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[2] / "dexengine" / "rl" / "tasks" / "mdp" / "actions" / "arm_ik_kinematics.py"


def _method():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_apply_fingers")
    fn.args.args[1].annotation = None
    ns = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(SRC), "exec"), ns)
    return ns["_apply_fingers"]


def _old_head_block(self):  # the pre-change code, verbatim
    head_joint_names = [n for n in self._asset.joint_names if "head_pitch" in n]
    if head_joint_names:
        head_id = self._asset.joint_names.index(head_joint_names[0])
        head_target = torch.full((self.num_envs, 1), 0.8, device=self.device)
        self._asset.set_joint_position_target(head_target, joint_ids=[head_id])


class _Asset:
    def __init__(self, names, n):
        self.joint_names = names
        self.calls = []
        self.target = torch.zeros(n, len(names))

    def set_joint_position_target(self, t, joint_ids):
        self.target[:, joint_ids] = t  # Isaac Lab copies into its buffer the same way
        self.calls.append((tuple(joint_ids), t.clone()))


class _Term:
    def __init__(self, names, n=5):
        self._asset, self.num_envs, self.device = _Asset(names, n), n, "cpu"
        self._processed_actions = torch.zeros(n, 12)
        self._finger_range, self._finger_lower = torch.ones(6), torch.zeros(6)
        self._finger_joint_ids = list(range(6))
        self._park = False
        self._has_coupling = False


NAMES = [f"finger_{i}" for i in range(6)] + ["torso_0", "head_yaw", "head_pitch_joint", "head_pitch_extra"]


@pytest.mark.parametrize("names", [NAMES, NAMES[:7]])
def test_same_writes_as_the_old_head_block(names):
    new, old = _Term(names), _Term(names)
    apply = _method()
    for k in range(4):
        new._processed_actions.uniform_(-1, 1, generator=torch.Generator().manual_seed(k))
        old._processed_actions.copy_(new._processed_actions)
        apply(new, None)
        apply(old, None)  # finger writes identical by construction; replace its head write by the oracle's
        old._asset.calls = old._asset.calls[:-1] if "head_pitch_joint" in names else old._asset.calls
        _old_head_block(old)
    assert len(new._asset.calls) == len(old._asset.calls)
    for (ia, ta), (ib, tb) in zip(new._asset.calls, old._asset.calls, strict=True):
        assert ia == ib and ta.dtype == tb.dtype and ta.shape == tb.shape
        assert ta.numpy().tobytes() == tb.numpy().tobytes()
    assert torch.equal(new._asset.target, old._asset.target)
