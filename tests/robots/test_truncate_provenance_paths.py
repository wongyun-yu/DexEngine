"""Truncated-hand provenance (dexengine/robots/urdf/truncate.py) records no absolute path.

The provenance file is committed next to the truncated URDFs. A generator run from a scratch worktree once wrote that
worktree's absolute path into it (the tree, the rule file), which a release must not carry and which changes with
every checkout. The rule: every recorded path is relative to the code tree or the data root, or `<outside>/<name>`.
"""

import json
import pathlib

from dexengine.core.repo import FULL_ROBOT_URDF_DIR
from dexengine.robots.urdf import truncate as T


def absolute_values(node, where="") -> list[str]:
    """Every string value (at any depth) that is an absolute path, as `<key path>=<value>`."""
    if isinstance(node, dict):
        return [hit for k, v in node.items() for hit in absolute_values(v, f"{where}/{k}")]
    if isinstance(node, list):
        return [hit for i, v in enumerate(node) for hit in absolute_values(v, f"{where}[{i}]")]
    if isinstance(node, str) and pathlib.PurePosixPath(node).is_absolute():
        return [f"{where}={node}"]
    return []


def test_committed_provenance_has_no_absolute_path():
    files = sorted(pathlib.Path(FULL_ROBOT_URDF_DIR).glob(f"*{T.SUFFIX}/{T.PROV_NAME}"))
    assert files, f"no {T.PROV_NAME} under {FULL_ROBOT_URDF_DIR}"
    for f in files:
        assert absolute_values(json.loads(f.read_text())) == [], f


def test_scanner_catches_a_planted_absolute_path():
    planted = {"rule": {"rule_file": "/tmp/scratch/tree/dexengine/retarget/ik/prior.py"}, "hands": [{"path": "/x"}]}
    assert absolute_values(planted) == [
        "/rule/rule_file=/tmp/scratch/tree/dexengine/retarget/ik/prior.py",
        "/hands[0]/path=/x",
    ]
    assert absolute_values({"rule": {"rule_file": "dexengine/retarget/ik/prior.py"}}) == []


def test_record_paths_are_relative(tmp_path):
    assert T._rel(T.REPO / "dexengine" / "robots") == "dexengine/robots"
    outside = tmp_path / "elsewhere" / "prior.py"
    assert T._rel(outside) == "<outside>/prior.py"
    assert "tree" not in T._git_state()
