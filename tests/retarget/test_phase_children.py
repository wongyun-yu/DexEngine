"""Stage-1 phase 4/5 children: they run without a GL backend, and the IK rejects an object body it has no reference
for (as the reference solver does)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dexengine.retarget.phases import robot
from dexengine.retarget.phases.common import RUN


@pytest.fixture
def child_env(monkeypatch, tmp_path):
    """Run one phase-4/5 clip with the child process faked; returns the env each child received."""
    monkeypatch.setenv("MUJOCO_GL", "egl")  # what batch.py puts into os.environ for the rendering phases
    monkeypatch.setenv("FXKEEP", "1")
    monkeypatch.setattr(RUN, "dataset_name", "fxds")
    monkeypatch.setattr(robot, "_solve_fp", lambda *a, **kw: {})
    monkeypatch.setattr(robot, "ensure_robot_assets", lambda *a, **kw: None)
    monkeypatch.setattr(robot, "xml_outputs_complete", lambda d: True)
    monkeypatch.setattr(robot, "ik_result_valid", lambda f, dt: True)
    monkeypatch.setattr(robot, "save_taco_text_annotation", lambda **kw: tmp_path / "a.json")
    monkeypatch.setattr(robot.RFP, "stamp", lambda *a, **kw: None)
    seen = []

    def fake_run(cmd, **kw):
        seen.append(kw.get("env"))
        # a real child proves its OpenBLAS kernel (blas_pin.apply); the parent forwards the line and dies without it
        proof = "[BLAS_PIN] child loaded libscipy_openblas64_=Haswell/0.3.30 (owner blas_pin)\n"
        return SimpleNamespace(returncode=0, stdout=proof, stderr="")

    monkeypatch.setattr(robot.subprocess, "run", fake_run)
    return seen


def test_generate_xml_child_has_no_gl_backend(child_env, tmp_path):
    assert robot.run_single_generate_xml(tmp_path, "t", None, "fxhand", "right", False, "decomposed")
    (env,) = child_env
    assert env is not None, "the scene child inherits the batch-wide MUJOCO_GL"
    assert env["MUJOCO_GL"] == "disable"
    assert env["FXKEEP"] == "1"


def test_ik_child_has_no_gl_backend(child_env, tmp_path):
    assert robot.run_single_ik(tmp_path, "t", None, "fxhand", "right", False, False, 0.02)
    (env,) = child_env
    assert env["MUJOCO_GL"] == "disable"
    assert env["FXKEEP"] == "1"


def test_object_body_without_reference_column_fails():
    """`supp_object_*` bodies are added by the base layer after stage 1; their poses are not in the keypoint archive,
    so the IK has no reference column for them and stops, like the reference solver (`_object_reference_column`)."""
    from dexengine.retarget.ik.inputs import object_col  # noqa: PLC0415

    assert [object_col(b, 1) for b in ("right_object", "left_object", "extra_object_0")] == [13, 14, 12]
    with pytest.raises(ValueError, match="'supp_object_0' is not an object body name"):
        object_col("supp_object_0", 0)


def test_passthrough_stops_on_supp_object_free_joint():
    mujoco = pytest.importorskip("mujoco")
    import numpy as np  # noqa: PLC0415

    from dexengine.retarget.ik import post as PP  # noqa: PLC0415
    from dexengine.retarget.ik.inputs import Keypoints  # noqa: PLC0415

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<body name="right_object"><freejoint/><geom size="0.01"/></body>'
        '<body name="supp_object_0"><freejoint/><geom size="0.01"/></body>'
        "</worldbody></mujoco>"
    )
    T = 5
    ref = np.zeros((T, 14, 7))
    ref[..., 3] = 1.0
    kp = Keypoints(T=T, ref=ref, n_extra=0, extra_rows=[], arti={}, contact_raw={}, contact_ref=np.zeros((T, 0)))
    sm = PP.Smoothed(qpos=np.zeros((T - 3, model.nq)), qvel=np.zeros((T - 3, model.nv)))
    with pytest.raises(ValueError, match="supp_object_0"):
        PP.object_passthrough(model, sm, kp, 0.02)
