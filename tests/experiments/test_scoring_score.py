"""The shared Tab. 2 scorer's completion rule on a synthetic clip (the scorer's former built-in self-test)."""

import json

import numpy as np

from experiments.scoring import score as S


def _clip(tmp_path, frames=199):
    """A bimanual SPIDER-form clip: 70-dof qpos, both objects moving 30 cm along x, identity orientation."""
    q = np.zeros((frames, 70))
    x = np.linspace(0.0, 0.3, frames)
    for lo in (-14, -7):
        q[:, lo] = x
        q[:, lo + 3] = 1.0  # quaternion w (wxyz)
    (tmp_path / "0").mkdir()
    np.savez(tmp_path / "0" / "trajectory_kinematic.npz", qpos=q)
    (tmp_path / "task_info.json").write_text(json.dumps({"ref_dt": 0.02, "embodiment_type": "bimanual"}))
    return q


def test_completion_rule_positive_and_negative(tmp_path):
    q = _clip(tmp_path)
    ref = S.load_reference(str(tmp_path))
    dt = ref["dt"]

    def run(qpos):
        return S.score_rollouts(S.from_qpos(qpos, dt, True), ref)

    assert run(q)["success"]  # the reference itself completes
    pushed = q.copy()
    pushed[S.GRACE + 10 :, -14:-12] += 0.06 / np.sqrt(2)  # right object 6 cm off after the grace
    assert not run(pushed)["success"]
    lagged = np.concatenate([np.repeat(q[:1], 10, 0), q[:-10]], 0)  # 10 frames late: inside the lag window
    assert run(lagged)["success"]
    assert not run(q[: q.shape[0] // 2])["success"]  # stops half way = not completed
    policy = run(np.stack([pushed] * 19 + [q]))  # 1 of 20 rollouts completes = PASS_FRAC
    assert policy["n_complete"] == 1 and policy["success"]


def test_rule_constants():
    assert (S.POS_M, S.ROT_DEG, S.PASS_FRAC) == (0.05, 90.0, 0.05)
    assert (S.LAG_BACK, S.LAG_AHEAD, S.GRACE) == (25, 5, 60)
