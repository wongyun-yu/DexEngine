"""Trajectory recorder × policy-free PARK probe (CPU): a probe batch stops every env at its last cut, so the episodes
still running are recorded as cut at that step (not successes, PARK validity kept); its rows are no candidates and not
judge episodes (the judge counts none in a probe batch), in the recorder's and the object trace's count equations; a
batch the probe did not mark still dies on an unfinished episode, and so does a mark that is not the recorder's last
step. The judge batch marks the recorder only when PARK runs as the probe."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dexengine.rl.eval import traj_record as TR
from dexengine.rl.judge import rollouts_batch as RB
from experiments.scoring import objtrace_recorder as OT

D, NE, T_LAST = 3, 4, 60


@pytest.fixture
def rec(monkeypatch):
    """A recorder in the state the judge loop leaves at end_batch: envs 0 · 1 ended at step 30, envs 2 · 3 running."""
    for name in ("score", "rank_key", "failure_key"):
        monkeypatch.setattr(TR.traj_select, name, lambda *a, **k: 0.0)
    r = object.__new__(TR.TrajRecorder)
    r.b, r.bs, r.n_env, r.t, r.warm, r.k_keep = 0, NE, NE, T_LAST, False, 2
    r.pin, r.dev, r.D = False, torch.device("cpu"), D
    r.buf = torch.zeros((T_LAST + 2, NE, D))
    r.length = torch.tensor([30, 30, -1, -1])
    r.early = torch.zeros(NE, dtype=torch.bool)
    r.assign = np.zeros(NE, dtype=np.int64)
    r.rows, r.kept, r.fail_best, r.probe_cut, r.probe_batches = [], {}, {}, None, set()
    monkeypatch.setattr(TR.TrajRecorder, "sides_cols", lambda self: [])
    return r


def test_a_probe_batch_records_its_cut_episodes(rec):
    rec.mark_probe_cut(T_LAST)
    rec.end_batch()
    assert [row[6] for row in rec.rows] == [30, 30, T_LAST, T_LAST]
    assert [row[5] for row in rec.rows] == [True, True, False, False]  # cut ≠ success
    assert all(row[4] for row in rec.rows) and rec.kept == {} and rec.fail_best == {}  # no candidates
    assert rec.probe_batches == {0}


def _pack(rec, monkeypatch):
    """The recorder's count equation against a probe run's judge (no episode counted)."""
    monkeypatch.setattr(TR, "_extra_slots_meta", lambda env: {})
    rec.env = SimpleNamespace(step_dt=0.02, physics_dt=0.005, cfg=SimpleNamespace(decimation=4))
    rec.tree, rec.clip_names, rec.warmup_batches, rec.sides, rec.meta_in, rec.t0 = {}, ["c0"], 0, [], {}, 0.0
    out = rec._pack_trajectories({"num_episodes": 0, "per_clip": [{"completed": 0}]})
    return out[5], out[7:11]  # meta · ok_c, ok_e, ok_f, ok_s


def test_a_probe_run_meets_the_judge_count_equation(rec, monkeypatch):
    rec.mark_probe_cut(T_LAST)
    rec.end_batch()
    meta, oks = _pack(rec, monkeypatch)
    assert all(oks) and meta["recorded"]["probe_batches"] == [0] and meta["recorded"]["real_episodes"] == 0


def test_probe_rows_counted_as_judge_episodes_break_the_equation(rec, monkeypatch):
    """Negative: the same rows without the probe batch set are 4 real episodes against the judge's 0."""
    rec.mark_probe_cut(T_LAST)
    rec.end_batch()
    rec.probe_batches = set()
    _meta, (ok_c, ok_e, ok_f, ok_s) = _pack(rec, monkeypatch)
    assert not ok_e and not ok_s


def test_the_object_trace_leaves_probe_batches_out_of_its_judge_count():
    ot = object.__new__(OT.ObjTrace)
    ot.eps = [(0, 0, e, 0, 1, 0, T_LAST) for e in range(2)] + [(0, 1, e, 0, 1, 1, 40) for e in range(2)]
    ot.obj_parts, ot.ref_parts, ot.rf_parts, ot.rule_mismatch = [], [], [], 0
    assert OT.ObjTrace.counts(ot, SimpleNamespace(probe_batches={0}))["real"] == 2
    assert OT.ObjTrace.counts(ot, SimpleNamespace(probe_batches=set()))["real"] == 4  # negative: every batch counted


def test_an_ordinary_batch_keeps_its_candidates(rec):
    """Negative of the probe rule: the same batch without the probe, every episode ended, keeps its candidates."""
    rec.length = torch.tensor([30, 30, 40, 40])
    rec.end_batch()
    assert all(row[4] for row in rec.rows) and rec.kept and rec.probe_batches == set()


def test_an_unmarked_batch_still_dies_on_an_unfinished_episode(rec):
    with pytest.raises(RuntimeError, match="unfinished episode"):
        rec.end_batch()


def test_a_mark_that_is_not_the_last_step_dies(rec):
    rec.mark_probe_cut(T_LAST - 1)
    with pytest.raises(RuntimeError, match="probe cut"):
        rec.end_batch()


class _Rec:
    def __init__(self):
        self.calls = []

    def mark_probe_cut(self, step):
        self.calls.append(("mark", step))

    def end_batch(self, park_invalid=False):
        self.calls.append(("end", park_invalid))


@pytest.mark.parametrize(("probe", "want"), [(True, [("mark", T_LAST), ("end", False)]), (False, [("end", False)])])
def test_the_judge_batch_marks_the_recorder_only_for_the_probe(monkeypatch, probe, want):
    park = SimpleNamespace(probe=probe, step=T_LAST, invalid=False, e_rows={}, excl_a={})
    monkeypatch.setattr(RB, "_patch_traj_mgr_reset", lambda *a, **k: (park, SimpleNamespace()))
    loader = SimpleNamespace(get_clip_lengths=lambda a: torch.ones_like(a))
    env = SimpleNamespace(env=SimpleNamespace(unwrapped=SimpleNamespace(device="cpu")), num_envs=NE)
    r = _Rec()
    RB._run_single_batch(
        env,
        None,
        SimpleNamespace(_loader=loader),
        torch.zeros(NE, dtype=torch.long),
        0,
        NE,
        arm_pen=None,
        traj_recorder=r,
    )
    assert r.calls == want
