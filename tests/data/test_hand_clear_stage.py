"""S-hand-clear step 2f (CPU): an excluded clip becomes one hand-scoped exclusion line with its counted reason, the line
applies to that hand only, a rerun replaces only this generator's lines of that hand, and a reader that names no hand
fails instead of ignoring the scope column."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dexengine.core.rules import ledger as L
from dexengine.data.objects import hand_clear as HC
from dexengine.placement.repair import hand_clear_stage as HS

H1, H2 = "xhand_pro", "wuji2"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "excluded_clips.txt"
    monkeypatch.setattr(L, "retarget_excluded_path", lambda ds: p)
    return p


def _rows(clip: str, **sides) -> list[dict]:
    return [{"clip": clip, "side": s, "status": st, "why": f"why {s}"} for s, st in sides.items()]


def test_the_drop_verdict_is_one_line_per_clip_with_the_reason_word(monkeypatch):
    K = HS.KEPT
    rows = _rows("a", right=K, left="CLEAN") + _rows("b", right="CLEAN") + _rows("c", right=K, left=K)
    assert HS.drop_verdicts(rows) == {}  # sides kept as recorded stay in the chain (ACTION keep)
    monkeypatch.setitem(HS.ACTION, K, "exclude")  # were they excluded: one line per clip with the hand's word
    v = HS.drop_verdicts(rows)
    assert sorted(v) == ["a", "c"] and {w for w, _ in v.values()} == {HC.KEPT_WORD}
    assert v["c"][1] == "right: why right | left: why left"


def test_a_hand_scoped_line_applies_to_its_hand_only(ledger):
    ledger.write_text("a\thuman decision\n", encoding="utf-8")
    drops = {"a": (HC.KEPT_WORD, "x"), "b": (HC.KEPT_WORD, "y")}
    L.retarget_excluded_record("taco", HS.GENERATOR, drops, ["a", "b", "c"], hand=H1)
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert lines == ["a\thuman decision", f"b\tHC:{HC.KEPT_WORD} — y\t{H1}"]  # a: already excluded for every hand
    assert L.excluded_from_file(ledger, H1) == {"a": "human decision", "b": f"HC:{HC.KEPT_WORD} — y"}
    assert L.excluded_from_file(ledger, H2) == {"a": "human decision"}


def test_a_rerun_replaces_only_its_own_hand(ledger):
    L.retarget_excluded_record("taco", HS.GENERATOR, {"b": (HC.KEPT_WORD, "y")}, ["b"], hand=H1)
    L.retarget_excluded_record("taco", HS.GENERATOR, {"b": (HC.KEPT_WORD, "z")}, ["b"], hand=H2)
    L.retarget_excluded_record("taco", HS.GENERATOR, {}, ["b"], hand=H1)  # b now clears for H1
    assert "b" not in L.excluded_from_file(ledger, H1) and "b" in L.excluded_from_file(ledger, H2)


def test_a_reader_without_a_hand_fails_on_a_scoped_line(ledger, monkeypatch):
    """Negative: the scope column is never ignored — no hand named and none active is a failure."""
    L.retarget_excluded_record("taco", HS.GENERATOR, {"b": (HC.KEPT_WORD, "y")}, ["b"], hand=H1)
    monkeypatch.delenv("DEXENGINE_HAND", raising=False)
    with pytest.raises((SystemExit, RuntimeError, KeyError, ValueError)):
        L.excluded_from_file(ledger)
    with pytest.raises(SystemExit):
        L.retarget_excluded_record("taco", HS.GENERATOR, {}, ["b"], hand="no_such_hand")


def test_the_rule_reads_nothing_for_a_dataset_that_opts_out(monkeypatch):
    """Negative: with S-hand-clear off for the dataset no still row is measured (no geometry is opened)."""
    from dexengine.data.objects import extra_slots as XS
    from dexengine.data.objects import scene_rules as SC

    rows = [{"name": "plate", "cls": XS.STATIC_SPAWN, "kind": XS.KIND_TOUCHED}]
    monkeypatch.setattr(SC, "on", lambda rid, ds: rid != HC.RULE_ID)
    assert HC.clip_verdicts("/nonexistent", {"dataset_name": "taco"}, None, rows) == {}


def test_the_targets_are_the_still_kinds_the_robot_collides_with(monkeypatch, run_dataset):
    """Targets come from collision_presets.extra_collide (robot channel ON in training or the delivered scene): every
    still kind, envelope included; a kind whose robot channel is OFF in both is not judged (no geometry opened)."""
    from dexengine.data.objects import extra_slots as XS
    from dexengine.rl.tasks.scene import collision_presets as CP

    still = (XS.KIND_TOUCHED, XS.KIND_RECEIVER, XS.KIND_TOUCHED_SOLID, XS.KIND_ENVELOPE)
    assert HC.target_kinds(run_dataset()) == still
    assert HC.target_kinds(run_dataset(robot_static=False)) == still  # the delivered scene collides with all of them
    real = CP.extra_collide
    monkeypatch.setattr(
        CP,
        "extra_collide",
        lambda k, full=None, ds=None: dict(real(k, full, ds), hand="OFF") if k == "envelope" else real(k, full, ds),
    )
    assert XS.KIND_ENVELOPE not in HC.target_kinds("taco")
    env = [{"name": "plate", "cls": XS.STATIC_SPAWN, "kind": XS.KIND_ENVELOPE}]
    assert HC.clip_verdicts("/nonexistent", {"dataset_name": "taco"}, None, env) == {}
    assert "targets=touched+receiver+touched_solid" in HS.rule_version("taco")


def test_a_pool_task_drops_the_shapes_its_tree_built(monkeypatch):
    """2f maps tree_job: the object shapes one tree built do not stay in the worker for its next tree (a worker grew
    with every object it had seen)."""
    import inspect

    from dexengine.data.objects import collider_shape as CS

    def tree_rows(tree):
        CS._CACHE[("hot3d", tree)] = object()
        return tree, [], ""

    monkeypatch.setattr(HS, "tree_rows", tree_rows)
    monkeypatch.setattr(HS.capacity.ctypes, "CDLL", lambda _: SimpleNamespace(malloc_trim=lambda n: None))
    assert HS.tree_job("t1") == ("t1", [], "") and not CS._CACHE
    assert "ex.map(tree_job," in inspect.getsource(HS.run)
