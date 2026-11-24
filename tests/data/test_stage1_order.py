"""Stage-1 order: every step declares what it reads and writes, the table check refuses a step that reads what no
earlier step writes, the sidecar path owner refuses the same at run time, and the driver walks the table's own list."""

from pathlib import Path

import pytest

from dexengine.core import layout as L
from dexengine.core import paths as P
from dexengine.data.stage1 import table as T

REPO = Path(__file__).resolve().parents[2]


def _ids(stages):
    return [T.step_id(g) for g in stages]


def _moved(step: str, after: str):
    """The table with `step` moved to right after `after`."""
    rows = [g for g in T.STAGES if T.step_id(g) != step]
    k = _ids(rows).index(after)
    return (*rows[: k + 1], next(g for g in T.STAGES if T.step_id(g) == step), *rows[k + 1 :])


def test_the_table_passes_and_2f_follows_every_writer_it_reads():
    made = T.check_inputs()
    ids = _ids(T.STAGES)
    reads = next(g for g in T.STAGES if T.step_id(g) == "2f")[3].reads
    assert {"case", "roles", "contact_base", "contact_all"} <= set(reads)
    for a in reads:
        writer = next(T.step_id(g) for g in T.STAGES if a in getattr(g[3], "writes", ()))
        assert ids.index(writer) < ids.index("2f"), (a, writer)
    assert set(reads) <= made["2f"]
    assert T.sidecars_ready("2e") == ["contact_all"] and T.sidecars_ready("2f") == ["contact_all", "contact_base"]


@pytest.mark.parametrize(
    ("after", "missing"),
    [("2e", "case"), ("4", "contact_base")],  # the old place (before 3), and between roles and the main-object sidecar
)
def test_the_two_order_defects_are_refused(after, missing):
    with pytest.raises(SystemExit, match=rf"step 2f reads '{missing}', which no earlier step writes"):
        T.check_inputs(_moved("2f", after))


def test_unknown_artifacts_and_second_writers_are_refused():
    g = next(g for g in T.STAGES if T.step_id(g) == "4b")
    bad_read = tuple(x if x is not g else (*g[:3], g[3]._replace(reads=("quality",))) for x in T.STAGES)
    with pytest.raises(SystemExit, match="unknown artifact 'quality'"):
        T.check_inputs(bad_read)
    two = tuple(x if x is not g else (*g[:3], g[3]._replace(writes=("roles",))) for x in T.STAGES)
    with pytest.raises(SystemExit, match="'roles' has two writers"):
        T.check_inputs(two)


def test_the_run_guard_refuses_a_sidecar_no_earlier_step_writes(tmp_path, monkeypatch):
    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(P.STAGE1_STEP_ENV, "2e")
    monkeypatch.setenv(P.STAGE1_SIDECARS_ENV, ",".join(T.sidecars_ready("2e")))
    assert P.sidecar_file("contact_all", "grab", "c", "right", "wuji2").name.endswith(".npz")
    with pytest.raises(RuntimeError, match=r"\[STAGE1_ORDER\] stage-1 step 2e opens the contact_base sidecar"):
        P.sidecar_file("contact_base", "grab", "c", "right", "wuji2")
    monkeypatch.delenv(P.STAGE1_STEP_ENV)  # outside a step run nothing is refused
    assert P.sidecar_file("contact_base", "grab", "c", "right", "wuji2").name.endswith(".npz")


def test_every_emitted_run_carries_the_guard():
    src = (REPO / "dexengine/data/stage1/plan.py").read_text(encoding="utf-8")
    assert 'argv = ["env", *guard,' in src and "T.sidecars_ready(step)" in src


def test_the_driver_walks_the_table_list():
    assert T.driven_steps() == [T.step_id(g) for g in T.STAGES if isinstance(g[3], T.Run)]
    assert "2f" in T.driven_steps()
