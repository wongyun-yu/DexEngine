"""Long-clip split (stage1.split_apply): the steps it reruns on new pieces come from the stage table
(``table.split_steps``), every driven step after the base layer is either rerun, carried by the cut or left to f0,
and split_apply keeps no step list of its own. Before the fix its list was the literal "2d,3,4,4b,5,5b,6": 2f (hand
clear) never ran on new pieces."""

import ast
from pathlib import Path

import pytest

from dexengine.data.segment import pieces as PC
from dexengine.data.stage1 import split_apply as SA
from dexengine.data.stage1 import table as T

REPO = Path(__file__).resolve().parents[2]


def _with(step: str, cell: str | None = None, **fields):
    """The table with one row's cell or Run fields replaced."""
    out = []
    for g in T.STAGES:
        if T.step_id(g) == step:
            g = (g[0], g[1] if cell is None else cell, g[2], g[3]._replace(**fields))
        out.append(g)
    return tuple(out)


def _step_of(argv: list[str]) -> str:
    """The table step a recorded argv runs: same script, the row whose own flags match best (present +1, absent -1)."""
    script = next(Path(a).name for a in argv if a.endswith(".py"))
    rows = [g for g in T.STAGES if isinstance(g[3], T.Run) and Path(g[3].argv[0]).name == script]
    assert rows, f"no table row runs {script}"
    return T.step_id(max(rows, key=lambda g: sum(1 if a in argv else -1 for a in g[3].argv if a.startswith("--"))))


def _apply(tmp_path, monkeypatch, dss=("hot3d",), argvs=None):
    """An Apply without its constructor (no data root): stubbed runs that record what they run (and the argv of the
    scoped runs in `argvs`); the interpreter lookup is stubbed too."""
    monkeypatch.setenv("DEXENGINE_HAND", "wuji2")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    monkeypatch.setattr(T, "py", lambda what: "py")
    a = object.__new__(SA.Apply)
    a.h, a.s, a.dss, a.pys, a.env = "wuji2", tmp_path, list(dss), "py", {}
    a.steps = T.split_steps()["run"]
    ran: list[str] = []

    def run(argv, log, what, err_pat=""):
        argv = [str(x) for x in argv]
        ran.append(_step_of(argv) if any(x.endswith(".py") for x in argv) else what)
        if argvs is not None:
            argvs.append(argv)
        Path(log).write_text("")

    def stage(step, ds, prefix):
        ran.append(step)
        return {"runs": []}

    monkeypatch.setattr(a, "run", run)
    monkeypatch.setattr(a, "stage", stage)
    return a, ran


def test_every_driven_step_after_the_base_layer_has_one_place():
    part = T.split_steps()
    after = T.driven_steps()[1:]
    placed = [s for grp in part.values() for s in grp]
    assert sorted(placed, key=after.index) == after and len(placed) == len(set(placed))
    for grp in part.values():
        assert grp == sorted(grp, key=after.index), "a group is out of row order"
    assert "2f" in part["run"] and "2e" in part["carried"] and "7" in part["f0"]


@pytest.mark.parametrize(
    ("step", "kw"),
    [
        ("2e", {"cut_copies": ""}),  # 2e no longer carried: its cell cannot judge a piece (no tree_fresh)
        ("2f", {"cut_copies": "NOT_COPIED"}),  # 2f claimed as carried by a key the cut does not copy
        ("2f", {"cell": T._KEYED + '"x"'}),  # 2f's cell no longer judges a tree
    ],
)
def test_a_step_the_split_cannot_place_dies(step, kw):
    with pytest.raises(SystemExit, match=rf"step {step}\b"):
        T.split_steps(_with(step, **kw))


def test_the_cut_dropping_a_carried_key_dies(monkeypatch):
    monkeypatch.setattr(PC, "PIECE_RULES_COPY", tuple(k for k in PC.PIECE_RULES_COPY if k != "WRIST_SLACK"))
    with pytest.raises(SystemExit, match=r"step 2e\b"):
        T.split_steps()


def test_downstream_runs_the_table_piece_steps_in_row_order(tmp_path, monkeypatch):
    a, ran = _apply(tmp_path, monkeypatch)
    a.downstream([["hot3d", "bimanual", "clip_t1"]])
    seq = [s for i, s in enumerate(ran) if i == 0 or ran[i - 1] != s]
    assert seq == T.split_steps()["run"], seq


def test_scoped_runs_are_the_table_argv_behind_the_run_guard(tmp_path, monkeypatch):
    """The piece-scoped runs (sidecars, cases, roles) are the table row's argv (plan.step_argv) with the scope appended,
    behind DEXENGINE_STAGE1_STEP; roles run once over the touched datasets."""
    argvs: list[list[str]] = []
    a, _ran = _apply(tmp_path, monkeypatch, dss=("grab", "hot3d"), argvs=argvs)
    a.downstream([["grab", "right", "g_t1"], ["hot3d", "bimanual", "h_t1"]])
    assert argvs
    for argv in argvs:
        step = _step_of(argv)
        assert argv[:2] == ["env", f"DEXENGINE_STAGE1_STEP={step}"], argv
        body = argv[argv.index("py") + 1 :]
        row = [x.replace("{hand}", "wuji2") for x in next(g[3].argv for g in T.STAGES if T.step_id(g) == step)]
        assert len(body) > len(row) and all(
            x == y or y == "{ds}" for x, y in zip(body[: len(row)], row, strict=True)
        ), (body, row)
    roles = [x for x in argvs if _step_of(x) == "4"]
    assert len(roles) == 1 and roles[0][roles[0].index("--datasets") + 1] == "grab,hot3d"


def test_todo_judges_the_table_piece_steps(tmp_path, monkeypatch):
    a, _ran = _apply(tmp_path, monkeypatch)
    seen = {}

    def run(argv, log, what, err_pat=""):
        argv = [str(x) for x in argv]
        seen["stages"] = argv[argv.index("--stages") + 1]
        Path(log).write_text("[PIECE_TODO] hand=wuji2 total pieces 1 to-do 1\n")
        Path(argv[argv.index("--out") + 1]).write_text("hot3d\tbimanual\tclip_t1\t2f\n")

    monkeypatch.setattr(a, "run", run)
    a.todo(["hot3d\tbimanual\tclip_t1"], ["hot3d\tbimanual\tclip_t1"])
    assert seen["stages"].split(",") == T.split_steps()["run"]


def test_split_apply_keeps_no_step_list():
    ids = set(T.driven_steps())
    src = (REPO / "dexengine/data/stage1/split_apply.py").read_text(encoding="utf-8")
    lits = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    bad = [s for s in lits if s.replace(",", " ").split() and set(s.replace(",", " ").split()) <= ids]
    assert not bad, f"split_apply.py names steps itself: {bad}"
    paths = [s for s in lits if s.startswith("dexengine/")]
    assert not paths, f"split_apply.py writes step argv itself (the table owns it): {paths}"


def test_split_apply_makes_the_campaign_folder(tmp_path, monkeypatch):
    """The documented call names a campaign folder that does not exist yet (CAMPAIGN_DIR=logs/split): split_apply makes
    it (with parents) before its first write, the units snapshot. Before the fix that write raised FileNotFoundError."""

    class Stop(Exception):
        pass

    def stop(self):
        raise Stop

    camp = tmp_path / "logs" / "split"
    monkeypatch.setenv("ARM", "rby1")
    monkeypatch.setenv("CAMPAIGN_DIR", str(camp))
    monkeypatch.setattr(SA.Apply, "split_steps", stop)
    with pytest.raises(Stop):
        SA.main(["xhand_pro", "--datasets", "hot3d"])
    assert (camp / "units_before_xhand_pro.tsv").is_file()
