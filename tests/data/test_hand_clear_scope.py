"""Stage 2f over a piece list (`hand_clear_stage --clips-file`, what stage1.split_apply passes): only the listed trees
are measured, the other trees keep their ledger lines byte for byte, and an empty list runs nothing — the split skips a
dataset without changed pieces, and the step refuses an empty list instead of reading it as 'measure nothing'."""

import concurrent.futures
from pathlib import Path

import pytest

from dexengine.data import rules as RR
from dexengine.data.stage1 import split_apply as SA
from dexengine.data.stage1 import table as T
from dexengine.placement.repair import hand_clear_stage as HCS

NAMES = ("s10__duck_pass_1_t0", "s10__duck_pass_1_t1", "s10__hammer_pass_1_t0")


class _Pool:
    """ProcessPoolExecutor stand-in: the same map, in this process."""

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def map(self, fn, it, chunksize=1):
        return map(fn, it)


def _row(name: str, tag: str) -> dict:
    return {
        "clip": name,
        "side": "right",
        "status": HCS.CLEAN,
        "rows": f"max_depth_m:{tag}",
        "qpos_sha": tag,
        "inputs": tag,
        "version": "v",
        "why": "",
        "depth_m": 0.0,
    }


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Three base trees, a ledger written by an earlier whole run, stubbed measuring (records what it measured) and
    exclusion writing (records what it scanned)."""
    trees = [tmp_path / "base" / "bimanual" / n for n in NAMES]
    led = tmp_path / "ledger_grab.tsv"
    old = ["\t".join(HCS.LEDGER_COLS)] + ["\t".join(str(_row(n, "old")[k]) for k in HCS.LEDGER_COLS) for n in NAMES]
    led.write_text("\n".join(old) + "\n", encoding="utf-8")
    measured, scanned = [], []

    def tree_rows(tree):
        measured.append(Path(tree).name)
        return Path(tree).name, [_row(Path(tree).name, "new")], ""

    monkeypatch.setattr(HCS, "stage_trees", lambda ds, spec: trees)
    monkeypatch.setattr(HCS, "tree_rows", tree_rows)
    monkeypatch.setattr(HCS, "ledger_path", lambda ds, hand, write: led)
    monkeypatch.setattr(HCS, "rule_version", lambda ds: "v")
    sha = dict.fromkeys(NAMES, "old")  # the input sha each tree reads now (= its ledger line's: every line current)
    monkeypatch.setattr(HCS, "qpos_sha_file", lambda p: sha[Path(p).parent.parent.name])
    monkeypatch.setattr(HCS, "input_key", lambda t: sha[Path(t).name])
    monkeypatch.setattr(HCS, "holds_repair", lambda t: False)
    monkeypatch.setattr(RR, "retarget_excluded_record", lambda ds, gen, drops, sc, hand=None: scanned.extend(sc))
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _Pool)
    return {"led": led, "old": old, "measured": measured, "scanned": scanned, "dir": tmp_path, "sha": sha}


def _clips(stage, names) -> str:
    f = stage["dir"] / "clips.txt"
    f.write_text("".join(n + "\n" for n in names))
    return str(f)


def test_unlisted_trees_keep_their_ledger_lines(stage):
    assert HCS.run("grab", "wuji2", True, jobs=1, clips=_clips(stage, [NAMES[1]])) == 0
    got = stage["led"].read_text(encoding="utf-8").splitlines()
    assert got[0] == stage["old"][0] and len(got) == len(stage["old"])
    assert got[1] == stage["old"][1] and got[3] == stage["old"][3]  # the unlisted trees: the earlier lines, verbatim
    assert got[2] != stage["old"][2] and got[2].split("\t")[:1] == [NAMES[1]] and "new" in got[2]


def test_only_listed_trees_are_measured_and_scanned(stage):
    assert HCS.run("grab", "wuji2", True, jobs=1, clips=_clips(stage, [NAMES[2], NAMES[0]])) == 0
    assert stage["measured"] == [NAMES[0], NAMES[2]]  # base order, the unlisted tree is not measured
    assert stage["scanned"] == [NAMES[0], NAMES[2]]  # exclusion lines of the unlisted clip stay (not scanned)
    whole = HCS.run("grab", "wuji2", True, jobs=1)  # no list: every tree, the ledger written whole
    assert whole == 0 and stage["measured"][2:] == list(NAMES)


@pytest.mark.parametrize("names", [[], ["s10__no_such_clip_t0"]])
def test_an_empty_or_unknown_list_fails_without_measuring(stage, names):
    assert HCS.run("grab", "wuji2", True, jobs=1, clips=_clips(stage, names)) == 2
    assert stage["measured"] == [] and stage["led"].read_text(encoding="utf-8").splitlines() == stage["old"]


def test_the_split_runs_2f_only_where_pieces_changed(tmp_path, monkeypatch):
    """split_apply's 2f run is per touched dataset with that dataset's pieces; no changed piece = no 2f run."""
    monkeypatch.setenv("DEXENGINE_HAND", "wuji2")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    monkeypatch.setattr(T, "py", lambda what: "py")
    a = object.__new__(SA.Apply)
    a.h, a.s, a.dss, a.pys, a.env = "wuji2", tmp_path, ["grab", "hot3d"], "py", {}
    runs = []
    monkeypatch.setattr(a, "run", lambda argv, log, what, err_pat="": runs.append([str(x) for x in argv]))
    step = next(s for s in T.split_steps()["run"] if s == "2f")
    a._hand_clear(step, [], {})
    assert runs == []  # nothing changed: 2f does not run
    a._hand_clear(step, [["grab", "right", "g_t1"]], {"grab": ["g_t1"]})
    assert len(runs) == 1 and "--clips-file" in runs[0] and runs[0][runs[0].index("--dataset") + 1] == "grab"
    listed = Path(runs[0][runs[0].index("--clips-file") + 1]).read_text().split()
    assert listed == ["g_t1"]


def test_a_short_list_fails_when_an_unlisted_line_is_stale(stage):
    """A changed tree left off the list would keep a stale line under a scoped run: the run fails before measuring."""
    stage["sha"][NAMES[2]] = "changed"  # the hammer tree changed, the list names only the first duck piece
    assert HCS.run("grab", "wuji2", True, jobs=1, clips=_clips(stage, [NAMES[0]])) == 2
    assert stage["measured"] == [] and stage["led"].read_text(encoding="utf-8").splitlines() == stage["old"]
    assert HCS.run("grab", "wuji2", True, jobs=1, clips=_clips(stage, [NAMES[0], NAMES[2]])) == 0  # listed: measured
    assert stage["measured"] == [NAMES[0], NAMES[2]]
