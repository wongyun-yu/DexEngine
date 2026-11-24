"""Campaign manifest rules and the release builder's refusals (dexengine.rl.manifest, manifest_rules) — no data."""

import json

import pytest

from dexengine.rl import cell_files as CF
from dexengine.rl import manifest as M
from dexengine.rl import manifest_axes as MM
from dexengine.rl import manifest_rules as BR


def _row(bid, clips, dups=None, rk="rk", n=None):
    keys = dict.fromkeys(clips, "k1")
    return {
        "id": bid,
        "clips": clips,
        "dups": dups or {},
        "run_key": rk,
        "n": n or len(clips),
        "keys": keys,
        "families": {"k1": "f"},
        "dataset": "taco",
    }


def test_r0_finds_each_violation():
    a = _row("a.01", ["c1", "c2"])
    b = _row("b.01", ["c3", "c1"], dups={"c1": "a.01"})
    assert BR.r0_problems(["c1", "c2", "c3"], [a, b], 64) == []
    assert BR.r0_problems(["c1", "c2", "c3", "c4"], [a, b], 64) == ["missing 1 ['c4']"]
    bad = _row("b.01", ["c3", "c1"], dups={"c1": "zz"})
    assert BR.r0_problems(["c1", "c2", "c3"], [a, bad], 64) == ["dup with a wrong origin 1 ['b.01:c1']"]
    assert BR.r0_problems(["c1", "c2", "c3"], [a, _row("b.01", ["c3", "c1"])], 64)[0].startswith("unrecorded dup 1")
    assert BR.r0_problems(["c1", "c2"], [a], 1) == ["over n 1 ['a.01']"]


def test_fingerprints_and_selection():
    assert BR.axis_fp([("b", "2"), ("a", "1")]) == BR.axis_fp([("a", "1"), ("b", "2")])
    full = BR.selection(BR.N_DEFAULT, BR.BLOCK_DEFAULT, set(), None, None)
    assert set(full.values()) == {""}
    srcs = [{"axis": "taco", "files": [{"sha256": "s"}]}]
    assert BR.manifest_fingerprint(srcs, full) != BR.manifest_fingerprint(srcs, BR.selection(32, 8, set(), None, None))
    with pytest.raises(SystemExit):
        BR.rule_record(BR.N_DEFAULT + 1, 8, MM.SMALL_POOL, M.CLASS_ORDER)
    assert BR.clip_table(_row("a.01", ["c1"])) == CF.TABLE_HEAD + "c1\ttaco\tk1\tf\t0\t-\n"


def test_release_rules_refuse_a_lineage_and_overwrite(tmp_path, data_root):
    with pytest.raises(SystemExit, match="site rules"):
        M.RELEASE.gate("h", {"gen": "g"}, [], set())
    with pytest.raises(SystemExit, match="no folded ledger"):  # the empty data root: no fallback
        M.build("wuji2", tmp_path, 64, 8, set())
    with pytest.raises(SystemExit, match="unknown optional"):
        M.build("wuji2", tmp_path, 64, 8, {"nope"})


def test_verify_refuses_an_old_schema(tmp_path):
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps({"schema": 2}))
    assert M.verify(f)[0].startswith("schema 2")


def test_klass_and_tag():
    g = {"artic": "rigid", "scene": "table", "shared": "0", "idle": "left"}
    assert MM.klass("bimanual", g) == "idle" and MM.gtag("bimanual", g) == "idleL"
    assert MM.klass("right", dict(g, idle="none")) == "uni" and MM.gtag("right", g) == "uniR"
