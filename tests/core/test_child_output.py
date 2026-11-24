"""The tail of a child's output is quoted through one reader (dexengine/core/child_output.py).

Rule (structure): a slice `X[-N:]` whose X is a child's output text — `.stdout` / `.stderr`, their sum, an
`or` of them, or those under `.strip()` / `.rstrip()` / `.decode()` — is a raw tail. A raw tail is allowed only
where tests/core/child_tails.tsv states it (column child = what was read; marker n = it prints no ✗; rows of the
files a deployment layer keeps are in that layer's own table, read only where it is present); a marker child
goes through `child_failure`, which keeps every ✗ line whole. A new raw tail, a stale row, an unparsable file, or a
row that says the child prints a marker all fail. Not caught (stated limit): a tail of a name bound earlier
(`out = r.stdout; out[-200:]`) and line-list tails (`.splitlines()[-k:]`).

Expiry: child_tails.tsv states the raw tails allowed today. Once every raw tail of a child's output goes through
`child_failure`, delete child_tails.tsv and make the statement test require zero raw tails."""

import ast
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from dexengine.core.child_output import child_failure

REPO = Path(__file__).resolve().parents[2]
TABLE = Path(__file__).with_name("child_tails.tsv")
#: The same statement for the files of a deployment layer (read only where that layer is present).
SITE_TABLE = REPO / "tests" / "site" / "child_tails.tsv"
STREAMS = {"stdout", "stderr"}
PEEL = {"strip", "rstrip", "decode"}


def _child_text(n: ast.AST) -> bool:
    if isinstance(n, ast.Attribute):
        return n.attr in STREAMS
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        return _child_text(n.left) and _child_text(n.right)
    if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or):
        return any(_child_text(v) for v in n.values) and all(
            _child_text(v) or (isinstance(v, ast.Constant) and isinstance(v.value, str)) for v in n.values
        )
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in PEEL:
        return _child_text(n.func.value)
    return False


def raw_tails(src: str) -> list[tuple[int, int, str]]:
    """(line, N, expression) of every raw tail `X[-N:]` of a child's output text in `src` (SyntaxError propagates)."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)):
            continue
        s = n.slice
        lo = s.lower
        if (
            s.upper is not None
            or s.step is not None
            or not isinstance(lo, ast.UnaryOp)
            or not isinstance(lo.op, ast.USub)
        ):
            continue
        if not (isinstance(lo.operand, ast.Constant) and isinstance(lo.operand.value, int)):
            continue
        if _child_text(n.value):
            out.append((n.lineno, lo.operand.value, " ".join(ast.get_source_segment(src, n.value).split())))
    return out


def scan(root: Path, names: list[str]) -> Counter:
    got = Counter()
    for f in names:
        p = root / f
        if not p.is_file():
            continue
        for _ln, num, expr in raw_tails(p.read_text(encoding="utf-8")):  # an unparsable file fails here
            got[(f, num, expr)] += 1
    return got


def measured() -> Counter:
    files = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "*.py"], capture_output=True, text=True, check=True
    )
    names = [f for f in files.stdout.split("\0") if f]
    assert names, "git ls-files found no python file — the scan would pass on nothing"
    return scan(REPO, names)


def stated() -> tuple[Counter, list[list[str]]]:
    rows = []
    for t in (TABLE, SITE_TABLE) if SITE_TABLE.is_file() else (TABLE,):
        rows += [ln.split("\t") for ln in t.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    assert all(len(r) == 5 for r in rows), "child_tails.tsv rows are file · N · expression · marker · child"
    return Counter((r[0], int(r[1]), r[2]) for r in rows), rows


def test_raw_tails_match_the_statement():
    got, (want, _rows) = measured(), stated()
    new, stale = got - want, want - got
    assert not new, f"raw tail of a child's output not stated — quote it through child_failure: {sorted(new)[:10]}"
    assert not stale, f"child_tails.tsv rows no longer in the code — remove them: {sorted(stale)[:10]}"


def test_no_stated_raw_tail_is_a_marker_child():
    _want, rows = stated()
    assert {r[3] for r in rows} <= {"n", "undecided"}, "a child that prints ✗ must go through child_failure"


def test_rule_catches_a_planted_raw_tail():
    src = (
        "import subprocess\nr = subprocess.run(['x'], capture_output=True, text=True)\nprint(r.stderr.strip()[-200:])\n"
    )
    assert raw_tails(src) == [(3, 200, "r.stderr.strip()")]
    assert raw_tails(src.replace("r.stderr.strip()[-200:]", "(r.stdout + r.stderr)[-90:]")) == [
        (3, 90, "r.stdout + r.stderr")
    ]
    assert raw_tails(src.replace("r.stderr.strip()[-200:]", "(r.stderr or '').strip()[-60:]"))[0][1] == 60
    assert raw_tails(src.replace("r.stderr.strip()[-200:]", "child_failure(r.stderr, 200)")) == []
    assert raw_tails(src.replace("r.stderr.strip()[-200:]", "r.stderr.splitlines()[-3:]")) == []


def test_planted_file_fails_the_scan(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "probe.py").write_text("def f(r):\n    return r.stdout[-80:]\n", encoding="utf-8")
    (tmp_path / "pkg" / "ok.py").write_text("def f(r):\n    return child_failure(r.stdout, 80)\n", encoding="utf-8")
    want, _rows = stated()
    assert scan(tmp_path, ["pkg/probe.py", "pkg/ok.py"]) - want == Counter({("pkg/probe.py", 80, "r.stdout"): 1})
    (tmp_path / "pkg" / "bad.py").write_text("def f(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        scan(tmp_path, ["pkg/bad.py"])


def test_unparsable_source_is_not_skipped():
    with pytest.raises(SyntaxError):
        raw_tails("def f(:\n")


def test_marker_line_survives_a_long_tail():
    long = "x" * 400
    text = f"[COLLECT_RESTORE] ✗ meta/info.json missing — /{long}/a\n[COLLECT_RESTORE] ✗ second /{long}/b\nend\n"
    out = child_failure(text, 200)
    assert out.count("[COLLECT_RESTORE] ✗ ") == 2 and out.endswith("/b") and long in out


def test_nested_marker_counts_once():
    line = "[COLLECT_SEED_USE] ✗ revive h c: [COLLECT_RESTORE] ✗ meta/info.json missing"
    assert child_failure(line + "\n", 30).count("[COLLECT_RESTORE] ✗") == 1


def test_no_marker_gives_the_tail():
    assert child_failure("  a\nb" + "z" * 300 + "\n", 200) == "z" * 200
    assert child_failure(None, 200) == "" and child_failure("", 5) == ""
