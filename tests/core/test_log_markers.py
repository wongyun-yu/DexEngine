"""dexengine/core/log_markers.py — the markers that loops outside the package read (six markers no reader checked
before + the capability list that replaces source greps).

  M1 each formatter's line is found and parsed by its own reader pattern (Python); near-miss lines (another stage
     word, a log path without its token, a probe line without job=) are not
  M2 the shell form: `--sh` assignments sourced in bash, `grep -E "$RX_…"` on the same lines (round trip)
  M3 the module loads by path with the system python in isolated mode (no package on the path) and imports only the
     standard library
  M4 structure: every emitter calls its formatter (LM.<fn>) and no file of the package keeps the marker text as an
     emit literal (f-string part or call argument); planted: a literal emitter is flagged, a missing call is flagged
  M5 INFRA: the retry kinds' death texts and the boot / runtime words match, the clip-fault kinds do not; the launcher
     and the attempt rule read the module's tables
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core import log_markers as LM
from tests.core.references_paths import SITE

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "dexengine/core/log_markers.py"

LINES = {  # marker → (a line its formatter makes, the fields the reader must get back)
    "EXT_PREPARE_DONE": (LM.ext_prepare_done(12, 2, "/r/ext/plan.json"), {}),
    "JUDGE_FP": (LM.judge_fp(3, "ne=1024 scene=s → /r/jfp.tsv"), {"schema": "3"}),
    "JUDGE_LOG": (
        f"[EXT] ✗ judge all rc 1 · DONE [] · {LM.judge_log('/r/ext/c1/judge.log')} · []",
        {"log": "/r/ext/c1/judge.log"},
    ),
    "JOBS_OK": (LM.jobs_ok("[RUN]", 6, "· hand wuji2 · manifest bundles 3"), {"n": "6"}),
    "PROBE": (
        LM.probe(
            "[COLLECT_POD]",
            "FAIL",
            "c__b__orig__p1of2",
            {"job": "j7", "alive": 0, "pod.rc": "-", "tar": "0"},
            "fail=-",
        ),
        {"verdict": "FAIL", "spec": "c__b__orig__p1of2", "job": "j7", "alive": "0"},
    ),
    "PROBE_RC": (LM.probe_rc("[RUN]", 3, {"DONE": 2, "WAIT": 1}), {"rc": "3", "counts": "{'DONE': 2, 'WAIT': 1}"}),
    "COLLECT_PLACE_OK": (
        LM.collect_place_ok("/r/collect/recv/c1/pieces/c__b__orig__p1of2.tar", "0123456789abcdef", "moved · root /r"),
        {"piece": "c__b__orig__p1of2", "sha": "0123456789ab"},
    ),
}
NEAR_MISS = {  # lines a reader must not take (another stage word, the path without its token, a missing field)
    "EXT_PREPARE_DONE": "[EXT_STAGE] prepare started · clips 12 · condition groups 2",
    "JUDGE_LOG": "[EXT] ✗ judge all rc 1 · DONE [] · end markers 0/3: /r/ext/c1/judge.log · []",
    "JOBS_OK": "[RUN] jobs OK · hand wuji2",
    "PROBE": "[RUN] probe FAIL c__b__orig__p1of2 alive=0",
    "COLLECT_PLACE_OK": "[COLLECT_PLACE] OK /r/collect/recv/c1/c__b__orig__p1of2.tar sha=0123456789ab",
}
EMITTERS = {  # formatter → the files of this tree that emit its marker
    "judge_fp": ["dexengine/placement/judge/boot.py"],
    "judge_log": ["dexengine/record/augment/cells_steps.py"],
}
LITERALS = {  # emit literals that would be a second owner of a marker's text
    "ext_prepare_done": r"prepare done",
    "judge_fp": r"\[JUDGE_FP\] schema=",
    "jobs_ok": r"jobs OK",
    "probe": r"^ ?probe $",
    "probe_rc": r"PROBE rc=",
    "collect_place_ok": r"\[COLLECT_PLACE\] OK",
}


def test_m1_round_trip_and_old_forms():
    by = {m.name: m for m in LM.MARKERS}
    for name, (line, want) in LINES.items():
        got = by[name].parse(line)
        assert got is not None, (name, line)
        assert {k: got[k] for k in want} == want, (name, got)
    assert LM.PROBE.parse(LM.probe("[COLLECT_CLUSTER]", "DONE", "s1", {"job": "9", "slurm": "R"}))["alive"] is None
    for name, line in NEAR_MISS.items():
        assert not by[name].found(line), f"{name}: a near-miss line matches"


def test_m2_shell_round_trip(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    for name, (line, _w) in LINES.items():
        script = f'eval "$({sys.executable} {MOD} --sh)"; printf "%s\\n" "$L" | grep -aqE "$RX_{name}"'
        r = subprocess.run(
            [bash, "-c", script], env={"L": line, "PATH": "/usr/bin:/bin"}, capture_output=True, text=True
        )
        assert r.returncode == 0, (name, line, r.stderr[-300:])
    for name, line in NEAR_MISS.items():
        script = f'eval "$({sys.executable} {MOD} --sh)"; printf "%s\\n" "$L" | grep -aqE "$RX_{name}"'
        r = subprocess.run(
            [bash, "-c", script], env={"L": line, "PATH": "/usr/bin:/bin"}, capture_output=True, text=True
        )
        assert r.returncode == 1, (name, "a near-miss line must not match in grep -E either")
    heads = subprocess.run([sys.executable, str(MOD), "--sh"], capture_output=True, text=True).stdout
    assert "MARK_COLLECT_PLACE_OK='[COLLECT_PLACE] OK'" in heads


def test_m3_standalone_stdlib_only():
    py = "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else sys.executable
    code = (
        "import importlib.util, sys; s = importlib.util.spec_from_file_location('lm', sys.argv[1]); "
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.JOBS_OK.head)"
    )
    r = subprocess.run([py, "-I", "-c", code, str(MOD)], capture_output=True, text=True, cwd="/")
    assert r.returncode == 0 and r.stdout.strip() == "jobs OK", r.stderr[-300:]
    tree = ast.parse(MOD.read_text())
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def _calls(tree, fn: str) -> int:
    return sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == fn
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "LM"
    )


def _emit_literals(tree) -> list[str]:
    """String parts of f-strings and string arguments of calls (docstrings and comments are not emits)."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.JoinedStr):
            out += [v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        elif isinstance(n, ast.Call):
            out += [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    return out


def problems(files: dict[str, str]) -> list[str]:
    """{path: source} → structure problems: an emitter without its formatter call, an emit literal of a marker text
    outside log_markers.py."""
    bad = []
    trees = {p: ast.parse(s) for p, s in files.items()}
    for fn, emitters in EMITTERS.items():
        bad += [f"{p}: no LM.{fn}( call" for p in emitters if p in trees and not _calls(trees[p], fn)]
    for p, t in trees.items():
        if p.endswith("dexengine/core/log_markers.py"):
            continue
        for fn, rx in LITERALS.items():
            bad += [f"{p}: emit literal {s!r} ({fn})" for s in _emit_literals(t) if re.search(rx, s)]
    return bad


def _package_sources() -> dict[str, str]:
    out = {}
    for top in ("dexengine", SITE, "experiments"):
        for f in sorted((REPO / top).rglob("*.py")):
            out[str(f.relative_to(REPO))] = f.read_text(encoding="utf-8")
    return out


def test_m4_structure_and_planted():
    src = _package_sources()
    assert problems(src) == []
    planted = dict(src)
    planted["<planted emitter>"] = 'TAG = "[X]"\nprint(f"{TAG} jobs OK {3} · hand h")\n'
    assert any("<planted emitter>: emit literal" in b for b in problems(planted))
    p = "dexengine/placement/judge/boot.py"
    planted = dict(src)
    planted[p] = src[p].replace("LM.judge_fp(", "_fp_line(")
    assert any(f"{p}: no LM.judge_fp(" in b for b in problems(planted))


def test_m5_infra_words():
    for k in LM.RETRY_KINDS:
        assert LM.INFRA.found(f"[COLLECT] {LM.KIND_TEXT[k]} · again — /w/session_1.log"), k
    for k in set(LM.KIND_TEXT) - set(LM.RETRY_KINDS):
        assert not LM.INFRA.found(f"[COLLECT] {LM.KIND_TEXT[k]} — /w/session_1.log"), f"{k} is a clip fault"
    for w in (*LM.RUNTIME_INFRA, *(lab for _t, lab in LM.BOOT_SIGNS)):
        assert LM.INFRA.found(f"FAIL x · {w} · y"), w
    from dexengine.record.session import attempts, launch

    assert launch.KIND_TEXT is LM.KIND_TEXT and launch.BOOT_SIGNS is LM.BOOT_SIGNS
    assert attempts.RETRY_KINDS is LM.RETRY_KINDS
