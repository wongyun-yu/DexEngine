"""Re-selection (`--place-reselect`, the re-selection rule) — the rule and
the judge's re-selection block (CPU only, synthetic ledger · records · clip lists).

  J  judge: boot reads the record once (marker `[PLACE_RESELECT] clips=<n> record=<path>`) · a record clip outside the
     judge list dies (ReselectError) · no record = {} · a margin on a tree out of the rule dies · structure: boot calls
     the reader on --place-reselect and clip prep passes the margin into place_rule.clip (negative variants caught)
  P1–P4 pending: GUARD_FAIL + the clip in its source chunk's record only · adopted later · other GUARD_FAIL · own chunk
  W  chunk record (--chunk-record): header + row when pending · no file when none (an old file is removed)
  G  carry guard: pending without the record dies naming the clip · same margin passes · other margin dies · own chunk
  C  the CLI runs by file path the way callers run it
  S  structure: stage2 calls the carry guard before the judge and before the bake driver (negative: removing a
     call is caught); callers outside this package that derive and pass the record are checked with their code
  E  stage2 ground_place_reselect: fresh run writes run.env place_reselect= (relative inside OUT) · resume with the same
     record passes · resume without it / with a record the judge never had dies (rc 2)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.placement.bake import place_check as PC
from dexengine.placement.judge import reselect as RS
from dexengine.placement.scene import place_rule as PR
from tests.imports import own_module

REPO = Path(__file__).resolve().parents[2]
HEAD = "\t".join(PC.RESELECT_COLS)
C1, C2, C3 = "cut_knife_pan__20231020_901", "cut_knife_pan__20231020_902", "cut_knife_pan__20231020_903"
ROW = [C1, "12.50", "14.50", "main:left_object@rest:robot_keepout"]


def _record(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEAD + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _clips(path: Path, names: list) -> Path:
    path.write_text("".join(n + "\n" for n in names), encoding="utf-8")
    return path


# ── J judge ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_load_marker_and_margin(tmp_path, capsys):
    rec = _record(tmp_path / PC.RESELECT_IN, [ROW])
    rs = RS.load(str(rec), [C1, C2])
    assert rs == {C1: pytest.approx(0.0145)}
    out = capsys.readouterr().out
    assert re.search(rf"^{re.escape(PC.RESELECT_TAG)} clips=1 record={re.escape(str(rec))} ", out, re.M)
    assert RS.margin(rs, C1) == pytest.approx(0.0145) and RS.margin(rs, C2) == 0.0
    assert RS.load("", [C1]) == {} and capsys.readouterr().out == ""


def test_load_stray_clip_dies(tmp_path):
    rec = _record(tmp_path / PC.RESELECT_IN, [ROW, [C3, "1.00", "3.00", "v"]])
    with pytest.raises(RS.ReselectError, match=C3) as e:
        RS.load(str(rec), [C1, C2])
    assert isinstance(e.value, PR.PlaceRuleError) and C1 not in str(e.value).split(":")[-1]


def test_clip_line_needs_the_rule_on():
    pr = {"on": True, "keepout": (0.1, 0.2, 0.3145)}
    assert RS.clip_line(C1, 0.0145, pr).startswith(
        f"{PC.RESELECT_TAG} {C1} margin 14.50 mm — robot spot disc radius 0.3145 m"
    )
    with pytest.raises(RS.ReselectError, match="place rule is off"):
        RS.clip_line(C1, 0.0145, {"on": False, "keepout": None})


def test_boot_reads_the_record_against_the_judge_list(tmp_path, capsys, monkeypatch):
    """_boot_clips on a stub judge: the record is checked against the judge list (stray → dies before anything else)
    and kept as _RESELECT."""
    boot = own_module("dexengine.placement.judge.boot")
    rec = _record(tmp_path / PC.RESELECT_IN, [ROW])
    a = SimpleNamespace(
        clips=str(_clips(tmp_path / "c.txt", [C1, C2])),
        dataset="taco",
        emb="bimanual",
        src_layer="base",
        place_reselect=str(rec),
        exhaust_resume="",
        desk_shrink_robot=0.0,
    )
    J = SimpleNamespace(a=a, _F0ROOT="f0", TRAJ_ROOT="t", PL_DIR="p", NBx=3, NBy=3, Wx=1, Wy=1, step=0.01)
    monkeypatch.setattr(boot.VC, "desk_bounds_dataset", lambda *x: ((0, 1), (0, 1)))
    monkeypatch.setattr(boot.VC, "desk_box_for", lambda *x: (0, 1, 0, 1))
    monkeypatch.setattr(boot.VC, "desk_size_mark", lambda *x: "[DESK] stub")
    boot.BootMixin._boot_clips(J)
    assert {C1: pytest.approx(0.0145)} == J._RESELECT
    assert f"{PC.RESELECT_TAG} clips=1 record={rec}" in capsys.readouterr().out
    a.clips = str(_clips(tmp_path / "c2.txt", [C2]))
    with pytest.raises(RS.ReselectError, match=C1):
        boot.BootMixin._boot_clips(J)
    a.place_reselect = ""
    boot.BootMixin._boot_clips(J)
    assert J._RESELECT == {}


def _judge_wired(boot_src: str, clip_src: str) -> bool:
    """Boot reads --place-reselect through reselect.load; clip prep passes reselect.margin into place_rule.clip and
    prints the per-clip line — call presence, comments ignored."""
    code = lambda s: "\n".join(ln for ln in s.splitlines() if not ln.lstrip().startswith("#"))  # noqa: E731
    b, c = code(boot_src), code(clip_src)
    return (
        "self._RESELECT = _RS.load(a.place_reselect, self.clips)" in b
        and "_PR.clip(tree_dir, arm=a.arm, margin_m=(_pm := _RS.margin(self._RESELECT, clip)))" in c
        and "_RS.clip_line(clip, _pm, pr)" in c
        and c.count("_PR.clip(") == 1
    )


def test_judge_wiring_structure():
    b = (REPO / "dexengine/placement/judge/boot.py").read_text(encoding="utf-8")
    c = (REPO / "dexengine/placement/judge/clip_inputs.py").read_text(encoding="utf-8")  # clip prep (_clip_objects)
    assert _judge_wired(b, c)
    assert not _judge_wired(b.replace("self._RESELECT = _RS.load", "# self._RESELECT = _RS.load"), c)
    assert not _judge_wired(b, c.replace(", margin_m=(_pm := _RS.margin(self._RESELECT, clip))", ""))
    assert not _judge_wired(b, c + "\n        pr = _PR.clip(tree_dir)\n")


def test_judge_parser_takes_the_record():
    from dexengine.placement.judge import args as JA  # noqa: PLC0415

    ap = JA._build_parser()
    act = {x.dest: x for x in ap._actions}["place_reselect"]
    assert act.option_strings == ["--place-reselect"] and act.default == ""


# ── P · W · G place_check (synthetic hand root) ─────────────────────────────────────────────────────────────────
@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(RR, "ledger_file", lambda axis, hand, root=None, **_: tmp_path / "ledger.tsv")
    monkeypatch.setattr(P, "run_path", lambda layer, hand="", **kw: tmp_path / "chunks" / kw["tag"])
    _record(tmp_path / "chunks" / "G1x_taco_01" / PC.RESELECT_FILE, [ROW])
    return tmp_path


def _ledger(root: Path, rows: list) -> None:
    (root / "ledger.tsv").write_text(
        "clip\tstatus\tsource_tag\tnote\n" + "".join("\t".join(r) + "\t\n" for r in rows), encoding="utf-8"
    )


def test_pending(root):
    _ledger(
        root, [[C1, "GUARD_FAIL", "G1x_taco_01"], [C2, "GUARD_FAIL", "G1x_taco_01"], [C3, "ADOPTED", "G1x_taco_02"]]
    )
    got = PC.pending("h", "taco", [C1, C2, C3])
    assert list(got) == [C1] and got[C1] == "\t".join(ROW)  # P1 · P3
    _ledger(root, [[C1, "ADOPTED", "G1x_taco_05"]])
    assert PC.pending("h", "taco", [C1]) == {}  # P2
    _ledger(root, [[C1, "GUARD_FAIL", "G1x_taco_01"]])
    assert PC.pending("h", "taco", [C1], this_tag="G1x_taco_01") == {}  # P4


def test_chunk_record_cli(root, capsys):
    _ledger(root, [[C1, "GUARD_FAIL", "G1x_taco_01"]])
    clips, out = _clips(root / "clips.txt", [C1, C2]), root / PC.RESELECT_IN
    base = ["--chunk-record", str(out), "--hand", "h", "--axis", "taco", "--clips", str(clips)]
    assert PC.main(base) == 0 and out.read_text(encoding="utf-8").startswith(HEAD + "\n")
    assert PC.read_reselect(out) == {C1: pytest.approx(0.0145)}
    assert f"{PC.RESELECT_TAG} chunk record 1 clips" in capsys.readouterr().out
    _ledger(root, [[C1, "ADOPTED", "G1x_taco_05"]])
    assert PC.main(base) == 0 and not out.exists()  # W: none = no file (old one removed)
    with pytest.raises(SystemExit):
        PC.main(["--chunk-record", str(out), "--hand", "h", "--clips", str(clips)])  # --axis missing


def test_carry_guard_cli(root):
    _ledger(root, [[C1, "GUARD_FAIL", "G1x_taco_01"]])
    base = ["--carry-guard", "--hand", "h", "--axis", "taco", "--clips", str(_clips(root / "clips.txt", [C1]))]
    with pytest.raises(PR.PlaceRuleError, match=C1):
        PC.main(base)
    out = root / PC.RESELECT_IN
    PC.write_record(out, PC.pending("h", "taco", [C1]))
    assert PC.main(base + ["--carried", str(out)]) == 0
    out.write_text(HEAD + f"\n{C1}\t12.50\t9.00\tx\n", encoding="utf-8")
    with pytest.raises(PR.PlaceRuleError):
        PC.main(base + ["--carried", str(out)])
    assert PC.main(base + ["--tag", "G1x_taco_01"]) == 0


def test_cli_by_file_path(tmp_path):
    """The lanes run `python3 dexengine/placement/bake/place_check.py …` from the repo root."""
    rec = _record(tmp_path / PC.RESELECT_IN, [ROW])
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "dexengine/placement/bake/place_check.py", "--reselect-check", str(rec)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0 and r.stdout.startswith(f"{PC.RESELECT_TAG} record {rec} clips 1 · margin 14.50–14.50 mm")


# ── S structure ───────────────────────────────────────────────────────────────────────────────────────────────────
def _code(path: str) -> str:
    return "\n".join(
        ln for ln in (REPO / path).read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")
    )


def guard_wired(src: str) -> bool:
    """stage2 checks the carry guard once before the judge call and once between it and the bake driver call."""
    lines = src.splitlines()
    g = [i for i, ln in enumerate(lines) if re.search(r"ground_carry_guard \"\$(CLIPS|OUT/passed\.txt)\"", ln)]
    j = [i for i, ln in enumerate(lines) if "-m dexengine.placement.judge.core" in ln]
    b = [i for i, ln in enumerate(lines) if "bash scripts/stage2/bake_drive.sh" in ln]
    return bool(g and j and b) and min(g) < min(j) and any(min(j) < x < min(b) for x in g)


def test_stage2_structure():
    st = _code("scripts/stage2_ground.sh")
    assert "    --place-reselect) PLACE_RESELECT=$2; shift ;;" in st and '"${PRS[@]}" \\' in st
    assert re.search(
        r'bake_drive\.sh "\$COMBO_V" "\$OUT/plans\.jsonl" "\$OUT/passed\.txt" "\$BAKE_NPAR" '
        r'"\$PLACE_RESELECT"',
        st,
    )
    assert guard_wired(st)
    assert not guard_wired("\n".join(ln for ln in st.splitlines() if "ground_carry_guard" not in ln))
    assert not guard_wired("\n".join(ln for ln in st.splitlines() if 'ground_carry_guard "$OUT/passed.txt"' not in ln))
    bd = _code("scripts/stage2/bake_drive.sh")
    assert '${RESELECT_IN:+--reselect-in "$RESELECT_IN"}' in bd and "RESELECT_IN=${5:-}" in bd


# ── E stage2 ground_place_reselect ────────────────────────────────────────────────────────────────────────────────
def _ground(out: Path, record: str, skip_judge: int, keep: int, clips: Path) -> subprocess.CompletedProcess:
    script = '. scripts/stage2/_ground_lib.sh; ground_place_reselect; rc=$?; echo "rc=$rc prs=${PRS[*]}"; exit $rc'
    env = dict(
        os.environ,
        PYTHONPATH=str(REPO),
        OUT=str(out),
        PYI=sys.executable,
        CLIPS=str(clips),
        SKIP_JUDGE=str(skip_judge),
        _KEEP_RUN_ENV=str(keep),
        _LEDGER_KEY="taco",
        PLACE_RESELECT=record,
    )
    return subprocess.run(["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_ground_place_reselect(tmp_path):
    out = tmp_path / "chunk"
    out.mkdir()
    clips = _clips(tmp_path / "clips.txt", [C1, C2])
    rec = _record(out / PC.RESELECT_IN, [ROW])
    (out / "run.env").write_text("tag=t\n", encoding="utf-8")
    r = _ground(out, str(rec), 0, 0, clips)  # fresh judge run
    assert r.returncode == 0 and f"prs=--place-reselect {rec}" in r.stdout, r.stderr
    assert (out / "run.env").read_text(encoding="utf-8").endswith(f"place_reselect={PC.RESELECT_IN}\n")
    r = _ground(out, str(rec), 1, 1, clips)  # resume, same record
    assert r.returncode == 0 and (out / "run.env").read_text(encoding="utf-8").count("place_reselect=") == 1
    r = _ground(out, "", 1, 1, clips)  # resume without it
    assert r.returncode == 2 and "re-selection record 'none' != the one the judge used" in r.stderr
    (out / "run.env").write_text("tag=t\n", encoding="utf-8")
    r = _ground(out, str(rec), 1, 1, clips)  # judge never had one
    assert r.returncode == 2 and "the one the judge used 'none'" in r.stderr
    assert _ground(out, "", 1, 1, clips).returncode == 0  # none / none = unchanged
    r = _ground(out, str(_record(tmp_path / "x.tsv", [[C3, "1.00", "3.00", "v"]])), 0, 0, clips)
    assert r.returncode == 2  # record clip outside --clips
