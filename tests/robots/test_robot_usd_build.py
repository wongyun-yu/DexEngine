"""Robot USD driver (dexengine.robots.usd.build, scripts/build_robot_usd.py): plan, input check, counted runner.
Isaac is never started: the runner test replaces the Isaac launcher with a stand-in that writes the step outputs."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.robots.usd import build as B

REPO = Path(__file__).resolve().parents[2]
SHARPA = "rby1_sharpa_wave"  # the hand whose vendor meshes are committed (assets/robots/hands)


def test_canonical_output_is_the_training_usd(tmp_path):
    from dexengine.robots import vendor

    un = vendor.unavailable_hands()
    plans = B.plan(B.default_dirs(), tmp_path)
    want = {P.combo_dirname(c) for c in P.combo_ids() if P.split_combo(c, P.ARM_IDS, P.HAND_IDS)[1] not in un}
    assert {p.dir for p in plans} == want
    for p in plans:
        assert p.canonical and p.urdf.is_file()
        rel = p.out("reporters").relative_to(tmp_path / B.ROBOT_USD_REL)
        assert str(rel) == P.combo_usd_relpath(P.combo_id(p.arm, p.hand))
    var = B.plan([SHARPA + "_truncated"], tmp_path)[0]
    assert not var.canonical and var.variant.endswith("_truncated")
    with pytest.raises(ValueError):
        B.plan(["not_a_combo"], tmp_path)


def test_commands_order_and_scripts(tmp_path):
    plans = B.plan([SHARPA], tmp_path)
    steps = [st for st, _d, _a in B.commands(plans)]
    assert steps == [s for s, _ in B.STEPS]
    for _st, _d, argv in B.commands(plans):
        assert (REPO / argv[0]).is_file()
    intermediates = [plans[0].out(s).name for s in ("convert", "flatten", "export")]
    assert all("_test" in n for n in intermediates)  # the step scripts refuse other names under usd_cache


def _cli(*args, root):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/build_robot_usd.py"), *args, "--data-root", str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )


def test_dry_run_checks_and_writes_nothing(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    r = _cli("--hand", "sharpa_wave", "--dry-run", root=root)
    assert r.returncode == 0, r.stderr[-400:]
    assert f"[ROBOT_USD_PLAN] {SHARPA} " in r.stdout and "missing 0 · importer≠resolver 0" in r.stdout
    assert r.stdout.count("[ROBOT_USD_CMD]") == len(B.STEPS)
    assert list(root.iterdir()) == []
    bad = _cli("--combos", SHARPA + "_nosuch", "--dry-run", root=root)
    assert bad.returncode == 2 and "merged URDF missing" in bad.stderr


def _converter_records(usd_dir, urdf, ah=None):
    """What Isaac Lab's converter leaves next to its output: config.yaml and .asset_hash (its formula unless ``ah``)."""
    usd_dir.mkdir(parents=True, exist_ok=True)
    cfg = f"asset_path: {urdf}\nusd_dir: {usd_dir}\nusd_file_name: a.usd\nfix_base: true\n"
    (usd_dir / "config.yaml").write_text(cfg)
    md5 = hashlib.md5(json.dumps({"fix_base": True}).encode())
    md5.update(Path(urdf).read_bytes())
    (usd_dir / ".asset_hash").write_text(ah or md5.hexdigest())


def test_built_from_reads_the_bake_record(tmp_path):
    """ "same" only while the URDF bytes, the converter record and the USD bytes are the ones the bake bound; each
    change is named, and undoing it is "same" again."""
    p = B.plan([SHARPA], tmp_path)[0]
    assert B.built_from(p) == "no record"
    _converter_records(p.usd_dir, p.urdf)
    usd = p.out("reporters")
    usd.write_text("canonical")
    assert B.built_from(p) == "no record"  # converted, never bound
    rec = B.write_bake_record(p.usd_dir, p.urdf, usd)
    assert rec["usd"] == usd.name and rec["usd_md5"] == hashlib.md5(b"canonical").hexdigest()
    assert B.built_from(p) == "same"
    usd.write_text("canonical, re-written after the bake")
    assert B.built_from(p) == B.STALE
    usd.write_text("canonical")
    assert B.built_from(p) == "same"
    ah = (p.usd_dir / ".asset_hash").read_text()
    (p.usd_dir / ".asset_hash").write_text("0" * 32)
    assert B.built_from(p) == "converter record changed"
    (p.usd_dir / ".asset_hash").write_text(ah)
    cfg = (p.usd_dir / "config.yaml").read_text()
    (p.usd_dir / "config.yaml").write_text(cfg.replace("fix_base: true", "fix_base: false"))
    assert B.built_from(p) == "converter config changed"
    (p.usd_dir / "config.yaml").write_text(cfg)
    assert B.built_from(p) == "same"


def test_a_bake_record_needs_the_converter_record_of_these_bytes(tmp_path):
    """A conversion that left an old .asset_hash (a re-bake that wrote the USD only) cannot be bound."""
    p = B.plan([SHARPA], tmp_path)[0]
    _converter_records(p.usd_dir, p.urdf, ah="0" * 32)
    p.out("reporters").write_text("canonical")
    with pytest.raises(RuntimeError, match="not the converter record"):
        B.write_bake_record(p.usd_dir, p.urdf, p.out("reporters"))
    assert not (p.usd_dir / B.BAKE_RECORD).exists()


_FAKE = r"""
import shutil, sys
from pathlib import Path
script, args = Path(sys.argv[1]).name, sys.argv[2:]
def val(flag): return args[args.index(flag) + 1]
fail = Path(sys.argv[0]).with_name("FAIL").read_text().strip() if Path(sys.argv[0]).with_name("FAIL").exists() else ""
if script == fail:
    print("Traceback: stand-in failure"); sys.exit(1)
if script == "urdf_to_usd.py":
    import hashlib, json
    out, urdf = Path(val("--output-usd")), Path(val("--input-urdf"))
    out.write_text("base")
    cfg = f"asset_path: {urdf}\nusd_dir: {out.parent}\nusd_file_name: {out.name}\nfix_base: true\n"
    (out.parent / "config.yaml").write_text(cfg)
    md5 = hashlib.md5(json.dumps({"fix_base": True}).encode()); md5.update(urdf.read_bytes())
    (out.parent / ".asset_hash").write_text("0" * 32 if fail == "asset_hash" else md5.hexdigest())
elif script == "flatten.py": Path(val("--output")).write_text("flat")
elif script == "mimic.py": Path(val("--output-usd")).write_text("stage3")
elif script == "contact_report_bake.py":
    root = Path(val("--usd-root"))
    for c in val("--combos").split(","):
        src = root / c / f"{c}_stage3_test.usd"
        if src.is_file():
            shutil.copy(src, root / c / f"{c}.usd"); print(f"[stage4] {c}: applied=1 already=0 joints 1->1 OK")
        else:
            print(f"[stage4] {c}: MISSING {src}"); sys.exit(1)
elif script == "check_colors.py": print(f"[VISUAL_COLORS] {Path(val('--usd')).name} · links compared 3 ✓")
"""


def _fake_launcher(tmp_path, monkeypatch, fail=""):
    f = tmp_path / "fake" / "fake_isaac.py"
    f.parent.mkdir()
    f.write_text(_FAKE)
    if fail:
        (f.parent / "FAIL").write_text(fail)
    monkeypatch.setattr(B, "_ISAAC_RUN", f'exec "{sys.executable}" "{f}" "$@"')


def test_runner_counts_outputs(tmp_path, monkeypatch, capsys):
    _fake_launcher(tmp_path, monkeypatch)
    root = tmp_path / "root"
    plans = B.plan([SHARPA], root)
    old = plans[0].out("reporters")
    old.parent.mkdir(parents=True)
    old.write_text("old canonical")
    os.utime(old, (1, 1))  # an old canonical file is replaced, not counted
    assert B.run(plans, tmp_path / "logs", root) == 0
    out = capsys.readouterr().out
    assert "COMBO_USD_DONE stage3 1/1 · stage4 rc=0 · canonical usd 1/1 · colour check 1/1 · bake record 1/1" in out
    assert old.read_text() == "stage3"
    assert B.built_from(plans[0]) == "same" and f"[BAKE_RECORD] {SHARPA} asset_hash" in out


def test_runner_fails_when_the_converter_record_is_stale(tmp_path, monkeypatch, capsys):
    _fake_launcher(tmp_path, monkeypatch, fail="asset_hash")
    root = tmp_path / "root"
    plans = B.plan([SHARPA], root)
    assert B.run(plans, tmp_path / "logs", root) == 1
    out = capsys.readouterr().out
    assert f"[BAKE_RECORD] {SHARPA} FAIL" in out and "bake record 0/1" in out
    assert B.built_from(plans[0]) == "no record"


def test_runner_fails_on_a_missing_step_output(tmp_path, monkeypatch, capsys):
    _fake_launcher(tmp_path, monkeypatch, fail="flatten.py")
    root = tmp_path / "root"
    assert B.run(B.plan([SHARPA], root), tmp_path / "logs", root) == 1
    out = capsys.readouterr().out
    assert f"{SHARPA}: no stage3 output" in out and "COMBO_USD_DONE stage3 0/1" in out


def test_default_dirs_leave_out_an_unavailable_hand_and_asking_for_it_dies(monkeypatch):
    """Negative: a hand whose vendor source is not set is not in the default bake list, and naming it dies."""
    from dexengine.robots import vendor

    monkeypatch.setattr(vendor, "unavailable_hands", lambda: {"sharpa_wave": "source ['x'] not set"})
    assert all("sharpa_wave" not in d for d in B.default_dirs())
    with pytest.raises(SystemExit, match="not available"):
        B.default_dirs(hand="sharpa_wave")
