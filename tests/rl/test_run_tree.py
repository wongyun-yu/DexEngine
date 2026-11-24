"""Where a launch's records land when the code tree, the cwd and the data root are three places (release item: run
logs and outputs landed in the code tree, the cwd of the trainer). One rule decides — paths.run_root_of(data root,
code tree) — and every reader asks it: the launcher (Request.run_tree), the trainer process (paths.work_area through
env_build / seqdir, and its shell window `paths --what work-area`), the coll-preset staging of replay and record
(cell_session.stage_coll_preset) and finalize (cell_state.work_areas · root_guard.guarded_import), for every kind of
data root (layout.root_kind). The cwd is always a third folder."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.core.layout import generations as G
from dexengine.core.layout import shadow as SH
from dexengine.core.layout.runs import RUN_CKPT_ROOT, RUN_RENDER_OUT, RUN_WORK_AREAS
from dexengine.rl import cell_session as CR
from dexengine.rl import cell_state as CH
from dexengine.rl import launch
from dexengine.rl import root_guard as HR
from dexengine.rl.launch import dispatch, process, records
from dexengine.rl.launch import plan as PL
from tests.rl.test_launch import MANIFEST, parse_case, setup_case

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[2]
#: canon_tree = the data root is the code tree (production) · canon_other = a main checkout apart from the code tree
#: (a dev worktree) · generation · folder (a release or copied data root, the GPU run) · shadow (over a folder base)
SHAPES = ("canon_tree", "canon_other", "generation", "folder", "shadow")


def _shape(shape: str, tree: Path, main: Path, tmp: Path) -> tuple[Path, Path]:
    """Make `main` a data root of kind `shape` for code tree `tree` → (the data root to give, the run tree it means)."""
    git = main / ".git"
    if shape == "canon_tree":
        return tree, tree
    if shape == "canon_other":
        git.mkdir(parents=True, exist_ok=True)
        return main, tree
    if git.is_dir():
        git.rmdir()
    if shape == "generation":
        (main / G.GENERATION_FILE).write_text(json.dumps(G.generation_record("hot3d_lfix")))
        return main, main
    if shape == "folder":
        return main, main
    base = tmp / "base"
    base.mkdir()
    SH.write_shadow_record(str(main), str(base), "aug_place")
    return main, base


def _same(a, b) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_rule_by_data_root_kind(shape, tmp_path):
    tree, main = tmp_path / "tree", tmp_path / "main"
    tree.mkdir()
    main.mkdir()
    data, want = _shape(shape, tree, main, tmp_path)
    assert _same(P.run_root_of(data, tree), want)
    assert P.work_area("work_log", want) == want / RUN_WORK_AREAS["work_log"]
    assert P.run_out("results/x", want) == want / "results/x" and P.run_out("/abs/x", want) == Path("/abs/x")


@pytest.mark.parametrize("shape", SHAPES)
def test_the_trainer_window_answers_the_same_rule(shape, tmp_path):
    """The trainer process (PYTHONPATH = the code tree, the launch's DEXENGINE_MAIN_ROOT, any cwd) finds the work areas
    where the launcher puts them — asked through the shell window, which calls the function env_build and seqdir
    call. The window answers in the --rel form (paths.run_rel): code-tree relative when the run tree is the code tree
    (what the judge sidecar, run from the code tree, has always read), else absolute."""
    main = tmp_path / "main"
    main.mkdir()
    data, want = _shape(shape, REPO, main, tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DEXENGINE_")}
    env.update(PYTHONPATH=str(REPO), DEXENGINE_MAIN_ROOT=str(data))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    for area, rel in RUN_WORK_AREAS.items():
        r = subprocess.run(
            [sys.executable, "-m", "dexengine.core.paths", "--what", "work-area", "--area", area],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, r.stderr
        got = r.stdout.strip()
        assert got == (rel if _same(want, REPO) else str(Path(os.path.realpath(want)) / rel))
        assert _same(REPO / got, P.run_root_of(data, REPO) / rel)
    assert list(cwd.iterdir()) == []


@pytest.mark.parametrize("shape", [s for s in SHAPES if s != "shadow"])
def test_launch_records_land_in_the_run_tree(shape, tmp_path, monkeypatch):
    """The four families the GPU run found in its code tree: launch logs (overfit/logs/refbase[.dryrun]), the roster
    record, the run folder with coll_preset.txt, the replay video folder."""
    case = HERE / "launch_cases" / "replay_bi.case"
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    data, want = _shape(shape, root, tmp_path / "main", tmp_path)
    sub["DEXENGINE_MAIN_ROOT"] = str(data)
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(data))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with contextlib.redirect_stdout(io.StringIO()):
        req = launch.from_env(run, sub, root=root)
        got = launch.resolve(req)
        man = dispatch.manifest(PL.prepare(req), dispatch.out_name(req))
    assert _same(req.run_tree, want)
    in_tree = _same(want, root)  # canonical shapes: records keep their relative form
    # launch logs (a dry run: the .dryrun twin)
    assert got and all(_same(p.log.parent, want / f"{RUN_WORK_AREAS['work_log']}.dryrun") for p in got)
    # roster record (the trainer writes it in work_log; the manifest names it)
    roster = next(t for ln in man.splitlines() for t in ln.split() if t.startswith("ROSTER_RECORD="))
    rel = f"{RUN_WORK_AREAS['work_log']}/{dispatch.out_name(req)}_roster.txt"
    assert roster == f"ROSTER_RECORD={rel if in_tree else req.run_tree / rel}"
    # replay video folder (argv of each trainer process)
    outs = {p.argv[p.argv.index("--videos_out_dir") + 1] for p in got}
    vid = f"{RUN_RENDER_OUT}/{run}_replay4view"
    assert outs == {vid if in_tree else str(req.run_tree / vid)}
    # run folder + coll_preset.txt: staged where the launcher's check reads it
    folder = tmp_path / "trained_run"
    folder.mkdir()
    (folder / "coll_preset.txt").write_text("canon_r2r3off\n")
    assert CR.stage_coll_preset(folder, root, "ALIAS", data_root=data)
    marker = records.coll_preset_marker(req.run_tree, "ALIAS")
    assert marker.read_text() == "canon_r2r3off\n" and _same(marker.parent, want / RUN_CKPT_ROOT / "ALIAS")
    if not in_tree:  # nothing of the run in the code tree or the cwd
        assert sorted(x.name for x in root.iterdir()) == ["assets", "configs", "dexengine", "scripts"]
        assert list(cwd.iterdir()) == []
        # the trainer process of this launch: its environment (process.child_environ) reads the same run tree
        isaac = tmp_path / "isaac"
        (isaac / "env_isaaclab/bin").mkdir(parents=True)
        (isaac / "env_isaaclab/bin/python").symlink_to(sys.executable)
        monkeypatch.setenv("ISAACLAB_DIR", str(isaac))
        cenv = process.child_environ(got[0], REPO)
        r = subprocess.run(
            [sys.executable, "-m", "dexengine.core.paths", "--what", "work-area", "--area", "work_log"],
            cwd=cwd,
            env=cenv,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, r.stderr
        assert _same(r.stdout.strip(), records.log_dir(False, req.run_tree))


@pytest.mark.parametrize("shape", [s for s in SHAPES if s != "shadow"])  # shadow: this process would install the
def test_this_process_run_tree(shape, tmp_path, monkeypatch):  # shadow write guard (window test has it)
    """run_root() / run_out() of this process (the default every stage uses — stage1 --variant, admit --log-dir,
    the trainer's video folders): a dev worktree naming the main checkout keeps its outputs (canon_other)."""
    main = tmp_path / "main"
    main.mkdir()
    data, want = _shape(shape, REPO, main, tmp_path)
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(data))
    assert _same(P.run_root(), want)
    assert _same(P.run_out("overfit/stage1_x/taco_retarget"), want / "overfit/stage1_x/taco_retarget")


@pytest.mark.parametrize("shape", ["generation", "folder"])
def test_finalize_imports_from_the_work_area(shape, tmp_path, monkeypatch):
    """Work areas and hand roots share one tree: guarded_import (os.rename only) takes a run folder out of the work
    area this process's launches wrote — no cross-tree move."""
    main = tmp_path / "main"
    main.mkdir()
    data, want = _shape(shape, REPO, main, tmp_path)
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(data))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    assert _same(P.run_root(), want)
    _ref, ckr = CH.work_areas()
    assert _same(ckr, want / RUN_CKPT_ROOT)
    (ckr / "RUN" / "checkpoints").mkdir(parents=True)
    (ckr / "RUN" / "checkpoints" / "agent_1.pt").touch()
    head = P.data_layout.RUN_LAYERS["run_root"].split("{", 1)[0].rstrip("/")
    dst = want / head / "sharpa_wave" / "campaigns" / "c" / "RUN"
    HR.guarded_import(ckr / "RUN", dst, "sharpa_wave")
    assert (dst / "checkpoints" / "agent_1.pt").is_file() and not (ckr / "RUN").exists()
    assert list(cwd.iterdir()) == []
