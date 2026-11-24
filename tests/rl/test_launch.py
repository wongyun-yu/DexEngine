"""The launcher resolves every recorded launch to the exact trainer argv and environment of the old shell chain.

Each `launch_cases/<case>.case` holds a launch environment (`## env`) and the trainer dumps the old launcher wrote
for it (`## dump <log>`: argv lines, then `ENV::` lines), after the mechanical renames (entry
`dexengine/rl/cli.py`, `DEXENGINE_` names). The test rebuilds the inputs under a temporary data root, resolves the
launch and compares dump by dump. Two inputs owned elsewhere are fixed to the recorded values: the bimanual data
checks (`preflight.check_bimanual`, which reads real trees) and the render look (`dexengine.record.looks.render_look`).

The recorded dumps carry the RL switches as `ENV::` lines (environment names); a resolved process carries them in its
switches file. `switch_view` compares a resolved process against such a dump: argv and the other environment lines byte
for byte, the
switches as typed values, and the set difference of the environment names = exactly the converted switches.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from pathlib import Path

import pytest

from dexengine.core.layout.runs import RUN_CKPT_ROOT
from dexengine.core.repo import REPO_ROOT
from dexengine.rl import launch
from dexengine.rl.launch import dispatch as run_mod
from dexengine.rl.launch import modes, preflight, recipe
from dexengine.rl.launch import plan as plan_mod
from dexengine.rl.launch import process as process_mod
from dexengine.rl.tasks import switches

CASES = sorted(Path(__file__).with_name("launch_cases").glob("*.case"))
LOOK_KEYS = ("AUG_DESK_MDL", "AUG_FLOOR_MDL", "AUG_LIGHT", "AUG_ROOM", "KEY_LIGHT", "LOG_EGO_PITCH", "CAMERA_FOCAL_MM")
TMPVIEW = re.compile(r"/[^\n]*/replay_seq_[A-Za-z0-9_]+")
MANIFEST = "<manifest>"
HEAD = re.compile(r"^date=\S+ git=\S* dirty=\S+ code=\S* codedirty=\S+ ")


def parse_case(path: Path) -> tuple[dict, dict]:
    """(launch env, {log name: dump text}); the old manifest is under the key MANIFEST."""
    env, dumps, cur = {}, {}, None
    for ln in path.read_text().splitlines(keepends=True):
        if ln.startswith("## env"):
            cur = None
        elif ln.startswith("## manifest"):
            cur = MANIFEST
            dumps[cur] = ""
        elif ln.startswith("## dump "):
            cur = ln[8:].strip()
            dumps[cur] = ""
        elif cur is None:
            k, _, v = ln.rstrip("\n").partition("=")
            env[k] = v
        else:
            dumps[cur] += ln
    return env, dumps


def build_root(tmp: Path, env: dict, dumps: dict) -> tuple[Path, dict]:
    """Code tree (symlinks to this checkout), data root <MAIN> and scratch <S>; returns (root, substituted env)."""
    root, main, scratch = tmp / "tree", tmp / "main", tmp / "s"
    root.mkdir()
    # The cases were recorded in the canonical checkout (data root = the main checkout), whose run tree is the code
    # tree the trainer runs from (paths.run_root_of): <MAIN> is a main checkout here too. A data root apart from the
    # code tree is tests/rl/test_run_tree.py.
    (main / ".git").mkdir(parents=True)
    for d in ("dexengine", "scripts", "configs", "assets"):
        (root / d).symlink_to(REPO_ROOT / d)
    sub = {k: v.replace("<MAIN>", str(main)).replace("<S>", str(scratch)) for k, v in env.items()}
    sub["DEXENGINE_MAIN_ROOT"] = str(main)
    seq = Path(sub["SEQ_DIR"])
    seq.mkdir(parents=True, exist_ok=True)
    for log in dumps:
        if m := re.match(r"EXT_replay_\w+?_(p\d{4}_.+_pl_rt)_replay_render\.log$", log):
            (seq / m.group(1)).mkdir(exist_ok=True)
    if sub.get("RENDER_RUNDIR"):
        (Path(sub["RENDER_RUNDIR"]) / "checkpoints").mkdir(parents=True)
        for ck in sub["RENDER_ONLY"].split():
            if any(f"_vid_{ck[:-3]}_" in log for log in dumps):
                (Path(sub["RENDER_RUNDIR"]) / "checkpoints" / ck).touch()
        (Path(sub["RENDER_RUNDIR"]) / "coll_preset.txt").write_text("canon_r2r3off\n")
    if sub.get("FROZEN_REPLAY_CKPT") and sub["SIDE"] == "both":
        d = root / RUN_CKPT_ROOT / RUN_OF[env_key(env)]
        d.mkdir(parents=True)
        (d / "coll_preset.txt").write_text("canon_r2r3off\n")
    return root, sub


RUN_OF: dict = {}


def parse_dump(text: str) -> tuple[list[str], dict]:
    argv, env = [], {}
    for ln in text.splitlines():
        if ln.startswith("ENV::"):
            k, _, v = ln[5:].partition("=")
            env[k] = v
        elif ln.strip():
            argv.append(ln)
    return argv, env


def switch_view(p, recorded: str, norm) -> list[str]:
    """Differences between resolved process `p` and a recorded pre-file dump ([] = the same launch)."""
    argv_old, env_old = parse_dump(recorded)
    argv_new, env_new = parse_dump(norm(p.dump()))
    bad = [] if argv_new == argv_old else ["argv"]
    path = env_new.pop(switches.SWITCHES_ENV, None)
    if path != norm(str(process_mod.switches_path(p.log))):
        bad.append(f"{switches.SWITCHES_ENV}={path}")
    sw_old = {k: v for k, v in env_old.items() if k in switches.FIELD_OF}
    rest_old = {k: v for k, v in env_old.items() if k not in switches.FIELD_OF}
    if env_new != rest_old:
        bad.append(f"env {sorted(set(env_new.items()) ^ set(rest_old.items()))}")
    if set(env_old) - set(env_new) != set(sw_old) or set(env_new) - set(env_old):
        bad.append("converted names")
    if set(switches.env_texts(p.switches, p.given)) != set(sw_old):
        bad.append(f"given {sorted(p.given)} vs recorded {sorted(sw_old)}")
    if p.switches != switches.from_env_names(sw_old)[0]:
        bad.append("switch values")
    return bad


def env_key(env: dict) -> str:
    return repr(sorted(env.items()))


@pytest.mark.parametrize("case", CASES, ids=[c.stem for c in CASES])
def test_resolved_launch_matches_recorded_dump(case, tmp_path, monkeypatch):
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        got = {p.log.name: p for p in launch.resolve(launch.from_env(run, sub, root=root))}

    def norm(v):
        return TMPVIEW.sub("<TMPVIEW>", v.replace(str(tmp_path / "main"), "<MAIN>").replace(str(tmp_path / "s"), "<S>"))

    assert sorted(got) == sorted(dumps)
    for log, text in dumps.items():
        assert switch_view(got[log], text, norm) == [], log


@pytest.mark.parametrize("case", CASES, ids=[c.stem for c in CASES])
def test_manifest_matches_recorded_manifest(case, tmp_path, monkeypatch):
    """The launch manifest (<OUT>_cmd.txt, read by the judge sidecar and record) keeps every line of the old one; only
    the date/commit/code-id head, the new RECIPE:: lines and the order of the environment lines may differ."""
    env, dumps = parse_case(case)
    old = dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        req = launch.from_env(run, sub, root=root)
        p = plan_mod.prepare(req)
        new = run_mod.manifest(p, run_mod.out_name(req))
    new = new.replace(str(tmp_path / "main"), "<MAIN>").replace(str(tmp_path / "s"), "<S>")
    new_lines = [HEAD.sub("", ln) for ln in new.splitlines() if not ln.startswith(("RECIPE::", "SNAPSHOT="))]

    def split(lines):
        return (
            [ln for ln in lines if not ln.startswith("DEXENGINE_")],
            sorted(ln for ln in lines if ln.startswith("DEXENGINE_")),
        )

    assert split(new_lines) == split(old.splitlines())


def setup_case(case, env, dumps, tmp_path, monkeypatch):
    run = ("EXT_" if any(k.startswith("EXT_") for k in dumps) else "GOLD_") + case.stem
    RUN_OF[env_key(env)] = run
    root, sub = build_root(tmp_path, env, dumps)
    want_env = {}
    for text in dumps.values():
        want_env.update(ln[5:].split("=", 1) for ln in text.splitlines() if ln.startswith("ENV::"))
    look = {f"DEXENGINE_{k}": want_env[f"DEXENGINE_{k}"] for k in LOOK_KEYS if f"DEXENGINE_{k}" in want_env}
    look = {k: v.replace("<MAIN>", str(tmp_path / "main")) for k, v in look.items()}
    monkeypatch.setattr(preflight, "check_bimanual", lambda *a, **k: "")
    monkeypatch.setattr(
        modes,
        "render_look_env",
        lambda plan, physx_lite: {
            **({"DEXENGINE_PHYSX_LITE": "1"} if physx_lite else {}),
            **look,
        },
    )
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(tmp_path / "main"))
    return run, root, sub


def test_recipe_columns_are_strings_and_paths_differ():
    cols = recipe.columns()
    assert set(cols) == {"common", "uni", "bi"}
    assert recipe.defaults("uni")["KL_THRESH"] == "0.02" and recipe.defaults("bi")["KL_THRESH"] == "0.04"
    assert recipe.defaults("bi")["COLL_PRESET"] == "canon_r2r3off"


def test_table_values_outer_value_wins_and_derives_usd_variant():
    env = {
        "CTE_PATH": "bi",
        "DEXENGINE_ARM": "rby1",
        "DEXENGINE_HAND": "sharpa_wave",
        "DEXENGINE_DATASET": "hot3d",
        "DEXENGINE_EMBODIMENT": "bimanual",
        "NE": "2048",
        "LAGW": "",
    }
    got = recipe.table_values(
        ("NE", "LAGW", "MI", "DEXENGINE_CKPT_INTERVAL", "DEXENGINE_COMBO_USD_VARIANT", "UNKNOWN_KEY"), env
    )
    assert got == {
        "NE": "2048",
        "LAGW": "25",
        "MI": "1600",
        "DEXENGINE_CKPT_INTERVAL": "50",
        "DEXENGINE_COMBO_USD_VARIANT": "_mi_armhull",
        "UNKNOWN_KEY": "",
    }
    with pytest.raises(recipe.RecipeError):
        recipe.table_values(("NE",), {**env, "DEXENGINE_DATASET": ""})


@pytest.mark.parametrize(
    "bad",
    [
        {"SMOKE": "1", "RENDER_ONLY": "replay"},  # two modes
        {"PROBE": "1"},  # removed input
        {"DEXENGINE_TRAJ_RECORD": "/x.npz"},  # judge-only input on a training launch
        {"RENDER_RUNDIR": "/x"},  # render-only input
    ],
)
def test_request_refuses_conflicting_or_removed_inputs(bad):
    base = {
        "SIDE": "both",
        "GPU": "0",
        "SEQ_DIR": "/x",
        "DEXENGINE_ARM": "rby1",
        "DEXENGINE_HAND": "sharpa_wave",
        "DEXENGINE_DATASET": "hot3d",
        "DEXENGINE_EMBODIMENT": "bimanual",
    }
    with pytest.raises(recipe.RecipeError):
        launch.from_env("r", {**base, **bad})


def test_preset_conflict_exits_7(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "check_bimanual", lambda *a, **k: "")
    env = {
        "SIDE": "both",
        "GPU": "0",
        "SEQ_DIR": str(tmp_path),
        "DEXENGINE_ARM": "rby1",
        "DEXENGINE_HAND": "sharpa_wave",
        "DEXENGINE_DATASET": "hot3d",
        "DEXENGINE_EMBODIMENT": "bimanual",
        "DEXENGINE_BI_SELFCOLL": "1",
    }
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as e:
        launch.resolve(launch.from_env("r", env, root=tmp_path))
    assert e.value.code == 7


def test_render_only_clips_goes_after_the_retry_count(tmp_path, monkeypatch):
    """VIDEOS_ONLY_CLIPS (the cell render tool's --clips) adds `--videos_only_clips <tokens>` where the old shell put
    it; every other argv and env line stays as recorded."""
    case = next(c for c in CASES if c.stem == "render_bi_sharpa_wave")
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        got = {
            p.log.name: p for p in launch.resolve(launch.from_env(run, {**sub, "VIDEOS_ONLY_CLIPS": "a,b"}, root=root))
        }
    assert sorted(got) == sorted(dumps)
    for log, text in dumps.items():
        new = got[log].argv
        i = new.index("--videos_allfail_retry_batches")
        assert new[i + 2 : i + 5] == ["--videos_only_clips", "a,b", "--videos_rank_metric"]
        del new[i + 2 : i + 4]

        def norm(v):
            return TMPVIEW.sub(
                "<TMPVIEW>", v.replace(str(tmp_path / "main"), "<MAIN>").replace(str(tmp_path / "s"), "<S>")
            )

        assert switch_view(got[log], text, norm) == [], log


#: The data root the suite was started with (the shared conftest gives every test an empty one).
REAL_DATA_ROOT = os.environ.get("DEXENGINE_MAIN_ROOT", "")


@pytest.mark.skipif(not REAL_DATA_ROOT, reason="needs the data root (DEXENGINE_MAIN_ROOT=<root> pytest ...)")
def test_bimanual_preflight_on_real_trees(monkeypatch):
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", REAL_DATA_ROOT)
    main = Path(REAL_DATA_ROOT)
    seq = main / "data/hot3d/trees/rby1_sharpa_wave/bimanual"
    if not seq.is_dir():
        pytest.skip(f"no tree {seq}")
    with contextlib.redirect_stdout(io.StringIO()):
        assert preflight.check_bimanual(str(seq), "rby1", "sharpa_wave", "hot3d", "bimanual", {}, REPO_ROOT) == ""
        with pytest.raises(recipe.RecipeError):
            preflight.check_bimanual(str(seq), "rby1", "xhand_pro", "hot3d", "bimanual", {}, REPO_ROOT)


def _uni_case(tmp_path, monkeypatch, **extra):
    case = next(c for c in CASES if c.stem == "train_uni_right_sharpa_wave")
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        return root, launch.resolve(launch.from_env(run, {**sub, **extra}, root=root))


def test_set_field_reaches_the_switches_file(tmp_path, monkeypatch):
    """`--set arm_ff=0` (scripts/stage3_rl.py puts it into the launch environment) ends in the process's file."""
    root, (p,) = _uni_case(tmp_path, monkeypatch, arm_ff="0", obj_spawn_hold="3")
    assert (p.switches.arm_ff, p.switches.obj_spawn_hold) == (False, 3)
    assert {"arm_ff", "obj_spawn_hold"} <= p.given
    assert process_mod.start(p, root, dry_run=True) == 0
    sw, given = switches.read(p.env[switches.SWITCHES_ENV])
    assert (sw, given) == (p.switches, p.given)
    argv, env = process_mod.load_dump(p.log)
    assert env[switches.SWITCHES_ENV] == str(process_mod.switches_path(p.log))


def test_field_and_old_name_must_agree(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(recipe.RecipeError, match="disagree"):
        _uni_case(tmp_path / "a", monkeypatch, arm_ff="0", DEXENGINE_ARM_FF="1")
    with pytest.raises(recipe.RecipeError, match="not 0 | 1"):
        _uni_case(tmp_path / "b", monkeypatch, arm_ff="maybe")


@pytest.mark.parametrize("case", CASES, ids=[c.stem for c in CASES])
def test_the_switches_path_is_the_only_switch_carrier(case, tmp_path, monkeypatch):
    """No converted switch name reaches a trainer process, even when the launching shell exported one."""
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        got = launch.resolve(launch.from_env(run, sub, root=root))
    monkeypatch.setenv("ISAACLAB_DIR", str(tmp_path / "isaac"))
    monkeypatch.setenv("DEXENGINE_ARM_FF", "0")
    for p in got:
        assert not set(p.env) & set(switches.FIELD_OF) and p.env[switches.SWITCHES_ENV]
        child = process_mod.child_environ(p, root)
        assert not set(child) & set(switches.FIELD_OF)
        assert child[switches.SWITCHES_ENV] == p.env[switches.SWITCHES_ENV]


def test_switch_view_catches_a_changed_value(tmp_path, monkeypatch):
    """Negative check of the comparison the golden cases use: a different value, an extra switch, a dropped one."""
    case = next(c for c in CASES if c.stem == "train_bi_sharpa_wave")
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    log, text = next(iter(dumps.items()))
    for extra, want in (
        ({"DEXENGINE_OBJ_SPAWN_HOLD": "9"}, "switch values"),
        ({"camera_res": "480"}, "given"),
        ({}, None),
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            (p,) = launch.resolve(launch.from_env(run, {**sub, **extra}, root=root))
        bad = switch_view(p, text, lambda v: v.replace(str(tmp_path / "main"), "<MAIN>"))
        assert (want is None and bad == []) or any(want in b for b in bad), (extra, bad)
    dropped = text.replace("ENV::DEXENGINE_OBJ_SPAWN_HOLD=10\n", "")
    assert any("given" in b for b in switch_view(p, dropped, lambda v: v.replace(str(tmp_path / "main"), "<MAIN>")))


READ_BACK = (
    "import json, sys; from dexengine.rl.tasks import env_flags, switches; switches.require_file(); "
    "json.dump({n: env_flags.get(n) for n in switches.FIELD_OF}, sys.stdout)"
)


@pytest.mark.parametrize("case", CASES, ids=[c.stem for c in CASES])
def test_trainer_reads_the_recorded_values(case, tmp_path, monkeypatch):
    """Per process: write its switches file, load it in a clean trainer-like process and read every switch through
    env_flags. A switch the recorded dump exported reads as that value; every other one reads as its unset text."""
    import json
    import subprocess
    import sys

    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        got = {p.log.name: p for p in launch.resolve(launch.from_env(run, sub, root=root))}
    for log, text in dumps.items():
        p = got[log]
        p.write_switches()
        r = subprocess.run(
            [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); " + READ_BACK],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", switches.SWITCHES_ENV: p.env[switches.SWITCHES_ENV]},
        )
        assert r.returncode == 0, r.stderr[-800:]
        read = json.loads(r.stdout)
        old = {k: v for k, v in parse_dump(text)[1].items() if k in switches.FIELD_OF}
        for name, f in switches.FIELD_OF.items():
            if name in old:
                assert switches.same(f, read[name], old[name]), (log, name, read[name], old[name])
            else:
                assert read[name] == switches.unset_text(f), (log, name, read[name])


JUDGE_ONLY = (
    "FROZEN_REPLAY_CKPT",
    "FROZEN_REPLAY_NE",
    "FROZEN_REPLAY_EVALS_PER_CLIP",
    "FROZEN_REPLAY_ITER",
    "FROZEN_RENDER_OUT",
    "TRAJ_RENDER_OUT",
    "DEXENGINE_TRAJ_RECORD",
    "DEXENGINE_TRAJ_RENDER",
)


def _training_record(run, sub, root, **train_extra):
    """Resolve the training launch of a judge case's inputs and write its run's switches record."""
    from dexengine.rl.launch import records

    train_env = {**{k: v for k, v in sub.items() if k not in JUDGE_ONLY}, **train_extra}
    with contextlib.redirect_stdout(io.StringIO()):
        (t,) = launch.resolve(launch.from_env(run, train_env, root=root))
    return t, records.write_switches_record(root, run, t)


@pytest.mark.parametrize("case", [c for c in CASES if c.stem.startswith("judge_")], ids=lambda c: c.stem)
def test_judge_over_the_training_record_matches_the_recorded_dump(case, tmp_path, monkeypatch):
    """A judge resolved over its training run's switches record (what the sidecar passes) is the recorded judge: the
    record holds the training's values and the judge adds only its context switches (render look)."""
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    _t, rec = _training_record(run, sub, root)
    with contextlib.redirect_stdout(io.StringIO()):
        got = {p.log.name: p for p in launch.resolve(launch.from_env(run, sub, root=root, switches_from=str(rec)))}

    def norm(v):
        return TMPVIEW.sub("<TMPVIEW>", v.replace(str(tmp_path / "main"), "<MAIN>").replace(str(tmp_path / "s"), "<S>"))

    assert sorted(got) == sorted(dumps)
    for log, text in dumps.items():
        assert switch_view(got[log], text, norm) == [], log


def test_set_at_training_reaches_the_judge_process_file(tmp_path, monkeypatch):
    """`--set arm_ff=0 --set obj_spawn_hold=3` on the training launch -> the run's record -> the judge's file. Without
    the record the judge would run on the recipe values; an explicit judge input that differs from the record fails."""
    case = next(c for c in CASES if c.stem == "judge_bi_sharpa_wave")
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    t, rec = _training_record(run, sub, root, arm_ff="0", obj_spawn_hold="3")
    assert (t.switches.arm_ff, t.switches.obj_spawn_hold) == (False, 3)
    with contextlib.redirect_stdout(io.StringIO()):
        (j,) = launch.resolve(launch.from_env(run, sub, root=root, switches_from=str(rec)))
        (plain,) = launch.resolve(launch.from_env(run, sub, root=root))
    assert process_mod.start(j, root, dry_run=True) == 0
    sw, given = switches.read(j.env[switches.SWITCHES_ENV])
    assert (sw.arm_ff, sw.obj_spawn_hold) == (False, 3) and {"arm_ff", "obj_spawn_hold"} <= given
    assert sw == t.switches  # nothing but context switches may differ
    assert (plain.switches.arm_ff, plain.switches.obj_spawn_hold) == (True, 10)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(recipe.RecipeError, match="trained with"):
        launch.resolve(launch.from_env(run, {**sub, "DEXENGINE_ARM_FF": "1"}, root=root, switches_from=str(rec)))
    with pytest.raises(recipe.RecipeError, match="judge or render input"):
        launch.from_env(run, {k: v for k, v in sub.items() if k not in JUDGE_ONLY}, root=root, switches_from=str(rec))
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(recipe.RecipeError, match="not found"):
        launch.resolve(launch.from_env(run, sub, root=root, switches_from=str(tmp_path / "none.json")))
