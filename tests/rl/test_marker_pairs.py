"""Printed markers and the judge sidecar patterns that parse them stay in step: each emitter's real output must match
the pattern the sidecar greps for (read from scripts/rl/judge_sidecar.sh itself)."""

from __future__ import annotations

import contextlib
import io
import re
import types

import pytest

from dexengine.core.repo import REPO_ROOT
from dexengine.rl.judge import terms
from dexengine.rl.launch import hand_recipe, run_dir

SIDECAR = (REPO_ROOT / "scripts/rl/judge_sidecar.sh").read_text(encoding="utf-8")


def sidecar_pattern(tag: str, contains: str = "") -> re.Pattern:
    """The ERE the sidecar greps the judge log with for `tag` (the one `grep -aqE "^\\[TAG\\] ..."` line holding
    `contains`)."""
    pats = [p for p in re.findall(r'grep -aqE "(\^\\\[' + re.escape(tag) + r'\\\][^"]*)"', SIDECAR) if contains in p]
    assert len(pats) == 1, (tag, pats)
    return re.compile(pats[0].replace("$JCT_STD", str(terms.CONTACT_TERM)))


def test_contact_term_marker_matches_the_sidecar():
    value, src = terms.contact_term({})
    assert sidecar_pattern("JUDGE_TERM").search(terms.marker(value, src))
    assert not sidecar_pattern("JUDGE_TERM").search(terms.marker(1, terms.GIVEN))


def judge_launch():
    pytest.importorskip("torch")  # dexengine.rl.judge.launch imports the PARK module (torch)
    from dexengine.rl.judge import launch

    return launch


def test_fresh_launch_marker_matches_the_sidecar():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert judge_launch().check_launch("fresh", {}) == "fresh"
    line = [ln for ln in out.getvalue().splitlines() if ln.startswith("[EVAL_LAUNCH]")]
    assert len(line) == 1 and sidecar_pattern("EVAL_LAUNCH").search(line[0]), line


def test_intrain_eval_first_word_is_the_recorded_word():
    for ctx, word in (("none", "off"), ("intrain", "on")):
        out = io.StringIO()
        env = {"DEXENGINE_EVAL_EXIT_RESET": "1"} if ctx == "intrain" else {}
        with contextlib.redirect_stdout(out):
            judge_launch().check_launch(ctx, env)
        ln = [x for x in out.getvalue().splitlines() if x.startswith("[INTRAIN_EVAL]")]
        assert len(ln) == 1
        assert [t for t in ln[0].split("[INTRAIN_EVAL]", 1)[1].split() if t != "⚠"][0] == word


@pytest.mark.parametrize("ledger", [False, True])
def test_checkpoint_ledger_marker_matches_the_sidecar(tmp_path, ledger):
    hand = sorted(hand_recipe.HAND_RL_RECIPES)[0]
    args = types.SimpleNamespace(hand=hand, finger_penalty_span=None)
    recipe = hand_recipe.rl_recipe_for(hand)
    for field, kind, _name, _default in hand_recipe.RECIPE_CHANNELS:
        if kind == "flag":
            setattr(args, field, getattr(recipe, field))
    env = {"DEXENGINE_HAND_RECIPE": "1"}
    (tmp_path / "checkpoints").mkdir()
    if ledger:
        (tmp_path / run_dir.LEDGER_NAME).write_text(run_dir.ledger_line(args, env) + "\n")
    line = run_dir.check_checkpoint(str(tmp_path / "checkpoints/agent_1.pt"), args, env, ctx="train", resume=False)
    assert sidecar_pattern("HAND_RECIPE", "ledger").search(line), line


def test_dirty_tree_launch_raises_the_type_the_scheduler_stops_on(tmp_path):
    """cell_state.launch_record refuses a dirty tree with UncommittedTreeError (a ValueError); the scheduler
    (dexengine.cluster.scheduler) stops launching on that type."""
    from dexengine.rl import cell_state

    with pytest.raises(cell_state.UncommittedTreeError) as e:
        cell_state.launch_record(tmp_path, {"hand": "h"}, {}, "RUN", {"dirty": 3, "tree": "/t"})
    assert isinstance(e.value, ValueError) and "RUN" in str(e.value)


# --- judge-log lines the sidecar parses with grep/sed: the writer's f-string (filled) through the sidecar's own text


def _fstring(rel: str, needle: str, fill: dict) -> str:
    """The one f-string of `rel` containing `needle`, placeholders replaced by fill[source] (source = ast.unparse)."""
    import ast

    hits = []
    for n in ast.walk(ast.parse((REPO_ROOT / rel).read_text())):
        if isinstance(n, ast.JoinedStr):
            text = "".join(
                v.value if isinstance(v, ast.Constant) else "\0" + ast.unparse(v.value) + "\0" for v in n.values
            )
            if needle in text:
                hits.append(re.sub(r"\0([^\0]*)\0", lambda m: fill[m.group(1)], text))
    assert len(hits) == 1, hits
    return hits[0]


def _const(rel: str, name: str) -> str:
    import ast

    for n in ast.parse((REPO_ROOT / rel).read_text()).body:
        if isinstance(n, ast.Assign) and [getattr(t, "id", "") for t in n.targets] == [name]:
            return ast.literal_eval(n.value)
    raise AssertionError(f"{name} not in {rel}")


def _sh(cmd: list[str], stdin: str) -> str:
    import subprocess

    return subprocess.run(cmd, input=stdin, capture_output=True, text=True).stdout


def _sh_func(name: str, *args: str) -> str:
    """Run the sidecar's own shell function `name` with args."""
    import subprocess

    m = re.search(rf"^{name}\(\) \{{.*?^\}}", SIDECAR, re.S | re.M)
    assert m, f"{name} not in judge_sidecar.sh"
    return subprocess.run(
        ["bash", "-c", m.group(0) + f'\n{name} "$@"', "_", *args], capture_output=True, text=True
    ).stdout


@pytest.mark.parametrize("needle", ["[C41][static]", "[C41][balanced]"])
def test_c41_batch_counts_are_what_the_sidecar_reads(needle):
    rel = "dexengine/rl/judge/rollouts_run.py"
    batches = _const(rel, "C41_BATCHES")
    fill = {
        "C41_BATCHES.format(n=_n_real, w=warmup_batches)": batches.format(n=7, w=1),
        "C41_BATCHES.format(n=n_real, w=warmup_batches)": batches.format(n=7, w=1),
        "int(_fixed_counts.min())": "3",
        "int(_fixed_counts.max())": "4",
        "evals_per_clip": "64",
        "num_clips": "5",
        "n_envs": "1024",
    }
    line = _fstring(rel, needle, fill) + "\n"
    m = re.search(r'read -r nreal w < <\(LC_ALL=C grep -aoE "([^"]+)" "\$log".*?sed -E \'([^\']+)\'\)', SIDECAR, re.S)
    assert m, "judge_sidecar.sh no longer reads the [C41] batch counts"
    assert _sh(["sed", "-E", m.group(2)], _sh(["/usr/bin/grep", "-aoE", m.group(1)], line)).split() == ["7", "1"]


def test_park_excl_a_tally_is_what_the_sidecar_notes(tmp_path):
    rel = "dexengine/rl/judge/rollouts_report.py"
    line = _fstring(rel, " · still scored", {"EXCL_A_TALLY": _const(rel, "EXCL_A_TALLY"), "len(xa)": "2"})
    log = tmp_path / "j.log"
    log.write_text(line + " · c1 3/9 batches\n")
    assert _sh_func("park_excl_a_note", str(log)).strip() and "2" in _sh_func("park_excl_a_note", str(log))
    log.write_text(line.replace(" 2 ", " 0 ") + "\n")
    assert _sh_func("park_excl_a_note", str(log)) == ""


def test_c651_frozen_replay_lines_are_what_the_sidecar_counts():
    rel = "dexengine/rl/train/finger_penalty.py"
    fill = {
        "FINGER_PENALTY_PATH": _const(rel, "FINGER_PENALTY_PATH"),
        "path": "frozen_replay",
        "span": "derived_norm",
        "source": "canonical default",
        "side": "right",
        "dims[0]": "6",
        "dims[1]": "26",
        "p['n']": "20",
        "w_scale": "1",
        "dz_scale": "1",
        "users": "x",
    }
    line = _fstring(rel, " w_scale=", fill)
    m = re.search(r'nc=\$\(grep -acE "([^"]+)" "\$log"', SIDECAR)
    assert m, "judge_sidecar.sh no longer counts the [FINGER_PENALTY] frozen_replay lines"
    assert _sh(["/usr/bin/grep", "-acE", m.group(1)], line + "\n" + line.replace("right", "left") + "\n").strip() == "2"
    ev = _fstring(
        "dexengine/rl/eval/run.py",
        "eval span=",
        {"_FINGER_PENALTY_PATH": fill["FINGER_PENALTY_PATH"], "_fspan_e": "x", "_fspan_src_e": "y"},
    )
    assert _sh(["/usr/bin/grep", "-acE", m.group(1)], ev + "\n").strip() == "0"  # the eval path is not counted


def test_roster_clip_count_is_read_from_training_logs():
    rel = "dexengine/placement/train_ready_util.py"
    fill = {
        "hand": "h",
        "axis": "a",
        "src": "ledger:x",
        "len(uni)": "64",
        "len(have)": "64",
        "mode": "list:x",
        "len(train)": "17",
        "len(excluded)": "0",
        "': ' + ', '.join(excluded[:5]) if excluded else ''": "",
        "' …' if len(excluded) > 5 else ''": "",
        "sha": "0",
    }
    line = _fstring(rel, "[ROSTER] hand=", fill)
    m = re.search(r'LC_ALL=C grep -aoE "(\^\\\[ROSTER\\\][^"]+)"', SIDECAR)
    assert m, "judge_sidecar.sh no longer reads the [ROSTER] clip count"
    assert _sh(["/usr/bin/grep", "-aoE", m.group(1)], line + "\n").strip().endswith(" 17")


def test_collider_payload_reads_training_and_judge_logs_alike(tmp_path, monkeypatch):
    from dexengine.rl.tasks.scene import collision_presets as CP

    monkeypatch.setattr(CP, "extra_has_collider", lambda k: k == "moving")
    kinds = ["moving", "touched"]
    new = CP.collider_line(kinds, {"moving": 2, "touched": 0, "empty": 0}, {"moving": 2, "touched": 1, "empty": 3})
    train = CP.collider_line(kinds, {"moving": 1, "touched": 0, "empty": 0}, {"moving": 1, "touched": 2, "empty": 1})
    (tmp_path / "t.log").write_text(train + "\n")
    (tmp_path / "j.log").write_text(new + "\n")
    pt, pj = _sh_func("coll_payload", str(tmp_path / "t.log")), _sh_func("coll_payload", str(tmp_path / "j.log"))
    assert pj.strip() and pt == pj


def test_outside_cell_marker_is_what_record_augment_counts(monkeypatch):
    from dexengine.rl import checkpoints as CK

    monkeypatch.setattr(CK, "cell_of", lambda ckpt, hand: None)
    monkeypatch.setattr(CK, "_commit", lambda: "c0de")
    line = CK.check("/w/ck.pt", "/s", [], "h")
    src = (REPO_ROOT / "dexengine/record/augment/run.py").read_text()
    assert 'f"{_CK.TAG} {_CK.OUTSIDE_CELL}" in txt' in src and f"{CK.TAG} {CK.OUTSIDE_CELL}" in line
