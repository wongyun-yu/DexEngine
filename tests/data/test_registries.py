"""Reader registry, rule registry API and the stage-1 step table (declarations that other code relies on)."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def rr(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    from dexengine.data import rules

    return rules


def test_rules_is_the_registry_module(rr):
    import dexengine.core.datasets as impl

    assert rr is impl  # one module object: shared state, lazy attributes
    assert rr.ACTIVE_HAND == "sharpa_wave"
    with pytest.raises(KeyError):
        rr.datasets_scope("no_such_dataset")


def test_ledger_key_round_trip(rr):
    for ds in ("hot3d", "arctic"):
        variant = rr.DATASET_PROFILE[ds].variant
        key = rr.ledger_key(ds, "bimanual", variant)
        assert rr.ledger_axis_parts(key)[:2] == (ds, "bimanual")
        assert rr.ledger_axis_variant(key) == variant


def test_support_thresholds_resolve_from_bands(rr):
    from dexengine.data.support import bands

    refs = [(r, label, ref) for r in rr.resolve() for label, ref in r.thresholds if "support/bands.py" in ref]
    assert refs
    for r, label, ref in refs:
        assert r.resolved.get(label) == getattr(bands, ref.split(":")[1]), ref


def test_reader_registry():
    from dexengine.data import readers as R

    assert R.spec("hot3d").driver == "stage" and R.spec("hocap").driver == "hoi"
    with pytest.raises(SystemExit):
        R.spec("show3d")  # dropped dataset: no row, no silent default
    with pytest.raises(SystemExit):
        R.stage_spec("hrdexdb")  # library reader: not the per-recording driver
    import importlib

    from dexengine.core import datasets as registry

    assert set(R.READERS) == set(registry.ALL) - {"show3d"}  # one reader per dataset of the registry
    for name, s in R.READERS.items():
        assert s.driver in R.DRIVERS, name
        assert (REPO / (s.module.replace(".", "/") + ".py")).is_file(), name
        if s.driver == "stage":
            assert "{task}" in s.marker and s.tasks and s.derive
        if s.driver == "mano":
            m = importlib.import_module(s.module)
            assert callable(m.enumerate_clips) and callable(m.convert) and callable(m.main), name


def test_stage_table(rr):
    from dexengine.data.stage1 import table as T

    ids = [T.step_id(g) for g in T.STAGES]
    assert len(ids) == len(set(ids))
    rule_ids = {r.id for r in rr.RULES}
    for name, pat, _slot, cmd in T.STAGES:
        if isinstance(cmd, T.Run):
            assert (REPO / cmd.argv[0]).is_file(), name
            assert cmd.rule is None or cmd.rule in rule_ids, name
            T.every_tree(pat, cmd)  # an every-tree cell must be a marker cell
    assert T.emb_sides("bimanual") == ("right", "left") and T.emb_sides("left") == ("left",)
    with pytest.raises(SystemExit):
        T.every_tree(T._INPUT_FP + "x", T.Run("", ("a",), every=True))


def _named_entries(text: str) -> tuple[list[str], list[str]]:
    """(repository files, `-m` modules) a printed command names."""
    import re

    files = re.findall(r"(?<![\w/])((?:scripts|dexengine)/[\w/]+\.(?:py|sh))", text)
    mods = re.findall(r"-m (dexengine(?:\.\w+)+)", text)
    return files, mods


def test_printed_stage_commands_exist(rr):
    """Every file and module the status table, the path tokens and the registration chain print exists — a person
    pastes these lines (the guidance once named scripts that no longer existed)."""
    import importlib.util

    from dexengine.data.stage1 import register as RG
    from dexengine.data.stage1 import table as T

    texts = [T.cmd_of(g) for g in T.STAGES] + list(T._DOC_TOKEN.values()) + [T._DOC_SCENE]
    texts += [" ".join(cmd) for _n, _i, cmd, _w in RG.REGISTER_STEPS]
    seen = 0
    for text in texts:
        files, mods = _named_entries(text)
        for f in files:
            assert (REPO / f).is_file(), f"{f} (printed: {text[:80]})"
        for m in mods:
            assert importlib.util.find_spec(m) is not None, f"{m} (printed: {text[:80]})"
        seen += len(files) + len(mods)
    assert seen >= len(T.STAGES)


def test_clip_scope_keys_are_step_scripts(tmp_path):
    """clip_scope rewrites the clip input of the step scripts it names; a key that is not a step script's basename
    (an old name) silently runs that step over the whole dataset."""
    from dexengine.data.stage1 import clip_scope as CS
    from dexengine.data.stage1 import table as T

    steps = {Path(c.argv[0]).name: c.argv[0] for _n, _p, _s, c in T.STAGES if isinstance(c, T.Run)}
    for script, (flag, form, unit) in CS.CLIP_ARGS.items():
        assert script in steps, f"{script} is not a step script of the table"
        assert f'"{flag}"' in (REPO / steps[script]).read_text(encoding="utf-8"), f"{steps[script]} has no {flag}"
        assert form in ("comma", "file") and unit in CS.UNITS
    (tmp_path / "family.txt").write_text("clip a\nclip b\n")
    argv = ["py", str(REPO / steps["taxonomy.py"]), "--dataset", "ds", "--write"]
    assert CS.scoped_argv(argv, tmp_path) == argv + ["--clips", "clip a,clip b"]


def test_rules_cli_prog():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "dexengine.data.rules", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
        timeout=120,
    )
    assert out.returncode == 0 and out.stdout.startswith("usage: python -m dexengine.data.rules"), out.stdout[:200]
