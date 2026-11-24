"""Training orchestration of the launcher against the old shell chain, with a fake Isaac Lab (fake_isaaclab.sh).

`orchestration_cases/<case>/` holds what an older shell chain (scripts/stage3_rl.sh -> shell trainers,
since replaced) did for one scenario, recorded in a sandbox with the same fake trainer: its exit code, the
trainer calls (run name, resume checkpoint and iteration), the summary-table row (train_sec dropped), the run-folder
files and the log-folder files (minus the old per-run script snapshot and wrap.out files, which the launcher no longer
writes; the switches file next to each trainer log and the run's switches record are new and checked apart). Scenarios:
a clean run, one CUDA out-of-memory death (automatic resume), two planned memory restarts, a trainer failure, MI below
the eval interval, and the one-hand path.

Two root shapes: `canon` (the data root is the code tree — the shape the cases were recorded in) and `apart` (a data
root apart from the code tree, the release shape): there every record lands under the data root (the run tree,
paths.run_root_of) and nothing under the code tree the trainer runs from.
"""

from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path

import pytest

from dexengine.core.layout.runs import RUN_CKPT_ROOT
from dexengine.core.repo import REPO_ROOT
from dexengine.rl import launch
from dexengine.rl.launch import preflight

HERE = Path(__file__).parent
CASES = sorted(p.name for p in (HERE / "orchestration_cases").iterdir() if p.is_dir())


@pytest.fixture
def fake_isaac(tmp_path):
    d = tmp_path / "isaac"
    (d / "env_isaaclab/bin").mkdir(parents=True)
    shutil.copy(HERE / "fake_isaaclab.sh", d / "isaaclab.sh")
    (d / "isaaclab.sh").chmod(0o755)
    (d / "env_isaaclab/bin/python").symlink_to(shutil.which("python3"))
    return d


def normalized(root: Path, run: str, path: str) -> dict[str, str]:
    ref = root / "overfit/logs/refbase"
    out = {"calls.txt": (root / "calls.txt").read_text().replace(run, "RUN")}
    rows = []
    summ = ref / ("bi_summary.tsv" if path == "bi" else "uni_lag_summary.tsv")
    for ln in summ.read_text().splitlines():
        f = ln.split("\t")
        del f[5 if path == "bi" else 6]  # train_sec
        f = [x.replace(run, "RUN") for x in f]
        if f[-1] != "JUDGE=sidecar":
            f[-1] = "<host>"
        else:
            f[-2] = "<host>"
        rows.append("\t".join(f))
    out["summary.tsv"] = "\n".join(rows) + "\n"
    files = sorted(str(x.relative_to(root)).replace(run, "RUN") for x in (root / "logs").rglob("*") if x.is_file())
    out["run_files.txt"] = "".join(f"{p}\n" for p in files if not p.endswith("/switches.json"))
    out["records"] = [p for p in files if p.endswith("/switches.json")]
    names = sorted(x.name.replace(run, "RUN") for x in ref.iterdir())
    out["refbase_files.txt"] = "".join(f"{p}\n" for p in names if not p.endswith(".switches.json"))
    out["switch_stems"] = sorted(p.removesuffix(".switches.json") for p in names if p.endswith(".switches.json"))
    out["log_stems"] = sorted(p.removesuffix(".log") for p in names if p.endswith(".log"))
    return out


@pytest.mark.parametrize("shape", ["canon", "apart"])
@pytest.mark.parametrize("case", CASES)
def test_training_orchestration_matches_old_chain(case, shape, tmp_path, fake_isaac, monkeypatch):
    scen, side = case.rsplit("_", 1)
    want = {f.name: f.read_text() for f in (HERE / "orchestration_cases" / case).iterdir()}
    root = tmp_path / "tree"
    root.mkdir()
    for d in ("dexengine", "scripts", "configs", "assets"):
        (root / d).symlink_to(REPO_ROOT / d)
    seq = tmp_path / ("bi2" if side == "both" else "right2")
    seq.mkdir()
    run = f"ORCH_{case}"
    data = root if shape == "canon" else tmp_path / "data"
    data.mkdir(exist_ok=True)
    env = {
        "GPU": "0",
        "SIDE": side,
        "SEQ_DIR": str(seq),
        "DO_VIDEO": "0",
        "BI_NEED_MIB": "0",
        "WANDB_GROUP": "dry",
        "DEXENGINE_ARM": "rby1",
        "DEXENGINE_HAND": "sharpa_wave",
        "DEXENGINE_DATASET": "hot3d",
        "DEXENGINE_EMBODIMENT": "bimanual" if side == "both" else side,
        "DEXENGINE_MAIN_ROOT": str(data),
    }
    if scen == "noeval":
        env["MI"] = "100"
        monkeypatch.setenv("FAKE_EVAL_INTERVAL", "200")
    monkeypatch.setenv("ISAACLAB_DIR", str(fake_isaac))
    monkeypatch.setenv("FAKE_CALLS", str(data / "calls.txt"))
    monkeypatch.setenv("FAKE_SCENARIO", scen)
    monkeypatch.setattr(preflight, "check_bimanual", lambda *a, **k: "")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = launch.run(launch.from_env(run, env, root=root))
    assert str(rc) == want["rc"].strip(), out.getvalue()[-1500:]
    got = normalized(data, run, "bi" if side == "both" else "uni")
    if shape == "apart":  # the code tree (the trainer's cwd) holds none of the run's records
        assert sorted(x.name for x in root.iterdir()) == ["assets", "configs", "dexengine", "scripts"]
    for name in ("calls.txt", "summary.tsv", "run_files.txt", "refbase_files.txt"):
        assert got[name] == want[name], name
    # one switches file per trainer process, next to its log; the run's record = its first training process's file
    assert got["switch_stems"] and set(got["switch_stems"]) <= set(got["log_stems"])
    assert got["records"] == [f"{RUN_CKPT_ROOT}/RUN/switches.json"]
    from dexengine.rl.tasks import switches

    rec = switches.read(data / RUN_CKPT_ROOT / run / "switches.json")
    assert rec == switches.read(data / "overfit/logs/refbase" / f"{run}_train.switches.json")
