"""record.driver --cpu-tail on: a worker's GPU part ends at the bundle's render; the store call (--gpu none), check,
merge and cleanup run in its tail thread while the next bundle renders. A fake bundle (render = files + a pause,
store = a dataset file) stands in for record.bundle: the outputs and results equal the off run's, renders never
overlap on the GPU, a store overlaps the next render, and a failure streak still stops the worker."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from dexengine.record import bundle as C
from dexengine.record import clean as CLN
from dexengine.record import driver as DRV
from dexengine.record import driver_bundle as DRB
from dexengine.record import sources as OWN

FAKE = r"""
import os, sys, time
from pathlib import Path
argv = sys.argv[3:]                                    # after "-m dexengine.record.bundle"
opt = {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1) if argv[i].startswith("--")}
out, gpu, until = Path(opt["--out"]), opt["--gpu"], opt.get("--until", "store")
ev = Path(os.environ["FAKE_EVENTS"])
def mark(what):
    with open(ev, "a") as f:
        f.write(f"{out.name} {what} {time.time():.4f} {gpu}\n")
print("SRC_LINE x", flush=True)
print("OWNER_LINE ok", flush=True)
clips = opt["--clips"].split(",")
render = out / "render"
if gpu == "none":                                      # store call: caches only
    if until != "store" or not all((render / f"{c}.mp4").is_file() for c in clips):
        print("cache miss in a CPU-only call", flush=True); sys.exit(9)
else:
    mark("render-start"); time.sleep(0.5)
    render.mkdir(parents=True, exist_ok=True)
    for c in clips:
        (render / f"{c}.mp4").write_text(c)
    mark("render-end")
    if until == "render":
        print("RENDER_DONE episodes", flush=True); sys.exit(0)
mark("store-start"); time.sleep(0.3)
if os.environ.get("FAKE_STORE_FAIL") == "1":
    print("✗ store failed: disk full", flush=True); sys.exit(5)
ds = out / "rounds" / "aug" / "dataset"
ds.mkdir(parents=True, exist_ok=True)
(ds / "stored.txt").write_text(",".join(clips))
mark("store-end")
print("DONE_LINE episodes", flush=True)
"""


def _setup(tmp: Path, monkeypatch, tail: str, n: int = 3):
    fake = tmp / "fakepy"
    fake.write_text(
        f"#!{sys.executable}\n"
        + FAKE.replace("OWNER_LINE", OWN.OWNER_LINE.strip())
        .replace("SRC_LINE", f"{OWN.MARK} src=")
        .replace("RENDER_DONE", C.RENDER_DONE_LINE)
        .replace("DONE_LINE", C.DONE_LINE)
    )
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_EVENTS", str(tmp / "events.txt"))
    monkeypatch.setattr(DRV, "tree_state", lambda tree: "t0")
    monkeypatch.setattr(DRV, "mem_available_gb", lambda: 1000.0)
    monkeypatch.setattr(DRB, "piece_dir", lambda hand, combo, main, b, rnd, part: tmp / "work" / b["bundle"])
    monkeypatch.setattr(DRB.PC, "round_dir", lambda out, rnd, part="": out / "rounds" / rnd)
    monkeypatch.setattr(DRB.SPM, "row0_of", lambda a, clips: [])
    monkeypatch.setattr(DRB.P, "layer_path", lambda *a, **k: tmp / "final")

    def verify(out, clips, ds, run_log=None):
        ok = (ds / "stored.txt").is_file()
        return {"ok": ok, "problems": [] if ok else ["nothing stored"], "target": len(clips), "episodes": len(clips)}

    monkeypatch.setattr(DRB, "verify", verify)

    def clean(out, v, rd, clips):
        gone = sorted(p.name for p in (out / "render").glob("*.mp4"))
        for p in (out / "render").glob("*.mp4"):
            p.unlink()
        return {"done": True, "files": gone}

    monkeypatch.setattr(CLN, "clean", clean)
    queue = [
        {"dataset": "grab", "campaign": "c", "bundle": f"b{i}", "clips": [f"clip{i}a", f"clip{i}b"]} for i in range(n)
    ]
    a = types.SimpleNamespace(
        run_dir=tmp / "run",
        queue=queue,
        tree=tmp,
        hand="h",
        combo="rby1_h",
        main=tmp,
        round="aug",
        part="",
        bundle_timeout_base=60,
        bundle_timeout_per_clip=10,
        py=str(fake),
        owner=tmp / "own.tsv",
        owner_sha="0",
        ledger=tmp / "ledger.tsv",
        min_rate=0.05,
        queue_root=str(tmp),
        merge_fn=None,
        spec_id="",
        cpu_tail=tail,
        retry_failed=False,
        only=set(),
        max_bundles=0,
        deadline_ts=time.time() + 600,
        exclusive=set(),
        min_mem_gb=1,
    )
    S = DRV.Pilot(a)
    S.workers["7"] = {"state": "starting", "bundle": None, "note": "", "gpu": "7"}
    return S


def _events(tmp: Path) -> list[tuple[str, str, float, str]]:
    return [(b, w, float(t), g) for b, w, t, g in (ln.split() for ln in (tmp / "events.txt").read_text().splitlines())]


def _outputs(tmp: Path) -> dict:
    files = sorted(str(p.relative_to(tmp)) for p in (tmp / "work").rglob("*") if p.is_file())
    res = {}
    for f in sorted((tmp / "run" / "bundles").glob("*/result.json")):
        r = json.loads(f.read_text().replace(str(tmp), "<T>"))  # the two runs' roots differ
        res[f.parent.name] = {k: v for k, v in r.items() if k not in ("sec", "started", "ended", "cpu_tail")}
    states = {f.parent.name: f.read_text().split()[0] for f in (tmp / "run" / "bundles").glob("*/state")}
    return {"files": files, "results": res, "states": states}


def test_tail_on_same_outputs_and_overlap(tmp_path, monkeypatch):
    off, on = tmp_path / "off", tmp_path / "on"
    for d, mode in ((off, "off"), (on, "on")):
        d.mkdir()
        S = _setup(d, monkeypatch, mode)
        S.worker("7", "7")
        assert S.workers["7"]["state"] == "idle", S.workers["7"]
    o_off, o_on = _outputs(off), _outputs(on)
    assert o_off == o_on and set(o_on["states"].values()) == {"DONE"} and len(o_on["results"]) == 3
    for d in (off, on):  # renders never overlap on the GPU; GPU calls see GPU 7
        ev = _events(d)
        r = [(t, w) for _b, w, t, _g in ev if w.startswith("render")]
        assert [w for _t, w in sorted(r)] == ["render-start", "render-end"] * 3
        assert all(g == "7" for _b, w, _t, g in ev if w.startswith("render"))
    ev_on = {(b, w): (t, g) for b, w, t, g in _events(on)}
    assert all(ev_on[(f"b{i}", "store-start")][1] == "none" for i in range(3))  # store calls: no GPU
    assert ev_on[("b1", "render-start")][0] < ev_on[("b0", "store-end")][0]  # b0 stores while b1 renders
    ev_off = {(b, w): t for b, w, t, _g in _events(off)}
    assert ev_off[("b0", "store-end")] < ev_off[("b1", "render-start")]  # off: strictly one after the other
    r1 = json.loads((on / "run" / "bundles").glob("*b1/result.json").__next__().read_text())
    assert set(r1["cpu_tail"]) == {"gpu_s", "cpu_s"} and r1["gpu"] == "7"
    log = next((on / "run" / "bundles").glob("*b1/collect.log")).read_text()
    assert "--until render" in log and "--gpu none" in log and log.count("# ") == 2


@pytest.mark.parametrize("mode, n_failed", [("off", 3), ("on", 4)])
def test_tail_failure_streak_stops_the_worker(tmp_path, monkeypatch, mode, n_failed):
    """Off stops after the third failure; on learns it at the third tail's end, after the fourth bundle rendered —
    that bundle still gets its store call (never left RUNNING)."""
    monkeypatch.setenv("FAKE_STORE_FAIL", "1")
    S = _setup(tmp_path, monkeypatch, mode, n=5)
    S.worker("7", "7")
    assert S.workers["7"]["state"] == "stopped" and "same failure" in S.workers["7"]["note"]
    states = _outputs(tmp_path)["states"]
    assert sorted(states.values()) == ["FAILED"] * n_failed
    assert all(
        "store failed" in json.loads(f.read_text())["reason"]
        for f in (tmp_path / "run" / "bundles").glob("*/result.json")
    )


@pytest.mark.parametrize("until, ok", [("render", False), ("record", False)])
def test_cpu_only_call_stores_only(tmp_path, until, ok):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexengine.record.bundle",
            "--hand",
            "h",
            "--gpu",
            C.CPU_ONLY,
            "--out",
            str(tmp_path),
            "--round",
            "orig",
            "--until",
            until,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode != 0 and f"--gpu {C.CPU_ONLY} stores from the caches only" in r.stdout + r.stderr


def test_cpu_only_call_hides_the_gpu(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexengine.record.bundle",
            "--hand",
            "h",
            "--gpu",
            C.CPU_ONLY,
            "--out",
            str(tmp_path),
            "--round",
            "orig",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert "CPU-only call: no GPU visible" in r.stdout  # then it dies on the missing queue root (no default)
