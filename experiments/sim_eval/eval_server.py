"""Policy servers of the closed-loop evaluation: start/stop one DP or GR00T server per GPU and write the policy JSON
the session's policy adapter reads.

DP: `experiments.policies.dp.server` in the DP python (lerobot) with `--out <server dir>`; it writes `READY` and
`modality.json` there after loading and exits when `STOP` appears. GR00T: `experiments.policies.gr00t.server start`
prints `[E5G] READY … modality=<path>`; `stop --name` ends it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: Actions executed per policy call: DP 8, GR00T 16 (the fine-tuning recipe's execution horizons).
EXEC_HORIZON = {"dp": 8, "gr00t": 16}
POLICY_TIMEOUT_MS = 60000
CAMERAS = {"camera_ego_left": "egoL", "camera_ego_right": "egoR"}


@dataclass
class Server:
    """A started policy server: its kind, port, modality file and how to stop it."""

    kind: str
    port: int
    modality: str
    out: Path
    proc: subprocess.Popen | None = None
    name: str = ""
    extra: dict = field(default_factory=dict)

    def alive(self) -> bool:
        """False once a DP server exited or was told to stop (GR00T: always True — the session times out instead)."""
        if self.kind != "dp":
            return True
        return self.proc is not None and self.proc.poll() is None and not (self.out / "STOP").exists()

    def stop(self) -> str:
        """Stop the server and return a one-line note."""
        if self.kind == "dp":
            (self.out / "STOP").touch()
            return f"STOP {self.out}"
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.policies.gr00t.server",
                "stop",
                "--root",
                self.extra["root"],
                "--name",
                self.name,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return (p.stdout.strip().splitlines() or [""])[-1]


def start_dp(
    ckpt: str,
    dataset_meta: str,
    port: int,
    gpu: int,
    out: Path,
    dp_python: str,
    server_args: list[str],
    wait_s: int = 180,
) -> Server:
    """Start the DP server on one GPU and wait for READY (dies on a traceback or timeout)."""
    so = out / "server"
    if so.exists():
        raise SystemExit(f"[OVFCL] server folder {so} already exists (same tag ran before) — use a new tag")
    so.mkdir(parents=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HUB_OFFLINE="1",
        HF_HOME=str(out / "hf_home"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(ROOT),
    )
    cmd = [
        "timeout",
        "14400",
        dp_python,
        "-m",
        "experiments.policies.dp.server",
        "--ckpt",
        ckpt,
        "--root",
        dataset_meta,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--out",
        str(so),
        *server_args,
    ]
    with open(so / "server.log", "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    for _ in range(wait_s // 2):
        if (so / "READY").exists():
            break
        txt = (so / "server.log").read_text(errors="replace")
        if "Traceback" in txt or "SystemExit" in txt or proc.poll() is not None:
            break
        time.sleep(2)
    if not (so / "READY").exists():
        (so / "STOP").touch()
        tail = " ".join((so / "server.log").read_text(errors="replace").splitlines()[-2:])
        raise RuntimeError(f"DP server did not become ready: {tail[:250]}")
    return Server("dp", port, str(so / "modality.json"), so, proc=proc)


def start_gr00t(ckpt: str, dataset_meta: str, port: int, gpu: int, out: Path, name: str, root: str) -> Server:
    """Start the GR00T server (with the processor ↔ training-set stats check) and parse its READY line."""
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.policies.gr00t.server",
            "start",
            "--root",
            root,
            "--ckpt",
            ckpt,
            "--port",
            str(port),
            "--gpu",
            str(gpu),
            "--dataset",
            dataset_meta,
            "--name",
            name,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    last = ((p.stdout + p.stderr).strip().splitlines() or [""])[-1]
    if not last.startswith("[E5G] READY"):
        raise RuntimeError(f"GR00T server: {last[:300]}")
    m = re.search(r"modality=(\S+)", last)
    if not m:
        raise RuntimeError(f"GR00T server READY line without modality=: {last[:300]}")
    return Server("gr00t", port, m.group(1), out, name=name, extra={"root": root})


def write_policy_json(
    path: Path, tag: str, srv: Server, dataset_meta: str, text_map: str | None, pylib: list[str]
) -> None:
    """The policy JSON of one batch (`{"policies": {tag: {...}}}`; session mode `policy-<tag>`)."""
    p = {
        "kind": srv.kind,
        "host": "127.0.0.1",
        "port": srv.port,
        "modality": srv.modality,
        "dataset_meta": dataset_meta,
        "video": dict(CAMERAS),
        "pylib": list(pylib),
        "exec_horizon": EXEC_HORIZON[srv.kind],
        "timeout_ms": POLICY_TIMEOUT_MS,
    }
    if text_map:
        p["instructions"] = text_map
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"policies": {tag: p}}, indent=1))
