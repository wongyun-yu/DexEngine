"""GR00T N1.7 policy server of the closed-loop evaluation: build its private environment, start, stop, status.

The server is Isaac-GR00T n1.7-release `gr00t/eval/run_gr00t_server.py`, unpatched, run from the Isaac-GR00T checkout
under `--root` with that checkout's uv venv (the same release, lock file and base weights as fine-tuning). Everything
(checkout, venv, Hugging Face cache, logs, run records, modality JSONs) lives under `--root`.

  python -m experiments.policies.gr00t.server setup  --root R [--isaac-python P --client-lib DIR] [--uv-offline]
  python -m experiments.policies.gr00t.server start  --root R --ckpt DIR --port N --gpu G (--dataset DS | --image-hw
  HxW)
                                                     [--name RUN] [--tag new_embodiment] [--host 127.0.0.1]
                                                     [--ready-timeout 900] [--threads 8]
  python -m experiments.policies.gr00t.server stop   --root R --name RUN
  python -m experiments.policies.gr00t.server status --root R

`start` checks the checkpoint (model type, tag), the training ↔ processor statistics (stats_check.py) and prints one
last line: `[E5G] READY <name> tcp://<host>:<port> pid=<pid> load_s=<s> modality=<json> log=<log>`, or
`[E5G] FAIL <why>` (exit 1) / `[E5G] TIMEOUT <why>` (exit 3). The modality JSON for the session client is built from
the checkpoint (keys, dims) and the dataset meta (image size) and checked against the live server.
`setup` prints `[SETUP] DONE` at the end, `[SETUP] FAIL <step>` (exit 1) on the first failing step.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GR00T_TAG = "n1.7-release"
GR00T_COMMIT = "23ace64f17aa5015259b8609d371eb61a357c776"
UV_VERSION = "0.8.22"  # = the fine-tuning environment's uv
BASE = ("nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495")
BACKBONE = ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561")
CLIENT_PINS = ("pyzmq==27.0.1", "msgpack==1.1.0")  # wire-compatible with the server's pyproject
ENTRY = "gr00t/eval/run_gr00t_server.py"
READY = "Server is ready and listening"
DIED = ("Traceback", "CUDA out of memory", "OutOfMemoryError", "403 Client Error", "Address already in use")


class Fail(SystemExit):
    def __init__(self, why: str, code: int = 1, tag: str = "[E5G] FAIL"):
        print(f"{tag} {why}", flush=True)
        super().__init__(code)


class Paths:
    """Layout under the server root."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.repo = self.root / "Isaac-GR00T"
        self.py = self.repo / ".venv/bin/python"
        self.logs, self.runs, self.modality = self.root / "logs", self.root / "runs", self.root / "modality"
        self.uv = self.root / "bin/uv"

    def env(self) -> dict:
        """Process environment: every cache under the root."""
        r = self.root
        e = dict(
            os.environ,
            HF_HOME=str(r / "hf"),
            XDG_CACHE_HOME=str(r / "cache"),
            UV_CACHE_DIR=str(r / "uv_cache"),
            UV_PYTHON_INSTALL_DIR=str(r / "uv_python"),
            UV_NO_MODIFY_PATH="1",
            PIP_CACHE_DIR=str(r / "cache/pip"),
            TORCH_HOME=str(r / "cache/torch"),
            TORCH_EXTENSIONS_DIR=str(r / "cache/torch_extensions"),
            TRITON_CACHE_DIR=str(r / "cache/triton"),
            CUDA_CACHE_PATH=str(r / "cache/nv"),
            MPLCONFIGDIR=str(r / "cache/mpl"),
            WANDB_DIR=str(r / "cache/wandb"),
            WANDB_MODE="disabled",
            PATH=f"{r / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        return e


def hf_token(env: dict) -> None:
    """The gated backbone is rebuilt from the hub at model construction, so the token is exported (read-only use of
    ~/.cache/huggingface/token). transformers 4.57.3 (pinned by the release lock) calls huggingface_hub.model_info for
    the
    Cosmos tokenizer whatever the local cache holds, so HF_HUB_OFFLINE must be off; weights still come from HF_HOME."""
    tok = Path.home() / ".cache/huggingface/token"
    if not env.get("HF_TOKEN") and tok.is_file():
        env["HF_TOKEN"] = env["HUGGING_FACE_HUB_TOKEN"] = tok.read_text().strip()
    if not env.get("HF_TOKEN"):
        raise Fail("no HF_TOKEN (and no ~/.cache/huggingface/token)")
    if env.get("HF_HUB_OFFLINE", "0") not in ("", "0"):
        raise Fail(f"HF_HUB_OFFLINE={env['HF_HUB_OFFLINE']}: the N1.7 processor build needs the hub API")
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)


def head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def alive(pidfile: Path) -> int:
    """Pid of a live run_gr00t_server.py recorded in `pidfile` (cmdline checked), else 0."""
    try:
        pid = int(pidfile.read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError):
        return 0
    return pid if ENTRY in cmd else 0


def check_ckpt(ckpt: Path, tag: str) -> None:
    cfg = json.loads((ckpt / "config.json").read_text())
    if cfg.get("model_type") != "Gr00tN1d7":
        raise Fail(
            f"checkpoint model_type {cfg.get('model_type')!r} is not Gr00tN1d7 (N1.6 checkpoints do not load "
            "under n1.7-release)"
        )
    pdir = ckpt if (ckpt / "processor_config.json").is_file() else ckpt / "processor"
    pc = json.loads((pdir / "processor_config.json").read_text())
    tags = sorted(pc.get("processor_kwargs", pc)["modality_configs"])
    if tag not in tags:
        raise Fail(
            f"checkpoint has no modality for tag {tag!r}; it has {tags} (a base model cannot serve a new embodiment)"
        )
    print(
        f"[E5G] checkpoint ok model_type=Gr00tN1d7 action_horizon={cfg.get('action_horizon')} tags={tags} "
        f"processor={pdir}",
        flush=True,
    )


def _check_start(a):
    """Refuse a start that cannot work: no env, wrong GR00T tree, port taken, no dims source, bad ckpt, name running."""
    P = Paths(a.root)
    if not P.py.is_file():
        raise Fail(f"no GR00T env at {P.py} (run: server setup --root {a.root})")
    for d in (P.logs, P.runs, P.modality):
        d.mkdir(parents=True, exist_ok=True)
    if head(P.repo) != GR00T_COMMIT:
        raise Fail(f"Isaac-GR00T tree is not {GR00T_COMMIT}")
    with socket.socket() as s:
        if s.connect_ex((a.host, a.port)) == 0:
            raise Fail(f"port {a.port} already listening")
    if not (a.dataset or a.image_hw):
        raise Fail("give --dataset (training LeRobot dir: dims + image size) or --image-hw HxW")
    ckpt = Path(a.ckpt).resolve()
    if not ckpt.is_dir():
        raise Fail(f"--ckpt DIR required (got {a.ckpt!r})")
    check_ckpt(ckpt, a.tag)
    name = a.name or f"{ckpt.parent.name}_{ckpt.name}_p{a.port}"
    if alive(P.runs / f"{name}.pid"):
        raise Fail(f"server {name} already running (pid {(P.runs / f'{name}.pid').read_text().strip()})")
    return P, ckpt, name


def _stats_check(P, ckpt):
    """Export the HF token and run stats_check.py on the checkpoint; fail before any server starts."""
    env = P.env()
    hf_token(env)
    print(f"[E5G] hf online (token exported; weights from {env['HF_HOME']})", flush=True)
    slog = P.logs / f"stats_check_{ckpt.name}.log"
    with open(slog, "w") as f:
        rc = subprocess.run(
            [
                str(P.py),
                str(ROOT / "experiments/policies/gr00t/stats_check.py"),
                "--ckpt",
                str(ckpt),
                "--load-processor",
            ],
            cwd=P.repo,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        ).returncode
    tail = slog.read_text(errors="replace").strip().splitlines() or [""]
    if rc != 0:
        print("\n".join(tail[-3:]))
        raise Fail(f"stats check failed for {ckpt} — not starting")
    print(f"[E5G] stats check PASS {tail[-1][:160]}", flush=True)
    return env


def _write_start_log(P, name, ckpt, a):
    """Open the server log with the start, tree and GPU lines; returns the server argv."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    log, mjson = P.logs / f"server_{name}_{ts}.log", P.modality / f"{name}.json"
    args = [
        "--model-path",
        str(ckpt),
        "--embodiment-tag",
        a.tag,
        "--device",
        "cuda",
        "--host",
        a.host,
        "--port",
        str(a.port),
    ]
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader", "-i", str(a.gpu)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    log.write_text(
        f"[E5G] start {ts} name={name} mode=model gpu={a.gpu} host={a.host} port={a.port} tag={a.tag}\n"
        f"[E5G] gr00t_commit={head(P.repo)} tag={GR00T_TAG} code={ROOT} hf_offline=0\n"
        f"[E5G] args {' '.join(args)} threads={a.threads}\n{gpu}\n"
    )
    return ts, log, mjson, args


def _spawn_server(env, a, log, P, args):
    """Start the GR00T policy server detached, on one GPU with a bounded CPU thread count."""
    # CPU threads: the torch/OMP default (every core) made the processor + collate step dominate a call on a shared
    # machine; 8 threads leave the model forward unchanged.
    senv = dict(
        env,
        CUDA_VISIBLE_DEVICES=str(a.gpu),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=str(a.threads),
        MKL_NUM_THREADS=str(a.threads),
    )
    with open(log, "a") as lf:
        p = subprocess.Popen(
            [str(P.py), ENTRY, *args],
            cwd=P.repo,
            env=senv,
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return p


def _record_run(P, name, p, a, log, ts, ckpt):
    """Write the pid/addr files and one ledger line for the started server."""
    (P.runs / f"{name}.pid").write_text(f"{p.pid}\n")
    (P.runs / f"{name}.addr").write_text(f"tcp://{a.host}:{a.port} gpu={a.gpu} log={log}\n")
    with open(P.runs / "ledger.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "ts": ts,
                    "name": name,
                    "mode": "model",
                    "pid": p.pid,
                    "gpu": a.gpu,
                    "host": a.host,
                    "port": a.port,
                    "tag": a.tag,
                    "ckpt": str(ckpt),
                    "replay": "",
                    "log": str(log),
                    "gr00t_commit": GR00T_COMMIT,
                }
            )
            + "\n"
        )


def _wait_ready(log, name, p, a):
    """Poll the server log until READY; fail on a start-up error, an early exit or the ready timeout."""
    t0 = time.time()
    while True:
        txt = log.read_text(errors="replace")
        if READY in txt:
            break
        if any(k in txt for k in DIED):
            err = next((ln for ln in txt.splitlines() if "rror" in ln), "")[-300:]
            raise Fail(f"server {name} died at start-up: {err} log={log}")
        if p.poll() is not None:
            raise Fail(f"server {name} exited before ready: {' '.join(txt.splitlines()[-3:])[-300:]} log={log}")
        if time.time() - t0 > a.ready_timeout:
            raise Fail(
                f"{name} not ready in {a.ready_timeout}s (still running pid={p.pid}) log={log}", 3, "[E5G] TIMEOUT"
            )
        time.sleep(2)
    load_s = int(time.time() - t0)
    return load_s


def _write_modality(a, mjson, ckpt, log, P, env, p, name, load_s):
    """Write the modality JSON through the live server and print the READY line."""
    mm = [
        "make-modality",
        "--host",
        a.host,
        "--port",
        str(a.port),
        "--embodiment-tag",
        a.tag,
        "--out",
        str(mjson),
        "--checkpoint",
        str(ckpt),
    ]
    mm += ["--dataset", a.dataset] if a.dataset else []
    mm += ["--image-hw", a.image_hw] if a.image_hw else []
    with open(log, "a") as lf:
        rc = subprocess.run(
            [str(P.py), "-m", "experiments.sim_eval.policy_client", *mm],
            cwd=ROOT,
            env=dict(env, PYTHONPATH=str(ROOT)),
            stdout=lf,
            stderr=subprocess.STDOUT,
        ).returncode
    if rc != 0:
        raise Fail(f"modality JSON (server up pid={p.pid}): {' '.join(log.read_text().splitlines()[-2:])} log={log}")
    line = f"[E5G] READY {name} tcp://{a.host}:{a.port} pid={p.pid} load_s={load_s} modality={mjson} log={log}"
    with open(log, "a") as lf:
        lf.write(line + "\n")
    print(line, flush=True)
    return 0


def start(a) -> int:
    P, ckpt, name = _check_start(a)
    env = _stats_check(P, ckpt)
    ts, log, mjson, args = _write_start_log(P, name, ckpt, a)
    p = _spawn_server(env, a, log, P, args)
    _record_run(P, name, p, a, log, ts, ckpt)
    load_s = _wait_ready(log, name, p, a)
    return _write_modality(a, mjson, ckpt, log, P, env, p, name, load_s)


def stop(a) -> int:
    f = Paths(a.root).runs / f"{a.name}.pid"
    pid = alive(f)
    if not pid:
        raise Fail(f"no live server named {a.name} (pidfile {f})")
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not alive(f):
            print(f"[E5G] STOPPED {a.name} pid={pid}")
            return 0
        time.sleep(1)
    os.kill(pid, signal.SIGKILL)
    print(f"[E5G] STOPPED(kill -9) {a.name} pid={pid}")
    return 0


def status(a) -> int:
    P = Paths(a.root)
    for f in sorted(P.runs.glob("*.pid")):
        pid = alive(f)
        addr = (P.runs / f"{f.stem}.addr").read_text().strip() if (P.runs / f"{f.stem}.addr").is_file() else ""
        print(f"[E5G] RUNNING {f.stem} pid={pid} {addr}" if pid else f"[E5G] DEAD {f.stem}")
    return 0


def _run(step: str, cmd: list, **kw) -> None:
    if subprocess.run(cmd, **kw).returncode != 0:
        raise Fail(step, 1, "[SETUP] FAIL")


def _sync_venv(P, a, env):
    """Build the venv from the release lock (frozen, optionally offline) and check that it imports."""
    _run(
        "uv-sync",
        [str(P.uv), "sync", "--python", "3.10", "--frozen", *(["--offline"] if a.uv_offline else [])],
        cwd=P.repo,
        env=env,
    )
    _run("uv.lock changed — env would differ from training's", ["git", "diff", "--quiet", "--", "uv.lock"], cwd=P.repo)
    _run(
        "import",
        [
            str(P.py),
            "-c",
            "import gr00t, torch, flash_attn, zmq, msgpack, transformers; print('[SETUP] import "
            "ok torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)",
        ],
        env=env,
    )


def _install_client_lib(a, env):
    """Install the client libs (zmq, msgpack) for the Isaac python into --client-lib and check the import."""
    if a.isaac_python:
        if not a.client_lib:
            raise Fail("--isaac-python needs --client-lib DIR", 1, "[SETUP] FAIL")
        _run(
            "client-lib",
            [
                a.isaac_python,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "--disable-pip-version-check",
                "--target",
                a.client_lib,
                *CLIENT_PINS,
            ],
        )
        _run(
            "client-lib-import",
            [
                a.isaac_python,
                "-c",
                "import zmq, msgpack; print('[SETUP] client lib ok zmq', zmq.__version__, 'msgpack', msgpack.version)",
            ],
            env=dict(env, PYTHONPATH=a.client_lib),
        )


def setup(a) -> int:
    """Build the private server env (idempotent): uv, Isaac-GR00T at the pinned tag, its venv from the release lock
    (never re-resolved), the base and backbone weights, and optionally the session client's libs for the Isaac
    python."""
    P = Paths(a.root)
    for d in ("bin", "logs", "hf", "cache/pip", "uv_cache", "uv_python"):
        (P.root / d).mkdir(parents=True, exist_ok=True)
    env = P.env()
    ver = subprocess.run([str(P.uv), "--version"], capture_output=True, text=True).stdout if P.uv.is_file() else ""
    if f" {UV_VERSION}" not in ver:
        url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
        with urllib.request.urlopen(url, timeout=300) as r, tarfile.open(fileobj=io.BytesIO(r.read())) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    m.name = Path(m.name).name
                    tf.extract(m, P.root / "bin")
    print(f"[SETUP] uv {subprocess.run([str(P.uv), '--version'], capture_output=True, text=True).stdout.strip()}")
    genv = dict(env, GIT_LFS_SKIP_SMUDGE="1")
    if not (P.repo / ".git").is_dir():
        _run("clone", ["git", "clone", "--quiet", "https://github.com/NVIDIA/Isaac-GR00T", str(P.repo)], env=genv)
    _run("checkout", ["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", GR00T_TAG], cwd=P.repo, env=genv)
    if head(P.repo) != GR00T_COMMIT:
        raise Fail(f"commit {head(P.repo)} != {GR00T_COMMIT}", 1, "[SETUP] FAIL")
    if not (P.repo / "gr00t/model/gr00t_n1d7").is_dir():
        raise Fail("not an N1.7 tree (no gr00t/model/gr00t_n1d7)", 1, "[SETUP] FAIL")
    _run("lfs-pull", ["git", "lfs", "pull", "--include", "scripts/deployment/dgpu/wheels/*"], cwd=P.repo, env=env)
    for w in sorted((P.repo / "scripts/deployment/dgpu/wheels").glob("*.whl")):
        if not zipfile.is_zipfile(w):
            raise Fail(f"lfs wheel still a pointer: {w}", 1, "[SETUP] FAIL")
    print(f"[SETUP] repo {head(P.repo)} ({GR00T_TAG})")
    # --uv-offline: install only from the uv cache (filled elsewhere with the same uv version and command).
    _sync_venv(P, a, env)
    hf_token(env)
    fetch = (
        "import sys\nfrom huggingface_hub import snapshot_download\nfor spec in sys.argv[1:]:\n"
        "    repo, rev = spec.split('@')\n    try:\n"
        "        p = snapshot_download(repo_id=repo, revision=rev, local_files_only=True); how = 'cached'\n"
        "    except Exception:\n        p = snapshot_download(repo_id=repo, revision=rev); how = 'downloaded'\n"
        "    print(f'[SETUP] weights {repo}@{rev[:8]} {how} -> {p}', flush=True)\n"
    )
    _run("hf-download", [str(P.py), "-c", fetch, "@".join(BASE), "@".join(BACKBONE)], env=env)
    _install_client_lib(a, env)
    print("[SETUP] DONE")
    return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("setup", "start", "stop", "status"):
        s = sub.add_parser(n)
        s.add_argument("--root", required=True, help="server env root (checkout, venv, caches, logs)")
        if n == "setup":
            s.add_argument("--isaac-python", default="", help="also install the session client's libs for this python")
            s.add_argument(
                "--client-lib", default="", help="pip --target folder of those libs (the policy JSON's pylib)"
            )
            s.add_argument("--uv-offline", action="store_true")
        if n == "start":
            s.add_argument("--ckpt", required=True)
            s.add_argument("--port", type=int, required=True)
            s.add_argument("--gpu", type=int, required=True)
            s.add_argument("--dataset", default="", help="training LeRobot dir (dims + image size)")
            s.add_argument("--image-hw", default="", help="HxW when no dataset is given, e.g. 180x320")
            s.add_argument("--tag", default="new_embodiment")
            s.add_argument("--name", default="")
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--ready-timeout", type=int, default=900)
            s.add_argument("--threads", type=int, default=8)
        if n == "stop":
            s.add_argument("--name", required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    return {"setup": setup, "start": start, "stop": stop, "status": status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
