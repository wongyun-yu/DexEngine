#!/usr/bin/env python3
"""Resident bake worker and its client.

`bake_drive.sh` starts one worker per lane; `chain_fast.sh` sends one clip's stage list to a free worker with the
`send` subcommand. A worker imports torch · pinocchio · mujoco · warp, hashes and loads the Warp kernels and builds
the pytorch_kinematics chains once instead of once per clip.

Isolation between clips (= the meaning of a new process), as rules:
  1. Every module of this repository has its globals snapshotted right after it executes (a hook on
     SourceFileLoader.exec_module — modules first loaded during a request too) and restored to that snapshot per
     request (containers, scalars, arrays; functions, classes and modules are code and stay). Globals created after
     import are deleted.
  2. Exceptions are declared by the module that owns a cache: a module global `_SERVER_KEEP = ("name", …)` — only
     pure caches whose keys do not depend on the clip. Without a declaration everything is restored.
     2b. A container global the snapshot cannot deep-copy (a mappingproxy, a Warp kernel …) and that the module does not
     declare stops its import with the module and name: restored by deep copy it would kill the worker at its next
     request (the first request ran, every later clip on that lane failed with 'worker … disconnected').
  3. Globals meaning "module load time" are declared as `_SERVER_LOADTIME = ("name", …)`; the server sets them to
     time.time() right before the stage is first called in a request (as if the module were loaded at that moment).
  4. The request environment must equal the worker's start environment (except PWD · OLDPWD · SHLVL · _ that the
     shell changes itself); otherwise rc 97 — import-time env reads must not silently run with other values.
  5. stdout and stderr are redirected per request to the clip log at fd level (C library output included).
  6. A client that disconnects mid-request (the chain was killed by timeout) makes the worker exit (os._exit)
     instead of continuing, so it never races the next anchor's re-bake on the same tree.
  7. The MuJoCo asset cache (process-wide mesh parse results, determined by the files, independent of the output)
     stops accepting meshes when full, losing hits even within one clip's recompiles. Its capacity is MJ_CACHE_MB
     and it is cleared at request start when more than half full — robot and hand meshes hit across clips, only
     object meshes churn.

Usage:
  server.py --sock <path> --ready <file>                                 one worker (bake_drive.sh)
  server.py send --pool <dir> --log <clip log> -- <stage> [<stage> …]     one clip (chain_fast.sh)
The `send` path uses the standard library only (start-up ≈ 20 ms).
"""

from __future__ import annotations

import contextlib
import copy
import importlib._bootstrap_external as _bext
import json
import os
import socket
import sys
import threading
import time
import traceback
import types
from pathlib import Path

from dexengine.core.errors import input_missing

_WTS = ""  # repository root + os.sep — set by main (the `send` path imports nothing from the package)
VOLATILE_ENV = {"PWD", "OLDPWD", "SHLVL", "_"}
MJ_CACHE_MB = 1024  # (7) MuJoCo asset cache capacity of one worker (16 lanes → up to 16 GB)
_SIMPLE: tuple = ()  # value types restored by deep copy — set by main (numpy types) before the hook is installed

# (1) import-time snapshots — the moment a module of this repository executes
_SNAP: dict[int, tuple[types.ModuleType, dict]] = {}
_orig_exec = _bext.SourceFileLoader.exec_module


def _is_repo(mod) -> bool:
    f = getattr(mod, "__file__", None)
    return bool(f) and os.path.realpath(f).startswith(_WTS)


def _snapshot(mod) -> dict:
    snap = {}
    keep = set(getattr(mod, "_SERVER_KEEP", ()) or ())
    for k, v in vars(mod).items():
        if k.startswith("__") and k.endswith("__"):
            continue
        if isinstance(v, _SIMPLE):
            try:
                snap[k] = copy.deepcopy(v)
            except Exception as e:  # noqa: BLE001 — rule 2b: restoring it would deep-copy it again and die
                if k not in keep:
                    raise RuntimeError(
                        f"[STAGE_SERVER] ✗ {mod.__name__}.{k} ({type(v).__name__}) cannot be deep-copied "
                        f"({type(e).__name__}: {e}) — the next request's restore would die on it. Declare it in that "
                        "module's _SERVER_KEEP (a clip-independent constant or cache) or make it copyable"
                    ) from e
                snap[k] = _Ref(v)  # declared: _restore_all leaves it as it is
        else:
            snap[k] = _Ref(v)
    return snap


class _Ref:
    """Code objects (functions, classes, modules) — the same object is put back."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


def _exec_hook(self, module):
    _orig_exec(self, module)
    if _is_repo(module):
        _SNAP[id(module)] = (module, _snapshot(module))


def _restore_all() -> int:
    n = 0
    live = {id(m) for m in list(sys.modules.values())}
    for key in [k for k in _SNAP if k not in live]:  # forget old modules whose name now points elsewhere
        del _SNAP[key]
    for mod, snap in list(_SNAP.values()):
        keep = set(getattr(mod, "_SERVER_KEEP", ()) or ())
        d = vars(mod)
        for k in [k for k in d if not (k.startswith("__") and k.endswith("__")) and k not in snap and k not in keep]:
            del d[k]
            n += 1
        for k, v in snap.items():
            if k in keep:
                continue
            if isinstance(v, _Ref):
                if d.get(k) is not v.v:
                    d[k] = v.v
                    n += 1
            else:
                d[k] = copy.deepcopy(v)
    return n


def _env_diff(req_env: dict, base_env: dict) -> list[str]:
    keys = (set(req_env) | set(base_env)) - VOLATILE_ENV
    return sorted(k for k in keys if req_env.get(k) != base_env.get(k))


def _run_request(req, base_env, nreq, SR):
    """One request: refuse an environment that differs from the start, else reset the globals, trim the MuJoCo asset
    cache and run the stage list. Returns rc."""
    diff = _env_diff(req["env"], base_env)
    if diff:
        print(
            f"[STAGE_SERVER] ✗ the request environment differs from the worker's start environment: "
            f"{diff[:8]} — it must run on a new worker",
            flush=True,
        )
        rc = 97
    else:
        os.chdir(req["cwd"])
        nres = _restore_all()  # (1)(2) isolation between clips
        import mujoco

        _C = mujoco.mj_getCache()
        _csz = mujoco.mj_getCacheSize(_C)
        if _csz > mujoco.mj_getCacheCapacity(_C) // 2:  # (7)
            mujoco.mj_clearCache(_C)
        loaded = set()

        def _before(name, mod, loaded=loaded):  # (3) load-time globals
            if name not in loaded:
                loaded.add(name)
                for k in getattr(mod, "_SERVER_LOADTIME", ()) or ():
                    setattr(mod, k, time.time())

        print(
            f"[STAGE_SERVER] request #{nreq} pid={os.getpid()} globals reset {nres} · "
            f"asset cache {_csz >> 20}MB · stages {len(req['specs'])}",
            flush=True,
        )
        try:
            rc = SR.run_specs(req["specs"], before=_before)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        except BaseException:  # noqa: BLE001 — an unhandled exception of a new process = rc 1
            traceback.print_exc()
            rc = 1
    return rc


def _args():
    """Command line: socket path and ready file."""
    import argparse

    ap = argparse.ArgumentParser(description="Resident bake worker (bake_drive.sh starts one per lane)")
    ap.add_argument("--sock", required=True)
    ap.add_argument("--ready", required=True)
    a = ap.parse_args()
    return a


def _warm_and_listen(mujoco, a, t0):
    """Size the MuJoCo asset cache, report the warm-up, bind the socket and write the ready file."""
    mujoco.mj_setCacheCapacity(mujoco.mj_getCache(), MJ_CACHE_MB << 20)  # (7)
    print(
        f"[STAGE_SERVER] pid={os.getpid()} sock={a.sock} cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"warm={time.time() - t0:.1f}s modules={len(_SNAP)}",
        flush=True,
    )
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(a.sock):
        os.unlink(a.sock)
    srv.bind(a.sock)
    srv.listen(4)
    Path(a.ready).write_text(f"{os.getpid()}\n")
    return srv


def main() -> int:
    """Worker: warm up, then serve stage-list requests on a Unix socket until a `stop` request."""

    a = _args()
    import numpy as np

    from dexengine.core.repo import REPO_ROOT as WT  # constants only (nothing to snapshot)

    global _SIMPLE, _WTS
    _SIMPLE = (dict, list, set, tuple, int, float, str, bytes, bool, type(None), np.ndarray, np.generic)
    _WTS = str(WT) + os.sep
    _bext.SourceFileLoader.exec_module = _exec_hook
    base_env = dict(os.environ)
    os.chdir(WT)
    t0 = time.time()
    from dexengine.placement.bake import chain as SR  # noqa: E402  (stage table and runner)

    for name in SR.STAGES:  # load the stage modules (= pay the import cost once)
        SR._mod(name)
    import mujoco

    from dexengine.placement.repair import xr_gpu  # noqa: F401,E402  — the cross-repair GPU path's heavy import (warp)

    srv = _warm_and_listen(mujoco, a, t0)
    out_fd, err_fd = os.dup(1), os.dup(2)
    nreq = 0
    while True:
        conn, _ = srv.accept()
        f = conn.makefile("rb")
        line = f.readline()
        if not line:
            conn.close()
            continue
        req = json.loads(line)
        if req.get("op") == "stop":
            conn.sendall(b'{"rc": 0}\n')
            conn.close()
            print(f"[STAGE_SERVER] stop request — {nreq} requests served", flush=True)
            return 0
        nreq += 1
        busy = {"on": True}

        def _watch(c=conn, b=busy):  # (6) client gone = worker exits
            try:
                x = c.recv(1, socket.MSG_PEEK)
            except OSError:
                x = b""
            if not x and b["on"]:
                os.write(
                    err_fd,
                    f"[STAGE_SERVER] the client disconnected mid-request — dropping the work and exiting"
                    f"(pid {os.getpid()})\n".encode(),
                )
                os._exit(3)

        threading.Thread(target=_watch, daemon=True).start()
        rc = 99
        logf = os.open(req["log"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(logf, 1)
        os.dup2(logf, 2)
        try:
            rc = _run_request(req, base_env, nreq, SR)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.close(logf)
            busy["on"] = False
        with contextlib.suppress(OSError):
            conn.sendall((json.dumps({"rc": int(rc)}) + "\n").encode())
        conn.close()


def client_main(argv=None) -> int:
    """Client (`send`): send one clip's stage list to a free worker of the pool and return its rc.

    Worker choice: among `<pool>/s*.<device>.sock` the free ones (flock on `<sock>.lock`) on **this request's
    CUDA_VISIBLE_DEVICES** device. No free worker before the deadline → rc 96 (never a silent new interpreter);
    a worker that disconnects without an answer → rc 95."""
    import argparse
    import fcntl
    import glob

    ap = argparse.ArgumentParser(prog="server.py send")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--wait", type=float, default=600.0, help="deadline (s) to wait for a free worker")
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args(argv)
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    socks = sorted(glob.glob(os.path.join(a.pool, f"s*.{dev or 'cpu'}.sock")))
    if not socks:
        print(input_missing(f"[stage_client] ✗ pool {a.pool} has no worker on device '{dev}'"), file=sys.stderr)
        return 96
    t_end = time.time() + a.wait
    while True:
        for sp in socks:
            lk = open(sp + ".lock", "a")  # noqa: SIM115 — the flock stays held until the request ends
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lk.close()
                continue
            try:
                c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                c.connect(sp)
            except OSError:
                lk.close()
                continue  # dead worker — next
            req = {"specs": a.specs, "log": os.path.abspath(a.log), "cwd": os.getcwd(), "env": dict(os.environ)}
            c.sendall((json.dumps(req) + "\n").encode())
            line = c.makefile("rb").readline()
            c.close()
            lk.close()
            if not line:
                print(f"[stage_client] ✗ worker {sp} disconnected without an answer", file=sys.stderr)
                return 95
            return int(json.loads(line)["rc"])
        if time.time() > t_end:
            print(
                input_missing(f"[stage_client] ✗ no free worker within {a.wait:.0f}s ({len(socks)} workers)"),
                file=sys.stderr,
            )
            return 96
        time.sleep(0.05)


if __name__ == "__main__":
    if sys.argv[1:2] == ["send"]:
        sys.exit(client_main(sys.argv[2:]))
    sys.exit(main())
