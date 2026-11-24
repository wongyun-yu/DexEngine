"""Run core of the Isaac record session: dump reading, post-import hooks, running the canonical entry, finishing, and
the stage-1 (record) hook.

The entry is dexengine/record/session/entry.py (stage choice, render-stage hooks); this file holds only the part
that touches the **physics record (stage 1)**: the code fingerprints of the stage-1 and augment cache keys
(record.cache.CODE_FILES) measure this file and not the entry. So adding or changing a render stage (only the entry
and render modules change) does not invalidate the physics records built elsewhere. Changing this file reruns every
physics record, so render work does not go here. The entry re-exports the names (SS.load_dump …).
Marker: ``[COLLECT_SESSION] …`` (same head as the entry).
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.repo import REPO_ROOT

HERE = Path(__file__).resolve().parent
ROOT = REPO_ROOT  # tree root
MARK = "[COLLECT_SESSION]"
HOOK = f"{MARK} hook:"  # head of every hook line (session.launch.HOOK_LINE reads it)
_STATE = {"done": False}
#: post-import hook target -> the marker line its hook returned. When a target is first imported during Kit startup
#: (extension start), the marker printed then does not reach the session log (Kit redirects Python output to its own
#: log); prove_hooks prints it again where it is visible, and dies if the hook did not run.
_FIRED: dict[str, str] = {}


#: the canonical trainer entry a DRYRUN dump starts with (its first argv)
ENTRY = "dexengine/rl/cli.py"


def load_dump(path: str | os.PathLike) -> tuple[list[str], dict]:
    """DRYRUN log -> (canonical entry argv, DEXENGINE_* env). A first argv other than the canonical entry dies."""
    argv, env = [], {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("ENV::"):
            k, _, v = ln[5:].partition("=")
            env[k] = v
        elif ln.strip():
            argv.append(ln)
    if not argv or not argv[0].endswith(ENTRY):
        raise SystemExit(f"{MARK} the dump's first line is not {ENTRY}: {argv[:1]} ({path})")
    return argv, env


def set_arg(argv: list[str], flag: str, val: str) -> None:
    """Replace a flag's value in argv; a flag missing from the dump dies (the canonical entry did not emit it, so the
    dump has another shape)."""
    if flag not in argv:
        raise InputMissingError(f"{MARK} the dump has no {flag}")
    argv[argv.index(flag) + 1] = val


def require(argv: list[str], *flags: str) -> None:
    miss = [f for f in flags if f not in argv]
    if miss:
        raise InputMissingError(f"{MARK} the dump has no {miss}: not a dump of the expected canonical branch")


class PostImport:
    """A meta path finder that calls the hook right after a module first runs (it does not import the module early)."""

    def __init__(self, hooks):
        self.hooks, self.done = dict(hooks), set()
        if early := sorted(n for n in self.hooks if n in sys.modules):  # a module already loaded cannot be hooked
            raise SystemExit(f"{MARK} ✗ hook targets {early} were imported before the hooks were set")

    def find_spec(self, name, path=None, target=None):
        if name not in self.hooks or name in self.done:
            return None
        import importlib.util

        self.done.add(name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            raise SystemExit(f"{MARK} hook target module {name} not found")
        orig, hook = spec.loader.exec_module, self.hooks[name]

        def exec_module(module):
            orig(module)
            _FIRED[name] = hook(module) or "fired"

        spec.loader.exec_module = exec_module
        return spec


def prove_hooks(names: list[str]) -> None:
    """Proof that the post-import hooks ran: print each hook's marker line again where it is visible (render function,
    stage-1 recorder entry). Any target whose hook did not run dies (a session whose hook silently fell out is
    invalid)."""
    if miss := [n for n in names if n not in _FIRED]:
        raise SystemExit(f"{MARK} ✗ post-import hooks did not run {miss}: the session is invalid")
    for n in names:
        print(f"{MARK} hook proof {n}: {_FIRED[n]}", flush=True)


def tree_line() -> str:
    def git(*c):
        return subprocess.run(["git", "-C", str(ROOT), *c], capture_output=True, text=True).stdout.strip()

    return f"tree={ROOT} commit={git('rev-parse', '--short', 'HEAD')} branch={git('rev-parse', '--abbrev-ref', 'HEAD')}"


def run_entry(argv: list[str], env: dict, hooks: dict) -> None:
    """Run the canonical trainer entry in this process with the dump's argv and env and the post-import hooks."""
    os.environ.update(env)
    os.environ["COLLECT_TREE"] = tree_line()
    print(f"{MARK} {os.environ['COLLECT_TREE']} · hook targets {sorted(hooks)}", flush=True)
    sys.meta_path.insert(0, PostImport(hooks))
    sys.argv = list(argv)
    if not argv[0].endswith(ENTRY):
        raise SystemExit(f"{MARK} dump entry {argv[0]} is not {ENTRY}")
    # started by module name (the canonical entry is a package module, not loaded by file path) · __main__ context kept
    runpy.run_module("dexengine.rl.cli", run_name="__main__", alter_sys=True)


def finish(ok: bool, wrapped) -> None:
    """End of a render stage: after the output check, close the env (module doc of session.entry, 'Shutdown') and
    return. On failure SystemExit (the canonical entry ends with rc 1)."""
    _STATE["done"] = ok
    wrapped.close()
    print(f"{MARK} env closed (STOP unsubscribed): the canonical entry's app exit follows · ok={ok}", flush=True)
    if not ok:
        raise SystemExit(f"{MARK} output count check failed")
    from dexengine.rl.isaac_cli import mark_closing  # noqa: PLC0415 — loaded by the canonical entry (stdlib only)

    mark_closing("record render DONE · output count check passed")


def _hook_record(mod):
    from dexengine.record.rollout import recorder as REC

    prove_hooks(["gymnasium"])  # Kit settings applied (session.kit.check_applied; stage 1's gymnasium hook)
    mod.TrajRecorder = REC.collect_recorder_class(mod.TrajRecorder)
    line = f"{HOOK} traj_record.TrajRecorder → rollout.recorder.CollectRecorder"
    print(line, flush=True)
    return line


def record(argv: list[str], env: dict) -> None:
    """Stage 1: run the frozen entry used by judging with its trajectory recorder replaced by the record recorder."""
    require(argv, "--num_envs")
    if not env.get("DEXENGINE_TRAJ_RECORD") or env.get("DEXENGINE_FROZEN_REPLAY") != "1":
        raise InputMissingError(f"{MARK} the record dump lacks DEXENGINE_TRAJ_RECORD and DEXENGINE_FROZEN_REPLAY=1")
    run_entry(argv, env, {"dexengine.rl.eval.traj_record": _hook_record})
