"""Start one trainer process (Isaac Lab) — or, in a dry run, write its argv and trainer environment instead.

The trainer runs as `<IsaacLab>/isaaclab.sh -p dexengine/rl/cli.py …` from the code tree, with the Isaac Lab
virtualenv active and the tree first on PYTHONPATH. The plan's and the call's values for the RL switches
(`dexengine.rl.tasks.switches`) go into one switches file next to the process log, written when the process starts (dry
runs included); the process gets its path in DEXENGINE_RL_SWITCHES. Its other DEXENGINE_* environment is exactly the
plan's plus the call's (the caller's other DEXENGINE_* variables are part of the plan). A dry-run dump is one argv
token per line followed by `ENV::KEY=VALUE` lines (sorted); data recording replays these dumps.
The process's GPU is the call's ``gpu`` (``GPU`` / ``--gpu``, required) and reaches it as CUDA_VISIBLE_DEVICES; a
CUDA_VISIBLE_DEVICES already set in the launcher's environment must name the same GPU (:func:`gpu_bind`).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import time
from pathlib import Path

from dexengine.rl.launch.argv import Call, trainer_args
from dexengine.rl.launch.plan import Plan
from dexengine.rl.launch.recipe import RecipeError
from dexengine.rl.tasks import switches

#: Threads of the trainer's CPU math libraries.
CPU_THREADS = "8"


@dataclasses.dataclass
class Launch:
    """A resolved trainer process: argv (argv[0] = dexengine/rl/cli.py), its DEXENGINE_* environment, the other
    environment it needs, the log file it writes, and its switches with the fields the launch gave."""

    argv: list
    env: dict
    process_env: dict
    log: Path
    switches: switches.RLSwitches = dataclasses.field(default_factory=switches.RLSwitches)
    given: frozenset = frozenset()

    def write_switches(self) -> Path:
        """Write the switches file this process reads (the path in its DEXENGINE_RL_SWITCHES)."""
        return switches.write(self.env[switches.SWITCHES_ENV], self.switches, self.given)

    def dump(self) -> str:
        """Dry-run dump text (argv lines, then ENV:: lines)."""
        return "".join(a + "\n" for a in self.argv) + "".join(f"ENV::{k}={v}\n" for k, v in sorted(self.env.items()))


def load_dump(path) -> tuple[list[str], dict]:
    """(argv, DEXENGINE_* env) of a dry-run dump."""
    argv, env = [], {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("ENV::"):
            k, _, v = ln[5:].partition("=")
            env[k] = v
        elif ln.strip():
            argv.append(ln)
    return argv, env


def isaaclab_dir(root: Path) -> Path:
    """Isaac Lab checkout: ISAACLAB_DIR, else a sibling `IsaacLab` of the code tree, the data root or the git
    common directory (the first that holds env_isaaclab/bin/python)."""
    if os.environ.get("ISAACLAB_DIR"):
        return Path(os.environ["ISAACLAB_DIR"])
    cands = [Path(root) / ".." / "IsaacLab", Path("..") / "IsaacLab"]
    if os.environ.get("DEXENGINE_MAIN_ROOT"):
        cands.append(Path(os.environ["DEXENGINE_MAIN_ROOT"]) / ".." / "IsaacLab")
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        cands.append((Path(root) / r.stdout.strip()).resolve().parent / ".." / "IsaacLab")
    for c in cands:
        if os.access(c / "env_isaaclab" / "bin" / "python", os.X_OK):
            return c.resolve()
    raise RecipeError(
        "Isaac Lab not found — set ISAACLAB_DIR=<Isaac Lab checkout>. Expected layout: "
        "$ISAACLAB_DIR/isaaclab.sh and an executable $ISAACLAB_DIR/env_isaaclab/bin/python (the "
        "Isaac Lab venv). Unset, only a sibling ../IsaacLab is searched (of the code tree, the cwd, "
        f"the data root, the main checkout; looked at {[str(c) for c in cands]})"
    )


def switches_path(log: Path) -> Path:
    """The switches file of the process that writes `log`."""
    return log.with_name(f"{log.stem}.switches.json")


def from_record(plan: Plan, call: Call, texts: dict) -> tuple:
    """Switches of a judge or render process over the training run's record: the record's values, then the call's
    context switches (render look). A switch the caller gives explicitly must equal the record's value; launch values
    from the recipe that differ are reported (the record wins: the checkpoint was trained under it)."""
    path = plan.req.switches_from
    base, given = switches.read(path)
    caller = {k: v for k, v in plan.req.env.items() if k in switches.FIELD_OF}
    of = switches.FIELD_OF
    bad = [
        f"{k}={v!r} (training: {switches.render(of[k], getattr(base, of[k]))!r})"
        for k, v in caller.items()
        if not switches.same(of[k], v, getattr(base, of[k]))
    ]
    if bad:
        raise RecipeError(
            f"{bad} differ from the training run's switches {path}: a checkpoint is judged under the "
            "switches it was trained with"
        )
    ctx = {k: texts[k] for k in call.context_switches if k in texts}
    differ = sorted(
        k
        for k, v in texts.items()
        if k not in ctx and not switches.same(switches.FIELD_OF[k], v, getattr(base, switches.FIELD_OF[k]))
    )
    print(
        f"[launch] switches of {call.exp}: training record {path} + context {sorted(ctx) or '-'}"
        + (f" · the record wins over the launch values of {differ}" if differ else ""),
        flush=True,
    )
    changes = {switches.FIELD_OF[k]: v for k, v in ctx.items()}
    return switches.build(changes, base), given | set(changes)


def gpu_bind(gpu: str, environ=None) -> str:
    """The CUDA_VISIBLE_DEVICES of a trainer process: the call's ``gpu``. A non-empty CUDA_VISIBLE_DEVICES in the
    launcher's own environment is a second statement of the GPU; when it names another GPU the launch dies here,
    before any process starts (overwriting it silently could put several 'per-GPU' runs on one GPU). An empty one
    names no GPU (the test suite hides GPUs that way), so the call's value is used. Prints
    ``[GPU_BIND] visible=<v> gpu=<g> source=<flag|env+flag>``."""
    environ = os.environ if environ is None else environ
    outer = environ.get("CUDA_VISIBLE_DEVICES", "")
    if outer and outer != gpu:
        raise RecipeError(
            f"CUDA_VISIBLE_DEVICES={outer} in the launcher's environment and GPU={gpu} name different GPUs — "
            "give one GPU, the same in both (or unset CUDA_VISIBLE_DEVICES)"
        )
    print(f"[GPU_BIND] visible={gpu} gpu={gpu} source={'env+flag' if outer else 'flag'}", flush=True)
    return gpu


def resolve_call(plan: Plan, call: Call, log_dir: Path) -> Launch:
    """The trainer process of one call: switch values (old environment names in the plan and the call) go to its
    switches file, the rest of the DEXENGINE_* values to its environment."""
    visible = gpu_bind(call.gpu)  # first: a launch that names two GPUs dies before anything is written
    from dexengine.core import desk as DK
    from dexengine.rl import app as TK

    env = {**plan.env, **call.env}
    given = [k for k in (TK.ENTRY_ENV, *TK.REMOVED) if k in env]
    if given:
        raise RecipeError(
            f"{given}: the launch mode names the entry and its kit (dexengine.rl.app) — a caller cannot set it"
        )
    if gone := [k for k in DK.REMOVED_DESK_ENV if k in env]:
        raise RecipeError(f"{gone} removed — one desk (Willow from the look mirror) for every scene")
    if call.entry not in TK.ENTRIES:
        raise RecipeError(f"call {call.exp}: entry {call.entry!r} not one of {TK.ENTRIES}")
    env[TK.ENTRY_ENV] = call.entry
    texts = {k: env.pop(k) for k in list(env) if k in switches.FIELD_OF}
    try:
        if plan.req.switches_from:
            sw, given = from_record(plan, call, texts)
        else:
            sw, given = switches.from_env_names(texts)
    except switches.SwitchError as e:
        raise RecipeError(str(e)) from None
    log = log_dir / call.log_name
    env[switches.SWITCHES_ENV] = str(switches_path(log))
    return Launch(
        argv=trainer_args(call, plan.values, plan.req.dataset),
        env=env,
        process_env={"CUDA_VISIBLE_DEVICES": visible, **call.process_env},
        log=log,
        switches=sw,
        given=given,
    )


def child_environ(launch: Launch, root: Path) -> dict:
    """Full environment of a trainer process."""
    lab = isaaclab_dir(root)
    venv = lab / "env_isaaclab"
    base = {k: v for k, v in os.environ.items() if not k.startswith("DEXENGINE_") and k != "PYTHONHOME"}
    pp = os.environ.get("PYTHONPATH", "")
    base.update(
        PYTHONUNBUFFERED="1",
        PYTHONFAULTHANDLER="1",
        OMP_NUM_THREADS=CPU_THREADS,
        MKL_NUM_THREADS=CPU_THREADS,
        OMNI_KIT_ACCEPT_EULA="YES",
        VIRTUAL_ENV=str(venv),
        PATH=f"{venv / 'bin'}:{os.environ.get('PATH', '')}",
        PYTHONPATH=str(root) + (f":{pp}" if pp else ""),
        ISAACLAB_DIR=str(lab),
    )
    return {**base, **launch.env, **launch.process_env}


def start(launch: Launch, root: Path, dry_run: bool) -> int:
    """Run the trainer to completion (stdout and stderr into its log) and return its exit code; a dry run writes
    the dump into the log and returns 0."""
    launch.log.parent.mkdir(parents=True, exist_ok=True)
    launch.write_switches()
    if dry_run:
        launch.log.write_text(launch.dump())
        return 0
    cmd = [str(isaaclab_dir(root) / "isaaclab.sh"), "-p", *launch.argv]
    print(f"[launch {time.strftime('%T')}] {launch.argv[0]} exp -> {launch.log}", flush=True)
    with open(launch.log, "wb") as f:
        rc = subprocess.run(
            cmd, cwd=root, env=child_environ(launch, root), stdout=f, stderr=subprocess.STDOUT
        ).returncode
    print(f"  rc={rc}", flush=True)
    return rc


def wait_gpu(gpu: str, need_mib: int, max_wait_s: int = 600) -> None:
    """Wait until the first GPU of `gpu` has `need_mib` free (exit 6 after `max_wait_s`)."""
    g, waited, free = gpu.split(",")[0], 0, None
    while waited < max_wait_s:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits", "-i", g],
            capture_output=True,
            text=True,
        )
        if r.returncode or not r.stdout.strip():
            print("[launch] nvidia-smi failed — not waiting for GPU memory", flush=True)
            return
        total, used = (int(x) for x in r.stdout.split(",")[:2])
        free = total - used
        if free >= need_mib:
            if waited:
                print(f"[launch] GPU{g} {free} MiB free after {waited}s", flush=True)
            return
        if waited == 0:
            print(f"[launch] GPU{g} has {free} MiB < {need_mib} MiB free — waiting for the previous job", flush=True)
        time.sleep(10)
        waited += 10
    print(
        f"[launch] GPU{g} did not free {need_mib} MiB within {max_wait_s}s ({free} MiB free) — not launching",
        flush=True,
    )
    raise SystemExit(6)
