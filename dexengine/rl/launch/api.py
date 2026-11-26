"""Public launch API: resolve a request into trainer processes, run it, read or write dry-run dumps.

from dexengine.rl import launch
req = launch.from_env("myrun", {"SIDE": "both", "GPU": "0", "SEQ_DIR": …, "DEXENGINE_ARM": "rby1", …})
for p in launch.resolve(req):      # argv (argv[0] = dexengine/rl/cli.py) + DEXENGINE_* env + log path
    ...
rc = launch.run(req)               # or run(dataclasses.replace(req, dry_run=True)) to write the dumps
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dexengine.rl.launch import modes, process, records
from dexengine.rl.launch import plan as PL
from dexengine.rl.launch.dispatch import run
from dexengine.rl.launch.process import Launch, load_dump
from dexengine.rl.launch.recipe import RecipeError, table_values
from dexengine.rl.launch.request import Request, from_env

__all__ = ["Launch", "RecipeError", "Request", "from_env", "load_dump", "resolve", "run", "table_values", "write_dump"]


def resolve(req: Request) -> list[Launch]:
    """The trainer processes a launch starts (training: the first training process; render dumps: one per checkpoint;
    replay dumps: one per clip, each with its own one-clip view folder under a fresh temporary directory). Nothing is
    started and no record is written; the log paths are those of a real launch (or of the dry folder when
    `req.dry_run`)."""
    plan = PL.prepare(req)
    ref = records.log_dir(req.dry_run, req.run_tree)
    if req.mode == "smoke":
        calls = [modes.smoke_call(plan)]
    elif req.mode == "judge":
        calls = [modes.judge_call(plan)]
    elif req.mode == "render":
        calls = modes.render_calls(plan, modes.scene_env(plan, physx_lite=True))
    elif req.mode == "replay":
        look = modes.scene_env(plan, physx_lite=req.path == "bi")
        calls = []
        for c in modes.replay_clips(plan):
            view = Path(tempfile.mkdtemp(prefix="replay_seq_"))
            (view / c).symlink_to(os.path.realpath(Path(req.seq_dir) / c))
            calls.append(modes.replay_call(plan, str(view), c, look))
    else:
        calls = [modes.train_call(plan)]
    return [process.resolve_call(plan, c, ref) for c in calls]


def write_dump(launch: Launch, path) -> Path:
    """Write a dry-run dump of one trainer process (the format `load_dump` and data recording read)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    launch.write_switches()
    p.write_text(launch.dump())
    return p
