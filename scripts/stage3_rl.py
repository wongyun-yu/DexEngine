#!/usr/bin/env python3
"""Stage 3 — RL: train or judge one run (paper Sec. 3.4). Thin CLI over `dexengine.rl.launch`.

  python3 scripts/stage3_rl.py RUN --side both --arm rby1 --hand sharpa_wave --dataset hot3d --emb bimanual \
      --seq-dir data/hot3d/trees/rby1_sharpa_wave/bimanual --seq-list <clip list> --gpu 0            # train
  … --mode judge --ckpt <run>/checkpoints/agent_51200.pt --judge-ne 1024 --judge-epc 64 --judge-iter 1600 \
      --switches-from <run>/switches.json            (the training's switches; the judge sidecar passes it)
  … --mode smoke   (64-env boot check)
  … --mode render --render-ckpts agent_51200.pt --render-rundir <run folder> --dry-run   |   … --mode replay --dry-run
                        (dumps only, for a record session: the RL trainer draws nothing — robot videos come only from
                         the record render, scripts/stage4_record.py; without --dry-run these modes die)
  … --set NE=2048 --set LAGW=0                                            (recipe overrides, configs/rl/recipe.yaml)
  … --set arm_ff=0 --set obj_spawn_hold=0       (RL switches: fields of dexengine.rl.tasks.switches.RLSwitches)
  … --dry-run           (write the manifest and the trainer argv dumps; start nothing)

Any input not given as a flag is read from a recorded launch environment (the form campaign cells and the judge
sidecar store: SIDE GPU SEQ_DIR DEXENGINE_ARM DEXENGINE_HAND DEXENGINE_DATASET DEXENGINE_EMBODIMENT
DEXENGINE_SEQ_LIST FROZEN_REPLAY_CKPT/_NE/_EVALS_PER_CLIP/_ITER RENDER_ONLY RENDER_RUNDIR SMOKE
DRYRUN and
recipe keys such as NE or KL_THRESH). Exit codes: see dexengine.rl.launch.dispatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

#: The code tree this script belongs to (not symlink-resolved: a tree that links dexengine/ and scripts/ runs in place).
ROOT = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(ROOT))
from dexengine.core import config  # noqa: E402
from dexengine.rl import launch  # noqa: E402
from dexengine.rl.tasks import switches  # noqa: E402

FLAG_ENV = {
    "side": "SIDE",
    "gpu": "GPU",
    "seq_dir": "SEQ_DIR",
    "arm": "DEXENGINE_ARM",
    "hand": "DEXENGINE_HAND",
    "dataset": "DEXENGINE_DATASET",
    "emb": "DEXENGINE_EMBODIMENT",
    "seq_list": "DEXENGINE_SEQ_LIST",
    "ckpt": "FROZEN_REPLAY_CKPT",
    "judge_ne": "FROZEN_REPLAY_NE",
    "judge_epc": "FROZEN_REPLAY_EVALS_PER_CLIP",
    "judge_iter": "FROZEN_REPLAY_ITER",
    "render_rundir": "RENDER_RUNDIR",
    "extra": "EXTRA",
}


def parse(argv):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run", help="run name (output names derive from it)")
    ap.add_argument("--mode", choices=("train", "smoke", "judge", "render", "replay"))
    for f in FLAG_ENV:
        ap.add_argument("--" + f.replace("_", "-"), dest=f)
    ap.add_argument("--render-ckpts", help="render dump: checkpoint file names, comma separated")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="recipe override (repeatable)")
    ap.add_argument(
        "--switches-from",
        metavar="FILE",
        help="judge/render: the training run's switches record (<run folder>/switches.json)",
    )
    ap.add_argument("--dry-run", action="store_true", help="write manifest + argv dumps, start nothing")
    ap.add_argument("--print", action="store_true", help="print the resolved trainer processes and exit")
    config.add_data_root_arg(ap)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    config.apply_data_root(a)  # before the launch environment is copied: the trainer inherits it
    env = dict(os.environ)
    for f, k in FLAG_ENV.items():
        if getattr(a, f) is not None:
            env[k] = getattr(a, f)
    for kv in a.set:
        k, sep, v = kv.partition("=")
        if not sep:
            print(f"[launch] --set needs KEY=VALUE (got {kv!r})", file=sys.stderr)
            return 2
        if k != k.upper() and k not in switches.FIELDS:
            print(f"[launch] --set {k}: not a switch (fields: {', '.join(switches.FIELDS)})", file=sys.stderr)
            return 2
        env[k] = v
    mode_env = {
        "smoke": {"SMOKE": "1"},
        "render": {"RENDER_ONLY": (a.render_ckpts or "").replace(",", " ")},
        "replay": {"RENDER_ONLY": "replay"},
    }
    if a.mode in mode_env:
        env.update(mode_env[a.mode])
    if a.dry_run:
        env["DRYRUN"] = "1"
    try:
        req = launch.from_env(a.run, env, root=ROOT, **({"switches_from": a.switches_from} if a.switches_from else {}))
        if a.mode and req.mode != a.mode:
            raise launch.RecipeError(f"--mode {a.mode} but the inputs select {req.mode}")
        if a.print:
            for p in launch.resolve(dataclasses.replace(req)):
                print(f"# log {p.log}\n{p.dump()}", end="")
            return 0
        return launch.run(req)
    except launch.RecipeError as e:
        print(f"[launch] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
