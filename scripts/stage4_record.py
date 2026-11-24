#!/usr/bin/env python3
"""Stage 4 (record): run one record tool from the tree root.

Usage: python scripts/stage4_record.py <command> [args...]      (retarget venv python; `<command> -h` for its options)

  drive          drive record bundles over GPUs (dexengine.record.driver); needs DEXENGINE_MAIN_ROOT and
                 DEXENGINE_RL_QUEUE_ROOT
  bundle         record one bundle: stage 1 rollouts, render, store (dexengine.record.bundle)
  merge          merge pieces into the final LeRobot dataset (dexengine.record.store.merge)
  scene-layout   write / check the scene layout sidecar of final datasets (dexengine.record.store.scene_layout)
  scene-manifest write / check the scene manifest sidecar of a final dataset (dexengine.record.store.scene_manifest)
  look-mirror    build / verify the local look-D asset mirror (dexengine.record.looks.look_d)
  assets         build the visual-augmentation asset root (dexengine.record.looks.augvis.assets)
  text           language annotations of final datasets (dexengine.record.text.pipeline)

  --data-root DIR (any command): the data root (dexengine.core.config), exported to the processes the command starts.
  (--set scene.… is gone: one desk, the Willow table of the look mirror, for every scene — render_unify.)
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "drive": "dexengine.record.driver",
    "bundle": "dexengine.record.bundle",
    "merge": "dexengine.record.store.merge",
    "scene-layout": "dexengine.record.store.scene_layout",
    "scene-manifest": "dexengine.record.store.scene_manifest",
    "look-mirror": "dexengine.record.looks.look_d",
    "assets": "dexengine.record.looks.augvis.assets",
    "text": "dexengine.record.text.pipeline",
}
NEEDS_ENV = {"drive": ("DEXENGINE_MAIN_ROOT", "DEXENGINE_RL_QUEUE_ROOT")}


def _scene_sets(args: list[str]) -> tuple[list[str], list[str]]:
    """(arguments for the command, `scene.<key>=<value>` texts of `--set scene.…` — removed, refused by main)."""
    rest, sets, i = [], [], 0
    while i < len(args):
        a = args[i]
        if a == "--set" and i + 1 < len(args) and args[i + 1].startswith("scene."):
            sets.append(args[i + 1])
            i += 2
            continue
        if a.startswith("--set=scene."):
            sets.append(a[len("--set=") :])
        else:
            rest.append(a)
        i += 1
    return rest, sets


def main() -> None:
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        return
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    sys.path.insert(0, str(ROOT))
    from dexengine.core import config

    args = config.take_data_root(sys.argv[2:])  # before the environment check: it sets DEXENGINE_MAIN_ROOT
    wants_help = any(x in ("-h", "--help") for x in args)
    if not wants_help and (missing := [k for k in NEEDS_ENV.get(cmd, ()) if not os.environ.get(k)]):
        raise SystemExit(f"[stage4_record] {cmd}: set {' and '.join(missing)}")
    os.chdir(ROOT)
    rest, scene_sets = _scene_sets(args)
    if scene_sets:
        raise SystemExit(
            f"[stage4_record] {scene_sets}: the scene section is removed — one desk (Willow from the look "
            "mirror, dexengine.record.looks.look_assets) for every scene"
        )
    sys.argv = [f"{COMMANDS[cmd]}", *rest]
    runpy.run_module(COMMANDS[cmd], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
