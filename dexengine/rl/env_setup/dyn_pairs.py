"""S-dyn-pairs at scene build: whether each loader clip's two main objects are a pair-off pair
(data/objects/dyn_pairs.scene_pair over off_pairs: deeper than the free-pair tolerance at frame 0 on the collision
shapes and not moving together) becomes the cfg statement `cfg.dyn_pair_off` = [bool per loader clip] (key
rl/tasks/scene/collision_dyn_pairs.CFG_KEY), which the pre-parse authoring turns into collision groups. Train, eval and
render build their scene through env_setup.supports.build_scene_supports, which calls this, so the three read one
statement. The drop verdicts (visual overlap, no path) belong to the exclusion step (dyn_pairs.dyn_pairs); the scene
reads the frame-0 pair only (dyn_pairs.scene_pair: a clip with more than one such pair, or a pair that is not its right
and left object, raises — the authoring cuts the right object root against the left one).
Marker (the run's S-dyn-pairs line) `[DYN_PAIRS] <ctx>: clips N · pair_off k · <clip: a×b> … · t s`.
"""

from __future__ import annotations

import json
import os
import time


def build_dyn_pairs(cfg, args_cli, ctx_label) -> None:
    """Read every loader clip (args_cli.sequence under args_cli.sequence_dir) into cfg.dyn_pair_off (module doc)."""
    import numpy as np  # noqa: PLC0415

    from dexengine.core import stdio  # noqa: PLC0415
    from dexengine.data.objects import dyn_pairs as DP  # noqa: PLC0415 — owner of the rule
    from dexengine.rl.tasks.scene import collision_dyn_pairs as DG  # noqa: PLC0415 — the reader's cfg key

    seqs = [t for t in (args_cli.sequence or "").split(",") if t]
    seq_dir = getattr(args_cli, "sequence_dir", None)
    if seqs and not seq_dir:
        raise RuntimeError(f"{DG.MARK} the tree folder (args_cli.sequence_dir) is unknown — no statement")
    t0 = time.time()
    off, cut = [], []
    for q in seqs:
        tree = os.path.join(str(seq_dir), q)
        with open(os.path.join(tree, "task_info.json"), encoding="utf-8") as f:
            ti = json.load(f)
        qpos = np.load(os.path.join(tree, "0", "trajectory_kinematic.npz"))["qpos"]
        with stdio.capture():
            pair = DP.scene_pair(tree, ti, qpos)  # one pair, right × left, or None (raises otherwise)
        off.append(pair is not None)
        cut += [f"{q}: {pair['a']}×{pair['b']}"] if pair else []
    setattr(cfg, DG.CFG_KEY, off)
    print(
        f"{DG.MARK} {ctx_label}: clips {len(seqs)} · pair_off {sum(off)}"
        + "".join(f" · {n}" for n in cut)
        + f" · t {time.time() - t0:.1f} s",
        flush=True,
    )
