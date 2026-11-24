"""S-float-settle at scene build: the data statement of every loader clip (data/objects/float_settle.statement) becomes
the env cfg statement `cfg.float_settle` = {side: [first-touch frame of that side's settle object per loader clip,
-1 = not lowered]}, read by the training pin (rl/tasks/scene/float_lower). Train, eval and render build their scene
through env_setup.supports.build_scene_supports, which calls this, so the three read one statement.

A clip whose statement is `drop` raises: the exclusion step drops it (float_settle.dropped), so reaching the env means
the clip list skipped that step. Marker `[FLOAT_SETTLE] <ctx>: clips k · lowered n · max x mm · t s`.
"""

from __future__ import annotations

import functools
import json
import os
import time

SIDES = ("right", "left")


def clip_statement(st: dict) -> tuple[dict, list]:
    """({side: first-touch frame or -1}, [gap m of each lowered object]) of one clip's statement; a drop verdict
    raises, and so does a support verdict (the tree's disc stamp has no support under that object yet: once the stamp
    holds it, the object rests on its own disc and the statement reads rest). A carried object is not lowered. A
    shared object (side both) is lowered for both sides' pins."""
    out, gaps = dict.fromkeys(SIDES, -1), []
    for name, r in st.items():
        if r["verdict"] == "drop":
            raise RuntimeError(
                f"[FLOAT_SETTLE] {name}: drop verdict (gap {r['gap_m']} m) reached the env — the exclusion step drops "
                "this clip (float_settle.dropped)"
            )
        if r["verdict"] == "stale_base":
            raise RuntimeError(
                f"[FLOAT_SETTLE] {name}: obj_id {r['ids'][0]} differs from its base's {r['ids'][1]} — the base was "
                "rebuilt after placement; place the tree again (float_settle.regenerate)"
            )
        if r["verdict"] == "support":
            raise RuntimeError(
                f"[FLOAT_SETTLE] {name}: support verdict (gap {r['gap_m']} m, still before its first touch) and no "
                "support under it in the disc stamp — re-stamp the tree (coords.discs writes the float supports)"
            )
        if r["verdict"] != "settle":
            continue
        for side in SIDES if r["side"] in ("both", "") else (r["side"],):
            out[side] = int(r["first_touch"])
        gaps.append(float(r["gap_m"]))
    return out, gaps


def build_float_settle(cfg, args_cli, ctx_label) -> None:
    """Read the statement of every loader clip (args_cli.sequence under args_cli.sequence_dir) into cfg.float_settle
    (module doc)."""
    import numpy as np  # noqa: PLC0415

    from dexengine.core import stdio  # noqa: PLC0415
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415 — the clip rows the statement may read
    from dexengine.data.objects import float_settle as FS  # noqa: PLC0415 — owner of the statement
    from dexengine.rl.tasks.scene import float_lower as FL  # noqa: PLC0415 — the reader's cfg key

    seqs = [t for t in (args_cli.sequence or "").split(",") if t]
    seq_dir = getattr(args_cli, "sequence_dir", None)
    if seqs and not seq_dir:
        raise RuntimeError("[FLOAT_SETTLE] the tree folder (args_cli.sequence_dir) is unknown — no statement")
    t0 = time.time()
    by_side, gaps = {s: [] for s in SIDES}, []
    for q in seqs:
        tree = os.path.join(str(seq_dir), q)
        with open(os.path.join(tree, "task_info.json"), encoding="utf-8") as f:
            ti = json.load(f)
        qpos = np.load(os.path.join(tree, "0", "trajectory_kinematic.npz"))["qpos"]
        with stdio.capture():
            rows_of = functools.partial(XS.clip_rows, tree, ti, qpos, strict=False)  # read only if a support is sought
            row, g = clip_statement(FS.statement(tree, ti, qpos, rows_of))
        for s in SIDES:
            by_side[s].append(row[s])
        gaps += g
    setattr(cfg, FL.CFG_KEY, by_side)
    print(
        f"{FS.MARK} {ctx_label}: clips {len(seqs)} · lowered {len(gaps)} · max {max(gaps, default=0.0) * 1000:.1f} mm "
        f"· t {time.time() - t0:.1f} s",
        flush=True,
    )
