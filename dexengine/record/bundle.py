#!/usr/bin/env python3
"""Entry of the public-data record pipeline (CPU coordination, retarget venv): a record-only path kept apart from the
inference and eval code.

  input  one of two:
        cell   a finished RL cell (--hand --campaign --bundle): the judge record's best checkpoint
               (dexengine.rl.cell_session._target, owner of the render target rule) and the per-clip [completed,
               evaluated] of its final judge (dexengine.rl.judge.metrics.CLIPS) -> clips with completion rate >=
               --min-rate (default 5
               %: try at 5 % or more, secure at least one at 10 % or more). With --clips, only among those.
        owner  --owner <rl_clip_owner.tsv> --clips <names,…>: per clip its owning checkpoint
               (dexengine.rl.acceptance.use_rows; a stale ledger or a missing checkpoint dies; entry record.sources:
               --owner-sha != the received file, missing, or an --owner-pin copy with a different owner dies; marker
               [CKPT_OWNER]). Bundles may differ; stage 1 runs per checkpoint. The completion rate is the clip record
               of that checkpoint's judge (ledger kind, iter).
  stage-1 record  the record session (session.entry --stage record) relaunches the same frozen entry as judging
        (dump = FROZEN_REPLAY_CKPT … DEXENGINE_TRAJ_RECORD): parallel rollouts per clip
        ROLLOUT_LEVELS[0]=64 (NE = 64 x clips, evaluations 64, warm-up 1 + real batch 1), low-dimensional only
        (rollout.recorder: state, reference, policy output, absolute joint targets, contact forces). Clips with 0
        successes are measured again at the next level (128 -> 256), only those; 0 at 256 too = not collected (kept in
        the table).
  choice  among successes (same definition as judging) the one with the smallest end error
        (rollout.recorder.SELECT_BY); only that trajectory is kept (K_TOP) · the top ranking table comes from the
        all-episode summary (rollout.recorder.selection_table).
  stage-2 render and store  per clip (render.clip_all.render_all: one session per clip; original + placement variants,
        look plan) · written without physics and captured on top of the canonical kinematic replay dump
        (RENDER_ONLY=replay). The static object float rule and the object texture cover ([RENDER_OBJ_USD]) are
        already applied by the canonical render entry (the dump keeps that entry's argv/env). Store = store.store
        (LeRobot v2.1 form + depth and forces) · preview store.preview · place = piece.round_dir · loader field =
        store.meta.loader_meta · meaning of --round = record.episodes.
Cache keys (stage-1 reuse, stage-2 render fingerprint) and code fingerprint rules = record.cache.
Writes only under --out (data root, cells and judge records are read only). Switches and placement augment =
record.chain (default on).
Record only (--until record: physics-only machines, no render or store): runs stage 1 (64 -> 128 -> 256) and the
  placement augment physics (generation, rollout, post gate, choice, chosen trajectory files), writes the round place
  record.json (targets, trajectories, exclusions) and ends. Render and store: calling the same command again without
  --until picks up the caches (stage-1 key, augment run.json, post.json); no second path. When moving, the augment
  shadow root is packed with the shadow packer's list.
Render only (--until render: the record driver's GPU part with --cpu-tail on): through the render, then the end line
  [COLLECT] RENDER DONE. The same command without --until and with --gpu none (CPU_ONLY) picks up every cache (stage 1,
  augment, render present) and stores; a CPU-only call runs its children with no GPU visible (NO_GPU_ENV: no CUDA
  device, no Vulkan or EGL driver), so a cache miss there dies at its session's start instead of using a GPU.
Several bundles: record.driver.
Markers: [COLLECT] … · [COLLECT_CACHE] … · default store end line [COLLECT] DONE episodes n/n · record-only end line
[COLLECT] RECORD DONE · augment lines = record.chain.
Exit codes: 0 · RC_PARTIAL (some clips excluded) · RC_AUG_FAILED (augment failed; wins over partial) · others = the
base stage failed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dexengine.core import config
from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.record import chain as CANON
from dexengine.record import tree_ref as TR  # clip folder tree record and fingerprint check
from dexengine.record.bundle_cell import cell_of, cells_arm, owner_cells
from dexengine.record.cache import (
    cache_key,
    cache_line,
    code_fingerprint,
    env_fingerprint,
    stage1_done,
    stage1_hit,
)
from dexengine.record.render import episode as CRN  # render frame rate for the loader meta

# owner of session runs, failure classes and the one boot-crash retry
from dexengine.record.session import launch as CLA
from dexengine.record.store import meta as CM
from dexengine.record.store import store as STORE
from dexengine.rl import app as TK
from dexengine.rl import cell_session as CR  # owner of the cell and render target (best) rules

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

ROLLOUT_LEVELS = (64, 128, 256)
# how far: store = to the end (default) · render = through the render (no store) · record = through physics only
UNTIL = ("store", "render", "record")
CPU_ONLY = "none"  # --gpu value of a call that must not use a GPU (the driver's store call after --until render)
_NO_DRIVER = "/nonexistent/no-gpu.json"
NO_GPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "VK_ICD_FILENAMES": _NO_DRIVER,
    "VK_DRIVER_FILES": _NO_DRIVER,
    "__EGL_VENDOR_LIBRARY_FILENAMES": _NO_DRIVER,
}
MIN_RATE = 0.05
MARK = "[COLLECT]"
DONE_LINE = f"{MARK} DONE"  # head of the store end line; the driver (record.driver) reads this value
# head of the record-only end line; read when the driver passed --until record
RECORD_DONE_LINE = f"{MARK} RECORD DONE"
RENDER_DONE_LINE = f"{MARK} RENDER DONE"  # head of the render-only end line (--until render)
RC_PARTIAL = 1  # some clips excluded (reasons record.chain base_exclusions); base store and augment finished
RC_AUG_FAILED = 3  # placement augment failed (marker record.chain AUG_FAIL): the bundle fails


def pick_clips(rates: dict, min_rate: float, names: list[str] | None) -> tuple[list[str], list[dict]]:
    """Completion rate table -> (chosen tree names, the whole table; row target = this cell's targets). A name missing
    from the table dies."""
    if names:
        tn = [P.tree_name(n) for n in names]
        miss = [n for n in tn if n not in rates]
        if miss:
            raise InputMissingError(f"{MARK} --clips {miss} are not in the judge record")
        cand = tn
    else:
        cand = sorted(rates)
    table = [
        {"clip": c, "completed": v[0], "evals": v[1], "rate": v[0] / v[1] if v[1] else 0.0, "target": c in cand}
        for c, v in rates.items()
    ]
    keep = [c for c in cand if rates[c][1] and rates[c][0] / rates[c][1] >= min_rate]
    return keep, table


def env_set(tokens: list[str], **kv) -> list[str]:
    keep = [t for t in tokens if t.split("=", 1)[0] not in kv]
    return keep + [f"{k}={v}" for k, v in kv.items() if v is not None]


def ref_dir(tree: Path, main: Path | None = None) -> Path:
    """Where the canonical launcher writes the dump: the dry log folder of the launch's run tree (owner
    dexengine.rl.launch.records.log_dir · rule paths.run_root_of(`main` = the DEXENGINE_MAIN_ROOT the launch gets,
    `tree`)). main None = the data root a command in `tree` reads (paths.data_root_of — the old one-argument form)."""
    from dexengine.rl.launch import records  # noqa: PLC0415

    return records.log_dir(True, P.run_root_of(main if main is not None else P.data_root_of(tree), tree))


def dryrun(tree: Path, tokens: list[str], alias: str, log: Path, launch_args=()) -> None:
    """Dump launch of the canonical entry (DUMP_ONLY=1: the trainer argv + env dumps a record session replays, no
    preflight, nothing started); `launch_args` follow the run name (e.g. `--set field=value`)."""
    cmd = ["env", *tokens, "DUMP_ONLY=1", "bash", "scripts/stage3_rl.sh", alias, *launch_args]
    with open(log, "w") as f:
        f.write("# " + shlex.join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, cwd=tree, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    if r.returncode:
        raise SystemExit(f"{MARK} dump launch (DUMP_ONLY) failed rc {r.returncode}: {log}")


def stage1(a, cell: dict, clips: list[str], level: int) -> Path:
    """One clip set and rollout count -> rollouts.npz (cache)."""
    # True = contact-partner decomposition (fixed on)
    key = cache_key(clips, cell["ckpt_sha"], level, code_fingerprint(a.tree), True)
    d = a.out / "stage1" / key
    rec = d / "record.json"
    # record sha check (record.cache: written once, never changed; one job owns the work place)
    if stage1_hit(d):
        cache_line(rec, key, len(clips), level, a.tree)
        return d / "rollouts.npz"
    cache_line(None, key, len(clips), level, a.tree)
    d.mkdir(parents=True, exist_ok=True)
    # list form = tree name - tree suffix (paths.TREE_SFX)
    (d / "clips.txt").write_text("".join(c.removesuffix(P.TREE_SFX) + "\n" for c in clips))
    alias = f"collect_{key}"
    # collision preset ledger: rule owner dexengine.rl.cell_session.stage_coll_preset
    staged = CR.stage_coll_preset(cell["folder"], a.tree, alias, data_root=a.main)
    print(
        f"{MARK} collision preset ledger {'placed' if staged else 'none (the launcher decides)'} · "
        f"{cell['folder'].name}",
        flush=True,
    )
    tok = env_set(
        cell["env"],
        DEXENGINE_SEQ_LIST=d / "clips.txt",
        GPU=a.gpu,
        DEXENGINE_MAIN_ROOT=a.main,
        DEXENGINE_RL_QUEUE_ROOT=a.queue_root,
        FROZEN_REPLAY_CKPT=cell["ckpt"],
        FROZEN_REPLAY_NE=level * len(clips),
        FROZEN_REPLAY_EVALS_PER_CLIP=level,
        FROZEN_REPLAY_ITER=cell["iter"],
        DEXENGINE_TRAJ_RECORD=d / "rollouts.npz",
    )
    dryrun(a.tree, tok, alias, d / "dryrun.out")
    dump = ref_dir(a.tree, a.main) / f"{alias}_frozenreplay_train.log"
    CLA.session(a.tree, a.gpu, ["--stage", "record", "--dump", str(dump)], d / "session.log", "[TRAJ_REC] DONE")
    TK.require_mark(d / "session.log")
    stage1_done(
        d,
        {
            "ok": (d / "rollouts.npz").is_file(),
            "key": key,
            "clips": clips,
            "rollouts": level,
            "ckpt": str(cell["ckpt"]),
            "ckpt_sha256": cell["ckpt_sha"],
            "code_fp": code_fingerprint(a.tree),
            "dump": str(dump),
            "tree": str(a.tree),
            "env_fp": env_fingerprint(a.tree),
        },
    )
    return d / "rollouts.npz"


def _generation_reason(CU, cell: dict, clip: str, a) -> str | None:
    """The dexengine.rl.checkpoints refusal of one clip for this cell's checkpoint, None when the clip is of its
    generation."""
    try:
        CU.check(cell["ckpt"], cell["seq_dir"], [clip], a.hand, a.queue_root)
    except ValueError as e:
        return str(e)
    return None


def generation_check(cell: dict, clips: list[str], a) -> dict:
    """Checkpoint vs clip generation check (owner dexengine.rl.checkpoints.check; the Isaac entry dies with the same
    function) -> {clip: reason}. Clips whose generation moved (TURNOVER, e.g. placement changed and retraining pending)
    are not collected with this checkpoint (it learned the old tree): one log line per clip with the reason · the cell
    summary generation_dropped (meta/collect.json)."""
    from dexengine.rl import checkpoints as CU

    os.environ.update(DEXENGINE_ARM=cells_arm([cell]), DEXENGINE_MAIN_ROOT=str(a.main))
    bad = {}
    for c in clips:
        why = _generation_reason(CU, cell, c, a)
        if why is not None:
            bad[c] = why
    for c in sorted(bad):
        print(f"{MARK} generation check (dexengine.rl.checkpoints) dropped {c}: {bad[c]}", flush=True)
    if bad:
        print(f"{MARK} generation check (dexengine.rl.checkpoints) dropped {len(bad)}/{len(clips)}", flush=True)
    return bad


def collect_cell(a, cell: dict, names: list[str] | None, items: list, sel: dict, cells: list) -> list[str]:
    """One cell: clip choice, generation check, stage 1 (64 -> 128 -> 256) -> added to items, sel and cells (render is
    render.clip). Returns the record target clips (tree names)."""
    clips, table = pick_clips(cell["rates"], a.min_rate, names)
    gen_bad = generation_check(cell, clips, a)
    if gen_bad and names:
        raise SystemExit(f"{MARK} chosen clips are not of this checkpoint's generation (canon_use): {gen_bad}")
    for row in table:
        row["generation"] = gen_bad.get(row["clip"], "current" if row["clip"] in clips else "not checked")
    clips = [c for c in clips if c not in gen_bad]
    print(
        f"{MARK} {cell['run']} · {cell['label']} iter {cell['iter']} (judge {cell['judge_agg_pct']} %) · "
        f"{cell['ckpt'].name} sha {cell['ckpt_sha'][:12]} · clips {len(clips)}/{len(table)}"
        f" (completion >= {a.min_rate:.0%}) "
        f"· code fingerprint {code_fingerprint(a.tree)} · tree {a.tree}",
        flush=True,
    )
    if not clips:
        raise InputMissingError(f"{MARK} {cell['bundle']}: no clips to choose")
    summary = {
        "campaign": cell["campaign"],
        "bundle": cell["bundle"],
        "run": cell["run"],
        "ckpt": str(cell["ckpt"]),
        "ckpt_sha256": cell["ckpt_sha"],
        "label": cell["label"],
        "iter": cell["iter"],
        "judge_agg_pct": cell["judge_agg_pct"],
        "clip_rates": table,
        "clips": clips,
        "generation_dropped": gen_bad,
    }
    cells.append(summary)
    if a.plan:
        return clips
    got, todo, files = {}, list(clips), {}
    for level in ROLLOUT_LEVELS:
        if not todo:
            break
        npz = stage1(a, cell, todo, level)
        TF, _TS = STORE.eval_modules()
        tf = TR.gate(TF.load(str(npz)), todo)
        for c in list(todo):
            if tf.selected(c) is not None:
                got[c], files[c] = CM.selection_of(tf, c), (npz, level)
                todo.remove(c)
        print(f"{MARK} rollouts {level}: chosen {len(clips) - len(todo)}/{len(clips)} · 0 successes {todo}", flush=True)
    summary["not_collected"] = todo
    cell_ref = {k: summary[k] for k in ("campaign", "bundle", "run", "ckpt", "ckpt_sha256", "iter", "label")}
    for c in clips:
        if c not in got:
            continue
        npz, level = files[c]
        items.append(
            {
                "traj": str(npz),
                "clip": c,
                "render_dir": None,
                "cell": cell_ref,
                "task_info": str(Path(cell["seq_dir"]) / c / "task_info.json"),  # owner of the case meta
                "selection": dict(got[c], rows=None, level=level),
            }
        )
        sel[c] = got[c]
    return clips


def record_done(a, plan: list, items: list, source: dict, want: list[str]) -> int:
    """Record-only end: placement augment physics (when on) -> <out>/record.json and the end line. Exit code rule as the
    store end."""
    from dexengine.record import piece as PC

    ok = CANON.after_store(a, plan, items, source)
    rec = PC.round_dir(a.out, a.round, a.part) / "record.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(
        json.dumps(
            dict(source, until="record", items=items, aug_ok=ok, env_fp=env_fingerprint(a.tree)),
            indent=1,
            ensure_ascii=False,
            default=str,
        )
    )
    print(
        f"{RECORD_DONE_LINE} chosen trajectories {len(items)}/{len(want)} · cells {len(source['cells'])} · augment "
        f"{'done' if ok else 'failed'} · {rec}",
        flush=True,
    )
    if not ok:
        return RC_AUG_FAILED
    return 0 if len(items) == len(want) else RC_PARTIAL


def _parser():
    """Command line of the record bundle (cell or owner-table entry, round, stages, roots)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--campaign", default="", help="cell entry: campaign")
    ap.add_argument("--bundle", default="", help="cell entry: bundle")
    ap.add_argument(
        "--owner",
        type=Path,
        default=None,
        help="owner-ledger entry: rl_clip_owner.tsv (owning checkpoint per clip; needs --clips)",
    )
    ap.add_argument(
        "--owner-sha",
        default=None,
        help="canonical owner ledger sha256 measured at launch (sources canon); a different --owner file or none dies",
    )
    ap.add_argument(
        "--owner-pin",
        type=Path,
        default=None,
        help="owner ledger copy frozen by the driver: clips whose owner differs from --owner die",
    )
    ap.add_argument(
        "--clips",
        default="",
        help="list names (comma separated); on the cell entry, without it every clip over the completion threshold",
    )
    ap.add_argument("--min-rate", type=float, default=MIN_RATE)
    ap.add_argument("--gpu", required=True, help=f"CUDA device index ({CPU_ONLY!r} = no GPU: store from the caches)")
    ap.add_argument(
        "--out", type=Path, required=True, help="output root = work piece (driver: run layer run_collect_work)"
    )
    ap.add_argument("--tree", type=Path, default=ROOT, help="code tree that runs the sessions (default this tree)")
    ap.add_argument("--main", type=Path, default=None, help="data root (default paths.main_root())")
    ap.add_argument("--plan", action="store_true", help="print the chosen clips and checkpoints only, then end")
    ap.add_argument(
        "--until",
        choices=UNTIL,
        default="store",
        help="record = through physics (stage 1, augment physics) only, no render or store · "
        "render = through the render, no store (record driver --cpu-tail on) · store = to the end "
        "(default)",
    )
    ap.add_argument("--round", required=True, choices=CANON.EPI.ROUNDS, help="record round (record.episodes)")
    ap.add_argument(
        "--part",
        default="",
        help="part tail of the round place (piece.part_key; sibling part specs only · none = the round place as it is)",
    )
    ap.add_argument("--orig-fov-out", default="")
    ap.add_argument(
        "--queue-root",
        default=os.environ.get("DEXENGINE_RL_QUEUE_ROOT", ""),
        help="queue root (env DEXENGINE_RL_QUEUE_ROOT; no default; canon_use checks checkpoint vs clip generation)",
    )
    config.add_data_root_arg(ap)
    return ap


def main() -> int:
    ap = _parser()
    a = ap.parse_args()
    config.apply_data_root(a)
    if a.gpu == CPU_ONLY:  # every child of this call sees no GPU (set before anything starts)
        if a.until != "store":
            raise SystemExit(f"{MARK} --gpu {CPU_ONLY} stores from the caches only: --until {a.until} needs a GPU")
        os.environ.update(NO_GPU_ENV)
        print(f"{MARK} CPU-only call: no GPU visible to the sessions ({', '.join(NO_GPU_ENV)})", flush=True)
    if not a.queue_root:
        raise InputMissingError(
            f"{MARK} no queue root (--queue-root, DEXENGINE_RL_QUEUE_ROOT): the entry that loads "
            "cell checkpoints (dexengine.rl.checkpoints) dies · no default"
        )
    names = P.read_clip_list([c for c in a.clips.split(",") if c], label="--clips", quiet=True) or None
    if bool(a.owner) == bool(a.campaign or a.bundle):
        raise SystemExit(f"{MARK} one entry only: --owner (ledger) or --campaign + --bundle (cell)")
    if a.owner and not names:
        raise SystemExit(f"{MARK} the owner-ledger entry needs --clips")
    if not a.owner and not (a.campaign and a.bundle):
        raise SystemExit(f"{MARK} the cell entry needs both --campaign and --bundle")
    a.main = (a.main or P.main_root()).resolve()
    a.out = a.out.resolve()
    rnd = CANON.apply_round(a.round)  # before switches
    switches = CANON.switches()
    plan = owner_cells(a, names) if a.owner else [(cell_of(a, a.campaign, a.bundle), names)]
    if not a.plan:
        a.out.mkdir(parents=True, exist_ok=True)
    if a.until != "record" and not a.plan:  # a stale statement dies before the render
        from dexengine.record.store import slots as CK

        CK.stmt_gate(a.hand, [cell for cell, _ in plan], a.main)
    items, sel, cells, want = [], {}, [], []
    for cell, cn in plan:
        want += collect_cell(a, cell, cn, items, sel, cells)
    if a.plan:
        return 0
    arm = cells_arm([cell for cell, _ in plan])
    source = {
        "hand": a.hand,
        "arm": arm,
        "entry": "owner" if a.owner else "cell",
        "owner": str(a.owner) if a.owner else None,
        "owner_sha": a.owner_sha if a.owner else None,
        "cells": cells,
        "min_rate": a.min_rate,
        "switches": switches,
        "not_collected": [c for c in want if c not in sel],
        "rollout_levels": list(ROLLOUT_LEVELS),
        "augmentation": None,
        "robot_type": P.combo_for(a.hand, arm),
        "code_fp": code_fingerprint(a.tree),
        "tree": str(a.tree),
        # controller text lands in meta/collect.json (data; the same text as experiments/policies/dense_render.py)
        "controller": "PD joint position control — action.joint_target = absolute target (controller setpoint) · "
        "action = policy output",
        "round": a.round,
        "round_switch": rnd,
        "loader": CM.loader_meta(CRN.FPS),
    }
    if a.until == "record":
        return record_done(a, plan, items, source, want)
    from dexengine.record.render import clip_all as CCA

    return CCA.render_all(a, plan, items, sel, source, want, a.round)


if __name__ == "__main__":
    sys.exit(main())
