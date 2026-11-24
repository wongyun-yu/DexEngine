#!/usr/bin/env python3
"""E5 kinematic baseline — the render and store driver (GPU machines). Trees and IK are built on CPU by prep.py.

One piece (--rows: ds, emb, clip, variants — comma list of passing variant clip names, empty when none), per clip:
  (1) record (no physics): DRYRUN dump (source clip, RENDER_ONLY=replay) → record session stage record_kin (the clip
  and
      its variants in one session — same objects; other clips get their own session: multi-object spawn env_i →
      asset[i % n]) → <out>/stage1/<key>/rollouts.npz (canonical recorder layout, row convention as our data).
  (2) render: render clip.stage2_clip — kinematic cell (cell.kin) = plan kin (no visibility choice, episodes.KIN_RULE),
      orig round (source row 0) and aug round (variant rows 1–3; the source is loaded to fix the camera). Looks = D +
      4 random; render, camera and session rules as the canonical record chain; exo camera on.
  (3) store: per round store.store (LeRobot v2.1, the layout of our final root) → <out>/rounds/<round>/dataset. Episode
      selection carries e5_kin (placement, reach, move table, variant draw) — reach rule owner reach.py.
Resume: record and render follow the record cache rules (same key, render fingerprint, plan sha = reuse) — call the
same command again after preemption.
Markers [KIN_COLLECT] …, last line [KIN_COLLECT] DONE clips n/m · episodes orig a · aug b.
  python experiments/policies/kin_data/collect.py --rows <tsv> --root <kinematic data root> --out <work dir> --gpu <n>
      [--tree <code tree>] [--queue-root <folder>] [--rounds orig,aug]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dexengine.record import hooks as record_hooks

# record hooks; plugins the caller named (the vkclamp layer's env line) stay loaded
record_hooks.add_plugin("experiments.policies.kin_data.plugin")


MARK = "[KIN_COLLECT]"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repo root
ARM, HAND = "rby1", "wuji2"


def cell_of(a, ds: str, emb: str) -> dict:
    """Kinematic cell — no checkpoint; env = the axis tokens of a record render cell (arm, hand, dataset, emb, side,
    extra
    slots, variant suffix)."""
    from dexengine.core import paths as P
    from experiments.policies.kin_data import offsets as KO

    fl = KO.ds_flags(ds)
    combo = f"{ARM}_{HAND}{fl['variant']}"
    side = "both" if emb == "bimanual" else emb
    env = [
        f"DEXENGINE_ARM={ARM}",
        f"DEXENGINE_HAND={HAND}",
        f"DEXENGINE_DATASET={ds}",
        f"DEXENGINE_EMBODIMENT={emb}",
        f"SIDE={side}",
        "DEXENGINE_EXTRA_SLOTS=1",
    ] + ([f"DEXENGINE_COMBO_VARIANT={fl['variant']}"] if fl["variant"] else [])
    seq_dir = P.trees_dir(combo, ds, emb)
    ref = {
        "campaign": "e5_kin",
        "bundle": f"{ds}.{emb}",
        "run": "e5_kin",
        "ckpt": f"kin:{ds}:{emb}",
        "iter": 0,
        "label": "E5 kinematic replay (no policy)",
        "ckpt_sha256": "",
    }
    return {"kin": True, "env": env, "seq_dir": str(seq_dir), "ckpt": ref["ckpt"], "combo": combo, "ref": ref}


def _key(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16]


def _record_session(a, cell: dict, dump: Path, names: Path, npz: Path, d: Path) -> None:
    """Run the record session stage record_kin on the replay dump; writes npz (session log in d)."""
    from dexengine.record.session import launch as CLA

    CLA.session(
        a.tree,
        a.gpu,
        [
            "--stage",
            "record_kin",
            "--dump",
            str(dump),
            "--seq-dir",
            cell["seq_dir"],
            "--seq-list",
            str(names),
            "--out",
            str(npz),
        ],
        d / "session.log",
        "[COLLECT_KIN] DONE",
    )


def record(a, cell: dict, trees: list[str]) -> Path:
    """One clip + its variants → kinematic record (one session). Cache = rollouts.npz + record.json ok (same tree
    fingerprint).
    """
    from dexengine.record import bundle as C
    from dexengine.record import tree_ref as TR

    key = _key([cell["ckpt"], *trees, *[TR.tree_sha(Path(cell["seq_dir"]), TR._base(t)) for t in trees]])
    d = a.out / "stage1" / key
    npz, rec = d / "rollouts.npz", d / "record.json"
    if rec.is_file() and json.loads(rec.read_text()).get("ok") and npz.is_file():
        print(f"{MARK} record exists {trees[0]} · {key}", flush=True)
        return npz
    d.mkdir(parents=True, exist_ok=True)
    seq1 = d / "seq"
    lsts = [TR.resolve(TR.write(seq1 / t, Path(cell["seq_dir"]), t))[1] for t in trees]
    names = d / "members.txt"
    names.write_text("".join(Path(f).read_text() for f in lsts))
    alias = f"kinrec_{key}"
    tok = C.env_set(
        cell["env"],
        SEQ_DIR=cell["seq_dir"],
        DEXENGINE_SEQ_LIST=lsts[0],
        GPU=a.gpu,
        DEXENGINE_MAIN_ROOT=a.main,
        DEXENGINE_RL_QUEUE_ROOT=a.queue_root,
        RENDER_ONLY="replay",
    )
    C.dryrun(a.tree, tok, alias, d / "dryrun.out")
    dump = C.ref_dir(a.tree, a.main) / f"{alias}_{trees[0]}_replay_render.log"
    if not dump.is_file():
        raise SystemExit(f"{MARK} {trees[0]}: no replay dump {dump}")
    _record_session(a, cell, dump, names, npz, d)
    rec.write_text(json.dumps({"ok": npz.is_file(), "trees": trees, "key": key, "dump": str(dump)}, indent=1))
    return npz


def kin_meta(cell: dict, tree: str) -> dict:
    """e5_kin of the episode selection — the tree's task_info.e5_kin (placement, reach, move table fingerprint written
    by prep.py) as is.
    """
    ti = json.loads((Path(cell["seq_dir"]) / tree / "task_info.json").read_text())
    k = ti.get("e5_kin")
    if not k or "reach" not in k:
        raise SystemExit(f"{MARK} {tree}: task_info.e5_kin.reach missing — tree made by a different kin_prep version")
    return k


def members_of(a, cell: dict, npz: Path, trees: list[str]) -> list[dict]:
    """stage2_clip members — [source, variants …] items (traj, clip, cell, task_info, selection)."""
    from dexengine.record.store import meta as CM
    from dexengine.record.store import store as ST

    TF, _TS = ST.eval_modules()
    tf = TF.load(str(npz))
    out = []
    for k, t in enumerate(trees):
        ek = kin_meta(cell, t)
        sel = dict(CM.selection_of(tf, t), rows=None, level=1, e5_kin=ek)
        if k:
            v = ek.get("variant") or {}
            sel.update(
                source_clip=trees[0],
                variant=t,
                k=v.get("k"),
                dx=v.get("dx_scene"),
                dy=v.get("dy_scene"),
                yaw=v.get("yaw_deg"),
                pick_rule="e5_kin_empirical_draw",
                pick_rank=k,
            )
        out.append(
            {
                "traj": str(npz),
                "clip": t,
                "render_dir": None,
                "cell": cell["ref"],
                "task_info": str(Path(cell["seq_dir"]) / t / "task_info.json"),
                "selection": sel,
            }
        )
    return out


def run_start(a) -> None:
    """Print where the dexengine package is imported from (find_spec) and the code commit; die if the root, code or work
    folder holds any symlink pointing outside those three (links are allowed only into one of them, on any machine)."""
    import importlib.util as ilu
    import subprocess

    sp = ilu.find_spec("dexengine")  # located, not executed
    if sp is None or not sp.origin:
        raise SystemExit(f"{MARK} ✗ dexengine is not importable (PYTHONPATH)")
    pkg = Path(sp.origin).resolve().parent
    if not pkg.is_relative_to(a.tree.resolve()):
        raise SystemExit(f"{MARK} ✗ the imported dexengine {pkg} is outside the code tree {a.tree}")
    head = (
        (a.tree / "COMMIT").read_text().strip()
        if (a.tree / "COMMIT").is_file()
        else subprocess.run(
            ["git", "-C", str(a.tree), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    )
    bad = []
    for top in (a.root, a.tree, a.out):
        for dp, dns, fns in os.walk(top):
            for n in dns + fns:
                p = Path(dp) / n
                if p.is_symlink() and not any(
                    Path(os.path.realpath(p)).is_relative_to(t.resolve()) for t in (a.root, a.tree, a.out)
                ):
                    bad.append(f"{p} -> {os.path.realpath(p)}")
    if bad:
        raise SystemExit(f"{MARK} ✗ outside symlinks {len(bad)} — {bad[:3]}")
    print(f"{MARK} [IMPORT] dexengine={pkg} · commit={head} · outside symlinks 0 (root · code · work)", flush=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--root", type=Path, required=True, help="kinematic data root (must equal DEXENGINE_MAIN_ROOT)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--tree", type=Path, default=ROOT)
    ap.add_argument(
        "--queue-root",
        default="",
        help="DEXENGINE_RL_QUEUE_ROOT (queue root the launcher reads — an empty location for kinematic data)",
    )
    ap.add_argument("--rounds", default="orig,aug")
    return ap.parse_args()


def _prepare_work(a) -> None:
    """Check the data root against DEXENGINE_MAIN_ROOT and create the work and queue folders."""
    from dexengine.core import paths as P

    a.main = a.root.resolve()
    if Path(os.environ.get("DEXENGINE_MAIN_ROOT", "")).resolve() != a.main or P.main_root().resolve() != a.main:
        raise SystemExit(f"{MARK} DEXENGINE_MAIN_ROOT differs from --root ({os.environ.get('DEXENGINE_MAIN_ROOT')})")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    a.queue_root = a.queue_root or str(a.out / "_queue_root")
    Path(a.queue_root).mkdir(parents=True, exist_ok=True)
    a.part, a.clips, a.orig_fov_out = "", "", ""


def _rounds_and_mode(a) -> tuple[list[str], bool]:
    """Validated render rounds and the dense per-variant mode (with its applied marker)."""
    from dexengine.record import episodes as EPI
    from dexengine.record.render import cameras as CC
    from experiments.policies.kin_data import piece as KPC

    rounds = [r for r in a.rounds.split(",") if r]
    for r in rounds:
        EPI.check_round(r)
    vo = (
        KPC.var_orig_d()
    )  # dense L1 (default off): one orig look-D episode per variant; the source is only recorded (owner piece.py)
    if vo:
        if rounds != ["orig"] or EPI.looks_mode() != EPI.LOOK_D:
            raise SystemExit(
                f"{MARK} {KPC.VAR_ORIG_ENV}=1 needs the single round orig and DEXENGINE_COLLECT_LOOKS=D"
                f" (got {rounds} · {EPI.looks_mode()!r})"
            )
        print(
            f"{MARK} {KPC.VAR_ORIG_MARK} on — a row's source + variants in one record session · in the "
            "orig round one look D"
            " per variant · the source is not drawn · stored cells clips = drawn variants",
            flush=True,
        )  # applied marker
    print(
        f"{MARK} code {a.tree} · root {a.main} · work {a.out} · GPU {a.gpu} · rounds {rounds} · exo "
        f"{'on' if CC.exo_on() else 'off'} · videos {CC.videos()}",
        flush=True,
    )
    return rounds, vo


def _record_all(a, rows: list[dict]) -> list[tuple]:
    """Stage 1: record every clip of the piece; returns (cell, trees, members) per row."""
    from dexengine.core import paths as P

    staged = []
    for row in rows:
        cell = cell_of(a, row["ds"], row["emb"])
        trees = [P.tree_name(row["clip"])] + [P.tree_name(v) for v in (row.get("variants") or "").split(",") if v]
        npz = record(a, cell, trees)
        staged.append((cell, trees, members_of(a, cell, npz, trees)))
    return staged


def _render_variants(a, cell: dict, trees: list[str], mem: list[dict], rnd: str, eps: dict, t0: float) -> None:
    """Dense mode: one stage2_clip per variant (the source mem[0] is not drawn); episodes appended to eps[rnd]."""
    from dexengine.record.render import clip as CCL
    from experiments.policies.kin_data import piece as KPC

    if len(mem) < 2:
        raise SystemExit(f"{MARK} {trees[0]}: {KPC.VAR_ORIG_MARK} is on but the row has 0 variants")
    a.round = rnd
    for mm in mem[1:]:
        got, why = CCL.stage2_clip(a, cell, mm["clip"], [mm], rnd)
        if why:
            raise SystemExit(f"{MARK} {mm['clip']} {rnd}: nothing to draw ({why}) — kinematic data does not select")
        gt, wt = Counter(g["clip"] for g in got), Counter({mm["clip"]: KPC.looks_per_traj(rnd)})
        if gt != wt:
            raise SystemExit(f"{MARK} {mm['clip']} {rnd}: drawn {dict(gt)} ≠ requested {dict(wt)}")
        eps[rnd] += got
        print(f"{MARK} {mm['clip']} {rnd}: episodes {len(got)} · {time.time() - t0:.0f} s", flush=True)


def _render_round(a, cell: dict, trees: list[str], mem: list[dict], rnd: str, eps: dict) -> None:
    """One round of one clip (orig = source only, aug = source + variants); episodes appended to eps[rnd]."""
    from dexengine.record.render import clip as CCL
    from experiments.policies.kin_data import piece as KPC

    m = mem[:1] if rnd == "orig" else mem
    if rnd == "aug" and len(m) == 1:
        print(f"{MARK} {trees[0]}: 0 passing variants — no aug round", flush=True)
        return
    a.round = rnd
    got, why = CCL.stage2_clip(a, cell, trees[0], m, rnd)
    if why:
        raise SystemExit(f"{MARK} {trees[0]} {rnd}: nothing to draw ({why}) — kinematic data does not select")
    # look count per received trajectory (no silent loss) · look count owner = piece.py
    gt = Counter(g["clip"] for g in got)
    wt = Counter({m_["clip"]: KPC.looks_per_traj(rnd) for m_ in (m if rnd == "orig" else m[1:])})
    if gt != wt:
        raise SystemExit(
            f"{MARK} {trees[0]} {rnd}: drawn trajectories · looks {dict(gt)} ≠ requested {dict(wt)} — "
            "kinematic data does not select"
        )
    eps[rnd] += got
    print(f"{MARK} {trees[0]} {rnd}: episodes {len(got)}", flush=True)


def _store_rounds(a, rows: list[dict], rounds: list[str], eps: dict, vo: bool) -> None:
    """Write each non-empty round's episodes as one dataset with its source record."""
    from dexengine.core import paths as P
    from dexengine.record import episodes as EPI
    from dexengine.record import piece as PC
    from dexengine.record.render import cameras as CC
    from dexengine.record.store import store as ST

    for rnd in rounds:
        if not eps[rnd]:
            continue
        rd = PC.round_dir(a.out, rnd, a.part)
        source = {
            "kind": "e5_kin",
            "rule": EPI.KIN_RULE,
            "round": rnd,
            "robot_type": f"{ARM}_{HAND}",
            "exo": CC.exo_on(),
            "videos": list(CC.videos()),
            "code_tree": str(a.tree),
            "data_root": str(a.main),
            "rows": a.rows,
            # source clip set (canonical merge check) · per variant (vo) = the drawn variants
            "cells": [
                {
                    "kind": "e5_kin",
                    "clips": ([g["clip"] for g in eps[rnd]] if vo else [P.tree_name(r["clip"]) for r in rows]),
                }
            ],
            "episodes_plan": {
                "version": EPI.PLAN_VERSION,
                "look_d": EPI.LOOK_D,
                "round": rnd,
                "random_per_traj": EPI.RANDOM_PER_TRAJ,
            },
        }
        ST.store(rd / "dataset", eps[rnd], source)


def main() -> int:
    a = _parse_args()
    run_start(a)
    _prepare_work(a)
    rounds, vo = _rounds_and_mode(a)
    with open(a.rows) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    t0 = time.time()
    eps: dict[str, list] = {r: [] for r in rounds}
    done = 0
    # Stage 1 (record) runs for every clip of the piece before stage 2 (render) — the canonical stage order.
    # Alternating per clip leaves the render worker (Kit, 10→16 GB) holding the card while the next clip's record_kin
    # Kit starts on it, which overflows a 24 GB card (VkResult ERROR_OUT_OF_DEVICE_MEMORY →
    # CUBLAS_STATUS_ALLOC_FAILED).
    staged = _record_all(a, rows)
    print(
        f"[KIN_ORDER] stage1_all_then_render n={len(staged)} · stage 1 {time.time() - t0:.0f} s — now rendering",
        flush=True,
    )
    for cell, trees, mem in staged:
        for rnd in rounds:
            if vo:  # one orig-round stage2_clip per variant (as the dense render) — the source (mem[0]) is not drawn
                _render_variants(a, cell, trees, mem, rnd, eps, t0)
                continue
            _render_round(a, cell, trees, mem, rnd, eps)
        done += 1
        print(f"{MARK} clips {done}/{len(rows)} · {time.time() - t0:.0f} s", flush=True)
    _store_rounds(a, rows, rounds, eps, vo)
    print(
        f"{MARK} DONE clips {done}/{len(rows)} · episodes "
        + " · ".join(f"{r} {len(eps[r])}" for r in rounds)
        + f" · {time.time() - t0:.0f} s",
        flush=True,
    )
    return 0 if done == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
