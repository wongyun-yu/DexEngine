#!/usr/bin/env python3
"""E5 closed-loop evaluation prep (CPU, no GPU): for every start scene, build what the evaluation session reads, with
the canonical record functions.

One scene (one JSONL line of the scene lists — test_episodes or aux_episodes):
  cell        source_cell (campaign · bundle · checkpoint sha) → bundle_cell.cell_of (judge env tokens, checkpoint). The
              checkpoint is that run folder's checkpoints/<name> under this machine's data root; a sha256 that differs
              from the episode record dies.
  trajectory  the trajectory that drew that data episode — source-tree scenes (start_rule orig_D, aug_D): the stage-1
              trajectory the render plan names; placement-variant scenes (row0_variant, aux_variant): the variant's
              augmented-physics trajectory (both from --variant-src, sha checked). No trajectory = death.
  tree        source = the data root's tree; variant = the variant tree and precompute placed in the stage's shadow root
              <stage>/shadow (augment.scene.init_shadow — the same overlay as augmented renders), which becomes the
              scene's
              data root (DEXENGINE_MAIN_ROOT of the dump). A variant tree whose aug_place differs from the episode's
              selection (dx, dy, yaw) dies.
  tree record tree_ref.write(<out>/seq1, tree root, clip) — DEXENGINE_SEQ_LIST of the render entry (as stage2_clip).
  plan        source scenes: episodes.plan (orig, look D only). Variant scenes: the record session plan that actually
  drew
              the episode (--variant-src --draw copy): trajectories [source first, variant], recorded look, look_key,
              deck,
              row/column, allow — the render fixes the head from the source (check_view) and gives the same look spec.
              Dense variant scenes (start_rule dense_D, only with --allow-dense): the dense render's plan copy, one
              trajectory (the variant, role orig, look D); the head is fixed from the variant. Slot = <variant tree>.D.
  dump        canonical entry DUMP_ONLY (launcher RENDER_ONLY=<checkpoint>, same tokens as stage2_clip) → dump.log.
Output: <stage>/clips/<slot>/{job.json, plan.json, dump.log, seq1/} (slot = test: the test clip's tree name; aux:
<variant tree>.<look>) and <stage>/prep.json (per-scene result, failure reason, input sha). Marker [E5_PREP]. Paths
stay absolute; the evaluation machine binds the same paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dexengine.core.hashing import sha256_file
from dexengine.rl.tasks import switches
from experiments.sim_eval.prep_scene import (
    DENSE_RULE,
    MARK,
    build_shadow,
    ckpt_path,
    render_traj,
    scene_plan,
    stage_path,
    traj_file,
)

TREE = Path(__file__).resolve().parents[2]

#: cells in which the evaluation (frozen render) dump may differ from the stage-1 record dump — render, record and
#: render
#: scene env only (not physics). Any other difference dies (physics must be equal). The render USD variant is the
#: repaint.
OK_ENV = {
    "DEXENGINE_FROZEN_REPLAY_EVALS_PER_CLIP",
    "DEXENGINE_FROZEN_REPLAY_ITER",
    "DEXENGINE_ENTRY",
    "DEXENGINE_TRAJ_RECORD",
    "DEXENGINE_CAMERA_FOCAL_MM",
    "DEXENGINE_EXTRA_SLOTS",
    "DEXENGINE_FROZEN_RENDER",
    "DEXENGINE_KEY_LIGHT",
    "DEXENGINE_PHYSX_LITE",
    "DEXENGINE_COMBO_USD_VARIANT",
    "DEXENGINE_SEQ_LIST",
}
OK_ARG = {
    "--enable_cameras",
    "--record_session",
    "--videos_out_dir",
    "--wandb",
    "--wandb_project",
    "--exp_name",
    "--num_envs",
    "--resume_from",
    "--enable_contact_termination",
}


def dump_diff(s1: Path, ev: Path) -> dict:
    """argv flag and env differences of two dumps — anything outside OK_ENV / OK_ARG goes to `unexpected`. The RL
    switches of each dump (its switches file) are compared under their old environment names."""

    def parse(p):
        L = Path(p).read_text().splitlines()
        env = switches.dump_env_view(dict(x[5:].split("=", 1) for x in L if x.startswith("ENV::")))
        argv, out, i = [x for x in L if not x.startswith("ENV::")], {}, 0
        while i < len(argv):
            t = argv[i]
            if t.startswith("--"):
                v = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None
                out.setdefault(t, []).append(v)
                i += 2 if v is not None else 1
            else:
                i += 1
        return out, env

    f1, e1 = parse(s1)
    f2, e2 = parse(ev)
    de = sorted(k for k in set(e1) | set(e2) if e1.get(k) != e2.get(k))
    da = sorted(k for k in set(f1) | set(f2) if f1.get(k) != f2.get(k))
    ce = {k: (e1.get(k), e2.get(k)) for k in de if k in ("DEXENGINE_COMBO_USD_VARIANT", "DEXENGINE_EXTRA_SLOTS")}
    return {
        "env": de,
        "arg": da,
        "checked": ce,
        "unexpected": [k for k in de if k not in OK_ENV] + [k for k in da if k not in OK_ARG],
        "contact_term": (f1.get("--enable_contact_termination"), f2.get("--enable_contact_termination")),
    }


def _fresh_scene_dir(a, stage, sub, data_root, seq_root, tname):
    """Recreate the scene folder and write its tree record (seq1) under the scene data root; returns (out, seq list)."""
    from dexengine.record import tree_ref as TR

    out = stage_path(a, stage / "clips" / sub)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    os.environ["DEXENGINE_MAIN_ROOT"] = str(
        data_root
    )  # the tree record (rel_of) reads the scene's data root — restored at the end
    try:
        _seq, lst = TR.resolve(TR.write(out / "seq1", seq_root, tname))
    finally:
        os.environ["DEXENGINE_MAIN_ROOT"] = str(a.main)
    return out, lst


def _prep_summary(sub, tname, base, ds, job, row, v):
    """The prep.json line of one scene."""
    return {
        "sub": sub,
        "clip": tname,
        "base": base,
        "dataset": ds,
        "cell": job["cell"],
        "start_rule": row.get("start_rule"),
        "look": (v or {}).get("look", "D"),
        "source": row["split_row"]["clip"],
    }


def prep_clip(a, row: dict, stage: Path) -> dict:
    from dexengine.core import paths as P
    from dexengine.record import bundle_cell as BC

    TF = traj_file()
    sc = row["source_cell"] or {}
    ds, base = row["dataset"], row["base"]
    v = row.get("variant")
    ck = ckpt_path(a.main, a.hand, ds, sc)
    cell = BC.cell_of(a, sc["campaign"], sc["bundle"], ck, int(sc["iter"]))
    if v:
        V = a.variants[v["name"]]
        tname, seq_root, traj, data_root = v["name"], V["seq_root"], V["traj"], a.shadow
        place = V["placement"]
        ck_orig, ck = ck, V["ckpt"]  # out-of-cell copy (build_shadow, augment run.cell_spec) — sha256 = source
    else:
        tname, seq_root, data_root = P.tree_name(base), Path(cell["seq_dir"]), a.main
        traj = render_traj(a, row)
    sub = row["key"]
    tf = TF.load(str(traj))
    if tf.selected(tname) is None:
        raise SystemExit(f"{MARK} {tname}: trajectory file {traj} has no selected trajectory")
    out, lst = _fresh_scene_dir(a, stage, sub, data_root, seq_root, tname)
    plan, lst = scene_plan(a, row, v, tname, traj, seq_root, data_root, out, lst, TF)
    (out / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    alias = _eval_dump(a, cell, out, sub, tname, seq_root, lst, data_root, ck)
    phys = _stage1_physics(a, cell, out, tname, alias, seq_root, data_root, ck, v)
    job = {
        "clip": tname,
        "base": base,
        "dataset": ds,
        "axis": row["split_row"]["axis"],
        "hand_use": row["split_row"]["hand_use"],
        "episode": row,
        "cell": dict(
            campaign=cell["campaign"],
            bundle=cell["bundle"],
            run=cell["run"],
            ckpt=str(ck),
            ckpt_sha256=cell["ckpt_sha"],
            iter=cell["iter"],
            **({"ckpt_orig": str(ck_orig)} if v else {}),
        ),
        "setup_traj": str(traj),
        "seq_dir": str(seq_root),
        "seq_list": str(lst),
        "plan": str(out / "plan.json"),
        "dump": str(out / "dump.log"),
        "physics_vs_stage1": phys,
        "tree_commit": os.popen(f"git -C {TREE} rev-parse HEAD").read().strip(),
    }
    if v:
        job["variant"] = {
            "placement": place,
            "look": v["look"],
            "name": v["name"],
            "k": v["k"],
            "start_rule": row["start_rule"],
            "frame": "scene (+x in front of the robot · +y left) · episode selection record = checked against the "
            "tree's aug_place",
        }
    (out / "job.json").write_text(json.dumps(job, indent=1, ensure_ascii=False))
    return _prep_summary(sub, tname, base, ds, job, row, v)


def _eval_dump(a, cell: dict, out: Path, sub: str, tname: str, seq_root, lst, data_root, ck: Path) -> str:
    """prep_clip step 3 — the evaluation dump: canonical frozen render DUMP_ONLY → <out>/dump.log; returns the alias."""
    from dexengine.record import bundle as C
    from dexengine.record.render import clip as CCL

    alias = f"e5eval_{hashlib.sha256(sub.encode()).hexdigest()[:12]}"
    slots = CCL.extra_slots_of(cell)
    base_tok = [t for t in cell["env"] if not t.startswith(("DEXENGINE_SEQ_LIST=", "SEQ_DIR="))]
    # evaluation dump = the canonical frozen render point (DEXENGINE_FROZEN_REPLAY=1 ∧ DEXENGINE_FROZEN_RENDER=1 →
    # the trainer's render seam, dexengine.rl.session_render), argv as the judge and stage 1. Bimanual = RL launcher
    # SIDE=both RENDER_ONLY=<checkpoint> (RENDER_RUNDIR = the run folder); one hand = its judge mode
    # FROZEN_REPLAY_CKPT + FROZEN_RENDER_OUT (its frozen render entry).
    side = next((t.split("=", 1)[1] for t in cell["env"] if t.startswith("SIDE=")), None)
    if side not in ("both", "right", "left"):
        raise SystemExit(f"{MARK} {tname}: cell env SIDE={side!r} — both|right|left")
    if side == "both":
        tok = C.env_set(
            base_tok,
            SEQ_DIR=seq_root,
            DEXENGINE_SEQ_LIST=lst,
            GPU="0",
            DEXENGINE_MAIN_ROOT=data_root,
            DEXENGINE_RL_QUEUE_ROOT=a.queue_root,
            RENDER_ONLY=ck.name,
            RENDER_RUNDIR=ck.parent.parent,
            DEXENGINE_EXTRA_SLOTS=slots,
        )
        dump_name = f"{alias}_vid_{ck.stem}_B_train.log"
    else:
        tok = C.env_set(
            base_tok,
            SEQ_DIR=seq_root,
            DEXENGINE_SEQ_LIST=lst,
            GPU="0",
            DEXENGINE_MAIN_ROOT=data_root,
            DEXENGINE_RL_QUEUE_ROOT=a.queue_root,
            FROZEN_REPLAY_CKPT=ck,
            FROZEN_REPLAY_NE=1,
            FROZEN_REPLAY_ITER=cell["iter"],
            FROZEN_RENDER_OUT=out / "_videos",
            DEXENGINE_EXTRA_SLOTS=slots,
        )
        dump_name = f"{alias}_frozenreplay_train.log"
    C.dryrun(TREE, tok, alias, out / "dryrun.out")
    dump = C.ref_dir(TREE, data_root) / dump_name
    if not dump.is_file():
        raise SystemExit(f"{MARK} {tname}: no dump {dump}")
    shutil.copy2(dump, out / "dump.log")
    return alias


def _stage1_physics(a, cell: dict, out: Path, tname: str, alias: str, seq_root, data_root, ck: Path, v) -> dict:
    """prep_clip step 4 — physics-equality evidence: the stage-1 record dump diffed against <out>/dump.log."""
    from dexengine.core import paths as P
    from dexengine.record import bundle as C

    # physics-equality evidence — DUMP_ONLY of the stage-1 record dump (the tokens, tree root and data root of stage 1);
    # argv and env differences are kept
    from dexengine.rl import cell_session as CR

    s1 = out / "s1"
    s1.mkdir()
    (s1 / "clips.txt").write_text(tname.removesuffix(P.TREE_SFX) + "\n")
    CR.stage_coll_preset(cell["folder"], TREE, alias + "s1")
    tok1 = C.env_set(
        cell["env"],
        DEXENGINE_SEQ_LIST=s1 / "clips.txt",
        GPU="0",
        DEXENGINE_MAIN_ROOT=data_root,
        DEXENGINE_RL_QUEUE_ROOT=a.queue_root,
        FROZEN_REPLAY_CKPT=ck,
        FROZEN_REPLAY_NE=64,
        FROZEN_REPLAY_EVALS_PER_CLIP=64,
        FROZEN_REPLAY_ITER=cell["iter"],
        DEXENGINE_TRAJ_RECORD=s1 / "rollouts.npz",
        **({"SEQ_DIR": seq_root} if v else {}),
    )
    C.dryrun(TREE, tok1, alias + "s1", s1 / "dryrun.out")
    d1 = C.ref_dir(TREE, data_root) / f"{alias}s1_frozenreplay_train.log"
    phys = dump_diff(d1, out / "dump.log")
    if phys["unexpected"]:
        raise SystemExit(f"{MARK} {tname}: physics differences eval dump ↔ stage-1 dump {phys['unexpected']}")
    return phys


_A = None


def _init(a):
    global _A
    _A = a
    os.environ["DEXENGINE_MAIN_ROOT"] = str(a.main)


def _one(row: dict):
    try:
        return row["key"], prep_clip(_A, row, _A.stage), None
    except (SystemExit, Exception) as e:  # noqa: BLE001 — a reason per scene (nothing dropped silently)
        return row["key"], None, f"{type(e).__name__}: {e}"


def key_of(r: dict) -> str:
    """Scene slot name — test: the test clip's tree name (one per clip, variant scenes too); aux: <variant
    tree>.<look>.
    """
    from dexengine.core import paths as P

    if r.get("start_rule") in ("aux_variant", DENSE_RULE):
        return f"{r['variant']['name']}.{r['variant']['look']}"
    if r.get("start_rule") == "orig_look":
        return f"{P.tree_name(r['base'])}.{r['look']['name']}"
    return P.tree_name(r["base"])


def _build_parser():
    """The command line (prep.py --help)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="scene list JSONL, one scene per line (test_episodes or aux_episodes)",
    )
    ap.add_argument(
        "--clips", default="", help="scene slot names to pick (comma · key_of) — default all (except missing rows)"
    )
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--hand", default="wuji2")
    ap.add_argument(
        "--queue-root", required=True, help="RL queue root written into the dump token (DEXENGINE_RL_QUEUE_ROOT)"
    )
    ap.add_argument("--augvis-assets", required=True, help="random-look asset root of the render (AUGVIS_ASSETS)")
    ap.add_argument(
        "--variant-src",
        type=Path,
        default=None,
        help="folder with the sim_eval.scenes.variant_src output (required when there are variant rows)",
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument(
        "--allow-dense",
        action="store_true",
        help=f"accept start_rule {DENSE_RULE} (dense variant scene) rows — default off: such rows die",
    )
    return ap


def _select_rows(a):
    """Episode rows to prepare: dense rows only with --allow-dense, unique slot names, the picked --clips."""
    want = {c for c in a.clips.split(",") if c}
    with open(a.episodes) as fh:
        allrows = [json.loads(x) for x in fh if x.strip()]
    nd = sum(r.get("start_rule") == DENSE_RULE for r in allrows)
    if nd and not a.allow_dense:
        raise SystemExit(f"{MARK} {DENSE_RULE} rows {nd} — not accepted without --allow-dense")
    if a.allow_dense:
        print(f"{MARK} [DENSE_RULE] {DENSE_RULE} on · rows {nd}", flush=True)
    rows = [dict(r, key=key_of(r)) for r in allrows if r["root"] != "missing"]
    if len({r["key"] for r in rows}) != len(rows):
        raise SystemExit(f"{MARK} scene slot names collide")
    rows = [r for r in rows if not want or r["key"] in want]
    if want and {r["key"] for r in rows} != want:
        raise SystemExit(f"{MARK} picked scenes without a source episode: {sorted(want - {r['key'] for r in rows})}")
    return rows


def _write_prep_record(jobs, a, fails, rows):
    """Write <stage>/prep.json; returns the scene count per start rule."""
    import collections

    by = collections.Counter(j["start_rule"] for j in jobs)
    (a.stage / "prep.json").write_text(
        json.dumps(
            {
                "episodes": str(a.episodes),
                "episodes_sha256": sha256_file(a.episodes),
                "tree_commit": os.popen(f"git -C {TREE} rev-parse HEAD").read().strip(),
                "jobs": sorted(j["sub"] for j in jobs),
                "by_start_rule": dict(by),
                "fails": fails,
                "n_rows": len(rows),
                "shadow": str(a.shadow) if a.shadow else None,
                "variants": len(a.variants),
                "job_rows": sorted(jobs, key=lambda j: j["sub"]),
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return by


def main() -> int:
    ap = _build_parser()
    a = ap.parse_args()
    from dexengine.core import paths as P

    a.main = P.main_root().resolve()
    if P.data_layout.data_shadow(str(a.main)) is not None:
        raise SystemExit(f"{MARK} data root {a.main} is a shadow — pass the canonical root")
    os.environ["AUGVIS_ASSETS"] = a.augvis_assets  # read by the record render code
    a.stage = Path(os.path.realpath(a.stage))
    if a.stage.is_relative_to(a.main) or a.main.is_relative_to(a.stage):
        raise SystemExit(f"{MARK} stage folder {a.stage} overlaps the data root {a.main}")
    rows = _select_rows(a)
    a.stage.mkdir(parents=True, exist_ok=True)
    vrows = [r for r in rows if r.get("variant")]
    a.shadow, a.variants = (None, {})
    if vrows:
        if a.variant_src is None:
            raise SystemExit(f"{MARK} variant scenes {len(vrows)} — needs --variant-src")
        a.shadow, a.variants = build_shadow(a, a.stage, vrows)
    jobs, fails = [], {}
    with ProcessPoolExecutor(max(1, a.jobs), initializer=_init, initargs=(a,)) as ex:
        for key, job, err in ex.map(_one, rows, chunksize=1):
            if err:
                fails[key] = err
                print(f"{MARK} ✗ {key}: {err[:400]}", flush=True)
            else:
                jobs.append(job)
                print(f"{MARK} ✓ {key}", flush=True)
    by = _write_prep_record(jobs, a, fails, rows)
    print(f"{MARK} DONE scenes {len(jobs)}/{len(rows)} · failed {len(fails)} · {dict(by)} → {a.stage}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
