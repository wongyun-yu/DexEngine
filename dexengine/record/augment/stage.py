#!/usr/bin/env python3
"""Placement augment record step, one step of the canonical record chain (record.bundle): for every clip whose base
record ended, it builds candidate trajectories (<= policy.CAND_MAX) up to physics, from which policy.PER_CLIP new
placements are chosen, and leaves them as <out>/aug/record_items.json; render and store happen in the per-clip render
(record.clip render_all) together with the original. The orchestration (run, one_clip, provenance) is
augment.stage_run; this module owns the switches, the cache fingerprint, the markers and the per-step functions.
Rules: per clip (PER_CLIP — the trained policy as it is, only the object place and direction change = mode A), a
post collision gate, and the main record = RL-successful clips + augments. Every rule belongs to its existing
owner (no copies):
  1. generation · pre gate  augment.place_gen: shadow root (<out>/aug/shadow) · candidates = the judged cell bundle
                            (judged_bundle: cells chosen from the region by policy.pick_cells, judged by judge mode L ·
                            seed policy.clip_seed, hand independent) · judge drop · desk at the last frame ·
                            motor limits · collision gate (final_guard) · physics cap (policy.phys_set) -> variant
                            status
  2. rollout                augment.run cell_spec + run_spec (mode A · one cell = one Isaac session (cell batch) ·
                            policy.EPC evaluations per variant): recorder rollout.recorder · the force partner
                            decomposition marker is checked (a mismatch dies)
  3. post gate              augment.coll_gate post: rule policy.post_ok (relative to the same clip's original
                            placement pv00)
  4. choice                 policy.pick_per_clip (salvaged = pre ∧ sim ∧ post · <= CAND_MAX candidates; the render
                            session chooses by visibility · rule PICK_RULE) · per variant the rollout = the post-passing
                            one with the smallest end error
  5. render · store         outside this step: record.clip render_all reads record_items.json
Counts: per clip [candidates · pre · sim salvaged · post · chosen · stored] -> <out>/aug/counts.json.
  stored != chosen dies. With chosen < PER_CLIP the first short field is written as the reason (nothing invented;
  policy). Target clips without a base store get a 'skipped' line from the caller (chain.after_store) ->
  counts.json skipped (holds also in held) · clip row picked_names (compared by driver_check clip_problems).
  End line `[AUG_COLLECT] RECORD DONE candidate trajectories n · expected clips c x PER_CLIP = e · short m · skipped
  s` (none = invalid).
On by default (decisions 1 · 9) · off = DEXENGINE_COLLECT_AUG=0 (off line).
Visual augment = the per-clip render · prefetch and pool size = augment.prefetch · elapsed seconds = record timing.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.core.repo import REPO_ROOT
from dexengine.record.augment import policy as AP
from dexengine.record.augment import prefetch as PF
from dexengine.record.augment import subrun as SR  # child run caps and the cause field of their failures

MARK = "[AUG_COLLECT]"
ENV = "DEXENGINE_COLLECT_AUG"
DEFAULT = "1"
HERE = Path(__file__).resolve().parent
ROOT = REPO_ROOT  # tree root
#: per-clip share added to the bundle time cap (generation, one session, post gate, PER_CLIP renders); read by
#: chain extra_s
TIMEOUT_PER_CLIP_S = 1200
POST_TIMEOUT_S = 1800  # fixed post-gate cap; the generation cap is augment.subrun.GenBound (work size, bundle deadline)
VISUAL_NOTE = "visual augmentation = per-clip render (collect_clip · collect_render_clip)"  # data (meta/collect.json)
#: cache fingerprint (clips/<fingerprint>/<clip>: generation, rollout and post gate are reused only with the same
#: fingerprint) = record code (record CODE_FILES) + augment and bake code (augment.bake bake_code_files) + judge code
#: (judge_key). The clip judge input fingerprint (judge_fp) rides in the generation record and is checked on reuse
#: (a difference dies).
AUG_CODE_FILES = tuple(
    f"dexengine/record/augment/{m}.py"
    for m in (
        "place_gen",
        "place_src",
        "scene",
        "judged",
        "judged_region",
        "speed_gate",
        "bake",
        "bake_run",
        "bake_post",
        "cells_pick",
        "coll_gate",
        "coll_check",
        "policy",
        "run",
        "run_session",
        "session",
        "stage",
        "stage_run",
    )
)


def on() -> bool:
    v = os.environ.get(ENV, DEFAULT)
    if v not in ("0", "1"):
        raise SystemExit(f"{MARK} {ENV}={v!r}: 0 or 1 only")
    return v == "1"


def explicit() -> str | None:
    """Explicit value (unset = None); read by the explicit field of the record fingerprint (chain switches)."""
    return os.environ.get(ENV)


def announce() -> str:
    line = (
        f"{MARK} off (explicit {ENV}=0; the canonical default is on)"
        if not on()
        else f"{MARK} placement augment on ({'explicit' if os.environ.get(ENV) else 'default'}) · per clip "
        f"{AP.PER_CLIP}"
    )
    print(line, flush=True)
    return line


def code_fp() -> str:
    from dexengine.record import cache as C
    from dexengine.record.augment import bake as AB

    return C.code_fingerprint(ROOT, C.CODE_FILES + AUG_CODE_FILES + AB.bake_code_files())


def say(*x) -> None:
    print(MARK, *x, flush=True)


def clip_info(cell: dict, clip: str) -> dict:
    """Generator inputs (dataset, emb, aerial); value owners are the cell env (judge command) and the clip task_info
    (scene)."""
    env = dict(t.split("=", 1) for t in cell["env"] if "=" in t)
    miss = [k for k in ("DEXENGINE_DATASET", "DEXENGINE_EMBODIMENT") if not env.get(k)]
    if miss:
        raise InputMissingError(f"{MARK} cell env lacks {miss}: {cell['run']}")
    ti = json.loads((Path(cell["seq_dir"]) / clip / "task_info.json").read_text())
    return {
        "ds": env["DEXENGINE_DATASET"],
        "emb": env["DEXENGINE_EMBODIMENT"],
        "aerial": str(ti.get("scene") or "") == "aerial",
    }


def run_logged(cmd: list, log: Path, limit, env: dict, done: str) -> None:
    """A generation or post-gate child (augment.subrun.run: cap = seconds | GenBound · failure = cause field first)."""
    SR.run(cmd, log, env, done, limit, MARK)


def judged_bundle(a, cell: dict) -> Path:
    """Judged cell bundle folder of this cell (RL campaign, bundle): place owner paths.run_path("run_aug_cells")
    alone · root = the data root a.main (the fetch place, the same root as the record output run_collect; worktree
    sessions also read the main's bundles) · a missing layer or folder dies."""
    b = P.run_path("run_aug_cells", a.hand, root=a.main, campaign=str(cell["campaign"]), bundle=str(cell["bundle"]))
    if not b.is_dir():
        raise InputMissingError(f"{MARK} judged cell bundle missing {b}: fetch the judge mode D · L outputs first")
    return b


def judged_same(recorded: str, bundle: Path, main) -> bool:
    """Is the generation record's judged cell bundle (absolute path on that machine's data root) this cell's bundle?
    Rule: the path relative to the data root (under the run_aug_cells layer: hand, campaign, bundle) is the same. The
    same bundle on another root = the same (a record rendered on on another machine); equal content is measured
    separately by judge_fp. When this cell's bundle is outside the data root the rule cannot apply and it dies."""
    try:
        rel = Path(bundle).resolve().relative_to(Path(main).resolve()).parts
    except ValueError:
        raise SystemExit(
            f"{MARK} judged cell bundle {bundle} is outside the data root {main}: the relative path rule cannot apply"
        ) from None
    return len(Path(recorded).parts) > len(rel) and Path(recorded).parts[-len(rel) :] == rel


_JVAL: dict = {}


def judge_key(fp: str, bundle: Path) -> str:
    """Clip work folder fingerprint = augment code fingerprint + the bundle's judge commits
    (augment.judged.bundle_commits: the cache changes when the bundle is judged again; read once per bundle)."""
    import hashlib

    from dexengine.record.augment import judged as AJ

    if str(bundle) not in _JVAL:
        _JVAL[str(bundle)] = AJ.bundle_commits(bundle)
    return hashlib.sha256(f"{fp}\0{_JVAL[str(bundle)]}".encode()).hexdigest()[:16]


#: technical failure reasons of the canonical cell (pv00): environment, input or code defects (held, not emptied as a
#: shortage; judge, speed and desk are shortages). 'canon_oracle' = read-only for older records (see
#: the chain ORACLE_DIFF comment).
CANON_TECH = ("bake", "length", "canon_oracle")


def canon_tech_fail(m: dict) -> str:
    """Generation record -> technical failure reason of the canonical cell (else "")."""
    v0 = next((v for v in m["variants"] if v["k"] == 0), None)
    if v0 is None or v0["status"] == "ok" or v0.get("why") not in CANON_TECH:
        return ""
    return f"{v0.get('why')} {v0.get('detail', '')}".strip()


def canon_problem(m: dict) -> str:
    """Generation record -> why the canonical cell (pv00, the control) did not pass (else ""). Rollout gate =
    render.fov gate. The reasons are data (counts.json skipped)."""
    v0 = next((v for v in m["variants"] if v["k"] == 0), None)
    if v0 is None:
        return "canonical cell (pv00) record missing"
    return (
        ""
        if v0["status"] == "ok"
        else f"canonical cell (pv00) rejected {v0.get('why', '')} {v0.get('detail', '')}".strip()
    )


def gen(a, cell: dict, clip: str, d: Path, shadow: Path, env: dict, bundle: Path) -> Path:
    name = clip.removesuffix(P.TREE_SFX)
    vj = d / "gen" / f"variants_{name}.json"
    if vj.is_file():
        say(f"generation present {name}")
        return vj
    info = clip_info(cell, clip)
    say(f"{name}: judged cell bundle {bundle} · clip_seed {AP.clip_seed(info['ds'], name)}")
    run_logged(
        [
            sys.executable,
            "dexengine/record/augment/place_gen.py",
            "--main",
            a.main,
            "--shadow",
            shadow,
            "--hand",
            a.hand,
            "--dataset",
            info["ds"],
            "--emb",
            info["emb"],
            "--clip",
            name,
            "--judged",
            bundle,
            "--jobs",
            PF.pool_sizes()["gen"],
            "--out",
            d / "gen",
            *(["--aerial"] if info["aerial"] else []),
        ],
        d / "gen.log",
        SR.GenBound(TIMEOUT_PER_CLIP_S),
        env,
        "[AUG_GEN] DONE",
    )
    return vj


def rollout(a, cell: dict, vj: Path, d: Path, shadow: Path, launch_args=()) -> tuple[dict, dict, Path]:
    from dexengine.record.augment import run as AR
    from dexengine.record.rollout import recorder as REC

    V, base = AR.load_variants(types.SimpleNamespace(status="ok", variants=[str(vj)], max_variants=0, hand=a.hand))
    spec = dict(AR.cell_spec(cell, a.hand, a.main, shadow, base), variants=V)
    rd = d / "run"
    rj = rd / "run.json"
    if rj.is_file() and json.loads(rj.read_text()).get("rc") == 0 and (rd / "rollouts.npz").is_file():
        say(f"rollout present {rd}")
    elif AR.run_spec(spec, shadow, rd, "A", AP.EPC, a.gpu, "", k_keep=AP.K_KEEP, launch_args=launch_args) != 0:
        raise SystemExit(f"{MARK} rollout failed: {rd}/session.log")
    log = (rd / "session.log").read_text(errors="replace")
    probs = REC.log_problems(log)
    if f"collect_record.K_TOP → {AP.K_KEEP} (AUG_K_KEEP)" not in log:  # marker that the keep cap really applied
        probs.append(f"keep cap marker 'collect_record.K_TOP → {AP.K_KEEP} (AUG_K_KEEP)' missing")
    if probs:
        raise SystemExit(f"{MARK} rollout record marker problems {probs}: {rd}/session.log")
    return V, spec, rd


def phys_extend(a, cell: dict, vj: Path, m: dict, d: Path, shadow: Path, env: dict, fo: bool = False):
    """Physics cap extension (acceptance ladder rule: when short, roll the remaining judge cells and raise the physics
    cap for that clip only): when the first physics salvaged fewer variants than AP.SECURE and variants held back by
    the cap (spare) exist, that clip alone rolls all spares once more (<d>/ext · post gate reference = pv00 of the first
    run --ref · fo = --no-ref). None without spares."""
    spare = [v for v in m["variants"] if v["status"] == "spare"]
    if not spare:
        return None
    de = d / "ext"
    (de / "gen").mkdir(parents=True, exist_ok=True)
    vj2 = de / "gen" / vj.name
    if not vj2.is_file():
        vs = [
            dict(v, status="ok", why="phys_ext")
            if v["status"] == "spare"
            else dict(v, status="phys_done")
            if v["status"] == "ok"
            else v
            for v in m["variants"]
        ]
        vj2.write_text(json.dumps(dict(m, variants=vs), indent=1))
    print(
        f"[AUG_PHYS_EXT] {m['clip']} physics cap {AP.SAMPLE['phys_max']} -> +{len(spare)} (that clip only, all "
        f"spares, first physics salvaged < {AP.SECURE}) · {de}",
        flush=True,
    )
    V2, _spec2, rd2 = rollout(a, cell, vj2, de, shadow)
    pj2 = de / "post.json"
    if not pj2.is_file():
        run_logged(
            [
                sys.executable,
                "dexengine/record/augment/coll_gate.py",
                "post",
                "--variants",
                vj2,
                "--main",
                a.main,
                "--shadow",
                shadow,
                "--run",
                rd2,
                "--out",
                pj2,
                *(["--no-ref"] if fo else ["--ref", d / "post.json"]),
            ],
            de / "post.log",
            POST_TIMEOUT_S,
            env,
            "[AUG_COLL] post",
        )
    return V2, json.loads(pj2.read_text()), rd2


def pick(V: dict, post: dict, scale: dict, n_max: int = AP.PER_CLIP, k0=None) -> tuple[list[str], dict, dict]:
    """Post-passing rollouts -> (chosen variant tree names <= n_max · rollout row to store per variant · record of the
    chosen set policy.pick_meta). Names are matched to tree names. scale = the generation record's scale (judge region
    grid, policy.scale_of)."""
    best = {}
    for r in post["kept"]:
        n = P.tree_name(r["name"])
        if r["post_ok"] and "_pv00" not in n and (n not in best or r["rank"] < best[n]["rank"]):
            best[n] = r
    rows = [{"k": v["k"], "name": P.tree_name(n), "dx": v["dx"], "dy": v["dy"], "yaw": v["yaw"]} for n, v in V.items()]
    picked = AP.fovx_pick_per_clip(rows, set(best), scale, n_max, k0)
    return picked, best, AP.pick_meta(rows, picked, scale)


def short_reason(c: dict) -> str:
    """The first short field when chosen < PER_CLIP (reasons are data: counts.json)."""
    for k, why in (("pre", "pre gate passed"), ("sim", "sim salvaged"), ("post", "post gate passed")):
        if c[k] < AP.PER_CLIP:
            return f"{why} {c[k]} < {AP.PER_CLIP}"
    return ""


def one_npz(z, ci: int, states: np.ndarray, score: float, out: Path) -> None:
    np.savez(
        out,
        meta=z["meta"],
        entities=z["entities"],
        clip_names=np.array([str(z["clip_names"][ci])]),
        ep_table=np.zeros((0, 7), np.int64),
        ep_score=np.zeros(0),
        states=states.astype(np.float32),
        kept_offset=np.array([0]),
        kept_length=np.array([len(states)]),
        kept_clip=np.array([0]),
        kept_rank=np.array([0]),
        kept_score=np.array([score]),
        kept_row=np.array([0]),
        sel=np.array([0]),
        fb_clip=np.zeros(0, np.int64),
        fb_offset=np.zeros(0, np.int64),
        fb_length=np.zeros(0, np.int64),
        fb_score=np.zeros(0),
        fb_row=np.zeros(0, np.int64),
    )
