"""Orchestration of the placement augment stage (record.augment.stage owns the switches, the cache fingerprint, the
markers and the per-step functions): run walks the target clips, one_clip takes one clip through generation, rollout,
post gate and pick, provenance writes the augmentation record. Part of the augment cache fingerprint
(AUG_CODE_FILES)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.record.augment import policy as AP
from dexengine.record.augment import prefetch as PF
from dexengine.record.augment import run as CB
from dexengine.record.augment.stage import (
    MARK,
    POST_TIMEOUT_S,
    ROOT,
    VISUAL_NOTE,
    canon_problem,
    canon_tech_fail,
    clip_info,
    code_fp,
    gen,
    judge_key,
    judged_bundle,
    judged_same,
    one_npz,
    phys_extend,
    pick,
    rollout,
    run_logged,
    say,
    short_reason,
)
from dexengine.record.render import fov as FOW


def gen_for(a, cell: dict, clip: str, root: Path, shadow: Path, env: dict, fp: str) -> tuple:
    bundle = judged_bundle(a, cell)
    fp = judge_key(fp, bundle)
    d = root / "clips" / fp / clip
    return bundle, d, gen(a, cell, clip, d, shadow, env, bundle)


def _post_check(pj, vj, a, shadow, rd, fo, d, env):
    """Run the collision post-check of the first physics run once (post.json is the cache)."""
    if not pj.is_file():
        run_logged(
            [
                sys.executable,
                "dexengine/record/augment/coll_gate.py",
                "post",
                "--variants",
                vj,
                "--main",
                a.main,
                "--shadow",
                shadow,
                "--run",
                rd,
                "--out",
                pj,
                *(["--no-ref"] if fo else []),
            ],
            d / "post.log",
            POST_TIMEOUT_S,
            env,
            "[AUG_COLL] post",
        )


def _merge_extension(ext, c, V, post, zs, m, k0, best, picked):
    """Physics extension: merge the salvaged of both runs and choose again (spare ∩ first physics = ∅)."""
    import numpy as np

    if ext is not None:  # merge the salvaged of both runs and choose again (spare ∩ first physics = ∅)
        V2, post2, rd2 = ext
        z2 = np.load(rd2 / "rollouts.npz", allow_pickle=False)
        kept2 = {P.tree_name(str(z2["clip_names"][int(ci)])) for ci in z2["kept_clip"]}
        c["sim"] += sum(P.tree_name(n) in kept2 for n, v in V2.items() if v["k"] != 0)
        c["phys_ext"] = sum(v["k"] != 0 for v in V2.values())
        V = {**V, **V2}
        post = dict(post, kept=[dict(r, src=0) for r in post["kept"]] + [dict(r, src=1) for r in post2["kept"]])
        zs[1] = z2
        picked, best, c["pick"] = pick(V, post, m["scale"], AP.CAND_MAX, k0)
    return V, picked, best


def _store_picked(picked, best, vby, zs, d, items, cell_ref, shadow, spec, clip, c):
    """Each picked rollout: its own rollouts.npz and an item for the store."""
    for rank, name in enumerate(picked, 1):
        r, v = best[name], vby[name]
        z = zs[int(r.get("src", 0))]  # the run that produced that rollout (0 = first physics · 1 = physics extension)
        j = int(r["j"])
        ci = int(z["kept_clip"][j])
        o, L = int(z["kept_offset"][j]), int(z["kept_length"][j])
        tdir = d / "traj" / name
        tdir.mkdir(parents=True, exist_ok=True)
        traj = tdir / "rollouts.npz"
        one_npz(z, ci, z["states"][o : o + L], float(z["kept_score"][j]), traj)
        items.append(
            {
                "traj": str(traj),
                "clip": name,
                "render_dir": None,
                "cell": cell_ref,
                "task_info": str(shadow / spec["seq_rel"] / name / "task_info.json"),
                "selection": {
                    "source_clip": clip,
                    "variant": name,
                    "k": v["k"],
                    "dx": v["dx"],
                    "dy": v["dy"],
                    "yaw": v["yaw"],
                    "r": v["r"],
                    "rollout_rank": int(r["rank"]),
                    "score": float(z["kept_score"][j]),
                    "post_values": r["values"],
                    "level": AP.EPC,
                    "pick_rule": AP.PICK_RULE,
                    "pick_rank": rank,
                    "seed": c["seed"],
                },
            }
        )
        c["stored"] += 1


def one_clip(a, cell: dict, clip: str, root: Path, shadow: Path, env: dict, items: list, fp: str) -> dict:
    import numpy as np

    bundle, d, vj = PF.take(f"{id(cell)}/{clip}") or gen_for(a, cell, clip, root, shadow, env, fp)
    m = json.loads(vj.read_text())
    nm = clip.removesuffix(P.TREE_SFX)
    seed = AP.clip_seed(clip_info(cell, clip)["ds"], nm)
    if m.get("seed") != seed:  # only generation records made with this clip's seed
        raise SystemExit(f"{MARK} {clip}: generation record seed {m.get('seed')} != clip_seed {seed}: {vj}")
    from dexengine.record.augment import judged as AJ
    from dexengine.record.augment import judged_region as JR

    jr = JR.read_region(bundle / "region" / f"region_{nm}.npz")
    # reuse only records made from the same judge result (judged_same · judge_fp)
    if not judged_same(m.get("judged") or "", bundle, a.main) or m.get("judge_fp") != jr["judge_fp"]:
        raise SystemExit(
            f"{MARK} {clip}: generation record judge {m.get('judged')} · {m.get('judge_fp')} != bundle "
            f"{bundle} · "
            f"{jr['judge_fp']} — {vj}"
        )
    new = [v for v in m["variants"] if v["k"] != 0]
    k0 = AJ.fovx_offset(bundle, nm)
    c = {
        "clip": clip,
        "seed": seed,
        "cand": len(new),
        "pre": sum(v["status"] == "ok" for v in new),
        "sim": 0,
        "post": 0,
        "picked": 0,
        "picked_names": [],
        "stored": 0,
        "reason": "",
        "judge_end": AJ.judge_end(bundle, nm),
    }
    if tech := canon_tech_fail(m):
        raise SystemExit(
            f"{MARK} ✗ {clip}: canonical cell (pv00) technical failure {tech[:300]}: job failure (retry · alert · {vj})"
        )
    stop, fo = FOW.gate(a, m, c, clip, canon_problem(m))
    if stop is not None:
        c["reason"] = stop or short_reason(c)
        return c
    V, spec, rd = rollout(a, cell, vj, d, shadow)
    z = np.load(rd / "rollouts.npz", allow_pickle=False)
    kept_names = {P.tree_name(str(z["clip_names"][int(ci)])) for ci in z["kept_clip"]}
    c["sim"] = sum(P.tree_name(n) in kept_names for n, v in V.items() if v["k"] != 0)
    pj = d / "post.json"
    _post_check(pj, vj, a, shadow, rd, fo, d, env)
    post = json.loads(pj.read_text())
    picked, best, c["pick"] = pick(V, post, m["scale"], AP.CAND_MAX, k0)
    zs = {0: z}
    ext = phys_extend(a, cell, vj, m, d, shadow, env, fo) if len(best) < AP.SECURE else None
    V, picked, best = _merge_extension(ext, c, V, post, zs, m, k0, best, picked)
    c["post"], c["picked"], c["picked_names"] = len(best), len(picked), list(picked)
    ac, adm = AP.admit_counts(m["variants"]), c["pick"]["admit"]
    c["admit"] = dict(ac, level=adm["level"], secured=adm["secured"], ext=ext is not None)
    print(
        f"[AUG_EXCLUDE] {clip} · level {AP.ladder_name(adm['level']) if adm['level'] is not None else 'none'} · "
        f"excluded cells {ac['excluded_l0']} / candidates {ac['cand']} · secured {adm['secured']}/{AP.SECURE}"
        + (" · short" if adm["secured"] < AP.SECURE else "")
        + (" · physics extension" if ext is not None else ""),
        flush=True,
    )
    c["reason"] = short_reason(c) if len(picked) < AP.PER_CLIP else ""
    vby = {P.tree_name(n): v for n, v in V.items()}
    cell_ref = {
        k: (str(cell[k]) if k == "ckpt" else cell[k]) for k in ("campaign", "bundle", "run", "ckpt", "iter", "label")
    }
    cell_ref["ckpt_sha256"] = cell["ckpt_sha"]
    _store_picked(picked, best, vby, zs, d, items, cell_ref, shadow, spec, clip, c)
    return c


def provenance(summary: dict, visual: dict | None) -> dict:
    """Owner of the placement augment statement (augmentation in the dataset's meta/collect.json); called by
    record.clip · visual = the visual augment statement (looks of the per-clip render). Values are data."""
    return {
        "kind": "placement",
        "mode": "A",
        "per_clip": AP.PER_CLIP,
        "sample": AP.SAMPLE,
        "cell_rule": AP.CELL_RULE,
        "seed_rule": "aug_policy.clip_seed(dataset, clip base name)",
        "pick_rule": AP.PICK_RULE,
        "epc": AP.EPC,
        "post_rel_mm": AP.POST_REL_MM,
        "post_terms": AP.POST_TERMS,
        "visual": visual,
        "visual_note": VISUAL_NOTE,
        "counts": summary,
    }


def run(a, per_cell: list[tuple[dict, list[str]]], skipped: dict | None = None, held: dict | None = None) -> dict:
    """Called after the base record of the bundle (chain.after_store): per_cell = [(cell, base-recorded clip tree
    names)] · skipped = target clips left out of the augment -> reason · held = the held clips among them -> {kind
    (policy.HOLD_*), why} (counts.json held). Up to the chosen trajectory files (no render or store) ·
    <out>/aug/record_items.json · end line RECORD DONE."""
    skipped = dict(skipped or {})
    if Path(a.tree).resolve() != ROOT:
        raise SystemExit(f"{MARK} session tree {a.tree} != augment code tree {ROOT}: they must be the same tree")
    root = a.out / "aug"
    shadow = root / "shadow"
    root.mkdir(parents=True, exist_ok=True)
    from dexengine.record.bundle_cell import cells_arm  # noqa: PLC0415 — the bundle's module (it imports rl)

    arm = cells_arm([cell for cell, _ in per_cell])
    env = dict(os.environ, DEXENGINE_ARM=arm, DEXENGINE_MAIN_ROOT=str(a.main), CUDA_VISIBLE_DEVICES=a.gpu)
    fp = code_fp()
    say(f"code fingerprint {fp} · tree {ROOT}")
    say(VISUAL_NOTE)
    print(PF.pool_line(), flush=True)
    items, counts = [], []
    PF.start(
        [(f"{id(c)}/{k}", (a, c, k, root, shadow, env, fp)) for c, ks in per_cell for k in ks],
        gen_for,
        PF.pool_sizes()["gen"],
    )
    try:
        for cell, clips in per_cell:
            CB.run_cell(
                a,
                cell,
                clips,
                [PF.take(f"{id(cell)}/{k}") or gen_for(a, cell, k, root, shadow, env, fp) for k in clips],
                shadow,
                canon_problem,
                canon_tech_fail,
            )
            for clip in clips:
                n0 = len(items)
                c = one_clip(a, cell, clip, root, shadow, env, items, fp)
                if c["stored"] != c["picked"] or len(items) - n0 != c["picked"]:
                    raise SystemExit(f"{MARK} {clip}: stored {c['stored']} != chosen {c['picked']}")
                counts.append(c)
                say(
                    f"{clip}: candidates {c['cand']} · pre {c['pre']} · sim {c['sim']} · post {c['post']} · chosen "
                    f"{c['picked']} · stored {c['stored']}" + (f" · short reason {c['reason']}" if c["reason"] else "")
                )
    finally:
        PF.stop()
    want = AP.PER_CLIP * len(counts)
    short = sum(max(0, AP.PER_CLIP - c["picked"]) for c in counts)  # the record-only path keeps <= CAND_MAX candidates
    unsecured = [c["clip"] for c in counts if (c.get("admit") or {}).get("secured", 0) < AP.SECURE]
    summary = {
        "per_clip": AP.PER_CLIP,
        "cand_max": AP.CAND_MAX,
        "code_fp": fp,
        "clips": len(counts),
        "expected": want,
        "stored": len(items),
        "short": short,
        "counts": counts,
        "skipped": skipped,
        "held": dict(held or {}),
        "secure": AP.SECURE,
        "unsecured": unsecured,
    }
    (root / "counts.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (root / "record_items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False, default=str))
    say(
        f"RECORD DONE candidate trajectories {len(items)} (<= {AP.CAND_MAX} per clip; the render session chooses) · "
        f"expected clips {len(counts)} x {AP.PER_CLIP} = {want} · short {short} · not secured {AP.SECURE} "
        f"{len(unsecured)} · skipped {len(skipped)} · no render or store · "
        f"{root / 'record_items.json'}"
    )
    return summary
