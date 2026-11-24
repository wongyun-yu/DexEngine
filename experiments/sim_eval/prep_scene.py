"""E5 closed-loop evaluation prep, scene inputs (called by experiments/sim_eval/prep.py — the module docstring there
is the contract): checkpoint path, trajectory, variant shadow tree, drawn render plan and the render plan of one
scene by start rule (scene_plan)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from dexengine.core.hashing import sha256_file

MARK = "[E5_PREP]"
#: start-rule name of dense variant scenes (owned here; source = the dense render plan copy, prep_clip's branch)
DENSE_RULE = "dense_D"


def traj_file():
    """traj_file (owner of the file layout; numpy only) — without importing the whole package (no Isaac)."""
    from dexengine.rl.eval import traj_file as TF

    return TF


def stage_path(a, p: Path) -> Path:
    """Real path of an output inside the stage folder (a symlink leading out of it dies)."""
    r = Path(os.path.realpath(p))
    if not r.is_relative_to(a.stage):
        raise SystemExit(f"{MARK} {p} → {r} is outside the stage folder {a.stage}")
    return r


def ckpt_path(main: Path, hand: str, ds: str, sc: dict) -> Path:
    """Map the episode record's checkpoint (a job scratch path) to the same run folder under this machine's data root by
    re-joining the relative tail overfit/hands/<hand>/rl_campaigns/…; missing or a different sha256 dies."""
    s = str(sc["ckpt"])
    key = f"overfit/hands/{hand}/rl_campaigns/"
    if key not in s:
        raise SystemExit(f"{MARK} checkpoint path lacks {key}: {s}")
    p = main / (key + s.split(key, 1)[1])
    if not p.is_file():
        raise SystemExit(f"{MARK} checkpoint {p} missing")
    got = sha256_file(p)
    if got != sc["ckpt_sha256"]:
        raise SystemExit(f"{MARK} checkpoint sha {got[:12]} ≠ episode record {sc['ckpt_sha256'][:12]} — {p}")
    return p


def place_of(v: dict, ti: dict) -> dict:
    """Variant tree task_info.aug_place ↔ the episode's selection record (variant.placement); a difference dies."""
    ap = ti.get("aug_place") or {}
    got = (ap.get("dx_scene"), ap.get("dy_scene"), ap.get("yaw_deg"))
    want = (*v["placement"]["xy_shift_m"], v["placement"]["yaw_deg"])
    if None in got or max(abs(float(x) - float(y)) for x, y in zip(got, want, strict=False)) > 1e-6:
        raise SystemExit(f"{MARK} {v['name']}: tree aug_place {got} ≠ episode selection {want}")
    return {"xy_shift_m": [float(want[0]), float(want[1])], "yaw_deg": float(want[2])}


def build_shadow(a, stage: Path, rows: list[dict]) -> tuple[Path, dict]:
    """Shadow root <stage>/shadow of placement-variant scenes (augment.scene.init_shadow, shadow_dir): every variant's
    shadow files from variant_src at the same relative place (sha and size checked). Rebuilt from scratch."""
    from dexengine.record.augment import scene as AS

    sh = stage_path(a, stage / "shadow")
    if sh.exists() or sh.is_symlink():
        shutil.rmtree(sh)
    AS.init_shadow(sh, a.main)
    out = {}
    for r in rows:
        v = r["variant"]
        if v["name"] in out:
            continue
        src = a.variant_src / r["dataset"] / v["name"]
        meta = json.loads((src / "src.json").read_text())
        for f in meta["files"]:
            fp = src / f["rel"]
            if not fp.is_file() or fp.stat().st_size != f["size"] or sha256_file(fp) != f["sha256"]:
                raise SystemExit(f"{MARK} {v['name']}: input {fp} size · sha differ from the data-server record")
        for f in meta["files"]:
            if not f["rel"].startswith("shadow/"):
                continue
            rel = Path(f["rel"]).relative_to("shadow")
            d = AS.shadow_dir(sh, a.main, str(rel.parent))
            t = d / rel.name
            if t.is_symlink():
                raise SystemExit(
                    f"{MARK} {v['name']}: shadow {t} has the same name as canonical {os.readlink(t)} — not "
                    "a variant's own file"
                )
            shutil.copy2(src / f["rel"], t)
        # Variant rows in the roster source (status_final_*.tsv per hand): place_gen.add_roster_rows (the source row
        # renamed,
        # appended to the shadow ledger copy). The shadow copy does not carry the ledger (the variant name is not in its
        # path), so
        # it is rewritten here with the canonical function. A missing source row dies.
        import types

        from dexengine.core import paths as P
        from dexengine.record.augment import place_src as APS

        srcn = types.SimpleNamespace(hand=a.hand, main=a.main, tree_name=P.tree_name(r["split_row"]["clip"]))
        added = APS.add_roster_rows(srcn, sh, [v["name"]])
        if not added:
            raise SystemExit(
                f"{MARK} {v['name']}: roster source has no row for source {srcn.tree_name} — cannot add the variant row"
            )
        # Checkpoint = an out-of-cell copy (augment run.cell_spec: shadow _ckpt/, sha256 = source). Variants are clips
        # outside
        # the training roster, so they cannot pass dexengine.rl.checkpoints's generation check (in-cell checkpoint) —
        # the same out-of-cell
        # checkpoint path as augmented records. Each cell is copied once (serially, outside the pool).
        from dexengine.record import bundle_cell as BC
        from dexengine.record.augment import run as AR

        sc = r["source_cell"]
        cell = BC.cell_of(a, sc["campaign"], sc["bundle"], ckpt_path(a.main, a.hand, r["dataset"], sc), int(sc["iter"]))
        spec = AR.cell_spec(cell, a.hand, a.main, sh, [r["split_row"]["clip"]])
        tree = sh / meta["tree_rel"]
        ti = json.loads((tree / "task_info.json").read_text())
        out[v["name"]] = {
            "seq_root": tree.parent,
            "traj": src / "traj" / "rollouts.npz",
            "placement": place_of(v, ti),
            "src_sha": sha256_file(src / "src.json"),
            "ckpt": sh / spec["ckpt_rel"],
        }
    n = relative_links(sh, Path.home().resolve())
    print(f"{MARK} shadow {sh} · variants {len(out)} · turned into relative links {n}", flush=True)
    return sh, out


def relative_links(root: Path, home: Path) -> int:
    """Rewrite symlinks under root whose absolute target is inside HOME into relative links to the same target (same
    realpath), so a copy of the tree resolves inside the copy on another machine. A target outside HOME, or a relative
    link
    leaving the root, dies."""
    n = 0
    for r, ds, fs in os.walk(root):
        for name in ds + fs:
            p = Path(r) / name
            if not p.is_symlink():
                continue
            t = os.readlink(p)
            tabs = Path(t) if os.path.isabs(t) else (p.parent / t)
            tabs = Path(os.path.normpath(tabs))
            if not tabs.is_relative_to(home):
                raise SystemExit(f"{MARK} link {p} → {t} is outside HOME — cannot resolve in the data-server copy")
            if os.path.isabs(t):
                rel = os.path.relpath(tabs, p.parent)
                p.unlink()
                p.symlink_to(rel)
                n += 1
    return n


def render_traj(a, row: dict) -> Path:
    """Source-tree scenes (orig_D, aug_D): the stage-1 trajectory of the render plan that drew the episode (from the
    variant-src stage1 copy; key and sha checked)."""
    from dexengine.core import paths as P

    if a.variant_src is None:
        raise SystemExit(f"{MARK} {row['base']}: scene trajectory input — needs --variant-src")
    src = a.variant_src / row["dataset"] / P.tree_name(row["base"])
    meta = json.loads((src / "src.json").read_text())
    if meta.get("kind") != "stage1" or meta.get("piece_id") != row.get("piece_id"):
        raise SystemExit(
            f"{MARK} {row['base']}: input {src} is not the trajectory of this episode's piece {row.get('piece_id')}"
        )
    f = meta["files"][0]
    t = src / f["rel"]
    if sha256_file(t) != f["sha256"]:
        raise SystemExit(f"{MARK} {row['base']}: {t} sha differs from the data-server record")
    return t


def reseal(plan: dict) -> None:
    """The stored plan now points at this scene's slot (trajectory, roster, output), so plan_sha is measured again over
    this
    content (episodes.plan's formula, out excluded). The stored plan's value stays in e5_src_plan.plan_sha."""
    plan["e5_src_plan"]["plan_sha"] = plan.pop("plan_sha", None)
    plan["plan_sha"] = hashlib.sha256(
        json.dumps({k: v for k, v in plan.items() if k != "out"}, sort_keys=True).encode()
    ).hexdigest()[:16]


def drawn_plan(a, row: dict, v: dict, tname: str) -> dict:
    """A variant scene's 'drawing session plan' copy (variant-src --draw → <variant_src>/_draw/<ds>/<variant>.<look>/):
    file sha checked against the record; the plan must hold one (variant, look) episode with trajs[0] = the source."""
    look = v.get("look") or "D"
    d = a.variant_src / "_draw" / row["dataset"] / f"{tname}.{look}"
    if not (d / "src.json").is_file():
        raise SystemExit(
            f"{MARK} {tname}: no copy of the drawn session plan {d} — see the variant_src --draw failure "
            "list (no fallback to look D)"
        )
    meta = json.loads((d / "src.json").read_text())
    # Source stage-1 trajectories: one copy per key (<variant_src>/_stage1/<ds>/<key>/rollouts.npz — copies of one key
    # had the
    # same sha); the rest (plan, allow) sit in the drawing plan copy. Either way the sha is checked against the record.
    s1 = a.variant_src / "_stage1" / row["dataset"] / meta["stage1_key"] / "rollouts.npz"
    for f in meta["files"]:
        fp = s1 if f["rel"] == "orig_rollouts.npz" else d / f["rel"]
        if sha256_file(fp) != f["sha256"]:
            raise SystemExit(f"{MARK} {tname}: {fp} sha differs from the data-server record")
    plan = json.loads((d / "plan.json").read_text())
    ep = [e for e in plan["episodes"] if e["traj"] == tname and e["look"] == look]
    t0 = plan["trajs"][0]
    if (
        len(ep) != 1
        or t0.get("role") != "orig"
        or t0.get("name") != plan["clip"]
        or meta.get("orig_tree") != plan["clip"]
    ):
        raise SystemExit(
            f"{MARK} {tname}: drawn plan {d}/plan.json — episodes {len(ep)} · trajs[0] "
            f"{t0.get('name')}/{t0.get('role')}"
        )
    if (look != "D") != (d / "allow.json").is_file():
        raise SystemExit(f"{MARK} {tname}: look {look} but allow.json {'missing' if look != 'D' else 'present'}")
    return {
        "plan": plan,
        "episode": ep[0],
        "orig": plan["clip"],
        "orig_traj": s1,
        "allow": (d / "allow.json") if look != "D" else None,
        "src": {
            "plan": meta["plan"],
            "archive": meta["archive"],
            "files": {f["rel"]: f["sha256"] for f in meta["files"]},
        },
    }


def _orig_look_plan(a, row, tname, traj, out, seq_root, lst):
    """Source-root random-look scene: the drawing session plan of the source round as it is (one source trajectory,
    recorded look, allow)."""
    dr = drawn_plan(a, row, {"look": (row.get("look") or {}).get("name")}, tname)
    if dr["orig"] != tname:
        raise SystemExit(f"{MARK} {tname}: source of the source-round plan {dr['orig']} ≠ scene tree")
    if sha256_file(Path(traj)) != sha256_file(dr["orig_traj"]):
        raise SystemExit(
            f"{MARK} {tname}: scene trajectory {traj} ≠ source trajectory of the drawn plan {dr['orig_traj']} (sha)"
        )
    plan = dict(
        dr["plan"],
        out=str(out / "render"),
        seq_dir=str(seq_root),
        seq_list=str(lst),
        trajs=[{"name": tname, "traj": str(traj), "role": "orig"}],
        episodes=[dr["episode"]],
        e5_src_plan=dr["src"],
    )
    reseal(plan)
    (out / "render" / "_look").mkdir(parents=True)
    shutil.copy2(dr["allow"], out / "render" / "_look" / "allow.json")
    return plan


def _dense_plan(v, tname, a, row, traj, out, seq_root, lst):
    """Dense variant scene: the dense render's plan copy (stage2_clip, orig round, DEXENGINE_COLLECT_LOOKS=D) as it is:
    one trajectory (the variant, role orig), one (variant, D) episode, no allow. The plan's source must be the scene
    tree and the plan trajectory sha the scene's."""
    if not v or (v.get("look") or "D") != "D":
        raise SystemExit(f"{MARK} {tname}: {DENSE_RULE} is for variant cells with look D only")
    dr = drawn_plan(a, row, v, tname)
    if dr["orig"] != tname or dr["allow"] is not None:
        raise SystemExit(f"{MARK} {tname}: source of the dense plan {dr['orig']} ≠ scene tree, or allow is present")
    if sha256_file(Path(traj)) != sha256_file(dr["orig_traj"]):
        raise SystemExit(
            f"{MARK} {tname}: scene trajectory {traj} ≠ trajectory of the drawn plan {dr['orig_traj']} (sha)"
        )
    if [(t.get("name"), t.get("role")) for t in dr["plan"]["trajs"]] != [(tname, "orig")]:
        raise SystemExit(
            f"{MARK} {tname}: dense plan trajectories {dr['plan']['trajs']} — must be one variant (role orig)"
        )
    plan = dict(
        dr["plan"],
        out=str(out / "render"),
        seq_dir=str(seq_root),
        seq_list=str(lst),
        trajs=[{"name": tname, "traj": str(traj), "role": "orig"}],
        episodes=[dr["episode"]],
        e5_src_plan=dr["src"],
    )
    reseal(plan)
    return plan


def _variant_plan(a, row, v, tname, seq_root, data_root, out, TF, traj):
    """Variant scene = the record session plan that actually drew the episode (stored copy, --draw) as it is:
    trajectories [source first, variant], recorded look, look_key, deck, row/column. Recording fixes the head from the
    source trajectory and pins it for the variant (check_view); the look spec follows allow, look_key, deck and
    row/column (spec_for), so the training video's head and look come back. Not found = death (no fallback to look D).
    Returns (seq list, plan)."""
    from dexengine.core import paths as P
    from dexengine.record import tree_ref as TR

    dr = drawn_plan(a, row, v, tname)
    if not (Path(seq_root) / dr["orig"] / "task_info.json").is_file():
        raise SystemExit(f"{MARK} {tname}: source tree {dr['orig']} is not in the scene tree root {seq_root}")
    os.environ["DEXENGINE_MAIN_ROOT"] = str(
        data_root
    )  # source tree fingerprint too (tree_ref — the tree that fixes the head)
    try:
        TR.resolve(TR.write(out / "seq1_orig", seq_root, dr["orig"]))
    finally:
        os.environ["DEXENGINE_MAIN_ROOT"] = str(a.main)
    if TF.load(str(dr["orig_traj"])).selected(dr["orig"]) is None:
        raise SystemExit(
            f"{MARK} {tname}: source stage-1 trajectory file {dr['orig_traj']} has no selected "
            f"trajectory for {dr['orig']}"
        )
    lst = out / "seq1" / "members.txt"
    lst.write_text(f"{P.clip_name(dr['orig'])}\n{P.clip_name(tname)}\n")
    plan = dict(
        dr["plan"],
        out=str(out / "render"),
        seq_dir=str(seq_root),
        seq_list=str(lst),
        trajs=[
            {"name": dr["orig"], "traj": str(dr["orig_traj"]), "role": "orig"},
            {"name": tname, "traj": str(traj), "role": "aug"},
        ],
        episodes=[dr["episode"]],
        e5_src_plan=dr["src"],
    )
    reseal(plan)
    if dr["allow"] is not None:
        (out / "render" / "_look").mkdir(parents=True)
        shutil.copy2(dr["allow"], out / "render" / "_look" / "allow.json")
    return lst, plan


def scene_plan(a, row: dict, v, tname: str, traj, seq_root, data_root, out: Path, lst, TF) -> tuple[dict, Path]:
    """prep_clip step 2 — the scene's render plan by start rule → (plan, seq list); a variant scene lists the source
    tree first and the variant second (<out>/seq1/members.txt)."""
    from dexengine.record import episodes as EPI

    if row.get("start_rule") == "orig_look":
        plan = _orig_look_plan(a, row, tname, traj, out, seq_root, lst)
    elif row.get("start_rule") == DENSE_RULE:
        plan = _dense_plan(v, tname, a, row, traj, out, seq_root, lst)
    elif v:
        lst, plan = _variant_plan(a, row, v, tname, seq_root, data_root, out, TF, traj)
    else:
        from dexengine.record.augment import policy as AP

        os.environ[EPI.LOOKS_ENV] = "D"
        plan = EPI.plan(
            tname,
            str(out / "render"),
            str(seq_root),
            str(lst),
            [{"name": tname, "traj": str(traj), "role": "orig"}],
            "orig",
            EPI.deck_args(AP.PER_CLIP),
        )
        if [e["look"] for e in plan["episodes"]] != ["D"]:
            raise SystemExit(f"{MARK} {tname}: plan episodes {plan['episodes']} — must be one look D")
    return plan, lst
