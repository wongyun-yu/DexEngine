"""Render step of record.bundle over every clip (render_all, one render.clip.stage2_clip session per clip): the clip
members by source clip, the held reasons, the random-look guard and the original's field-of-view records (fov_record,
write_fov_records). render.clip owns the per-clip session layout and plan. Part of the render fingerprint
(record.cache.RENDER_CODE_FILES)."""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record import chain as CANON
from dexengine.record import episodes as EPI
from dexengine.record import tree_ref as TR
from dexengine.record.render import fov as FOW
from dexengine.record.render.clip import MARK, ROOT_DIR, fov_owner_out, members_key, stage2_clip
from dexengine.record.store import lerobot as CL
from dexengine.record.store import meta as CMT

# head of the round exclusion reason (round_excluded, stored in data): held clips are not rendered
HELD_WHY = "canonical cell held"


def held_why(held: dict, clip: str) -> str | None:
    """Round exclusion reason of a held clip (counts.json held: CANON_HOLD, technical failure of the canonical cell
    and so on), else None. A held clip does not get the row-0 original fallback either: that fallback is only for
    clips with truly 0 variants, and a held clip is owned by the reassignment (the same clip
    in two pieces blocked the merge)."""
    if clip not in held:
        return None
    return f"{HELD_WHY}({(held[clip] or {}).get('kind')}) — no row-0 original fallback drawn · reassigned"


def members_by_clip(items: list[dict], aug_items: list[dict]) -> dict[str, list[dict]]:
    """Original items + that clip's variant items (selection.source_clip) -> {original tree name: [original,
    variants …]}. A variant without an owner dies."""
    out = {it["clip"]: [it] for it in items}
    for v in aug_items:
        src = (v.get("selection") or {}).get("source_clip")
        if src not in out:
            raise InputMissingError(f"{MARK} variant {v['clip']}: its original {src} is not in the base record")
        out[src].append(v)
    return out


def looks_guard(rd: Path, ds: Path) -> None:
    """Write the look branch drawn into the round place (DEXENGINE_COLLECT_LOOKS); a dataset already written by
    another branch is never overwritten (dies)."""
    mode, f = EPI.looks_mode() or "all", rd / "looks_mode.txt"
    had = f.read_text().strip() if f.is_file() else mode
    if had != mode:
        raise SystemExit(f"{MARK} {rd}: store branch {mode} over {had}: draw into another --out (piece place)")
    rd.mkdir(parents=True, exist_ok=True)
    f.write_text(mode + "\n")


def fov_record(a, clip: str, members: list[dict]) -> dict:
    """View record of a clip whose original was skipped as off screen in the orig round: the orig_fov of that clip's
    session summary (<out>/stage2c/<key>/<clip>/render_clip.json; rule, orig_kept false, view shape render.fov.check)
    or die (an off-screen clip without a record would get its judge cells by the plain rule)."""
    from dexengine.record.render import fov as FOV

    f = a.out / ROOT_DIR / members_key(members) / clip / "render_clip.json"
    of = (json.loads(f.read_text()) if f.is_file() else {}).get("orig_fov") or {}
    if of.get("rule") != EPI.ORIG_FOV_RULE or of.get("orig_kept") is not False or not of.get("view"):
        raise InputMissingError(
            f"{MARK} {clip}: skipped as off screen but no view record (rule {of.get('rule')} · "
            f"orig_kept {of.get('orig_kept')} · view {bool(of.get('view'))}) — {f}"
        )
    FOV.check(of["view"])
    return {"clip": clip, "orig_fov": of}


def write_fov_records(rd: Path, recs: dict) -> None:
    """Off-screen records -> the round place ORIG_FOV_DIR/<clip>.json (carried in the piece; the put_fov shape of the
    judge cells) · one marker line."""
    from dexengine.record import piece as PC

    if not recs:
        return
    d = rd / PC.ORIG_FOV_DIR
    d.mkdir(parents=True, exist_ok=True)
    for c, r in recs.items():
        (d / f"{c}.json").write_text(json.dumps(r, ensure_ascii=False))
    print(f"{EPI.ORIG_FOV_MARK} off-screen view records {len(recs)} -> {d} (piece carry = CARRY)", flush=True)


def _aug_provenance(a, source, aug_items, held):
    """Aug round: the augment record items, held canonical cells and the augmentation provenance in source."""
    from dexengine.record.augment.stage_run import provenance

    if CANON.AUGC.on():
        aug_items = json.loads((a.out / "aug" / "record_items.json").read_text())
        cnt = json.loads((a.out / "aug" / "counts.json").read_text())
        held = dict(cnt.get("held") or {})  # held canonical cells (CANON_HOLD, technical failure) -> {kind, why}
        source["augmentation"] = provenance(
            cnt,
            {
                "kind": "look",
                "entry": "collect_render_clip.render_clip",
                "plan": "collect_episodes",
                "look_d": EPI.LOOK_D,
                "random_per_traj": EPI.RANDOM_PER_TRAJ,
                "assets": source.get("augvis_assets"),
            },
        )
    return aug_items, held


def _render_clips(items, aug_items, held, skip, a, cells, rnd, fov, tsha, eps):
    """Render each clip (held and empty clips skipped with their reason); episodes, skips, off-screen records, tree
    shas."""
    for clip, members in members_by_clip(items, aug_items).items():
        if hw := held_why(held, clip):  # held clip: no row-0 original fallback either (the reassignment owns it)
            skip[clip] = hw
            print(f"{MARK} {clip}: skipped {hw}", flush=True)
            continue
        got, why = stage2_clip(a, cells[members[0]["cell"]["ckpt"]], clip, members, rnd)
        # the owner says off screen but aug did not draw row 0 = the two verdicts disagree
        if why and EPI.aug_on(rnd) and fov_owner_out(a, clip):
            raise SystemExit(
                f"{FOW.ORIG_FOV_OWNER_MARK} ✗ {clip}: owner = off screen but the aug round skipped "
                f"({why}): row 0 (a variant, else the original) must be drawn (no clip loss)"
            )
        if why:  # nothing to draw (off-screen orig, visible + 0 variants aug): an exclusion with a reason
            skip[clip] = why
            if rnd == "orig" and why.startswith(EPI.ORIG_FOV_WHY):
                fov[clip] = fov_record(a, clip, members)
            print(f"{MARK} {clip}: skipped {why}", flush=True)
            continue
        print(f"{MARK} {clip}: trajectories {len(members)} · episodes {len(got)}", flush=True)
        CMT.attach_traj_short(clip, got, a.out)
        # tree records written and checked by stage2_clip (seq1c/<key>/<tree>), verified only
        for m in members:
            tsha[m["clip"]] = TR.sha_of(a.out / "seq1c" / members_key(members) / m["clip"])
        eps += got
        CL.prewarm(got)  # probe the videos while rendering
    return eps


def render_all(a, plan: list, items: list, sel: dict, source: dict, want: list[str], rnd: str) -> int:
    """The clip branch of the bundle main: augment physics (aug round) -> render per clip -> one store into the round
    place. Exit codes follow record.bundle."""
    from dexengine.record import bundle as C
    from dexengine.record import piece as PC  # owner of the round place (round_dir)
    from dexengine.record.looks.augvis import scene as AVS
    from dexengine.record.store import preview as PV
    from dexengine.record.store import store as ST

    (PC.round_dir(a.out, rnd, a.part) / PC.EMPTY_FILE).unlink(missing_ok=True)  # a rerun removes the old empty record
    # random looks need the canonical assets: die before physics and augment (the sessions inherit the env)
    source["augvis_assets"] = str(AVS.assets_root())
    if not CANON.after_store(a, plan, items, source):
        return C.RC_AUG_FAILED
    aug_items, held = [], {}
    aug_items, held = _aug_provenance(a, source, aug_items, held)
    cells = {str(c["ckpt"]): c for c, _ in plan}
    eps, skip, fov, tsha = [], {}, {}, {}
    eps = _render_clips(items, aug_items, held, skip, a, cells, rnd, fov, tsha, eps)
    rd = PC.round_dir(a.out, rnd, a.part)
    write_fov_records(rd, fov)
    if not eps:  # 0 for the whole bundle (can be normal): no death, reasons and one alert line
        why = PC.write_empty(rd, rnd, skip, CANON.base_exclusions(source, want))
        print(
            f"[COLLECT_ALERT] round {rnd}: 0 episodes to draw · reasons {len(why)} · {rd / PC.EMPTY_FILE}", flush=True
        )
        print(f"{C.MARK} DONE episodes 0 · clips {len(items)}/{len(want)} · empty round {rnd}", flush=True)
        return 0 if len(items) == len(want) else C.RC_PARTIAL
    if a.until == "render":  # the GPU part ends here; the same command without --until stores (record.bundle)
        print(
            f"{C.RENDER_DONE_LINE} episodes {len(eps)} · clips {len(items)}/{len(want)} · store in the next call",
            flush=True,
        )
        return 0 if len(items) == len(want) else C.RC_PARTIAL
    from dexengine.record.augment import policy as AP

    source["round_excluded"] = skip
    source["episodes_plan"] = {
        "version": EPI.PLAN_VERSION,
        "look_d": EPI.LOOK_D,
        "round": rnd,
        "deck": EPI.deck_args(AP.PER_CLIP),
        "random_per_traj": EPI.RANDOM_PER_TRAJ,
    }
    source["tree_sha"] = tsha  # fingerprint per drawn tree (piece evidence; merged into meta/provenance tree_sha)
    ds = rd / "dataset"
    looks_guard(rd, ds)
    ST.store(ds, eps, source)
    # preview = one original look D per clip (a sample for people)
    first: dict = {}  # first look D episode per source clip (the first variant when the original was not chosen)
    for e in eps:
        s = e["clip"] if e["clip"] in sel else (e.get("selection") or {}).get("source_clip")
        if e["look"] == EPI.LOOK_D and s in sel:
            first.setdefault(s, e)
    PV.preview(ds, rd / "preview", list(first.values()), sel)
    print(
        f"{C.MARK} DONE episodes {len(eps)} · clips {len(items)}/{len(want)} · cells {len(source['cells'])} · "
        f"dataset {ds}",
        flush=True,
    )
    return 0 if len(items) == len(want) else C.RC_PARTIAL
