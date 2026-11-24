#!/usr/bin/env python3
"""E5 placement-variant scenes: gather their inputs (archive copies are read only; the one write is --out).

For every variant row (the variant field of the episodes output — test row0_variant, auxiliary aux_variant), copy only
the variant-specific files the render session read, from the piece's archive copy (variant.archive =
<layout.collect_job_archive_roots('aug')>/jobs/<hand>/<ds>/<piece_id>):
  shadow/             everything under the piece's shadow root (aug/shadow — real files only) whose path names the
                      variant tree (data/<ds>/trees/<root>/<branch>/<variant>/… ·
                      precompute/<ds>/robots/<combo>/<kind>/<variant>.*),
                      kept at its shadow-relative path
  traj/rollouts.npz   aug/clips/<key>/<source tree>/traj/<variant>/rollouts.npz (the variant's augmented physics
  trajectory, exactly one)
  src.json            provenance (piece, archive, file sha256 and size) — prep checks it
Layout: <out>/<ds>/<variant tree>/{shadow/…, traj/rollouts.npz, src.json}; a variant on several rows (auxiliary looks)
is copied once.
Original-tree start scenes (start_rule orig_D · aug_D): the stage-1 trajectory (stage1/<key>) that the render plan
which actually drew that data episode points at (piece archive stage2c/*/<tree>/plan*.json, trajs entry of this tree
with role orig) is copied from the stage-1 archive (layout.collect_stage1_variant(…, piece_id) when present, else
collect_stage1_archive) to <out>/<ds>/<tree>/traj/rollouts.npz (kind stage1; for aug_D the key must equal
selection.stage1_rerecord). The seed store is not used: a candidate picked by checkpoint sha is no evidence that it is
the trajectory that drew the episode.
Missing (tree, trajectory, archive) or ambiguous inputs put that variant on the failure list (fails.json) — never
dropped
silently. Marker [E5_VARSRC]."""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from dexengine.core import layout as DL

MARK = "[E5_VARSRC]"


def sha(p: str) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one(v: dict, ds: str, out: str) -> dict:
    """Copy one variant's shadow files and trajectory to <out>/<ds>/<variant>."""
    vt = v["name"]
    arc = v.get("archive")
    if not arc or not os.path.isdir(arc):
        raise RuntimeError(f"no archive {arc} (piece {v.get('piece_id')})")
    sh = os.path.join(arc, "aug", "shadow")
    files = []
    for top in ("data", "precompute"):
        for r, _d, fs in os.walk(os.path.join(sh, top)):
            for f in fs:
                p = os.path.join(r, f)
                rel = os.path.relpath(p, sh)
                if vt in rel.split(os.sep) or f.startswith(vt + "."):  # under the variant tree folder
                    files.append(rel)
    trees = sorted(
        {os.sep.join(x.split(os.sep)[: x.split(os.sep).index(vt) + 1]) for x in files if vt in x.split(os.sep)}
    )
    if len(trees) != 1 or not any(x.endswith(os.path.join(vt, "task_info.json")) for x in files):
        raise RuntimeError(f"shadow variant trees {trees} — exactly one with task_info.json expected")
    src_tree = vt.replace(f"_pv{v['k']:02d}_", "_", 1) if f"_pv{v['k']:02d}_" in vt else None
    tj = sorted(glob.glob(os.path.join(arc, "aug", "clips", "*", "*", "traj", vt, "rollouts.npz")))
    if len(tj) != 1:
        raise RuntimeError(f"{len(tj)} variant trajectories {tj[:3]} — exactly one expected")
    dst = os.path.join(out, ds, vt)
    part = dst + ".part"
    if os.path.isdir(part):
        shutil.rmtree(part)
    rec = []
    for rel in sorted(files) + ["__traj__"]:
        s = tj[0] if rel == "__traj__" else os.path.join(sh, rel)
        d = os.path.join(part, "traj", "rollouts.npz") if rel == "__traj__" else os.path.join(part, "shadow", rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        rec.append({"rel": os.path.relpath(d, part), "src": s, "size": os.path.getsize(d), "sha256": sha(d)})
    with open(os.path.join(part, "src.json"), "w") as fo:
        json.dump(
            {"variant": v, "dataset": ds, "tree_rel": trees[0], "source_tree": src_tree, "files": rec},
            fo,
            ensure_ascii=False,
            indent=1,
        )
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.rename(part, dst)
    return {"variant": vt, "files": len(rec), "bytes": sum(x["size"] for x in rec)}


STAGE1_RE = re.compile(r"/stage1/([0-9a-f]{16})/rollouts\.npz$")


def load_stage1_map(path: str) -> dict:
    """Optional stage-1 key table (--stage1-map): used only for orig_D scenes whose original piece archive is gone.
    Returns {} when no path is given."""
    if not path:
        return {}
    with open(path) as f:
        rows = {
            (x["dataset"], x["base"]): x
            for x in csv.DictReader((ln for ln in f if not ln.startswith("#")), delimiter="\t")
        }
    return {"path": path, "rows": rows, "sha": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _stage1_key(r, s1map):
    """The stage-1 trajectory key of the original tree start scene."""
    ds, piece = r["dataset"], r.get("piece_id")
    tname = r["base"] + "_pl_rt"
    arc = next(
        (
            d
            for root in DL.collect_job_archive_roots(r["round"])
            if os.path.isdir(d := os.path.join(root, "jobs", "wuji2", ds, piece or "-"))
        ),
        None,
    )
    how = {}
    S1MAP = s1map.get("rows") or {}
    if arc is None and S1MAP and r["start_rule"] == "orig_D":
        # no original piece archive: the key comes only from the stage-1 key table (record.json pairs + a full numeric
        # match ≤ TAU). Not in the table or unresolved raises; the table sha goes into src.json.
        m = S1MAP.get((ds, r["base"]))
        if m is None or not m["key"]:
            raise RuntimeError(
                f"no piece archive · stage-1 key table {('no row' if m is None else m['status'])} "
                f"(round {r['round']} · piece {piece})"
            )
        how = {
            "resolved_by": "stage1_map",
            "map": s1map["path"],
            "map_sha256": s1map["sha"],
            "map_status": m["status"],
            "map_detail": m["detail"][:400],
        }
    elif arc is None:
        raise RuntimeError(f"no piece archive (round {r['round']} · piece {piece})")
    keys = set() if not how else {S1MAP[(ds, r["base"])]["key"]}
    for pf in glob.glob(os.path.join(arc, "stage2c", "*", tname, "plan*.json")) if arc else []:
        with open(pf) as f:
            pl = json.load(f)
        for t in pl.get("trajs") or []:
            if t.get("name") == tname and t.get("role") == "orig":
                m = STAGE1_RE.search(t.get("traj") or "")
                if not m:
                    raise RuntimeError(f"{pf}: original trajectory {t.get('traj')} is not of the form stage1/<key>")
                keys.add(m.group(1))
    if len(keys) != 1:
        raise RuntimeError(f"render-plan stage-1 keys {sorted(keys)} — exactly one expected ({arc}/stage2c/*/{tname})")
    key = next(iter(keys))
    return ds, piece, tname, arc, how, key


def _copy_stage1(out, ds, tname, src, key, piece, r, arc, how):
    """Copy the stage-1 rollouts into the variant folder with a src.json record."""
    dst = os.path.join(out, ds, tname)
    part = dst + ".part"
    if os.path.isdir(part):
        shutil.rmtree(part)
    d = os.path.join(part, "traj", "rollouts.npz")
    os.makedirs(os.path.dirname(d))
    shutil.copy2(src, d)
    rec = [{"rel": "traj/rollouts.npz", "src": src, "size": os.path.getsize(d), "sha256": sha(d)}]
    with open(os.path.join(part, "src.json"), "w") as fo:
        json.dump(
            dict(
                kind="stage1",
                stage1_key=key,
                piece_id=piece,
                start_rule=r["start_rule"],
                dataset=ds,
                tree=tname,
                archive=arc,
                files=rec,
                **how,
            ),
            fo,
            ensure_ascii=False,
            indent=1,
        )
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.rename(part, dst)
    return {"variant": tname, "files": 1, "bytes": rec[0]["size"]}


def stage1_of(r: dict, out: str, s1map: dict) -> dict:
    """Original-tree start scene (orig_D · aug_D): the stage-1 trajectory key that the render plan which drew the data
    episode points at (piece archive stage2c/<key>/<tree>/plan.json, trajs entry of this tree with role orig) → the
    stage-1 archive (collect_stage1_variant(…, piece_id) when present, else collect_stage1_archive). Plans disagreeing
    on
    the key, an aug_D key different from selection.stage1_rerecord, or a missing file raise."""
    ds, piece, tname, arc, how, key = _stage1_key(r, s1map)
    if r["start_rule"] == "aug_D":
        k2 = ((r.get("selection") or {}).get("stage1_rerecord") or {}).get("stage1_key")
        if k2 != key:
            raise RuntimeError(f"render-plan key {key} != selection.stage1_rerecord {k2}")
    cands = [
        os.path.join(DL.collect_stage1_variant("wuji2", ds, key, piece), "rollouts.npz"),
        os.path.join(DL.collect_stage1_archive("wuji2", ds, key), "rollouts.npz"),
    ]
    src = next((c for c in cands if os.path.isfile(c)), None)
    if src is None:
        raise RuntimeError(f"no stage-1 trajectory {cands}")
    return _copy_stage1(out, ds, tname, src, key, piece, r, arc, how)


def stage1_src(ds: str, piece: str, traj: str) -> tuple[str, str]:
    """Original trajectory path of a render plan (…/stage1/<key>/rollouts.npz) → (key, stage-1 archive file); the same
    rule as stage1_of."""
    m = STAGE1_RE.search(traj or "")
    if not m:
        raise RuntimeError(f"original trajectory {traj} is not of the form stage1/<key>")
    key = m.group(1)
    cands = [
        os.path.join(DL.collect_stage1_variant("wuji2", ds, key, piece), "rollouts.npz"),
        os.path.join(DL.collect_stage1_archive("wuji2", ds, key), "rollouts.npz"),
    ]
    src = next((c for c in cands if os.path.isfile(c)), None)
    if src is None:
        raise RuntimeError(f"no stage-1 trajectory {cands}")
    return key, src


def pick_drawn(hits: list, rec_look: dict, vt: str, look: str) -> tuple:
    """When several plans hold the same (variant, look) episode (brightness-gate redraws plan.partN.rK.json), pick the
    one the record names: with a spare(k) in the record spec the plan with redraw {kind spare, k}, else the plan without
    redraw. Anything but exactly one raises."""
    if len(hits) == 1:
        return hits[0]
    spare = ((rec_look or {}).get("spec") or {}).get("spare")
    if spare is not None:
        pick = [
            h
            for h in hits
            if (h[1].get("redraw") or {}).get("kind") == "spare"
            and int((h[1].get("redraw") or {}).get("k", -1)) == int(spare["k"])
        ]
    else:
        pick = [h for h in hits if h[1].get("redraw") is None]
    if len(pick) != 1:
        raise RuntimeError(
            f"drawing session plans {[(h[0], (h[1].get('redraw') or {})) for h in hits]} · record spare {spare} — "
            f"exactly one plan expected for (variant {vt}, look {look})"
        )
    return pick[0]


def _plan_hits(v, look):
    """Plans of the variant archive that list this variant, with the fixed-look rule checked."""
    vt, arc, piece = v["name"], v.get("archive"), v.get("piece_id")
    if not arc or not os.path.isdir(arc):
        raise RuntimeError(f"no archive {arc} (piece {piece})")
    hits = []
    for pf in sorted(glob.glob(os.path.join(arc, "stage2c", "*", "*", "plan*.json"))):
        with open(pf) as f:
            pl = json.load(f)
        eps = [e for e in pl.get("episodes") or [] if e.get("traj") == vt and e.get("look") == look]
        if eps:
            hits.append((pf, pl, eps))
    # The drawing session is per_look fixed: the original round's plan.json (whole plan) and plan.head.json (head
    # session, draws no episode) hold the same episodes but did not draw them. Random looks can only be drawn by fixed
    # sessions.
    fixed = [h for h in hits if h[1].get("per_look") == "fixed"]
    if fixed or look != "D":
        hits = fixed
    if not hits or any(len(h[2]) != 1 for h in hits):
        raise RuntimeError(
            f"drawing session plans {[(h[0], len(h[2])) for h in hits]} — a drawing (per_look fixed) plan holding "
            f"one (variant {vt}, look {look}) episode is required"
        )
    return arc, piece, vt, hits


def _copy_draw_sources(out, ds, vt, look, srcs, key, piece, arc, pf, pl):
    """Copy the plan, the stage-1 rollouts (and the look allow file) into the variant folder with a src.json record."""
    dst = os.path.join(out, "_draw", ds, f"{vt}.{look}")
    part = dst + ".part"
    if os.path.isdir(part):
        shutil.rmtree(part)
    os.makedirs(part)
    rec = []
    for s, name in srcs:
        if (
            name == "orig_rollouts.npz"
        ):  # one original stage-1 trajectory per key (<out>/_stage1/<ds>/<key>/rollouts.npz)
            d = os.path.join(out, "_stage1", ds, key, "rollouts.npz")
            if not os.path.isfile(d):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d + ".part")
                os.rename(d + ".part", d)
        else:
            d = os.path.join(part, name)
            shutil.copy2(s, d)
        rec.append({"rel": name, "src": s, "size": os.path.getsize(d), "sha256": sha(d), "src_sha256": sha(s)})
    if any(x["sha256"] != x["src_sha256"] for x in rec):
        raise RuntimeError(f"copy sha mismatch {[(x['rel'], x['sha256'][:12], x['src_sha256'][:12]) for x in rec]}")
    with open(os.path.join(part, "src.json"), "w") as fo:
        json.dump(
            {
                "kind": "draw",
                "variant": vt,
                "look": look,
                "dataset": ds,
                "piece_id": piece,
                "archive": arc,
                "plan": pf,
                "stage1_key": key,
                "orig_tree": pl["clip"],
                "files": rec,
            },
            fo,
            ensure_ascii=False,
            indent=1,
        )
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.rename(part, dst)
    return {"variant": f"{vt}.{look}", "files": len(rec), "bytes": sum(x["size"] for x in rec)}


def draw_of(v: dict, look: str, ds: str, out: str, rec_look: dict | None = None) -> dict:
    """Make a variant scene (auxiliary aux_variant, test row0_variant) render with the same head and look as its
    training video: copy the collection session plan that actually drew that episode (piece archive
    stage2c/<key>/<original tree>/plan*.json whose episodes hold (variant, look) — exactly one), the allow file that
    session read (_look/allow.json, required for random looks) and the stage-1 trajectory of the plan's trajs[0]
    (original, role orig). Collection fixes the head from the original trajectory and keeps that angle for variants; the
    look spec follows from allow · look_key · deck · row/column. Not found raises (no silent fall back to look D).
    Several plans from brightness-gate redraws are told apart by the record spec's spare (pick_drawn).
    Output <out>/_draw/<ds>/<variant>.<look>/{plan.json (archive copy as is), allow.json?, src.json (sources, sha256)};
    the original stage-1 trajectory is stored once per key at <out>/_stage1/<ds>/<key>/rollouts.npz (the src.json row
    rel orig_rollouts.npz; the same key with another sha raises)."""
    arc, piece, vt, hits = _plan_hits(v, look)
    pf, pl, _eps = pick_drawn(hits, rec_look, vt, look)
    t0 = (pl.get("trajs") or [{}])[0]
    if t0.get("role") != "orig" or t0.get("name") != pl.get("clip"):
        raise RuntimeError(f"{pf}: trajs[0] {t0.get('name')}/{t0.get('role')} != original {pl.get('clip')}")
    tv = [t for t in pl["trajs"] if t.get("name") == vt]
    if vt != pl["clip"] and (
        len(tv) != 1 or not (tv[0].get("traj") or "").endswith(os.path.join("traj", vt, "rollouts.npz"))
    ):
        raise RuntimeError(f"{pf}: variant trajectory entries {tv} — exactly one traj/<variant>/rollouts.npz expected")
    key, s1 = stage1_src(ds, piece, t0["traj"])
    allow = os.path.join(os.path.dirname(pf), "_look", "allow.json")
    srcs = [(pf, "plan.json"), (s1, "orig_rollouts.npz")]
    if look != "D":
        if not os.path.isfile(allow):
            raise RuntimeError(f"random look {look} without allow file {allow}")
        srcs.append((allow, "allow.json"))
    return _copy_draw_sources(out, ds, vt, look, srcs, key, piece, arc, pf, pl)


def main() -> int:
    """CLI entry; exit 1 if any scene input could not be gathered."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, nargs="+", help="episodes.py output JSONL (test, auxiliary)")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--draw",
        action="store_true",
        help="also gather each variant scene's drawing plan, allow file and original stage-1 trajectory (draw_of)",
    )
    ap.add_argument(
        "--stage1-map", default="", help="stage-1 key table for orig_D scenes without a piece archive (off by default)"
    )
    a = ap.parse_args()
    out = a.out
    s1map = load_stage1_map(a.stage1_map)
    if s1map:
        print(
            f"{MARK} [STAGE1_MAP] on — without an original piece archive, keys come only from {s1map['path']} "
            f"(sha {s1map['sha'][:12]} · rows {len(s1map['rows'])})",
            flush=True,
        )
    want, s1rows, draws = {}, {}, {}
    for f in a.episodes:
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                v = r.get("variant")
                if r.get("root") != "missing" and v:
                    want.setdefault((r["dataset"], v["name"]), v)
                    draws.setdefault((r["dataset"], v["name"], v.get("look") or "D"), (v, r.get("look") or {}))
                elif r.get("start_rule") in ("orig_D", "aug_D", "orig_look"):
                    s1rows[(r["dataset"], r["base"])] = r
                    if (
                        r["start_rule"] == "orig_look"
                    ):  # original-root random look — the original round's drawing plan (one trajectory)
                        lk = (r.get("look") or {}).get("name")
                        draws.setdefault(
                            (r["dataset"], r["base"] + "_pl_rt", lk),
                            (
                                {
                                    "name": r["base"] + "_pl_rt",
                                    "archive": r.get("archive"),
                                    "piece_id": r.get("piece_id"),
                                },
                                r.get("look") or {},
                            ),
                        )
    os.makedirs(out, exist_ok=True)
    done, fails = [], {}
    for (ds, b), r in sorted(s1rows.items()):
        try:
            done.append(stage1_of(r, out, s1map))
        except (RuntimeError, OSError, KeyError) as e:  # noqa: PERF203  one failing item is recorded and the batch goes on
            fails[f"{ds}/{b}"] = str(e)
            print(f"{MARK} ✗ {ds}/{b}: {e}", flush=True)
    for (ds, vt), v in sorted(want.items()):
        try:
            done.append(one(v, ds, out))
        except (RuntimeError, OSError) as e:  # noqa: PERF203  one failing item is recorded and the batch goes on
            fails[f"{ds}/{vt}"] = str(e)
            print(f"{MARK} ✗ {ds}/{vt}: {e}", flush=True)
    for (ds, vt, look), (v, rec_look) in sorted(draws.items()):
        if not a.draw:
            break
        try:
            done.append(draw_of(v, look, ds, out, rec_look))
        except (RuntimeError, OSError, KeyError) as e:  # noqa: PERF203  one failing item is recorded and the batch goes on
            fails[f"_draw/{ds}/{vt}.{look}"] = str(e)
            print(f"{MARK} ✗ draw {ds}/{vt}.{look}: {e}", flush=True)
    with open(os.path.join(out, "fails.json"), "w") as fo:
        json.dump(fails, fo, ensure_ascii=False, indent=1)
    print(
        f"{MARK} DONE variants + aug_D + drawing plans "
        f"{len(done)}/{len(want) + len(s1rows) + (len(draws) if a.draw else 0)} · "
        f"failed {len(fails)} · files {sum(d['files'] for d in done)} · "
        f"{sum(d['bytes'] for d in done) / 1e6:.1f} MB → {out}",
        flush=True,
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
