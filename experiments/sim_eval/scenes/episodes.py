#!/usr/bin/env python3
"""E5 evaluation scenes: pick the source data episode of each start scene (one JSONL line per scene). Read only; the
one write is --out.

Test rows (--which test): one source episode per test clip of the split, by the rules in START_RULES order:
  (1) orig_D        original root · not superseded · look D · round orig · no placement variant (no _pv)
  (2) aug_D         augmentation root · not superseded · look D · no placement variant (a clip whose original is out
  of
                    view has no variant 0 → row 0 = the original)
  (3) row0_variant  augmentation root · not superseded · look D · the variant on deck row 0 of that clip = the
  episode's
                    traj_choice.row0 — the variant the original-out-of-view rule (traj_choice.orig_fov.rule) put on row
                    0
                    instead of the original (the first slot of the data's own fill order). Test clips are outside
                    training, so any variant is fair; the one picked is the one the data fixed. A row0 that is not
                    unique within the clip, or a missing row0 episode (look D), gives missing (with a reason).
  (4) missing       when none of the above exist: a 'missing' row with its reason (never dropped silently).
Auxiliary rows (--aux aux_holdout.tsv): per table row (training clip × held-out variant × look), the augmentation-root
  episode with (clip = aug_clip, look). Found by key, not by episode_index (final-root renumbering). None or several →
  missing (with a reason).
Variant rows (row0_variant, auxiliary) carry variant{name, k, placement{xy_shift_m[dx, dy], yaw_deg} (the episode's
  selection record, scene plane), look, archive} — archive = the first existing archive copy of that piece in
  dexengine.core.layout.collect_job_archive_roots('aug') order.
The Put-release measurement (data force end window) is not here: judge_io.data_object_force reads files.forces.
Data file paths follow info.json data_path · video_path · collect.force_path.
The row fields (why, placement.frame) are stored in the scene jobs and keep their wording.
Marker [E5_EPISODES] test n · orig_D a · aug_D b · row0_variant c · missing d | aux n · found a · missing b."""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from dexengine.core import layout as DL
from experiments.textio import read_lines

PV = re.compile(r"_pv(\d+)$")
HAND = "wuji2"
START_RULES = ("orig_D", "aug_D", "row0_variant")
KEEP = (
    "episode_index",
    "length",
    "video_frames",
    "clip",
    "source_cell",
    "camera_pose",
    "body",
    "head_joints",
    "spawn_hold_steps",
    "f0_hold",
    "selection",
    "look",
    "round",
    "tracking",
    "object_present",
    "side",
    "side_object",
    "shared_object",
    "supports",
    "support_objects",
    "case",
    "idle_hand",
    "resting_arm",
    "action_fill",
    "extras_visual",
    "render_preset",
    "piece_id",
    "src_episode_index",
    "tasks",
    "traj_choice",
)


def base_of(c: str) -> tuple[str, str]:
    """Clip name → (base clip, placement variant index or '')."""
    t = c[:-6] if c.endswith("_pl_rt") else c
    m = PV.search(t)
    return PV.sub("", t), (m.group(1) if m else "")


def archive_of(ds: str, piece_id: str, rnd: str = "aug") -> str | None:
    """First existing archive copy of a collected piece, or None."""
    for r in DL.collect_job_archive_roots(rnd):
        d = os.path.join(r, "jobs", HAND, ds, piece_id)
        if os.path.isdir(d):
            return d
    return None


def scan(want_ds: set, roots: tuple) -> dict:
    """(root tag, ds) → (dir, info, superseded indices, [episode rows])."""
    out = {}
    for tag, root in roots:
        for ds in sorted(want_ds):
            d = os.path.join(root, "data", ds, "collect", "rby1_" + HAND)
            f = os.path.join(d, "meta", "episodes.jsonl")
            if not os.path.isfile(f):
                continue
            info = json.loads(Path(os.path.join(d, "meta", "info.json")).read_text())
            sp = os.path.join(d, "meta", "superseded.jsonl")
            gone = (
                {int(json.loads(x)["episode_index"]) for x in read_lines(sp) if x.strip()}
                if os.path.isfile(sp)
                else set()
            )
            with open(f) as fh:
                eps = [json.loads(x) for x in fh if x.strip()]
            out[(tag, ds)] = (d, info, gone, eps)
    return out


def files_of(d: str, info: dict, ei: int) -> dict:
    """Data files (parquet, ego videos, forces) of episode ei."""
    cs = int(info["chunks_size"])
    fm = {"episode_chunk": ei // cs, "episode_index": ei}
    f = dict(
        parquet=os.path.join(d, info["data_path"].format(**fm)),
        **{
            k: os.path.join(d, info["video_path"].format(video_key=k, **fm))
            for k in ("observation.images.ego_left", "observation.images.ego_right")
        },
    )
    fp = (info.get("collect") or {}).get("force_path")
    if fp:
        f["forces"] = os.path.join(d, fp.format(**fm))
    return f


def row_of(ds, base, split_row, tag, d, info, e, rule, variant=None) -> dict:
    """One scene row from a data episode."""
    ei = int(e["episode_index"])
    r = dict(
        dataset=ds,
        base=base,
        split_row=split_row,
        root=tag,
        root_dir=d,
        fps=info["fps"],
        video_fps=(info.get("collect") or {}).get("video_fps"),
        files=files_of(d, info, ei),
        start_rule=rule,
        **{k: e.get(k) for k in KEEP},
    )
    if variant is not None:
        r["variant"] = variant
    return r


def variant_of(ds: str, e: dict, look: str) -> dict:
    """The placement-variant record of an episode (from its selection)."""
    s = e.get("selection") or {}
    for k in ("k", "dx", "dy", "yaw"):
        if s.get(k) is None:
            raise SystemExit(f"[E5_EPISODES] {e['clip']}: selection.{k} missing — no placement-variant record")
    return {
        "name": e["clip"],
        "k": int(s["k"]),
        "placement": {
            "xy_shift_m": [float(s["dx"]), float(s["dy"])],
            "yaw_deg": float(s["yaw"]),
            "frame": "scene (+x in front of the robot · +y left)",
        },
        "look": look,
        "piece_id": e.get("piece_id"),
        "archive": archive_of(ds, e.get("piece_id") or ""),
    }


def test_rows(split: str, which: str, db: dict) -> tuple[list, dict]:
    """Test (or train/all) rows of the split by START_RULES, plus per-rule counts."""
    with open(split) as fh:
        want = {
            (r["dataset"], r["clip"]): r
            for r in csv.DictReader(fh, delimiter="\t")
            if which == "all" or r["split"] == which
        }
    best, row0s, why = {}, {}, {}
    for (tag, ds), (d, info, gone, eps) in db.items():
        for e in eps:
            b, pv = base_of(e["clip"])
            if (ds, b) not in want or int(e["episode_index"]) in gone:
                continue
            if tag == "aug" and pv:
                r0 = ((e.get("traj_choice") or {}).get("orig_fov") or {}).get("row0")
                if r0:
                    row0s.setdefault((ds, b), set()).add(r0)
            if ((e.get("look") or {}).get("name")) != "D" or pv:
                continue
            rank = 0 if tag == "orig" and e.get("round") == "orig" else 1
            cur = best.get((ds, b))
            if cur is not None and cur[0] <= rank:
                continue
            best[(ds, b)] = (rank, row_of(ds, b, want[(ds, b)], tag, d, info, e, START_RULES[rank]))
    # (3) row-0 variant — only for clips without an original or variant-0 episode
    for key in sorted(set(want) - set(best)):
        ds, b = key
        r0 = row0s.get(key, set())
        if len(r0) != 1:
            why[key] = f"row0 {sorted(r0)} — not exactly one"
            continue
        name = next(iter(r0))
        d, info, gone, eps = db[("aug", ds)]
        hit = [
            e
            for e in eps
            if e["clip"] == name and (e.get("look") or {}).get("name") == "D" and int(e["episode_index"]) not in gone
        ]
        if len(hit) != 1:
            why[key] = f"row0 {name} look D episodes {len(hit)}"
            continue
        best[key] = (2, row_of(ds, b, want[key], "aug", d, info, hit[0], "row0_variant", variant_of(ds, hit[0], "D")))
    rows = [
        best[key][1]
        if key in best
        else {
            "dataset": key[0],
            "base": key[1],
            "split_row": want[key],
            "root": "missing",
            "start_rule": "missing",
            "why": why.get(key, "no source episode"),
        }
        for key in sorted(want)
    ]
    n = {k: sum(r["start_rule"] == k for r in rows) for k in (*START_RULES, "missing")}
    return rows, n


def aux_rows(aux: str, split: str, db: dict, aux_split: str = "train", aug_orig: bool = False) -> tuple[list, dict]:
    """Auxiliary rows. aux_split = the split side the row's clip must be on (train | test — test = all test variants).
    aug_orig: augmentation-root rows without a placement variant (the augmentation root re-rendering an
    original-out-of-view clip at its original place) are emitted like the test aug_D rows (look D only; other looks
    raise). Off = the aux_variant path."""
    with open(split) as fh:
        sp = {(r["dataset"], r["clip"]): r for r in csv.DictReader(fh, delimiter="\t")}
    rows, miss = [], 0
    idx: dict = {}
    for (tag, ds), (_d, _info, gone, eps) in db.items():
        for e in eps:
            if int(e["episode_index"]) in gone:
                continue
            idx.setdefault((tag, ds, e["clip"], (e.get("look") or {}).get("name")), []).append(e)
    with open(aux) as fh:
        for a in csv.DictReader(fh, delimiter="\t"):
            tag = a.get("root") or "aug"
            if tag not in ("aug", "orig"):
                raise SystemExit(f"[E5_EPISODES] aux {a['clip']}: root {tag!r} — aug|orig")
            key = (tag, a["dataset"], a["aug_clip"], a["look"])
            hit = idx.get(key, [])
            srow = sp.get((a["dataset"], a["clip"]))
            if srow is None or srow["split"] != aux_split:
                raise SystemExit(
                    f"[E5_EPISODES] aux {a['clip']}: not on the {aux_split} side of the split({srow and srow['split']})"
                )
            base = a["aug_clip"][:-6] if a["aug_clip"].endswith("_pl_rt") else a["aug_clip"]
            if len(hit) != 1:
                miss += 1
                rows.append(
                    {
                        "dataset": a["dataset"],
                        "base": base,
                        "split_row": srow,
                        "root": "missing",
                        "start_rule": "aux_missing",
                        "aux": a,
                        "why": f"(clip {a['aug_clip']} · look {a['look']}) episodes {len(hit)}",
                    }
                )
                continue
            d, info, _g, _e = db[(tag, a["dataset"])]
            # original root (no placement variant): look D = like test orig_D; other looks = the original round's plan
            if tag == "orig":
                if base_of(hit[0]["clip"])[1]:
                    raise SystemExit(
                        f"[E5_EPISODES] aux {a['aug_clip']}: placement-variant name on an original-root row"
                    )
                r = row_of(
                    a["dataset"], base, srow, "orig", d, info, hit[0], "orig_D" if a["look"] == "D" else "orig_look"
                )
                r["archive"] = archive_of(a["dataset"], hit[0].get("piece_id") or "", "orig")
            elif (
                aug_orig and not base_of(hit[0]["clip"])[1]
            ):  # augmentation root at the original place (no _pv) — like test aug_D
                if a["look"] != "D":
                    raise SystemExit(
                        f"[E5_EPISODES] aux {a['aug_clip']}: --aux-aug-orig takes look D only ({a['look']})"
                    )
                r = row_of(a["dataset"], base, srow, "aug", d, info, hit[0], "aug_D")
            else:
                r = row_of(
                    a["dataset"],
                    base,
                    srow,
                    "aug",
                    d,
                    info,
                    hit[0],
                    "aux_variant",
                    variant_of(a["dataset"], hit[0], a["look"]),
                )
            r["aux"] = a
            rows.append(r)
    return rows, {"rows": len(rows), "found": len(rows) - miss, "missing": miss}


def main() -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orig-root", required=True, help="original collection root (holds data/<ds>/collect/rby1_wuji2)")
    ap.add_argument("--aug-root", required=True, help="augmentation collection root")
    ap.add_argument("--which", default="test", choices=("test", "train", "all"))
    ap.add_argument("--aux", default="", help="aux_holdout.tsv — emit auxiliary rows (instead of test rows)")
    ap.add_argument(
        "--aux-split",
        default="train",
        choices=("train", "test"),
        help="split side of the auxiliary rows' clips (default train) — test = all test variants",
    )
    ap.add_argument(
        "--aux-aug-orig",
        action="store_true",
        help="emit augmentation-root auxiliary rows without _pv like aug_D (default off = aux_variant)",
    )
    a = ap.parse_args()
    out = a.out
    with open(a.split) as fh:
        ds_all = {r["dataset"] for r in csv.DictReader(fh, delimiter="\t")}
    db = scan(ds_all, (("orig", a.orig_root), ("aug", a.aug_root)))
    if a.aux:
        if a.aux_split != "train" or a.aux_aug_orig:
            print(f"[E5_EPISODES] aux options aux_split={a.aux_split} aux_aug_orig={int(a.aux_aug_orig)}", flush=True)
        rows, n = aux_rows(a.aux, a.split, db, a.aux_split, a.aux_aug_orig)
    else:
        rows, n = test_rows(a.split, a.which, db)
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(f"[E5_EPISODES] {'aux' if a.aux else 'test'} {len(rows)} · {n} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
