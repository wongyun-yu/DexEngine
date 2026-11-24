#!/usr/bin/env python3
"""E5 policy-training manifest — one row per source episode, keyed by identity, never by episode number.

The final collect roots get renumbered (episodes deleted · indices rewritten), so a row names its episode by
(dataset, root, clip, round, look) — the key that is unique across both collect roots (measured: 78,955 episode rows of
the two roots, 0 duplicates) — and `variant` (the placement variant `_pv<n>` of the clip name, '' for the original
placement) for reading.
The converter (convert.py) resolves the key to an episode_index in that root's episodes.jsonl at
conversion time and dies on a key it cannot find.

Inputs: the split table (made by experiments/sim_eval/split/make_split.py; its sha256 is recorded), the roots, and the
instruction source. Without per-clip text the instruction is one template sentence per task
axis (--lang template) and every row says so (lang_source=template).

Rows: split clips of --split-name whose base clip matches an episode in a root; episodes listed in a root's
meta/superseded.jsonl are left out (counted). A split clip with no episode in any root dies unless --allow-missing
— then the missing clips are listed in the sidecar.

Writes <out>.tsv and <out>.meta.json (split sha · roots · counts · missing · superseded dropped) and
<out>.counts.tsv (episodes per axis × dataset × hand_use — the kinematic rows match these counts).
Standard library only — runs with any python3 (reads only).
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

TAG = "[E5_MANIFEST]"
PV = re.compile(r"_pv(\d+)$")
PL_RT = "_pl_rt"
#: Template instruction per task axis (lang_source=template). The per-clip text replaces it (--lang <tsv>).
TEMPLATES = {
    "Lift": "Pick up the object and lift it.",
    "Put": "Pick up the object and put it down.",
    "Pour": "Pick up the container and pour from it.",
    "Sweep": "Sweep the object across the table.",
}
COLS = [
    "dataset",
    "root",
    "clip",
    "round",
    "look",
    "variant",
    "base_clip",
    "axis",
    "hand_use",
    "split",
    "side",
    "video_frames",
    "task_text",
    "lang_source",
    "verify_pending",
]


def keyed_list(path: str, roots: list[tuple[str, str]], combo: str) -> dict:
    """An episode list → {(dataset, root tag, clip, look): row}. Two key forms (a row with neither dies):
    `store` (= <root>/data/<ds>/collect/<combo> — a store outside the roots dies) + `clip` + `look`, or
    `dataset` + `root` (a root tag of --roots) + `aug_clip`|`clip` (the episode clip with its variant suffix) +
    `look`."""
    out = {}
    tags_ok = {t for t, _ in roots}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("store"):
                st = r["store"].rstrip("/")
                hit = [
                    (tag, os.path.relpath(st, os.path.join(p, "data")).split("/")[0])
                    for tag, p in roots
                    if st.startswith(os.path.join(p, "data") + "/") and st.endswith("/collect/" + combo)
                ]
                if len(hit) != 1:
                    sys.exit(f"{TAG} FAIL {path}: store {st} not under exactly one root")
                tag, ds = hit[0]
                clip = r["clip"]
            elif r.get("dataset") and r.get("root") in tags_ok and (r.get("aug_clip") or r.get("clip")):
                tag, ds, clip = r["root"], r["dataset"], (r.get("aug_clip") or r["clip"])
            else:
                sys.exit(f"{TAG} FAIL {path}: row has neither store nor dataset+root+clip: {dict(r)}")
            k = (ds, tag, clip, r["look"])
            if k in out:
                sys.exit(f"{TAG} FAIL {path}: duplicate key {k}")
            out[k] = r
    return out


def base_of(clip: str) -> tuple[str, str]:
    """Episode clip name → (split clip name, placement variant). `<clip>[_pv<n>][_pl_rt]`."""
    t = clip[: -len(PL_RT)] if clip.endswith(PL_RT) else clip
    m = PV.search(t)
    return PV.sub("", t), (m.group(1) if m else "")


def look_name(e: dict) -> str:
    lk = e.get("look")
    return str(lk.get("name") if isinstance(lk, dict) else (lk or ""))


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="e5_split.tsv")
    ap.add_argument("--split-name", required=True, choices=("train", "test"))
    ap.add_argument("--roots", required=True, help="tag=path,… — each path holds data/<ds>/collect/<combo>/meta")
    ap.add_argument("--combo", required=True, help="collect combo folder, e.g. rby1_wuji2")
    ap.add_argument("--lang", required=True, help="'template' or a TSV with columns dataset, clip, task_text")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--allow-missing", action="store_true", help="split clips with no episode are listed, not fatal")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="TSV of episodes to leave out (keyed_list forms) — repeatable · every row must match an episode",
    )
    ap.add_argument(
        "--include-unsplit",
        action="store_true",
        help="also every episode whose base clip has NO row in the split table (any split) — axis = '' · "
        "hand_use = '' · split = --split-name (GR00T mid-training corpus · "
        "convert with convert.py --allow-axis-none) — off = byte-identical",
    )
    ap.add_argument(
        "--tag", default="", help="TSV (store · clip · look [· tag]) — rows keep the episode and set verify_pending"
    )
    ap.add_argument("--tag-value", default="", help="verify_pending value for --tag rows without a tag column")
    a = ap.parse_args()
    return a


def _read_inputs(a) -> tuple:
    """The split table (its sha256 · this split's rows · the other split's clips), the text source and the roots."""
    split_bytes = Path(a.split).read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    with open(a.split) as fh:
        all_rows = list(csv.DictReader(fh, delimiter="\t"))
    rows = [r for r in all_rows if r["split"] == a.split_name]
    other_split = {(r["dataset"], r["clip"]) for r in all_rows if r["split"] != a.split_name}
    want = {(r["dataset"], r["clip"]): r for r in rows}
    if len(want) != len(rows):
        sys.exit(f"{TAG} FAIL split has duplicate (dataset, clip) rows")
    if a.lang == "template":
        text_of = None
    else:
        with open(a.lang) as fh:
            text_of = {(r["dataset"], r["clip"]): r["task_text"] for r in csv.DictReader(fh, delimiter="\t")}
    roots = []
    for item in a.roots.split(","):
        tag, path = item.split("=", 1)
        roots.append((tag, path))
    return split_sha, all_rows, other_split, want, text_of, roots


def _superseded_episodes(meta):
    """Episode indices listed in the meta superseded.jsonl (none when the file is absent)."""
    sp = os.path.join(meta, "superseded.jsonl")
    gone = set()
    if os.path.isfile(sp):
        with open(sp) as fh:
            gone = {int(json.loads(ln)["episode_index"]) for ln in fh if ln.strip()}
    return gone


def _split_row(a, want, other_split, in_split_any, unsplit, tag, ds, base):
    """The split-table row of the clip; with --include-unsplit a clip in no split gets an axis-less row (counted)."""
    r = want.get((ds, base))
    if r is None and a.include_unsplit and (ds, base) not in other_split and (ds, base) not in in_split_any:
        r = {
            "axis": "",
            "hand_use": "",
            "split": a.split_name,
        }  # outside the split table (in no split) — no axis
        unsplit[(tag, ds)] += 1
    return r


def _task_text(r, text_of, ds, base):
    """Task text of the clip: the axis template, or the per-clip text when a text table is given."""
    if text_of is None:
        text, src = (
            (TEMPLATES[r["axis"]], "template") if r["axis"] else ("", "none")
        )  # a row without an axis = no text (the text tool fills it)
    else:
        if (ds, base) not in text_of:
            sys.exit(f"{TAG} FAIL no task_text for {(ds, base)}")
        text, src = text_of[(ds, base)], "per_clip"
    return text, src


def _scan_roots(a, roots, all_rows, other_split, want, text_of, excl, tags) -> tuple:
    """Every episode of every root (superseded and excluded ones counted, not kept) -> the manifest rows."""
    excl_hit, tag_hit, excl_train = set(), set(), collections.Counter()
    in_split_any = {
        (r["dataset"], r["clip"]) for r in all_rows
    }  # --include-unsplit: clips in the table (any split) are decided by the table
    unsplit = collections.Counter()
    out_rows, seen_keys, found = [], set(), collections.Counter()
    dropped_sup = collections.Counter()
    for tag, path in roots:
        for ds in sorted(os.listdir(os.path.join(path, "data"))):
            meta = os.path.join(path, "data", ds, "collect", a.combo, "meta")
            ep_f = os.path.join(meta, "episodes.jsonl")
            if not os.path.isfile(ep_f):
                continue
            gone = _superseded_episodes(meta)
            with open(ep_f) as fh:
                for ln in fh:
                    if not ln.strip():
                        continue
                    e = json.loads(ln)
                    base, pv = base_of(e["clip"])
                    ek = (ds, tag, e["clip"], look_name(e))
                    if ek in excl:
                        excl_hit.add(ek)
                    if ek in tags:
                        tag_hit.add(ek)
                    r = _split_row(a, want, other_split, in_split_any, unsplit, tag, ds, base)
                    if r is None:
                        continue
                    if int(e["episode_index"]) in gone:
                        dropped_sup[(tag, ds)] += 1
                        continue
                    if ek in excl:
                        excl_train[(tag, ds)] += 1
                        continue
                    key = (ds, tag, e["clip"], str(e.get("round")), look_name(e))
                    if key in seen_keys:
                        sys.exit(f"{TAG} FAIL duplicate episode key {key}")
                    seen_keys.add(key)
                    found[(ds, base)] += 1
                    text, src = _task_text(r, text_of, ds, base)
                    out_rows.append(
                        {
                            "dataset": ds,
                            "root": tag,
                            "clip": e["clip"],
                            "round": str(e.get("round")),
                            "look": look_name(e),
                            "variant": pv,
                            "base_clip": base,
                            "axis": r["axis"],
                            "hand_use": r["hand_use"],
                            "split": r["split"],
                            "side": str(e.get("side")),
                            "video_frames": str(e.get("video_frames")),
                            "task_text": text,
                            "lang_source": src,
                            "verify_pending": ((tags[ek].get("tag") or a.tag_value) if ek in tags else ""),
                        }
                    )
    return out_rows, found, excl_hit, tag_hit, excl_train, unsplit, dropped_sup


def _check_lists(a, excl_lists, excl_hit, tags, tag_hit, want, found, out_rows) -> list:
    """Every exclude / tag row matched an episode, every split clip has one (or --allow-missing); rows sorted."""
    for x, d in excl_lists:
        if set(d) - excl_hit:
            sys.exit(
                f"{TAG} FAIL exclude list {x}: {len(set(d) - excl_hit)}/{len(d)} rows match no episode "
                f"(first {sorted(set(d) - excl_hit)[:3]})"
            )
    if set(tags) - tag_hit:
        sys.exit(f"{TAG} FAIL tag list: {len(set(tags) - tag_hit)}/{len(tags)} rows match no episode")
    missing = sorted(f"{d}/{c}" for (d, c) in want if found[(d, c)] == 0)
    if missing and not a.allow_missing:
        sys.exit(
            f"{TAG} FAIL {len(missing)} split clips have no episode (first: {missing[:5]}) — --allow-missing to list"
        )
    # tagged (verify_pending) rows go last: a later 'drop' is then the last episodes of the converted set
    out_rows.sort(key=lambda r: (r["verify_pending"] != "", r["dataset"], r["root"], r["clip"], r["round"], r["look"]))
    if a.tag and not all(
        r["verify_pending"] for r in out_rows if (r["dataset"], r["root"], r["clip"], r["look"]) in tags
    ):
        sys.exit(f"{TAG} FAIL a --tag row has no tag value (give --tag-value)")
    return missing


def _write_rows(a, out_rows, want, found) -> str:
    """<out>.tsv and <out>.counts.tsv; returns the manifest sha256."""
    with open(a.out + ".tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    man_sha = hashlib.sha256(Path(a.out + ".tsv").read_bytes()).hexdigest()
    counts = collections.Counter((r["axis"], r["dataset"], r["hand_use"]) for r in out_rows)
    with open(a.out + ".counts.tsv", "w") as fh:
        fh.write("axis\tdataset\thand_use\tepisodes\tclips\n")
        clips = collections.Counter()
        for (d, c), r in want.items():
            if found[(d, c)]:
                clips[(r["axis"], d, r["hand_use"])] += 1
        for k in sorted(counts):
            fh.write("\t".join(k) + f"\t{counts[k]}\t{clips[k]}\n")
    return man_sha


def _unsplit_meta(a, meta, unsplit, out_rows):
    """--include-unsplit: record and print the episodes whose base clip is in no split row."""
    if a.include_unsplit:  # the off mode has no such key (unchanged — besides the creation time)
        meta["include_unsplit"] = {
            "episodes": sum(unsplit.values()),
            "by_root_dataset": {f"{t}/{d}": n for (t, d), n in sorted(unsplit.items())},
            "rows_axis_empty": sum(1 for r in out_rows if r["axis"] == ""),
        }
        print(
            f"[UNSPLIT_OK] n={sum(unsplit.values())} episodes whose base clip is in no split row · axis '' "
            f"(rows {meta['include_unsplit']['rows_axis_empty']})",
            flush=True,
        )


def _manifest_meta(
    a,
    split_sha,
    roots,
    text_of,
    out_rows,
    want,
    missing,
    dropped_sup,
    excl_hit,
    excl_lists,
    excl_train,
    excl,
    other_split,
    tags,
    tag_hit,
    man_sha,
    unsplit,
):
    """The .meta.json record: inputs and hashes, counts by root/axis, list matches and the set-difference proofs."""
    meta = {
        "tool": "experiments/policies/data/manifest.py",
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "split": os.path.abspath(a.split),
        "split_sha256": split_sha,
        "split_sha12": split_sha[:12],
        "split_name": a.split_name,
        "roots": dict(roots),
        "combo": a.combo,
        "lang": a.lang,
        "templates": TEMPLATES if text_of is None else None,
        "episodes": len(out_rows),
        "clips_wanted": len(want),
        "clips_found": len(want) - len(missing),
        "missing": missing,
        "superseded_dropped": {f"{t}/{d}": n for (t, d), n in sorted(dropped_sup.items())},
        "exclude": [
            {
                "path": os.path.abspath(x),
                "sha256": hashlib.sha256(Path(x).read_bytes()).hexdigest(),
                "rows": len(d),
                "matched": len(set(d) & excl_hit),
                "in_split_before_exclusion": sum(1 for k in d if (k[0], base_of(k[2])[0]) in want),
            }
            for x, d in excl_lists
        ],
        "excluded_from_manifest": {f"{t}/{d}": n for (t, d), n in sorted(excl_train.items())},
        # set-difference proofs: no manifest row is an excluded key, none belongs to the other split
        "proof": {
            "rows_in_exclude": sum(1 for r in out_rows if (r["dataset"], r["root"], r["clip"], r["look"]) in excl),
            "rows_in_other_split": sum(1 for r in out_rows if (r["dataset"], r["base_clip"]) in other_split),
            "other_split_clips": len(other_split),
        },
        "tag": (
            {
                "path": os.path.abspath(a.tag),
                "sha256": hashlib.sha256(Path(a.tag).read_bytes()).hexdigest(),
                "rows": len(tags),
                "matched": len(tag_hit),
                "value": a.tag_value,
                "in_manifest": sum(1 for r in out_rows if r["verify_pending"]),
                "in_manifest_by_split_axis": dict(
                    collections.Counter(f"{r['split']}/{r['axis']}" for r in out_rows if r["verify_pending"])
                ),
            }
            if a.tag
            else None
        ),
        "manifest_sha256": man_sha,
        "by_axis": dict(collections.Counter(r["axis"] for r in out_rows)),
        "by_root": dict(collections.Counter(r["root"] for r in out_rows)),
    }
    _unsplit_meta(a, meta, unsplit, out_rows)
    return meta


def main() -> int:
    a = _args()

    split_sha, all_rows, other_split, want, text_of, roots = _read_inputs(a)

    excl_lists = [(x, keyed_list(x, roots, a.combo)) for x in a.exclude]
    excl = {k: r for _, d in excl_lists for k, r in d.items()}
    tags = keyed_list(a.tag, roots, a.combo) if a.tag else {}
    out_rows, found, excl_hit, tag_hit, excl_train, unsplit, dropped_sup = _scan_roots(
        a, roots, all_rows, other_split, want, text_of, excl, tags
    )
    missing = _check_lists(a, excl_lists, excl_hit, tags, tag_hit, want, found, out_rows)
    man_sha = _write_rows(a, out_rows, want, found)
    meta = _manifest_meta(
        a,
        split_sha,
        roots,
        text_of,
        out_rows,
        want,
        missing,
        dropped_sup,
        excl_hit,
        excl_lists,
        excl_train,
        excl,
        other_split,
        tags,
        tag_hit,
        man_sha,
        unsplit,
    )
    if meta["proof"]["rows_in_exclude"] or meta["proof"]["rows_in_other_split"]:
        sys.exit(f"{TAG} FAIL proof {meta['proof']}")
    with open(a.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(
        f"{TAG} split_sha={split_sha[:12]} {a.split_name} clips {len(want) - len(missing)}/{len(want)} "
        f"episodes={len(out_rows)} superseded_dropped={sum(dropped_sup.values())} "
        f"excluded={sum(excl_train.values())} (list {len(excl_hit)}/{len(excl)} matched) "
        f"tagged={sum(1 for r in out_rows if r['verify_pending'])} manifest_sha={man_sha[:12]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
