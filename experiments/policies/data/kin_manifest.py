#!/usr/bin/env python3
"""The kinematic conversion list of one E5 formal task = OUR training scenes, row for row, on the kinematic roots.

For every row of our text-filled list (text_<v>/manifest_<task>.tsv) the same scene key (dataset · clip · round ·
look) is looked up in the kinematic roots' meta/episodes.jsonl (not superseded): our root orig → kin_orig, aug →
kin_aug; a key found in no main root but in an extra root (a third --roots entry, e.g. kin_aug_s1 = re-rendered
scenes) goes there. A key found in two roots, or in none, dies. Kept: every column of ours except root ·
video_frames (the kinematic episode's) · task_text and lang_source emptied (the builder's text step fills them from
ours with --pair) · new columns pair_root · pair_round = OUR root · round (the text tool's pair key). Order = ours.
Usage: python -m experiments.policies.data.kin_manifest --ours <manifest_task.tsv>
  --roots kin_orig=<r>,kin_aug=<r>[,kin_x=<r>] --out <tsv>
Marker [KIN_MANIFEST] DONE | FAIL."""

import argparse
import collections
import csv
import json
import os
import re
import sys


def tsv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def lines(p):
    with open(p) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def die(m):
    sys.exit(f"[KIN_MANIFEST] FAIL {m}")


def keys(root, ds, cache):
    b = f"{root}/data/{ds}/collect/rby1_wuji2/meta"
    if b not in cache:
        out = {}
        if os.path.isdir(b):
            sp = f"{b}/superseded.jsonl"
            gone = {int(x["episode_index"]) for x in lines(sp)} if os.path.isfile(sp) else set()
            for e in lines(f"{b}/episodes.jsonl"):
                if int(e["episode_index"]) in gone:
                    continue
                lk = e.get("look")
                k = (e["clip"], str(e.get("round")), str(lk.get("name") if isinstance(lk, dict) else (lk or "")))
                if k in out:
                    die(f"{b}: key {k} has two episodes")
                out[k] = e
        cache[b] = out
    return cache[b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cross-round",
        action="store_true",
        help="an ORIGINAL-placement scene of our aug round (clip without _pv<k>) absent from kin_aug may "
        "take the kin_orig episode of the same clip, round orig and look (the kinematic set holds "
        "that placement once, under orig: Pour 1, Lift 4); default off = dies; rows printed",
    )
    a = ap.parse_args()
    R = dict(x.split("=", 1) for x in a.roots.split(","))
    if list(R)[:2] != ["kin_orig", "kin_aug"]:
        die(f"--roots: the first two must be kin_orig · kin_aug ({list(R)})")
    extra = list(R)[2:]
    if os.path.exists(a.out):
        die(f"{a.out} already exists")
    rows = tsv(a.ours)
    cache, out, where = {}, [], collections.Counter()
    for r in rows:
        k = (r["clip"], r["round"], r["look"])
        main_root = {"orig": "kin_orig", "aug": "kin_aug"}.get(r["root"]) or die(f"unknown root of ours {r['root']!r}")
        hit = [(n, k) for n in [main_root] + extra if k in keys(R[n], r["dataset"], cache)]
        if not hit and a.cross_round and r["round"] == "aug" and not re.search(r"_pv\d+_pl_rt$", r["clip"]):
            k2 = (r["clip"], "orig", r["look"])
            hit = [("kin_orig", k2)] if k2 in keys(R["kin_orig"], r["dataset"], cache) else []
            if hit:
                print(
                    f"[KIN_MANIFEST] other round {r['dataset']} {r['clip']} · our aug/aug → kin_orig/orig "
                    f"(same clip · source placement · {r['look']})",
                    flush=True,
                )
                where["cross_round"] += 1
        if len(hit) != 1:
            die(f"{r['dataset']} {k}: kinematic roots {[h[0] for h in hit] or 'none'}")
        ((n, kk),) = hit
        e = keys(R[n], r["dataset"], cache)[kk]
        out.append(
            dict(
                r,
                root=n,
                round=kk[1],
                video_frames=str(e["video_frames"]),
                task_text="",
                lang_source="",
                pair_root=r["root"],
                pair_round=r["round"],
            )
        )
        where[n] += 1
    with open(a.out + ".tmp", "x", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=list(rows[0]) + ["pair_root", "pair_round"], delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(out)
    os.replace(a.out + ".tmp", a.out)
    dvf = sum(int(o["video_frames"]) != int(r["video_frames"]) for o, r in zip(out, rows, strict=True))
    print(
        f"[KIN_MANIFEST] DONE {a.out} · rows {len(out)} = ours {len(rows)} · roots {dict(where)} · "
        f"frames differ from ours {dvf} · kinematic frames {sum(int(o['video_frames']) for o in out)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
