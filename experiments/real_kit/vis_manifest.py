#!/usr/bin/env python3
"""DP visual-augmentation fine-tuning set manifest: the chosen clips' ours train variants × 17 looks.

  rows = the ours train manifest's rows of the chosen base clips (look D · the successful placement variants the ours
  DP
         trained on — sim L1's 5 held-out placements are not among them; checked against the L1 list) + every render
         episode of those same variants (16 looks r<k>_s<nn>), one root tag per render piece.
  text and labels (axis · hand_use · side · task_text · lang_source) of a render row = its variant's ours D row.
  Dies when a variant lacks a look, a look repeats, an L1 placement is in the set, or a chosen clip has no ours row.
  --looks N (default 17): object-colour looks (17 + red · green · blue = 20) are render episodes named
  r<k>_s<nn>_<color>
  whose look spec carries obj_color (3 floats); a plain r<k>_s<nn> look must not carry it.
  --d-in-pieces: look D rows come from the render pieces too (a D re-render run · one episode look "D" per variant ·
  no
  obj_color · frames = the ours row); --d-from/--d-to are then refused. Without it the D rows come from the ours roots.
  --drop-looks a,b (real-print set: v1 minus red + the new 16 = 35): those looks stay out of the set · a named look
  that
  no
  render episode has = death. Print looks r<k>_s<nn>_ptex carry obj_tint (3 floats) and no obj_color.

  python -m experiments.real_kit.vis_manifest --ours <ours_train.tsv> --ours-roots <its .roots> --d-from <render dir>
      --d-to <local dir> --pieces <extracted pieces: <piece>/out_<p>/root> --clips a,b,c --l1 <lists/<task>_l1.txt>
      --out <manifest.tsv>      → <manifest.tsv> · <manifest.tsv>.roots · <manifest.tsv>.counts.json
Last line [E5_VISMAN] OK|FAIL.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

from experiments.policies.data.convert import look_name  # the converter's own key

TAG = "[E5_VISMAN]"
LOOKS = 17


def die(m: str) -> None:
    print(f"{TAG} FAIL {m}", flush=True)
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    for k in ("--ours", "--ours-roots", "--pieces", "--clips", "--l1", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--d-from")
    ap.add_argument("--d-to")
    ap.add_argument("--d-in-pieces", action="store_true")
    ap.add_argument("--combo", default="rby1_wuji2")
    ap.add_argument("--looks", type=int, default=LOOKS)
    ap.add_argument("--drop-looks", default="")
    ap.add_argument(
        "--print-rules",
        default="",
        # real-print set rules JSON (default none; vis_rules.py): dropped variants stay out, print looks' obj_keep = the
        # table, else death
    )
    a = ap.parse_args()
    if a.d_in_pieces != (a.d_from is None and a.d_to is None) or (not a.d_in_pieces and not (a.d_from and a.d_to)):
        die("look D source: --d-in-pieces, or both --d-from and --d-to")
    return a


def _ours_rows(ours, clips, rules):
    """Ours train rows of the chosen clips (print-rules drops applied); all look D, every clip present."""
    drows = [r for r in ours if r["base_clip"] in clips]
    if rules:
        n0 = len(drows)
        drows = [r for r in drows if r["clip"] not in rules["drop"]]
        print(
            f"{TAG} [PRINT_RULES] on · kept objects {rules['keep']} · dropped-variant table "
            f"{len(rules['drop'])} · dropped from this set's ours {n0 - len(drows)}",
            flush=True,
        )
    if {r["look"] for r in drows} != {"D"}:
        die(f"ours rows of the chosen clips have looks {sorted({r['look'] for r in drows})} (want D only)")
    miss = [c for c in clips if not any(r["base_clip"] == c for r in drows)]
    if miss:
        die(f"no ours train row for {miss}")
    return drows


def _variants_and_l1(drows, a):
    """Ours rows by variant (each once) and the L1 placement names (none may be an ours variant)."""
    by_var = {}
    for r in drows:
        if r["clip"] in by_var:
            die(f"ours variant {r['clip']} twice")
        by_var[r["clip"]] = r
    with open(a.l1) as fh:
        l1 = {re.sub(r"\.[^./]+/job\.json$", "", x.strip()).rsplit("/", 1)[-1] for x in fh if x.strip()}
    bad = sorted(set(by_var) & l1)
    if bad:
        die(f"L1 placements in the ours variants {bad[:3]}")
    return by_var, l1


def _d_rows(a, drows, roots):
    """Look D rows from the ours root moved to this machine (none when D comes from the pieces)."""
    out_roots, rows = {}, []
    for r in () if a.d_in_pieces else drows:  # D rows: the ours root, moved to this machine
        p = roots[r["root"]]
        if not p.startswith(a.d_from.rstrip("/") + "/"):
            die(f"ours root {p} not under --d-from {a.d_from}")
        out_roots[r["root"]] = os.path.join(a.d_to, p[len(a.d_from.rstrip("/")) + 1 :])
        rows.append(dict(r))
    return out_roots, rows


def _piece_root(a, piece):
    """The single out_* root of one render piece and its single dataset."""
    sub = [d for d in os.listdir(os.path.join(a.pieces, piece)) if d.startswith("out_")]
    if len(sub) != 1:
        die(f"{piece}: out_* folders {sub}")
    root = os.path.join(a.pieces, piece, sub[0], "root")
    dss = os.listdir(os.path.join(root, "data"))
    if len(dss) != 1:
        die(f"{piece}: datasets {dss}")
    return root, dss


def _look_spec(e, lk, a, piece):
    """Look spec of one render episode; dies unless the look is plain, painted, tinted or a D redo."""
    sp = ((e.get("look") or {}).get("spec") or {}) if isinstance(e.get("look"), dict) else {}
    oc, ot = sp.get("obj_color"), sp.get("obj_tint")
    plain = re.fullmatch(r"r\d+_s\d+", lk) is not None and oc is None and ot is None
    painted = (
        re.fullmatch(r"r\d+_s\d+_[a-z]+", lk) is not None
        and isinstance(oc, list)
        and len(oc) == 3
        and all(isinstance(x, (int, float)) for x in oc)
    )
    tinted = (
        re.fullmatch(r"r\d+_s\d+_ptex", lk) is not None
        and oc is None
        and isinstance(ot, list)
        and len(ot) == 3
        and all(isinstance(x, (int, float)) for x in ot)
    )
    redo = lk == "D" and a.d_in_pieces and oc is None
    if not (plain or painted or tinted or redo):
        die(f"{piece}: look '{lk}' of {e['clip']} (obj_color {oc} obj_tint {ot})")
    return sp, redo


def _check_rows(drop_looks, dropped, rules, rows, seen, a):
    """Every dropped look was seen, no dropped variant, no repeated look, every variant has all looks."""
    if drop_looks - set(dropped):
        die(f"--drop-looks {sorted(drop_looks - set(dropped))} not in any render episode")
    if rules and {r["clip"] for r in rows} & rules["drop"]:
        die(f"dropped variants in the set {sorted({r['clip'] for r in rows} & rules['drop'])}")
    dup = [k for k, v in seen.items() if v > 1]
    if dup:
        die(f"look repeats {dup[:3]}")
    per = collections.Counter(r["clip"] for r in rows)
    NL = a.looks
    short = sorted((v, n) for v, n in per.items() if n != NL)
    if short:
        die(f"variants without {NL} looks (render not finished?) {short[:5]} · {len(short)} variants")
    looks = sorted({r["look"] for r in rows})
    if len(looks) != NL:
        die(f"distinct looks {len(looks)} ≠ {NL}: {looks}")
    return NL, looks


def _read_ours(a):
    """The chosen clips (no repeats), the ours table rows and columns, and its root map."""
    clips = [c for c in a.clips.split(",") if c]
    if len(set(clips)) != len(clips):
        die(f"clips repeat {clips}")
    with open(a.ours) as fh:
        ours = list(csv.DictReader(fh, delimiter="\t"))
    cols = list(ours[0])
    roots = dict(x.split("=", 1) for x in Path(a.ours_roots).read_text().strip().split(","))
    return clips, ours, cols, roots


def main() -> int:
    a = _parse_args()
    clips, ours, cols, roots = _read_ours(a)
    rules = None
    if a.print_rules:
        from experiments.real_kit import vis_rules as VR

        rules = VR.load(a.print_rules)
    drows = _ours_rows(ours, clips, rules)
    by_var, l1 = _variants_and_l1(drows, a)
    out_roots, rows = _d_rows(a, drows, roots)
    seen = collections.Counter()
    skipped = collections.Counter()
    drop_looks = {x for x in a.drop_looks.split(",") if x}
    dropped = collections.Counter()
    for piece in sorted(os.listdir(a.pieces)):
        base = piece.split("@")[0]
        if base not in clips:
            continue
        root, dss = _piece_root(a, piece)
        tag = "v_" + re.sub(r"[^A-Za-z0-9_]", "_", piece)
        out_roots[tag] = root
        meta = os.path.join(root, "data", dss[0], "collect", a.combo, "meta", "episodes.jsonl")
        with open(meta) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                e = json.loads(ln)
                d = by_var.get(e["clip"])
                if d is None:
                    skipped[e["clip"]] += 1
                    continue
                lk = look_name(e)
                if lk in drop_looks:
                    dropped[lk] += 1
                    continue
                # look D carries spec null
                sp, redo = _look_spec(e, lk, a, piece)
                if rules:
                    why = VR.check_episode(d["base_clip"], e["clip"], lk, sp, rules)
                    if why:
                        die(f"{piece}: {why}")
                if redo and int(e["video_frames"]) != int(d["video_frames"]):
                    die(f"{piece}: look D of {e['clip']} frames {e['video_frames']} ≠ ours {d['video_frames']}")
                if d["dataset"] != dss[0]:
                    die(f"{piece}: dataset {dss[0]} ≠ ours {d['dataset']}")
                seen[(e["clip"], lk)] += 1
                rows.append(dict(d, root=tag, round=str(e.get("round")), look=lk, video_frames=str(e["video_frames"])))
    NL, looks = _check_rows(drop_looks, dropped, rules, rows, seen, a)
    _write_outputs(a, cols, rows, out_roots, by_var, clips, skipped, l1, NL, looks)
    return 0


def _write_outputs(a, cols, rows, out_roots, by_var, clips, skipped, l1, NL, looks) -> None:
    """The manifest TSV, its .roots and .counts.json, and the OK line."""
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    used = sorted({r["root"] for r in rows})
    Path(a.out + ".roots").write_text(",".join(f"{t}={out_roots[t]}" for t in used) + "\n")
    pc = {
        c: {
            "variants": sum(1 for v in by_var if by_var[v]["base_clip"] == c),
            "episodes": sum(1 for r in rows if r["base_clip"] == c),
        }
        for c in clips
    }
    res = {
        "clips": pc,
        "episodes": len(rows),
        "frames": sum(int(r["video_frames"]) for r in rows),
        "looks": looks,
        "roots": len(used),
        "skipped_variants": dict(skipped),
        "l1_checked": len(l1),
    }
    with open(a.out + ".counts.json", "w") as fh:
        json.dump(res, fh, indent=1)
    per_clip = " · ".join(f"{c} {v['variants']}×{NL}" for c, v in pc.items())
    print(
        f"{TAG} OK {len(rows)} episodes = Σ variants × {NL} looks ({per_clip}) · frames {res['frames']} · roots "
        f"{len(used)} · render episodes of other variants skipped {sum(skipped.values())} → {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
