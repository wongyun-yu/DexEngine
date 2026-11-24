"""E5 clip curation inputs: one row per wuji2 episode of the final collection roots (original, augmentation) —
visibility, length, look. Read only; the output is one gzip TSV on stdout.
Visibility = the rendered measurement visibility{object, hand}{in_frame, visible} (left eye, occlusion included) and the
superseded flag. Usage: python -m experiments.sim_eval.scenes.clip_vis --root orig=<path> --root aug=<path> >
vis.tsv.gz"""

import argparse
import gzip
import json
import os
import re
import sys

from experiments.textio import read_lines

PV = re.compile(r"_pv(\d+)$")
COLS = [
    "root",
    "ds",
    "clip",
    "base",
    "pv",
    "episode_index",
    "superseded",
    "length",
    "round",
    "look",
    "obj_in",
    "obj_vis",
    "obj_pts",
    "hand_in",
    "hand_vis",
    "vis_frames",
    "head1_deg",
    "cam_z",
    "f0_hold_steps",
    "src_window",
]


def main() -> None:
    """Write the gzip TSV of all episodes of the given roots to stdout."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        action="append",
        required=True,
        metavar="TAG=PATH",
        help="collection root holding data/<ds>/collect/rby1_wuji2 (repeat; e.g. orig=… aug=…)",
    )
    roots = [(v.split("=", 1)[1], v.split("=", 1)[0]) for v in ap.parse_args().root]
    with gzip.open(sys.stdout.buffer, "wt") as out:
        out.write("\t".join(COLS) + "\n")
        n = 0
        for root, tag in roots:
            base_dir = os.path.join(root, "data")
            for ds in sorted(os.listdir(base_dir)):
                f = os.path.join(base_dir, ds, "collect", "rby1_wuji2", "meta", "episodes.jsonl")
                if not os.path.isfile(f):
                    continue
                sp = os.path.join(os.path.dirname(f), "superseded.jsonl")
                gone = (
                    {int(json.loads(x)["episode_index"]) for x in read_lines(sp) if x.strip()}
                    if os.path.isfile(sp)
                    else set()
                )
                with open(f) as fh:
                    for ln in fh:
                        if not ln.strip():
                            continue
                        e = json.loads(ln)
                        c = e["clip"]
                        t = c[:-6] if c.endswith("_pl_rt") else c
                        m = PV.search(t)
                        v = e.get("visibility") or {}
                        o, h = v.get("object") or {}, v.get("hand") or {}
                        look = e.get("look") or {}
                        hj = (e.get("head_joints") or {}).get("head_1") or {}
                        cam = ((e.get("camera_pose") or {}).get("ego_left") or {}).get("pos") or [None, None, None]
                        cs = e.get("case") if isinstance(e.get("case"), dict) else {}
                        row = [
                            tag,
                            ds,
                            c,
                            PV.sub("", t),
                            m.group(1) if m else "",
                            e["episode_index"],
                            int(int(e["episode_index"]) in gone),
                            e.get("length"),
                            e.get("round"),
                            look.get("name") if isinstance(look, dict) else look,
                            o.get("in_frame"),
                            o.get("visible"),
                            o.get("points"),
                            h.get("in_frame"),
                            h.get("visible"),
                            v.get("frames"),
                            hj.get("render_deg"),
                            cam[2],
                            (e.get("f0_hold") or {}).get("steps"),
                            json.dumps(cs.get("source_window")),
                        ]
                        out.write("\t".join("" if x is None else str(x).replace("\t", " ") for x in row) + "\n")
                        n += 1
    print(f"[CLIP_VIS] {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
