"""E5 clip curation: one feature row per clip, the same function for the E5 clip list and the kinematic pool.

Inputs: clip list TSVs (wuji2_e5_clips.tsv, kinematic_pool.tsv[in_pool=1]), the data root (--data-root, read only, no
default) and the episode visibility table (clip_vis.py output — rendered measurement; empty cells without it).
Material = base clips (data/<ds>/base/<hand>/<emb>/<clip>/0/trajectory_kinematic.npz — object poses, fingertip contact)
and task_info (objects, support).
Frame = base frame (z up); row rate = npz frequency. Objects = role main only (background and extras excluded).
Features (moved object = the main object with the largest 3D displacement):
  len_s · rise_max_m · rise_end_m · hdisp_end_m (start→end horizontal) · hpath_max_m · tilt_max_deg (tool = moved
  object) ·
  f0: fingertips in contact (that object's hand) · bottom height (object lowest point − desk z) · speed over the
  first
  0.1 s ·
  end: speed · bottom height · fingertips in contact ·
  sweep contact slide (judge_v1.contact_travel — the judge's own function; tool/target = judge_v1.tool_target;
  target-rule
  version in the tgt_cand column) ·
  sideways slide on the target's top face (top_slide — task rule (1) · slide_top_m · slide_top_np_m · top_axis ·
  top_h_m).
Visibility: min · median of visibility.object.in_frame · visible (left-eye render measurement) over all training
episodes of the clip, and the original-stage value.
Output: TSV (--out) · marker [CLIP_FEAT] with a count match (unique input clips = output rows; rc 1 unless failures are
0)."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.sim_eval.judges import judge_v1 as J
from experiments.sim_eval.judges.judge_v1_pair import contact_points, target_guard_hit
from experiments.sim_eval.scenes.clip_motion import DC_M, low_z, over_target, sup_slide, tips_of

MARK = "[CLIP_FEAT]"
HAND = "wuji2"
COLS = [
    "ds",
    "clip",
    "emb",
    "label",
    "axis",
    "src",
    "T",
    "hz",
    "len_s",
    "n_main",
    "tags",
    "moved",
    "rise_max_m",
    "rise_end_m",
    "hdisp_end_m",
    "hpath_max_m",
    "disp3_max_m",
    "tilt_max_deg",
    "f0_tips",
    "f0_low_m",
    "f0_speed",
    "end_speed",
    "end_low_m",
    "end_tips",
    "max_tips",
    "desk_z",
    "desk_mode",
    "other_rise_max_m",
    "slide_m",
    "pair",
    "dmin_min_m",
    "tgt_r_m",
    "over_tilt_max_deg",
    "pour_over_s",
    "dtilt_over_s",
    "slide_np_m",
    "stir_in_s",
    "stir_turns",
    "stir_net_turns",
    "tgt_h_m",
    "deep_in_s",
    "slide_deep_m",
    "slide_np_nd_m",
    "sup_slide_m",
    "tool_id",
    "tgt_id",
    "slide_tray_m",
    "tgt_tag",
    "tgt_cand",
    "tgt_attached_skip",
    "slide_top_m",
    "slide_top_np_m",
    "top_axis",
    "top_h_m",
    "top_contact_frac",
    "vis_in_min",
    "vis_in_med",
    "vis_vis_min",
    "vis_vis_med",
    "vis_n",
    "vis_d_in",
    "err",
]


def lit(v):
    """Parse a dict/list literal stored as text; {} on failure."""
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return {}
    return v if v is not None else {}


def one(job: tuple) -> dict:
    """Feature row of one clip; a failure is written into the row's err column (counted at the end)."""
    ds, clip, emb, label, axis, src, base = job
    row = {"ds": ds, "clip": clip, "emb": emb, "label": label, "axis": axis, "src": src, "tgt_cand": J.TARGET_RULE}
    try:
        bd = Path(base)
        ti, objs = J.task_objects(bd)
        kin, hz = J.kin_scene(bd, objs)
        z = np.load(bd / "0" / "trajectory_kinematic.npz")
        contact = (z["contact"] > 0).astype(float) if "contact" in z.files else np.zeros((kin.T, 10))
        sup = lit(ti.get("support"))
        desk = sup.get("desk_z") if isinstance(sup, dict) else None
        mains = [n for n, v in kin.o.items() if v["meta"].get("role") == "main"]
        if not mains:  # no manipulated object (all support/background) — a feature (rule M candidate), not a failure
            row.update(
                T=kin.T,
                hz=hz,
                len_s=round(kin.T / hz, 3),
                n_main=0,
                tags="|".join(str(v["meta"].get("tag")) for v in kin.o.values()),
            )
            return row
        d3 = {n: float(np.linalg.norm(kin.o[n]["p"] - kin.o[n]["p"][0], axis=1).max()) for n in mains}
        mv = max(mains, key=d3.get)
        o = kin.o[mv]
        p, R = o["p"], o["R"]
        lin, _ = J.speeds(p, R, hz)
        k = max(1, int(round(0.1 * hz)))
        lz = low_z(ds, o, p, R)
        tips = tips_of(contact, list(ti.get("hands") or []), o["meta"].get("side") or "")
        rise = p[:, 2] - p[0, 2]
        oth = [float((kin.o[n]["p"][:, 2] - kin.o[n]["p"][0, 2]).max()) for n in mains if n != mv]
        row.update(
            T=kin.T,
            hz=hz,
            len_s=round(kin.T / hz, 3),
            n_main=len(mains),
            tags="|".join(str(kin.o[n]["meta"].get("tag")) for n in mains),
            moved=mv,
            rise_max_m=round(float(rise.max()), 4),
            rise_end_m=round(float(rise[-1]), 4),
            hdisp_end_m=round(float(np.linalg.norm(p[-1, :2] - p[0, :2])), 4),
            hpath_max_m=round(float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max()), 4),
            disp3_max_m=round(d3[mv], 4),
            tilt_max_deg=round(float(J.tilt_deg(R).max()), 2),
            f0_tips=int(tips[0]),
            f0_low_m=(round(float(lz[0] - desk), 4) if desk is not None else ""),
            f0_speed=round(float(lin[1 : k + 1].mean()), 4),
            end_speed=round(float(lin[-k:].mean()), 4),
            end_low_m=(round(float(lz[-1] - desk), 4) if desk is not None else ""),
            end_tips=int(tips[-1]),
            max_tips=int(tips.max()),
            desk_z=("" if desk is None else round(float(desk), 4)),
            desk_mode=(sup.get("desk_mode") if isinstance(sup, dict) else ""),
            other_rise_max_m=(round(max(oth), 4) if oth else ""),
            tool_id=str(o["meta"].get("obj_id", "")),
            sup_slide_m=sup_slide(p, lz, desk),
        )
        pair = J.tool_target(kin)  # target rule = the judge's own function (J.TARGET_RULE)
        row["tgt_attached_skip"] = int(target_guard_hit(kin))  # target changed by the attached-object guard (counted)
        if pair:
            row["tgt_id"] = str(kin.o[pair[1]]["meta"].get("obj_id", ""))
            row["tgt_tag"] = str(kin.o[pair[1]]["meta"].get("tag", ""))
            pts = contact_points(kin, pair[0], pair[1], ds)
            L, dmin, cum = J.contact_travel(kin, pair[0], pair[1], ds, DC_M, pts=pts)
            row.update(
                slide_m=round(L, 4),
                pair="/".join(pair),
                dmin_min_m=round(float(dmin.min()), 4),
                **over_target(kin, pair[0], pair[1], ds, hz, cum, pts=pts, dc=DC_M),
            )
    except Exception as e:  # noqa: BLE001 — failures go into the row and are counted at the end (rc 1)
        row["err"] = f"{type(e).__name__}: {e}"[:200]
    return row


def base_of(root: Path, ds: str, emb: str, clip: str) -> Path:
    """Base clip folder."""
    return root / "data" / ds / "base" / HAND / emb / clip


def _parse_args():
    """Command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", type=Path, required=True)
    ap.add_argument("--kin", type=Path, required=True)
    ap.add_argument("--vis", type=Path, required=True, help="clip_vis.py output (gzip TSV)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument(
        "--extra",
        type=Path,
        nargs="*",
        default=[],
        help="clip lists of the same shape (wuji2_e5_clips columns) — unassigned clips etc. · src extra",
    )
    a = ap.parse_args()
    return a


def _collect_jobs(a, root):
    """One job per (dataset, clip) from the kin cells, the E5 list and the extra lists."""
    with open(a.kin) as fh:
        kin = list(csv.DictReader(fh, delimiter="\t"))
    emb_of = {(r["ds"], r["clip"]): r["emb"] for r in kin}
    jobs = {}
    with open(a.e5) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            key = (r["dataset"], r["clip"])
            emb = emb_of.get(key)
            # clip not in the kinematic table — find its base folder (must be under exactly one emb folder)
            if emb is None:
                hits = sorted((root / "data" / key[0] / "base" / HAND).glob(f"*/{key[1]}"))
                if len(hits) != 1:
                    raise SystemExit(f"{MARK} {key} base folders {len(hits)} — exactly one expected")
                emb = hits[0].parent.name
            jobs[key] = [
                r["dataset"],
                r["clip"],
                emb,
                r["source_label"],
                r["axis"],
                "e5",
                str(base_of(root, key[0], emb, key[1])),
            ]
    for f in a.extra:
        with open(f) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                key = (r["dataset"], r["clip"])
                if key in jobs:
                    continue
                hits = sorted((root / "data" / key[0] / "base" / HAND).glob(f"*/{key[1]}"))
                if len(hits) != 1:
                    raise SystemExit(f"{MARK} {key} base folders {len(hits)} — exactly one expected")
                jobs[key] = [
                    r["dataset"],
                    r["clip"],
                    hits[0].parent.name,
                    r["source_label"],
                    r["axis"],
                    "extra",
                    str(hits[0]),
                ]
    for r in kin:
        if r["in_pool"] != "1":
            continue
        key = (r["ds"], r["clip"])
        if key in jobs:
            jobs[key][5] = "e5+kin"
            continue
        jobs[key] = [r["ds"], r["clip"], r["emb"], r["label"], r["axis"], "kin", str(root / r["base_dir"])]
    return jobs


def _read_vis(a):
    """Per-clip visibility rows of clip_vis.py."""
    vis: dict = {}
    vis_d: dict = {}
    with gzip.open(a.vis, "rt") as fh:
        for v in csv.DictReader(fh, delimiter="\t"):
            if v["superseded"] != "0" or not v["obj_in"]:
                continue
            vis.setdefault((v["ds"], v["base"]), []).append((float(v["obj_in"]), float(v["obj_vis"])))
            if v["root"] == "orig" and v["look"] == "D" and not v["pv"]:
                vis_d[(v["ds"], v["base"])] = float(v["obj_in"])
    return vis, vis_d


def _write_rows(a, rows, jobs):
    """Write the feature table and report failures; rc 1 unless every input clip has one failure-free row."""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r.get("err")]
    keys = {(r["ds"], r["clip"]) for r in rows}
    print(f"{MARK} rows {len(rows)} · unique {len(keys)} · input {len(jobs)} · failed {len(bad)} → {a.out}", flush=True)
    print(
        f"{MARK} rows whose target the attached-object guard changed "
        f"{sum(1 for r in rows if r.get('tgt_attached_skip') == 1)}",
        flush=True,
    )
    for r in bad[:10]:
        print(f"{MARK} FAIL {r['ds']} {r['clip']}: {r['err']}", flush=True)
    return 0 if (not bad and len(keys) == len(jobs) == len(rows)) else 1


def main() -> int:
    """CLI entry; rc 1 unless every input clip has one failure-free row."""
    a = _parse_args()
    print(
        f"{MARK} target_rule={J.TARGET_RULE} — target = judge_v1.tool_target (the judge's function · "
        "background included · "
        f"objects attached to the tool excluded)",
        flush=True,
    )
    root = J.set_data_root(a.data_root)
    jobs = _collect_jobs(a, root)
    # visibility — render measurement over all non-superseded training episodes of the base clip (both roots, all looks
    # and
    # placement variants). vis_d_in = the original-stage episode value (root orig · look D · no variant) when present.
    vis, vis_d = _read_vis(a)
    print(
        f"{MARK} clips {len(jobs)} (e5 {sum(j[5] != 'kin' for j in jobs.values())} · kinematic only "
        f"{sum(j[5] == 'kin' for j in jobs.values())}) · visibility-table clips {len(vis)} · jobs {a.jobs}",
        flush=True,
    )
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, [tuple(j) for j in jobs.values()], chunksize=8))
    for r in rows:
        vv = vis.get((r["ds"], r["clip"]))
        if vv:
            a_in, a_vis = np.array([x[0] for x in vv]), np.array([x[1] for x in vv])
            r.update(
                vis_in_min=round(float(a_in.min()), 4),
                vis_in_med=round(float(np.median(a_in)), 4),
                vis_vis_min=round(float(a_vis.min()), 4),
                vis_vis_med=round(float(np.median(a_vis)), 4),
                vis_n=len(vv),
                vis_d_in=vis_d.get((r["ds"], r["clip"]), ""),
            )
    return _write_rows(a, rows, jobs)


if __name__ == "__main__":
    sys.exit(main())
