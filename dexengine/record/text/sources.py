#!/usr/bin/env python3
"""Input records of the clip instructions (CPU): per target clip one JSON line with the source label, objects,
hands, scene, measured motion and (optionally) render frames.

Input: the union of RL clip-owner tables (rl_clip_owner.tsv per hand, completion rate >= --min-rate) or --pairs,
and --data-root (explicit, no default). A clip is hand-independent: its tree is the first by name of
data/<ds>/trees/<hand combo>/<emb>/<clip>_pl_rt/. Source label = the tree's task_info.json reference field, parsed
by SOURCE_RULES (unknown dataset dies). Motion summary = object poses of the tree's 0/trajectory_kinematic.npz
(qpos[:, qpos_start:+7] = xyz + wxyz). Marker [COLLECT_TEXT_SRC]; last line OK n=<records>; clips without a tree go
to missing.tsv.

Phase table (phases): [{phase, start_s, end_s}] per record + phases_src.
Frames (optional, --with-frames): row 0 x look-D episode of the clip in the stored datasets (--dataset-glob; video
keys store.store.VIDEO_KEYS; several hands -> the first by order.source_rank). Final datasets (data/<ds>/collect/
<combo>) are read as they are; a piece-layer dataset must match the current tree (tree_ref fingerprint). The phase
table is measured **on that episode** (phases.from_episode, same time axis as the frames); per view (--views,
default exo,egoL) the frame plan (phases.text_picks, TEXT_FRAMES) is extracted at --frame-size as JPEG (ffmpeg)
-> record field frames = [{path, view, t_s, w, h, phase, at}] + frames_src. Clips without a usable render get
frames = [], frames_missing = reason, and a phase table from the source reference motion
(phases.from_reference); they are listed in frames_missing.tsv (--require-frames dies on any).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.record import phases as PH
from dexengine.record.text import motion as MOT
from dexengine.record.text import vocab as CN
from dexengine.record.text.finals import final_wait, is_final, tree_fresh
from dexengine.record.text.motion import motion_summary
from dexengine.record.text.source_labels import SOURCE_RULES, case_now, hands_of

MARK = "[COLLECT_TEXT_SRC]"
STEP = "dexengine.record.text.sources"  # capacity.NEEDS key: concurrent ffmpeg frame grabs per clip


def die(msg: str) -> None:
    """Print a FAIL line and exit with code 2."""
    print(f"{MARK} FAIL {msg}", flush=True)
    raise SystemExit(2)


def handover_picks(hand: list[dict]) -> list[dict]:
    """Remaining handover windows (record field handover) -> frame times (start, mid, end; phase name handover), so
    the frames show the moment of the pass."""
    out = []
    for h in hand:
        a, b = (float(x) for x in h["time_s"])
        out += [
            {"t_s": a, "phase": "handover", "at": "start"},
            {"t_s": (a + b) / 2, "phase": "handover", "at": "mid"},
            {"t_s": b, "phase": "handover", "at": "end"},
        ]
    return out


def merge_picks(picks: list[dict], extra) -> list[dict]:
    """Phase frames + handover frames in time order (in the same 0.01 s slot the phase frame stays)."""
    seen = {round(float(p["t_s"]), 2) for p in picks}
    more = [e for e in extra if round(float(e["t_s"]), 2) not in seen]
    return sorted([*picks, *more], key=lambda p: float(p["t_s"]))


# ───────────────────── objects, hands, motion ─────────────────────
def record(ds: str, clip: str, tree: Path, data_root: Path) -> dict:
    """The input record of one clip from its tree (no frames)."""
    ti = json.loads((tree / "task_info.json").read_text(encoding="utf-8"))
    rule = SOURCE_RULES.get(ds)
    if rule is None:
        die(f"unknown dataset {ds!r}: add a rule to SOURCE_RULES")
    src = rule(ti, clip)
    mo = motion_summary(tree, ti)
    moved = {r["obj"]: r for r in (mo or {}).get("objects", [])}
    objs = []
    for o in ti.get("objects") or []:
        name, flags = CN.readable(ds, str(o["obj_id"]), clip, data_root)
        objs.append(
            dict(
                name=name,
                id=o["obj_id"],
                role=o.get("role"),
                side=o.get("side"),
                tag=o.get("tag"),
                moved=(moved.get(o["obj_id"]) or {}).get("moved"),
                **({"flags": flags} if flags else {}),
            )
        )
    for s in ti.get("supplemented_objects") or []:  # e.g. oakink2 receiving objects: names only
        if isinstance(s, dict) and s.get("obj_id") and s.get("for_hand"):
            name, flags = CN.readable(ds, str(s["obj_id"]), clip, data_root)
            objs.append({"name": name, "id": s["obj_id"], "role": f"supplemented:{s['for_hand']}"})
    if mo:  # summary object ids -> readable names
        by = {o["id"]: o["name"] for o in objs}
        for r in mo["objects"]:
            r["obj"] = by.get(r["obj"], r["obj"])
            if "ends_near" in r:
                r["ends_near"]["obj"] = by.get(r["ends_near"]["obj"], r["ends_near"]["obj"])
    seg = ti.get("segment") or {}
    case, hand = case_now(ti)
    readable = {str(o["name"]): r["name"] for o, r in zip(ti.get("objects") or [], objs, strict=False)}
    for h in hand:
        h["object"] = readable.get(h.pop("unit"))
    return dict(
        dataset=ds,
        clip=clip,
        parent_clip=ti.get("parent_clip"),
        segment={"index": seg.get("index"), "of": seg.get("n")} if (seg.get("n") or 1) > 1 else None,
        source=src,
        objects=objs,
        hands=hands_of(ti),
        case=case,
        **({"handover": hand} if hand else {}),
        scene=ti.get("scene") or "table",
        motion=mo,
        tree=str(tree),
    )


# ───── frames (optional) ─────
def dataset_index(patterns: list[str]) -> dict[str, dict]:
    """Stored datasets -> {clip: row 0 x look-D episode (video paths)}; a row without a look field dies. Final-layer
    folders (is_final) are scanned orig first, then their aug twins (only for clips without an orig row 0); an
    episode with a final_wait reason is skipped, and a clip with no usable episode in any hand maps to
    dict(wait=reason)."""
    from dexengine.record.looks.look_d import LOOK as LOOK_D
    from dexengine.record.store import merge_check as MC
    from dexengine.record.store import merge_rows as MR
    from dexengine.record.store.store import VIDEO_KEYS
    from dexengine.record.text import order as ORD

    idx: dict[str, dict] = {}
    wait: dict[str, str] = {}
    roots = sorted(
        {Path(d) for pat in patterns for d in glob.glob(pat) if (Path(d) / "meta" / "info.json").is_file()},
        key=ORD.source_rank,
    )
    twins = [t for t in (aug_twin(r) for r in roots if is_final(r)) if t and (t / "meta" / "info.json").is_file()]
    for root in roots + twins:  # every orig final first; aug twins fill clips without an orig row 0
        info = json.loads((root / "meta" / "info.json").read_text())
        gone = MC.superseded_indices(root)
        hand = ORD.hand_of(root)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            src = MR.row0_source(e, gone)
            if src is None:
                continue
            look = (e.get("look") or {}).get("name") if isinstance(e.get("look"), dict) else None
            if not look:
                raise InputMissingError(f"{MARK} {root}: episode {e.get('episode_index')} has no look field")
            clip = P.clip_name(src)
            if look != LOOK_D or clip in idx:
                continue
            why = final_wait(e) if is_final(root) else None
            if why:
                wait.setdefault(clip, f"{hand}: {why}")
                continue
            i = int(e["episode_index"])

            def mp4(v, root=root, info=info, i=i):
                if v not in VIDEO_KEYS:
                    raise SystemExit(f"{MARK} unknown view {v!r} — store.VIDEO_KEYS {sorted(VIDEO_KEYS)}")
                return root / info["video_path"].format(
                    episode_chunk=i // int(info["chunks_size"]), video_key=VIDEO_KEYS[v], episode_index=i
                )

            idx[clip] = {
                "src": f"{root}#episode_{i}",
                "mp4": mp4,
                "root": root,
                "i": i,
                "scene": CN.placed_ids(e),
                "sig": CN.ep_sig(e),
                "tree": e["clip"],
                "final": is_final(root),
                "aug": root in twins,
            }
    if twins:
        print(
            f"{MARK} aug row 0 used for {sum(1 for v in idx.values() if v['aug'])} clips (no orig row 0 · aug "
            f"finals {len(twins)})",
            flush=True,
        )
    return idx | {c: {"wait": w} for c, w in wait.items() if c not in idx}


def aug_twin(final) -> Path | None:
    """The aug-round final folder with the same dataset and combo as a final folder (<root>/data/<ds>/collect/
    <combo>), from layout.collect_final; None when the aug round has no root of its own. Dies on another form."""
    f = Path(final)
    if f.parent.name != "collect" or f.parents[2].name != "data":
        die(f"not a final dataset folder: {f} — <root>/data/<ds>/collect/<combo>")
    twin = Path(P.data_layout.collect_final(f.parent.parent.name, f.name, "aug", str(f.parents[3])))
    return None if twin == f else twin


def with_twins(roots) -> list[Path]:
    """Final folders + their existing aug twins."""
    tw = [aug_twin(r) for r in roots if is_final(r)]
    return sorted({Path(r) for r in roots} | {t for t in tw if t and (t / "meta" / "episodes.jsonl").is_file()})


def video_seconds(mp4: Path) -> float:
    from dexengine.record.store import lerobot as LR

    out = subprocess.run(
        [LR.ffprobe_bin(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def extract_frames(
    src: dict, views: list[str], ph: list[dict], dt: float, size: tuple[int, int], out: Path, extra: list[dict] = ()
) -> list[dict]:
    """Per view the phases.text_picks times + extra (handover windows) as JPEGs of at most `size` (name = time +
    render fingerprint). Existing files are reused; missing ones are grabbed by concurrent ffmpeg calls (capacity) (temp
    name -> final name)."""
    from concurrent.futures import ThreadPoolExecutor

    from dexengine.record.store import lerobot as LR

    out.mkdir(parents=True, exist_ok=True)
    picks, todo = [], []
    for v in views:
        mp4 = src["mp4"](v)
        if not mp4.is_file():
            raise InputMissingError(f"{MARK} {src['src']} has no {v} video {mp4}")
        dur = video_seconds(mp4)
        for pk in merge_picks(PH.text_picks(ph, dt, v, dur), extra):
            t = min(float(pk["t_s"]), max(0.0, dur - 0.05))
            f = out / f"{v}_t{int(round(t * 100)):06d}_{src['sig']}.jpg"
            picks.append((f, v, t, pk))
            todo += [(mp4, t, f)] if not f.is_file() and (mp4, t, f) not in todo else []
    vf = f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"

    def grab(job):
        mp4, t, f = job
        tmp = f.with_name(f".{f.name}")
        subprocess.run(
            [
                LR.ffmpeg_bin(),
                "-v",
                "error",
                "-y",
                "-ss",
                f"{t:.3f}",
                "-i",
                str(mp4),
                "-frames:v",
                "1",
                "-vf",
                vf,
                "-q:v",
                "3",
                "-f",
                "image2",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(f)

    from dexengine.core import capacity  # noqa: PLC0415

    with capacity.sized(STEP, len(todo)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        list(ex.map(grab, todo))
    return [
        {"path": str(f), "view": v, "t_s": round(t, 2), "w": w, "h": h, "phase": pk["phase"], "at": pk["at"]}
        for f, v, t, pk in picks
        for w, h in [jpeg_size(f)]
    ]


def jpeg_size(f: Path) -> tuple[int, int]:
    """(width, height) from the JPEG header (SOF marker; standard library only)."""
    b = f.read_bytes()
    i = 2
    while i < len(b):
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(b[i + 7 : i + 9], "big"), int.from_bytes(b[i + 5 : i + 7], "big")
        i += 2 + int.from_bytes(b[i + 2 : i + 4], "big")
    raise SystemExit(f"{MARK} cannot read the JPEG size of {f}")


# ───── input list ─────
def clips_from_owner(paths: list[str], min_rate: float) -> list[tuple[str, str]]:
    """(dataset, clip) of the canon rows with rate_pct >= min_rate in RL clip-owner tables."""
    seen = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                try:
                    ok = r["state"] == "canon" and float(r["rate_pct"]) >= min_rate
                except (KeyError, ValueError):
                    ok = False
                if ok:
                    seen[(r["dataset"], r["clip"])] = 1
    return sorted(seen)


def find_tree(root: Path, ds: str, clip: str) -> Path | None:
    hits = sorted(root.glob(f"{ds}/trees/*/*/{P.tree_name(clip)}/task_info.json"))
    return hits[0].parent if hits else None


def _parser():
    """Command line of the records builder."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", action="append", default=[], help="RL clip-owner table tsv (repeat)")
    ap.add_argument("--min-rate", type=float, default=None, help="with --owner: minimum completion rate (%%)")
    ap.add_argument("--pairs", help="dataset<TAB>clip list file (instead of --owner)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N only (trial)")
    ap.add_argument(
        "--with-frames",
        dest="frames",
        action="store_true",
        help="extract frames (phases.TEXT_FRAMES per view; off = text only)",
    )
    ap.add_argument("--views", default="exo,egoL", help="render video names (comma separated)")
    ap.add_argument("--frame-size", default="960x540", help="maximum frame size WxH")
    ap.add_argument(
        "--dataset-glob",
        action="append",
        default=[],
        help="glob of stored dataset folders (repeat; with --with-frames)",
    )
    ap.add_argument("--frames-dir", default="", help="frame folder (default: frames/ next to --out)")
    ap.add_argument("--require-frames", action="store_true", help="die when any clip has no render")
    ap.add_argument(
        "--frames-only",
        action="store_true",
        help="clips without a usable render get no record, only a frames_missing.tsv row (wait for it)",
    )
    return ap


def _build_records(
    pairs, root, missing, a, ridx, no_frames, size, fdir, leps, fin, recs, scene_bad, unmeasured, waiting
):
    """One record per (dataset, clip): tree, frame source, frames and phases (or the reference phases); fills recs,
    missing and no_frames."""
    for i, (ds, clip) in enumerate(pairs):
        t = find_tree(root, ds, clip)
        if t is None:
            missing.append((ds, clip))
            continue
        src = ridx.get(clip) if a.frames else None
        why = (
            ("no render" if src is None else src.get("wait") or (None if src["final"] else tree_fresh(src)))
            if a.frames
            else None
        )
        if why is None and a.frames and any(not src["mp4"](v).is_file() for v in a.views.split(",")):
            why = "copy body missing (file missing on this machine)"  # the episode's video file is missing
        if a.frames_only and why is not None:  # wait for the render: no record
            no_frames.append((ds, clip, why))
            waiting += 1
            continue
        rec = record(ds, clip, t, root)
        if a.frames and why is None:  # frames and phase table share a time axis: measured on that episode
            ph, dt = PH.from_episode(src["root"], src["i"])
            rec["frames"] = extract_frames(
                src, a.views.split(","), ph, dt, size, fdir / ds / clip, handover_picks(rec.get("handover") or [])
            )
            other = [(r, e) for r, e in leps.get(clip, []) if (r, int(e["episode_index"])) != (src["root"], src["i"])]
            more = [PH.from_meta(r, e) for r, e in other if "subtasks" in e]  # phases stored with the episode
            unmeasured += len(other) - len(more)
            rec.update(
                frames_src=src["src"], frames_sig=src["sig"], phases=PH.union_table(ph, dt, more), phases_src="episode"
            )
            rec = MOT.inputs(rec, fin)
            rec["scene_objects"], nb = CN.scene_names(ds, src["scene"], clip, root, {o["name"] for o in rec["objects"]})
            scene_bad += nb
        else:  # text only: phase table from the source reference motion (tree)
            ph, dt = PH.from_reference(t)
            rec.update(phases=PH.table(ph, dt), phases_src="reference")
            if a.frames:
                rec.update(frames=[], frames_src=None, frames_missing=why)
                no_frames.append((ds, clip, why))
        recs.append(rec)
        if (i + 1) % 500 == 0:
            print(f"{MARK} {i + 1}/{len(pairs)}", flush=True)
    return waiting, unmeasured, scene_bad


def main(argv=None) -> int:
    """CLI: build records.jsonl (+ missing.tsv, frames_missing.tsv) for the given clips."""
    ap = _parser()
    a = ap.parse_args(argv)
    if bool(a.owner) == bool(a.pairs):
        die("exactly one of --owner and --pairs")
    if a.owner and a.min_rate is None:
        die("--owner needs an explicit --min-rate")
    pairs = (
        clips_from_owner(a.owner, a.min_rate)
        if a.owner
        else [tuple(x.split("\t")[:2]) for x in Path(a.pairs).read_text().splitlines() if x.strip()]
    )
    if a.limit:
        pairs = pairs[: a.limit]
    root, out = Path(a.data_root), Path(a.out)
    if a.frames and not a.dataset_glob:
        die("--with-frames needs an explicit --dataset-glob")
    ridx = dataset_index(a.dataset_glob) if a.frames else {}
    from dexengine.record.text import order as ORD  # every loaded episode (the phase table covers all their phases)

    leps, unmeasured = (ORD.loaded_episodes(a.dataset_glob) if a.frames else {}), 0
    fin = with_twins({r for v in leps.values() for r, _ in v})  # objects missing from the render (R21), aug included
    size = tuple(int(x) for x in a.frame_size.lower().split("x"))
    fdir = Path(a.frames_dir) if a.frames_dir else out.parent / "frames"
    recs, missing, waiting, scene_bad, no_frames = [], [], 0, 0, []
    if a.frames_only and not a.frames:
        die("--frames-only needs --with-frames")
    waiting, unmeasured, scene_bad = _build_records(
        pairs, root, missing, a, ridx, no_frames, size, fdir, leps, fin, recs, scene_bad, unmeasured, waiting
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    tmp.replace(out)
    (out.parent / "missing.tsv").write_text("".join(f"{d}\t{c}\n" for d, c in missing))
    kinds = {}
    for r in recs:
        kinds.setdefault(r["dataset"], {}).setdefault(r["source"]["kind"], 0)
        kinds[r["dataset"]][r["source"]["kind"]] += 1
    print(
        f"{MARK} targets {len(pairs)} = records {len(recs)} + no tree {len(missing)} + waiting for render "
        f"{waiting} · source kinds {kinds}",
        flush=True,
    )
    if a.frames:
        (out.parent / "frames_missing.tsv").write_text("".join(f"{d}\t{c}\t{w}\n" for d, c, w in no_frames))
        have = len(recs) - (len(no_frames) - waiting)  # waiting clips have no record
        print(
            f"{MARK} frames {PH.TEXT_FRAMES} · {a.frame_size} — with render {have} · without {len(no_frames)} "
            f"(frames_missing.tsv · waiting {waiting}) · other episodes without a subtasks field in the phase union "
            f"{unmeasured} · scene object names unreadable {scene_bad}",
            flush=True,
        )
        if no_frames and a.require_frames:
            die(f"{len(no_frames)} clips without a render — --require-frames")
    if len(recs) + len(missing) + waiting != len(pairs):
        die("count mismatch")
    print(f"{MARK} OK n={len(recs)} → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
