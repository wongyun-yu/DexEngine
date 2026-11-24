#!/usr/bin/env python3
"""E5 policy dataset — episodes named by a manifest (manifest.py) → one LeRobot v2.1 dataset in the E5 contract.

Contract:
  state  = <--state-key>  (observation.joint_pos.robot) · action = <--action-key> (action.joint_target.robot)
           both 56-D, selected BY NAME into the four groups of GROUPS (the RB-Y1 + Wuji 2 joint groups, torso left out);
           any source joint outside GROUPS or any GROUPS joint absent from the source dies.
  rate   = the video rate: one row per video frame — for frame j the row with video_frame_index == j whose timestamp
           is nearest j / video_fps (the source rows run at info.fps = 50 Hz, the video at collect.video_fps = 30).
  video  = head stereo only (--cameras, default ego_left,ego_right → observation.images.camera_ego_{left,right}),
           ffmpeg scale=<W>:<H>:flags=area, libx264 crf 18 -g 2 yuv420p, the source frame rate kept.
  clip   = --action-clip-limits <limits.json> (default off = byte-identical): after the action is picked, each joint is
           clipped to [lo, hi] of the URDF limits, by name (the json is the output of
           dexengine.rl.tasks.scene.asset_checks.urdf_limits on dexengine.robots.selection.combo_urdf_path). Recorded
           per episode (action_clip_frames · action_clip_max_rad · runaway_frames = frames with a
           joint past its limit AND |target − state| > 1 rad on it, which clipping does not repair)
           and in meta/e5_conversion.json action_clip.
           Marker: [E5_CONVERT] action_clip ON.
  text   = the manifest's task_text (tasks.jsonl one row per distinct text) · axis_index = the manifest's axis
           (AXES order) on every frame (DP's task one-hot) · lang_source recorded per episode.
The manifest names an episode by (dataset, root, clip, round, look); its episode_index is looked up here in that root's
meta/episodes.jsonl — a key absent there, listed in superseded.jsonl, or matching twice dies (the roots get
renumbered; the manifest never stores numbers).

Phases (one call does all; a rerun resumes): resolve → convert (process pool; an episode is done when its
_work/ep_<i>.json exists — written last, by rename) → finalize (meta/*: info · episodes · tasks · episodes_stats ·
stats with q01/q99 · modality · e5_conversion; then the in = out check on the unique key set and on file counts).
Sources are read only. Needs numpy · pandas · pyarrow and an ffmpeg binary (--ffmpeg).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dexengine.core.hashing import sha256_file
from experiments.policies.data import convert_meta
from experiments.policies.data.convert_episode import AXES, GROUPS, NAMES, TAG, die, work_one
from experiments.policies.data.convert_meta import (
    column_stats,
    count_files,
    load_recs,
    write_episodes,
    write_info,
    write_modality,
    write_stats,
)
from experiments.textio import read_lines

assert len(NAMES) == 56 and len(set(NAMES)) == 56


def look_name(e: dict) -> str:
    lk = e.get("look")
    return str(lk.get("name") if isinstance(lk, dict) else (lk or ""))


def perm_of(info: dict, state_key: str, action_key: str) -> list[int]:
    if info.get("codebase_version") != "v2.1":
        die(f"source codebase_version {info.get('codebase_version')!r} ≠ v2.1")
    F = info["features"]
    sn, an = F[state_key]["names"], F[action_key]["names"]
    if sn != an:
        die(f"{state_key} and {action_key} joint orders differ")
    if len(set(an)) != len(an):
        die("source joint names not unique")
    if set(an) != set(NAMES):
        die(f"source joints ≠ contract: missing {sorted(set(NAMES) - set(an))} extra {sorted(set(an) - set(NAMES))}")
    idx = {n: i for i, n in enumerate(an)}
    return [idx[n] for n in NAMES]


# ---------------------------------------------------------------- resolve
def resolve(rows: list[dict], roots: dict[str, str], combo: str, state_key: str, action_key: str) -> tuple[list, dict]:
    """manifest rows → [(row, src_dir, src_episode_record, perm)] in manifest order · per-source info."""
    cache, infos = {}, {}
    out = []
    for r in rows:
        src = os.path.join(roots[r["root"]], "data", r["dataset"], "collect", combo)
        if src not in cache:
            meta = os.path.join(src, "meta")
            info = json.loads(Path(os.path.join(meta, "info.json")).read_text())
            sp = os.path.join(meta, "superseded.jsonl")
            gone = (
                {int(json.loads(x)["episode_index"]) for x in read_lines(sp) if x.strip()}
                if os.path.isfile(sp)
                else set()
            )
            by_key = {}
            with open(os.path.join(meta, "episodes.jsonl")) as fh:
                for ln in fh:
                    if not ln.strip():
                        continue
                    e = json.loads(ln)
                    k = (e["clip"], str(e.get("round")), look_name(e))
                    if k in by_key:
                        die(f"{src}: key {k} matches two episodes ({by_key[k]['episode_index']}, {e['episode_index']})")
                    by_key[k] = e
            cache[src] = (by_key, gone)
            infos[src] = {"info": info, "perm": perm_of(info, state_key, action_key)}
        by_key, gone = cache[src]
        k = (r["clip"], r["round"], r["look"])
        e = by_key.get(k)
        if e is None:
            die(f"{src}: manifest key {k} not in episodes.jsonl")
        if int(e["episode_index"]) in gone:
            die(f"{src}: manifest key {k} is superseded (episode {e['episode_index']})")
        out.append((r, src, e))
    return out, infos


# ---------------------------------------------------------------- finalize (meta/* in convert_meta.py)


def guard_out(out: str, out_root: str, roots: dict, reuse: str) -> str:
    """Every writer of this tool (parquet · mp4 · _work markers · meta · .partial temps · --reuse copies) writes
    under
    --out, so --out alone decides where bytes land. Rule (a mkdir -p must never follow a symlink into a source tree):
    --out has no symlink on its path (realpath == abspath) and lies under --out-root when one is given (same rule for
    it),
    must not overlap any source root or the --reuse folder in either direction, and the folder must hold no symlink.
    Any breach dies before a byte is written; finalize calls it again as the post-condition."""
    ab, re_ = os.path.abspath(out), os.path.realpath(out)
    if ab != re_:
        die(f"guard: --out {out} passes through a symlink (realpath {re_})")
    rre = os.path.realpath(out_root) if out_root else ""
    if out_root and os.path.abspath(out_root) != rre:
        die(f"guard: --out-root {out_root} passes through a symlink (realpath {rre})")
    if out_root and not re_.startswith(rre.rstrip("/") + "/"):
        die(f"guard: --out {re_} is not under --out-root {rre}")

    def inside(a_, b_):
        return a_ == b_ or a_.startswith(b_.rstrip("/") + "/")

    for tag, r in [*sorted(roots.items()), *([("reuse", reuse)] if reuse else [])]:
        rr = os.path.realpath(r)
        if inside(re_, rr) or inside(rr, re_):
            die(f"guard: --out {re_} overlaps {tag} root {rr}")
    links = []
    if os.path.isdir(out):
        for d, dirs, files in os.walk(out, followlinks=False):
            links += [os.path.join(d, n) for n in dirs + files if os.path.islink(os.path.join(d, n))]
    if links:
        die(f"guard: {len(links)} symlinks inside --out (first {links[0]} -> {os.readlink(links[0])})")
    print(
        f"{TAG} guard out={re_} under {rre or '-'} · no symlink on path or inside · disjoint from {len(roots)} roots"
        f"{' + reuse' if reuse else ''}",
        flush=True,
    )
    return re_


def _conversion_record(a, man_sha, infos, src_rows, recs, stats_filled, N, total):
    """The e5_conversion.json record: tool/manifest hashes, keys, rate/action rules, clip and video settings, counts."""
    langs = sorted({r["lang_source"] for r in src_rows})
    conv = dict(
        schema=1,
        tool="experiments/policies/data/convert.py",
        tool_sha256=sha256_file(__file__),
        tool_sibling_sha256={"experiments/policies/data/convert_meta.py": sha256_file(convert_meta.__file__)},
        stats_filled_features=stats_filled,
        git_commit=a.git_commit,
        built=datetime.datetime.now().isoformat(timespec="seconds"),
        manifest=os.path.abspath(a.manifest),
        manifest_sha256=man_sha,
        manifest_meta=(
            json.loads(Path(a.manifest[:-4] + ".meta.json").read_text())
            if a.manifest.endswith(".tsv") and os.path.isfile(a.manifest[:-4] + ".meta.json")
            else None
        ),
        limit=a.limit,
        roots=dict(r.split("=", 1) for r in a.roots.split(",")),
        combo=a.combo,
        state_key=a.state_key,
        action_key=a.action_key,
        groups={g: list(v) for g, v in GROUPS.items()},
        perms={src: v["perm"] for src, v in infos.items()},
        rate_rule="one row per video frame j: video_frame_index == j, timestamp nearest j/video_fps",
        action_shift=a.action_shift,
        action_rule=f"action of frame j = {a.action_key} at state row + {a.action_shift} (clamped to the last row)",
        action_clamped_frames=sum(r["action_clamped"] for r in recs),
        **(
            {
                "action_clip": dict(
                    a.clip_meta,
                    frames_clipped=sum(r.get("action_clip_frames", 0) for r in recs),
                    episodes_clipped=sum(1 for r in recs if r.get("action_clip_frames")),
                    max_excess_rad=max((r.get("action_clip_max_rad", 0.0) for r in recs), default=0.0),
                    runaway_frames=sum(r.get("runaway_frames", 0) for r in recs),
                    runaway_episodes=sum(1 for r in recs if r.get("runaway_frames")),
                )
            }
            if a.clip_meta
            else {}
        ),
        force_keys=[],
        force_rule=None,  # force rows are not emitted (fields kept for the record format)
        video={
            "cameras": list(a.cameras),
            "to": [a.width, a.height],
            "filter": "area",
            "codec": "h264",
            "crf": 18,
            "gop": 2,
        },
        axes=list(AXES),
        pilot_lang=("template" if langs == ["template"] else ",".join(langs)),
        episodes=N,
        frames=total,
        reused=sum(1 for r in recs if r.get("reused_from")),
        reuse_from=(os.path.abspath(a.reuse) if a.reuse else None),
    )
    if a.allow_axis_none:  # off: the record has no such key
        conv["axis_none"] = {
            "n": sum(1 for r in src_rows if r["axis"] == ""),
            "axis_index": -1,
            "environment_state": "all 0",
            "why": "GR00T mid-training corpus — GR00T modality does not read the axis",
        }
    return conv


def finalize(out: str, jobs: list[dict], tasks: list[str], a, man_sha: str, infos: dict, src_rows: list[dict]) -> None:
    N = len(jobs)
    recs, keys_in = load_recs(out, jobs)
    total = sum(r["length"] for r in recs)
    stats = column_stats(out, jobs, recs, a, total)
    meta = os.path.join(out, "meta")
    os.makedirs(meta, exist_ok=True)
    src_info = next(iter(infos.values()))["info"]
    features = write_info(meta, src_info, a, N, total, tasks)
    write_episodes(meta, jobs, recs, tasks)
    stats_filled = write_stats(out, jobs, a, meta, features, stats)
    write_modality(meta, a)
    conv = _conversion_record(a, man_sha, infos, src_rows, recs, stats_filled, N, total)
    with open(os.path.join(meta, "e5_conversion.json"), "w") as fh:
        json.dump(conv, fh, indent=1)
    # file-count check (in = out)
    n_pq, n_mp4 = count_files(out, a, N)
    print(
        f"{TAG} DONE episodes={N} frames={total} tasks={len(tasks)} parquet={n_pq} mp4={n_mp4} "
        f"keys_in=keys_out={len(keys_in)} pilot_lang={conv['pilot_lang']} out={out}"
    )


def _args():
    """Command line (help text = the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--roots", required=True, help="tag=path,… (the manifest's root column names a tag)")
    ap.add_argument("--combo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--git-commit", required=True, help="commit of this tool's tree (recorded)")
    ap.add_argument("--state-key", default="observation.joint_pos.robot")
    ap.add_argument("--action-key", default="action.joint_target.robot")
    ap.add_argument("--cameras", default="ego_left,ego_right")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument(
        "--action-shift",
        type=int,
        required=True,
        help="action of a frame = target this many rows after its state row (the training sets: 1)",
    )
    ap.add_argument(
        "--action-clip-limits",
        default="",
        help="D1b: URDF joint-limit json (owner output · limits {name: [lo, hi]}) — clip the action to it "
        "(default off)",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument(
        "--allow-axis-none",
        action="store_true",
        help="manifest rows with an empty axis are converted with axis_index -1 and an all-zero "
        "environment_state (GR00T mid-training corpus) — off = byte-identical",
    )
    ap.add_argument("--limit", type=int, default=0, help="first K manifest rows after --sample-seed shuffle (test)")
    ap.add_argument("--sample-seed", type=int, default=None, help="shuffle rows (seeded) before --limit")
    ap.add_argument("--reuse", default="", help="an earlier --out with the same contract: copy its finished episodes")
    ap.add_argument(
        "--out-root", default="", help="if given, --out must resolve under this folder (no symlinks) — guard_out"
    )
    a = ap.parse_args()
    return a


def _action_clip(a):
    """--action-clip-limits: per-joint URDF limits in output column order (a.clip) and their record (a.clip_meta)."""
    if a.action_clip_limits:
        with open(a.action_clip_limits) as fh:
            lj = json.load(fh)
        lim = lj.get("limits") or {}
        miss = [n for n in NAMES if n not in lim]
        if miss:
            die(f"--action-clip-limits {a.action_clip_limits}: no limit for {miss[:5]}… ({len(miss)})")
        bad = [n for n in NAMES if not float(lim[n][0]) < float(lim[n][1])]
        if bad:
            die(f"--action-clip-limits: lo ≥ hi for {bad}")
        a.clip = ([float(lim[n][0]) for n in NAMES], [float(lim[n][1]) for n in NAMES])
        csha = sha256_file(a.action_clip_limits)
        a.clip_meta = {
            "limits_file": os.path.abspath(a.action_clip_limits),
            "limits_sha256": csha,
            "provenance": {
                k: lj.get(k)
                for k in ("owner", "hand_usd", "variant", "urdf_variant", "urdf", "urdf_sha256", "tree_commit")
            },
            "rule": "action = clip(action, lo, hi) per joint by name — after the action-shift pick; "
            "stats are of the clipped action",
        }
        print(
            f"{TAG} action_clip ON limits={a.action_clip_limits} sha={csha[:12]} urdf_sha={lj.get('urdf_sha256')} "
            f"owner={lj.get('owner')}",
            flush=True,
        )


def _read_rows(a):
    """Read the manifest rows (optional seeded shuffle and --limit) and check their axes."""
    with open(a.manifest) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    man_sha = sha256_file(a.manifest)
    if a.sample_seed is not None:
        rng = np.random.default_rng(a.sample_seed)
        rows = [rows[k] for k in rng.permutation(len(rows))]
    if a.limit:
        rows = rows[: a.limit]
        rows.sort(key=lambda r: (r["dataset"], r["root"], r["clip"], r["round"], r["look"]))
    bad_axis = sorted({r["axis"] for r in rows} - set(AXES) - ({""} if a.allow_axis_none else set()))
    if bad_axis:
        die(f"axes outside {AXES}: {bad_axis}")
    n_axis_none = sum(1 for r in rows if r["axis"] == "")
    if a.allow_axis_none:  # mark: on, and how many rows — a corpus without this line is void
        print(
            f"[AXIS_NONE_OK] n={n_axis_none} of {len(rows)} rows · axis_index -1 · environment_state all 0", flush=True
        )
    return rows, man_sha


def _check_video_keys(a, infos):
    """Every read camera key exists in every source; print read, source and unread video keys."""
    # which video keys are read (kinematic roots add observation.images.exo — only the head eyes are read)
    read_keys = [f"observation.images.{c}" for c in a.cameras]
    src_keys = sorted(
        {k for v in infos.values() for k, f in v["info"]["features"].items() if f.get("dtype") == "video"}
    )
    missing = [k for k in read_keys if any(k not in v["info"]["features"] for v in infos.values())]
    if missing:
        die(f"video keys {missing} absent from a source")
    print(
        f"{TAG} video_keys read={','.join(read_keys)} (→ observation.images.camera_<cam>) · "
        f"source video keys={','.join(src_keys)} · "
        f"not read={','.join(k for k in src_keys if k not in read_keys) or 'none'}",
        flush=True,
    )


def _write_marker(a, man_sha):
    """Write the contract signature of this --out; an --out started with other inputs dies."""
    os.makedirs(os.path.join(a.out, "_work"), exist_ok=True)
    marker = os.path.join(a.out, "_work", "manifest.sha256")
    sig = (
        f"{man_sha} limit={a.limit} seed={a.sample_seed} shift={a.action_shift} force= "
        f"cams={','.join(a.cameras)} {a.width}x{a.height}"
        + (f" clip={a.clip_meta['limits_sha256'][:12]}" if a.clip_meta else "")
        + (" axis_none=1" if a.allow_axis_none else "")
    )
    if os.path.isfile(marker) and Path(marker).read_text().strip() != sig:
        die(f"{a.out} was started with other inputs ({Path(marker).read_text().strip()}) — use a new --out")
    Path(marker).write_text(sig + "\n")
    return sig


def _build_jobs(a, resolved, infos):
    """One conversion job per resolved episode (task index, axis index, global frame offset); returns tasks, jobs."""
    tasks, task_idx = [], {}
    jobs, g = [], 0
    for i, (r, src, e) in enumerate(resolved):
        if r["task_text"] not in task_idx:
            task_idx[r["task_text"]] = len(tasks)
            tasks.append(r["task_text"])
        jobs.append(
            dict(
                src=src,
                e=e,
                episode_index=i,
                out=a.out,
                width=a.width,
                height=a.height,
                ffmpeg=a.ffmpeg,
                cameras=a.cameras,
                info=infos[src]["info"],
                perm=infos[src]["perm"],
                state_key=a.state_key,
                action_key=a.action_key,
                chunks_size=a.chunks_size,
                action_shift=a.action_shift,
                **({"clip": a.clip} if a.clip else {}),
                global_index=g,
                task_index=task_idx[r["task_text"]],
                axis_index=(AXES.index(r["axis"]) if r["axis"] else -1),
                key=[r["dataset"], r["root"], r["clip"], r["round"], r["look"]],
                row=r,
            )
        )
        g += int(e["video_frames"])
    return tasks, jobs


def _reuse_finished(a, jobs, sig):
    """The jobs not finished yet; with --reuse, jobs finished in an earlier output of the same contract are copied."""
    todo = [j for j in jobs if not os.path.isfile(os.path.join(a.out, "_work", f"ep_{j['episode_index']:06d}.json"))]
    if a.reuse:
        # an earlier output with the SAME contract signature (all but the manifest sha) lends its finished episodes
        osig = Path(os.path.join(a.reuse, "_work", "manifest.sha256")).read_text().split(" ", 1)[1].strip()
        if osig != sig.split(" ", 1)[1]:
            die(f"--reuse {a.reuse}: contract differs ({osig} ≠ {sig.split(' ', 1)[1]})")
        old = {}
        for fn in os.listdir(os.path.join(a.reuse, "_work")):
            if fn.startswith("ep_") and fn.endswith(".json"):
                r = json.loads(Path(os.path.join(a.reuse, "_work", fn)).read_text())
                old[tuple(r["key"])] = int(r["episode_index"])
        nre = 0
        for j in todo:
            if tuple(j["key"]) in old:
                j["reuse"] = [os.path.abspath(a.reuse), old[tuple(j["key"])]]
                nre += 1
        print(f"{TAG} reuse {nre}/{len(todo)} from {a.reuse} ({len(old)} finished there)", flush=True)
    return todo


def _run_jobs(a, todo):
    """Convert the remaining jobs in a process pool; any failed episode dies after all are listed."""
    fails, done, t1 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(work_one, {k: v for k, v in j.items() if k != "row"}): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                f.result()
                done += 1
            except Exception as exc:  # noqa: BLE001 — every failure is listed, then the run dies
                fails.append((j["episode_index"], j["key"], repr(exc)[:300]))
                print(f"{TAG} EPFAIL {j['episode_index']} {j['key']} {repr(exc)[:300]}", flush=True)
            if done and done % 200 == 0:
                el = time.time() - t1
                print(
                    f"{TAG} progress {done}/{len(todo)} {el / done:.2f}s/ep wall · eta "
                    f"{(len(todo) - done) * el / done / 60:.1f} min",
                    flush=True,
                )
    if fails:
        die(f"{len(fails)} episodes failed (first: {fails[0]})")
    print(f"{TAG} converted {done} in {time.time() - t1:.0f}s", flush=True)


def main() -> int:
    a = _args()
    a.cameras = tuple(c for c in a.cameras.split(",") if c)
    roots = dict(r.split("=", 1) for r in a.roots.split(","))
    a.clip, a.clip_meta = None, None
    _action_clip(a)
    guard_out(a.out, a.out_root, roots, a.reuse)
    rows, man_sha = _read_rows(a)
    t0 = time.time()
    resolved, infos = resolve(rows, roots, a.combo, a.state_key, a.action_key)
    print(
        f"{TAG} resolved {len(resolved)} episodes from {len(infos)} sources in {time.time() - t0:.1f}s "
        f"manifest_sha={man_sha[:12]}",
        flush=True,
    )
    _check_video_keys(a, infos)
    sig = _write_marker(a, man_sha)
    tasks, jobs = _build_jobs(a, resolved, infos)
    todo = _reuse_finished(a, jobs, sig)
    print(f"{TAG} convert {len(todo)}/{len(jobs)} (rest done) jobs={a.jobs}", flush=True)
    _run_jobs(a, todo)
    guard_out(a.out, a.out_root, roots, a.reuse)
    finalize(a.out, jobs, tasks, a, man_sha, infos, rows)
    guard_out(a.out, a.out_root, roots, a.reuse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
