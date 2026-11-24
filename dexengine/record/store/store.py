"""Recording store (CPU, retarget venv, pyarrow): stage-1 record (.npz) + stage-2 render -> LeRobot dataset.

Layout (LeRobot v2.1; differences are described by the `collect` field of meta/info.json):
  meta/info.json (collect.cameras = camera intrinsics: K, distortion, resolution, fps, baseline) ·
  meta/episodes.jsonl (per episode: selection table, camera_pose = extrinsics of both ego eyes, exo and head joints) ·
  meta/tasks.jsonl · meta/collect.json (provenance: cell, clip completion rates, switches, controller). Camera values
  come from render_meta.json `camera`, copied by store.meta.camera_intrinsics / camera_pose (never recomputed).
  data/chunk-000/episode_<i>.parquet    one row per policy step (50 Hz): joint state, policy output, absolute joint
                                        targets, objects, reference, tracking error, video_frame_index (nearest frame)
  videos/chunk-000/observation.images.{ego_left,ego_right,exo}/episode_<i>.mp4   H.264 30 fps
  depth/chunk-000/observation.depth.ego_left/episode_<i>.npy.zst  uint16 mm, 0 = invalid, 30 fps (render.writer)
  forces/chunk-000/episode_<i>.npz      t, force_total, force_object, quality.*, *.force_n_clean, partner split
                                        (keys and rules: store.force)
  head_view/chunk-000/episode_<i>.npz   hand-in-head-view marks (render.visibility)
Differences from LeRobot: low-dimensional data at 50 Hz vs video at 30 fps (videos align to the nearest step via
video_frame_index); depth and forces are extra folders outside the LeRobot standard.
Tracking error = rl.eval.traj_select.errors (the same function as the selection score). Success = the judge's
definition (stage-1 recorder). Head joint columns = render measurement, episode meta head_joints and f0_hold
(store.meta).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError
from dexengine.record.looks import look_d as LOOK
from dexengine.record.render import cameras as CAMS
from dexengine.record.render import visibility as HV
from dexengine.record.render import writer as CD
from dexengine.record.rollout import recorder as REC
from dexengine.record.store import force as CF
from dexengine.record.store import form as FM
from dexengine.record.store import lerobot as CL
from dexengine.record.store import meta as CM
from dexengine.record.store import slots as CK
from dexengine.record.store.lerobot import write_parquet
from dexengine.record.store.tasks import episode_file, episode_subtasks, episode_task, write_task_meta

CODEBASE = "v2.1"
VIDEO_KEYS = {
    k: v
    for k, v in {
        "egoL": "observation.images.ego_left",
        "egoR": "observation.images.ego_right",
        "exo": "observation.images.exo",
    }.items()
    if k in CAMS.videos()
}  # with the exo camera off: both head eyes only
DEPTH_KEY = "observation.depth.ego_left"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"  # info.json path patterns
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000
PREVIEW_MP4_MAX_MB = 8.0
PREVIEW_IMG_W = 960  # preview still width (JPEG); the dataset mp4 keeps full size
LOOK_KEYS = ("tonemap", "whitepoint", "rtx", "panels", "materials", "props", "light_variation", "supports")
PREVIEW_VIDEO_DIR = "videos"  # preview video folder (copies / downscales of dataset videos), removed by record.clean


def eval_modules():
    """(rl.eval.traj_file, rl.eval.traj_select): trajectory file reader and selection/error functions."""
    from dexengine.rl.eval import traj_file, traj_select

    return traj_file, traj_select


ASSIST_FLAGS = frozenset({"assist_hand", *(f"touch_only_{s[0]}" for s in ("right", "left"))})
LOOPHOLES = ("odd_piece", "assist_idle")


def is_piece(ti: dict) -> bool:
    """A piece = a tree that segment.split built from a window (`segment_source` field). A rest-based cut
    (`_seg<k>`) is not a piece; that clip counts as an original clip."""
    return "segment_source" in ti


def loophole(ti: dict) -> str | None:
    """Case label for later filtering (metadata instead of a rejudge). One rule: with a resting hand (idle_sides) and
    an assist / touch-only hand in the case flags (ASSIST_FLAGS), a piece is 'odd_piece' (a piece cut so that one
    hand rests) and an uncut clip is 'assist_idle' (an original clip whose assisting hand rests); otherwise None.
    Missing case fields (case.flags, idle_sides) are an error: an unjudgeable clip is never written as clean."""
    c = ti.get("case")
    if not isinstance(c, dict) or not isinstance(c.get("flags"), list) or not isinstance(ti.get("idle_sides"), list):
        raise InputMissingError(
            f"[COLLECT_STORE] {ti.get('clip_id')}: task_info has no case.flags / idle_sides — "
            "cannot judge the loophole case"
        )
    if not (ti["idle_sides"] and ASSIST_FLAGS & set(c["flags"])):
        return None
    return LOOPHOLES[0] if is_piece(ti) else LOOPHOLES[1]


def clip_case(ti: dict) -> dict:
    """Episode case meta: the clip's task_info values copied as they are, plus the loophole label.
    parent_clip / source_window = the original take and the window in it (capture frames [start, end));
    segment_range = the window inside the parent clip for a piece; alignment = episode step <-> original frame
    (f0_prepad padding frames, then the retiming factor)."""
    hole = loophole(ti)
    c, sg = ti["case"], ti.get("segment") or {}
    return {
        "case": c.get("case"),
        "flags": list(c["flags"]),
        "idle_sides": list(ti["idle_sides"]),
        "n_extra": c.get("n_extra"),
        "piece": is_piece(ti),
        "parent_clip": ti.get("parent_clip"),
        "segment_range": ti.get("segment_range"),
        "source_window": [sg.get("start"), sg.get("end")],
        "cut_source": sg.get("cut_source"),
        "alignment": {"f0_prepad": ti.get("f0_prepad"), "retime_safety": ti.get("retime_safety")},
        "loophole": hole,
    }


def episode(tf, TS, clip: str) -> dict:
    """The selected trajectory of a clip -> columns {name: [L, ...] array} + meta."""
    st = tf.selected(clip)
    if st is None:
        raise InputMissingError(f"[COLLECT_STORE] {clip}: no selected trajectory")
    meta, ents = tf.meta, {e["name"]: e for e in tf.entities}
    col = meta["env"]["collect"]
    REC.require_partner(col, clip)
    L, step_dt = st.shape[0], float(meta["step_dt"])
    cols: dict = {"timestamp": np.arange(L) * step_dt, "frame_index": np.arange(L)}
    names: dict = {}  # column -> dim names (trajectory entity joint_names, joint-target channel names, or null)
    for e in tf.entities:
        blk = st[:, e["lo"] : e["hi"]]
        if e["kind"] == "artic" and not e["name"].startswith("object"):
            cols[f"observation.joint_pos.{e['name']}"] = blk[:, 7:]
            names[f"observation.joint_pos.{e['name']}"] = e.get("joint_names")
        else:
            cols[f"observation.object_pose.{e['name']}"] = blk[:, :7]
            if e["kind"] == "artic":
                jk = FM.OBJ_JOINT + e["name"]
                cols[jk], names[jk] = blk[:, 7:], e.get("joint_names")
    errs = {}
    for i, s in enumerate(meta["sides"]):
        ref_lo = tf.d_state + 7 * i
        cols[f"reference.object_pose.{s['side']}"] = st[:, ref_lo : ref_lo + 7]
        pe, re = TS.errors(st, int(ents[s["entity"]]["lo"]), ref_lo)
        cols[f"tracking.pos_err_m.{s['side']}"], cols[f"tracking.rot_err_rad.{s['side']}"] = pe, re
        errs[s["side"]] = {
            "pos_mean_m": float(pe.mean()),
            "pos_max_m": float(pe.max()),
            "rot_mean_deg": float(np.rad2deg(re.mean())),
            "rot_max_deg": float(np.rad2deg(re.max())),
            "pos_tol_m": s["pos_tol"],
            "rot_tol_deg": float(np.rad2deg(s["rot_tol"])),
            "end_pos_m": float(pe[-1]),
            "end_rot_deg": float(np.rad2deg(re[-1])),
        }
    forces = {}
    for c in col["channels"]:
        blk = st[:, c["lo"] : c["hi"]]
        if c["kind"] == "action":
            cols["action"] = blk
            names["action"] = c.get("dim_names")  # rollout.channels.action_dim_names
        elif c["kind"] == "joint_target":
            cols[f"action.joint_target.{c['name'].split('/', 1)[1]}"] = blk
            names[f"action.joint_target.{c['name'].split('/', 1)[1]}"] = c.get("names")
        elif (k := REC.contact_kind(c)) is not None:
            forces[f"{k}.{c['name'].split('/', 1)[1]}"] = (REC.channel_block(c, blk, L), c["names"])
        else:
            raise SystemExit(f"[COLLECT_STORE] unknown record channel {c['name']} kind {c['kind']}")
    cols, names, forces, chans = CK.own(cols, names, forces, meta, clip, col["channels"])
    return {
        "clip": clip,
        "L": L,
        "step_dt": step_dt,
        "cols": cols,
        "names": names,
        "forces": forces,
        "errors": errs,
        "channels": chans,
        "physics_dt": float(col["physics_dt"]),
        "spawn_hold_steps": col.get("spawn_hold_steps"),
    }


def rendered_episode(tf, TS, it: dict) -> tuple[dict, dict, dict]:
    """episode + render-measurement fix (store.meta.render_fix: head joints, world shift); the store and the
    preview use the same columns."""
    ep = episode(tf, TS, it["clip"])
    rmeta = json.loads((Path(it["render_dir"]) / "render_meta.json").read_text())
    return ep, rmeta, CM.render_fix(ep["cols"], tf.entities, ep["channels"], rmeta)


def _write_episode_files(out, i, fm, extra, it, rmeta, ep):
    """Parquet, videos, depth, head video and forces of episode i."""
    feats = write_parquet(out / f"data/chunk-000/episode_{i:06d}.parquet", fm["cols"], extra, fm["names"])
    for v, key in VIDEO_KEYS.items():
        dst = out / f"videos/chunk-000/{key}/episode_{i:06d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(it["render_dir"]) / f"{v}.mp4", dst)
    CD.place(Path(it["render_dir"]) / CD.DEPTH_FILE, out / f"depth/chunk-000/{DEPTH_KEY}/episode_{i:06d}{CD.EXT}")
    HV.store(Path(it["render_dir"]), rmeta, out, i, it["clip"])
    fo, ft = CF.force_timeline(ep["forces"], ep["step_dt"], ep["physics_dt"], ep["L"])
    frate = None
    if ft is not None and len(ft) > 1:  # force rate measured on the actual time axis
        frate = round(1.0 / float(np.median(np.diff(ft))), 3)
    pin = CF.pin_steps(ep["spawn_hold_steps"], Path(it["traj"]), ep["clip"])
    masks, fq = CF.quality({k: (v["force_n"], v["body_names"]) for k, v in fo.items()}, ft, ep["step_dt"], pin)
    CF.write(out / f"forces/chunk-000/episode_{i:06d}.npz", fo, ft, masks)
    return feats, fo, frate, pin, fq


def _add_optional_fields(rmeta, line, it, tx, tt, out, i, vfi, tinfo, tf):
    """Line fields present only for some renders (place marker, look, choice, text, piece, extra/clamp)."""
    if rmeta.get("place_marker"):  # place marker (render.markers), only when on
        line["place_marker"] = rmeta["place_marker"]
    if rmeta.get("look"):  # look field of a clip render (render.clip)
        if rmeta["look"].get("name") != it.get("look"):
            raise SystemExit(
                f"[COLLECT_STORE] {it['render_dir']}: render_meta look {rmeta['look'].get('name')} != "
                f"plan {it.get('look')}"
            )
        line["look"] = rmeta["look"]
    if it.get("choice"):  # trajectory choice and shortfall (render.clip)
        line.update(traj_choice=it["choice"], traj_short=it.get("traj_short"))
    if tx:  # instruction text, only when a text table is set
        line.update(text={k: v for k, v in tx.items() if k != "subtasks"}, source_label=tt["source_label"])
        if tx.get("subtasks"):  # stage texts: intervals measured on this episode
            line["subtasks"] = episode_subtasks(
                out / f"data/chunk-000/episode_{i:06d}.parquet", out / f"forces/chunk-000/episode_{i:06d}.npz", vfi, tx
            )
    seg = tinfo.get("segment") or {}
    if int(seg.get("n") or 1) > 1:  # piece clip: original take, index, source range
        line["piece"] = {
            "parent_clip": tinfo.get("parent_clip"),
            "index": seg.get("index"),
            "of": seg.get("n"),
            "source_range": [seg.get("start"), seg.get("end")],
            "source_fps": seg.get("fps"),
            "cut_source": seg.get("cut_source"),
        }
    xr = CM.extra_ref_motion(Path(it["task_info"]), tf.meta)  # only for records with extra slots
    if xr is not None:
        line[CM.EXTRA_REF_KEY] = xr
    rc = CM.artic_range_clamp(Path(it["task_info"]))  # only clips with clamped joint ranges
    if rc is not None:
        line[CM.RANGE_CLAMP_KEY] = rc


def write_episode(out: Path, i: int, it: dict, index0: int, tasks: dict, TF, TS, K: int, J: int = 0) -> dict:
    """Write episode i of a store (parquet = form columns, store.form with the dataset K)."""
    tf = TF.load(it["traj"])
    ep, rmeta, rfix = rendered_episode(tf, TS, it)
    fm = FM.form_episode(ep, tf.entities, tf.meta, it["clip"], K, J)
    hold = CM.f0_hold(
        Path(it["task_info"]),
        ep["step_dt"],
        np.concatenate([v for k, v in sorted(ep["cols"].items()) if k.startswith("reference.object_pose.")], axis=1),
    )
    steps = np.asarray(rmeta["frame_step"])
    vfi = np.clip(np.rint(np.arange(ep["L"]) * ep["step_dt"] * rmeta["fps"]).astype(np.int64), 0, len(steps) - 1)
    task, tx, tt = episode_task(it["clip"], (it.get("selection") or {}).get("source_clip"))
    tinfo = json.loads(Path(it["task_info"]).read_text())
    case = clip_case(tinfo)
    ti = tasks.setdefault(task, len(tasks))
    extra = {
        "episode_index": np.full(ep["L"], i),
        "index": index0 + np.arange(ep["L"]),
        "task_index": np.full(ep["L"], ti),
        "video_frame_index": vfi,
    }
    feats, fo, frate, pin, fq = _write_episode_files(out, i, fm, extra, it, rmeta, ep)
    # render fields = every line field that comes from render_meta
    rpart = dict(
        **rfix,
        render_preset=rmeta["render_preset"],
        render_look={k: rmeta["render_look"][k] for k in LOOK_KEYS},
        render_settle=LOOK.settle_row(rmeta["look_settle"]),
        **CM.hand_usd_carry(rmeta),
    )
    line = dict(
        episode_index=i,
        tasks=[task],
        length=ep["L"],
        clip=it["clip"],
        segment=tt["segment"],
        video_frames=len(steps),
        tracking=ep["errors"],
        end_err=TS.end_err(ep["errors"]),
        selection=it.get("selection", {}),
        source_cell=it.get("cell"),
        loophole=case.pop("loophole"),
        case=case,
        f0_hold=hold,
        force_quality=fq,
        spawn_hold_steps=pin,
        **fm["line"],
        **{CL.ABSENT: fm["absent"]},
        **rpart,
    )
    _add_optional_fields(rmeta, line, it, tx, tt, out, i, vfi, tinfo, tf)
    return dict(
        ep,
        line=line,
        render_keys=tuple(rpart),
        feats=feats,
        frate=frate,
        rmeta=rmeta,
        pin=pin,
        fo=fo,
        cam=CM.camera_intrinsics(rmeta),
        form=(fm["joints"], fm["width"]),
        env=data_env(tf.meta),
    )


def data_env(meta: dict) -> tuple[str, str, str]:
    """Record meta -> (hand, arm, dataset): the axis of the form K (store.slots.dataset_k)."""
    e = meta["env"]
    return e["hand"], e["arm"], e["dataset"]


def cameras_info(cams: list[dict]) -> dict:
    """Per-episode intrinsics -> one dataset value (info.json collect.cameras) plus the stream -> camera map.
    Episodes that differ are an error (the differing fields are named)."""
    ref = json.dumps(cams[0], sort_keys=True)
    bad = sorted(
        {
            f"ep{i}:{k}"
            for i, c in enumerate(cams)
            if json.dumps(c, sort_keys=True) != ref
            for k in set(c) | set(cams[0])
            if json.dumps(c.get(k), sort_keys=True) != json.dumps(cams[0].get(k), sort_keys=True)
        }
    )
    if bad:
        raise SystemExit(f"[COLLECT_STORE] camera intrinsics differ between episodes {bad[:6]} — one value per dataset")
    # stream labels are stored in info.json (kept byte-identical to existing datasets)
    streams = {
        VIDEO_KEYS["egoL"]: "ego (left eye)",
        VIDEO_KEYS["egoR"]: "ego (right eye)",
        DEPTH_KEY: "ego (left eye)",
        **({VIDEO_KEYS["exo"]: "exo"} if "exo" in VIDEO_KEYS else {}),
    }
    return dict(cams[0], streams=streams)


def fresh_out(out: Path) -> int:
    """Empty and recreate the store folder; returns the number of old episodes removed (a re-run rewrites it)."""
    old = len(list((out / "data/chunk-000").glob("episode_*.parquet"))) if out.is_dir() else 0
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return old


def _write_meta(out, info, source, eps, pq_path, features, absent, vids):
    """info.json, collect.json and the per-episode stats."""
    (out / "meta/info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=float))
    (out / "meta/collect.json").write_text(json.dumps(source, indent=1, ensure_ascii=False, default=float))
    CL.write_stats(
        out,
        [
            {
                "episode_index": i,
                "stats": CL.episode_stats(
                    pq_path[i], features, absent[i], {k: v[i] for k, v in vids.items()}, _ffprobe, CL.ffmpeg_bin()
                ),
            }
            for i in range(len(eps))
        ],
    )


def _count_check(out, info, items, total):
    """Count the written episodes and depth files; the DONE line; a count mismatch dies."""
    got = len(list((out / "data/chunk-000").glob("episode_*.parquet")))
    holes = {k: v for k, v in info["collect"]["loophole_counts"].items() if v}
    dfiles = list((out / f"depth/chunk-000/{DEPTH_KEY}").glob(f"*{CD.EXT}"))
    print(
        f"[COLLECT_STORE] DONE episodes {got}/{len(items)} · frames {total} · loophole {holes or 0} · unique depth "
        f"{len({f.stat().st_ino for f in dfiles})}/{len(dfiles)} (same content = hard link) · {out}",
        flush=True,
    )
    if got != len(items):
        raise SystemExit("[COLLECT_STORE] episode count mismatch")
    return {"episodes": got, "frames": total}


def store(out: Path, items: list[dict], source: dict) -> dict:
    """items = [{traj, clip, render_dir, ...}] -> dataset at out. Returns a summary (counts)."""
    TF, TS = eval_modules()
    if n_old := fresh_out(out):
        print(f"[COLLECT_STORE] removed old store — {n_old} episodes · {out}", flush=True)
    tasks, eps, total, frates, pins, cams, feats, chans, fo = {}, [], 0, set(), set(), [], [], [], {}
    env0 = data_env(TF.load(items[0]["traj"]).meta)
    K, ksrc = CK.dataset_k(*env0)  # form K and articulated width J from the statement
    J = CK.dataset_j(*env0)
    ep0 = None
    for i, it in enumerate(items):
        ep = write_episode(out, i, it, total, tasks, TF, TS, K, J)
        ep0 = ep0 or ep
        if ep["env"] != env0 or ep["form"] != ep0["form"]:
            raise SystemExit(
                f"[COLLECT_STORE] {it['clip']}: hand x dataset {ep['env']} or form differs from the "
                f"first episode {env0} — one store = one form"
            )
        frates |= {ep["frate"]} - {None}
        pins.add(ep["pin"])
        cams.append(ep["cam"])
        eps.append(ep["line"])
        feats.append(ep["feats"])
        rmeta, total = ep["rmeta"], total + ep["L"]
        chans.append(ep["channels"])
        fo |= ep["fo"]  # force keys = union over episodes (they differ by side)
    lay = {"chunks_size": CHUNKS_SIZE}  # episode_file paths use the info path patterns below
    pq_path = [episode_file(out, lay, DATA_PATH, i) for i in range(len(eps))]
    features, extra = CL.unify(feats)  # one form: every episode must have the same features
    if any(extra):
        raise SystemExit(
            f"[COLLECT_STORE] episode features differ from the form "
            f"{[(i, a[:2]) for i, a in enumerate(extra) if a][:3]}"
        )
    absent = [FM.stats_skip(e) for e in eps]  # features left out of the stats
    vids = {
        k: [episode_file(out, lay, VIDEO_PATH, i, video_key=k) for i in range(len(eps))] for k in VIDEO_KEYS.values()
    }
    features |= CL.video_features(_ffprobe, vids, rmeta["fps"])
    write_task_meta(out, eps, tasks)
    info = dict(
        codebase_version=CODEBASE,
        robot_type=source.get("robot_type"),
        **CL.totals(len(eps), total, len(tasks), CHUNKS_SIZE, len(VIDEO_KEYS)),
        chunks_size=CHUNKS_SIZE,
        fps=round(1.0 / ep["step_dt"]),
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
        features=features,
        collect=dict(
            video_fps=rmeta["fps"],
            video_keys=list(VIDEO_KEYS.values()),
            depth_path=("depth/chunk-{episode_chunk:03d}/" + DEPTH_KEY + "/episode_{episode_index:06d}" + CD.EXT),
            depth_codec=CD.CODEC,
            force_path="forces/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz",
            **HV.info_fields(eps),
            **CF.info_fields(fo, pins, frates),
            physics_hz=round(1.0 / ep["physics_dt"], 3),
            channels=FM.union_field("channels", chans),
            cameras=cameras_info(cams),
            loophole_counts={k: sum(e["loophole"] == k for e in eps) for k in LOOPHOLES},
            **{FM.FORM_KEY: FM.spec(K, ksrc, *ep0["form"], J=J)},
            # note text is stored in info.json (the old Korean value is in phase5 legacy_values.tsv)
            note="low-dim = policy steps · video aligned by video_frame_index · depth and force are outside the "
            "LeRobot standard · depth files with the same content are hard links (one copy on disk — move them with "
            "tar · rsync -H) · episode loophole = collect_store.loophole",
        ),
    )
    _write_meta(out, info, source, eps, pq_path, features, absent, vids)
    return _count_check(out, info, items, total)


# ---- video probing (frame counts; shared with the preview and the checks) ----
def _ffprobe(p: Path) -> dict:
    """Probe one video (decoded frame count, codec, size, fps); each content is measured once per process (the cache
    owner is store.lerobot, keyed by content, so the dataset's copies of the rendered videos are not decoded again)."""
    return CL.probe_video(p)
