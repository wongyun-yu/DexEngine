"""Per-episode corrections and metadata of the store step (CPU, retarget venv); store.store calls these per episode.

Head joints (render_head): the stored head joints are what the render session actually drew. The source of truth is
  the render measurement (render_meta.json `head`: render.rig.Rig.head_readback reads every head joint back from PhysX
  each video frame; `pinned_rad` = joints the rig pins). Pinned joints replace the whole state and absolute joint-target
  columns with the measured value (it must be constant over frames); unpinned joints (driven by the trajectory) are
  checked against the stored state frame by frame. A mismatch is an error.
  The rollout range stays in the episode meta `head_joints` (provenance of the post-rollout parquet edit; it is kept in
  meta only, so no consumer can read a stale column). Premise: the head is not in the policy's observation or action,
  the torso is a fixed joint (no load path to the arms), and the head colliders (camera links) do not collide with the
  arms, hands or discs, so drawing a different head angle leaves the manipulation physics unchanged.
F0 hold (f0_hold): the length of the frame-0 padding at the episode start (frame 0 repeated P times; wrists and
  objects still) is read from its owners (coords.tree_f0_pad, coords._base_to_tree, retiming included) and written in
  steps and seconds (0 for clips without padding). Independent check: the reference object pose must stay still during
  the padding; the hold = the measured still steps (capped by the padding length). The only reason an object may move
  inside the padding is the start ramp of a cross repair (a C1 ramp outside the window that carries a held object),
  so movement inside the padding without that repair's trace (xr_wrist_shift_world_*) is an error. The step after the
  padding (original frame 0) is a fractional retiming interpolation and is not checked.
Cameras (camera_intrinsics, camera_pose): the values in render_meta.json `camera` (render.rig.Rig.calibrate) are copied
  verbatim, never recomputed; a missing field is an error.
  Intrinsics (one per dataset, info.json collect.cameras): ego = pinhole of one eye of the rectified stereo pair (both
  eyes share K), distortion, resolution, fps, baseline; exo = K and resolution (Isaac pinhole, no distortion recorded).
  Extrinsics (per episode, episodes.jsonl camera_pose; head pitch and the exo dolly differ per clip): both ego eye
  positions and quat_ros_wxyz (camera -> world, optical axis +z, image down +y), exo position and quat_opengl_wxyz
  (optical axis -z, image up +y), the canonical position before the dolly, the dolly distance and the head joints.
  Frame = render scene env origin, metres (the render scene with the robot body on the floor; episode
  body.world_shift_m). The robot-base pose is not in render_meta and is not stored.
Hand USD (hand_usd_meta): the spawned hand USD (render_meta and the episode line `hand_usd`): a re-baked file with the
  same name differs from the old copy only by md5. Path = robots.selection.combo_usd_path (the variant env of the render
  context, i.e. the file the spawn opened); md5 = robots.derived_usd.file_md5; pin = robots.derived_usd.check_pin's
  line (it dies on a mismatch with its expected pins). The render writes it per episode because the spawn marker is
  printed once per process (a reused Kit worker would not log it again).
Short trajectories (traj_short): how many trajectories a clip got versus how many it needed, and why it is short.

Some strings below (CAMERA_FRAME, CAMERA_AXES, the camera `model`/`source` texts, loader `note`, the traj_short
reasons) are written into dataset metadata; they are kept byte-identical to existing datasets.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError

MARK = "[COLLECT_STORE]"
HEAD_TOL_RAD = 1e-6  # measured vs stored (float32) tolerance
REF_STATIC_TOL = 1e-6  # "still" tolerance of the reference object pose (padding is a byte copy; only rounding)
XR_SHIFT_PREFIX = "xr_wrist_shift_world_"  # wrist shift columns that cross repair leaves in a tree
EXTRA_REF_KEY = "extra_ref_motion"  # episode meta field: reference-driven motion of extra objects
HAND_USD_KEY = "hand_usd"  # render_meta and episode line field of the spawned hand USD
SHORT_ORIG_OUT, SHORT_UNKNOWN = "original out of view", "no judge end record"  # traj_short reasons (stored in data)
TRAJ_SHORT_MARK = "[TRAJ_SHORT]"


def selection_of(tf, clip: str) -> dict:
    """The clip's selection table: a revived record (traj_file version 102 — no all-episode summary) gives the stored
    line's copy (restored.selection), any other the recorder's table (rollout.recorder.selection_table). The revived
    branch stays out of the recorder: it is in the stage-1 code fingerprint (record.cache.CODE_FILES)."""
    from dexengine.record.rollout import recorder as REC

    if getattr(tf, "restored", None):
        sel = tf.restored.get("selection")
        if not sel or sel.get("clip") != clip:
            raise SystemExit(
                f"[COLLECT_STORE] {clip}: the revived record has no selection copy of this clip — {tf.path}"
            )
        print(
            f"[COLLECT_STORE] [RESTORED_SELECTION] {clip}: selection = the stored line's copy (no kept_row oracle)",
            flush=True,
        )
        return sel
    return REC.selection_table(tf, clip)


def _deg(x) -> float:
    return round(math.degrees(float(x)), 4)


def head_entity(entities: list, names: list) -> dict:
    """The one robot articulation (not an object) that has the head joints; none or several is an error."""
    hits = [
        e
        for e in entities
        if e["kind"] == "artic" and not e["name"].startswith("object") and set(names) <= set(e["joint_names"])
    ]
    if len(hits) != 1:
        raise SystemExit(f"{MARK} robot entities with head joints {names}: {[e['name'] for e in hits]} — need one")
    return hits[0]


def render_head(cols: dict, entities: list, channels: list, rmeta: dict) -> dict:
    """Set the head joints of cols (episode columns, replaced in place) to the render measurement.

    Returns the episode meta `head_joints`."""
    h = rmeta.get("head")
    if not h:
        raise InputMissingError(f"{MARK} render_meta has no head (head joint measurement) — re-render")
    names, pinned = list(h["joints"]), dict(h["pinned_rad"])
    fr, steps = np.asarray(h["frame_rad"], float), np.asarray(rmeta["frame_step"], int)
    if fr.shape != (len(steps), len(names)) or not set(pinned) <= set(names):
        raise SystemExit(
            f"{MARK} render_meta head shape {fr.shape} · pinned {sorted(pinned)} != frames {len(steps)} x {names}"
        )
    ent = head_entity(entities, names)
    sk = f"observation.joint_pos.{ent['name']}"
    tch = [c for c in channels if c["kind"] == "joint_target" and c["name"].split("/", 1)[1] == ent["name"]]
    tk = f"action.joint_target.{ent['name']}" if tch else None
    cols[sk] = np.array(cols[sk], copy=True)
    if tk:
        cols[tk] = np.array(cols[tk], copy=True)
    out = {}
    for j, n in enumerate(names):
        i = ent["joint_names"].index(n)
        rec = {"rollout_state_deg": [_deg(cols[sk][:, i].min()), _deg(cols[sk][:, i].max())]}
        ti = tch[0]["names"].index(n) if tk and n in tch[0]["names"] else None
        if ti is not None:
            rec["rollout_target_deg"] = [_deg(cols[tk][:, ti].min()), _deg(cols[tk][:, ti].max())]
        if n in pinned:
            v = float(fr[0, j])
            d = float(max(np.abs(fr[:, j] - v).max(), abs(v - float(pinned[n]))))
            if d > HEAD_TOL_RAD:
                raise SystemExit(
                    f"{MARK} pinned head joint {n}: frame measurement is {d:.2e} rad off the pinned "
                    f"value {pinned[n]:.6f}"
                )
            cols[sk][:, i] = v
            if ti is not None:
                cols[tk][:, ti] = v
            out[n] = dict(source="render_pinned", render_deg=_deg(v), frame_max_diff_rad=d, **rec)
        else:
            d = float(np.abs(fr[:, j] - cols[sk][steps, i]).max())
            if d > HEAD_TOL_RAD:
                raise SystemExit(
                    f"{MARK} head joint {n}: render measurement != stored state (trajectory) by "
                    f"{d:.2e} rad — the render drew a different value"
                )
            out[n] = dict(source="trajectory", frame_max_diff_rad=d, **rec)
    return out


def world_shift_m(body: dict) -> float:
    """World z shift (m) measured by the render session; non-zero only for the world-moving modes
    (render.desk_stretch.world_moved: aerial world_shift, desk desk_stretch)."""
    from dexengine.record.render import desk_stretch as DS

    return float(body["world_shift_m"]) if DS.world_moved(body) else 0.0


#: Stored column prefixes that move with the world: objects (incl. support discs and receiving-object slots, from
#: which store.form derives the line's supports / support_objects) and the reference object pose.
WORLD_SHIFT_COLS = ("observation.object_pose.", "reference.object_pose.")


def world_shift(cols: dict, body: dict, sign: float = 1.0) -> int:
    """Apply the render session's world shift to the stored columns: add sign × world_shift_m to z (index 2 of the
    7-wide pose) of every object / reference object pose column (copies). sign +1 = store (render_fix) · -1 = undo
    (a reviver: stored columns back to the stage-1 frame). Other
    modes are unchanged. Returns columns moved."""
    from dexengine.record.render import desk_stretch as DS

    if sign not in (1.0, -1.0):
        raise SystemExit(f"{MARK} world_shift sign {sign!r} — +1 (store) or -1 (undo) only")
    if not DS.world_moved(body):
        return 0
    keys = [k for k in cols if k.startswith(WORLD_SHIFT_COLS)]
    for k in keys:
        v = np.array(cols[k], dtype=np.float64, copy=True)
        if v.ndim != 2 or v.shape[1] != 7:
            raise SystemExit(f"{MARK} {k} shape {v.shape} — must be 7-wide poses")
        v[:, 2] += sign * world_shift_m(body)
        cols[k] = v.astype(np.asarray(cols[k]).dtype)
    return len(keys)


CAMERA_KEYS = (
    "zed",
    "K_left",
    "K_exo",
    "exo_res",
    "ego_left_pose",
    "ego_right_pose",
    "exo",
    "head_joint",
    "head_pinned_deg",
)
ZED_KEYS = ("width", "height", "fx", "fy", "cx", "cy", "distortion", "fps", "baseline_m", "mode")
EGO_POSE_KEYS = ("pos", "quat_ros_wxyz")
EXO_POSE_KEYS = ("pos", "quat_opengl_wxyz", "canon_pos", "delta_m")
CAMERA_FRAME = (
    "render scene env origin — position m · rotation = camera → world (wxyz) · the render scene puts "
    "the robot body on the floor (collect_body — episode body.world_shift_m)"
)
CAMERA_AXES = {
    "quat_ros_wxyz": "optical axis +z · image right +x · image down +y",
    "quat_opengl_wxyz": "optical axis −z · image right +x · image up +y",
}


def _camera(rmeta: dict) -> dict:
    """render_meta `camera`; any missing field is an error."""
    cam = rmeta.get("camera")
    miss = (
        ["camera"]
        if not isinstance(cam, dict)
        else [k for k in CAMERA_KEYS if k not in cam]
        + [f"zed.{k}" for k in ZED_KEYS if k not in (cam.get("zed") or {})]
        + [
            f"{e}.{k}"
            for e in ("ego_left_pose", "ego_right_pose")
            for k in EGO_POSE_KEYS
            if k not in (cam.get(e) or {})
        ]
        + [f"exo.{k}" for k in EXO_POSE_KEYS if k not in (cam.get("exo") or {})]
    )
    if miss:
        raise InputMissingError(f"{MARK} render_meta camera fields missing {miss[:6]} — re-render")
    return cam


def camera_intrinsics(rmeta: dict) -> dict:
    """Camera intrinsics (one per dataset; store.store checks they agree across episodes) from render_meta `camera`."""
    cam = _camera(rmeta)
    z = cam["zed"]
    return {
        "ego": dict(
            {k: z[k] for k in ZED_KEYS},
            K=cam["K_left"],
            model="pinhole · rectified stereo — same K for both eyes · baseline = left eye → right eye",
        ),
        "exo": {
            "K": cam["K_exo"],
            "width": cam["exo_res"][0],
            "height": cam["exo_res"][1],
            "model": "pinhole (Isaac render — no distortion field in render_meta)",
        },
        "video_fps": rmeta["fps"],
        "source": "render_meta.json camera (collect_rig.Rig.calibrate) as is",
    }


def camera_pose(rmeta: dict) -> dict:
    """Camera extrinsics (per episode, episodes.jsonl camera_pose) from render_meta `camera`."""
    cam = _camera(rmeta)
    return {
        "frame": CAMERA_FRAME,
        "axes": CAMERA_AXES,
        "ego_left": {k: cam["ego_left_pose"][k] for k in EGO_POSE_KEYS},
        "ego_right": {k: cam["ego_right_pose"][k] for k in EGO_POSE_KEYS},
        "exo": {k: cam["exo"][k] for k in EXO_POSE_KEYS},
        "head_joint": cam["head_joint"],
        "head_pinned_deg": cam["head_pinned_deg"],
    }


def loader_meta(video_fps) -> dict:
    """LeRobotDataset loader tolerance (meta/collect.json `loader`): tolerance_s = 1/(2 x video fps), so the nearest
    video frame of every low-dimensional timestamp (info fps = 1/step_dt) is within half a frame."""
    fps = int(video_fps)
    tol = 1.0 / (2 * fps)
    return {
        "video_fps": fps,
        "tolerance_s": tol,
        "rule": "tolerance_s = 1/(2 × video_fps)",
        "note": f"LeRobotDataset(repo_id, root=<final dataset>, tolerance_s={tol:.6f}) — video {fps} fps · "
        "low-dimensional table = info fps (episode video_frame_index = the nearest video frame)",
    }


def render_fix(cols: dict, entities: list, channels: list, rmeta: dict) -> dict:
    """Align the stored columns with the render measurement (head joints, world shift); return the episode meta."""
    if "body" not in rmeta:
        raise InputMissingError(f"{MARK} render_meta has no body (body height) — re-render")
    head = render_head(cols, entities, channels, rmeta)
    world_shift(cols, rmeta["body"])
    fr = (rmeta.get("camera") or {}).get("framing") or {}
    # Optional fields are copied only when the render wrote them (extras_visual is None when absent).
    return dict(
        head_joints=head,
        body=rmeta["body"],
        idle_hand=rmeta["idle_hand"],
        visibility=rmeta["visibility"],
        camera_pose=camera_pose(rmeta),
        extras_visual=rmeta.get("extras_visual"),
        **({"head_view": rmeta["head_view"]} if "head_view" in rmeta else {}),
        **({"ego_view": rmeta["ego_view"]} if "ego_view" in rmeta else {}),
        **({"exo_view": rmeta["exo_view"]} if "exo_view" in rmeta else {}),
        **({"round": rmeta["round"]} if "round" in rmeta else {}),
        framing={
            k: fr.get(k)
            for k in (
                "rule",
                "head_rule",
                "scene",
                "before",
                "after",
                "desk_fill",
                "subset_frac",
                "yaw_deg",
                "pitch_deg",
                "grasp_fail",
            )
        },
    )


def _runs(mask: np.ndarray) -> list[list[int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return [[int(a), int(b)] for a, b in zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]], strict=True)]


def extra_ref_motion(task_info: Path, traj_meta: dict) -> list | None:
    """Episode meta `extra_ref_motion`: intervals in which an extra object moves by the reference, not by the robot
    (so users can filter them). Targets = objects the slot rule (data.objects.extra_slots.slot_rows) made
    reference-driven (static spawns excluded). Per object:
      object, name, slot;
      intervals_steps / intervals_s: tree-frame runs where the reference pose differs from the previous frame
        (> REF_STATIC_TOL; inclusive ends; 1 step = 1 reference frame = ref_dt), moving_s = their total;
      robot_contact from the trajectory record meta extra_slots (accumulated over the judge session, per clip):
        held_frac = share of human-held steps with robot contact, moving_contact_frac = share of moving steps with
        robot contact.
    None when slots were off for the record (no extra_slots in the meta). Rule rows and recorded objects must match."""
    if "extra_slots" not in traj_meta:
        return None
    from dexengine.data.objects import extra_slots as XS

    tree = Path(task_info).parent
    ti = json.loads(Path(task_info).read_text())
    clip = tree.name
    summ = {d["obj"]: d for d in (traj_meta["extra_slots"].get(clip) or [])}
    rows, _ = XS.slot_rows(str(tree), ti)
    if sorted(summ) != sorted(r["obj_id"] for r in rows):
        raise SystemExit(
            f"{MARK} {clip}: slot rule objects {sorted(r['obj_id'] for r in rows)} != recorded {sorted(summ)}"
        )
    q = np.load(tree / "0" / "trajectory_kinematic.npz")["qpos"]
    dt = float(ti.get("ref_dt") or 0.0)
    if dt <= 0:
        raise InputMissingError(f"{MARK} {clip}: no ref_dt in task_info — cannot convert intervals to seconds")
    out = []
    for k, r in enumerate(rows):
        if r["static"]:
            continue
        p = q[:, r["qpos_start"] : r["qpos_start"] + 7].astype(np.float64)
        mv = np.r_[False, np.abs(np.diff(p, axis=0)).max(axis=1) > REF_STATIC_TOL]
        iv = _runs(mv)
        d = summ[r["obj_id"]]
        frac = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
        out.append(
            {
                "object": r["obj_id"],
                "name": r["name"],
                "slot": k,
                "intervals_steps": iv,
                "intervals_s": [[round(a * dt, 3), round(b * dt, 3)] for a, b in iv],
                "moving_s": round(float(mv.sum()) * dt, 3),
                "robot_contact": {
                    "scope": "judge_session_clip",
                    "held_side": d["held_side"],
                    "held_frac": frac(d["held_contact"], d["held_steps"]),
                    "moving_contact_frac": frac(d["moving_steps"] - d["moving_nocontact"], d["moving_steps"]),
                },
            }
        )
    return out


RANGE_CLAMP_KEY = "artic_range_clamp"


def artic_range_clamp(task_info: Path) -> dict | None:
    """Episode meta `artic_range_clamp`: tree-frame runs where an articulated object's joint was clamped to its range
    (the converter's rules.D1g.range_clamp verbatim). None for clips without clamped frames."""
    ti = json.loads(Path(task_info).read_text())
    return ((ti.get("rules") or {}).get("D1g") or {}).get("range_clamp") or None


def static_steps(ref: np.ndarray) -> int:
    """Number of leading steps of a reference pose column [L, D] equal (REF_STATIC_TOL) to step 0."""
    same = np.all(np.abs(ref - ref[:1]) <= REF_STATIC_TOL, axis=1)
    return int(len(same) if same.all() else np.argmin(same))


def f0_hold(task_info: Path, step_dt: float, ref: np.ndarray) -> dict:
    """The still part of the frame-0 padding: steps [0, steps) and seconds.

    The tree clock must equal the policy step. If the reference object moves inside the padding, the still part ends
    there (ramp_steps = padding - still) and the tree must carry a cross-repair trace; otherwise it is an error."""
    from dexengine.placement.coords import core as CO

    tree, ti = task_info.parent, json.loads(task_info.read_text())
    dt = CO.tree_dt(tree, ti)
    if abs(dt - step_dt) > 1e-9:
        raise InputMissingError(
            f"{MARK} {tree.name}: tree clock {dt} != policy step {step_dt} — cannot map the padding"
        )
    pad = CO.tree_f0_pad(tree, ti)
    hold = CO._base_to_tree(tree, ti)(0)
    n = static_steps(ref)
    steps = min(int(hold), n)
    if steps < hold:
        with np.load(tree / "0" / "trajectory_kinematic.npz") as z:
            xr = sorted(k for k in z.files if k.startswith(XR_SHIFT_PREFIX))
        if not xr:
            raise SystemExit(
                f"{MARK} {tree.name}: the reference object moves inside the padding (still {n} < "
                f"padding {hold}) without a cross-repair trace ({XR_SHIFT_PREFIX}*)"
            )
    return {
        "steps": steps,
        "seconds": round(steps * dt, 6),
        "pad_steps": int(hold),
        "ramp_steps": int(hold) - steps,
        "pad_frames": int(pad),
        "reference_static_steps": n,
        "owner": "coords.tree_f0_pad · coords._base_to_tree",
    }


def hand_usd_meta() -> dict:
    """{name, md5, pin} of the spawned hand USD (inside Kit only)."""
    from dexengine.robots import derived_usd as DU
    from dexengine.robots import selection as RS

    usd = RS.combo_usd_path(os.environ["DEXENGINE_ARM"], os.environ["DEXENGINE_HAND"], RS.USD_CACHE_DIR)
    return {"name": Path(usd).stem, "md5": DU.file_md5(usd), "pin": DU.check_pin(usd)}


def hand_usd_carry(rmeta: dict) -> dict:
    """render_meta -> the field to add to the episode line ({HAND_USD_KEY: ...} or {} when the render had none)."""
    return {HAND_USD_KEY: rmeta[HAND_USD_KEY]} if HAND_USD_KEY in rmeta else {}


def traj_short(choice: dict, judge_end: str | None, detail: str = "") -> dict:
    """Augmented-round choice record (episodes.choose_rows traj_choice) + judge end reason ->
    {trajs, need, reason, detail}.

    trajs = trajectories of the clip (original visible: original 1 (orig round) + rows 1..; original off screen:
    row 0 + rows 1..), need = 1 + n. reason = None when not short; SHORT_ORIG_OUT when off screen and row 0 is the
    original; else judge_end (SHORT_UNKNOWN when None). detail = the record stage's shortfall reason."""
    n, chosen, of = int(choice["n"]), list(choice["chosen"]), choice.get("orig_fov") or {}
    out = not of.get("orig_kept", True)
    trajs = ((1 if choice.get("row0") else 0) if out else 1) + len(chosen)
    reason = None
    if trajs < 1 + n:
        reason = SHORT_ORIG_OUT if (out and of.get("row0_source") == "orig") else (judge_end or SHORT_UNKNOWN)
    return {"trajs": trajs, "need": 1 + n, "reason": reason, "detail": detail or ""}


def attach_traj_short(clip: str, got: list[dict], out) -> None:
    """Set traj_short on every episode item of an augmented clip (items with `choice`), from <out>/aug/counts.json
    (always present in the aug round). Prints one line when the clip is short."""
    if not got or not got[0].get("choice"):
        return
    counts = json.loads((Path(out) / "aug" / "counts.json").read_text()).get("counts", [])
    cc = {c["clip"]: c for c in counts}.get(clip) or {}
    ts = traj_short(got[0]["choice"], cc.get("judge_end"), cc.get("reason") or "")
    for e in got:
        e["traj_short"] = ts
    if ts["reason"]:
        print(
            f"{TRAJ_SHORT_MARK} {clip}: trajectories {ts['trajs']}/{ts['need']} · reason {ts['reason']}"
            + (f" · {ts['detail']}" if ts["detail"] else ""),
            flush=True,
        )
