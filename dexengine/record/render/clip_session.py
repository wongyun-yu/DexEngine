"""Per-clip render (one session = one clip: original + placement-variant trajectories, an episode plan);
session.entry --stage render_clip. This module owns the plan contract and the fixed per-session state; the render
loop (render_clip) and its head phases (choose_phase, orig_camera, orig_fov_phase) live in render.clip_phases.

Contract (plan.json v2, planned by record.episodes / render.clip):
  {"version": 2, "clip": <original tree name>, "out": <folder>, "seq_dir": <tree root; the shadow side for augment>,
   "seq_list": <list file: original + variants, list names (no tree suffix)>,
   "trajs": [{"name": <tree name>, "traj": <chosen trajectory .npz, TF.load(..).selected(name)>, "role": "orig"|"aug"}
             … (original first)], "look_key": <random look key>, "round" · "deck" (record.episodes.plan),
   "episodes": [{"traj": <tree name>, "look": "D" | "r<row>_s<col>", "row": k, "col": c | null} …] | null,
   "choose": {"n": <number of trajectories>, "rule": record.episodes.CHOOSE_RULE} (only then; episodes = null)}
  Trajectory choice = choose_phase (rule record.episodes.choose) -> render_clip.json choice · episodes_plan (the
  receiving render.clip checks it against the outputs as a set).
  Output = per episode <out>/<tree name>/<look>/ with the MEDIA of render_selected + render_meta.json (field look
  {name, spec, seed_key}, episode_index added) · bundle summary <out>/render_clip.json · end line MARK DONE
  episodes=<n>/<m> (n < m gives rc != 0). Writing (mp4, depth) runs on the render.writer background thread: closing
  i overlaps the render of i+1 and is joined at the end of i+1's frames, then i's render_meta and DONE line (overlap
  depth 1; summary maxrss_gb = the session's peak RSS).
  allow (the random-look allow list) is measured on the original trajectory only (looks.look_run,
  record.episodes.allow_traj; for the augment round choose_phase measures it while the original is loaded).
How one session loads four trajectories (LOAD_METHOD): the entry overrides the dump argv --sequence_dir with
plan.seq_dir and env DEXENGINE_SEQ_LIST with plan.seq_list -> the dataset loader loads every listed clip (clip_names)
and trajectory k is chosen by a fixed reset clip_ids = clip_names.index(name) (render.episode.Session).
Fixed for the session (owned here):
  cams_off(rig)   render off for unused cameras (OFF_KEYS; only their pose and intrinsics are read), once after
                  calibrate. An Isaac Lab Camera holds only its render product path, so the HydraTexture is found in
                  the replicator texture registry (the same path annotators.attach uses), updates_enabled = False,
                  read back, and removed from the Rig update list (active) (updating a camera that is off gives a CUDA
                  illegal memory access a few renders later). Marker OFF_MARK.
                  Measured effect: settle 921 -> ~219 renders · step 83 -> 49 ms · first-frame brightness -0.008.
  check_view      later trajectories: the camera is not refitted (head angle = the original's); check that calibrate
                  would give the same answer (upper body, both eyes, desk), else die. Hand and object in-frame ratios
                  are only recorded (view.fill -> render_meta).
  restore         before switching trajectories, remove that trajectory's supports and extra prims and run the scene
                  prim oracle (prim set = the first trajectory's set before supports and extras, look-layer prims
                  LOOK_OWNED excluded); a difference dies.
Random look (episode look != D): the canonical measurement and application of looks.augvis.render._Run
  (measure_base, build_swept, apply_spec) in the session's trajectory context (LookRun); convergence is
  render_episode's single LOOK.settle. Plan order = every look D first (there is no way back to D: D after a random
  look dies) · one swept volume per trajectory. The asset root env AUGVIS_ASSETS is required (no default).
"""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.looks import rev as RD  # re-render (plan redraw field)
from dexengine.record.render import cameras as CAMS
from dexengine.record.render import markers as PM  # clears the place markers (off = no prims)
from dexengine.record.scene import extras as EXV
from dexengine.record.scene import supports as SUP

MARK = "[RENDER_CLIP]"
OFF_KEYS = ("head_camera", "front_camera") + (() if CAMS.exo_on() else ("collect_exo_front",))
OFF_MARK = "[COLLECT_CAM] off head · front"  # must be in the render log (log_problems; invalid without it)
FIXED_TOL_M = 1e-4  # check_view: upper-body links, both eyes, desk within this of the first calibrate
DOLLY_TOL_M = 1e-3  # check_view: exo dolly Δ re-measured with this trajectory's points within this of the first Δ
LOAD_METHOD = (
    "the entry (session.entry --stage render_clip) overrides dump argv --sequence_dir <- plan.seq_dir and "
    "env DEXENGINE_SEQ_LIST <- plan.seq_list · trajectory k = fixed reset clip_ids (clip_names.index)"
)
PLAN_KEYS = ("clip", "out", "seq_dir", "seq_list", "trajs", "episodes", "round", "deck")
#: plan.per_look values (planned by render.clip.per_look); the two head sessions draw no episode:
#:   choose  augment choice + allow measurement (original) -> episodes_plan · _look/allow.json
#:   measure allow measurement of an orig random-look plan (incl. the original off-screen rule) -> _look/allow.json
#:   fixed   a drawing session: record.episodes.EPISODES_PER_SESSION episodes · with a random look, allow comes from
#:           the file only (LookRun.adopt; 0 measure renders: ghost of the original object) · for augment the
#:           original is loaded only for calibrate
#: a plan beyond that (episodes > the session cap, a choice without per_look) dies at the session entry.
PER_LOOK_PHASES = ("choose", "measure", "fixed")
HEAD_PHASES = ("choose", "measure")


def log_problems(text: str) -> list[str]:
    """Render-stage log check (session.launch.classify): a missing camera-off marker is a problem."""
    return [] if OFF_MARK in text else [f"{OFF_MARK} marker missing: unused cameras were not turned off"]


def zero_by_rule(text: str) -> bool:
    """The session drew 0 episodes because of the original off-screen rule (orig skipped, nothing to draw for
    augment): only when both the end line DONE episodes=0/0 and an [ORIG_FOV] line are present. The per-episode
    markers (occlusion, head view) are then rightly absent (classify)."""
    from dexengine.record import episodes as EPI

    lines = text.splitlines()
    return f"{MARK} DONE episodes=0/0 " in text and any(ln.startswith(EPI.ORIG_FOV_MARK) for ln in lines)


def load_plan(path) -> dict:
    """plan.json check: version, fields, original first, no name overlap, list = trajectory names, episode
    trajectories, looks. A mismatch dies."""
    from dexengine.core import paths as P

    d = json.loads(Path(path).read_text())
    miss = [k for k in PLAN_KEYS if k not in d]
    from dexengine.record import episodes as EPI

    try:
        EPI.check_plan(d)
    except ValueError as e:
        miss.append(str(e))
    if miss:
        raise SystemExit(f"{MARK} plan {path}: missing fields · version {miss}")
    names = [t["name"] for t in d["trajs"]]
    if not names or names[0] != d["clip"] or d["trajs"][0].get("role") != "orig" or len(set(names)) != len(names):
        raise InputMissingError(
            f"{MARK} plan trajectories {names}: the original ({d['clip']}) must be first, role "
            "orig, with no name overlap"
        )
    listed = [ln.strip() for ln in Path(d["seq_list"]).read_text().splitlines() if ln.strip()]
    if sorted(P.tree_name(x) for x in listed) != sorted(names):
        raise SystemExit(f"{MARK} list {d['seq_list']} {listed} != plan trajectories {names}")
    ph = d.get("per_look")
    if ph is not None and ph not in PER_LOOK_PHASES:
        raise SystemExit(f"{MARK} plan {path}: per_look {ph!r}: {PER_LOOK_PHASES} only")
    if d.get("choose"):  # trajectory choice: the session picks, then fixes the episodes (episodes.plan n_choose)
        if d["episodes"] is not None or not isinstance(d["choose"].get("n"), int) or d["choose"]["n"] < 1:
            raise InputMissingError(f"{MARK} plan {path}: choose {d['choose']} but episodes already set or no n")
        if ph != "choose":  # old one-session shape (choose and 15 episodes in one session)
            raise SystemExit(
                f"{MARK} plan {path}: a choice plan with per_look={ph!r}; a choice session is "
                "per_look=choose (first episode only, every other look in a new session)"
            )
        return d
    if ph == "choose" or (ph == "measure" and d.get("round") != "orig"):
        raise InputMissingError(
            f"{MARK} plan {path}: per_look={ph} · no choice · round {d.get('round')}: "
            "choose needs a choice plan, measure is orig only"
        )
    bad = [e for e in d["episodes"] if e.get("traj") not in names or not str(e.get("look") or "")]
    if bad or not d["episodes"]:
        raise InputMissingError(f"{MARK} {len(d['episodes'])} episodes · outside the trajectories / no look {bad[:3]}")
    k = EPI.episodes_per_session(d.get("session_rule"))
    if ph not in HEAD_PHASES and len(d["episodes"]) > k:  # old one-session many-looks shape
        raise SystemExit(
            f"{MARK} plan {path}: {len(d['episodes'])} episodes in one session > {k}: "
            "a new session per look (render.clip.per_look)"
        )
    if ph != "fixed" and any(e["look"] != "D" for e in d["episodes"]) and ph not in HEAD_PHASES:
        raise SystemExit(
            f"{MARK} plan {path}: a random look drawn in a per_look={ph!r} session; only fixed sessions "
            "that read the head session's allow file (measure render ghost)"
        )
    if RD.redraw_plan_problems(d):
        raise SystemExit(f"{MARK} plan {path}: {RD.redraw_plan_problems(d)}")
    return d


def cams_off(rig) -> dict:
    """Turn off the render of unused cameras (module doc), once after Rig.calibrate. A read-back other than False
    dies."""
    from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager

    if rig.view_ref is None:
        raise SystemExit(
            "[COLLECT_CAM] cameras turned off before calibrate: turn off only after pose and intrinsics are read"
        )
    if rig.off is not None:
        return rig.off
    vm, rows = ViewportManager(), []
    for k in OFF_KEYS:
        for rp in rig.cams[k].render_product_paths:
            tex = vm._hydra_textures.get(vm._context, rp).hydra_texture
            tex.updates_enabled = False
            if tex.updates_enabled is not False:
                raise SystemExit(f"[COLLECT_CAM] {k} {rp} render-off read-back {tex.updates_enabled!r} != False")
            rows.append({"camera": k, "render_product": rp, "updates_enabled": False})
    rig.active = [k for k in rig.active if k not in OFF_KEYS]
    rig.off = {"cameras": list(OFF_KEYS), "render_products": rows, "active": list(rig.active)}
    print(
        f"{OFF_MARK} · render product {[r['render_product'] for r in rows]} · read-back updates_enabled=False · "
        f"updating {rig.active}",
        flush=True,
    )
    return rig.off


def check_view(rig, groups: dict | None) -> dict:
    """Later trajectory (would calibrate give the same answer?): with the head at its pinned value (the angle the
    first trajectory = the original set; desk fill or trajectory framing, the variants keep that angle) (1) the
    non-arm upper-body links, both eyes and the desk are within FIXED_TOL_M of the first calibrate (input of the head
    pitch rule and the front camera place) (2) the exo dolly Δ re-measured with this trajectory's upper-body points
    (arms included, first frame) is within DOLLY_TOL_M of the first Δ; else die. (3) this trajectory's hand and object
    in-frame ratios (render.framing.stats) are only recorded (out['fill'])."""
    import numpy as np

    from dexengine.record.render import cameras as CC
    from dexengine.record.render import fov as FOW  # numpy: after AppLauncher only ([PRE_APP])

    ref = rig.view_ref
    rig.set_heads(rig.pinned)
    rig.render(2)
    pl, ql, _f, _ = rig.pose("collect_ego_left")
    pr, _q, _f2, _ = rig.pose("collect_ego_right")
    names, P_up = rig._upper_points()
    if names != ref["upper"]:
        raise SystemExit(
            f"[COLLECT_CAM] upper-body link list differs from the first calibrate {names} != {ref['upper']}"
        )
    P = np.vstack([P_up, pl, pr])
    fixed = np.array([("arm_" not in n) for n in names] + [True, True])
    dP = float(np.abs(P[fixed] - ref["P"][fixed]).max())
    d = rig.desk()
    dd = max(abs(float(d[k]) - float(v)) for k, v in ref["desk"].items())
    K, (W, H) = ref["Kf"], ref["WH"]
    delta = float(CC.exo_dolly((ref["R0"].T @ (P - ref["c0"]).T).T, ref["pads"], float(K[0, 0]), float(K[1, 1]), W, H))
    out = {
        "fixed_max_m": dP,
        "desk_max_m": dd,
        "arm_moved_max_m": float(np.abs(P - ref["P"]).max()),
        "delta_ref_m": float(ref["delta"]),
        "delta_now_m": delta,
        "head_rule": ref["framing"].get("head_rule"),
    }
    if dP > FIXED_TOL_M or dd > FIXED_TOL_M or abs(delta - ref["delta"]) > DOLLY_TOL_M:
        raise SystemExit(
            f"[COLLECT_CAM] this trajectory has another camera: torso, head, both eyes {dP * 1000:.3f} mm"
            f" · desk {dd * 1000:.3f} mm (tol {FIXED_TOL_M * 1000:.1f}) · exo dolly {delta:.4f} != first "
            f"{ref['delta']:.4f} m (tol {DOLLY_TOL_M * 1000:.0f} mm): the fixed session camera breaks"
        )
    if groups is not None:
        out.update(FOW.view_measures(groups, pl, ql, ref["kz"]))  # fill + ladder
    print(
        f"[COLLECT_CAM] fixed-camera check · torso, head, both eyes max {dP * 1000:.4f} mm · desk {dd * 1000:.4f} mm"
        f" · exo dolly {delta:.4f} / first {ref['delta']:.4f} m · arm moved {out['arm_moved_max_m'] * 1000:.1f} mm · "
        f"framing {out['head_rule']} (original angle fixed) · this trajectory in frame {out.get('fill')}",
        flush=True,
    )
    return out


def prims() -> set[str]:
    import omni.usd

    return {str(p.GetPath()) for p in omni.usd.get_context().get_stage().TraverseAll()}


def _clear(raw_env, extras: dict, supports: dict) -> list[str]:
    """Remove the extra objects (prims made by scene.extras_apply.apply, empty ROOT) and the supports (discs shown by
    SUP.show -> hidden, material unbound, pillars, support material, empty ROOT). Returns the removed paths."""
    import omni.usd
    from pxr import UsdGeom, UsdShade

    stage, gone = omni.usd.get_context().get_stage(), []

    def rm(p: str) -> None:
        if stage.GetPrimAtPath(p).IsValid():
            stage.RemovePrim(p)
            gone.append(p)

    for r in extras.get("placed", []):
        if r.get("prim"):
            rm(r["prim"])
    for d in supports.get("discs", []):
        path = raw_env.scene[d["slot"]].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        prim = stage.GetPrimAtPath(path.replace("env_.*", "env_0"))
        UsdGeom.Imageable(prim).MakeInvisible()
        UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
        rm(f"/World/AugMaterials/look_support_{prim.GetName()}")
    for k in range(int(supports.get("slots", 0))):
        rm(f"{SUP.ROOT}/pillar_{k}")
        rm(f"/World/AugMaterials/look_support_pillar_{k}")
    gone += PM.clear(stage)
    for root in (EXV.ROOT, SUP.ROOT):
        if stage.GetPrimAtPath(root).IsValid() and not list(stage.GetPrimAtPath(root).GetChildren()):
            rm(root)
    return gone


# prims the look layer changes
LOOK_OWNED = ("/World/AugVis", "/World/AugMaterials/augvis_", "/World/AugMaterials/ph_")


def _traj_owned(ps: set) -> set:
    """The set the prim oracle measures: without the look-layer prims (LOOK_OWNED; they change per look; owner
    looks.augvis.scene)."""
    return {p for p in ps if not p.startswith(LOOK_OWNED)}


def restore(raw_env, T: dict, before: set) -> dict:
    """Scene prim oracle: after removing trajectory T's supports and extras, the prim set (look layer excluded) must
    equal the first trajectory's set before supports and extras (before)."""
    gone = _clear(raw_env, T["extras"], T["supports"])
    now = _traj_owned(prims())
    before = _traj_owned(before)
    extra, missing = sorted(now - before), sorted(before - now)
    if extra or missing:
        raise SystemExit(
            f"{MARK} scene prim oracle: after removing {T['clip']} left {len(extra)} {extra[:6]} · gone "
            f"{len(missing)} {missing[:6]}"
        )
    print(
        f"{MARK} scene prim oracle passed · {T['clip']} · removed {len(gone)} · prims {len(now)} = first trajectory "
        "before",
        flush=True,
    )
    return {"clip": T["clip"], "removed": gone, "prims": len(now), "ok": True}


def traj_row(T: dict, tm: dict, role) -> dict:
    """One row of render_clip.json trajs: the loaded trajectory's setup time, camera check, supports, extras."""
    return {
        "name": T["clip"],
        "role": role,
        "prep_s": round(T["prep"], 2),
        "load_s": {k: round(v, 2) for k, v in tm.items() if v},
        "view": T["view"],
        "body_reused": bool(T["body"].get("visual", {}).get("reused")),
        "extras_placed": len(T["extras"].get("placed", [])),
        "supports_visible": T["supports"].get("visible", 0),
    }
