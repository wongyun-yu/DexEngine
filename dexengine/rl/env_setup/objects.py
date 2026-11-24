"""Object entities of the scene: rigid and articulated object cfgs, the support constants and the per-object body
list."""

from __future__ import annotations

import json
import os

from dexengine.placement.coords import (
    core as _coords_hz,
)  # home of the empty-slot hidden z (imports without Isaac; the same value as the collection form)
from dexengine.rl.env_setup.object_mass import object_collision_props, object_rigid_props
from dexengine.rl.tasks import env_flags


def abspath_multi(p):
    """abspath each token of a possibly comma-separated (--multi_object) path list."""
    return ",".join(os.path.abspath(t) for t in str(p).split(",") if t)


#: Passive properties of articulated object joints (rule D1; this is the one owner). Object joints are not driven: the
#: physics only follows the hand opening and closing. Stiffness 0, a little damping for numerical stability, and an
#: effort limit as the cap on the force the hand hands over.
ARTIC_JOINT_DAMPING = 0.05
ARTIC_JOINT_EFFORT = 5.0
#: Joint velocity limit (rad/s, PhysX maxJointVelocity). In a canary, the hand-object penetration kick of RSI mid-frame
#: resets (also present on rigid objects, |v_root| 27–154 m/s) spun a free hinge to |ω| 622 rad/s and blew up PhysX
#: aggregate pairs (597M) and NaN at NE=1024. Measured reference joint speeds (736 arctic trees, frame difference ×
#: 50 Hz): max 44.3 rad/s (a ketchup spike) · p99 of the clip maxima 21.3 · median 3.9, so 60 is a safe cap that never
#: clips the reference (a guard, not a fix of the kick).
ARTIC_JOINT_VEL_LIMIT = 60.0


def build_object_entity_cfg(args_cli, ctx_label, side: str, usd_attr: str, init_pos=(0.0, 0.0, 0.78)):
    """**Asset cfg** of an object slot, rigid (RigidObjectCfg) or articulated (ArticulationCfg, rule D1): the single
    owner for training and eval.

    Articulation is decided by `args_cli.object_articulated_<side>` alone (set by the sequence_dir entry from the tree's
    `objects[].articulation`). Spawning an articulated object from a rigid USD makes the lid a fixture, and training
    learns it as 'a robot that cannot open it' (a silent wrong answer). An articulated object is two links,
    base_link + moving_link, and one joint (robots/objects/usd_bake.py --articulated); the reset
    (events.reset_object_to_trajectory) overwrites the initial joint value with the reference. Marker `[ARTIC][spawn]`.
    """
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg

    prim_path = "{ENV_REGEX_NS}/Object" + side.capitalize()
    artic = bool(int(getattr(args_cli, f"object_articulated_{side}", 0) or 0))
    if not artic:
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=build_object_spawn_cfg(args_cli, ctx_label, usd_attr=usd_attr),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(init_pos), rot=(1.0, 0.0, 0.0, 0.0)),
        )
    from dexengine.rl.tasks.mdp.artic_parts import require_one_kind

    usds = _object_usds(args_cli, usd_attr)
    kind = require_one_kind(usds, f"{ctx_label} object_{side}")
    spawn = build_object_spawn_cfg(args_cli, ctx_label, usd_attr=usd_attr, artic_kind=kind)
    print(
        f"[ARTIC][spawn] {ctx_label}: object_{side} = ArticulationCfg(base_link+moving_link, 1 {kind} joint) · "
        f"{len(usds)} USDs · "
        f"joint stiffness=0 damping={ARTIC_JOINT_DAMPING} effort={ARTIC_JOINT_EFFORT} "
        f"vel_limit={ARTIC_JOINT_VEL_LIMIT} rad/s · fix_root_link=False",
        flush=True,
    )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(init_pos), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        actuators={
            "artic": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=ARTIC_JOINT_DAMPING,
                effort_limit_sim=ARTIC_JOINT_EFFORT,
                velocity_limit_sim=ARTIC_JOINT_VEL_LIMIT,
            )
        },
    )


#: Disc slot count of the aerial scene (rule D20). Measured on adopted aerial trees: arctic max 3 · hot3d max 4. Too few
# slots fail.
AERIAL_DISC_SLOTS = 4
#: Pedestal slot cap (desk scene, coords.PEDESTAL_COLLIDE): one per disc slot plus one per fixed object (the extra slot
#: cap extra_slots.K_MAX; support objects one per main object). The real slot count is the bundle's maximum pedestal
#: count (_build_disc_slots).
PEDESTAL_SLOTS_MAX = AERIAL_DISC_SLOTS + 9 + 2
#   (stamped z = top = the floor of the supported object, the same meaning in the render; the centre transform is done
# by reset and tick)
AERIAL_HIDDEN_Z = (
    _coords_hz.SLOT_HIDDEN_Z
)  # hidden z of unused slots; value home coords (the collection form reads it too)


def build_all_obj_bodies(cfg, args_cli, ctx_label) -> None:
    """All-object collision (collision_presets.ALL_OBJ_ENV=1; default off = does nothing, byte-identical): one invisible
    kinematic body for each static object the render only shows (rows scene.extras.decide adds; the same owner as the
    collection render and the eval scene; visual only, render-only source). Collision shape = the training USD's convex
    pieces. The visual prims stay, so the picture does not change. Starts at the hidden pose; after the render entry's
    decisions all_obj_collision.place_all_obj_bodies writes the same pose as the visual prim. Marker =
    author_all_obj_collision."""
    from dexengine.rl.tasks.scene import collision_presets as _cp

    if not _cp.all_obj_collision():
        return

    from dexengine.placement.coords import core as _coords  # noqa: F401, I001 — import kept for its side effect
    import isaaclab.sim as _su
    from isaaclab.assets import RigidObjectCfg
    from dexengine.data.objects import hull as _oh  # owner of the training USD paths
    from dexengine.record.scene import extras as _exv  # owner of the added rows

    seqs = [t for t in (args_cli.sequence or "").split(",") if t]
    seq_dir = getattr(args_cli, "sequence_dir", None)
    if not seqs and seq_dir and env_flags.get("DEXENGINE_SEQ_LIST").strip() not in ("", "all"):
        with open(
            env_flags.get("DEXENGINE_SEQ_LIST").strip(), encoding="utf-8"
        ) as f:  # roster session (augmentation rollout)
            seqs = [ln.strip() for ln in f if ln.strip()]
    _stand_static_bodies(cfg, ctx_label, RigidObjectCfg, _cp, _exv, _oh, _su, seq_dir, seqs)


def _stand_static_bodies(cfg, ctx_label, RigidObjectCfg, _cp, _exv, _oh, _su, seq_dir, seqs):
    """Stand the static object bodies of every slot (one-clip or per-env branch) on the scene cfg."""
    if seq_dir and (len(seqs) > 1 or int(cfg.scene.num_envs) > 1):
        K, _EventTerm, _xc, _xe, per = _slot_rows_for_envs(cfg, _cp, _exv, _oh, seq_dir, seqs)
        if not K:
            print(
                f"{_cp.ALL_OBJ_TAG} static bodies stood 0 · clips {len(seqs)} (all with 0 added rows) · {ctx_label}",
                flush=True,
            )
            return
        rule = _xe.standin_rule()
        kin = _su.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        for k in range(K):
            stand = _xe.stand_in([[x["usd"] for x in r] for r in per], k, rule)
            _slot_entity(
                cfg,
                f"{_cp.ALL_OBJ_KEY}{k}",
                f"{_cp.ALL_OBJ_LEAF}{k}",
                [
                    _su.UsdFileCfg(
                        usd_path=r[k]["usd"] if k < len(r) else stand,
                        visible=False,
                        rigid_props=kin,
                        collision_props=_su.CollisionPropertiesCfg(collision_enabled=k < len(r)),
                    )
                    for r in per
                ],
            )
        cfg.all_obj_multi = {"clips": list(seqs), "rows": per, "K": K, "hidden_z": float(AERIAL_HIDDEN_Z)}
        if len(seqs) == 1:
            cfg.scene.replicate_physics = False
        cfg.events.reset_all_obj_bodies = _EventTerm(func=_xc.reset_all_obj_multi, mode="reset", params={})
        n = [len(r) for r in per]
        print(
            f"{_cp.ALL_OBJ_TAG} static bodies stood for {len(seqs)} clips · slots {K} · rows {sum(n)} (per clip "
            f"{min(n)}–{max(n)}) · "
            f"stand-in bodies {sum(K - x for x in n)} (hidden · collider off · rule {rule}) · env i -> clip i % "
            f"{len(seqs)} · "
            f"{ctx_label}",
            flush=True,
        )
        return
    if len(seqs) != 1 or not seq_dir:
        raise RuntimeError(f"{_cp.ALL_OBJ_TAG} {len(seqs)} clips · tree folder {seq_dir!r} — one-clip sessions only")
    tree = os.path.join(str(seq_dir), seqs[0])
    with open(os.path.join(tree, "task_info.json"), encoding="utf-8") as f:
        ti = json.load(f)
    put, _drop, _by = _exv.summary(_exv.decide(tree, ti)["rows"])
    rows = []
    for k, r in enumerate(put):
        usd = str(_oh.usd_file(str(ti["dataset_name"]), r["obj_id"], False))
        if not os.path.isfile(usd):
            raise RuntimeError(
                f"{_cp.ALL_OBJ_TAG} {r['name']}({r['obj_id']}) has no training USD {usd} — "
                "it cannot go into the physics"
            )
        setattr(
            cfg.scene,
            f"{_cp.ALL_OBJ_KEY}{k}",
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"{_cp.ALL_OBJ_LEAF}{k}",
                spawn=_su.UsdFileCfg(
                    usd_path=usd,
                    visible=False,
                    rigid_props=_su.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, AERIAL_HIDDEN_Z), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        )
        rows.append({"name": r["name"], "obj_id": r["obj_id"], "source": r["source"], "usd": usd})
    cfg.all_obj_bodies = rows
    print(f"{_cp.ALL_OBJ_TAG} static bodies stood {len(rows)} · {ctx_label}", flush=True)


def _slot_rows_for_envs(cfg, _cp, _exv, _oh, seq_dir, seqs):
    """The env count picks the branch: per-env slot rows for the one-clip branch."""
    # The env count picks the branch: the one-clip branch below takes one env only (the render entry writes the
    # pose; the author_all_obj_collision contract). One clip with several envs (an augmentation retry round rolling
    # one variant) also takes it, with replicate_physics off (authoring and measurement run per env).
    # Multi-clip session (augmentation rollout, v4): static bodies = the added rows per clip (scene.extras.decide,
    # the same owner as the render, the same scene pose). Slot k = clip c's k-th row: fixed slots (_slot_entity; env
    # i -> clip i % C, the loader order contract) stand that clip's objects per env. Places missing from a clip get
    # a stand-in body (extra_slots.stand_in; hidden, collider off). Pose = the reset event
    # all_obj_collision.reset_all_obj_multi (per env, per reset); authoring and measurement =
    # author_all_obj_collision (per env, that clip's rows).
    from isaaclab.managers import EventTermCfg as _EventTerm  # noqa: I001
    from dexengine.rl.tasks.scene import all_obj_collision as _xc
    from dexengine.rl.tasks.scene import extra_slots as _xe

    per = []
    for s_ in seqs:
        tree_ = os.path.join(str(seq_dir), s_)
        with open(os.path.join(tree_, "task_info.json"), encoding="utf-8") as f:
            ti_ = json.load(f)
        rows_ = []
        for r in _exv.summary(_exv.decide(tree_, ti_)["rows"])[0]:
            usd = str(_oh.usd_file(str(ti_["dataset_name"]), r["obj_id"], False))
            if not os.path.isfile(usd):
                raise RuntimeError(
                    f"{_cp.ALL_OBJ_TAG} {s_}: {r['name']}({r['obj_id']}) has no training USD {usd} — "
                    "it cannot go into the physics"
                )
            rows_.append(
                {
                    "name": r["name"],
                    "obj_id": r["obj_id"],
                    "source": r["source"],
                    "usd": usd,
                    "scene_xyz": [float(v) for v in r["scene_xyz"]],
                    "scene_wxyz": [float(v) for v in r["scene_wxyz"]],
                }
            )
        per.append(rows_)
    K = max(len(r) for r in per)
    cfg.all_obj_bodies = []
    return K, _EventTerm, _xc, _xe, per


def _slot_entity(cfg, name: str, prim_leaf: str, assets: list, contact: bool = False) -> None:
    """One fixed slot (shared by discs, support objects and extra objects): a rigid body with per-clip assets through
    MultiAssetSpawnerCfg (env_i -> asset[i % clip count]; loader clip order, the same contract as multi-object),
    starting
    at the hidden pose (the reset event writes the pose of that env's clip). contact = the contact report API (for
    contact sensors); the wrapping cfg's value overrides the asset cfg (IsaacLab spawn_multi_asset), so it is given
    here."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg

    setattr(
        cfg.scene,
        name,
        RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + prim_leaf,
            spawn=MultiAssetSpawnerCfg(assets_cfg=assets, random_choice=False, activate_contact_sensors=contact),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, AERIAL_HIDDEN_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        ),
    )


def build_start_held(cfg, args_cli, ctx_label) -> None:
    """Exclusion statement of the start-held probe (hands that start holding their own object are left out
    of the count): per clip (loader order = sorted) {side: did that hand start holding its own main object}
    (coords.start_held_hands; canonical contact sidecar, start window) -> `cfg.start_held_hands`. The only consumer is
    the probe's reset wrist gate (rl.judge.park.Park), so it is built for probes only (other launches do not lean on the
    sidecar layer; a probe without the statement makes Park fail). Marker `[PARK_PROBE] … start-held statement`."""
    from dexengine.rl.judge.park import probe_steps

    if not probe_steps():
        return
    from dexengine.placement.coords import core as _coords  # owner of the 'held' definition

    seqs, seq_dir = [t for t in (args_cli.sequence or "").split(",") if t], getattr(args_cli, "sequence_dir", None)
    if seqs != sorted(seqs) or not seq_dir:
        raise RuntimeError(
            f"[PARK_PROBE] start-held statement: --sequence is not sorted ({seqs[:2]}) or the tree folder"
            f" (args_cli.sequence_dir={seq_dir!r}) is unknown — clip order contract, cannot read"
        )
    held = []
    for q in seqs:
        with open(os.path.join(str(seq_dir), q, "task_info.json"), encoding="utf-8") as f:
            held.append(_coords.start_held_hands(os.path.join(str(seq_dir), q), json.load(f)))
    cfg.start_held_hands = held
    print(
        f"[PARK_PROBE] {ctx_label}: start-held statement (coords.start_held_hands) clips {len(held)} · hands that "
        f"start "
        f"holding their own object R {sum(h['right'] for h in held)} · L {sum(h['left'] for h in held)} — the reset "
        f"wrist gate leaves them out",
        flush=True,
    )


def _object_usds(args_cli, usd_attr: str) -> list[str]:
    """USD list of an object slot (absolute paths, loader clip order): the only interpretation, read by both the spawn
    cfg and the joint kind guard."""
    return [os.path.abspath(t) for t in str(getattr(args_cli, usd_attr, None) or "").split(",") if t]


def _render_object_usds(usds: list[str], label: str) -> list[str]:
    """Object USDs read by the spawn only: in a render context (robots.selection.render_context) the texture overlays
    (artifact_paths.render_spawn_usds; the physics composes the training USD as a sublayer, so it is the same),
    otherwise
    the training USDs as they are (byte-identical). Guards and fingerprints look at the training USDs."""
    from dexengine.robots.selection import render_context

    if not render_context():
        return usds
    from dexengine.core import artifact_paths as _tp

    out, fb = _tp.render_spawn_usds(usds)
    root = _tp.overlay_root()
    print(
        f"[RENDER_OBJ_USD] {label}: overlays {len(usds) - len(fb)}/{len(usds)} · training USD fallback {len(fb)}"
        + (f" · overlay root {root}" if root else "")
        + (
            f" (no overlay: {', '.join(os.path.basename(os.path.dirname(u)) for u in fb[:8])}"
            f"{' …' if len(fb) > 8 else ''} — bake dexengine/robots/objects/render_overlay.py)"
            if fb
            else ""
        ),
        flush=True,
    )
    # Measured marker (texture o/x): shells without textures among the spawned overlays, with the root (canonical |
    # DEXENGINE_RENDER_OVERLAY_ROOT)
    sh = _tp.overlay_shells(out)
    ids = ", ".join(os.path.basename(os.path.dirname(p)) for p in sh[:8])
    print(
        f"[TEX_OVERLAY] {label}: root={root or 'canonical'} shells={len(sh)}/{len(out)}" + (f" ({ids})" if sh else ""),
        flush=True,
    )
    # Recurrence gate: the measured md5 of overlays spawned from the canonical root are compared with the canonical
    # overlay manifest; a difference fails (a stale copy, or a new bake without a manifest update). A non-canonical root
    # (texture x, ...) is skipped and left in the marker. Rule and manifest owner =
    # artifact_paths.overlay_manifest_check.
    print(f"[OVERLAY_MANIFEST] {label}: {_tp.overlay_manifest_check(out, root)}", flush=True)
    return out


def build_object_spawn_cfg(args_cli, ctx_label, usd_attr="object_usd", artic_kind: str | None = None):
    """Spawn cfg for the main 'object' scene entity (train + eval single place).

    `artic_kind` = the joint kind of an articulated slot (hinge | slide, read by build_object_entity_cfg from the slot
    USDs) · None = rigid. An articulated object's self-collision (parts within the same articulation) follows that
    kind's statement artifact_paths.artic_self_collisions; marker `[ARTIC_SELFCOLL]`.

    Single-object (default): UsdFileCfg — byte-identical to the previous inline blocks.
    --multi_object 1: deterministic MultiAssetSpawnerCfg — one USD per clip in LOADER
    order; random_choice=False maps env_i -> assets[i % n], matching the trajectory
    manager's static env->clip assignment (clip i % n over the same sorted order).

    Props (all objects): contact sensors ON (object-filtered force for the contact
    termination); gravity disabled at spawn (ramp_object_gravity event applies it);
    mass_props NOT set here — apply_object_mass() overrides to the single-source
    0.2 kg default later; friction set at runtime by randomize_object_friction.
    """
    import isaaclab.sim as _su

    from dexengine.rl.tasks.cfg.constants import MAX_DEPENETRATION_VELOCITY as _mdv

    rigid = object_rigid_props(_su)
    print(
        f"[SPAWN_PHYS] {ctx_label}: object max_depenetration_velocity={_mdv:g} m/s "
        "(owner cfg.constants.MAX_DEPENETRATION_VELOCITY; the robot spawn gets the same value)",
        flush=True,
    )
    coll = object_collision_props(_su)
    usds = _object_usds(args_cli, usd_attr)
    for u in usds:
        if not os.path.exists(u):
            raise FileNotFoundError(
                f"active object USD path does not exist: {u} (--object_usd_right / --object_usd_left)"
            )
    # usds = the one interpretation of the slot USD list (the same as the kind guard and fingerprints) · spawn_usds =
    # its
    # spawn version (overlays in a render context); one name never carries two meanings.
    spawn_usds = _render_object_usds(usds, f"{ctx_label} {usd_attr}")
    # Articulated (rule D1): an articulated USD has the ArticulationRootAPI and is a free object, so
    # fix_root_link=False.
    # The rigid path without this argument stays byte-identical.
    articulated = artic_kind is not None
    artic_props = None
    if articulated:
        from dexengine.core import artifact_paths as _tp

        _self_coll = _tp.artic_self_collisions(artic_kind)
        print(
            f"[ARTIC_SELFCOLL] {ctx_label} {usd_attr}: kind={artic_kind} {'on' if _self_coll else 'off'} "
            f"n={len(usds)} (parts within the same articulation · env spawn enabled_self_collisions={_self_coll})",
            flush=True,
        )
        artic_props = _su.ArticulationRootPropertiesCfg(
            articulation_enabled=True, fix_root_link=False, enabled_self_collisions=_self_coll
        )
    _usd_kw = {"activate_contact_sensors": True, "rigid_props": rigid, "collision_props": coll}
    if articulated:
        _usd_kw["articulation_props"] = artic_props
    if getattr(args_cli, "multi_object", 0) and len(usds) > 1:
        from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg

        print(
            f"[{ctx_label}] multi-object spawn: {len(usds)} USDs, env_i -> asset[i % {len(usds)}]"
            + (" [ARTIC]" if articulated else "")
        )
        return MultiAssetSpawnerCfg(
            assets_cfg=[
                _su.UsdFileCfg(usd_path=u, **({"articulation_props": artic_props} if articulated else {}))
                for u in spawn_usds
            ],
            random_choice=False,  # deterministic cycle == loader clip order
            activate_contact_sensors=True,
            rigid_props=rigid,
            collision_props=coll,
        )
    return _su.UsdFileCfg(usd_path=spawn_usds[0], **_usd_kw)
