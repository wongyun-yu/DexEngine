"""Collision curriculum: cross-hand and cross-object pairs authored off at start and switched on during training."""

from __future__ import annotations

from pxr import Usd, UsdPhysics

from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.scene.collision_bodies import (
    _object_cfg_fields,
    author_filtered_pairs,
    classify_body,
    object_prim_paths,
    object_rigid_bodies,
)

#: Returned by a split-out step that did not return early (its caller then goes on).
_CONTINUE = object()


def author_collcurr_cross_pair_exclusions(
    scene,
    env_root: str,
    include_objpair: bool = False,
    objpair_only: bool = False,
    shared_object: bool = False,
) -> dict[str, list[str]]:
    """COLLISION CURRICULUM start-OFF state (flag-gated ``--collision_curriculum 1``).

    Authors ``FilteredPairsAPI`` exclusions on env_0 (BEFORE cloning) for the three
    CROSS-pair classes:
      (1) right-hand links (incl. link7) <-> left-hand links
      (2) right-hand links <-> LEFT object
      (3) left-hand links  <-> RIGHT object
    FULL ISOLATION (``include_objpair=True``, flag ``--collision_curriculum_objpair 1``)
    adds the last remaining cross channel:
      (4) RIGHT object <-> LEFT object (rigid<->rigid — runtime toggling verified by
          a feasibility study)
    so early training decomposes into two independent uni-equivalent problems.
    Each class lives on DEDICATED source prims (RH<->LH sourced on the LH bodies;
    object pairs sourced on the object bodies) so the runtime flip removes exactly
    these rel targets and never touches the permanent inner-hand/inner-arm
    exclusions (authored enumeration-order-keyed by
    ``author_cross_hand_only_self_collision``). KEPT ON: each hand <-> its own
    object, everything <-> desk/ground, all existing uni behavior.

    This is the ONE live mechanism for cross-side OBJECT isolation (B5b); classes
    (2) and (3) are exactly what the deleted ``author_cross_side_object_isolation``
    claimed to do, and (4) adds its object<->object channel.

    Args:
        scene: the ``InteractiveScene`` being built. Both the stage and the object
            prim paths come from it: the cfg it was constructed from is the
            authority that CREATES those prims, so no object prim name is spelled
            in this module. See ``object_prim_paths``.
        env_root: the env_0 prim path to author on.
        include_objpair: also filter (4) RIGHT object <-> LEFT object.

    Returns:
        ``{source_prim_path: [added_target_paths]}`` — the EXACT rel edits to undo
        at the flip (``BimanualSelfCollisionScene.collcurr_enable_cross_pairs``:
        env_0 TEMPLATE edit propagates to ALL clones live, no sim re-init).
    """
    stage = scene.stage
    robot_root = stage.GetPrimAtPath(f"{env_root}/Robot")
    rh_bodies, lh_bodies = [], []
    for p in Usd.PrimRange(robot_root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            g = classify_body(p.GetPath().pathString.rsplit("/", 1)[-1])
            if g == "RH":
                rh_bodies.append(p.GetPath().pathString)
            elif g == "LH":
                lh_bodies.append(p.GetPath().pathString)

    return _object_prim_paths(
        scene, env_root, include_objpair, objpair_only, shared_object, lh_bodies, rh_bodies, stage
    )


def _object_prim_paths(scene, env_root, include_objpair, objpair_only, shared_object, lh_bodies, rh_bodies, stage):
    """Object prim paths come from the cfg that spawns them — never from a name table in this file. Printed below so a
    run's log carries the exact paths the exclusions were authored against.
    """
    obj_paths = object_prim_paths(scene, env_root)

    objr_bodies = object_rigid_bodies(stage, obj_paths, "right")
    objl_bodies = object_rigid_bodies(stage, obj_paths, "left")
    _ret = _shared_object_pairs(
        scene,
        env_root,
        include_objpair,
        objpair_only,
        shared_object,
        lh_bodies,
        obj_paths,
        objl_bodies,
        objr_bodies,
        rh_bodies,
    )
    if _ret is not _CONTINUE:
        return _ret

    pairs: dict[str, list[str]] = {}

    def _exclude(src_paths, target_paths):
        # Record ONLY targets we actually add, so the flip undoes exactly ours.
        # Delegates to the shared author_filtered_pairs primitive (same authoring
        # path as author_arm_obj_nocollision below) — accumulates into the outer
        # `pairs` dict this closure captures.
        added, n = author_filtered_pairs(stage, src_paths, target_paths)
        for s, ts in added.items():
            pairs.setdefault(s, []).extend(ts)
        return n

    # With the split articulation (1)(2)(3) are skipped entirely (see the objpair_only comment above).
    # Category choice: DEXENGINE_COLLCURR_CATS="hh,ho,oo" (default = as before)
    #   hh = (1) hand<->hand,  ho = (2)(3) hand<->other object,  oo = (4) object<->object.  Categories missing from
    #   the list get **no filter** (= collision ON). Training (collcurr), EVAL_CROSS_OFF and frozen all go through this
    #   function, so one variable moves all three. E.g. abl_ho = "hh,oo", abl_hoo = "hh".
    _cats = {c.strip() for c in env_flags.get("DEXENGINE_COLLCURR_CATS").split(",") if c.strip()}
    # With a single articulation and the self-collision flag OFF (canonical SELFCOLL=0), hand<->hand (hh) is **already
    # OFF**: pairs inside an articulation follow the flag and FilteredPairs adds nothing to the physics. Yet authoring
    # it in every env, 529 pairs/env × 4096 = 2.17M targets, failed a 12.9 GB PhysX GPU narrowphase allocation and gave
    # nan at step 0. Physically meaningless authoring is not done (the audit still judges R2 OFF through the flag).
    _hh_redundant = (not objpair_only) and env_flags.get("DEXENGINE_BI_SELFCOLL") != "1"
    if _hh_redundant and "hh" in _cats:
        print(
            "[COLLCURR] hh (hand<->hand) authoring skipped — already OFF through the single-articulation "
            "self-collision flag (PhysX GPU memory)",
            flush=True,
        )
    n_hh = (
        0 if (objpair_only or "hh" not in _cats or _hh_redundant) else _exclude(lh_bodies, rh_bodies)
    )  # (1) cross-hand, on LH
    # The source is the **link**, the target the object. With object prims as sources (as before) the PhysX GPU handles
    # dynamic-body source rels very expensively: 47 pairs/env filled 47 GB of GPU at 4096 envs (link-source desk with
    # 61 pairs/env = 20 GB).
    _ho_off = objpair_only or "ho" not in _cats or shared_object
    n_rl = 0 if _ho_off else _exclude(rh_bodies, objl_bodies)  # (2) RH <-> LEFT obj  (source = link)
    n_lr = 0 if _ho_off else _exclude(lh_bodies, objr_bodies)  # (3) LH <-> RIGHT obj (source = link)
    # (4) FULL ISOLATION: objR <-> objL, sourced on objL bodies (recorded like the
    # rest, so the same flip re-enables it). Shared object: same prim — nothing to isolate.
    n_oo = _exclude(objl_bodies, objr_bodies) if (include_objpair and "oo" in _cats and not shared_object) else 0
    _oo_seg = f", objRxobjL={n_oo}" if include_objpair else ""
    print(
        ("[COLLCURR] ★split articulation (objpair only) — " if objpair_only else "")
        + f"cats={sorted(_cats)} "
        + f"start-OFF authored on {env_root}: "
        f"filtered pairs RHxLH={n_hh}, RHxobjL={n_rl}, LHxobjR={n_lr}{_oo_seg} "
        f"(bodies RH={len(rh_bodies)}, LH={len(lh_bodies)}, objR={len(objr_bodies)}, "
        f"objL={len(objl_bodies)}). Object prims (from scene cfg): "
        f"right={obj_paths.get('right', '<none>')} left={obj_paths.get('left', '<none>')}. "
        f"KEPT ON: hand<->own object, desk/ground, uni behavior."
    )
    return pairs


def _shared_object_pairs(
    scene,
    env_root,
    include_objpair,
    objpair_only,
    shared_object,
    lh_bodies,
    obj_paths,
    objl_bodies,
    objr_bodies,
    rh_bodies,
):
    """Shared object: the scene **deliberately has no** object_left; both hands handle the one object_right. Then (2)
    RH<->objL has no target, (3) LH<->objR is **the left hand's own object**, not a "cross" pair, and must not be
    turned off, and (4) objR<->objL is the same prim. Only when the caller stated shared_object=True are these three
    skipped; an empty objL without the statement fails as before (a build error is never covered). A statement that
    disagrees with the evidence (no object_left in the cfg) fails.
    """
    if shared_object:
        if "left" in obj_paths:
            raise RuntimeError(
                f"[COLLCURR] shared_object=True but scene.cfg has object_left ({obj_paths['left']}) — "
                "_apply_shared_object did not remove the left prim. Statement and scene disagree."
            )
        print(
            f"[COLLCURR] {env_root}: SHARED OBJECT — no ho (hand<->other object) · oo (object<->object) authoring "
            f"(the left hand's object = objR, {len(objr_bodies)} bodies; not a cross pair). hh only.",
            flush=True,
        )
    # Split articulation (objpair_only=True): hand·arm crossings are already **ordinary rigid collisions between
    # articulations**, so there is nothing to author, and the left links are not under /Robot in the first place
    # (-> LH=0). Filters targeting articulation links also never resolve on instanced colliders, so authoring them
    # would do nothing. Only object<->object remains (rigid pairs, proven to work).
    if objpair_only:
        if not include_objpair or shared_object:
            print(
                f"[COLLCURR] {env_root}: split articulation path — nothing authored "
                f"(cross hand·arm are inter-articulation collisions, no filter needed; "
                + ("shared object, so no objpair either" if shared_object else "objpair not requested")
                + ")."
            )
            return {}
        if not objr_bodies or not objl_bodies:
            raise RuntimeError(
                f"[COLLCURR] objpair_only but no object bodies found under {env_root}: "
                f"objR={len(objr_bodies)} objL={len(objl_bodies)} (obj prims: {obj_paths})"
            )
    elif not rh_bodies or not lh_bodies or not objr_bodies or (not objl_bodies and not shared_object):
        raise RuntimeError(
            f"[COLLCURR] enumeration failed under {env_root}: RH={len(rh_bodies)} "
            f"LH={len(lh_bodies)} objR={len(objr_bodies)} objL={len(objl_bodies)} — "
            f"collision curriculum needs all four groups. Object prims derived from "
            f"the scene cfg: {obj_paths or '<none declared>'} (cfg fields carrying a "
            f"prim_path: {_object_cfg_fields(scene)})."
        )
    return _CONTINUE


def _collcurr_remap(path: str, i: int) -> str:
    return path if i == 0 else path.replace("/env_0/", f"/env_{i}/")


def collcurr_replicate_physics(scene) -> bool:
    """`scene.cfg.replicate_physics` (True when missing = the IsaacLab default)."""
    return bool(getattr(getattr(scene, "cfg", None), "replicate_physics", True))


def authoring_roots(scene) -> list[str]:
    """Env roots to author collision filters on: single owner of which envs have real prims at the authoring point
    (collision_authoring, right after the spawners, before parsing). With `replicate_physics=True` IsaacLab builds only
    env_0 and clones the rest (interactive_scene: `_add_entities_from_cfg` -> `clone_environments`) and PhysX replicates
    env_0's physics, so **only the env_0 template** (it propagates); with False the spawners make real prims in every
    env, so **all of them**."""
    roots = list(getattr(scene, "env_prim_paths", []) or [])
    return roots[:1] if collcurr_replicate_physics(scene) else roots


def collcurr_author_envs(
    scene, include_objpair: bool = False, objpair_only: bool = False, shared_object: bool = False
) -> dict[str, list[str]]:
    """Author the cross-pair start-OFF in **every env**.

    The target envs used to be chosen by `DEXENGINE_COLLCURR_ENVS` (default = env_0 only). With reference clones
    (`replicate_physics=True`) the env_0 template propagated to the clones, but the canonical multi-object setup uses
    `replicate_physics=False`, where nothing propagates: 4095 of 4096 ABL training envs then ran physics different from
    the preset (cross collision ON; measured filtered_targets=[637,0,0,0]).
    One rule: **a collision the preset turns off is off in every env.** The subset option is gone.

    The record (`_collcurr_pairs`) holds env_0 paths: the runtime switch (`_collcurr_edit`) works by substituting
    `/env_0/` -> `/env_i/`. env_1.. are authored here with the same substitution.
    With `replicate_physics=True` the clone prims do not exist, so only the env_0 template is authored.
    """
    roots = authoring_roots(scene)
    base = author_collcurr_cross_pair_exclusions(
        scene, roots[0], include_objpair=include_objpair, objpair_only=objpair_only, shared_object=shared_object
    )
    scene._collcurr_pairs = base
    n0 = sum(len(v) for v in base.values())
    if len(roots) <= 1:
        scene._collcurr_env_ids = [0]
        print(
            f"[COLLCURR] cross-pair start-OFF authored: env_0 template {n0} targets "
            f"(replicate_physics=True -> propagates to the clones)",
            flush=True,
        )
        return base
    n_rel, n_env = _collcurr_edit(scene, base, range(1, len(roots)), add=True)
    if n_env != len(roots) - 1 or n_rel != n0 * (len(roots) - 1):
        raise RuntimeError(
            f"[COLLCURR] cross-pair replication incomplete: env {n_env}/{len(roots) - 1}, "
            f"targets {n_rel}/{n0 * (len(roots) - 1)} — env_k prim paths differ from the `/env_0/`->`/env_k/` "
            f"substitution of env_0 (substitution contract broken). Training like this would learn "
            f"different physics per env."
        )
    scene._collcurr_env_ids = list(range(len(roots)))
    print(
        f"[COLLCURR] cross-pair start-OFF authored: env_0 {n0} targets + env_1..{len(roots) - 1} replicated "
        f"{n_rel} targets/{n_env} envs (replicate_physics=False, the same physics in every env)",
        flush=True,
    )
    return base


def collcurr_authored_env_ids(scene) -> list[int]:
    """Env indices really authored. `[0]` without a record (the old path)."""
    ids = getattr(scene, "_collcurr_env_ids", None)
    return list(ids) if ids else [0]


def collcurr_flip_env_ids(scene) -> list[int]:
    """Env indices the runtime switch **must edit**.

    With reference clones (`replicate_physics=True`) editing the env_0 template switches globally, and editing clone
    prims does nothing (reason (2) above). With real replication (`False`) every authored env must be edited
    (reason (3)).
    """
    if collcurr_replicate_physics(scene):
        return [0]
    return collcurr_authored_env_ids(scene)


def _collcurr_edit(scene, pairs: dict[str, list[str]], env_ids, add: bool) -> tuple[int, int]:
    """Add/remove `pairs` (env_0 paths) in each env of `env_ids`.

    Returns: (edited rel targets, envs really touched): the **result**, not the intent.
    """
    from pxr import Sdf

    n_rel = 0
    n_env = 0
    with Sdf.ChangeBlock():
        for i in env_ids:
            touched = False
            for src, targets in pairs.items():
                prim = scene.stage.GetPrimAtPath(_collcurr_remap(src, i))
                if not prim.IsValid():
                    continue
                api = UsdPhysics.FilteredPairsAPI.Apply(prim) if add else UsdPhysics.FilteredPairsAPI(prim)
                rel = api.GetFilteredPairsRel()
                if add and not rel:
                    rel = api.CreateFilteredPairsRel()
                if not rel:
                    continue
                for t in targets:
                    tt = _collcurr_remap(t, i)
                    if add:
                        rel.AddTarget(tt)
                    else:
                        rel.RemoveTarget(tt)
                    n_rel += 1
                    touched = True
            if touched:
                n_env += 1
    return n_rel, n_env


# ── Runtime cross-pair switches: the contract of the physical effect (read before changing) ──────────────
# They are module functions, not scene methods: the canonical bi (DEXENGINE_BI_SELFCOLL=0) is never swapped to
# BimanualSelfCollisionScene, and a `hasattr` gate on the scene method silently did nothing there. The authoring
# record (`scene._collcurr_pairs`) is attached by the env POSTINIT block whatever the scene class.
# Measured in a feasibility study (Isaac Sim 5.1.0 / PhysX 107.3.26, GPU pipeline):
#   (1) editing the rels of `UsdPhysics.FilteredPairsAPI` during playback changes the contact behaviour **at once**;
#       omni.physx has no Python API for pair filters, and `UsdNoticeListener` reading pure USD authoring live is
#       the only mechanism.
#   (2) with **reference clones** (`replicate_physics=True`, `copy_from_source=False`) RemoveTarget on a clone prim's
#       pre-clone filter does **nothing** (the composition is empty, PhysX does not change); removing it on the env_0
#       **template** prim flips every env, untouched env_2/3 included = a global switch.
#   (3) so the edit works **on the prim that owns the local spec**. With reference clones that is env_0 alone; with
#       `replicate_physics=False` (canonical multi-clip) the spawners make **real prims** per env and each env_k owns
#       its spec, so edits must be **per env** (editing env_0 only reaches env_0).
# So these functions read `replicate_physics` to pick the lever: True -> edit the env_0 template once (global);
# False -> edit every authored env (`/env_0/` -> `/env_i/` substitution, the same contract as the ramp). They print
# the **result**, not the intent: edited rel targets and **envs really touched**.


def collcurr_enable_cross_pairs(scene) -> int:
    """One-off switch: **remove** the cross-pair filters to turn collision on (= the same physics as strict).

    Only the rel targets recorded at authoring are removed, so permanent exclusions (hand internal, arm internal) stay.
    Idempotent: the record is consumed by the first call. Call between physics steps.

    Returns: removed rel targets (0 = nothing to remove).
    """
    pairs = getattr(scene, "_collcurr_pairs", None)
    if not pairs:
        return 0
    scene._collcurr_pairs = None
    env_ids = collcurr_flip_env_ids(scene)
    n_rel, n_env = _collcurr_edit(scene, pairs, env_ids, add=False)
    print(
        f"[COLLCURR] one-off switch: rels removed {n_rel} · envs edited {n_env}/"
        f"{len(getattr(scene, 'env_prim_paths', []) or [])} "
        f"(replicate_physics={collcurr_replicate_physics(scene)} → "
        + (
            "env_0 template edit = propagated to every clone"
            if len(env_ids) == 1 and collcurr_replicate_physics(scene)
            else "per-env edit"
        )
        + ")",
        flush=True,
    )
    return n_rel


def collcurr_enable_fraction(scene, frac: float) -> tuple[int, int]:
    """Remove the cross-pair filters only in **the leading frac share of envs** (cumulative, monotone).

    Args:
        frac: 0.0~1.0, the share of envs to turn on. Smaller than the previous call does nothing
              (monotone increase only: turning on and off would make the reward abnormal).
    Returns:
        (envs turned on this time, envs turned on so far)
    """
    pairs = getattr(scene, "_collcurr_pairs_perenv", None)
    if pairs is None:
        base = getattr(scene, "_collcurr_pairs", None)
        if not base:
            return (0, getattr(scene, "_collcurr_on_n", 0))
        scene._collcurr_pairs_perenv = base
        scene._collcurr_pairs = None  # mutually exclusive with the one-off switch
        scene._collcurr_on_n = 0
        pairs = base
        # Existence check: without env_1's corresponding prim the clone is not a copy -> fail loudly.
        n_env = len(scene.env_prim_paths)
        if n_env > 1:
            probe = next(iter(base))
            p1 = probe.replace("/env_0/", "/env_1/")
            if not scene.stage.GetPrimAtPath(p1).IsValid():
                raise RuntimeError(
                    "[COLLCURR] the gradual switch needs per-env prims. "
                    f"{p1} is missing — the clones are USD instances sharing env_0. "
                    "clone_environments(copy_from_source=True) is needed, or "
                    "this combo can only use the one-off switch (--collision_curriculum_on_frac)."
                )
    n_env = len(scene.env_prim_paths)
    want = max(0, min(n_env, int(round(float(frac) * n_env))))
    have = getattr(scene, "_collcurr_on_n", 0)
    if want <= have:
        return (0, have)
    n_rel, n_touched = _collcurr_edit(scene, pairs, range(have, want), add=False)
    scene._collcurr_on_n = want
    print(
        f"[COLLCURR] gradual switch: env {have}->{want}/{n_env} "
        f"({100.0 * want / max(1, n_env):.0f}%) · rels removed {n_rel} · envs really edited {n_touched}"
    )
    return (want - have, want)


def collcurr_verify_envs(scene, n_check: int = 4) -> str:
    """One-line summary for the propagation check: counts FilteredPairs targets per env (authoring unchanged).

    The same computation as `_bsc_verify_filters`, run outside the scene class. Verdict: `envs_covered` 1 with a large
    `num_envs` means only env_0 has them.
    """
    roots = list(getattr(scene, "env_prim_paths", []) or [])
    n_env = len(roots)
    covered = 0
    counts = []
    for k in range(min(max(0, n_check), n_env)):
        # Walks the whole env root: cross-pair filters sit on both /Robot links **and** object prims (the sources of
        # objRxobjL, RHxobjL).
        prim = scene.stage.GetPrimAtPath(roots[k])
        if not prim.IsValid():
            counts.append(-1)
            continue
        n_tgt = 0
        for p in Usd.PrimRange(prim):
            if not p.HasAPI(UsdPhysics.FilteredPairsAPI):
                continue
            rel = UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel()
            if rel:
                n_tgt += len(list(rel.GetTargets()))
        counts.append(n_tgt)
        if n_tgt > 0:
            covered += 1
    return (
        f"[COLLCURR] verify envs_covered={covered}/{min(max(0, n_check), n_env)} "
        f"checked (num_envs={n_env}) filtered_targets={counts} "
        f"replicate_physics={collcurr_replicate_physics(scene)}"
    )
