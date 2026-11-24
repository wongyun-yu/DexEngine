"""Legacy bimanual self-collision scene: articulation self-collision on, every self pair filtered except cross-hand (and
arm/hand <-> torso).


The robot is ONE articulation holding both arms + both hands + head. Its
articulation self-collision is OFF by default (``enabled_self_collisions=False``
in the combo articulation cfg), so NO intra-articulation pair collides — including
the left hand vs the right hand. For physically-correct bimanual manipulation the
two hands MUST be able to collide (a flailing policy should not pass one hand
through the other).

Naively flipping ``enabled_self_collisions=True`` is HARMFUL: it switches on the
FULL self-collision matrix — every inner-hand (finger/palm hull) pair and every
inner-arm (link) pair — which the current convention deliberately keeps OFF and
which would produce spurious depenetration explosions at rest/grasp. The current
collision convention (no inner-hand, no inner-arm self-collision) MUST be
preserved.

So we enable self-collision and then author ``UsdPhysics.FilteredPairsAPI`` to
EXCLUDE every self-pair EXCEPT right-hand-body <-> left-hand-body. Net effect:
  - left_hand <-> right_hand : COLLIDE (the one wanted addition)
  - inner-hand (right<->right, left<->left)        : excluded (unchanged = OFF)
  - inner-arm  (within either arm)                 : excluded (unchanged = OFF)
  - arm <-> same/other hand, arm <-> arm, head/base: excluded (unchanged = OFF)

"Hand" scope: the palm/wrist mount (the ARM's ``ee_body_name(side)``,
which carries the palm hulls after the importer fixed-merges them) PLUS every body
the **ACTIVE** hand's own spec declares -- ``HandSpec.contact_body_regex_right``
and its palm frames, mirrored per side. Everything else (the arm links, head,
base, ...) is "OTHER" and collides with nothing intra-articulation. The scope is
RESOLVED FROM THE SPECS, never spelled here, and only from the spec of the hand
THIS robot carries; see the note above ``classify_body``.

TECHNIQUE / LIFECYCLE — author
``FilteredPairsAPI`` on the env_0 TEMPLATE robot-link prims BEFORE the cloner
replicates env_0 (``replicate_physics=True`` parses env_0 physics once, then the
replicator remaps each relationship target into every clone's namespace). We
subclass ``InteractiveScene`` and override ``clone_environments`` to author the
exclusions first, then call ``super().clone_environments(...)``. The env class
swaps in this subclass only when bimanual (single-hand path never constructs it).
FilteredPairsAPI is symmetric, so each excluded pair is authored once.
"""

from __future__ import annotations

from collections import defaultdict

from pxr import Usd, UsdPhysics

from isaaclab.scene import InteractiveScene

from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.scene.collision_bodies import (
    _SIDE_GROUP,
    _SIDES,
    _active_arm_name,
    _active_hand_name,
    classify_bodies,
)
from dexengine.rl.tasks.scene.collision_curriculum import (
    author_collcurr_cross_pair_exclusions,
    collcurr_enable_cross_pairs,
    collcurr_enable_fraction,
)
from dexengine.rl.tasks.scene.collision_pairs import author_robot_desk_nocollision

# Arm/hand <-> torso pairs stay colliding in the bimanual self-collision filter (no pass-through).
_ARM_TORSO_COLL = True


def author_cross_hand_only_self_collision(stage, env_root: str, quiet: bool = False) -> None:
    """Author FilteredPairsAPI on env_0 so only right-hand<->left-hand self-pairs collide.

    Enumerates the robot's rigid-body link prims under ``{env_root}/Robot``,
    classifies each, and excludes every self-pair except (RH, LH).

    Args:
        stage: the USD stage.
        env_root: the env prim path, e.g. "/World/envs/env_0".
        quiet: suppress the per-env census line. The per-env fix (DEXENGINE_CROSS_FILTER_PERENV=1) calls this
            function for every env, and 2048 envs × one summary line would make the log useless. env_0 always
            stays loud (quiet=False), as in the old log.
    """
    robot_root = f"{env_root}/Robot"
    root_prim = stage.GetPrimAtPath(robot_root)
    if not root_prim.IsValid():
        print(f"[BimanualSelfCollision] WARNING: robot prim not found, skipping: {robot_root}")
        return

    body_paths, census, excl, n_excl, n_keep, n_keep_lt, n_lh, n_other, n_rh = _author_selfcoll_filters(
        robot_root, root_prim
    )

    # ── authoring ──
    for src, targets in excl.items():
        prim = stage.GetPrimAtPath(src)
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = api.GetFilteredPairsRel()
        if not rel:
            rel = api.CreateFilteredPairsRel()
        existing = set(rel.GetTargets())
        for t in targets:
            if t not in existing:
                rel.AddTarget(t)

    if quiet:
        return
    _tipmsg = ", ".join(f"{s}={len(census[s][3])}/{len(census[s][1])} ({census[s][0]})" for s in _SIDES)
    print(
        f"[BimanualSelfCollision] env_root={env_root}: bodies={len(body_paths)} "
        f"(RH={n_rh}, LH={n_lh}, OTHER={n_other}); fingertips in the collision set: "
        f"{_tipmsg}; KEPT cross-hand pairs={n_keep}; "
        f"EXCLUDED self-pairs={n_excl}"
        + (
            f" (of which arm·hand<->torso KEPT={n_keep_lt}, arm-torso ON)"
            if _ARM_TORSO_COLL
            else " (arm<->torso excluded too, arm-torso OFF)"
        )
        + ". Inner-hand + inner-arm collision OFF "
        "(preserves current convention); ONLY right-hand<->left-hand collides."
    )


def _author_selfcoll_filters(robot_root, root_prim):
    """Authoring location axis (DEXENGINE_SELFCOLL_FILTER_ON_SHAPES). Why this axis exists: a measurement
    established that "the filters never reach PhysX" (the same CLEAN clip on env_0~7 replayed with
    SC1NULL -> **the same blow-up in every env**; the same clip with SELFCOLL=0 is fine). The one remaining question
    is then **"which prim must carry the authoring for PhysX to read it"**. This asset's measured structure narrows
    the candidates: * 61 links = `Xform + RigidBodyAPI`, **no CollisionAPI at all** * all 72 real colliders are
    `link_X/collisions/<mesh>/<shape>`, and `link_X/collisions` is **instanceable** (a prototype reference), so the
    colliders are **instance proxies**; API schemas cannot be applied to proxies (`UsdPhysics.FilteredPairsAPI.Apply`
    only warns "Unable to create primSpec" and **silently returns an invalid schema**, measured). So the old code
    could only author on **body (link) prims**, and unless PhysX expands that into shape pairs inside the
    articulation, every filter does nothing.  Collect rigid-body link prims (the articulation links carry
    RigidBodyAPI).
    """
    body_paths = [
        prim.GetPath().pathString for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]

    body_names = [p.rsplit("/", 1)[-1] for p in body_paths]
    by_name, census = classify_bodies(body_names)
    groups = {p: by_name[n] for p, n in zip(body_paths, body_names, strict=True)}
    n_rh = sum(1 for g in groups.values() if g == "RH")
    n_lh = sum(1 for g in groups.values() if g == "LH")
    n_other = sum(1 for g in groups.values() if g == "OTHER")

    bad = _hand_classification_problems(census, n_lh, n_rh)
    if bad:
        raise RuntimeError(
            f"[BimanualSelfCollision] classification failed under {robot_root}: "
            + "; ".join(bad)
            + f". Counts: RH={n_rh} LH={n_lh} OTHER={n_other} over "
            f"{len(body_paths)} bodies. Active selection: arm={_active_arm_name()!r} "
            f"hand={_active_hand_name()!r}. Body names: {body_names}. "
            "Names are resolved from HandSpec/ArmSpec — if a hand was just "
            "ported, fix its contact_body_regex_right / tip_body_names_right rather "
            "than adding a literal here."
        )

    # Author exclusions for every unordered pair that is NOT (RH,LH)/(LH,RH).
    excl = defaultdict(list)
    n_keep = 0
    n_keep_lt = 0
    n_excl = 0
    for i in range(len(body_paths)):
        gi = groups[body_paths[i]]
        for j in range(i + 1, len(body_paths)):
            gj = groups[body_paths[j]]
            _p = {gi, gj}
            is_cross_hand = _p == {"RH", "LH"}
            # ── **the arms must not pass through the torso** ─────────────────────
            # It breaks sim-to-real. Until now articulation self-collision was off entirely
            # (enabled_self_collisions=False in the combo articulation), so the arms passed through the torso. The
            # bimanual build turned it on and **excluded everything again except RH<->LH**, so arm<->torso was
            # excluded too.
            #   Unimanual is not touched (its runs stay as trained).
            #   Default OFF = no change. The bimanual launcher turns it on.
            # Pairs turned on: (arm|hand) <-> torso/base/head. **Adjacent (jointed) pairs are excluded automatically
            # by PhysX inside an articulation**, so parent-child pairs such as arm_0<->torso pass as before; that is
            # the rule "contact with each one's parent is fine".
            # **Not** turned on (kept as before): hand internal · same arm internal · arm<->own hand
            #   (combinations that blow up with false depenetration while grasping)
            is_limb_torso = _ARM_TORSO_COLL and (("OTHER" in _p) and bool(_p & {"RH", "LH", "RA", "LA"}))
            if is_cross_hand or is_limb_torso:
                n_keep += 1
                if is_limb_torso:
                    n_keep_lt += 1
                continue
            excl[body_paths[i]].append(body_paths[j])
            n_excl += 1
    return body_paths, census, excl, n_excl, n_keep, n_keep_lt, n_lh, n_other, n_rh


def _hand_classification_problems(census, n_lh, n_rh):
    """Fail loud if the classification is obviously wrong (would silently disable or mis-enable collisions) — bimanual
    must find both hands, FINGERS INCLUDED.  A guard of ``n_rh == 0 or n_lh == 0`` passes on a hand
    whose every fingertip is misclassified, because ONE palm body is enough to make the count non-zero — a silent
    degradation. The guard now asks the question the
    physics actually cares about — are this hand's FINGERTIPS in the collision set? — and a hand that classifies zero
    of them stops the run instead of quietly training palm-vs-palm.  B3b: this guard aborts EVERY bimanual run, so it
    was checked against the REAL body lists before being trusted (the merged URDFs under
    assets/data/Robots/full_robot_gen/, minus the fixed-joint children the Isaac importer merges away). All 5
    declared tips of both hands survive the merge on both sides — rby1_sharpa_wave
    right/left_{thumb,index,middle,ring,pinky}_DP (and, at the time, the legacy pair's
    right/left_hand_{thumb_4,index_2,middle_2, ring_2,little_2}) — so on those assets the guard PASSED 5/5 per side.
    What it can still catch is the case it exists for: a newly ported hand.  A tip the spec DECLARES but the stage
    does not HAVE is reported (below) but is not by itself fatal: the importer legitimately merges a fixed-joint tip
    into its parent (observed in a past hand port — the ``*_tip_link`` case, core/hands), and the porter's fix is
    in the spec. Fatal is only "this side's fingers are not in the collision set at all" and "a tip that IS on the
    stage did not classify" — the two states that make the physics silently wrong.
    """
    bad: list[str] = []
    if n_rh == 0:
        bad.append("no RIGHT-hand bodies")
    if n_lh == 0:
        bad.append("no LEFT-hand bodies")
    for side in _SIDES:
        spec_name, declared, found, ok = census[side]
        grp = _SIDE_GROUP[side]
        missing = [t for t in declared if t not in found]
        if not ok:
            bad.append(
                f"{side}: 0 fingertips classified {grp} — spec {spec_name!r} declares "
                f"{len(declared)} tips {list(declared)}, {len(found)} of them are on "
                f"the stage; MISSING from the stage: {missing or 'none'}. A hand whose "
                f"fingers land in OTHER collides palm-only"
            )
        elif len(ok) != len(found):
            bad.append(
                f"{side}: {len(found) - len(ok)}/{len(found)} present fingertips did "
                f"NOT classify {grp} (spec {spec_name!r}): "
                f"{[t for t in found if t not in ok]}. Those bodies exist on the stage "
                f"but the active hand's contact_body_regex does not match them"
            )
        elif missing:
            print(
                f"[BimanualSelfCollision] ⚠ {side}: HandSpec {spec_name!r} declares "
                f"fingertip bod{'y' if len(missing) == 1 else 'ies'} {missing}, which "
                f"{'is' if len(missing) == 1 else 'are'} NOT on the stage "
                f"({len(ok)}/{len(declared)} tips ARE in the collision set, so the run "
                f"continues). Most likely cause: the Isaac URDF importer MERGES a "
                f"fixed-joint child into its parent, so a tip that names a merged "
                f"frame disappears (observed in a past hand port, the '*_tip_link' "
                f"case). FIX in dexengine/core/hands: "
                f"point {spec_name}.tip_body_names_right at the SURVIVING parent body "
                f"and carry the pad in fingertip_site_offsets — do NOT add a body-name "
                f"literal here.",
                flush=True,
            )
    return bad


class BimanualSelfCollisionScene(InteractiveScene):
    """InteractiveScene enabling ONLY right-hand<->left-hand self-collision.

    Authors ``FilteredPairsAPI`` on the env_0 robot links right BEFORE cloning so
    the replicator remaps the exclusions into every cloned env. Used ONLY for
    bimanual training (wired by the env class); the single-hand scene never
    instantiates this. Requires the robot's ``enabled_self_collisions=True``
    (set bimanual-gated in ``TrainArmResidualEnvCfg.__post_init__``); without it
    PhysX generates no intra-articulation pairs and these exclusions are moot.

    Cross-side OBJECT isolation is NOT authored here. It belongs to the collision
    curriculum below (``author_collcurr_cross_pair_exclusions``, flag
    ``--collision_curriculum``, which the launcher's collision preset owns —
    dexengine/rl/launch/plan.py PRESET_FLAGS). The old env gate ``DEXENGINE_CROSS_SIDE_NOCOLL`` and
    its ``author_cross_side_object_isolation`` were DELETED:
    nothing in the repo ever set that variable, so the branch could only ever look
    engaged while doing nothing. Isolating the objects while the two hands still
    collide is not supported.
    ``DEXENGINE_DESK_ROBOT_NOCOLL=1``: additionally excludes robot<->desk (see
    author_robot_desk_nocollision). Unset = no change.
    """

    # COLLISION CURRICULUM request flags (class attrs: the env cannot pass ctor kwargs
    # through the InteractiveScene symbol swap). Set by ArmResidualTrainEnv right
    # before construction when cfg.collision_curriculum is on; reset in its
    # `finally`. Defaults False = byte-identical (no curriculum authoring).
    # collcurr_objpair: FULL ISOLATION — also filter objR<->objL at start.
    collcurr_start_off = False
    collcurr_objpair = False

    def _add_entities_from_cfg(self):
        """Authoring point: a clone_environments override is never called by the current IsaacLab, and post-clone
        FilteredPairs edits do not reach the physics. InteractiveScene.__init__ calls clone_environments right after
        _add_entities_from_cfg(), so **the end of this method (just before replication)** is the live authoring point,
        exactly the original design (author the template -> remap on replication).

        **Correction**: the reason first given for the two switches below, "inherited clones keep the rels' absolute
        target paths pointing at env_0", was **wrong**. Checked directly with pxr: USD **remaps rel targets that point
        into the source subtree into the copy's namespace**, through an `inherits` arc as well as `Sdf.CopySpec`,
        whether the authoring comes before or after the copy. Targets go empty in one combination only, **`Sdf.CopySpec`
        first and authoring later**, which is exactly the spawner's per-env path with `replicate_physics=False`. PERENV
        below fills **only that hole** (and did: verify prints `targets_foreign=0`). And the observation "env0 fine /
        env1~3 blow up" was itself **clip confounding**: with multiple objects the env number is the clip number (the
        trajectory manager's static assignment). Both default OFF:
          `DEXENGINE_CROSS_FILTER_PERENV=1` — author on **every env prim separately**, not only env_0
          `DEXENGINE_BSC_VERIFY=N`          — count the authoring per env and judge from the log (no replay needed)
        PERENV needs real prims per env, so it only means something **together with**
        `DEXENGINE_REPLICATE_PHYSICS=0`: that one first copies the env xforms with `copy_from_source=True` and the
        spawners build prims per env (interactive_scene `_add_entities_from_cfg`).
        """
        super()._add_entities_from_cfg()
        _perenv = False
        _roots = [self.env_prim_paths[0]]
        print(
            f"[BSC] authoring at the end of _add_entities_from_cfg "
            f"(start_off={type(self).collcurr_start_off}, perenv={_perenv}, "
            f"envs={len(_roots)}/{len(self.env_prim_paths)}, "
            f"replicate_physics={getattr(self.cfg, 'replicate_physics', None)})",
            flush=True,
        )
        _desk = env_flags.get("DEXENGINE_DESK_ROBOT_NOCOLL") == "1"
        for _k, _root in enumerate(_roots):
            # env_0 keeps the old log; the rest of PERENV authors quietly
            # (2048 envs × hundreds of log lines would make a run useless).
            _quiet = _k > 0
            author_cross_hand_only_self_collision(self.stage, _root, quiet=_quiet)
            if _desk:
                author_robot_desk_nocollision(self.stage, _root)
        if type(self).collcurr_start_off:
            # The curriculum record (_collcurr_pairs) stays **relative to env_0**: the runtime switch
            # (collcurr_enable_fraction) works by substituting `/env_0/` -> `/env_i/`, and changing the format here
            # would break that path.
            self._collcurr_pairs = author_collcurr_cross_pair_exclusions(
                self, self.env_prim_paths[0], include_objpair=type(self).collcurr_objpair
            )
        self._bsc_verify_filters()

    # ── Authoring result check hook ──────────────────────────────────────────
    # Why count in USD (not the PhysX tensor API): this runs during scene creation, before `sim.reset()`, so no PhysX
    # view exists yet. And the failure **is already visible in USD**: env_1's links have no FilteredPairsAPI, or the
    # targets point at **another env (env_0)**. Both are counted as they are.
    # Verdict: nonzero `targets_foreign`, or `prims_with_api` different from env_0, means not propagated.
    def _bsc_verify_filters(self):
        n_env = len(self.env_prim_paths)
        n_check = 0 if getattr(self.cfg, "replicate_physics", True) else 4
        if n_check <= 0:
            return
        for k in range(min(n_check, n_env)):
            root = self.env_prim_paths[k]
            robot_root = f"{root}/Robot"
            prim = self.stage.GetPrimAtPath(robot_root)
            if not prim.IsValid():
                print(f"[BSC] verify env_{k}: no Robot prim ({robot_root})", flush=True)
                continue
            n_api = n_tgt = n_foreign = 0
            # Count separately whether the authoring sat on **body prims** or **shape (collider) prims**; only these
            # two numbers show whether DEXENGINE_SELFCOLL_FILTER_ON_SHAPES took effect.
            n_on_body = n_on_shape = 0
            for p in Usd.PrimRange(prim):
                if not p.HasAPI(UsdPhysics.FilteredPairsAPI):
                    continue
                rel = UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel()
                if not rel:
                    continue
                tgts = list(rel.GetTargets())
                if not tgts:
                    continue
                n_api += 1
                n_tgt += len(tgts)
                n_foreign += sum(1 for t in tgts if not str(t).startswith(root + "/"))
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    n_on_shape += 1
                elif p.HasAPI(UsdPhysics.RigidBodyAPI):
                    n_on_body += 1
            n_col = sum(1 for c in Usd.PrimRange(prim) if c.HasAPI(UsdPhysics.CollisionAPI))
            n_col_ip = sum(
                1 for c in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if c.HasAPI(UsdPhysics.CollisionAPI)
            )
            print(
                f"[BSC] verify env_{k} filtered={n_tgt} prims_with_api={n_api} "
                f"targets_foreign={n_foreign} "
                f"on_shapes={n_on_shape} on_bodies={n_on_body} "
                f"colliders_real={n_col}/{n_col_ip}"
                + (
                    "  ★all colliders are instance proxies = no shape authoring possible"
                    if n_col == 0 and n_col_ip > 0
                    else ""
                )
                + ("  ★not propagated (targets point at another env)" if n_foreign else "")
                + ("  ★not propagated (no filter)" if n_tgt == 0 and k > 0 else ""),
                flush=True,
            )

    def clone_environments(self, copy_from_source: bool = False):
        """Authoring just before replication (env_0 only). PERENV is not done here: this method is called only with
        `replicate_physics=True`, when the clones do not exist yet and env_1.. prims are missing. PERENV is handled on
        the `_add_entities_from_cfg` side (= the `replicate_physics=False` path). This authoring duplicates
        `_add_entities_from_cfg`, but AddTarget skips existing targets, so it is idempotent."""
        print(f"[BSC] clone_environments ENTER (start_off={type(self).collcurr_start_off})", flush=True)
        author_cross_hand_only_self_collision(self.stage, self.env_prim_paths[0])
        if env_flags.get("DEXENGINE_DESK_ROBOT_NOCOLL") == "1":
            author_robot_desk_nocollision(self.stage, self.env_prim_paths[0])
        # Collision curriculum: author the start-OFF cross-pair filters LAST (after
        # the permanent exclusions) and remember the exact rel edits for the flip.
        # Passes SELF, not self.stage: the object prim paths are read off the cfg
        # this scene was built from.
        if type(self).collcurr_start_off:
            self._collcurr_pairs = author_collcurr_cross_pair_exclusions(
                self, self.env_prim_paths[0], include_objpair=type(self).collcurr_objpair
            )
        super().clone_environments(copy_from_source=copy_from_source)

    # ── The switch logic lives in module functions ─────────────────────────────────
    # Why: the canonical bi (SELFCOLL=0) is not swapped to this scene class, so a `hasattr` gate closed silently. The
    # two below are **thin delegations** kept for existing callers; implementation, comments and the evidence of the
    # physics effect live in the module functions `collcurr_enable_cross_pairs` / `collcurr_enable_fraction`.
    def collcurr_enable_cross_pairs(self) -> int:
        """One-off switch (delegates to the module function). Returns the removed rel targets."""
        return collcurr_enable_cross_pairs(self)

    def collcurr_enable_fraction(self, frac: float) -> tuple[int, int]:
        """Gradual switch (delegates to the module function). Returns (envs turned on this time, envs so far)."""
        return collcurr_enable_fraction(self, frac)
