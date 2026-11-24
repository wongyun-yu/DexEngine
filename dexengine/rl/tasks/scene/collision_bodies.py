"""Body and prim lookups of the collision filters: hand/arm body classes, object, support and slot prims, and the
FilteredPairs writer."""

from __future__ import annotations

import re
from functools import cache

from pxr import Sdf, Usd, UsdPhysics

_SIDES = ("right", "left")
_SIDE_GROUP = {"right": "RH", "left": "LH"}


def _to_side(name: str, side: str) -> str:
    """Map a right-hand-spelled spec name/regex to ``side`` (the ONE mirror rule)."""
    from dexengine.core.hands import mirror_right_to_left as _mirror

    return name if side == "right" else _mirror(name)


def _active_arm_name() -> str:
    """Arm name of the robot this process builds; ``naming`` is the one owner.

    The import stays lazy (this module asks for the arm only at scene clone time).
    """
    from dexengine.rl.tasks.mdp.naming import _active_arm_name as _aan

    return _aan()


def _active_hand_name() -> str:
    """Hand name of the robot this process builds: the env name **and the unset default** come from the owner.

    The default was once restated here as a legacy hand name while robots.selection already picked the canonical hand,
    so paths that bypass the launcher built another hand's matchers. The accessor ``dexengine.core.arms`` is the one
    owner now (lazy import, like ``_active_arm_name``).
    """
    from dexengine.core.arms import active_hand_name

    return active_hand_name()


# Body classification: read off the ASSET SPECS, never off name literals.
# A substring test such as ``"right_hand" in body_name`` only fits one vendor's naming: on SharpA Wave it caught
# nothing but the arm's EE link (the importer fixed-merges the three matching bodies into it), so every finger was
# classified OTHER while ``n_rh`` stayed non-zero and cross-hand collision was palm-only. The names now come from
# the ACTIVE hand's spec (``contact_body_regex_right``, palm frames, ``tip_body_names_right``), mirrored to the left
# by the one mirror rule (``core.hands.mirror_right_to_left``), plus the arm's ``ee_body_name(side)`` (the palm hulls
# fixed-merge into it). Only the active hand's pattern is compiled: the classifier must never be broader than the
# contact sensor built from the same spec (pooling every registered hand would let ``right_.*`` swallow a
# ``right_wheel`` on any robot). The arm's own non-EE links are hard-OTHER before any regex runs; the fingertip
# census still probes every registered spec, because it compares exact declared names (no over-match possible) and
# so diagnoses a mixed robot (a different hand per side) per side.


@cache
def _hand_matchers(arm_name: str, hand_name: str):
    """Per-side body matchers for the active robot, compiled once per process.

    Args:
        arm_name: ``DEXENGINE_ARM`` (or the legacy arm). Keyed rather than read
            inside so the cache cannot go stale if a test flips the env var.
        hand_name: ``DEXENGINE_HAND`` (or the legacy hand).

    Returns:
        ``(pats, lits, tips, ee, arm_links)``, each keyed by side:
          * ``pats[side]``  compiled ``contact_body_regex`` of the ACTIVE hand
            ONLY (fullmatch) — see the B3b note above
          * ``lits[side]``  frozenset of the ACTIVE hand's declared palm names
          * ``tips[side]``  ``(spec_name, tip_body_names)`` for EVERY registered
            hand, active first — exact names, census/diagnosis only
          * ``ee[side]``    the arm's end-effector body name (carries the palm)
          * ``arm_links[side]`` the arm's OTHER link names — hard-OTHER
    """
    from dexengine.core.arms import ARM_REGISTRY
    from dexengine.core.hands import HAND_REGISTRY

    arm = ARM_REGISTRY.get(arm_name)
    if arm is None:
        raise RuntimeError(
            f"[BimanualSelfCollision] arm {arm_name!r} is not in ARM_REGISTRY "
            f"(known: {sorted(k for k, v in ARM_REGISTRY.items() if v is not None)}). "
            "Cross-hand collision needs the arm's EE body name — a draft spec that "
            "has not been registered must not reach a training run."
        )

    registered = [s for s in HAND_REGISTRY.values() if s is not None]
    active = HAND_REGISTRY.get(hand_name)
    if active is None:
        raise RuntimeError(
            f"[BimanualSelfCollision] hand {hand_name!r} is not in HAND_REGISTRY "
            f"(known: {sorted(s.name for s in registered)}). The classifier reads "
            "body names off the HandSpec; without one it would have to guess, and "
            "guessing here is what made B3 invisible."
        )
    ordered = [active] + [s for s in registered if s is not active]

    pats, lits, tips, ee, arm_links = {}, {}, {}, {}, {}
    for side in _SIDES:
        # ACTIVE hand only. A tuple (not a bare pattern) because a spec may one
        # day declare several patterns; it is NOT a place to add other hands.
        pats[side] = tuple(re.compile(_to_side(rx, side)) for rx in (active.contact_body_regex_right,))
        lits[side] = frozenset(
            _to_side(n, side) for n in (active.palm_fk_body_right, active.urdf_palm_frame_right) if n
        )
        # Census probes every registered hand: EXACT names, so no over-match.
        tips[side] = tuple((s.name, tuple(_to_side(b, side) for b in s.tip_body_names_right)) for s in ordered)
        ee[side] = arm.ee_body_name(side)
        # Every arm link EXCEPT the EE. The EE is excluded because the palm
        # bodies fixed-merge into it, which is precisely why it counts as hand.
        arm_links[side] = frozenset(arm.body_name(side, i) for i in range(arm.index_base, arm.index_hi + 1)) - {
            ee[side]
        }
    return pats, lits, tips, ee, arm_links


def classify_body(body_name: str) -> str:
    """Classify a robot link by name into right-hand / left-hand / other.

    Hand = every body the ACTIVE ``HandSpec`` declares for that side (its contact
    regex + its palm frames) PLUS the arm's EE link, which carries the palm hulls
    after the fixed-joint merge. An arm link that is not the EE is OTHER before
    any regex is consulted. See the B3/B3b note above: nothing here is spelled as
    a literal, so a new hand is covered by registering its spec — and no other
    hand's naming convention can reach this robot.
    """
    pats, lits, _tips, ee, arm_links = _hand_matchers(_active_arm_name(), _active_hand_name())
    for side in _SIDES:
        if body_name == ee[side] or body_name in lits[side]:
            return _SIDE_GROUP[side]
    for side in _SIDES:
        if body_name in arm_links[side]:
            # ARM link — never a hand body. It must be told apart from the torso, so it returns the per-side arm
            # group instead of one "OTHER" lump. Old callers only check `!= "RH"/"LH"`, so they behave the same;
            # only the places using the new rule tell "RA"/"LA" apart.
            return "RA" if side == "right" else "LA"
    for side in _SIDES:
        if any(p.fullmatch(body_name) for p in pats[side]):
            return _SIDE_GROUP[side]
    return "OTHER"


def classify_bodies(body_names) -> tuple[dict, dict]:
    """Classify a whole body-name list AND census the fingertips per side.

    The census is what the B3 guard needs: "did we find this hand's FINGERTIPS?"
    is the question ``n_rh > 0`` could not answer — a palm alone answered it yes.

    Args:
        body_names: robot link names (prim leaf names).

    Returns:
        ``(groups, census)`` where ``groups`` maps name -> ``"RH"/"LH"/"OTHER"``
        and ``census`` maps side -> ``(spec_name, declared, present, classified)``:
        the best-matching hand spec for that side, its declared tip names, those
        actually present on the stage, and those the classifier put on that side.
        ``declared - present`` (tips the spec names but the stage does not have)
        is what the guard reports as a spec/importer mismatch.
    """
    _pats, _lits, tips, _ee, _arm = _hand_matchers(_active_arm_name(), _active_hand_name())
    groups = {n: classify_body(n) for n in body_names}
    present = set(body_names)
    census = {}
    for side in _SIDES:
        # Best-matching spec = the one with the most declared tips actually on the
        # stage. On a mixed robot that picks a different hand per side, which is
        # exactly right; ties keep the active selection (it is ordered first).
        best = (tips[side][0][0], tips[side][0][1], ())
        for spec_name, declared in tips[side]:
            hit = tuple(t for t in declared if t in present)
            if len(hit) > len(best[2]):
                best = (spec_name, declared, hit)
        spec_name, declared, found = best
        ok = tuple(t for t in found if groups[t] == _SIDE_GROUP[side])
        census[side] = (spec_name, declared, found, ok)
    return groups, census


# Scene object prims: read off the cfg that creates them. ``InteractiveScene`` keeps the cfg it was built from
# (``self.cfg``) and resolves every entity's ``prim_path`` template before cloning, so at authoring time
# ``scene.cfg.object_right.prim_path`` already is the real namespaced path; it is read and the env namespace swapped
# for env_0's. No prim name is spelled here. What is spelled is the scene entity KEY (``object_<side>``), the same
# derivation the launcher treats as canonical (``rl/cli.py`` ``_ACTIVE_OBJ_KEY``); a stale key raises, listing the cfg
# fields that do carry a prim_path, instead of authoring nothing. Cross-side object isolation in a live bimanual run
# comes from ``author_collcurr_cross_pair_exclusions`` alone.


def _object_cfg_keys(side: str) -> tuple[str, ...]:
    """Scene-cfg field names that may hold ``side``'s task object.

    The old right-only key ``"object"`` (an archived OakInk cfg) fallback is gone: no live scene cfg has that field,
    and identity inputs get no silent fallback.
    """
    return (f"object_{side}",)


def scene_object_key(scene_keys, side: str, shared_object: bool, idle_side: str = "") -> str | None:
    """The scene-entity key that holds ``side``'s task object — the ONE rule runtime code uses.

    Plain rule: ``object_<side>``. Under an explicit ``shared_object=True`` statement (env cfg attribute written by
    env_setup.build._apply_shared_object; ``--shared_object 1``) the left hand's object IS the single
    ``object_right`` prim, and ``object_left`` must be absent — both hands manipulate one physical object.
    The flag is a statement, never inferred from key presence: a uni scene without ``object_left`` keeps
    resolving ``left`` -> ``object_left`` exactly as before, so callers see the same KeyError/None they always did.

    Args:
        scene_keys: ``scene.keys()`` (or any container supporting ``in``); used only to check the statement.
        side: "right" | "left".
        shared_object: ``bool(env.cfg.shared_object)`` — the caller passes the statement, this function checks it.
        idle_side: ``env.cfg.idle_side`` (rule D16, env_setup.sides.apply_idle_side) — that side has NO object: the
        answer
            is None and the caller skips the object (a prim of that name in the scene contradicts the statement →
            raise).
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    key = f"object_{side}"
    if idle_side and side == idle_side:
        if key in scene_keys:
            raise RuntimeError(
                f"[idle_side] cfg says idle_side={idle_side} but the scene declares '{key}' — "
                "apply_idle_side did not remove it (statement/scene mismatch)."
            )
        return None
    if shared_object and side == "left":
        if key in scene_keys:
            raise RuntimeError(
                "[shared_object] cfg says shared_object=True but the scene declares 'object_left' — "
                "_apply_shared_object did not remove the left prim (statement/scene mismatch)."
            )
        return "object_right"
    return key


def _object_cfg_fields(scene) -> list[str]:
    """Every scene-cfg field that carries a ``prim_path`` — for error messages."""
    cfg = getattr(scene, "cfg", None)
    if cfg is None:
        return []
    return sorted(k for k, v in vars(cfg).items() if v is not None and hasattr(v, "prim_path"))


def object_prim_paths(scene, env_root: str) -> dict[str, str]:
    """Per-side task-object prim paths under ``env_root``, derived from the CFG.

    Args:
        scene: the ``InteractiveScene`` being built (``scene.cfg`` is the
            authority that CREATES the object prims; ``scene.env_regex_ns`` is
            the namespace its ``prim_path``s were resolved into).
        env_root: the env prim path to rewrite them onto, e.g.
            "/World/envs/env_0" (this module authors on the env_0 TEMPLATE only).

    Returns:
        ``{side: prim_path}`` for each side whose object the cfg declares. A side
        the cfg does not declare is simply absent (a uni scene has one side); the
        CALLER decides whether that is fatal.
    """
    cfg = getattr(scene, "cfg", None)
    if cfg is None:
        raise RuntimeError(
            "[BimanualObjects] scene has no .cfg — the object prim names are read "
            "off the cfg that creates them; there is no literal table to "
            "fall back to."
        )
    ns = scene.env_regex_ns
    out: dict[str, str] = {}
    for side in _SIDES:
        for key in _object_cfg_keys(side):
            entity = getattr(cfg, key, None)
            raw = getattr(entity, "prim_path", None) if entity is not None else None
            if not raw:
                continue
            # The scene resolves the template in place before cloning, so this is
            # normally a no-op; it also covers a caller that clones by hand.
            resolved = raw.format(ENV_REGEX_NS=ns) if "{" in raw else raw
            if resolved.startswith(ns + "/"):
                out[side] = env_root + resolved[len(ns) :]
            elif resolved.startswith(env_root + "/"):
                out[side] = resolved
            else:
                raise RuntimeError(
                    f"[BimanualObjects] scene.cfg.{key}.prim_path={resolved!r} is "
                    f"under neither the env namespace {ns!r} nor {env_root!r}. "
                    "FilteredPairs are authored on the env_0 TEMPLATE only."
                )
            break
    return out


def _object_root_prim(stage, obj_paths: dict[str, str], side: str):
    """Resolve one side's object root prim from an ``object_prim_paths`` result.

    Returns:
        ``(prim, path)`` if the cfg-derived path exists on the stage, else
        ``(None, path_or_empty)``. Callers decide how loudly to fail.
    """
    path = obj_paths.get(side, "")
    if not path:
        return None, ""
    prim = stage.GetPrimAtPath(path)
    return (prim, path) if prim.IsValid() else (None, path)


def object_rigid_bodies(stage, obj_paths: dict[str, str], side: str) -> list[str]:
    """Rigid-body prim paths under the object root of ``side`` (empty when the scene has no object there)."""
    root, _path = _object_root_prim(stage, obj_paths, side)
    if root is None:
        return []
    return [p.GetPath().pathString for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


#: Scene cfg field prefix of the support discs (rules D20 aerial · D26′ desk support · D40 start support). Writer
#: `env_setup.supports._build_disc_slots` (`support_disc_<k>`); readers `object_pin.support_disc_state` and this module.
SUPPORT_DISC_PREFIX = "support_disc_"


def support_disc_prim_paths(scene, env_root: str) -> list[str]:
    """Prim paths of the support disc slots (under `env_root`), read from the `prim_path` of **the cfg that makes
    them** (no literal table). A scene without discs -> []. Paths resolve by the same rule as `object_prim_paths`."""
    return slot_prim_paths(scene, env_root, SUPPORT_DISC_PREFIX, "DISC_COLL")


def support_object_prim_paths(scene, env_root: str) -> list[str]:
    """Prim paths of the support object slots (rule D42; writer `env_setup.supports._build_support_slots`): the same
    rule as the discs."""
    from dexengine.rl.tasks.scene.object_pin import SUPPORT_OBJ_PREFIX

    return slot_prim_paths(scene, env_root, SUPPORT_OBJ_PREFIX, "SUPPORT_COLL")


def slot_prim_paths(scene, env_root: str, prefix: str, tag: str) -> list[str]:
    cfg = getattr(scene, "cfg", None)
    if cfg is None:
        raise RuntimeError(f"[{tag}] scene has no .cfg — slot prim paths are read from the cfg that makes them")
    ns = scene.env_regex_ns
    out: list[str] = []
    for key in sorted(k for k in vars(cfg) if k.startswith(prefix)):
        raw = getattr(getattr(cfg, key), "prim_path", None)
        if not raw:
            raise RuntimeError(f"[{tag}] scene.cfg.{key} has no prim_path")
        resolved = raw.format(ENV_REGEX_NS=ns) if "{" in raw else raw
        if resolved.startswith(ns + "/"):
            out.append(env_root + resolved[len(ns) :])
        elif resolved.startswith(env_root + "/"):
            out.append(resolved)
        else:
            raise RuntimeError(f"[{tag}] scene.cfg.{key}.prim_path={resolved!r} is outside the env namespace {ns!r}")
    return out


def support_disc_slot_paths(scene, env_root: str) -> dict[int, str]:
    """{slot number k: disc prim path}; the number is the k of the cfg field name `support_disc_<k>` (an integer, not
    lexical order, which breaks at 10+ slots). Used to pair with the disc owner statement
    (`cfg.support_disc_owner[clip][k]`)."""
    paths = support_disc_prim_paths(scene, env_root)  # same order; fails there without a cfg
    keys = sorted(k for k in vars(scene.cfg) if k.startswith(SUPPORT_DISC_PREFIX))
    out: dict[int, str] = {}
    for key, path in zip(keys, paths, strict=True):
        tail = key[len(SUPPORT_DISC_PREFIX) :]
        if not tail.isdigit():
            raise RuntimeError(f"[DISC_COLL] cannot read the slot number of disc cfg field {key!r}")
        out[int(tail)] = path
    return out


def _rigid_bodies(stage, root: str, what: str) -> list[str]:
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        raise RuntimeError(f"[DISC_COLL] {what} prim {root} is not on the stage — called before the spawn?")
    got = [p.GetPath().pathString for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not got:
        raise RuntimeError(f"[DISC_COLL] no rigid body (RigidBodyAPI) under {what} {root}")
    return got


def author_filtered_pairs(stage, src_paths, target_paths) -> tuple[dict[str, list[str]], int]:
    """Author ``FilteredPairsAPI`` on each ``src`` in ``src_paths``, excluding every
    ``target`` in ``target_paths`` not already excluded.

    THE shared authoring primitive for every cross-pair exclusion in this module
    (collision curriculum RHxobjL/LHxobjR/objRxobjL, and the arm<->object physics
    ablation below) — one code path, so "does this link<->RigidObject exclusion
    reach PhysX" only has to be validated once. It DOES: unlike
    articulation-link<->articulation-link self-collision (colliders are
    instance proxies, so ``FilteredPairsAPI`` on the link body prim never reaches
    the shape), a RigidObject prim is not instanced, so authoring on it (or on the
    articulation link prim it targets) is fully valid.

    Returns:
        ``({src: [added_targets]}, n_added)`` — only the NEW rel edits (so a caller
        can record exactly what to undo later, mirroring the collision-curriculum
        flip bookkeeping) and an accurate count for logging.

    Reads come first (each source's targets so far), then every write under one ``Sdf.ChangeBlock`` — one change
    notice per call instead of one per edit; nothing is read inside the block (Usd reads there are stale: the
    callers' read-backs would miss the new targets). A source listed twice sees its first pass's targets, as when
    each pass read them back (GetTargets gives ``Sdf.Path``s, so string targets never match: kept as it was).
    """
    pairs: dict[str, list[str]] = {}
    n = 0
    have: dict[str, set] = {}
    for s in src_paths:
        if s not in have:
            rel = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(s)).GetFilteredPairsRel()
            have[s] = set(rel.GetTargets()) if rel else set()
    with Sdf.ChangeBlock():
        for s in src_paths:
            api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(s))
            rel = api.GetFilteredPairsRel() or api.CreateFilteredPairsRel()
            existing, added = have[s], []
            for t in target_paths:
                if t not in existing:
                    rel.AddTarget(t)
                    added.append(t)
                    pairs.setdefault(s, []).append(t)
                    n += 1
            have[s] = existing | {Sdf.Path(t) for t in added}
    return pairs, n


def _split_robot_bodies(stage, env_root: str) -> list[str]:
    """Every RigidBodyAPI link under the robot (right) articulation."""
    root = stage.GetPrimAtPath(f"{env_root}/Robot")
    if not root.IsValid():
        raise RuntimeError(f"[SPLIT_OFF] robot prim not found: {env_root}/Robot")
    return [p.GetPath().pathString for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _split_robot_left_bodies(stage, env_root: str) -> list[str]:
    """Every link under the robot_left articulation (``RobotL`` prim, split_artic.py)."""
    from dexengine.rl.tasks.scene import split_artic as _sa

    left_root = f"{env_root}/{_sa.LEFT_PRIM_LEAF}"
    root = stage.GetPrimAtPath(left_root)
    if not root.IsValid():
        raise RuntimeError(
            f"[SPLIT_OFF] robot_left prim not found: {left_root} "
            "(DEXENGINE_BI_SPLIT_ARTIC=1 but the assets are missing or were not spawned)."
        )
    return [p.GetPath().pathString for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
