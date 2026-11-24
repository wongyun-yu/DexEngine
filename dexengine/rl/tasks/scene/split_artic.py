"""Shared helpers of the path that splits the two arms into **separate articulations**.

Why. Two earlier designs died on the same root: all 72 colliders of this asset are **USD instance proxies**, and API
schemas cannot be applied to proxies, so no ``FilteredPairsAPI`` ever resolves on an articulation collider (neither
pairs inside an articulation nor proxy body -> link). **Only a design without any filter survives.**

Splitting the robot into two articulations yields the canonical collision set **without authoring**:

===========================================  ==========================================
Pair (split articulation, before presets)     What the split articulation does
===========================================  ==========================================
ON  hand·arm <-> other hand·arm               ordinary rigid collision **between** articulations (no filter needed)
ON  left arm·left hand <-> torso              collision between articulations (the torso belongs to R)
ON  hand <-> other object · object <-> object  rigid pairs as before, untouched
OFF hand internal · arm internal · parent-child  ``enabled_self_collisions=False`` on each articulation
===========================================  ==========================================

⚠ **Known asymmetry**: with the two-piece split the torso sits in the R articulation, so
**right arm·right hand <-> torso is still self-collision**, and with the self-collision flag permanently OFF it stays
off. Only the left side becomes ON. Only a three-piece split (option B: a separate torso articulation) removes it
(follow-up).

Switch
------
``DEXENGINE_BI_SPLIT_ARTIC=1`` (default ``0``). With the default no branch of this module runs, so it is
**byte-identical** with before. Every place that asks "is the left side a separate articulation" looks at
:func:`enabled` alone. (The legacy pair-only diagnostic switch ``DEXENGINE_SPLIT_ARTIC`` went away with that arm;
``robots.selection`` refuses it when set.)

Assets
------
:mod:`dexengine.robots.usd.split_arm` splits the canonical USD by **reference + deactivation** into
``<combo>_split/{<combo>_split_R.usd, <combo>_split_L.usd}`` (not flattened, so instancing stays alive and the
U1DEIN/OOM wall is avoided). The path rule lives in :func:`split_usd_pair` (pure derivation; :func:`split_usd_paths`
requires existence on top of it) and the generator uses the same rule. The floating hand (``DEXENGINE_ARM=floating``)
bakes only this pair without a merged version (experiments/ablations/floating_hand/build_floating_usd.sh); the
per-side URDF places are :func:`split_urdf_pair`.
"""

from __future__ import annotations

import pathlib

from dexengine.core.hands import mirror_right_to_left as _to_left
from dexengine.rl.tasks import env_flags

#: Master switch (combo path). Default 0.
BI_ENV = "DEXENGINE_BI_SPLIT_ARTIC"
#: Scene entity name of the left articulation (downstream wiring: the `left_hand` alias of actions and observations
# points here).
LEFT_ENTITY = "robot_left"
#: Scene entity names of the robot articulations (whole robot, or torso + right side · the split left side).
ROBOT_KEYS = ("robot", LEFT_ENTITY)
#: Prim name the left articulation spawns under; a **sibling** of ``/Robot``.
LEFT_PRIM_LEAF = "RobotL"


def bi_enabled() -> bool:
    """Is the split articulation path of a combo (rby1+sharpa_wave, ...) on?"""
    return env_flags.get(BI_ENV) == "1"


def enabled() -> bool:
    """Are the left arm and hand a **separate articulation**? The answer to the downstream question "the left side is
    ``robot_left``".

    Every wiring point looks at this one. (The legacy pair diagnostic switch used to make it true as well; it went
    away with that arm.)
    """
    return bi_enabled()


def left_robot_ns() -> str:
    """Prim namespace prefix of the left links (for sensor prim_paths).

    Splits into ``/RobotL`` only with the split spawn (``bi``).
    """
    return "{ENV_REGEX_NS}/" + (LEFT_PRIM_LEAF if bi_enabled() else "Robot")


# ---------------------------------------------------------------------------
# Asset paths: the home of the split name rule. **Derivation** (pure, never looks at existence) and **requirement**
# (fails when missing) are separate: readiness checks (robots.selection.combo_ready and the reference check, through
# robots.selection.missing_robot_files) take the derivation only and report what is missing in their own words, while
# the spawn (split_usd_paths) fails when missing. The floating hand bakes no merged version, so the merged path these
# functions receive is only a **place** that does not exist.
# ---------------------------------------------------------------------------
#: Side tags: the right articulation R (with the torso) · left L. The split USD pair and the floating hand's per-side
# URDFs share them.
SPLIT_SIDE_TAGS = (("right", "R"), ("left", "L"))


def split_usd_pair(canon_usd) -> tuple[pathlib.Path, pathlib.Path]:
    """Canonical combo USD place -> ``(R usd, L usd)`` places; pure derivation.

    Rule (one copy with the generator): ``<dir>/<combo>.usd`` ->
    ``<dir>/../<combo>_split/<combo>_split_{R,L}.usd``.
    """
    src = pathlib.Path(canon_usd)
    stem = src.stem
    d = src.parent.parent / f"{stem}_split"
    right, left = (d / f"{stem}_split_{t}.usd" for _side, t in SPLIT_SIDE_TAGS)
    return right, left


def split_urdf_pair(combo_urdf) -> tuple[pathlib.Path, pathlib.Path]:
    """Combo URDF place -> the floating hand's per-side URDF ``(R, L)`` places; pure derivation. The URDFs live in **the
    same folder** (``<dir>/<combo>.urdf`` -> ``<dir>/<combo>_split_{R,L}.urdf``, keeping the depth of the relative mesh
    paths ``../../…``).
    The generator experiments/ablations/floating_hand/make_floating_urdf.py writes through this function."""
    src = pathlib.Path(combo_urdf)
    right, left = (src.parent / f"{src.stem}_split_{t}.urdf" for _side, t in SPLIT_SIDE_TAGS)
    return right, left


def split_usd_paths(canon_usd: str) -> tuple[str, str]:
    """Canonical combo USD -> ``(R usd, L usd)``. Fails **loudly** when missing (path rule = :func:`split_usd_pair`)."""
    r_usd, l_usd = split_usd_pair(canon_usd)
    missing = [str(p) for p in (r_usd, l_usd) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{BI_ENV}=1 but the split articulation assets are missing: {missing}. Bake them first — "
            f"`python -m dexengine.robots.usd.split_arm --src {canon_usd}` "
            "(the canonical USD is not touched)."
        )
    return str(r_usd), str(l_usd)


# ---------------------------------------------------------------------------
# Side classification, derived from the one mirror rule (core.hands.mirror_right_to_left).
# ---------------------------------------------------------------------------
def side_of(name: str) -> str | None:
    """Side of a joint/body name or regex. ``None`` without a side token (neutral: head, torso, catch-all)."""
    if _to_left(name) != name:
        return "right"
    if name.replace("left_", "right_", 1) != name:
        return "left"
    return None


def _is_catchall(key: str) -> bool:
    """Is this an all-directions pattern with ``.*`` or a lookahead (valid on both views)?"""
    return ".*" in key or key.startswith("(?")


def keep_for(view_side: str, key: str) -> bool:
    """Keep ``key`` in the ``view_side`` articulation cfg?

    A neutral key stays on **both sides when it is a catch-all**, otherwise (a real name such as a head joint) only on
    the R (=right) view that holds the torso. Isaac's ``resolve_matching_names_values`` **dies on 0 matches**, so
    without this filter the left view dies at boot (the reason the legacy path emptied ``init_state.joint_pos``
    entirely).
    """
    s = side_of(key)
    if s is None:
        return _is_catchall(key) or view_side == "right"
    return s == view_side


def _filter_value(view_side: str, value):
    """Filter dict values (joint name -> number) by side. Scalars and None pass unchanged."""
    if not isinstance(value, dict):
        return value
    out = {k: v for k, v in value.items() if keep_for(view_side, k)}
    return out or None


def restrict_actuators_to_side(art_cfg, view_side: str) -> list[str]:
    """Keep only one side's actuator groups in an articulation cfg. Returns the names of the removed groups.

    A group whose ``joint_names_expr`` disappears entirely on that side (e.g. the head on the left view) is **removed
    as a whole**: IsaacLab dies with ValueError when a regex matches 0. Gain and limit dicts are filtered by the same
    rule, preventing "right key missing from the left view" errors.
    """
    dropped = []
    for gname in list(art_cfg.actuators):
        g = art_cfg.actuators[gname]
        exprs = [e for e in list(g.joint_names_expr) if keep_for(view_side, e)]
        if not exprs:
            del art_cfg.actuators[gname]
            dropped.append(gname)
            continue
        g.joint_names_expr = exprs
        for fld in (
            "stiffness",
            "damping",
            "armature",
            "effort_limit_sim",
            "velocity_limit_sim",
            "effort_limit",
            "velocity_limit",
        ):
            if hasattr(g, fld):
                setattr(g, fld, _filter_value(view_side, getattr(g, fld)))
    return dropped


def restrict_init_state_to_side(art_cfg, view_side: str) -> None:
    """Keep only that side's ``init_state.joint_pos/joint_vel`` keys."""
    st = art_cfg.init_state
    for fld in ("joint_pos", "joint_vel"):
        v = getattr(st, fld, None)
        if isinstance(v, dict):
            setattr(st, fld, {k: x for k, x in v.items() if keep_for(view_side, k)})


# ---------------------------------------------------------------------------
# Runtime: keeping the two roots in sync (the quietest place to go wrong)
# ---------------------------------------------------------------------------
def left_articulation(env):
    """The scene's left articulation. ``None`` without the split."""
    if not enabled():
        return None
    scene = env.scene
    try:
        keys = scene.keys()
    except Exception:  # noqa: BLE001 — an env without an interactive scene has no split articulation
        return None
    if LEFT_ENTITY in keys:
        return scene[LEFT_ENTITY]
    return None


def robot_articulations(env) -> list[tuple[str, object]]:
    """Every robot articulation of the scene as (scene name, articulation): ``robot`` and, with the split, the left one
    (left_articulation). A runtime write of a robot physics value goes to each of them — writing ``robot`` alone
    leaves the split left side at its USD value. A scene that holds ``robot_left`` while the split is off contradicts
    the flag and dies."""
    scene = env.scene
    try:
        keys = scene.keys()
    except Exception:  # noqa: BLE001 — an env without an interactive scene has no robot articulation
        return []
    if LEFT_ENTITY in keys and not enabled():
        raise RuntimeError(f"[SPLIT_ARTIC] the scene holds {LEFT_ENTITY!r} but {BI_ENV} is off")
    out = [("robot", scene["robot"])] if "robot" in keys else []
    left = left_articulation(env)
    return out + ([(LEFT_ENTITY, left)] if left is not None else [])


def mirror_root_pose(env, root_pose, env_ids) -> bool:
    """Write **the same root pose** written to the right articulation to the left one too.

    The two articulations each have a fixed base; updating only one stands the two sides at different heights (the
    half-write family). Every place that moves the root, like the ENVZ root z, must go through this function. Without
    the split it does nothing (byte-identical).

    Returns: whether the left side was written.
    """
    left = left_articulation(env)
    if left is None:
        return False
    import torch

    left.write_root_pose_to_sim(root_pose, env_ids)
    left.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids)
    if not getattr(env, "_split_root_sync_logged", False):
        env._split_root_sync_logged = True
        print(
            "[SPLIT_ARTIC] syncing the two articulation roots — the same root pose goes to robot and robot_left "
            "(writing one would stand the two sides at different heights)",
            flush=True,
        )
    return True


# ---------------------------------------------------------------------------
# The original joints the split cut: the side asking about joint neighbours (dexengine.rl.judge.arm_penetration, ...)
# looks at this one
# ---------------------------------------------------------------------------
#: Tolerance for treating the articulation-relative transforms of the mount (L root) and the R root link as equal (m ·
#: rotation matrix elements). The generator copies the transform as is, so the measured difference is float noise only;
#: anything larger means the split contract 'mount = stand-in of the R root' is broken.
CUT_POSE_TOL = 1e-5


def _world_fixed_root(stage, art_root: str) -> str:
    """Under articulation prim `art_root`, the body1 of the **world-fixed joint** (empty body0) = that articulation's
    root link path. Anything but exactly one fails: without the root the cut joints cannot be derived."""
    from pxr import Usd, UsdPhysics

    prim = stage.GetPrimAtPath(art_root)
    if not prim.IsValid():
        raise RuntimeError(f"[split] articulation prim {art_root} is missing")
    roots = []
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not b0 and len(b1) == 1:
            roots.append(str(b1[0]))
    if len(roots) != 1:
        raise RuntimeError(
            f"[split] {art_root}: {len(roots)} bodies of world-fixed joints (no body0) {roots} — "
            "the cut joints can be derived only with one articulation root"
        )
    return roots[0]


def cut_joint_pairs(stage, right_root: str, left_root: str) -> tuple[set, dict]:
    """The **original asset joints** the two-piece split cut -> {frozenset((body name, body name))} and an evidence
    dict.

    Rule (split contract; this module's 'Assets' header and ``dexengine/robots/usd/split_arm.py``): the L articulation's
    root link (the mount) is a stand-in with **the same transform** as the R articulation's root link, and the branch
    joints only had body0 moved from the R root to the mount. So for every L joint with body0 = mount, the original
    asset joint is (R root link, that joint's body1). No names are hard-coded: the root comes from the world-fixed
    joint, children from the joint graph.
    Fails when it cannot derive: not exactly one root per articulation · 0 joints leaving the mount · a different
    articulation-relative transform of mount and R root (CUT_POSE_TOL); a silent empty set would count cut neighbours as
    penetrations."""
    from pxr import Usd, UsdGeom, UsdPhysics

    r_base, mount = _world_fixed_root(stage, right_root), _world_fixed_root(stage, left_root)
    children = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(left_root)):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if len(b0) == 1 and str(b0[0]) == mount and len(b1) == 1:
            children.append((p.GetName(), str(b1[0])))
    if not children:
        raise RuntimeError(
            f"[split] no joint leaves the L root {mount} — without the branch joint the cut original "
            "joints cannot be derived"
        )
    cache = UsdGeom.XformCache()

    def rel(path, art):  # articulation-prim-relative transform (independent of the spawn place)
        return (
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(art)).GetInverse()
        )

    a, b = rel(mount, left_root), rel(r_base, right_root)
    err = max(abs(a[i][k] - b[i][k]) for i in range(4) for k in range(4))
    if err > CUT_POSE_TOL:
        raise RuntimeError(
            f"[split] the articulation-relative transforms of L root {mount} and R root {r_base} differ"
            f" (max difference {err:.2e} > {CUT_POSE_TOL}) — the 'mount = stand-in of the R root' contract is broken"
        )
    base = stage.GetPrimAtPath(r_base).GetName()
    pairs = {frozenset((base, stage.GetPrimAtPath(c).GetName())) for _, c in children}
    return pairs, {
        "r_base": base,
        "mount": stage.GetPrimAtPath(mount).GetName(),
        "joints": [n for n, _ in children],
        "pose_err": err,
    }
