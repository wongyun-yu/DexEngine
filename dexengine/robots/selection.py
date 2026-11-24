"""Which (arm, hand) this process builds — one resolver, read from environment variables.

The gym task ids are fixed while the robot varies: ``--arm rby1 --hand sharpa_wave`` produces the same training run
with only the hardware swapped. The selection must reach the env-cfg factories, which run at module import time
inside the gym entry point, so the launcher (``dexengine.rl``) records it in the environment of the Isaac process and
this module reads it there. Environment variables read here (all set by the launcher for the Isaac worker):

    DEXENGINE_ARM / DEXENGINE_HAND       the (arm, hand) pair (names owned by ``dexengine.core.arms``); required.
    DEXENGINE_COMBO_USD_VARIANT          explicit robot-asset variant suffix (unset = the arm's canonical sim variant;
                                         set to the empty string = the unsuffixed old generation, on purpose).
    DEXENGINE_COMBO_VARIANT              data-world suffix of the combo (an alternative preprocessing world of the
                                         same pair, e.g. ``_reopt``); unset = "".
    DEXENGINE_DATASET / DEXENGINE_EMBODIMENT   the data axes of the run; required by ``combo_data_paths``.
    DEXENGINE_ALLOW_GENERATION_MISMATCH  "1" = run a deliberate asset/base-pose generation mismatch (A/B experiments).

The pair must name REGISTERED specs with a mount, otherwise a loud error names what is missing: an unregistered
draft spec does not exist here, which keeps placeholder geometry out of training.

Adding a robot: register the HandSpec/ArmSpec, fill MOUNT_REGISTRY, bake the combo USD. ``combo_ready()`` is the
single readiness judgment the launcher gate consults — no per-robot edits in this file.
"""

from __future__ import annotations

import os
import pathlib

from dexengine.core import layout as _layout
from dexengine.core.arms import (
    ARM_ENV,
    ARM_GENERATIONS,
    ARM_REGISTRY,
    ARM_RENDER_GEN,
    HAND_ENV,
    ArmSpec,
    active_arm_name,
    active_hand_name,
    canon_variant,
    generation_of,
    plate_variant,
    retired_plate_why,
    sim_body_variant,
    urdf_variant,
)
from dexengine.core.embodiment import base_pos_env_name
from dexengine.core.floating import MARK as _FH_MARK
from dexengine.core.floating import data_world_arm, mode_canon_path
from dexengine.core.hands import HAND_REGISTRY, HandSpec
from dexengine.core.mounts import MOUNT_REGISTRY
from dexengine.core.repo import FULL_ROBOT_URDF_DIR, REPO_ROOT
from dexengine.rl.tasks.scene.split_artic import split_urdf_pair, split_usd_pair
from dexengine.robots import derived_usd
from dexengine.robots.bake_meta import bake_meta_file, baked_urdf_path  # noqa: F401 — re-exported

#: Robot USD cache root (untracked ``data/usd_cache``) — under the data root (``layout.data_root``: env
#: DEXENGINE_MAIN_ROOT), so a code-only worktree reads the main checkout's cache; without one, under this checkout.
#: Read before Kit starts (launcher gate) and at spawn time, so both see the same value.
USD_CACHE_DIR = os.path.join(os.path.normpath(_layout.data_root(str(REPO_ROOT))), "data", "usd_cache")

#: The robot-asset variant override (see the module docstring). ARM_ENV / HAND_ENV come from ``dexengine.core.arms``.
USD_VARIANT_ENV = "DEXENGINE_COMBO_USD_VARIANT"

_said: set[str] = set()


def _say_once(key: str, msg: str) -> None:
    """Announce once per process (no silent fallback, no log flooding)."""
    if key not in _said:
        _said.add(key)
        print(msg, flush=True)


def _usd_literal(arm: str, hand: str, usd_cache_dir: str, variant: str) -> str:
    return f"{usd_cache_dir}/robots/full_robot_gen/{arm}_{hand}{variant}/{arm}_{hand}{variant}.usd"


#: Merged arm + hand URDFs (package assets). The variant suffix is the same string as the USD's (arms.RobotGeneration).
URDF_GEN_DIR = FULL_ROBOT_URDF_DIR


def combo_urdf_path(arm: str, hand: str, urdf_variant_suffix: str) -> pathlib.Path:
    """``<combo><urdf variant>/<combo><urdf variant>.urdf`` — the suffix is ``arms.urdf_variant``'s. Missing = error."""
    d = f"{arm}_{hand}{urdf_variant_suffix}"
    p = URDF_GEN_DIR / d / f"{d}.urdf"
    if not p.is_file():
        raise FileNotFoundError(f"[robot_selection] merged URDF missing: {p}")
    return p


def _require_variant_asset(usd_cache_dir, arm, hand, env, _render):
    """An explicitly named variant must exist on disk (a retired generation name stops here)."""
    if usd_cache_dir is not None:
        _p = _usd_literal(arm, hand, usd_cache_dir, env)
        if not pathlib.Path(_p).is_file():
            raise RuntimeError(
                f"[robot_selection] the asset named by {USD_VARIANT_ENV}={env!r} does not exist:\n"
                f"  looked for : {_p}\n"
                + (
                    f"  this is the render body — bake it: python dexengine/robots/usd/collider_rev.py "
                    f"--arm {arm} --hand {hand} (without a render body that command prints the render_body.py "
                    "command)\n"
                    if _render
                    else ""
                )
                + f"  canonical plate of this (arm, hand): {plate_variant(arm, hand) or '(none)'}\n"
                f"  registered generations: {sorted(ARM_GENERATIONS.get(arm, {}))}\n"
                f"  If a generation was renamed, move every reference to the old name with it.\n"
                f"  To really run an old generation, restore its asset and also set that generation's "
                f"{base_pos_env_name(arm)} (setting only one makes the generation guard stop the run)."
            )


def resolve_usd_variant(arm: str, hand: str, usd_cache_dir: str | None = None) -> str:
    """The asset variant suffix this (arm, hand) spawns — the code default is canonical.

    The environment override distinguishes three states:
      * unset        → the canonical plate (``arms.plate_variant`` — every process spawns it: render_unify);
      * ``=<suffix>`` → that variant, chosen explicitly (the launcher always states the plate this way); a missing
                        asset is an error, there is no other body to spawn instead. For an arm with a generation axis
                        the empty value is such a choice too: the body generation whose suffix is empty;
      * ``=`` (empty) on an arm without a generation axis → that arm's only body.
    Joint angles are relative to the base, so a body of another generation would run without any guard firing and
    only the fingertips would drift — hence the canonical default and the loud messages.
    """
    env = os.environ.get(USD_VARIANT_ENV)
    if env is not None:
        env = env.strip()
        if env or arm in ARM_GENERATIONS:
            # An explicit name is checked for existence too: a script holding a retired generation name must hear at
            # once that the asset no longer exists, not fail later somewhere else.
            if why := retired_plate_why(arm, hand, env):
                raise RuntimeError(f"[robot_selection] {USD_VARIANT_ENV}={env!r}: {why}")
            _render = arm in ARM_RENDER_GEN and env == plate_variant(arm, hand)
            _require_variant_asset(usd_cache_dir, arm, hand, env, _render)
            _say_once(
                f"var:{arm}_{hand}:{env}",
                f"[robot_selection] {'plate' if _render else 'explicit asset variant override'}: "
                f"{arm}_{hand}{env} (canonical plate {arm}_{hand}{plate_variant(arm, hand) or '(none)'})",
            )
        else:
            # An arm without a generation axis (e.g. a floating hand): the empty value is its only body (the
            # canonical answer of `paths --what sim-usd-variant`).
            _say_once(
                f"var:{arm}_{hand}:noaxis",
                f"[robot_selection] {arm}_{hand}: arm without a generation axis — the unsuffixed body is its "
                f"only one ({USD_VARIANT_ENV}='' = the canonical value)",
            )
        return env

    v = plate_variant(arm, hand)
    if not v:
        return ""
    if usd_cache_dir is not None:
        _vp = _usd_literal(arm, hand, usd_cache_dir, v)
        if v != canon_variant(arm) and not pathlib.Path(_vp).is_file():
            # The collider-revision derivative is not baked on this machine yet. Falling back to the body-generation
            # USD would silently run different physics.
            raise RuntimeError(
                f"[robot_selection] USD of the canonical sim variant {arm}_{hand}{v} is missing: {_vp}\n"
                f"  bake it: python dexengine/robots/usd/collider_rev.py --arm {arm} --hand {hand}  "
                f"(the source body {arm}_{hand}{sim_body_variant(arm, hand)} is left as is; without a body the "
                f"command prints how to make one)\n"
                f"  to run the old colliders ON PURPOSE (a comparison), set "
                f"{USD_VARIANT_ENV}={sim_body_variant(arm, hand)}."
            )
        if not pathlib.Path(_vp).is_file():
            raise RuntimeError(
                f"[robot_selection] USD of the canonical variant {arm}_{hand}{v} is missing: {_vp} — there is no "
                f"other body to spawn instead (build: python scripts/build_robot_usd.py)"
            )
    return v


def render_context() -> bool:
    """Whether this process renders = its entry is a render entry (``dexengine.rl.app.KIT_RENDER``: render · replay —
    the launcher names every process's entry). The robot plate cannot tell: every process spawns the same plate
    (``arms.plate_variant``). Read by the object appearance overlay (``dexengine.rl.env_setup.objects``) and the
    one-env render slots (``extra_slots``). A process without an entry dies (``app.entry_of`` — no default).
    The entry prints the marker ([RENDER_CTX] · app.context_line)."""
    from dexengine.rl import app as TK  # noqa: PLC0415 — stdlib only

    return TK.entry_of(os.environ) in TK.KIT_RENDER


def combo_usd_path(arm: str, hand: str, usd_cache_dir: str) -> str:
    """The full-robot USD for a pair. PAIR-scoped on purpose.

    A hand alone cannot answer this — a USD that contains the ARM is a pair artifact (paths.combo_usd_relpath). The
    variant suffix (``resolve_usd_variant``: an alternative asset bake of the same pair, e.g. a different collider
    decomposition) is independent of the data-world suffix ``DEXENGINE_COMBO_VARIANT``.
    """
    return _usd_literal(arm, hand, usd_cache_dir, resolve_usd_variant(arm, hand, usd_cache_dir))


def _urdf_of_baked_usd(usd_path: str) -> str | None:
    """File name of the URDF the USD was baked from (``baked_urdf_path``) — None when absent; the check is then
    skipped."""
    p = baked_urdf_path(usd_path)
    return None if p is None else p.name


def _check_base_pairing(gen, has_axis, arm, hand, variant, base, fell_back, canon, usd_path):
    """Rule 1: the variant's generation and the runtime base pose are a pair (an explicit approval passes loudly)."""
    if gen is None:
        if has_axis:
            _say_once(
                f"gen:{arm}_{hand}:{variant}:unreg",
                f"[robot_selection] ⚠ unregistered asset variant '{variant}' — skipping the generation check. "
                f"The matching base pose is the operator's responsibility "
                f"(the code checks only the variants derived from the arm spec's generations).",
            )
        else:
            # An arm with no generation axis at all (a new arm's first state): skip, but leave a marker.
            _say_once(
                f"gen:{arm}:noaxis",
                f"[robot_selection] ⚠ arm '{arm}' has no generation axis (ARM_GENERATIONS) — skipping the "
                f"asset variant / base pose pairing check. Register it to have the code check it.",
            )
    elif base is not None:
        b = tuple(float(x) for x in base)
        if max(abs(b[k] - gen.base_pos[k]) for k in range(3)) > 1e-6:
            if fell_back:
                remedy = (
                    f"  This pair has NO canonical variant asset ({arm}_{hand}{canon}) — it fell back to the old "
                    f"generation body while the base pose is canonical, so the pair broke.\n"
                    f"  Fix (one of):\n"
                    f"    (a) recommended — bake the canonical generation of this pair "
                    f"(dexengine/robots/urdf/merge.py → USD bake).\n"
                    f"    (b) run the old generation CONSISTENTLY: "
                    f"{base_pos_env_name(arm)}={gen.base_pos[0]},{gen.base_pos[1]},{gen.base_pos[2]}"
                )
            else:
                remedy = (
                    f"  The canonical pair is variant '{canon}' + base "
                    f"{generation_of(arm, canon).base_pos}. To reproduce an old generation, set BOTH "
                    f"DEXENGINE_COMBO_USD_VARIANT and {base_pos_env_name(arm)} to that generation's values "
                    f"(setting only one stops here again)."
                )
            head = (
                f"  USD to spawn : {usd_path}\n"
                f"  its variant  : '{variant}' = chest pitch {gen.chest_pitch_deg:+.2f}°, "
                f"paired base {gen.base_pos}\n"
                f"  runtime base : {b}\n"
            )
            # Reproducing a mismatched pair ON PURPOSE (an A/B that revisits a failure) is legitimate: what is banned
            # is silence, so an explicit approval passes with a large banner in the log.
            if os.environ.get("DEXENGINE_ALLOW_GENERATION_MISMATCH", "0") == "1":
                _say_once(
                    f"gen:{arm}_{hand}:{variant}:allow",
                    "[robot_selection] ⚠⚠ generation mismatch EXPLICITLY APPROVED, continuing "
                    "(DEXENGINE_ALLOW_GENERATION_MISMATCH=1).\n"
                    + head
                    + "  The fingertips of this run are structurally off the reference — "
                    "stop now unless this is the intended A/B.",
                )
            else:
                raise RuntimeError(
                    "[robot_selection] generation mismatch — the asset and the base pose are not a pair.\n"
                    + head
                    + "  Joint angles are RELATIVE to the base, so this pair runs without any error while only the "
                    "fingertips drift (measured p50 12.1 cm) — hence the stop here.\n"
                    + remedy
                    + "\n  If this is an experiment that reproduces a mismatched pair on purpose, approve it "
                    "explicitly with DEXENGINE_ALLOW_GENERATION_MISMATCH=1."
                )


def _check_derived_body(gen, arm, usd_path, variant, hand, usd_cache_dir):
    """A derived body pairs with its source generation's USD (derived_usd.verify)."""
    if gen is not None and gen.derived_from is not None:
        # A derived body: it must pair with its source generation's USD (source md5, generator, the current arm
        # collider digest, and the training body's md5 for a render body) — a stale derivative stops here.
        _mat = generation_of(arm, gen.derived_from).material_of
        prov = derived_usd.verify(
            usd_path,
            variant,
            gen.derived_from,
            _usd_literal(arm, hand, usd_cache_dir, gen.derived_from),
            arm=arm,
            hand=hand,
            material_source_usd=None if _mat is None else _usd_literal(arm, hand, usd_cache_dir, _mat),
        )
        _say_once(
            f"gen:{arm}_{hand}:{variant}:derived",
            f"[robot_selection] derived USD {variant} ← source {gen.derived_from} "
            f"(md5 {prov.get('source_md5', '')[:8]} matches · "
            f"{prov.get('kind', '?')})",
        )


def _check_baked_identity(arm, usd_path, gen, hand, variant):
    """Rule 2: the md5 pins and the URDF the USD was baked from (config.yaml) match the variant."""
    # Identity gate: a body re-baked under the same name differs from the old copy only by md5 (expected values:
    # derived_usd.USD_MD5_PINS). Checked on what is actually spawned (required_robot_files: one merged body for an
    # arm, the per-side split pair for a floating hand, which has no merged body).
    for _pin_usd in required_robot_files(arm, "usd", usd_path):
        _say_once(f"pin:{_pin_usd}", derived_usd.check_pin(_pin_usd))
    if gen is not None:
        src = _urdf_of_baked_usd(usd_path)
        if src is not None:
            want = f"{arm}_{hand}{urdf_variant(arm, variant)}.urdf"
            if src != want:
                raise RuntimeError(
                    f"[robot_selection] asset identity mismatch — this USD was baked from another URDF.\n"
                    f"  USD             : {usd_path}\n"
                    f"  baked source    : {src}   (config.yaml asset_path)\n"
                    f"  expected source : {want}\n"
                    f"  The variant folder name and its content differ — re-bake it."
                )


def assert_generation_coherent(arm: str, hand: str, usd_cache_dir: str) -> None:
    """Check that the asset to spawn and the runtime base pose belong to the same generation, or stop.

    Three axes define the robot's body: the URDF (offline IK, judging), the USD (what the sim spawns) and the base
    pose (both). Only the base axis is stamped into caches (``base_xyz``, guarded at reset); the posture axis has
    no stamp, so a mismatch there would be silent. This checks the pairing contract instead — variant and base pose
    must come from the same generation, so the base stamp protects the posture axis transitively:
      1. (variant ↔ base pose) pairing — mismatch = RuntimeError;
      2. (USD ↔ the URDF it was baked from) — ``config.yaml`` naming another variant = RuntimeError.
    An unregistered variant (experimental assets) cannot be judged: the check is skipped with a loud message.
    """
    spec = ARM_REGISTRY.get(arm)
    base = getattr(spec, "base_pos", None) if spec is not None else None
    variant = resolve_usd_variant(arm, hand, usd_cache_dir)
    usd_path = _usd_literal(arm, hand, usd_cache_dir, variant)

    has_axis = arm in ARM_GENERATIONS
    canon = plate_variant(arm, hand)  # the plate every process should spawn (render body + collider revision)
    # The canonical variant was wanted, another one came out and nobody chose it = fell back for a missing asset.
    fell_back = has_axis and os.environ.get(USD_VARIANT_ENV) is None and variant != canon

    gen = generation_of(arm, variant)
    _check_base_pairing(gen, has_axis, arm, hand, variant, base, fell_back, canon, usd_path)

    _check_derived_body(gen, arm, usd_path, variant, hand, usd_cache_dir)
    _check_baked_identity(arm, usd_path, gen, hand, variant)

    if gen is not None:
        _what = f" · chest pitch {gen.chest_pitch_deg:+.2f}°"
    elif has_axis:
        _what = " (unregistered — generation check skipped)"
    else:
        _what = " (this arm has no torso generation axis)"
    _say_once(
        f"gen:{arm}_{hand}:{variant}:ok",
        f"[robot_selection] spawned robot = {usd_path}\n"
        f"                  variant '{variant}'"
        + _what
        + f" · base {tuple(base) if base is not None else '(spec default)'}",
    )


def required_robot_files(arm: str, kind: str, merged) -> tuple[pathlib.Path, ...]:
    """The robot assets the readiness check requires to exist — the one answer to this question (``combo_ready``).

    ``kind`` = ``"usd"`` (``merged`` = the merged combo USD, ``combo_usd_path``) | ``"urdf"`` (the merged combo URDF).
    A normal arm = the merged body only (the split R/L USDs of split wiring are required separately when the cfg is
    assembled, by ``split_artic.split_usd_paths``). A floating hand (``ArmSpec.is_floating``) = the per-side split
    pair only, it has no merged body. The pair's naming rule belongs to ``split_artic``."""
    if kind not in ("usd", "urdf"):
        raise ValueError(f"[robot_selection] required_robot_files: kind must be usd|urdf (got {kind!r})")
    if not getattr(ARM_REGISTRY.get(arm), "is_floating", False):
        return (pathlib.Path(merged),)
    return (split_usd_pair if kind == "usd" else split_urdf_pair)(mode_canon_path(merged))


def missing_robot_files(arm: str, hand: str, kind: str, merged) -> tuple[list[pathlib.Path], str | None]:
    """(missing files, the message for a floating hand — None for a normal arm, whose caller words its own).
    Both readiness checks stop with this one message (no fallback to the merged body)."""
    missing = [p for p in required_robot_files(arm, kind, merged) if not p.is_file()]
    if not missing or not getattr(ARM_REGISTRY.get(arm), "is_floating", False):
        return missing, None
    how = (
        f"bake (GPU): HAND={hand} bash experiments/ablations/floating_hand/build_floating_usd.sh"
        if kind == "usd"
        else f"generate: python -m experiments.ablations.floating_hand.make_floating_urdf --hand {hand}"
    )
    return missing, (
        f"{_FH_MARK} the per-side split {kind.upper()} of floating hand {arm}_{hand} is missing: "
        + " · ".join(str(p) for p in missing)
        + f" — the merged body ({pathlib.Path(merged).name}) is absent by design (two articulations "
        f"only). {how}"
    )


def combo_ready(arm: str, hand: str, usd_cache_dir: str | None = None) -> tuple[bool, str]:
    """(ready, reason-if-not). The ONE readiness judgment for a pair.

    Checks, in dependency order, naming the FIRST missing artifact so the
    porter knows what to do next rather than reading a wall of state:
      1. specs registered (drafts excluded by design),
      2. mount measured/approved (MOUNT_REGISTRY entry, non-None),
      3. the robot USD actually on disk — the merged combo USD, or for a floating hand the per-side split pair
         (required_robot_files — a floating hand has no merged USD).
    """
    if ARM_REGISTRY.get(arm) is None:
        return False, f"ArmSpec {arm!r} not registered (draft? see arm_specs.py)"
    if HAND_REGISTRY.get(hand) is None:
        return False, f"HandSpec {hand!r} not registered (draft? see core/hands)"
    m = MOUNT_REGISTRY.get((arm, hand))
    if m is None:
        return False, f"no mount for ({arm}, {hand}) — MOUNT_REGISTRY entry is None/absent"
    if usd_cache_dir is not None:
        missing, why = missing_robot_files(arm, hand, "usd", combo_usd_path(arm, hand, usd_cache_dir))
        if missing:
            return False, why or f"combo USD missing: {missing[0]}"
    return True, ""


def combo_data_paths(arm_spec: ArmSpec | None, hand_spec: HandSpec | None) -> dict | None:
    """Per-combo training data / precompute dirs — ONE rule, under the data root.

    Every path is pair-keyed: placement (dx, dy, yaw, dz) is baked into the data per combo, so the trees and the
    wrist FK are combo products. The hand-keyed folders (``robots/<hand>/wrist_poses_*``) are raw staging inputs of
    the placement solver only and are never read by the env. ``DEXENGINE_COMBO_VARIANT`` appends a data-world suffix
    (an alternative preprocessing world of the same pair, e.g. ``_reopt``); unset = "". The dataset and embodiment
    come from ``DEXENGINE_DATASET`` / ``DEXENGINE_EMBODIMENT`` (no defaults). The arm-IK directory suffix is
    appended by the reset event after the side swap, so these names end in ``_right``.

    Layout (relative to the data root; path strings are owned by ``dexengine.core.layout``):
      trajectories : data/<ds>/trees/<arm>_<hand>/<emb>/
      wrist poses  : the combo's precompute folder / wrist_poses_right
      arm-IK cache : the combo's precompute folder / arm_ik_poses_right
    """
    if arm_spec is None or hand_spec is None:
        raise RuntimeError(
            "robot_selection.combo_data_paths: arm_spec/hand_spec is None — the caller must name a registered "
            "(arm, hand); there is no default pair."
        )
    arm, hand = arm_spec, hand_spec
    # A floating hand reads the data world (trees, wrist tables, placement) of the arm body: the same clips, objects
    # and wrist references with only the body swapped. Which arm's world is decided by floating.data_world_arm.
    _data_arm = data_world_arm(arm.name)
    if _data_arm != arm.name:
        print(
            f"{_FH_MARK} data world = trees · wrist tables (incl. placement) of {_data_arm}_{hand.name} — "
            "only the body floats",
            flush=True,
        )
    combo = f"{_data_arm}_{hand.name}{os.environ.get('DEXENGINE_COMBO_VARIANT', '')}"
    try:
        ds, emb = os.environ["DEXENGINE_DATASET"], os.environ["DEXENGINE_EMBODIMENT"]
    except KeyError as e:
        raise RuntimeError(
            f"[robot_selection] {e.args[0]} is not set — the dataset and embodiment axes are explicit "
            f"inputs (no default). Canonical entry: scripts/stage3_rl.sh"
        ) from None
    return {
        "data_dir": _layout.data_path(_layout.render("trees", ds, combo=combo, emb=emb)),
        "wrist_poses_dir_right": _layout.data_path(
            _layout.render("precompute", ds, combo=combo) + "/wrist_poses_right"
        ),
        "arm_ik_poses_dir_right": _layout.data_path(
            _layout.render("precompute", ds, combo=combo) + "/arm_ik_poses_right"
        ),
    }


def resolve_robot_selection() -> tuple[ArmSpec | None, HandSpec | None]:
    """The (arm_spec, hand_spec) pair this process should build, from env vars.

    Never returns None: an unregistered (arm, hand) or a pair without a mount stops here (no silent fallback).
    """
    arm, hand = active_arm_name(), active_hand_name()
    a, h = ARM_REGISTRY.get(arm), HAND_REGISTRY.get(hand)
    missing = [n for n, v in ((arm, a), (hand, h)) if v is None]
    if missing:
        raise RuntimeError(
            f"dexengine.robots.selection: {ARM_ENV}={arm!r} {HAND_ENV}={hand!r} but "
            f"{missing} is/are not registered. Drafts stay out of the registry "
            "until their smoke passes — placeholder geometry must not train."
        )
    if MOUNT_REGISTRY.get((arm, hand)) is None:
        raise RuntimeError(f"dexengine.robots.selection: ({arm}, {hand}) has no mount in MOUNT_REGISTRY.")
    return a, h
