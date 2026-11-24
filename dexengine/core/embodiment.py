"""The embodiment spec loader: hand, arm and arm×hand mount files → HandSpec, ArmSpec and MountSpec.

One file per hand, per arm and per (arm, hand) mount, under one spec root (default ``configs/robots``):

    hands/<hand>.yaml         every HandSpec field, plus ``finger_gains`` (optional rule) and ``rl_recipe``
    arms/<arm>.yaml           every ArmSpec field (``flange``, ``skeleton``, ``torso_ruler`` included), plus the body
                              sections ``generations`` and ``collider_rev`` (arm_generations.body_sections)
    mounts/<arm>+<hand>.yaml  ``arm`` · ``hand`` · ``translation`` · ``rotation`` · every other MountSpec field

The file name is the id: ``hands/xhand_pro.yaml`` must say ``name: xhand_pro``, a mount file must name its arm and
hand. A spec states **every** field of its dataclass (typed construction: dexengine.core.typed_spec — a missing key,
an unknown key or a value of the wrong type stops the load with the file and the key, ``SpecError``).

Rules the loader evaluates (each one replaces a literal that was copied by hand before):

    hand  ``finger_gains: from_effort_cap``  finger_kp / finger_kd = hands.spec.finger_gains_from_cap(finger_effort)
                                             (the two keys are then absent from the file)
    hand  ``rl_recipe.finger_effort: {hardware_cap_times: k}``  = round(k · finger_effort, 6)
    arm   ``base_pos: canon_generation``     = the base of the file's canonical generation (generations.canon),
                                             overridable by env DEXENGINE_<ARM>_BASE_POS together with the variant
    arm   ``skeleton.posture.<joint>: {generation: <field>}``  = that field of the file's canonical generation; the
                                             joint bound to torso3_pitch_rad is ArmSpec.generation_pitch_joint
    mount ``translation: {stack: {adapter_thickness: t}}``
              offset (both sides) = arm.flange.xyz + arm.flange.approach · (t + hand.mount_face_to_root)
              — valid only when the mount rotation maps the hand root's +z onto the flange approach (checked)
    mount ``translation: {measured: {right: xyz, left: xyz}}``  a transform read from a vendor document, as is
    mount ``render_visual_parts[i].ee_to_part: hand_mount_face``  the part frame at the hand's mounting face
              (mounts.parent_link_pose — the vendor link one fixed joint above the hand root)

The marker ``[EMBODIMENT] hand=<h> arm=<a> mount=<a>+<h> spec_sha=<16 hex> root=<dir>`` (stderr, once per combo and
root) names the three files a combo was resolved from; spec_sha is the sha256 of their bytes.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import os
import sys
import typing
from pathlib import Path

import yaml

from dexengine.core.repo import REPO_ROOT
from dexengine.core.typed_spec import SpecError, build, convert  # noqa: F401 — SpecError re-exported (callers catch it)
from dexengine.core.typed_spec import fail as _fail

#: The spec root of this code tree.
SPEC_ROOT = REPO_ROOT / "configs" / "robots"
MARK = "[EMBODIMENT]"
KINDS = ("hands", "arms", "mounts")
#: The rule names a spec file may use in place of a value.
FROM_EFFORT_CAP = "from_effort_cap"
CANON_GENERATION = "canon_generation"
HAND_MOUNT_FACE = "hand_mount_face"
#: |R(q)·z − approach| above this means the stack rule does not apply to the mount rotation.
_AXIS_TOL = 1e-6


# ── files ───────────────────────────────────────────────────────────────────────────────────────────────────────


def read_spec(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        _fail(str(path), f"not valid YAML: {e}")
    if not isinstance(raw, dict):
        _fail(str(path), "the file must hold one mapping")
    return raw


def spec_files(root: Path, kind: str) -> dict[str, Path]:
    d = root / kind
    if not d.is_dir():
        _fail(str(root), f"no {kind}/ directory under the spec root")
    return {p.stem: p for p in sorted(d.glob("*.yaml"))}


def _pop_rule(raw: dict, key: str, where: str, allowed: tuple):
    if key not in raw:
        return None
    v = raw.pop(key)
    if v not in allowed:
        _fail(where, f"{key}: {v!r} is not a known rule (known: {list(allowed)})")
    return v


# ── hands ───────────────────────────────────────────────────────────────────────────────────────────────────────


def _hand(path: Path):
    from dexengine.core.hands import spec as HS  # noqa: PLC0415 — the dataclass module (no registry import)

    where, raw = str(path), dict(read_spec(path))
    recipe_raw = raw.pop("rl_recipe", dataclasses.MISSING)
    if recipe_raw is dataclasses.MISSING:
        _fail(where, "missing key ['rl_recipe'] — null when the hand has no RL recipe")
    if _pop_rule(raw, "finger_gains", where, (FROM_EFFORT_CAP,)) == FROM_EFFORT_CAP:
        for k in ("finger_kp", "finger_kd"):
            if k in raw:
                _fail(where, f"{k} is computed by finger_gains: {FROM_EFFORT_CAP} — state one or the other")
        effort = convert(raw.get("finger_effort"), dict[str, float] | float, f"{where}.finger_effort")
        raw["finger_kp"], raw["finger_kd"] = HS.finger_gains_from_cap(effort)
    spec = build(HS.HandSpec, raw, where)
    if spec.name != path.stem:
        _fail(where, f"name {spec.name!r} ≠ the file name {path.stem!r}")
    recipe = None
    if recipe_raw is not None:
        if not isinstance(recipe_raw, dict):
            _fail(where, "rl_recipe: a mapping of HandRLRecipe fields, or null")
        recipe_raw = dict(recipe_raw)
        fe = recipe_raw.get("finger_effort")
        if isinstance(fe, dict):
            if set(fe) != {"hardware_cap_times"} or not isinstance(spec.finger_effort, float):
                _fail(where, "rl_recipe.finger_effort: {hardware_cap_times: k} needs a scalar finger_effort")
            k = convert(fe["hardware_cap_times"], float, f"{where}.rl_recipe.finger_effort.hardware_cap_times")
            recipe_raw["finger_effort"] = round(k * spec.finger_effort, 6)
        recipe = build(HS.HandRLRecipe, recipe_raw, f"{where}.rl_recipe")
    return spec, recipe


# ── arms ────────────────────────────────────────────────────────────────────────────────────────────────────────


#: The generation field that fingerprints a body (its posture joint's origin pitch) and the ArmSpec field that names
#: that joint (computed here from the posture rule, read by placement.arm_ik.urdf_torso3_pitch).
PITCH_FIELD, PITCH_JOINT = "torso3_pitch_rad", "generation_pitch_joint"


def base_pos_env_name(arm: str) -> str:
    """The env that overrides an arm's canonical base position (set together with DEXENGINE_COMBO_USD_VARIANT)."""
    return f"DEXENGINE_{arm.upper()}_BASE_POS"


def _env_base_pos(default, env: str):
    v = os.environ.get(env, "").strip()
    if not v:
        return default
    xyz = tuple(float(x) for x in v.split(","))
    if len(xyz) != 3:
        raise ValueError(f"{env}: must have the form 'x,y,z' (got {v!r})")
    return xyz


def _arm(path: Path):
    from dexengine.core import arm_generations as AG  # noqa: PLC0415
    from dexengine.core import arms as AS  # noqa: PLC0415

    where, raw = str(path), dict(read_spec(path))
    name = raw.get("name")
    gens, rev = AG.body_sections(raw, where)  # this file's generations and collider revision (popped)
    gen = gens[0] if gens is not None else None
    if raw.get("base_pos") == CANON_GENERATION:
        if gen is None:
            _fail(where, f"base_pos: {CANON_GENERATION} but arm {name!r} has no canonical generation")
        raw["base_pos"] = _env_base_pos(gen.base_pos, base_pos_env_name(name))
    if PITCH_JOINT in raw:
        _fail(where, f"{PITCH_JOINT} is computed from skeleton.posture ({{generation: {PITCH_FIELD}}}) — not a key")
    raw[PITCH_JOINT] = None
    sk = raw.get("skeleton")
    if isinstance(sk, dict) and isinstance(sk.get("posture"), dict):
        posture = {}
        for j, v in sk["posture"].items():
            if isinstance(v, dict):
                if set(v) != {"generation"} or gen is None or not hasattr(gen, v["generation"]):
                    _fail(where, f"skeleton.posture.{j}: {{generation: <field of the canonical generation>}}")
                if v["generation"] == PITCH_FIELD:
                    if raw[PITCH_JOINT] is not None:
                        _fail(where, f"skeleton.posture: two joints bound to {PITCH_FIELD}")
                    raw[PITCH_JOINT] = j
                v = getattr(gen, v["generation"])
            posture[j] = v
        raw["skeleton"] = {**sk, "posture": posture}
    spec = build(AS.ArmSpec, raw, where)
    if spec.name != path.stem:
        _fail(where, f"name {spec.name!r} ≠ the file name {path.stem!r}")
    if spec.torso_ruler != bool(rev and (rev.torso_replace or rev.torso_add)):
        _fail(
            where,
            f"torso_ruler {spec.torso_ruler} disagrees with the collider revision's torso lists "
            "(collider_rev torso_replace · torso_add)",
        )
    return spec


# ── mounts ──────────────────────────────────────────────────────────────────────────────────────────────────────

#: MountSpec fields the loader computes from ``translation`` and ``rotation``.
_MOUNT_COMPUTED = ("ee_to_wrist_offset", "ee_to_wrist_offset_left", "ee_to_wrist_rot_right", "ee_to_wrist_rot_left")
_QUAT = tuple[float, float, float, float]
_XYZ = tuple[float, float, float]


def _rot_z(q) -> tuple[float, float, float]:
    """R(q)·(0, 0, 1) for a wxyz quaternion."""
    w, x, y, z = q
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def _stack_offset(where, t, arm, hand, rot) -> _XYZ:
    if arm.flange is None:
        _fail(where, f"translation.stack needs the arm's flange (arms/{arm.name}.yaml flange)")
    for side, q in rot.items():
        if math.dist(_rot_z(q), arm.flange.approach) > _AXIS_TOL:
            _fail(
                where,
                f"translation.stack: rotation.{side} maps the hand's +z to {_rot_z(q)}, not the flange approach "
                f"{arm.flange.approach} — state the transform with translation.measured instead",
            )
    d = t + hand.mount_face_to_root
    return tuple(p + a * d for p, a in zip(arm.flange.xyz, arm.flange.approach, strict=True))


def _translation(where, tr, arm, hand, rot) -> dict:
    if not isinstance(tr, dict) or len(tr) != 1 or next(iter(tr)) not in ("stack", "measured"):
        _fail(where, "translation: exactly one of {stack: {adapter_thickness: t}} | {measured: {right: , left: }}")
    form, body = next(iter(tr.items()))
    if form == "stack":
        if not isinstance(body, dict) or set(body) != {"adapter_thickness"}:
            _fail(where, "translation.stack: {adapter_thickness: <m>}")
        t = convert(body["adapter_thickness"], float, f"{where}.translation.stack.adapter_thickness")
        return {"ee_to_wrist_offset": _stack_offset(where, t, arm, hand, rot), "ee_to_wrist_offset_left": None}
    if not isinstance(body, dict) or set(body) != {"right", "left"}:
        _fail(where, "translation.measured: {right: [x, y, z], left: [x, y, z]}")
    return {
        "ee_to_wrist_offset": convert(body["right"], _XYZ, f"{where}.translation.measured.right"),
        "ee_to_wrist_offset_left": convert(body["left"], _XYZ, f"{where}.translation.measured.left"),
    }


def _parts(where, parts, kw) -> list:
    from dexengine.core import mounts as MS  # noqa: PLC0415

    if not isinstance(parts, list):
        _fail(where, "render_visual_parts: a list (empty when there is none)")
    out = []
    for i, p in enumerate(parts):
        p = dict(p) if isinstance(p, dict) else p
        if isinstance(p, dict) and p.get("ee_to_part") == HAND_MOUNT_FACE:
            del p["ee_to_part"]
            r, lo = kw["ee_to_wrist_rot_right"], kw["ee_to_wrist_rot_left"]
            off = kw["ee_to_wrist_offset"]
            if r != lo or (kw["ee_to_wrist_offset_left"] not in (None, off)):
                _fail(where, f"render_visual_parts[{i}]: {HAND_MOUNT_FACE} needs one pose for both sides")
            pos, quat = MS.parent_link_pose(off, r, (0.0, 0.0, kw["_mount_face"]))
            p["ee_to_part_pos"], p["ee_to_part_quat"] = pos, quat
        out.append(p)
    return out


def _mount(path: Path, arms: dict, hands: dict):
    from dexengine.core import mounts as MS  # noqa: PLC0415

    where, raw = str(path), dict(read_spec(path))
    for k in ("arm", "hand", "translation", "rotation"):
        if k not in raw:
            _fail(where, f"missing key [{k!r}]")
    arm_id, hand_id = raw["arm"], raw["hand"]
    if path.stem != f"{arm_id}+{hand_id}":
        _fail(where, f"arm+hand {arm_id}+{hand_id} ≠ the file name {path.stem!r}")
    if arm_id not in arms or hand_id not in hands:
        _fail(where, f"unknown arm {arm_id!r} or hand {hand_id!r} (spec files arms/, hands/ of the same root)")
    arm, hand = arms[arm_id], hands[hand_id]
    rot = raw.pop("rotation")
    if not isinstance(rot, dict) or set(rot) != {"right", "left"}:
        _fail(where, "rotation: {right: [w, x, y, z], left: [w, x, y, z]}")
    rot = {s: convert(rot[s], _QUAT, f"{where}.rotation.{s}") for s in ("right", "left")}
    for k in _MOUNT_COMPUTED:
        if k in raw:
            _fail(where, f"{k} is computed from translation/rotation — not a key of a mount file")
    kw = {"ee_to_wrist_rot_right": rot["right"], "ee_to_wrist_rot_left": rot["left"]}
    kw |= _translation(where, raw.pop("translation"), arm, hand, rot)
    parts = _parts(where, raw.pop("render_visual_parts", []), kw | {"_mount_face": hand.mount_face_to_root})
    return build(MS.MountSpec, {**raw, **kw, "render_visual_parts": parts}, where)


# ── catalog ─────────────────────────────────────────────────────────────────────────────────────────────────────


class Catalog:
    """The specs of one root, loaded per kind on first use (an owner module reads only its own kind at import).

    Each kind first imports its owner module (core.hands · core.arms · core.mounts): the owner fills its registry
    from the default catalog at import, so a cold first call lets that import finish and then returns the rows it
    stored, instead of loading the files twice."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def _once(self, name: str, owner: str, make):
        import importlib  # noqa: PLC0415

        importlib.import_module(owner)
        if name not in self.__dict__:
            self.__dict__[name] = make()
        return self.__dict__[name]

    @property
    def _hand_rows(self) -> dict:
        return self._once(
            "_hand_rows_v",
            "dexengine.core.hands",
            lambda: {k: _hand(p) for k, p in spec_files(self.root, "hands").items()},
        )

    @property
    def hands(self) -> dict:
        return {k: v[0] for k, v in self._hand_rows.items()}

    @property
    def recipes(self) -> dict:
        return {k: v[1] for k, v in self._hand_rows.items()}

    @property
    def arms(self) -> dict:
        return self._once(
            "_arms_v", "dexengine.core.arms", lambda: {k: _arm(p) for k, p in spec_files(self.root, "arms").items()}
        )

    @property
    def mounts(self) -> dict:
        return self._once("_mounts_v", "dexengine.core.mounts", self._load_mounts)

    def _load_mounts(self) -> dict:
        out = {}
        for p in spec_files(self.root, "mounts").values():
            m = _mount(p, self.arms, self.hands)
            out[(m.arm, m.hand)] = m
        bare = sorted(h for h in self.hands if not any(k[1] == h for k in out))
        if bare:
            _fail(
                str(self.root),
                f"hand(s) {bare} have no mount file (mounts/<arm>+<hand>.yaml) — a mount is measured or "
                "vendor-defined, never invented",
            )
        return out

    def path(self, kind: str, key: str) -> Path:
        p = self.root / kind / f"{key}.yaml"
        if not p.is_file():
            _fail(str(self.root), f"no {kind} spec {key!r} ({p})")
        return p


@functools.cache
def default_catalog() -> Catalog:
    return Catalog(SPEC_ROOT)


def catalog(root: str | os.PathLike | None = None) -> Catalog:
    """The catalog of ``root`` (None = this tree's spec root)."""
    return default_catalog() if root is None else _catalog_at(str(Path(root).resolve()))


@functools.cache
def _catalog_at(root: str) -> Catalog:
    return Catalog(Path(root))


@dataclasses.dataclass(frozen=True)
class Embodiment:
    """One resolved (arm, hand) combo: the three specs and the identity of their files."""

    arm: typing.Any
    hand: typing.Any
    mount: typing.Any
    spec_sha: str
    root: Path


_SAID: set = set()


def resolve(arm: str, hand: str, root: str | os.PathLike | None = None) -> Embodiment:
    """The (arm, hand) combo of a spec root, with the [EMBODIMENT] marker. A missing spec raises SpecError."""
    cat = catalog(root)
    files = [cat.path("arms", arm), cat.path("hands", hand), cat.path("mounts", f"{arm}+{hand}")]
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.parent.name}/{f.name}".encode() + b"\0" + f.read_bytes() + b"\0")
    emb = Embodiment(cat.arms[arm], cat.hands[hand], cat.mounts[(arm, hand)], h.hexdigest()[:16], cat.root)
    key = (arm, hand, str(cat.root))
    if key not in _SAID:
        _SAID.add(key)
        print(
            f"{MARK} hand={hand} arm={arm} mount={arm}+{hand} spec_sha={emb.spec_sha} root={cat.root}",
            file=sys.stderr,
            flush=True,
        )
    return emb
