"""Switches of the RL task package: one typed object per trainer process (stdlib only).

`RLSwitches` lists every switch with the value the package uses when nothing sets it. A launch states its values in
`configs/rl/recipe.yaml` and with `scripts/stage3_rl.py --set <field>=<value>`; the launcher resolves them per trainer
process, writes them to one JSON file next to the process log and passes only that file's path, in
`DEXENGINE_RL_SWITCHES`, across the process boundary. The trainer entry (`dexengine/rl/cli.py`) calls
`require_file()` before anything reads a switch: without the file it stops, and a switch's environment name left in its
environment with a different value stops it too (that value would not apply). A process that has DEXENGINE_RL_SWITCHES
loads the file on its first switch read, so no read in it falls back to a default.

Each field has an environment name (`env` in the field metadata, `DEXENGINE_<FIELD>`). Recorded launch environments
(stored launch records, the judge sidecar's inherited environment) spell switches that way and the launcher accepts it;
the shell view of the recipe table uses it too. A process without DEXENGINE_RL_SWITCHES (stage-2 shell steps, placement,
CPU tools) reads those names from its environment through the same table.

The file holds every field and the list of fields the launch gave (`given`); a field not given holds its default.
A file written before a switch existed lacks that field. It is read only when the switch declares `before` — the value
that reproduces the behaviour from before it existed (for a new default-off flag: off) — and the reader fills it with
that value, counts it as given and prints `[RL_SWITCHES] … filled <field>=<value> (declared value before the switch
existed)`. A missing field without that declaration fails.
`text(name)` returns a switch in its text form ("1"/"0" for on/off, "" or None when unset) for readers that parse text;
`current()` returns the typed object.

  python3 -m dexengine.rl.tasks.switches --write <file> [--set field=value ...]   # a file for a manual trainer run
  python3 -m dexengine.rl.tasks.switches --show <file>                            # the values and the given fields
  python3 -m dexengine.rl.tasks.switches --to-set K=V ...    # `--set field=value` args of the switch tokens (shells)
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import get_type_hints

#: The one process-boundary variable of the switches: the path of the trainer process's switches file.
SWITCHES_ENV = "DEXENGINE_RL_SWITCHES"
FORMAT = "dexengine.rl.switches/1"
MARK = "[RL_SWITCHES]"
_UNSET = object()


class SwitchError(ValueError):
    """A switch name, value or file the package does not accept."""


def _sw(default, env: str, doc: str, unset=_UNSET, before=_UNSET):
    """A switch field: `env` = its environment name, `unset` = the text a reader gets when the switch is not set
    (default: the default rendered as text), `before` = the value that reproduces the behaviour from before the switch
    existed (declared only by switches added after records were written; module doc)."""
    return dataclasses.field(default=default, metadata={"env": env, "doc": doc, "unset": unset, "before": before})


@dataclasses.dataclass(frozen=True)
class RLSwitches:
    """Every switch of the RL task package; the defaults are the values used when a launch does not set them."""

    # plugins (the arm and hand ids belong to dexengine.core.config.ENV_VARS; arm 'floating' needs the plugin)
    plugins: str = _sw("", "DEXENGINE_PLUGINS", "comma-separated experiment modules imported before any cfg (hooks)")
    fh_nan_reset: bool = _sw(
        False, "DEXENGINE_FH_NAN_RESET", "floating hand: reset envs whose reward turns NaN", unset=""
    )
    fh_palm_point: str = _sw("", "DEXENGINE_FH_PALM_POINT", "floating hand: 'arm_ee' = palm at the arm-body EE point")
    right_park: bool = _sw(False, "DEXENGINE_RIGHT_PARK", "the right arm and fingers hold their default pose")
    # arm control
    arm_kp: float | None = _sw(
        None, "DEXENGINE_ARM_KP", "arm joint stiffness override (None = cfg default)", unset=None
    )
    arm_kd: float | None = _sw(None, "DEXENGINE_ARM_KD", "arm joint damping override (None = cfg default)", unset=None)
    left_arm_kp_scale: float | None = _sw(
        None, "DEXENGINE_LEFT_ARM_KP_SCALE", "scale of the left arm stiffness in bimanual runs (None = 1)", unset=""
    )
    arm_ff: bool = _sw(True, "DEXENGINE_ARM_FF", "arm feed-forward (gravity compensation + reference velocity lead)")
    arm_ff_mode: str = _sw("lead", "DEXENGINE_ARM_FF_MODE", "feed-forward channel: lead | vel | torque | both")
    arm_ff_vel_scale: float = _sw(1.0, "DEXENGINE_ARM_FF_VEL_SCALE", "scale of the reference velocity fed forward")
    arm_joint_residual: bool = _sw(
        False,
        "DEXENGINE_ARM_JOINT_RESIDUAL",
        "ablation: the policy outputs an arm-joint residual instead of a wrist one",
    )
    arm_joint_residual_rad: float = _sw(0.15, "DEXENGINE_ARM_JOINT_RESIDUAL_RAD", "ablation: arm-joint residual cap")
    action_abs: bool = _sw(False, "DEXENGINE_ACTION_ABS", "ablation: absolute wrist and finger targets, no residuals")
    zero_residual: bool = _sw(False, "DEXENGINE_ZERO_RESIDUAL", "eval/render: force the learned residual to 0")
    arm_ik_reset_suffix: str = _sw(
        "", "DEXENGINE_ARM_IK_RESET_SUFFIX", "ablation: reset from an arm IK cache solved per frame (directory suffix)"
    )
    # robot placement
    envz: bool = _sw(True, "DEXENGINE_ENVZ", "per-env robot height offset (desk-relative) applied at reset")
    robot_z_off: float = _sw(0.0, "DEXENGINE_ROBOT_Z_OFF", "robot root height offset (m)", unset="0")
    mount_tol: float = _sw(0.0025, "DEXENGINE_MOUNT_TOL", "tolerance (m) of the base height in the arm IK cache")
    # collision
    coll_preset: str = _sw("", "DEXENGINE_COLL_PRESET", "collision preset name of the launch (canon_r2r3off, ...)")
    bi_split_artic: bool = _sw(False, "DEXENGINE_BI_SPLIT_ARTIC", "bimanual: one articulation per arm")
    bi_split_off: str = _sw("", "DEXENGINE_BI_SPLIT_OFF", "split articulation: rows authored off (subset of r2,r3l)")
    bi_selfcoll: bool = _sw(False, "DEXENGINE_BI_SELFCOLL", "legacy bimanual self-collision filter scene")
    desk_robot_nocoll: bool = _sw(False, "DEXENGINE_DESK_ROBOT_NOCOLL", "no robot <-> desk collision")
    arm_obj_nocoll: bool = _sw(False, "DEXENGINE_ARM_OBJ_NOCOLL", "no arm <-> object collision")
    eval_cross_off: bool = _sw(False, "DEXENGINE_EVAL_CROSS_OFF", "eval: cross-hand and cross-object pairs off")
    collcurr_cats: str = _sw("hh,ho,oo", "DEXENGINE_COLLCURR_CATS", "collision curriculum categories")
    all_obj_collision: bool = _sw(
        False, "DEXENGINE_ALL_OBJ_COLLISION", "every object collides (policy scenes)", unset=""
    )
    full_physics: bool = _sw(
        False,
        "DEXENGINE_FULL_PHYSICS",
        "every slot collides with the robot and the main objects (delivered scene · physical-validity judge)",
        before=False,
    )
    # objects
    obj_spawn_hold: int = _sw(0, "DEXENGINE_OBJ_SPAWN_HOLD", "steps after reset with objects pinned to the reference")
    obj_ref_vel_init: bool = _sw(False, "DEXENGINE_OBJ_REF_VEL_INIT", "objects start with the reference velocity")
    # physics
    physx_lite: bool = _sw(False, "DEXENGINE_PHYSX_LITE", "small PhysX GPU buffers (1-few env renders and checks)")
    physx_cap_auto: bool = _sw(False, "DEXENGINE_PHYSX_CAP_AUTO", "contact / patch caps from the card size")
    physx_contact_cap_log2: int | None = _sw(
        None, "DEXENGINE_PHYSX_CONTACT_CAP_LOG2", "contact stream cap, log2 (None = derived)", unset=""
    )
    physx_patch_cap_log2: int | None = _sw(
        None, "DEXENGINE_PHYSX_PATCH_CAP_LOG2", "patch stream cap, log2 (None = derived)", unset=""
    )
    disable_robot_gravity: bool = _sw(
        False, "DEXENGINE_DISABLE_ROBOT_GRAVITY", "audit run with robot gravity off", unset=None
    )
    # gates and diagnostics
    left_gate_ramp: str = _sw("", "DEXENGINE_LEFT_GATE_RAMP", "left object gate ramp start 'm,deg' (empty = constant)")
    objtrack_debug: bool = _sw(False, "DEXENGINE_OBJTRACK_DEBUG", "per-step env trace of the object gate", unset=None)
    # rendering
    camera_res: int = _sw(1920, "DEXENGINE_CAMERA_RES", "head camera square resolution")
    camera_focal_mm: float = _sw(8.0, "DEXENGINE_CAMERA_FOCAL_MM", "head camera focal length (mm)")
    key_light: str = _sw("", "DEXENGINE_KEY_LIGHT", "render key light 'intensity[,r,g,b]' (empty = none)")


FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(RLSwitches))
_META = {f.name: f.metadata for f in dataclasses.fields(RLSwitches)}
# _META holds dataclass field metadata (mappingproxy — deepcopy cannot copy it) and the _UNSET sentinel by
# identity; it is derived from the class definition, so the resident bake worker keeps it as it is
# (placement/bake/server.py rule 2 · a deep-copied restore would also break the `is _UNSET` checks below).
_SERVER_KEEP = ("_META",)
_TYPES = get_type_hints(RLSwitches)
_DEFAULTS = {f.name: f.default for f in dataclasses.fields(RLSwitches)}
#: field -> environment name, and back.
ENV_NAME: dict[str, str] = {k: m["env"] for k, m in _META.items()}
FIELD_OF: dict[str, str] = {v: k for k, v in ENV_NAME.items()}
#: {field: its value before it existed} — the switches a record may lack (module doc).
BEFORE: dict[str, object] = {k: m["before"] for k, m in _META.items() if m["before"] is not _UNSET}


def _base(field: str):
    tp = _TYPES[field]
    return (tp.__args__[0], True) if getattr(tp, "__args__", None) else (tp, False)


def render(field: str, value) -> str | None:
    """A field value in its text form."""
    tp, _opt = _base(field)
    if value is None:
        unset = _META[field]["unset"]
        return "" if unset is _UNSET else unset
    if tp is bool:
        return "1" if value else "0"
    if tp is float:
        return repr(float(value))
    return str(value)


def unset_text(field: str) -> str | None:
    """The text a reader gets when the switch is not set."""
    unset = _META[field]["unset"]
    return render(field, _DEFAULTS[field]) if unset is _UNSET else unset


def parse(field: str, value) -> object:
    """`value` (text form, or a JSON / YAML scalar) as the field's type. Empty text means off for an on/off
    switch, None for an optional one and the default otherwise; anything else that does not fit fails."""
    if field not in _TYPES:
        raise SwitchError(f"{MARK} unknown switch {field!r}; known: {', '.join(FIELDS)}")
    tp, opt = _base(field)
    where = f"{MARK} {field} ({ENV_NAME[field]})"
    if value is None:
        if opt:
            return None
        raise SwitchError(f"{where}: null is not allowed")
    if isinstance(value, str):
        s = value.strip()
        if tp is str:
            return value
        if s == "":
            return False if tp is bool else (None if opt else _DEFAULTS[field])
        if tp is bool:
            if s.lower() in ("1", "true"):
                return True
            if s.lower() in ("0", "false"):
                return False
            raise SwitchError(f"{where}: {value!r} is not 0 | 1")
        try:
            return int(s) if tp is int else float(s)
        except ValueError:
            raise SwitchError(f"{where}: {value!r} is not {'an integer' if tp is int else 'a number'}") from None
    if tp is bool and isinstance(value, bool):
        return value
    if tp is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise SwitchError(f"{where}: {value!r} is not of type {tp.__name__}")


def check(sw: RLSwitches) -> RLSwitches:
    """The value rules the readers apply, checked once when the object is built."""
    if sw.obj_spawn_hold < 0:
        raise SwitchError(f"{MARK} obj_spawn_hold={sw.obj_spawn_hold} — integers >= 0 only")
    if sw.camera_res <= 0:
        raise SwitchError(f"{MARK} camera_res={sw.camera_res} — a positive integer")
    return sw


def build(values: Mapping[str, object] = (), base: RLSwitches | None = None) -> RLSwitches:
    """`base` (default: the defaults) with `values` ({field: value}) parsed onto it; unknown fields fail."""
    changes = {k: parse(k, v) for k, v in dict(values).items()}
    return check(dataclasses.replace(base or RLSwitches(), **changes))


def from_env_names(texts: Mapping[str, str]) -> tuple[RLSwitches, frozenset[str]]:
    """(object, given fields) from {environment name: text}; a name that is not a switch fails."""
    bad = sorted(set(texts) - set(FIELD_OF))
    if bad:
        raise SwitchError(f"{MARK} {bad} are not switches (dexengine.rl.tasks.switches.ENV_NAME)")
    given = {FIELD_OF[k]: v for k, v in texts.items()}
    return build(given), frozenset(given)


def env_texts(sw: RLSwitches, given) -> dict[str, str]:
    """{environment name: text} of the given fields."""
    return {ENV_NAME[f]: render(f, getattr(sw, f)) for f in FIELDS if f in given}


# -- the file ------------------------------------------------------------------------------------------------------


def write(path, sw: RLSwitches, given=()) -> Path:
    """Write the switches file (every field + the given ones) and return its path."""
    given = sorted(set(given))
    bad = sorted(set(given) - set(FIELDS))
    if bad:
        raise SwitchError(f"{MARK} given fields {bad} are not switches")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"format": FORMAT, "values": {f: getattr(sw, f) for f in FIELDS}, "given": given}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def _fill_before(p: Path, vals: dict, given: list) -> tuple[dict, list]:
    """The fields of BEFORE a record lacks, filled with their declared value and counted as given (with a marker)."""
    fill = sorted(set(BEFORE) - set(vals))
    for k in fill:
        print(f"{MARK} {p}: filled {k}={BEFORE[k]!r} (declared value before the switch existed)", flush=True)
    return {**vals, **{k: BEFORE[k] for k in fill}}, [*given, *fill]


def read(path) -> tuple[RLSwitches, frozenset[str]]:
    """(object, given fields) of a switches file. A missing file, another format, an unknown or a missing field, a
    value of the wrong type, or a field that is not given but differs from its default fails."""
    p = Path(path)
    if not p.is_file():
        raise SwitchError(f"{MARK} switches file not found: {p}")
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SwitchError(f"{MARK} {p}: not JSON ({e})") from None
    if not isinstance(body, dict) or body.get("format") != FORMAT:
        raise SwitchError(f"{MARK} {p}: format is not {FORMAT}")
    vals, given = body.get("values"), body.get("given")
    if not isinstance(vals, dict) or not isinstance(given, list):
        raise SwitchError(f"{MARK} {p}: needs 'values' (mapping) and 'given' (list)")
    unknown, missing = sorted(set(vals) - set(FIELDS)), sorted(set(FIELDS) - set(vals) - set(BEFORE))
    if unknown or missing or set(given) - set(FIELDS):
        raise SwitchError(f"{MARK} {p}: unknown {unknown + sorted(set(given) - set(FIELDS))} · missing {missing}")
    vals, given = _fill_before(p, vals, given)
    for k, v in vals.items():
        if isinstance(v, str) and _base(k)[0] is not str:
            raise SwitchError(f"{MARK} {p}: {k}={v!r} — the file holds typed values, not text")
    sw = build(vals)
    odd = [f for f in FIELDS if f not in given and getattr(sw, f) != _DEFAULTS[f]]
    if odd:
        raise SwitchError(f"{MARK} {p}: {odd} differ from their defaults but are not given")
    return sw, frozenset(given)


def derive(path, changes: Mapping[str, object], out) -> str:
    """Write `out` = the switches file `path` with `changes` ({field: value}) applied and marked given; returns `out`.
    For a session that replays a recorded trainer dump with a few switches changed."""
    sw, given = read(path)
    write(out, build(changes, sw), given | set(changes))
    return str(out)


# -- this process ---------------------------------------------------------------------------------------------------

_STATE: dict = {"path": None, "sw": None, "given": frozenset(), "file": None, "same": set(), "env_reads": {}}


def same(field: str, a, b) -> bool:
    """Two values of one switch (text or typed; None = unset) mean the same."""
    if a == b:
        return True
    opt = _base(field)[1]
    a, b = ("" if x is None and not opt else x for x in (a, b))
    try:
        return parse(field, a) == parse(field, b)
    except SwitchError:
        return False


def _stale_env(sw: RLSwitches, environ: Mapping[str, str]) -> list[str]:
    out = []
    for f in FIELDS:
        name = ENV_NAME[f]
        if name in environ and not same(f, environ[name], getattr(sw, f)):
            out.append(f"{name}={environ[name]!r} (file: {render(f, getattr(sw, f))!r})")
    return out


def require_file(environ: Mapping[str, str] | None = None) -> str:
    """Load this trainer process's switches file (named by DEXENGINE_RL_SWITCHES) and return the marker line.

    Stops when the variable is unset, the file is bad, a switch was read from the environment earlier with a value
    the file contradicts, or a switch's environment name carries a value that differs from the file. Loading the same
    file again is a no-op; a second file must hold the same switches."""
    environ = os.environ if environ is None else environ
    path = environ.get(SWITCHES_ENV, "")
    if loaded() and path:
        _check_same_file(path)
        return f"{MARK} {_STATE['path']} · given {len(_STATE['given'])}: {', '.join(sorted(_STATE['given'])) or '-'}"
    if not path:
        raise SystemExit(
            f"{MARK} {SWITCHES_ENV} is not set: the trainer reads its switches from the file the launcher "
            "writes (scripts/stage3_rl.py); for a manual run write one with "
            "`python3 -m dexengine.rl.tasks.switches --write <file>`"
        )
    try:
        sw, given = read(path)
    except SwitchError as e:
        raise SystemExit(str(e)) from None
    early = [f"{n}={v!r}" for n, v in _STATE["env_reads"].items() if not same(FIELD_OF[n], v, getattr(sw, FIELD_OF[n]))]
    stale = _stale_env(sw, environ)
    if early or stale:
        raise SystemExit(
            f"{MARK} values outside the switches file {path}: read before it was loaded {early} · in the "
            f"environment {stale} — set switches with `scripts/stage3_rl.py --set <field>=<value>` (or "
            "switches.derive for a replayed dump)"
        )
    _STATE.update(path=str(path), sw=sw, given=given, file=(sw, given), same={str(path)})
    return f"{MARK} {path} · given {len(given)}: {', '.join(sorted(given)) or '-'}"


def loaded() -> bool:
    """True once this process runs on a switches file."""
    return _STATE["sw"] is not None


def _check_same_file(path: str) -> None:
    """A process that serves several dumps (a persistent render session) may name another file with equal content."""
    if path in _STATE["same"]:
        return
    try:
        other = read(path)
    except SwitchError as e:
        raise SystemExit(str(e)) from None
    if other != _STATE["file"]:
        raise SystemExit(
            f"{MARK} {SWITCHES_ENV}={path} holds other switches than the loaded {_STATE['path']}: one "
            "process runs on one set of switches"
        )
    _STATE["same"].add(path)


def _ensure(environ: Mapping[str, str] | None) -> None:
    """Load the file DEXENGINE_RL_SWITCHES names on the first read (and check a changed name)."""
    path = (os.environ if environ is None else environ).get(SWITCHES_ENV, "")
    if path and not loaded():
        require_file(environ)
    elif path:
        _check_same_file(path)


def override(**changes) -> None:
    """Change switches of this process after the file was loaded (the trainer entry derives right_park from --side)."""
    if not loaded():
        raise SwitchError(f"{MARK} override({sorted(changes)}) before the switches file was loaded (require_file)")
    _STATE["sw"] = build(changes, _STATE["sw"])
    _STATE["given"] = _STATE["given"] | set(changes)


def current(environ: Mapping[str, str] | None = None) -> RLSwitches:
    """The typed switches of this process: its switches file, else the environment names (processes without a file)."""
    _ensure(environ)
    if loaded():
        return _STATE["sw"]
    environ = os.environ if environ is None else environ
    return build({f: environ[ENV_NAME[f]] for f in FIELDS if ENV_NAME[f] in environ})


def text(name: str, default=_UNSET, environ: Mapping[str, str] | None = None) -> str | None:
    """A switch by its environment name, in its text form; `default` is returned when the switch is not set (not given
    in the file, or not in the environment of a process without a file)."""
    if name not in FIELD_OF:
        raise SwitchError(f"{MARK} {name} is not a switch of the RL task package")
    f = FIELD_OF[name]
    _ensure(environ)
    if loaded():
        if f in _STATE["given"]:
            return render(f, getattr(_STATE["sw"], f))
        return unset_text(f) if default is _UNSET else default
    environ = os.environ if environ is None else environ
    val = environ.get(name, unset_text(f) if default is _UNSET else default)
    _STATE["env_reads"].setdefault(name, environ.get(name, unset_text(f)))
    return val


def legacy_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The process environment with the given switches under their environment names (for readers that take one)."""
    out = dict(os.environ if environ is None else environ)
    if loaded():
        out = {k: v for k, v in out.items() if k not in FIELD_OF}
        out.update(env_texts(_STATE["sw"], _STATE["given"]))
    return out


def dump_env_view(env: Mapping[str, str]) -> dict[str, str]:
    """A recorded trainer environment with the switches file replaced by the given switches under their environment
    names (for comparing or keying dumps by their switch values instead of the file path)."""
    out = {k: v for k, v in env.items() if k != SWITCHES_ENV}
    if env.get(SWITCHES_ENV):
        sw, given = read(env[SWITCHES_ENV])
        out.update(env_texts(sw, given))
    return out


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Write or show an RL switches file.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="FILE", help="write the defaults with the --set values given")
    g.add_argument("--show", metavar="FILE", help="print the values (given ones marked *)")
    g.add_argument(
        "--to-set",
        nargs="*",
        metavar="K=V",
        help="print one `--set field=value` per line for the tokens that name a switch (others: nothing)",
    )
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    a = ap.parse_args(argv)
    try:
        if a.to_set is not None:
            for kv in a.to_set:
                k, _, v = kv.partition("=")
                if k in FIELD_OF:
                    parse(FIELD_OF[k], v)
                    print(f"--set\n{FIELD_OF[k]}={v}")
            return 0
        if a.write:
            vals = dict(kv.partition("=")[::2] for kv in a.set)
            write(a.write, build(vals), vals)
            print(f"{MARK} wrote {a.write} · given {sorted(vals) or '-'}")
        else:
            sw, given = read(a.show)
            for f in FIELDS:
                print(f"{'*' if f in given else ' '} {f} = {getattr(sw, f)!r}")
    except SwitchError as e:
        print(e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
