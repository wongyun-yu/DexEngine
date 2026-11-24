"""Typed configuration: dataclasses filled from `configs/*.yaml`, command-line overrides and a few environment
variables.

A configuration section is a (frozen or plain) dataclass whose field defaults are the values the pipeline uses when
nothing is configured. A section is read with `load`::

    from dataclasses import dataclass
    from dexengine.core import config

    @dataclass(frozen=True)
    class RenderConfig:
        width: int = 640
        fps: float = 30.0
        looks: tuple[str, ...] = ("orig",)

    cfg = config.load(RenderConfig, "record/render")          # configs/record/render.yaml, if present
    cfg = config.load(RenderConfig, "record/render", overrides=["width=320", "looks=[orig,aug]"])

Sources, lowest to highest priority:
  1. the dataclass defaults;
  2. the YAML file `configs/<name>.yaml` (or an explicit path); a missing default file is not an error,
     a missing explicit file is;
  3. overrides `key=value` (a dotted key reaches a nested dataclass field; the value is parsed as YAML, so
     `true`, `3`, `0.5`, `[a, b]` and `{k: v}` are typed); usually given on the command line with `--set`.

Every key is checked against the dataclass: an unknown key or a value of the wrong type raises `ConfigError`. Values
are converted to the declared field type (`int`, `float`, `bool`, `str`, `Path`, `tuple[...]`, `list[...]`,
`dict[...]`, `Optional[...]`, `Literal[...]` and nested dataclasses).

Command line::

    parser = argparse.ArgumentParser()
    config.add_cli(parser)              # adds --config FILE and --set KEY=VALUE (repeatable)
    args = parser.parse_args()
    cfg = config.from_args(RenderConfig, "record/render", args)

Environment variables. Configuration does not come from the environment, with one exception: values a child process
(an Isaac Lab worker, a shell stage) has to inherit. Those are listed in `ENV_VARS`, each with its meaning; read them
with `env(name)` so the list stays the only statement of which variables exist.

`dump(obj)` returns a plain dict (YAML/JSON-ready) and `to_yaml(obj)` its YAML text, for recording the effective
configuration next to an output.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import types
import typing
from pathlib import Path
from typing import Any

import yaml

from dexengine.core.repo import CONFIGS_DIR

#: Environment variables that cross a process boundary. Name → meaning. Anything else is configuration (YAML + CLI).
ENV_VARS: dict[str, str] = {
    "DEXENGINE_MAIN_ROOT": "data root: the folder that holds data/, precompute/, overfit/, logs/ "
    "(set by --data-root; unset = configs/layout.yaml data_root, else the code checkout)",
    "DEXENGINE_ARM": "arm id of the running process (registry: dexengine.core.arms.ARM_REGISTRY)",
    "DEXENGINE_HAND": "hand id of the running process (registry: dexengine.core.hands.HAND_REGISTRY)",
    "DEXENGINE_DATASET": "dataset id of the running process (registry: dexengine.core.layout.DATASETS)",
    "DEXENGINE_EMBODIMENT": "embodiment of the running process: right, left or bimanual",
    "DEXENGINE_LAYOUT_CONFIG": "layout configuration file used instead of configs/layout.yaml (a site's storage "
    "roots; every child process must see the same file)",
    "DEXENGINE_SEQ_LIST": "file listing the clip folders of the running process (set by --seq-list; record sessions "
    "replace it per session)",
    "DEXENGINE_ROSTER": "named roster of the clip folder (skips the roster check when it matches)",
    "DEXENGINE_ARM_IK_DIR_SUFFIX": "suffix of the per-clip arm IK cache directories (owner "
    "dexengine.core.locations.names.ik_suffix; unset = _cont25)",
    "DEXENGINE_RL_SWITCHES": "path of the RL trainer process's switches file (written by dexengine.rl.launch; "
    "fields: dexengine.rl.tasks.switches.RLSwitches)",
    "DEXENGINE_MEM_GIB": "memory budget [GiB] process pools plan for when the machine's limit cannot be read "
    "(a container hiding its cgroup, a Slurm job whose request is not in memory.max); owner dexengine.core.capacity",
    "DEXENGINE_POOL_LOCK": "reach maps used as locked inputs although their wrist inputs are stale or unrecorded: "
    "comma list of <map dir>@<12 hex of sha256(meta.npz)> (owner dexengine.placement.reach_map.store; unset = none)",
}


class ConfigError(ValueError):
    """A configuration file, override or value that does not match its dataclass."""


def env(name: str, default: str = "") -> str:
    """The value of a process-boundary environment variable (must be listed in `ENV_VARS`)."""
    if name not in ENV_VARS:
        raise ConfigError(f"{name} is not a process-boundary variable (dexengine.core.config.ENV_VARS)")
    return os.environ.get(name, default)


def config_path(name: str) -> Path:
    """`configs/<name>.yaml` for a section name such as `layout` or `record/render`."""
    return CONFIGS_DIR / f"{name}.yaml"


def read_yaml(path: str | os.PathLike) -> dict:
    """The mapping stored in a YAML file (an empty file is an empty mapping)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping, got {type(data).__name__}")
    return data


def parse_override(text: str) -> tuple[list[str], Any]:
    """`a.b=value` → (["a", "b"], value parsed as YAML)."""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"override {text!r}: expected KEY=VALUE")
    try:
        value = yaml.safe_load(raw) if raw.strip() else ""
    except yaml.YAMLError as e:
        raise ConfigError(f"override {text!r}: value is not valid YAML ({e})") from e
    return key.split("."), value


def _nest(keys: list[str], value: Any) -> dict:
    out: dict = {}
    cur = out
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return out


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _type_name(tp) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _coerce_union(value, args, where):
    """Union/Optional: the first member type that accepts the value."""
    if value is None and type(None) in args:
        return None
    errors = []
    for a in args:
        if a is type(None):
            continue
        try:
            return coerce(value, a, where)
        except ConfigError as e:
            errors.append(str(e))
    raise ConfigError(f"{where}: {value!r} matches none of {[_type_name(a) for a in args]}")


def _coerce_sequence(value, where, args, origin):
    """list/tuple (fixed-length tuples check the item count)."""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected a list, got {value!r}")
    if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
        if len(args) != len(value):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        items = [coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True))]
    else:
        item_tp = args[0] if args else Any
        items = [coerce(v, item_tp, f"{where}[{i}]") for i, v in enumerate(value)]
    return tuple(items) if origin is tuple else items


def _coerce_mapping(value, where, args):
    """dict with typed keys and values."""
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {value!r}")
    kt, vt = args if args else (Any, Any)
    return {coerce(k, kt, f"{where} key"): coerce(v, vt, f"{where}[{k}]") for k, v in value.items()}


def _coerce_scalar(tp, value, where):
    """bool/int/float/str/Path and plain classes (no implicit bool→int)."""
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected true/false, got {value!r}")
    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    if tp is str:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
        raise ConfigError(f"{where}: expected a string, got {value!r}")
    if tp is Path:
        if isinstance(value, (str, os.PathLike)):
            return Path(value)
        raise ConfigError(f"{where}: expected a path, got {value!r}")
    if isinstance(tp, type) and isinstance(value, tp):
        return value
    raise ConfigError(f"{where}: cannot convert {value!r} to {_type_name(tp)}")


def coerce(value: Any, tp: Any, where: str = "value") -> Any:
    """`value` converted to the annotated type `tp` (raises ConfigError when it cannot be)."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if tp is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        return _coerce_union(value, args, where)
    if origin is typing.Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of {list(args)}")
        return value
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping for {tp.__name__}, got {value!r}")
        return build(tp, value, where)
    if origin in (tuple, list):
        return _coerce_sequence(value, where, args, origin)
    if origin is dict:
        return _coerce_mapping(value, where, args)
    return _coerce_scalar(tp, value, where)


def build(cls, data: dict, where: str = "") -> Any:
    """An instance of dataclass `cls` from a mapping (unknown keys raise; missing keys keep their defaults)."""
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where or cls.__name__}: unknown key(s) {unknown}; known: {sorted(fields)}")
    kwargs = {k: coerce(v, hints[k], f"{where}.{k}" if where else k) for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where or cls.__name__}: {e}") from e


def dump(obj) -> Any:
    """A dataclass instance (or nested values) as plain YAML/JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: dump(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [dump(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def to_yaml(obj) -> str:
    """YAML text of `dump(obj)`."""
    return yaml.safe_dump(dump(obj), sort_keys=False, allow_unicode=True)


def load(cls, name: str | None = None, *, path: str | os.PathLike | None = None, overrides=()) -> Any:
    """Dataclass `cls` filled from its YAML file and `overrides` (see the module docstring for the order).

    `name` selects `configs/<name>.yaml` (skipped when the file does not exist); `path` selects an explicit file
    (which must exist) instead."""
    data: dict = {}
    if path is not None:
        if not os.path.isfile(path):
            raise ConfigError(f"config file not found: {path}")
        data = read_yaml(path)
    elif name is not None and config_path(name).is_file():
        data = read_yaml(config_path(name))
    for text in overrides:
        keys, value = parse_override(text)
        data = _merge(data, _nest(keys, value))
    return build(cls, data, name or cls.__name__)


def add_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add `--config FILE` and `--set KEY=VALUE` (repeatable) to an argparse parser."""
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="YAML configuration file (default: the section's configs/<name>.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="config_set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one configuration key (dotted for nested keys; value parsed as YAML)",
    )
    return parser


def from_args(cls, name: str | None, args: argparse.Namespace) -> Any:
    """`load` with the file and overrides given on a command line built with `add_cli`."""
    return load(cls, name, path=getattr(args, "config", None), overrides=getattr(args, "config_set", ()) or ())


def add_data_root_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add `--data-root DIR` (the data root; see `apply_data_root`). Every stage entry point declares it through this
    function and calls `apply_data_root` right after parsing, before it reads any path. The entry modules read no data
    root at import time (tests/core/test_data_root_cli.py), so a value applied after parsing reaches every reader."""
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help="data root (holds data/, precompute/, overfit/, logs/); default: $DEXENGINE_MAIN_ROOT, "
        "else configs/layout.yaml data_root, else the code checkout",
    )
    return parser


def apply_data_root(args_or_path) -> str:
    """Make a `--data-root` value effective for this process and its children (it is exported as
    DEXENGINE_MAIN_ROOT, the process-boundary variable). Returns the value ("" = nothing given, nothing changed)."""
    root = (
        getattr(args_or_path, "data_root", args_or_path)
        if not isinstance(args_or_path, (str, os.PathLike))
        else args_or_path
    )
    if root:
        os.environ["DEXENGINE_MAIN_ROOT"] = os.path.abspath(os.fspath(root))
        return os.environ["DEXENGINE_MAIN_ROOT"]
    return ""


def take_data_root(argv: list[str]) -> list[str]:
    """For a dispatcher that hands the rest of its command line to another parser: apply a `--data-root DIR` in `argv`
    (`apply_data_root`) and return `argv` without it. Other options pass through unchanged (no abbreviations)."""
    pre = add_data_root_arg(argparse.ArgumentParser(add_help=False, allow_abbrev=False))
    ns, rest = pre.parse_known_args(argv)
    apply_data_root(ns)
    return rest
