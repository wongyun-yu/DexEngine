"""Typed construction of frozen dataclasses from parsed spec data (YAML): ``build(cls, raw, where)``.

Every field of ``cls`` must be stated in ``raw`` and nothing else (a missing or unknown key raises ``SpecError`` naming
``where`` and the key); each value is checked against the field's type hint and converted: a sequence becomes a tuple,
an integer becomes a float where the field is a float, a mapping becomes a nested dataclass. A value equal to the
field's declared default is that default. The embodiment loader (dexengine.core.embodiment) is the reader.
"""

from __future__ import annotations

import dataclasses
import types
import typing

MARK = "[SPEC]"


class SpecError(ValueError):
    """Spec data that does not match its dataclass schema."""


def fail(where: str, msg: str) -> typing.NoReturn:
    raise SpecError(f"{MARK} {where}: {msg}")


def _is_dataclass_type(t) -> bool:
    return isinstance(t, type) and dataclasses.is_dataclass(t)


def _union_args(hint):
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return typing.get_args(hint)
    return None


def _scalar(value, hint, where):
    if hint is typing.Any:
        return value
    if hint is bool:
        if isinstance(value, bool):
            return value
    elif hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif hint is str:
        if isinstance(value, str):
            return value
    elif hint is type(None):
        if value is None:
            return None
    else:
        fail(where, f"unsupported field type {hint!r}")
    return fail(where, f"expected {getattr(hint, '__name__', hint)}, got {type(value).__name__} {value!r}")


def _tuple(value, args, where):
    if not isinstance(value, (list, tuple)):
        fail(where, f"expected a sequence, got {type(value).__name__} {value!r}")
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(convert(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
    if len(value) != len(args):
        fail(where, f"expected {len(args)} items, got {len(value)}")
    return tuple(convert(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True)))


_NO = object()


def _try_convert(value, hint, where: str, errs: list):
    """convert(), or _NO with the error appended to errs (one alternative of a union)."""
    try:
        return convert(value, hint, where)
    except SpecError as e:
        errs.append(str(e))
        return _NO


def convert(value, hint, where: str):
    """``value`` (as YAML gave it) checked against the type ``hint`` and converted (sequence → tuple, int → float,
    mapping → dataclass). A mismatch raises SpecError naming ``where``."""
    alts = _union_args(hint)
    if alts is not None:
        errs = []
        for a in alts:
            got = _try_convert(value, a, where, errs)
            if got is not _NO:
                return got
        fail(where, f"matches none of {hint!r}: {' | '.join(errs)}")
    origin = typing.get_origin(hint)
    if origin is tuple:
        return _tuple(value, typing.get_args(hint), where)
    if origin is dict:
        if not isinstance(value, dict):
            fail(where, f"expected a mapping, got {type(value).__name__} {value!r}")
        kt, vt = typing.get_args(hint)
        return {convert(k, kt, f"{where} key"): convert(v, vt, f"{where}.{k}") for k, v in value.items()}
    if _is_dataclass_type(hint):
        if not isinstance(value, dict):
            fail(where, f"expected a mapping for {hint.__name__}, got {type(value).__name__}")
        return build(hint, value, where)
    return _scalar(value, hint, where)


def _default_of(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _is_default(v, d) -> bool:
    """``v`` (YAML) states the field's default ``d`` — a sequence equals a tuple default item by item."""
    if d is dataclasses.MISSING:
        return False
    if isinstance(d, tuple):
        return isinstance(v, list | tuple) and len(v) == len(d) and all(map(_is_default, v, d))
    if isinstance(v, bool) or isinstance(d, bool):
        return type(v) is type(d) and v == d
    if isinstance(d, int | float) and isinstance(v, int | float):
        return v == d
    return type(v) is type(d) and v == d


def build(cls, raw: dict, where: str):
    """``cls(**raw)`` after checking the keys (every field stated, nothing else) and every value's type."""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(raw) - set(names))
    if unknown:
        fail(where, f"unknown key(s) {unknown} for {cls.__name__} (known: {names})")
    missing = [n for n in names if n not in raw]
    if missing:
        fail(where, f"missing key(s) {missing} for {cls.__name__} — a spec states every field")
    kw = {}
    for f in dataclasses.fields(cls):
        v, d = raw[f.name], _default_of(f)
        kw[f.name] = d if _is_default(v, d) else convert(v, hints[f.name], f"{where}.{f.name}")
    try:
        return cls(**kw)
    except (TypeError, ValueError) as e:
        fail(where, f"{cls.__name__} refused the values: {e}")
