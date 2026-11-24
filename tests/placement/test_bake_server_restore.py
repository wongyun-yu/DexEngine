"""Resident bake worker isolation (placement/bake/server.py rules 2 · 2b): a module global the snapshot cannot
deep-copy either is declared in `_SERVER_KEEP` (kept as it is, the restore never touches it) or stops the import —
before the fix it was snapshotted raw and the next request's restore died on it ('cannot pickle mappingproxy'), so
only the first clip of every lane was baked."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from dexengine.placement.bake import server as SV


@pytest.fixture(autouse=True)
def simple_types(monkeypatch):
    """The value types the worker restores by deep copy (set by server.main before the hook is installed)."""
    monkeypatch.setattr(
        SV, "_SIMPLE", (dict, list, set, tuple, int, float, str, bytes, bool, type(None), np.ndarray, np.generic)
    )
    monkeypatch.setattr(SV, "_SNAP", {})


def _module(name: str, **glob) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(glob)
    return m


def test_undeclared_uncopyable_global_stops_the_import():
    m = _module("fake_meta", _META={"a": types.MappingProxyType({"env": "X"})})
    with pytest.raises(RuntimeError, match=r"fake_meta\._META .*_SERVER_KEEP"):
        SV._snapshot(m)


def test_declared_uncopyable_global_survives_two_restores(monkeypatch):
    proxy = types.MappingProxyType({"env": "X"})
    m = _module("fake_kept", _META={"a": proxy}, _SERVER_KEEP=("_META",), counter=[0])
    monkeypatch.setitem(sys.modules, "fake_kept", m)  # the restore forgets snapshots of modules no longer loaded
    SV._SNAP[id(m)] = (m, SV._snapshot(m))
    held = m._META
    for _ in range(2):  # request 1 and request 2 of the same worker
        m.counter.append(1)  # a clip writes into an undeclared container
        SV._restore_all()
        assert m._META is held  # kept by identity
        assert m.counter == [0]  # everything else is back at its import value


def test_real_switches_module_snapshots_and_restores():
    from dexengine.rl.tasks import switches as SW

    before = [SW.unset_text(f) for f in SW.FIELDS]
    SV._SNAP[id(SW)] = (SW, SV._snapshot(SW))
    meta = SW._META
    SV._restore_all()
    SV._restore_all()
    assert SW._META is meta
    assert [SW.unset_text(f) for f in SW.FIELDS] == before  # the `is _UNSET` checks still hold after the restores
