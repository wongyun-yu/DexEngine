"""Import a module of this repository in a test, skipping only when a third-party package it needs is absent.

`pytest.importorskip` on a repository module would also turn a broken import of our own code into a skip; here a
missing module of this repository, or any other import error, fails the test."""

from __future__ import annotations

import importlib

import pytest

OWN = ("dexengine", "experiments", "tests")


def own_module(name: str):
    """The module `name`; a skip when a third-party module it imports is not installed."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name is None or e.name.split(".")[0] in OWN:
            raise
        pytest.skip(f"{name} needs {e.name}, which is not installed", allow_module_level=True)
