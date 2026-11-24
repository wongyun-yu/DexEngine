"""dexengine.core.config: typed sections from YAML, overrides and the command line."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from dexengine.core import config as C


@dataclass(frozen=True)
class Inner:
    k: int = 1


@dataclass(frozen=True)
class Section:
    width: int = 640
    fps: float = 30.0
    looks: tuple[str, ...] = ("orig",)
    inner: Inner = field(default_factory=Inner)
    out: Path | None = None
    mode: Literal["a", "b"] = "a"
    tags: dict[str, int] = field(default_factory=dict)


def test_defaults_without_file():
    assert C.load(Section, "no_such_section") == Section()


def test_yaml_file_then_overrides(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("width: 320\ninner: {k: 3}\nlooks: [orig, aug]\n")
    cfg = C.load(Section, path=f, overrides=["inner.k=5", "fps=2", "out=/x", "mode=b", "tags={a: 1}"])
    assert cfg == Section(
        width=320, fps=2.0, looks=("orig", "aug"), inner=Inner(5), out=Path("/x"), mode="b", tags={"a": 1}
    )
    assert isinstance(cfg.fps, float)


@pytest.mark.parametrize("bad", ["nope=1", "width=1.5", "width=true", "mode=c", "inner.z=1", "looks=3", "width"])
def test_bad_overrides_raise(bad):
    with pytest.raises(C.ConfigError):
        C.load(Section, overrides=[bad])


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(C.ConfigError):
        C.load(Section, path=tmp_path / "missing.yaml")


def test_cli_round_trip(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("width: 100\n")
    ap = C.add_cli(argparse.ArgumentParser())
    args = ap.parse_args(["--config", str(f), "--set", "fps=10", "--set", "looks=[x]"])
    assert C.from_args(Section, "unused", args) == Section(width=100, fps=10.0, looks=("x",))
    assert C.dump(Section()) == {
        "width": 640,
        "fps": 30.0,
        "looks": ["orig"],
        "inner": {"k": 1},
        "out": None,
        "mode": "a",
        "tags": {},
    }
    g = tmp_path / "dump.yaml"
    g.write_text(C.to_yaml(Section()))
    assert C.load(Section, path=g) == Section()


def test_env_names_are_declared(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "wuji2")
    assert C.env("DEXENGINE_HAND") == "wuji2"
    with pytest.raises(C.ConfigError):
        C.env("DEXENGINE_SOMETHING_ELSE")


def test_data_root_cli_exports_env(tmp_path):
    ap = C.add_data_root_arg(argparse.ArgumentParser())
    C.apply_data_root(ap.parse_args(["--data-root", str(tmp_path)]))
    assert os.environ["DEXENGINE_MAIN_ROOT"] == str(tmp_path)
    assert C.apply_data_root(ap.parse_args([])) == ""
