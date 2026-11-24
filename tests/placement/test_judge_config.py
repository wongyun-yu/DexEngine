"""Judge search conditions: the paper's Sec. 3.3 constants, the parser defaults and the input-record flags."""

from __future__ import annotations

import pytest

from dexengine.placement.judge import config as C


def test_paper_constants():
    c = C.load()
    assert c.vol_frac == 0.80  # 80 % coverage of the screening
    assert c.dp_err_mm == 30.0  # 3 cm tracking threshold (stretch-weighted p99)
    assert c.random_starts(c.track_seeds) == 32  # NS = max(4 · 8, 32)
    assert c.random_starts(12) == 48 and c.random_starts(1) == 32
    assert c.swivel_max_deg == 180.0


def test_parser_defaults_follow_config():
    from dexengine.placement.judge import args as A

    c = C.load()
    a = A._build_parser().parse_args(
        ["--arm", "rby1", "--hand", "sharpa_wave", "--dataset", "taco", "--emb", "bimanual", "--out", "/nonexistent"]
    )
    for f in C.dataclasses.fields(C.JudgeConfig):
        assert getattr(a, f.name) == getattr(c, f.name), f.name
    # the flags whose effective values enter the judge input record, as default text
    assert C.canon_flags() == {
        "--desk-shrink-robot": "0.15",
        "--exhaustive": "",
        "--exhaust-max": "1024",
        "--batch-cells": "512",
        "--n-candidates": "3",
        "--exhaust-cands": "3",
        "--jump-max": "1",
        "--order-nat": "1",
        "--desk-obj-frame0": "1",
        "--gate-batch": "256",
        "--clip-budget-s": "1200",
        "--arm-flip": "1",
    }


def test_load_rejects_unknown_and_missing_keys(tmp_path):
    text = C.CONFIG_FILE.read_text()
    bad = tmp_path / "unknown.yaml"
    bad.write_text(text + "\nextra:\n  not_a_field: 1\n")
    with pytest.raises(ValueError, match="unknown keys"):
        C.load(bad)
    short = tmp_path / "missing.yaml"
    short.write_text(text.replace("  vol_frac:", "  # vol_frac:"))
    with pytest.raises(ValueError, match="missing keys"):
        C.load(short)
