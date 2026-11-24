"""A body of another generation at a canonical body's cache name never spawns.

The spawned plate's identity record (derived_provenance.json) holds the md5 of the bodies it was made from, and the
spawn check (selection._check_derived_body -> derived_usd.verify) compares them with the files now at those names.
Negative: a folder of another generation planted at the canonical training-body name stops the spawn. With an empty
generation suffix that name is `rby1_<hand>`, the name an earlier torso-tilted generation used."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dexengine.core.arm_generations import generation_of
from dexengine.core.arms import plate_variant
from dexengine.robots import derived_usd as DU
from dexengine.robots import selection as SEL

ARM = "rby1"


@pytest.mark.parametrize("hand", ["sharpa_wave", "xhand_pro", "dg5f_s", "wuji2"])
def test_old_generation_at_the_training_body_name_stops(tmp_path, monkeypatch, hand):
    monkeypatch.delenv(DU.PROVENANCE_DIR_ENV, raising=False)
    monkeypatch.setattr(DU, "collider_digest_now", lambda *a, **k: "digest")
    cache = str(tmp_path)
    plate = plate_variant(ARM, hand)
    gen = generation_of(ARM, plate)
    src = gen.derived_from
    mat = generation_of(ARM, src).material_of
    paths = {v: Path(SEL._usd_literal(ARM, hand, cache, v)) for v in (plate, src, mat)}
    for v, p in paths.items():
        p.parent.mkdir(parents=True)
        p.write_bytes(f"canonical body {v}".encode())
    prov = {
        "variant": plate,
        "source_variant": src,
        "source_md5": DU.file_md5(paths[src]),
        DU.GENERATOR_KEY: DU.generator_digest(),
        DU.DIGEST_KEY: "digest",
        DU.MATERIAL_SOURCE_KEY: DU.file_md5(paths[mat]),
    }
    DU.provenance_file(paths[plate]).write_text(json.dumps(prov), encoding="utf-8")
    SEL._check_derived_body(gen, ARM, str(paths[plate]), plate, hand, cache)  # the canonical set spawns
    paths[mat].write_bytes(b"a body of the earlier generation")
    with pytest.raises(RuntimeError, match="training body USD changed"):
        SEL._check_derived_body(gen, ARM, str(paths[plate]), plate, hand, cache)
