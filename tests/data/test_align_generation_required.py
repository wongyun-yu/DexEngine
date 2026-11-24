"""The reference hand direction refuses a missing generation declaration where the constant would be wrong.

Before the fix, `align.for_dataset` returned the constant REF_DEG (119.91) for every dataset without
`data/<ds>/GENERATION`. Six datasets were built under 119.79, so a data root made from the raw downloads without the
shipped input `generation` rotated their scenes by 0.12 deg and nothing said so. Now:
- a dataset of BUILT_OFF_REF without its declaration fails and names the shipped input (negative: fails if the
  fallback returns);
- with the declaration the value is the declared one, as before;
- the datasets built under the constant (and any unlisted dataset) keep the constant without a declaration;
- `ship` refuses to write the shipped input when the declared values disagree with BUILT_OFF_REF (oracle).
"""

from __future__ import annotations

import pytest

from dexengine.core import paths as P
from dexengine.core.errors import INPUT_MISSING
from dexengine.data import align

KEPT = ("hot3d", "oakink2", "taco")


def _declare(ds: str, deg: float, name: str = "g1") -> None:
    f = align.decl_path(P.layer_path("intake", ds))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"generation={name}\n{align.KEY}={deg}\n", encoding="utf-8")


def test_the_six_and_the_three_cover_the_registered_datasets():
    assert set(align.BUILT_OFF_REF) | set(KEPT) == set(P.data_layout.DATASETS)
    assert not set(align.BUILT_OFF_REF) & set(KEPT)


@pytest.mark.parametrize("ds", align.BUILT_OFF_REF)
def test_a_missing_declaration_fails_where_the_constant_would_rotate_the_scenes(ds):
    with pytest.raises(align.RefDirError) as e:
        align.for_dataset(ds)
    msg = str(e.value)
    assert INPUT_MISSING in msg and f"`{align.SHIPPED_INPUT}`" in msg
    assert f"data/{ds}/{align.GENERATION_FILE}" in msg and f"SHA256SUMS.{align.SHIPPED_INPUT}" in msg


def test_a_present_declaration_gives_its_value():
    _declare("grab", 119.79)
    assert align.for_dataset("grab") == (pytest.approx(119.79), "generation:g1")


@pytest.mark.parametrize("ds", KEPT)
def test_datasets_built_under_the_constant_keep_it(ds):
    assert align.for_dataset(ds) == (align.REF_DEG, "fixed")


def test_an_unlisted_dataset_keeps_the_constant(tmp_path):
    assert align.for_dataset("newds", tmp_path / "data" / "newds" / "INTAKE") == (align.REF_DEG, "fixed")


def test_ship_checks_the_tuple_against_the_declared_values(tmp_path):
    for ds in align.BUILT_OFF_REF:
        _declare(ds, 119.79)
    for ds in KEPT:
        _declare(ds, align.REF_DEG)
    out = tmp_path / "shipped_generation"
    align.ship(out)
    assert sorted(p.parent.name for p in out.glob(f"*/{align.GENERATION_FILE}")) == sorted(P.data_layout.DATASETS)
    _declare("grab", align.REF_DEG)
    with pytest.raises(align.RefDirError, match="BUILT_OFF_REF"):
        align.ship(tmp_path / "shipped_again")
