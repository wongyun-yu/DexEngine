"""The render-only record's measuring module is owned by its dataset, not by a module name stored in the record."""

import pytest

from dexengine.record.scene import render_only as RO


def _rec(dataset: str, key_module: str | None = None) -> dict:
    rec = {"version": RO.VERSION, "dataset": dataset, "clip": f"{dataset}_clip"}
    if key_module is not None:
        rec["key_module"] = key_module
    return rec


def test_record_written_by_another_tree_resolves_to_the_dataset_reader():
    km = RO.frame_module(_rec("oakink2", key_module="scene_oracle_oakink2"))
    assert km.__name__ == "dexengine.data.readers.raw.oakink2"
    assert hasattr(km, "frame_key") and hasattr(km, "frame_inputs")


def test_record_without_key_module_resolves_the_same_way():
    assert RO.frame_module(_rec("oakink2")).__name__ == "dexengine.data.readers.raw.oakink2"


def test_dataset_without_a_raw_reader_dies():
    with pytest.raises(RO.RenderOnlyError, match="no raw scene reader"):
        RO.frame_module(_rec("no_such_dataset"))
