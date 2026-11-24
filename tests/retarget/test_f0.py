"""f0 (first-frame repair) pieces that run without the GPU engine: the task_info projection of the input key and the
split of non-judgeable ledger rows."""

from __future__ import annotations

import json

from dexengine.retarget.f0 import ledger
from dexengine.retarget.f0 import shadow_key as F
from dexengine.retarget.f0 import shadow_status as ST


def _same(a: dict, b: dict) -> bool:
    return json.dumps(F.ti_projection(a), sort_keys=True) == json.dumps(F.ti_projection(b), sort_keys=True)


def test_unread_task_info_fields_do_not_move_the_key(monkeypatch):
    base = {"clip_id": "x", "reference": {"note": "n", "source": "s"}, "support": {"rest_intervals": [[0, 5]]}}
    parts = json.loads(json.dumps(base))
    parts["reference"].update(sibling_parts=[{"raw_obj": "O02@0039@00004"}], raw_fit_residual_m=0.0123)
    assert _same(base, parts)
    other = json.loads(json.dumps(base))
    other["reference"]["note"] = "another value"
    assert not _same(base, other)  # a field f0 may read still moves the key
    sup = json.loads(json.dumps(base))
    sup["support"]["rest_intervals_window"] = [0, 9]
    sup["clip_id"] = "y"
    assert _same(base, sup)
    monkeypatch.setattr(F, "TI_UNREAD_REFERENCE", ())
    assert not _same(base, parts)  # negative: without the list the unread fields count


def test_split_failures(tmp_path, capsys):
    led = tmp_path / "f0" / "hand"
    led.mkdir(parents=True)
    rows = [
        "clip\tstatus\tc3\tc4\tc5\tc6\tc7\tnote",
        "a\tFAIL_EXC\t-\t-\t-\t-\t-\told",
        "a\tCLEAN\t-\t-\t-\t-\t-\tok",
        "b\tDATA_DEFECT\t-\t-\t-\t-\t-\tdefect",
        "c\tFAIL_EXC\t-\t-\t-\t-\t-\tboom",
    ]
    (led / ST.LEDGER_NAME).write_text("\n".join(rows) + "\n")
    clips = tmp_path / "clips.txt"
    clips.write_text("a\nb\nc\nd\n")
    rf = tmp_path / "run_failed.txt"
    assert ledger.split_failures([str(tmp_path / "f0"), str(clips), "hand", "--run-failed", str(rf)]) == 0
    assert capsys.readouterr().out == "b\n"  # declared data defects only
    assert rf.read_text() == "c\tFAIL_EXC\tboom\n"  # run failures; the last row of a wins; d is absent
