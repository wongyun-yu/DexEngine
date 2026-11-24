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
    base = {
        "clip_id": "x",
        "objects": [{"role": "main", "obj_id": "o1"}],
        "reference": {"note": "n", "source": "s", "time": "2026-10-13T04:24:27+0900"},
        "support": {"rest_intervals": [[0, 5]]},
    }
    ref = json.loads(json.dumps(base))
    ref["reference"].update(time="2026-10-13T19:51:19+0900", sibling_parts=[{"raw_obj": "O02@0039@00004"}])
    assert _same(base, ref)  # the reference block (f0 reads none of it — its time stamp moved v3 keys)
    other = json.loads(json.dumps(base))
    other["objects"][0]["obj_id"] = "o2"
    assert not _same(base, other)  # a field f0 reads still moves the key
    sup = json.loads(json.dumps(base))
    sup["support"]["rest_intervals_window"] = [0, 9]
    sup["clip_id"] = "y"
    assert _same(base, sup)
    monkeypatch.setattr(F, "TI_UNREAD_TOP", tuple(k for k in F.TI_UNREAD_TOP if k != "reference"))
    assert not _same(base, ref)  # negative: without the entry the reference block counts


def test_f0_key_reads_no_reference():
    """`reference` is out of the f0 key because no f0 module reads it: the only task_info readers of that block are
    the two non-f0 modules below. A new reader anywhere fails here — decide whether f0 reads it (then take
    `reference` out of TI_UNREAD_TOP) and add it to the list."""
    import re
    from pathlib import Path

    root = Path(F.__file__).resolve().parents[3]
    pat = re.compile(r"""\[\s*["']reference["']\s*\]|\.get\(\s*["']reference["']""")
    readers = sorted(
        str(p.relative_to(root))
        for p in (root / "dexengine").rglob("*.py")
        if p.name != "shadow_key.py" and pat.search(p.read_text())
    )
    assert readers == ["dexengine/data/filtering.py", "dexengine/record/text/source_labels.py"], readers


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
    rf, held = tmp_path / "run_failed.txt", tmp_path / "held.txt"
    assert (
        ledger.split_failures([str(tmp_path / "f0"), str(clips), "hand", "--run-failed", str(rf), "--held", str(held)])
        == 0
    )
    assert capsys.readouterr().out == "b\n"  # declared data defects only
    assert rf.read_text() == "c\tFAIL_EXC\tboom\n"  # run failures; the last row of a wins; d is absent
    assert held.read_text() == ""  # always written
