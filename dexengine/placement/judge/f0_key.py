"""f0 shadow identity key — lets the region-dump reuse gate see that the f0 shadow was built again.

Judge mode D (placement.judge.aug_cells) stores the clip's f0 marker key in the region dump (field f0_key); the
augmentation runner (record.augment.cells_cond.f0_again) compares it with the current marker and judges the clip again
when they differ. Key = the marker's KEY_FIELDS — rebuilding f0 changes input_key; it is a content key, so the same
inputs give the same key on any machine. A keyless old dump is judged again when its marker is newer than the dump.
A missing marker dies: cell judging never runs without the f0 shadow. This module owns the marker file name
(retarget.f0.prejudge, which writes the marker, reads MARKER from here). Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.core.errors import InputMissingError

MARKER = "F0_PREJUDGE.json"
#: the f0 object repair's rows marker in a shadow clip (written by retarget.f0.repair_obj_write.write_obj_window; its
#: rows_sha is the f0 output part of the judge record's base field, judge.input_clip._f0_obj)
OBJ_WINDOW_MARKER = "F0_OBJ_WINDOW.json"
KEY_FIELDS = ("input_key", "fp_design", "status")
TAG = "[F0_KEY]"


def marker(f0root, spec_dir: str, emb: str, clip: str) -> Path:
    """Place of the f0 shadow marker = <f0 root>/<hand spec dir>/<emb>/<clip>/MARKER (where prejudge writes it)."""
    return Path(f0root) / spec_dir / emb / clip / MARKER


def judge_marker(J, clip: str) -> Path:
    """The marker of the f0 shadow the judge reads — f0 root = J._F0ROOT (paths.f0_root · env DEXENGINE_F0_STAGE)."""
    from dexengine.core.paths import hand_spec_dir  # noqa: PLC0415

    if not getattr(J.a, "wrist_root", ""):
        raise SystemExit(
            f"{TAG} ✗ {clip}: the judge does not read the f0 shadow (--wrist-root off) — cell judging must read f0"
        )
    return marker(J._F0ROOT, hand_spec_dir(J.a.hand), J.a.emb, clip)


def key_of(mk: Path) -> str:
    """The marker's key — json of KEY_FIELDS (sorted keys); the stored f0_key value."""
    if not Path(mk).is_file():
        raise InputMissingError(f"{TAG} ✗ no f0 marker {mk} — there is no f0 shadow for the judge to read")
    m = json.loads(Path(mk).read_text())
    return json.dumps({k: m.get(k) for k in KEY_FIELDS}, sort_keys=True)


def unchanged(mk: Path, key: str) -> None:
    """Dies when f0 was built again during the judge — the dump may have read a half-built f0 (an f0 rebuild racing
    the judge)."""
    if key_of(mk) != key:
        raise SystemExit(
            f"{TAG} ✗ the f0 marker changed during the judge {mk} — the dump carries the old key, so the "
            "next run judges it again"
        )


def stale_why(dump_key: str, dump_mtime: float, mk: Path) -> str:
    """'' = the dump came from the current f0 shadow · otherwise the reason to judge it again."""
    now = key_of(mk)
    if dump_key:
        return "" if dump_key == now else f"f0 key differs (dump {dump_key} != now {now})"
    dt = Path(mk).stat().st_mtime - dump_mtime
    return f"old dump without an f0 key · the marker is {dt:.0f} s newer" if dt > 0 else ""
