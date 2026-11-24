"""Judge reject rows (noconfirm_<combo>.tsv) written before the English names: the no-place heads (no volume · no
place spot · no receiver spot) were Korean words. The state fold (ledger_states.noconfirm_fold) reads the reason only
for `-` rows; a head in the old names is not migrated data, so the fold dies instead of setting nothing (the clip would
otherwise stay without its EXHAUSTED state). The status ledger names the file and line. With an
external values table (env DEXENGINE_LEGACY_VALUES) mapping the old heads to no_volume / no_place_spot, the mapped
reasons fold to EXHAUSTED.
"""

from __future__ import annotations

import pytest

from dexengine.core import legacy_text as LT
from dexengine.placement import ledger as L
from dexengine.placement.ledger_states import noconfirm_fold


def _ko(*cps: int) -> str:
    """Korean text built at run time from code points (the tree holds no Korean text)."""
    return "".join(map(chr, cps))


VOLUME_KO = _ko(0xBCFC, 0xB968, 0xC5C6, 0xC74C)  # the old no-volume head
PLACE_KO = _ko(0xB193, 0xC784, 0xC790, 0xB9AC, 0xC5C6, 0xC74C)  # the old no-place-spot head
RECV_KO = _ko(0xBC1B, 0xB294, 0xBB3C, 0xCCB4, 0xC790, 0xB9AC, 0xC5C6, 0xC74C)  # the old no-receiver-spot head
BOTH_KO = _ko(0xACF5, 0xD1B5)  # 'both' in the coverage detail
COVER_KO = _ko(0xCEE4, 0xBC84)  # 'cover'
CONFIRM_KO = _ko(0xD655, 0xC815, 0xC2E4, 0xD328)  # 'confirm failed'
MAIN_KO = _ko(0xC8FC, 0xBB3C, 0xCCB4)  # 'main object'
OLD_VOLUME = f"{VOLUME_KO}(dz=-5 r12% l3% {BOTH_KO}0%; dz=-6 r1% {BOTH_KO}0%)"
OLD_PLACE = f"{PLACE_KO}(off_desk n=3 robot_keepout n=2 — {MAIN_KO})"


@pytest.mark.parametrize("why", [OLD_VOLUME, OLD_PLACE, RECV_KO, f"{VOLUME_KO}: x"])
def test_legacy_no_place_head_dies(why):
    with pytest.raises(ValueError, match="legacy value"):
        noconfirm_fold(None, "-", why)
    with pytest.raises(ValueError, match="legacy value"):
        noconfirm_fold("ADOPTED", "-", why)  # unmigrated data dies whatever the clip's state


def test_migrated_heads_fold_and_korean_detail_is_not_a_head():
    assert noconfirm_fold(None, "-", "no_volume(dz=-5 r12% l3% both0%)") == "EXHAUSTED"
    assert noconfirm_fold(None, "-", f"no_volume(dz=-5 r12% l3% {BOTH_KO}0%)") == "EXHAUSTED"  # display detail
    assert noconfirm_fold(None, "-", "no_place_spot(off_desk n=3 …)") == "EXHAUSTED"
    assert noconfirm_fold("UNPROVEN", "-", "no_volume(dz=-5)") == "UNPROVEN"


def test_reason_of_other_verdicts_is_not_read():
    """TRUNCATED / EXHAUSTED / anchor rows decide by the verdict; their (display) reason may still hold Korean words."""
    assert noconfirm_fold(None, "TRUNCATED", f"{COVER_KO} 0.40") == "UNPROVEN"
    assert noconfirm_fold(None, "EXHAUSTED", f"{COVER_KO} 0.10") == "EXHAUSTED"
    assert noconfirm_fold(None, "anchor0", f"{CONFIRM_KO}(3)") is None


def test_status_ledger_names_the_file_and_line(tmp_path):
    root = tmp_path / "judge"
    (root / "p0").mkdir(parents=True)
    f = root / "p0" / "noconfirm_rby1_x.tsv"
    f.write_text(f"c1\t-\tno_volume(dz=-5)\nc2\t-\t{OLD_VOLUME}\n", encoding="utf-8")

    class A:
        judge_root = [str(root)]

    with pytest.raises(SystemExit, match=r"noconfirm_rby1_x\.tsv:2: .*legacy value"):
        L._fold_noconfirm(A, {}, lambda c: c)


def test_legacy_rule_is_hangul_only():
    assert LT.is_legacy(VOLUME_KO) and not LT.is_legacy("no_volume") and not LT.is_legacy("dz=-5 r12% · —")
