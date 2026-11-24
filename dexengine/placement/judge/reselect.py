"""Re-selection margin of the judge (the re-selection rule) — `--place-reselect <record>`.

The bake writes a re-selection record (placement.bake.place_check.RESELECT_FILE) for a clip whose every candidate
failed the post-bake check on the robot spot. The re-selection chunk judges only those clips once more with the robot
spot disc widened by the record's margin (place_rule.clip margin_m); the post-bake check and the audit keep margin 0,
so no tree's verdict changes. Without the option nothing changes. The record is read with place_check.read_reselect
(form owner); every clip of the record must be in the judge list (a stray clip dies — ReselectError). Marker
`[PLACE_RESELECT] clips=<n> record=<path>` once at boot and one line per widened clip.
The judge has no reuse path (the old --reuse-from was removed), so no result found without the margin can be carried
into a re-selection run. The margin is a search condition outside the judge input record (input_record), as before.
"""

from __future__ import annotations

from pathlib import Path

from dexengine.placement.bake import place_check as PC
from dexengine.placement.scene.place_rule import PlaceRuleError

TAG = PC.RESELECT_TAG


class ReselectError(PlaceRuleError):
    """The re-selection record does not fit this judge run (a clip outside the judge list)."""


def load(record: str, clips) -> dict[str, float]:
    """{clip: margin m} of the record ('' = no record → {}); a record clip outside `clips` raises ReselectError."""
    if not record:
        return {}
    rs = PC.read_reselect(record)
    if stray := sorted(set(rs) - set(clips)):
        raise ReselectError(
            f"{TAG} ✗ clips of the re-selection record {record} are not in the judge list: {stray[:5]}"
            f" ({len(stray)} of {len(rs)})"
        )
    print(
        f"{TAG} clips={len(rs)} record={Path(record)} — only these clips are judged with the robot spot disc + "
        "their margin",
        flush=True,
    )
    return rs


def margin(rs: dict, clip: str) -> float:
    """The robot spot margin (m) of a clip in this run — 0 outside the record (the rule)."""
    return float(rs.get(clip, 0.0))


def clip_line(clip: str, margin_m: float, pr: dict) -> str:
    """The per-clip marker of a widened clip (pr = place_rule.clip with that margin). A record clip whose tree is out
    of the place rule (no robot spot disc to widen) raises ReselectError — the margin is never silently unused."""
    if not pr.get("on") or pr.get("keepout") is None:
        raise ReselectError(
            f"{TAG} ✗ {clip}: margin {margin_m * 1000:.2f} mm in the re-selection record but the "
            "place rule is off for this tree (no robot spot disc to widen)"
        )
    return (
        f"{TAG} {clip} margin {margin_m * 1000:.2f} mm — robot spot disc radius {pr['keepout'][2]:.4f} m in the "
        "search (post-bake check · audit keep margin 0)"
    )
