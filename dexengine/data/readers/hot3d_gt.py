"""HOT3D recordings with ground truth: the rule shared by the reader (``hot3d.load_clips``) and the raw re-read
driver (``python -m dexengine.data.read_raw hot3d read``).

HOT3D does not publish hand/object GT for 130 of its 424 recordings (BOP challenge test split, Aria 62 · Quest 68):
their hand and object files are 0 bytes and metadata says ``have_hand_object_pose_gt=false``. The 294 recordings with GT
are the scope — the same count as CHORD's "294 raw recordings". GT = the metadata flag is true **and** the hand pose
file is not empty. A recording whose flag is missing or disagrees with its pose file is an error, not a default.

Standard library only, so any interpreter can list the recordings.
Usage: python3 -m dexengine.data.readers.hot3d_gt <recordings root>
  prints the GT recording names, one per line · rc 2 when a recording is inconsistent (nothing printed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GT_KEY = "have_hand_object_pose_gt"
POSE_FILE = "mano_hand_pose_trajectory.jsonl"


def has_ground_truth(seq_dir: str | Path) -> bool:
    """True when the recording has hand/object GT; ValueError when its flag is missing or disagrees with its pose
    file."""
    seq_dir = Path(seq_dir)
    meta = json.loads((seq_dir / "metadata.json").read_text())
    if GT_KEY not in meta:
        raise ValueError(f"{seq_dir.name}: metadata has no {GT_KEY}")
    f = seq_dir / POSE_FILE
    pose = f.is_file() and f.stat().st_size > 0
    if bool(meta[GT_KEY]) != pose:
        state = "present" if pose else "empty"
        raise ValueError(f"{seq_dir.name}: {GT_KEY}={meta[GT_KEY]} but the hand pose file is {state}")
    return pose


def _sort_recording(d: Path, good: list[str], bad: list[str]) -> None:
    """Append ``d``'s name to ``good`` when it has GT, or its inconsistency to ``bad``."""
    try:
        if has_ground_truth(d):
            good.append(d.name)
    except (ValueError, OSError) as e:
        bad.append(str(e))


def ground_truth_recordings(root: str | Path) -> list[str]:
    """Every recording under ``root`` (``P*`` folders) with GT, sorted. ValueError lists every inconsistent one."""
    good, bad = [], []
    for d in sorted(p for p in Path(root).glob("P*") if p.is_dir()):
        _sort_recording(d, good, bad)
    if bad:
        raise ValueError(f"inconsistent recordings {len(bad)}: " + " · ".join(bad[:5]))
    return good


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="the recordings root (P* folders)")
    a = ap.parse_args(argv)
    try:
        names = ground_truth_recordings(a.root)
    except ValueError as e:
        print(f"[hot3d_gt] ✗ {e}", file=sys.stderr)
        return 2
    print("\n".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
