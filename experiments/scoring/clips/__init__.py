"""The frozen clip lists of the comparison with prior pipelines (Tab. 2) and the RL ablations.

``E123_CLIPS_v1.tsv`` is the 64-clip TACO set every method of Tab. 2 is scored on (``select_B0_pool.py`` drew the pool,
the B0r replacement record is in ``provenance/``); ``E123_HELDOUT_v1.tsv`` is the 64-clip held-out set of the transfer
probe (``select_heldout_v1.py``). Both files are records: they are read, never rewritten.
"""

from __future__ import annotations

from pathlib import Path

DIR = Path(__file__).resolve().parent
E123_CLIPS_V1 = DIR / "E123_CLIPS_v1.tsv"
E123_HELDOUT_V1 = DIR / "E123_HELDOUT_v1.tsv"


def e123_clips(path: Path = E123_CLIPS_V1) -> list[str]:
    """The 64 clip names of a frozen list in file order (column ``clip``; ``#`` lines are comments, the first row is the
    header). Raises when the list does not hold 64 distinct clips."""
    clips = [ln.split("\t")[1] for ln in Path(path).read_text().splitlines() if ln and not ln.startswith("#")][1:]
    assert len(clips) == 64 and len(set(clips)) == 64, f"{path}: {len(clips)} clips, {len(set(clips))} distinct"
    return clips
