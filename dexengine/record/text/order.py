"""Which episode feeds the text, and which clips are text targets.

Hand order (source_rank): for each clip the row-0 x look-D episode of the first hand in SOURCE_HANDS is the text
input (sources.dataset_index scans the final datasets in this order). Hand name = the <hand> of the final combo
rby1_<hand>. A final dataset of a hand outside the table dies (it is never silently appended).

Targets from the final datasets (final_targets): every clip actually present in the finals (all hands). Each
episode must link to its recorded parent clip (piece / case parent_clip — the clip the segment was cut from: the
same take by paths.parent_clip, the key being that clip or a derivative of it); a broken link dies.
"""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError

MARK = "[TEXT_ORDER]"
#: Text source hand order: wuji2 -> SharpA -> Tesollo (DG-5F-S) -> XHand Pro -> wuji (first generation).
SOURCE_HANDS = ("wuji2", "sharpa_wave", "dg5f_s", "xhand_pro", "wuji")


def hand_of(final) -> str:
    """Final dataset folder (data/<ds>/collect/rby1_<hand>) -> <hand>."""
    return Path(final).name.split("_", 1)[-1]


def source_rank(root) -> tuple:
    """Scan order of dataset folders: final layer (data/<ds>/collect/rby1_<hand>, parent named collect) ->
    (1 + hand rank, folder), dying on an unknown hand; piece layer (<piece>/rounds/<round>/dataset — a single hand)
    -> (0, folder)."""
    if Path(root).parent.name != "collect":
        return 0, str(root)
    h = hand_of(root)
    if h not in SOURCE_HANDS:
        raise SystemExit(
            f"{MARK} hand {h!r} ({root}) is not in the text source hand order — add it to order.SOURCE_HANDS"
        )
    return 1 + SOURCE_HANDS.index(h), str(root)


def loaded_episodes(patterns: list[str]) -> dict[str, list[tuple[Path, dict]]]:
    """Clip -> every loaded episode [(final folder, meta row)] in hand order (source_rank), superseded rows removed,
    key = ep_clip (placement variants map to their source clip). The text input phase table covers the phase names
    of all these episodes (phases.union_table)."""
    import glob

    from dexengine.record.store import merge_check as MC

    out: dict[str, list] = {}
    roots = sorted(
        {Path(d) for pat in patterns for d in glob.glob(pat) if (Path(d) / "meta" / "info.json").is_file()},
        key=source_rank,
    )
    for root in roots:
        gone = MC.superseded_indices(root)
        for ln in (root / "meta" / "episodes.jsonl").read_text().splitlines():
            e = json.loads(ln) if ln.strip() else None
            if e is not None and int(e["episode_index"]) not in gone:
                out.setdefault(ep_clip(e), []).append((root, e))
    return out


def ep_clip(e: dict) -> str:
    """Final episode row -> text table key (clip name); a placement variant maps to its source clip
    (selection.source_clip)."""
    return P.clip_name((e.get("selection") or {}).get("source_clip") or e["clip"])


def linked(key: str, recorded: str) -> bool:
    """The key links to the recorded parent clip: same take, and the clip itself or a derivative (suffix _...)."""
    return P.parent_clip(recorded) == P.parent_clip(key) and (key == recorded or key.startswith(recorded + "_"))


def final_targets(finals: list) -> list:
    """(dataset, clip) actually present in the final datasets (all hands); an unlinkable clip dies."""
    out: dict = {}
    for f in finals:
        for ln in (Path(f) / "meta" / "episodes.jsonl").read_text().splitlines():
            e = json.loads(ln) if ln.strip() else None
            if e is None:
                continue
            k = ep_clip(e)
            rec = {r for r in ((e.get(x) or {}).get("parent_clip") for x in ("piece", "case")) if r}
            if not rec or not all(linked(k, r) for r in rec):
                raise SystemExit(
                    f"{MARK} final clip {k} ({f}) cannot be linked to its parent clip — recorded "
                    f"{sorted(rec)} · paths.parent_clip {P.parent_clip(k)}"
                )
            out[(Path(f).parts[-3], k)] = None
    return list(out)


def cap(todo: list, n: int) -> list:
    """Cap on new clips per round (track.json round_cap, 0 = none), so frame extraction of one tick stays bounded;
    the rest follows in the next tick (target order kept)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise InputMissingError(f"[COLLECT_TEXT] round_cap {n!r} — an integer >= 0 (0 = none)")
    if n and len(todo) > n:
        print(f"[COLLECT_TEXT] round cap round_cap {n} — this round {n} · next tick {len(todo) - n}", flush=True)
        return todo[:n]
    return todo
