"""J-self-pen in the final merge: every trajectory of a piece is measured on its executed joints (owner
placement.collision.self_pen) before the final lock is taken (merge.measure_piece: the ruler never blocks another
merge), together with a fingerprint of the bytes measured (the piece meta hash and each data file's sha256). Under the
lock merge_piece reads the piece again and dies when the fingerprint differs (the piece changed after it was measured).
A trajectory over the line leaves the merge with all its episodes (looks), and the pieces line records it
(`self_pen_dropped`); a clip leaves the final only when every trajectory of it is dropped. A piece whose every
trajectory is dropped still gets its pieces line (no episodes, `self_pen_dropped` with every trajectory), so the piece
count of the final sees it.

Trajectory = the episodes of one episode clip (variant suffix included) of the piece. Its looks are renders of one
rollout; each distinct joint table is measured once, and the trajectory's value is the deepest of its tables.
Callers: store.merge.measure_piece (measure, fingerprint) · store.merge.merge_piece (same_piece, plan: after the input
check, before the overlap rules, so a dropped trajectory never takes part in a supersede, skip or top-up decision) ·
store.merge_check (line_problems). Markers: [SELF_PEN] per trajectory · [SELF_PEN_SUM] per piece.
"""

from __future__ import annotations

from dexengine.core.hashing import sha256_file
from dexengine.placement.collision import self_pen as SP
from dexengine.record.store import merge_overlap as MO
from dexengine.record.store.merge_check import DATA, die

RULE = "J-self-pen"
SUM = "[SELF_PEN_SUM]"


def measure(pc: dict) -> dict:
    """{episode clip: (worst Result, [piece episode positions])} of a read piece, in episode order."""
    combo, names = str(pc["info"]["robot_type"]), list(pc["info"]["features"][SP.JOINTS]["names"])
    by: dict[str, list[int]] = {}
    for k, e in enumerate(pc["eps"]):
        by.setdefault(e["clip"], []).append(k)
    seen, out = {}, {}
    for clip, ks in by.items():
        worst = None
        for k in ks:
            Q = SP.read_joints(pc["files"][k][DATA])
            h = SP.table_key(Q)
            if h not in seen:
                seen[h] = SP.measure_joints(combo, names, Q)
            if worst is None or seen[h].depth_m > worst.depth_m:
                worst = seen[h]
        out[clip] = (worst, ks)
    return out


def fingerprint(pc: dict) -> dict:
    """The bytes a measurement read: the piece meta hash and the sha256 of every episode's data file (episode order)."""
    return {"sha": pc["sha"], "data": [sha256_file(f[DATA]) for f in pc["files"]]}


def same_piece(pc: dict, measured: dict, piece) -> None:
    """Dies unless the piece read under the lock is the one that was measured (fingerprint)."""
    if fingerprint(pc) != measured["fp"]:
        die(f"piece {piece} changed after it was measured (J-self-pen fingerprint) — measure it again, nothing written")


def plan(pc: dict, piece_id: str, got: dict) -> tuple[dict, list]:
    """(the piece without its dropped trajectories — no episodes when none is left, the pieces-line records of the
    dropped ones) from the measurement `got` (measure). Prints the markers."""
    dropped, keep = [], []
    for clip, (res, ks) in got.items():
        print(SP.marker(clip, res), flush=True)
        if SP.verdict(res) == "drop":
            dropped.append(
                {"clip": clip, "rule": RULE, "why": SP.reason(res), "statement": SP.statement(res), "piece_eps": ks}
            )
        else:
            keep += ks
    print(
        f"{SUM} {piece_id} — trajectories {len(got)} · dropped {len(dropped)} · episodes kept {len(keep)} / "
        f"{len(pc['eps'])} · recorded in the pieces line self_pen_dropped",
        flush=True,
    )
    if not dropped:
        return pc, []
    return MO.keep_eps(pc, sorted(keep)), dropped


def line_problems(r: dict) -> list:
    """A pieces line's self_pen_dropped records: each a drop of this rule, its clip not among the line's episodes."""
    bad = []
    for d in r.get("self_pen_dropped") or []:
        if d.get("rule") != RULE or (d.get("statement") or {}).get("verdict") != "drop" or not d.get("piece_eps"):
            bad.append(f"piece {r['piece_id']}: self_pen_dropped record {d.get('clip')} is not a {RULE} drop")
        if d.get("clip") in set(r.get("episode_clips") or []):
            bad.append(f"piece {r['piece_id']}: self_pen_dropped clip {d.get('clip')} is in the final")
    return bad
