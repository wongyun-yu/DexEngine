"""Trajectory selection rule: which successful episode of each clip is rendered.

The score is pure numpy; ``tracked_sides`` reads the env by duck typing, so both run without Isaac.

Score (smaller is better): per tracked side (position error / position tolerance + rotation error / rotation
tolerance) / 2, averaged over sides; the time mean plus the same quantity over the last TAIL_FRAC of the episode (the
end pose matters: a mug that finishes lying on its side). The errors use the same formulas as the object-tracking
termination term (ObjectTrackingExceeded): position = norm, rotation = 2·acos|q_obj · q_ref|. The tolerances are read
from the runtime termination-term cfg and written to the file meta (tracked_sides).
Rank rule (rank_key, chosen by the recorder class's RANK_BY): RANK_SCORE = smallest score (default) · RANK_END =
smallest normalised error at the last frame, ties broken by score (the success test allows lagging tracking, so a
success can end with the object off target — RANK_END does not pick those rollouts). The success definition does not
depend on the rank rule.
"""

from __future__ import annotations

import numpy as np

TAIL_FRAC = 0.10
RANK_SCORE = "score"
RANK_END = "end"
RANK_RULE = {
    RANK_SCORE: "lowest score among the success candidates (success ∧ real batch ∧ PARK valid)",
    RANK_END: "among the success candidates (success ∧ real batch ∧ PARK valid), the lowest side mean of the last "
    "frame's (position error / position threshold + rotation error / rotation threshold) / 2 · ties by lowest score",
}


def failure_key(length: int, sc: float) -> tuple[int, float]:
    """Archive rank of a no-success clip's failure (failure_best, not a rendered video): longest failure first (most
    progress), then smallest score; smaller is better. By score alone a short early failure would win on its low
    mean."""
    return (-int(length), float(sc))


def errors(states: np.ndarray, obj_lo: int, ref_lo: int) -> tuple[np.ndarray, np.ndarray]:
    """[L] position error (m) and rotation error (rad) between the object root columns (obj_lo: pos 3 + quat 4) and
    the reference columns (ref_lo, same layout)."""
    p, q = states[:, obj_lo : obj_lo + 3], states[:, obj_lo + 3 : obj_lo + 7]
    rp, rq = states[:, ref_lo : ref_lo + 3], states[:, ref_lo + 3 : ref_lo + 7]
    pe = np.linalg.norm(p.astype(np.float64) - rp, axis=-1)
    d = np.abs(np.sum(q.astype(np.float64) * rq, axis=-1))
    re = 2.0 * np.arccos(np.clip(d, 0.0, 1.0))
    return pe, re


def _norm_err(states: np.ndarray, sides: list[dict]) -> np.ndarray:
    """[L] per frame, the side mean of (position error / position tolerance + rotation error / rotation tolerance) / 2.
    Dies on an empty side list or a zero-length trajectory."""
    if not sides:
        raise ValueError("[traj_select] no tracked side — cannot score")
    if states.shape[0] < 1:
        raise ValueError("[traj_select] zero-length trajectory")
    per = []
    for s in sides:
        pe, re = errors(states, int(s["obj_lo"]), int(s["ref_lo"]))
        per.append(0.5 * (pe / float(s["pos_tol"]) + re / float(s["rot_tol"])))
    return np.mean(np.stack(per, 0), axis=0)


def score(states: np.ndarray, sides: list[dict], tail_frac: float = TAIL_FRAC) -> float:
    """Episode score (module docstring). sides = [{obj_lo, ref_lo, pos_tol, rot_tol}]; an empty list dies (the idle
    hand's score is never used instead)."""
    e = _norm_err(states, sides)
    n_tail = max(1, int(round(tail_frac * e.shape[0])))
    return float(e.mean() + e[-n_tail:].mean())


def end_error(states: np.ndarray, sides: list[dict]) -> float:
    """Normalised error at the last frame (same formula as score, side mean) — the RANK_END criterion."""
    return float(_norm_err(states, sides)[-1])


def rank_key(by: str, states: np.ndarray, sides: list[dict], sc: float) -> tuple:
    """Candidate rank key (smaller is better), see RANK_RULE. An unknown rule dies."""
    if by == RANK_SCORE:
        return (float(sc),)
    if by == RANK_END:
        return (end_error(states, sides), float(sc))
    raise ValueError(f"[traj_select] unknown rank rule {by!r} — {tuple(RANK_RULE)}")


def end_err(errs: dict) -> dict:
    """Per-side tracking summaries (end_pos_m, end_rot_deg) -> episode end error {pos_m, rot_deg} (max over sides)."""
    return {
        "pos_m": max(e["end_pos_m"] for e in errs.values()),
        "rot_deg": max(e["end_rot_deg"] for e in errs.values()),
    }


def tracked_sides(raw_env, ents: list[dict], term_cls=None) -> list[dict]:
    """[{side, entity, pos_tol, rot_tol, progress}] for the object sides (env.held_object_assets) that have an
    object-tracking termination term; zero sides dies. Tolerances are interpolated at the trajectory manager's
    schedule_progress. term_cls = the tracking termination class (default ObjectTrackingExceeded)."""
    if term_cls is None:
        from dexengine.rl.tasks.mdp.terminations.task import ObjectTrackingExceeded as term_cls
    held = raw_env.held_object_assets()
    tm = raw_env.termination_manager
    terms = {}
    for cfg in getattr(tm, "_class_term_cfgs", []) or []:
        if isinstance(cfg.func, term_cls):
            p = cfg.params
            terms[str(p.get("side", "right"))] = p
    prog = float(getattr(raw_env.trajectory_manager, "schedule_progress", 1.0))
    by_id = {}
    for e in ents:
        a = (
            raw_env.scene.articulations.get(e["name"])
            if e["kind"] == "artic"
            else raw_env.scene.rigid_objects.get(e["name"])
        )
        by_id[id(a)] = e["name"]
    out = []
    for side, obj in held:
        p = terms.get(side)
        if p is None:
            print(
                f"[TRAJ_REC] side {side}: has an object but no tracking termination term — left out of the ranking "
                "(marker)",
                flush=True,
            )
            continue
        name = by_id.get(id(obj))
        if name is None:
            raise RuntimeError(f"[TRAJ_REC] side {side}: the object is not in the entity list")
        ps, pe = float(p.get("pos_start", 0.04)), float(p.get("pos_end", 0.02))
        rs, re_ = float(p.get("rot_start_deg", 60.0)), float(p.get("rot_end_deg", 30.0))
        out.append(
            {
                "side": side,
                "entity": name,
                "pos_tol": ps + (pe - ps) * prog,
                "rot_tol": np.deg2rad(rs + (re_ - rs) * prog),
                "progress": prog,
            }
        )
    if not out:
        raise RuntimeError(
            "[TRAJ_REC] 0 tracked sides — no object side to rank (an idle hand's score is not used instead)"
        )
    return out
