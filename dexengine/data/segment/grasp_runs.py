"""Owner of grasp runs and the **sustained grasp** (D36) rule; a module independent of hand and data root (the piece
rule
segment.split and the TACO side assignment producer taco_side_assign read the same rule; no second verdict function).

  grasp_runs      case.grasp_s ({object: {hand: [[t0, t1] s]}}) → one hand's grasp sequence [[start, end, object]]
                  (frames; consecutive runs of the same object merged)
  sustained_runs  only sustained grasps of that sequence: ① a grasp lying wholly inside the same hand's grasp of another
                  object = holding together (assist), dropped ② grasps shorter than the minimum (min_s; callers pass
                  dexengine.data.candidates.MIN_DURATION_S) = brushes, dropped. Consecutive grasps of the same object
                  left after that merge into one.
  break_tie       breaks ties within one hand: smaller mean hand–object distance → earlier grasp start → else None.
"""

from __future__ import annotations


def grasp_runs(case: dict, side: str, fps: float) -> list[list]:
    """Grasp sequence of one hand (case.grasp_s): [[start, end, object]] (frames · time order · consecutive runs of the
    same object merged)."""
    seq = sorted(
        (int(round(float(iv[0]) * fps)), int(round(float(iv[1]) * fps)), str(o))
        for o, per in (case.get("grasp_s") or {}).items()
        for iv in ((per or {}).get(side) or [])
    )
    runs: list[list] = []
    for a, b, o in seq:
        if runs and runs[-1][2] == o:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b, o])
    return runs


def sustained_runs(allr: list[list], fps: float, min_s: float) -> list[list]:
    """Only the sustained grasps (① ② of the module docstring) of one hand's grasp sequence ([[start, end, object]]
    frames · time order)."""
    minf = int(min_s * fps)
    runs: list[list] = []
    for a, b, o in allr:
        if b - a < minf or any(q[2] != o and q[0] <= a and b <= q[1] for q in allr):
            continue
        if runs and runs[-1][2] == o:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b, o])
    return runs


def break_tie(cands: list[dict]) -> str | None:
    """Rule breaking ties within one hand (objects whose sustained grasp times differ by less than the minimum grasp):
    cands = [{obj, mean_gap_m, start}] (mean_gap_m = mean hand surface–object surface distance · start = start frame of
    that object's first sustained grasp). ① smaller mean distance ② then earlier grasp start ③ else None (the caller
    records a reason and stops)."""
    for key in ("mean_gap_m", "start"):
        best = min(c[key] for c in cands)
        win = [c for c in cands if c[key] == best]
        if len(win) == 1:
            return win[0]["obj"]
        cands = win
    return None
