"""E5 clip curation, motion features of one clip (called by experiments/sim_eval/scenes/clip_features.py): object
lowest point, sideways slide on the target's top face (top_slide), tilt / position over the target (over_target),
slide on the support (sup_slide) and the fingertips in contact (tips_of). Thresholds below are the values the
feature tables were measured with.
"""

from __future__ import annotations

import numpy as np

from experiments.sim_eval.judges import judge_v1 as J

# on the support = object lowest point within desk z + 3 mm (1 cm counted the slanted motion of lifting/placing as
# sliding)
SUP_ON_M = 0.003
DEEP_M, CONTAINER_H_M = (
    0.015,
    0.04,
)  # inside a container = target height >= 4 cm and tip >= 1.5 cm below the top (excludes plates, boards)
POUR_TILT_DEG, FOOT_MARGIN_M = (
    45.0,
    0.04,
)  # tilted over the target (task signature Pour) — the judge's tilt limit and footprint margin
DC_M = 0.01  # contact = nearest-surface distance <= 1 cm (sweep contact slide and top-face slide)


def low_z(ds: str, x: dict, p: np.ndarray, R: np.ndarray) -> np.ndarray:
    """World z of the object's lowest mesh point per row."""
    pts = J.mesh_pts(ds, x["meta"], 400)
    return p[:, 2] + np.einsum("tj,nj->tn", R[:, 2, :], pts).min(axis=1)


def runs(mask: np.ndarray):
    """Runs of True as (i, j), both ends included."""
    i, T = 0, len(mask)
    while i < T:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < T and mask[j + 1]:
            j += 1
        yield i, j
        i = j + 1


def top_slide(kin, tool: str, tgt: str, ds: str, pts: tuple, pose: np.ndarray, dc: float) -> dict:
    """Sideways slide on the target's top face (task rule (1) — the Sweep signature input). Insertion (contact running
    down a hole or wall), placing, the vertical component and rows below the top face are not counted. Operational
    definition (this function owns it):
      top direction up = the target body axis closest to world up at row 0 (sign included) — measured along the top
      face
        when the target is lifted and tilted (in world height, the rim of a hand-held tilted TACO plate became the top
        and rubbing on the plate fell 'below the top').
      on-top row = contact (nearest distance <= dc) ∧ contact point inside the target footprint (in the plane normal to
        up: target radius p95 + FOOT_MARGIN_M) ∧ not deep (for targets with height along up >= CONTAINER_H_M, below
        top − DEEP_M is deep; lower targets — plates, trays, boards — are all top face).
      amount = over each run of on-top rows, the maximum displacement normal to up of the contact material point at the
        run's first row (that point of the tool body) relative to the target, summed (m) — displacement, not path
        length (like sup_slide: in long insertions, contact-point jitter added up to several cm).
    slide_top_m = all rows · slide_top_np_m = without pour-pose rows (containers drop pour-pose rows, tools keep them —
    rule (4) was a draft that was never applied) · top_contact_frac = share of on-top rows among contact rows
    (explains whether a miss is 'contact below the top' or 'on top but small sideways displacement')."""
    dmin, cpt, jj = pts
    a, b = kin.o[tool], kin.o[tgt]
    tp, gp = J.mesh_pts(ds, a["meta"], 1500), J.mesh_pts(ds, b["meta"], 1500)
    k = int(np.argmax(np.abs(b["R"][0][2, :])))
    up = np.zeros(3)
    up[k] = float(np.sign(b["R"][0][2, k])) or 1.0
    hgt = gp @ up
    top, h = float(hgt.max()), float(np.ptp(hgt))
    r = float(np.percentile(np.linalg.norm(gp - np.outer(hgt, up), axis=1), 95))
    pr, Rr = J.rel_pose(kin, tool, tgt)
    ch = cpt @ up
    foot = np.linalg.norm(cpt - np.outer(ch, up), axis=1) <= r + FOOT_MARGIN_M
    deep = (ch < top - DEEP_M) if h >= CONTAINER_H_M else np.zeros(kin.T, dtype=bool)
    on = (dmin <= dc) & foot & ~deep
    out = {}
    for col, m in (("slide_top_m", on), ("slide_top_np_m", on & ~pose)):
        tot = 0.0
        for i, j in runs(m):
            q = (
                np.einsum("tij,j->ti", Rr[i : j + 1], tp[jj[i]]) + pr[i : j + 1]
            )  # target-frame path of the run's first contact material point
            dq = q - q[0]
            dq = dq - np.outer(dq @ up, up)  # drop the up component
            tot += float(np.linalg.norm(dq, axis=1).max())
        out[col] = round(tot, 4)
    nc = int((dmin <= dc).sum())
    return dict(
        out,
        top_axis=f"{'+' if up[k] > 0 else '-'}{'xyz'[k]}",
        top_h_m=round(h, 4),
        top_contact_frac=(round(float(on.sum()) / nc, 4) if nc else ""),
    )


def _turns_and_slides(rel, inside, tilt, foot, cum, kin, h):
    """Turns of the tip around the target axis and the slide lengths split by pose."""
    th = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    both = inside[1:] & inside[:-1]
    dth = np.diff(th)[both]
    turns, net = float(np.abs(dth).sum() / (2 * np.pi)), float(abs(dth.sum()) / (2 * np.pi))
    pose = (
        np.abs(tilt - tilt[0]) >= POUR_TILT_DEG
    ) & foot  # pour pose = tilt change from start >= 45° with the tip in the footprint
    slide_np = float(np.diff(cum)[~pose[1:]].sum()) if cum is not None and len(cum) == kin.T else float("nan")
    slide_tray = (
        (float(np.diff(cum)[inside[1:]].sum()) if h < CONTAINER_H_M else 0.0)
        if cum is not None and len(cum) == kin.T
        else float("nan")
    )
    return net, turns, pose, slide_np, slide_tray


def over_target(
    kin,
    tool: str,
    tgt: str,
    ds: str,
    hz: float,
    cum: np.ndarray | None = None,
    pts: tuple | None = None,
    dc: float = 0.01,
) -> dict:
    """Motion of the tool's lowest point (tip) relative to the target — the task-signature input.
    tgt_r_m = target mesh horizontal radius p95 · footprint = tgt_r + FOOT_MARGIN_M.
    over_tilt_max_deg = max tool tilt while the tip is in the footprint · pour_over_s = longest run of 'tilt >=
    POUR_TILT_DEG ∧
    tip in footprint'.
    dtilt_over_s = longest run of 'tilt change from start >= POUR_TILT_DEG ∧ tip in footprint' (pour pose) — objects
    whose
    body z lies flat from the start (TACO knife, brush) always have a large absolute tilt, so the change is used ·
    slide_np_m = contact slide summed over rows not in pour pose (contact_travel cumulative).
    tgt_h_m = target height · deep_in_s = time the tip is deep inside a container (target height >= CONTAINER_H_M,
    inside
    the radius, below top − DEEP_M) · slide_deep_m = contact slide at that depth (stir-signature input — separates it
    from rubbing on a flat board).
    slide_tray_m = only for shallow containers (target height < CONTAINER_H_M): contact slide over rows with the tip
    inside (inside the radius, below the top — the same band as stir_in); rule (3) 'stirring on a shallow tray =
    rubbing
    inside a container' · 0 when height >= CONTAINER_H_M (deep containers use slide_deep).
    stir_in_s = time the tip is inside the target radius and below its top (inside the container) · stir_turns = summed
    angular motion of the tip around the target axis there / 360° (target-frame horizontal angle, |Δθ| between
    consecutive inside rows — back-and-forth counts) · stir_net_turns = |signed sum of the same Δθ| / 360° (net
    rotation
    in
    one direction — back-and-forth and crossings cancel)."""
    a, b = kin.o[tool], kin.o[tgt]
    tp, gp = J.mesh_pts(ds, a["meta"], 1500), J.mesh_pts(ds, b["meta"], 1500)
    r = float(np.percentile(np.linalg.norm(gp[:, :2], axis=1), 95))
    world = np.einsum("tij,nj->tni", a["R"], tp) + a["p"][:, None]
    low = world[np.arange(kin.T), np.argmin(world[:, :, 2], axis=1)]
    tilt = J.tilt_deg(a["R"])
    dh = np.linalg.norm(low[:, :2] - b["p"][:, :2], axis=1)
    foot = dh <= r + FOOT_MARGIN_M
    top = (np.einsum("tij,nj->tni", b["R"], gp) + b["p"][:, None])[:, :, 2].max(axis=1)
    rel = np.einsum("tji,tj->ti", b["R"], low - b["p"])  # tip in the target body frame
    inside = (np.linalg.norm(rel[:, :2], axis=1) <= r) & (low[:, 2] <= top)
    gz = gp[:, 2]
    h = float(gz.max() - gz.min())  # target height (body-frame z range)
    deep = (
        (np.linalg.norm(rel[:, :2], axis=1) <= r) & (low[:, 2] <= top - DEEP_M) & (h >= CONTAINER_H_M)
    )  # deep inside a container
    slide_deep = float(np.diff(cum)[deep[1:]].sum()) if cum is not None and len(cum) == kin.T else float("nan")
    net, turns, pose, slide_np, slide_tray = _turns_and_slides(rel, inside, tilt, foot, cum, kin, h)
    top = top_slide(kin, tool, tgt, ds, pts, pose, dc) if pts is not None else {}
    return dict(
        tgt_r_m=round(r, 4),
        over_tilt_max_deg=(round(float(tilt[foot].max()), 2) if foot.any() else 0.0),
        pour_over_s=round(J.longest((tilt >= POUR_TILT_DEG) & foot, hz), 3),
        dtilt_over_s=round(J.longest(pose, hz), 3),
        slide_np_m=round(slide_np, 4),
        stir_in_s=round(float(inside.sum()) / hz, 3),
        stir_turns=round(turns, 3),
        stir_net_turns=round(net, 3),
        tgt_h_m=round(h, 4),
        deep_in_s=round(float(deep.sum()) / hz, 3),
        slide_deep_m=round(slide_deep, 4),
        slide_np_nd_m=(
            round(float(np.diff(cum)[~(pose | deep)[1:]].sum()), 4)
            if cum is not None and len(cum) == kin.T
            else float("nan")
        ),
        slide_tray_m=round(slide_tray, 4),
        **top,
    )


def sup_slide(p: np.ndarray, lz: np.ndarray, desk) -> float | str:
    """Slide on the support (desk): over each run where the object's lowest point is within desk z + SUP_ON_M, the
    maximum horizontal displacement from the run's first row, summed (m). Not path length — pose noise of resting
    objects added up and HRDexDB / GRAB lifts (352 · 167) were caught as 'sliding'. Empty for scenes without a desk."""
    if desk is None:
        return ""
    on = (lz - float(desk)) <= SUP_ON_M
    tot, i, T = 0.0, 0, len(on)
    while i < T:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < T and on[j + 1]:
            j += 1
        tot += float(np.linalg.norm(p[i : j + 1, :2] - p[i, :2], axis=1).max())
        i = j + 1
    return round(tot, 4)


def tips_of(contact: np.ndarray, hands: list, side: str) -> np.ndarray:
    """Fingertip contact (T, 5·hands) → number of contacting fingertips of the object's hand (T,); shared (both) = sum
    of both hands.
    """
    n = contact.shape[1] // max(1, len(hands))
    cols = [i for i, h in enumerate(hands) if side in (h, "both")]
    if not cols:
        return np.zeros(len(contact))
    return sum(contact[:, i * n : (i + 1) * n].sum(axis=1) for i in cols)
