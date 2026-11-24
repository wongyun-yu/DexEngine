"""E5 judge v1, the tool–target pair: the target rule (tool_target · TARGET_RULE · the one-body guard), the tool pose
in the target frame (rel_pose), the Pour rule and the Sweep contact travel (contact_points · contact_travel).

Part of the frozen v1 judge: judge_v1.py imports these back (callers read them as judge_v1.<name>, judge_v2 rebinds
judge_v1.contact_travel), and run_v2.py hashes this file together with judge_v1.py (v2_marks frozen_md5).
"""

from __future__ import annotations

import numpy as np

from experiments.sim_eval.judges.geometry import first_hold, longest, rot_deg, tilt_deg
from experiments.sim_eval.judges.judge_io import Scene, mesh_pts

ATTACHED_REL_M = (
    0.02  # "moves as one body with the tool": the tool moves at least this much while the candidate's position
)
# in the tool body frame spans less than this. An object that moves less than the sweep threshold
# (2 cm) relative to the tool cannot slide over it (measured "sliding" there is sample jitter);
# such candidates are other pieces of the same object (e.g. OakInk2 *_00001 / *_00002 parts).


def attached(kin: Scene, tool: str, n: str) -> bool:
    """True when object n moves as one body with the tool (see ATTACHED_REL_M)."""
    a, b = kin.o[tool], kin.o[n]
    if float(np.linalg.norm(a["p"] - a["p"][0], axis=1).max()) < ATTACHED_REL_M:
        return False  # a tool that hardly moves cannot "move together" (resting neighbours stay candidates)
    rel = np.einsum("tji,tj->ti", a["R"], b["p"] - a["p"])
    return float(np.ptp(rel, axis=0).max()) < ATTACHED_REL_M


TARGET_RULE = (
    "v3"  # target-rule version; its one owner is tool_target (the judge, the task classifier clip_features and
)
# the extra-slot table call that function). Written into judge records and the feature table (tgt_cand).
# v3 = background objects are candidates + the one-body guard.


def _moved_main(kin: Scene) -> str | None:
    mains = [n for n, v in kin.o.items() if v["meta"].get("role") == "main"]
    if not mains:
        return None
    return max(mains, key=lambda n: float(np.linalg.norm(kin.o[n]["p"] - kin.o[n]["p"][0], axis=1).max()))


def _nearest(kin: Scene, tool: str, names: list) -> str | None:
    return (
        min(names, key=lambda n: float(np.linalg.norm(kin.o[n]["p"] - kin.o[tool]["p"], axis=1).min()))
        if names
        else None
    )


def tool_target(kin: Scene) -> tuple[str, str] | None:
    """(tool, target) under TARGET_RULE v3: tool = the main object that moves most; target = the other object that comes
    closest to the tool. Background-tagged objects are candidates (e.g. a background cup brushed inside); objects that
    move as one body with the tool (attached) are not (otherwise another piece of the same object becomes the "target"
    and sliding over itself counts as Sweep)."""
    tool = _moved_main(kin)
    if tool is None:
        return None
    tgt = _nearest(kin, tool, [n for n in kin.o if n != tool and not attached(kin, tool, n)])
    return (tool, tgt) if tgt is not None else None


def target_guard_hit(kin: Scene) -> bool:
    """Did the one-body guard change the target (feature table tgt_attached_skip)? True when the nearest object without
    the guard moves as one body with the tool."""
    tool = _moved_main(kin)
    if tool is None:
        return False
    near = _nearest(kin, tool, [n for n in kin.o if n != tool])
    return near is not None and attached(kin, tool, near)


def rel_pose(sc: Scene, tool: str, tgt: str) -> tuple[np.ndarray, np.ndarray]:
    """Tool position and rotation in the target's body frame, per row."""
    a, b = sc.o[tool], sc.o[tgt]
    return np.einsum("tji,tj->ti", b["R"], a["p"] - b["p"]), np.einsum("tji,tjk->tik", b["R"], a["R"])


def label_of(clip: str, ds: str) -> str:
    """Action label of a clip name (TACO: first '_' token; other datasets: before '__')."""
    return clip.split("_")[0] if ds == "taco" else clip.split("__")[0]


def _pour_a(P, lab, tilt_k, prel_k, tool, tgt, Rrel_k, kin, sc):
    """Variant A: reach the reference key pose of the tool relative to the target and hold it."""
    A = P["A"]
    keyset = "tilt" if any(lab == t or lab.endswith(t) or lab.startswith(t) for t in A["key"]["tilt"]) else "nearest"
    K = int(np.argmax(tilt_k)) if keyset == "tilt" else int(np.argmin(np.linalg.norm(prel_k, axis=1)))

    def cond(s):
        pr, Rr = rel_pose(s, tool, tgt)
        return (np.linalg.norm(pr - prel_k[K], axis=1) <= A["eps_pos_m"]) & (
            rot_deg(Rr, Rrel_k[K][None]) <= A["eps_rot_deg"]
        )

    H = max(A["hold_min_s"], min(A["hold_cap_s"], A["hold_frac"] * longest(cond(kin))))
    m = cond(sc)
    tip = tilt_deg(sc.o[tgt]["R"]) - float(tilt_deg(kin.o[tgt]["R"]).max())
    row = first_hold(m, H)
    tipped = (
        bool((tip > P["target_tip_deg"]).any()) if row is None else bool((tip[: row + 1] > P["target_tip_deg"]).any())
    )
    return {
        "ok": (row is not None and not tipped),
        "row": row,
        "variant": "A",
        "key": keyset,
        "K": K,
        "hold_s": round(H, 3),
        "tool": tool,
        "target": tgt,
        "target_tipped": tipped,
    }


def _pour_b(P, tilt_k, ds, sc, tool, tgt, kin):
    """Variant B: tool tilted past the threshold with its lowest point over the target footprint, held."""
    B = P["B"]
    thr = min(B["tilt_cap_deg"], B["tilt_frac"] * float(tilt_k.max()))
    tp, gp = mesh_pts(ds, sc.o[tool]["meta"], 1500), mesh_pts(ds, sc.o[tgt]["meta"], 1500)
    rad = float(np.percentile(np.linalg.norm(gp[:, :2], axis=1), 95)) + B["footprint_margin_m"]
    extra = {}
    if B["ref_radius_delta_m"] is not None:
        # B′: reference radius d_ref = the smallest r for which the reference (kinematic) holds "tilt ≥ threshold and
        # horizontal distance of the lowest point ≤ r" for hold_s (candidates = horizontal distances of tilted frames,
        # ascending); footprint = max(target p95 + margin, d_ref + δ). No d_ref (reference never passes the threshold)
        # → the reference fails → not judgeable.
        tk, dk = _tilt_low_dist(kin, tool, tgt, tp)
        cand = sorted(set(np.round(dk[tk >= thr], 4).tolist()))
        dref = next(
            (float(r) for r in cand if first_hold((tk >= thr) & (dk <= r + 1e-9), B["hold_s"]) is not None), None
        )
        if dref is None:
            return {
                "ok": False,
                "row": None,
                "variant": "B′",
                "why": "B′: the reference does not pass the tilt threshold (no d_ref)",
                "tilt_thr_deg": round(thr, 2),
                "tool": tool,
                "target": tgt,
            }
        extra = {
            "variant": "B′",
            "d_ref_m": round(dref, 4),
            "delta_m": B["ref_radius_delta_m"],
            "footprint_target_m": round(rad, 4),
        }
        rad = max(rad, dref + B["ref_radius_delta_m"])
    a, b = sc.o[tool], sc.o[tgt]
    world = np.einsum("tij,nj->tni", a["R"], tp) + a["p"][:, None]
    low = world[np.arange(sc.T), np.argmin(world[:, :, 2], axis=1)]
    m = (tilt_deg(a["R"]) >= thr) & (np.linalg.norm(low[:, :2] - b["p"][:, :2], axis=1) <= rad)
    row = first_hold(m, B["hold_s"])
    out = {
        "ok": row is not None,
        "row": row,
        "variant": "B",
        "tilt_thr_deg": round(thr, 2),
        "footprint_m": round(rad, 4),
        "tool": tool,
        "target": tgt,
    }
    out.update(extra)  # B′ overrides variant and adds d_ref, δ and the target footprint (empty for plain B)
    return out


def pour(kin: Scene, sc: Scene, P: dict, lab: str, ds: str, pair: tuple[str, str], ref_scene: bool) -> dict:
    """Pour. Variant A: reach the reference key pose of the tool relative to the target (max-tilt frame for tilt labels,
    nearest frame otherwise) and hold it, without tipping the target. Variant B (/B′): tool tilted past a threshold
    with
    its lowest point over the target footprint."""
    tool, tgt = pair
    if tool not in sc.o or tgt not in sc.o:
        return {"ok": False, "why": f"tool/target {tool}/{tgt} not in the scene"}
    prel_k, Rrel_k = rel_pose(kin, tool, tgt)
    tilt_k = tilt_deg(kin.o[tool]["R"])
    if P["variant"] == "A":
        return _pour_a(P, lab, tilt_k, prel_k, tool, tgt, Rrel_k, kin, sc)
    return _pour_b(P, tilt_k, ds, sc, tool, tgt, kin)


def _tilt_low_dist(s: Scene, tool: str, tgt: str, tp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tool tilt (T,), horizontal distance tool lowest point ↔ target centre (T,)) — the quantities of variant B."""
    a, b = s.o[tool], s.o[tgt]
    world = np.einsum("tij,nj->tni", a["R"], tp) + a["p"][:, None]
    low = world[np.arange(s.T), np.argmin(world[:, :, 2], axis=1)]
    return tilt_deg(a["R"]), np.linalg.norm(low[:, :2] - b["p"][:, :2], axis=1)


def contact_points(sc: Scene, tool: str, tgt: str, ds: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row, the tool point nearest to the target (shared by contact_travel and clip_motion.top_slide):
    (min surface distance (T,), that tool point in the target body frame (T, 3), its index in mesh_pts(…, 1500)
    (T,))."""
    from scipy.spatial import cKDTree

    tp, gp = mesh_pts(ds, sc.o[tool]["meta"], 1500), mesh_pts(ds, sc.o[tgt]["meta"], 1500)
    kd = cKDTree(gp)
    pr, Rr = rel_pose(sc, tool, tgt)
    dmin, cpt, jj = np.zeros(sc.T), np.zeros((sc.T, 3)), np.zeros(sc.T, dtype=int)
    for t in range(sc.T):
        x = tp @ Rr[t].T + pr[t]
        d, _i = kd.query(x)
        j = int(np.argmin(d))
        dmin[t], cpt[t], jj[t] = d[j], x[j], j
    return dmin, cpt, jj


def contact_travel(
    sc: Scene, tool: str, tgt: str, ds: str, dc: float, pts: tuple | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """(travel while in contact, per-row min surface distance, per-row cumulative contact travel (row 0 = 0)). Travel =
    path length of the tool contact point in the target body frame. pts = an already measured contact_points result
    for the same scene, tool and target."""
    dmin, cpt, _jj = pts if pts is not None else contact_points(sc, tool, tgt, ds)
    c = dmin <= dc
    both = c[1:] & c[:-1]
    step = np.where(both, np.linalg.norm(np.diff(cpt, axis=0), axis=1), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    return float(cum[-1]), dmin, cum
