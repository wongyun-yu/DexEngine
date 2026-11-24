#!/usr/bin/env python3
"""Arm ↔ torso surface — the torso part of the one arm ruler (arm_surface; read through collision.core).

Shapes are read, never generated: the torso = sim_collider.torso_bodies (root frame — the static ``world`` body, torso
joints fixed; one hull per part of the decomposition rule torso_split, keyed by body name, paired as its link); the
moving bodies = arm_surface.build_surface (capsules with r · revision hulls) + hand_surface.hand_tree (palm · fingers).
Pairs = TORSO_PAIRS: "vendor_colaffinity" — an arm body meets a torso link when the vendor bitmasks pair them
((ca_a & ct_b) or (ca_b & ct_a)); a link with no vendor collider takes its nearest URDF ancestor's (arm_6 · palm →
arm_5, torso_3 → torso_2) — a rule, not a table · "all_arm_links" — every arm body meets every torso link but its URDF
joint neighbour (arm_0 ↔ torso_5 touch at the joint in every pose).
Two layers, one ruler:
  guard   (collision.core.pen_cross_guard, every adoption site) exact: capsule ↔ hull = hull surface samples to the
          axis − r + an axis-inside check; hull ↔ hull = deepest surface sample of either inside the other. Frames and
          retiming chords; bodies = arm + palm + fingers (the clip's finger values on the same time axis —
          hand_surface.guard_finger_q; none → death); tolerance TORSO_OVERLAP_MAX_MM; pairs proven clear skipped
          (guard_band).
  filter  (the search — collision.core.desk_penetration(_each)(_t)) a signed-distance grid of the static torso read at
          the moving bodies' points (capsule axis samples − r, hull vertices) + the torso vertices inside each moving
          hull (numpy = torch formula). Bodies = arm + palm (clip-free), a subset of the guard's, so the filter is
          meant not to reject what the guard accepts (torso_guard reports it). Proxy geometry: arm_proxy.COLL_PROXY.
In-hull lower bound (``lb_planes`` · ``in_hull_lb`` · ``torso_torch.lb_t``): the filter's in-hull term (min over the
torso vertices of the moving hull's max plane value) is read exactly only where it can change the value — a pose keeps
its grid value when the bound already exceeds it (min(grid, in-hull) = grid). The bound = the max over LB_PLANES of the
hull's planes (farthest-point on the unit normals) — one of the full max's arguments, so never above it; LB_EPS_M
absorbs the rounding of carrying those planes to world. A bound only chooses rows, so its arithmetic (numpy · eager
torch · compiled torch) never reaches a value. Markers: ``[TORSO_RULER]`` · ``[TORSO_HANDS]`` (per side · URDF) ·
``[TORSO_GUARD] filter≠guard`` (each disagreement, with the deciding body — a finger there is the guard's own reach)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError
from dexengine.placement.collision import arm_proxy as AP
from dexengine.placement.collision import guard_band as GB
from dexengine.placement.collision import hand_surface as HS
from dexengine.placement.collision import sdf_grid as HG
from dexengine.placement.collision.arm_surface import hull_samples, pt_seg
from dexengine.placement.collision.geom import arm_link_transforms, capsule_endpoints, load_capsules

TORSO_OVERLAP_MAX_MM = 2.0  # allowed arm ↔ torso surface overlap (the strict 2 mm)
#: Which arm bodies meet which torso links — "vendor_colaffinity" | "all_arm_links" (module doc).
TORSO_PAIRS = "vendor_colaffinity"
if TORSO_PAIRS not in ("vendor_colaffinity", "all_arm_links"):
    raise SystemExit(f"[TORSO_RULER] TORSO_PAIRS={TORSO_PAIRS!r} — vendor_colaffinity | all_arm_links")
AXIS_M = 0.005  # capsule axis sample spacing (filter)
NEAR_M = 0.040  # the guard measures exactly the (body, pose) pairs whose filter distance is below this
SKIP_M = 0.020  # filter: a body whose bounding sphere is this far from the torso keeps the sphere bound
DENSE_M = 0.004  # torso surface sample spacing (grid outside distance · exact guard)
CRS_M = 0.012  # coarse moving-hull samples: the guard's grid prescreen runs on these first (exact, see _coarse)
CHUNK = 1024  # poses per block where an array grows with poses × torso vertices × planes (numpy · torch)
LB_PLANES = 24  # planes of the in-hull lower bound
LB_EPS_M = 1e-9  # rounding margin of the bound carried to world
_TORSO_CACHE: dict = {}
_SERVER_KEEP = ("_TORSO_CACHE",)  # URDF-keyed pure cache (torso hulls · grids)


def tol_m() -> float:
    """TORSO_OVERLAP_MAX_MM in metres."""
    return TORSO_OVERLAP_MAX_MM / 1000.0


def _said(key: str, msg: str) -> None:
    if ("_said", key) not in _TORSO_CACHE:
        _TORSO_CACHE[("_said", key)] = True
        print(msg, flush=True)


def _hull_w(v: np.ndarray) -> dict:
    from scipy.spatial import ConvexHull, cKDTree

    from dexengine.robots import wrist_hull as G

    ctr = v.mean(0)
    smp = hull_samples(v, G.hull_triangles(v), DENSE_M)
    return {
        "v": v,
        "eq": ConvexHull(v).equations,
        "ctr": ctr,
        "rad": float(np.linalg.norm(v - ctr, axis=1).max()),
        "smp": smp,
        "kd": cKDTree(smp),
    }


def _parents(urdf: str) -> dict:
    return {
        j.find("child").get("link"): (j.find("parent").get("link"), j.get("type"))
        for j in ET.parse(urdf).getroot().findall("joint")
    }


def _masks(link: str, parent: dict, vend: dict) -> tuple[int, int]:
    """(coltype, colaffinity) of a link: its vendor capsule's, else its nearest URDF ancestor's that has one."""
    cur = link
    while cur not in vend:
        if cur not in parent:
            raise InputMissingError(
                f"[TORSO_RULER] {link}: no ancestor link with a vendor collider — cannot derive the pair masks"
            )
        cur = parent[cur][0]
    return vend[cur]


def _pair(link: str, torso: dict, parent: dict, vend: dict) -> tuple[str, ...]:
    """Torso bodies an arm body meets (TORSO_PAIRS) — a part by its link."""
    tl = {t: torso[t]["link"] for t in sorted(torso)}
    if TORSO_PAIRS == "all_arm_links":
        return tuple(t for t, ln in tl.items() if parent.get(link, ("",))[0] != ln and parent.get(ln, ("",))[0] != link)
    ct, ca = _masks(link, parent, vend)
    return tuple(t for t, ln in tl.items() if (ca & _masks(ln, parent, vend)[0]) or (_masks(ln, parent, vend)[1] & ct))


def _torso(arm: str, urdf: str) -> dict:
    key = ("torso", urdf)
    if key not in _TORSO_CACHE:
        from dexengine.placement.collision import sim_collider as SC

        hs = {b.name: dict(_hull_w(np.asarray(b.verts, float)), link=b.link) for b in SC.torso_bodies(arm, urdf)}
        if not hs:
            raise InputMissingError(
                f"[TORSO_RULER] {arm}: no torso colliders stated (torso_replace · torso_add of the "
                "arms collider revision)"
            )
        _TORSO_CACHE[key] = hs
    return _TORSO_CACHE[key]


def _grid(torso: dict, mask: tuple, urdf: str) -> dict:
    """The signed-distance grid of the union of the masked torso hulls (dexengine.placement.collision.sdf_grid; shell ·
    margin: arm_proxy
    .grid_shell), cached in memory per URDF and on disk in the geometry cache (link_shapes.MESH_CACHE)."""
    key = ("grid", urdf, mask, AP.kind())
    if key not in _TORSO_CACHE:
        from dexengine.placement.collision import link_shapes as LS

        g, msg = HG.grid(
            [torso[t] for t in mask], *AP.grid_shell(NEAR_M + 3 * HG.GRID_M, HG.MARGIN_M), cache_dir=Path(LS.MESH_CACHE)
        )
        _said(f"grid:{msg}", f"[TORSO_RULER] {msg}")
        _TORSO_CACHE[key] = g
    return _TORSO_CACHE[key]


def _hull_body(link: str, k: int, v, mask: tuple) -> dict:
    h = _hull_w(np.asarray(v, float))
    return {
        "kind": "hull",
        "link": link,
        "k": k,
        "mask": mask,
        "hull": h,
        "pts": h["v"],
        "r": 0.0,
        "ctr": h["ctr"],
        "rad": h["rad"],
    }


def _cap_body(c: dict, link: str, k: int, mask: tuple) -> dict:
    n = max(2, int(np.ceil(float(c["L"]) / AXIS_M)) + 1)
    az = np.asarray(c["R"], float) @ np.array([0.0, 0.0, 1.0])
    return {
        "kind": "cap",
        "link": link,
        "k": k,
        "mask": mask,
        "cap": c,
        "r": float(c["r"]),
        "ctr": np.asarray(c["p"], float),
        "pts": c["p"][None] + (np.linspace(-0.5, 0.5, n) * float(c["L"]))[:, None] * az[None],
        "rad": 0.5 * float(c["L"]),
    }


def _report_torso(torso, arm, side, urdf, bodies, grids, ht, fingers):
    """The torso parts and the ruler's body/finger counts (once per arm side)."""
    nparts: dict[str, int] = {}
    for t in torso.values():
        nparts[t["link"]] = nparts.get(t["link"], 0) + 1
    tdesc = ",".join(f"{ln.removeprefix('link_')}×{n}" for ln, n in nparts.items())
    _said(
        f"{arm}/{side}/{urdf}",
        f"[TORSO_RULER] surface · arm–torso = surface · tolerance {TORSO_OVERLAP_MAX_MM:g} mm"
        f" · pairs {TORSO_PAIRS} · "
        f"{Path(urdf).stem}/{side} · torso (parts) {tdesc} · "
        + " ".join(
            f"{b['link'].split('_')[-1] if b['kind'] == 'cap' or b['link'] != f'{side}_palm' else 'palm'}→"
            + "/".join(dict.fromkeys(torso[t]["link"].split("_")[-1] for t in b["mask"]))
            for b in bodies
        )
        + f" · grids {len(grids)} · in-hull bound planes {LB_PLANES} (margin {LB_EPS_M:g} m)",
    )
    _said(
        f"hands/{side}/{urdf}",
        f"[TORSO_HANDS] {Path(urdf).stem}/{side}: palm (links "
        f"{len(ht['groups'][0]['links'])}) + finger bodies {len(fingers)} (finger joints {ht['n']}) — fingers are "
        "placed by the clip's finger values on the same time axis",
    )


def _hand_part(ctx: dict, arm: str, hand: str, side: str) -> tuple:
    """(hand tree, finger groups with a shape) — the hand half of a torso record (the support surface reads it)."""
    from dexengine.placement.collision import sim_collider as SC

    ht = HS.hand_tree(ctx["urdf"], SC.arm_link_names(arm, side)[-1], HS.finger_names(hand, side))
    return ht, [gi for gi, g in enumerate(ht["groups"]) if gi and g["verts"] is not None]


def build_torso(ctx: dict, arm: str, hand: str, side: str) -> dict:
    """Torso hulls, the moving bodies with their pair masks and filter points, one grid per mask. bodies = arm + palm
    (filter's and guard's, k = arm joint index); fingers (guard's only, k = arm joints + i on the frames torso_guard
    appends: group fsel[i] of the hand tree; mesh: + the guard band). No torso (ArmSpec.torso_ruler false): empty."""
    from dexengine.core.arms import ARM_REGISTRY
    from dexengine.placement.collision import sim_collider as SC

    if not ARM_REGISTRY[arm].torso_ruler:  # no torso bodies → the filter and the guard measure nothing
        ht, fsel = _hand_part(ctx, arm, hand, side)
        _said(f"none/{arm}", f"[TORSO_RULER] {arm}: no torso (ArmSpec.torso_ruler false) — filter and guard off")
        none = {"torso": {}, "bodies": [], "grids": {}, "tverts": {}}
        return {**none, "hand": ht, "fsel": fsel, "fingers": [], "proxy": AP.kind()}
    urdf = ctx["urdf"]
    torso = _torso(arm, urdf)
    parent = _parents(urdf)
    vend = {c["link"]: (c["ct"], c["ca"]) for c in load_capsules(urdf)}
    sf, bodies = ctx["surf"], []
    bodies.extend(_cap_body(c, c["link"], int(c["k"]), _pair(c["link"], torso, parent, vend)) for c in sf["caps"])
    bodies.extend(
        {
            "kind": "hull",
            "link": h["link"],
            "k": int(h["k"]),
            "mask": _pair(h["link"], torso, parent, vend),
            "hull": h,
            "pts": h["v"],
            "r": 0.0,
            "ctr": h["ctr"],
            "rad": h["rad"],
        }
        for h in sf["hulls"]
    )
    last, nj = SC.arm_link_names(arm, side)[-1], len(ctx["joints"])
    ht, fsel = _hand_part(ctx, arm, hand, side)
    proxy = AP.capsule()
    bodies.append(
        (AP.palm_body if proxy else _hull_body)(
            f"{side}_palm", nj - 1, ht["groups"][0]["verts"], _pair(last, torso, parent, vend)
        )
    )
    fingers = []
    for i, g in enumerate(ht["groups"][gi] for gi in fsel):
        m = _pair(g["root"], torso, parent, vend)
        fingers.append(
            _cap_body(AP.finger_cap(hand, side, g), g["root"], nj + i, m)
            if proxy
            else _hull_body(g["root"], nj + i, g["verts"], m)
        )
        if not proxy:
            fingers[-1]["hull"]["band"] = GB.band(AP.hand_key(hand, side, g["root"]), g["verts"])
    bodies, fingers = [b for b in bodies if b["mask"]], [b for b in fingers if b["mask"]]
    for b in bodies + fingers:
        if b["kind"] == "hull":
            b["lb"] = lb_planes(b["hull"]["eq"])
    grids = {b["mask"]: _grid(torso, b["mask"], urdf) for b in bodies + fingers}
    tverts = {m: np.concatenate([torso[t]["v"] for t in m]) for m in grids}
    _report_torso(torso, arm, side, urdf, bodies, grids, ht, fingers)
    AP.mark_torso(f"{Path(urdf).stem}/{side}", bodies, fingers)
    return {
        "torso": torso,
        "bodies": bodies,
        "grids": grids,
        "tverts": tverts,
        "hand": ht,
        "fsel": fsel,
        "fingers": fingers,
        "proxy": AP.kind(),
    }


# ── in-hull lower bound (filter rows only) ─────────────────────────────────────────────────────────────────


def lb_planes(eq: np.ndarray) -> np.ndarray:
    """The bound's planes (LB_PLANES rows of a hull's eq): farthest-point on the unit normals from the +x-most one —
    spread over every direction, so an elongated hull keeps its caps and sides. Fewer planes than that = all."""
    if len(eq) <= LB_PLANES:
        return np.asarray(eq, float)
    sel, d = [int(np.argmax(eq[:, 0]))], np.full(len(eq), np.inf)
    for _ in range(LB_PLANES - 1):
        d = np.minimum(d, 1.0 - eq[:, :3] @ eq[sel[-1], :3])
        sel.append(int(np.argmax(d)))
    return np.asarray(eq[sel], float)


def in_hull_lb(tv, E, R, t, chunk: int) -> np.ndarray:
    """Bound per pose (numpy): min over tv (V, 3) of the max over the planes E (K, 4) carried to world (R n, d − R n·t)
    — the vertices are not transformed. R (P, 3, 3) · t (P, 3) · chunk = caller's block (CHUNK)."""
    out = np.empty(len(t))
    nw = np.einsum("tij,fj->tfi", R, E[:, :3])
    dw = E[:, 3][None] - np.einsum("tfi,ti->tf", nw, t)
    for i in range(0, len(t), chunk):
        pv = np.matmul(tv[None], nw[i : i + chunk].transpose(0, 2, 1)) + dw[i : i + chunk, None, :]  # (C, V, K)
        out[i : i + chunk] = pv.max(-1).min(-1)
    return out


# ── filter (the search) ─────────────────────────────────────────────────────────────────────────────────────


def _torso_in_hull(tz: dict, b: dict, R, t) -> np.ndarray:
    """min over the masked torso vertices of the moving hull's max plane value (negative = a torso vertex inside).
    Poses in blocks of CHUNK (poses × torso vertices × planes is the peak array)."""
    tv, eq, out = tz["tverts"][b["mask"]], b["hull"]["eq"], np.empty(len(t))
    for i in range(0, len(t), CHUNK):
        pb = np.einsum("tji,tmj->tmi", R[i : i + CHUNK], tv[None] - t[i : i + CHUNK, None, :])  # world → body (Rᵀ)
        out[i : i + CHUNK] = (pb @ eq[:, :3].T + eq[:, 3]).max(-1).min(-1)
    return out


def _in_hull_min(tz: dict, b: dict, R, t, v) -> np.ndarray:
    """min(v, _torso_in_hull) per pose, the in-hull term read only on the poses whose lower bound does not clear v
    (elsewhere min(v, in-hull) = v: in-hull ≥ bound > v — module doc)."""
    j = np.nonzero(in_hull_lb(tz["tverts"][b["mask"]], b["lb"], R, t, CHUNK) <= v + LB_EPS_M)[0]
    v = v.copy()
    if len(j):
        v[j] = np.minimum(v[j], _torso_in_hull(tz, b, R[j], t[j]))
    return v


def _body_filter_sd(tz: dict, b: dict, Ts) -> np.ndarray:
    """One body's filter distance per pose: the bounding-sphere bound where it is ≥ SKIP_M, else the min over the body's
    points of the grid value (− r) and, for a hull, the torso vertices inside it."""
    g = tz["grids"][b["mask"]]
    Tk = Ts[:, b["k"]]
    R, t = Tk[:, :3, :3], Tk[:, :3, 3]
    sd = HG.interp(g, np.einsum("tij,j->ti", R, b["ctr"]) + t) - b["rad"] - b["r"]
    ix = np.nonzero(sd < SKIP_M)[0]
    if len(ix):
        sd[ix] = (HG.interp(g, np.einsum("tij,nj->tni", R[ix], b["pts"]) + t[ix, None, :]) - b["r"]).min(-1)
        if b["kind"] == "hull":
            sd[ix] = _in_hull_min(tz, b, R[ix], t[ix], sd[ix])
    return sd


def torso_filter_sd(tz: dict, Ts) -> np.ndarray:
    """Filter signed surface distance arm/palm ↔ torso per pose (P,) [m]."""
    out = np.full(Ts.shape[0], 1e9)
    for b in tz["bodies"]:
        out = np.minimum(out, _body_filter_sd(tz, b, Ts))
    return out


# ── guard (the decision) ────────────────────────────────────────────────────────────────────────────────────


def _exact_cap(H: dict, b: dict, T2) -> np.ndarray:
    """hull_cap_sd(torso hull, capsule) restricted to the torso samples that can lie within r of the axis (KD ball
    around the capsule centre, radius L/2 + r) + an axis-inside check — identical below the tolerance, a bound above
    it."""
    a1, b1 = capsule_endpoints(b["cap"], T2[:, b["k"]])
    r, L = b["r"], 2.0 * b["rad"]
    sd = np.full(len(T2), 1e9)
    lists = H["kd"].query_ball_point(0.5 * (a1 + b1), 0.5 * L + r + 0.005)
    lens = np.fromiter((len(x) for x in lists), int, len(lists))
    if lens.sum():
        idx = np.concatenate([np.asarray(x, int) for x in lists if len(x)])
        pid = np.repeat(np.arange(len(T2)), lens)
        np.minimum.at(sd, pid, pt_seg(H["smp"][idx][:, None, :], a1[pid][:, None, :], b1[pid][:, None, :])[:, 0] - r)
    sd = np.where(sd > 1e8, 0.005, sd)
    axs = a1[:, None, :] + np.linspace(0, 1, 9)[None, :, None] * (b1 - a1)[:, None, :]
    return np.minimum(sd, (axs @ H["eq"][:, :3].T + H["eq"][:, 3]).max(-1).min(-1) - r)


def _coarse(A: dict) -> tuple:
    """(coarse samples, fine samples grouped by nearest coarse sample, cut) of a moving hull, cached on its record.
    A fine sample can read < 10 mm on the grid only if its coarse parent reads < cut = 10 mm + √3·(farthest
    fine→parent) + 5 mm (the trilinear grid is √3-Lipschitz; 5 mm covers the KD-distance step at the surface) — the
    prescreen keeps exactly the fine samples the full one keeps."""
    if "crs" not in A:
        from scipy.spatial import cKDTree

        from dexengine.robots import wrist_hull as G

        c = hull_samples(A["v"], G.hull_triangles(A["v"]), CRS_M)
        d, par = cKDTree(c).query(A["smp"])
        cnt = np.bincount(par, minlength=len(c))
        A["crs"] = (c, np.argsort(par, kind="stable"), cnt, np.cumsum(cnt) - cnt, 0.015 + np.sqrt(3.0) * float(d.max()))
    return A["crs"]


def _exact_hull(Hs: list, b: dict, T2, g: dict) -> np.ndarray:
    """min over the masked torso hulls H of hull_hull_sd(moving hull A, H): min of (deepest A sample inside H, deepest
    H sample inside A), clipped at 10 mm. A samples are prescreened once by the mask's torso grid (< 10 mm) and per H by
    its bounding sphere; H samples by a KD ball around A's centre and A's box — every sample inside the other hull is
    kept, so a negative value is exact and a positive one is a bound."""
    A, Tk = b["hull"], T2[:, b["k"]]
    R, t = Tk[:, :3, :3], Tk[:, :3, 3]
    P = len(T2)
    c, order, cnt, start, cut = _coarse(A)
    tc, ci = np.nonzero(HG.interp(g, np.einsum("tij,pj->tpi", R, c) + t[:, None, :]) < cut)
    n = cnt[ci]
    tt = np.repeat(tc, n)
    si = order[np.repeat(start[ci], n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]
    pw = np.einsum("nij,nj->ni", R[tt], A["smp"][si]) + t[tt]
    keep = HG.interp(g, pw) < 0.010
    ti, near = tt[keep], pw[keep]
    out = np.full(P, 0.010)
    lo, hi = A["v"].min(0), A["v"].max(0)
    for H in Hs:
        m = np.linalg.norm(near - H["ctr"], axis=1) < H["rad"] + 0.010
        _seg_min(out, ti[m], (near[m] @ H["eq"][:, :3].T + H["eq"][:, 3]).max(-1))
        lists = H["kd"].query_ball_point(np.einsum("tij,j->ti", R, A["ctr"]) + t, A["rad"] + 0.002)
        lens = np.fromiter((len(x) for x in lists), int, len(lists))
        if lens.sum():
            idx = np.concatenate([np.asarray(x, int) for x in lists if len(x)])
            pid = np.repeat(np.arange(P), lens)
            pb = np.einsum("nji,nj->ni", R[pid], H["smp"][idx] - t[pid])  # world → A link frame (Rᵀ)
            m = np.all((pb >= lo) & (pb <= hi), 1)
            _seg_min(out, pid[m], (pb[m] @ A["eq"][:, :3].T + A["eq"][:, 3]).max(-1))
    return out


def _seg_min(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """out[i] = min(out[i], vals of every row with idx == i) for a sorted idx (np.minimum.at, without its cost)."""
    if len(idx):
        st = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
        out[idx[st]] = np.minimum(out[idx[st]], np.minimum.reduceat(vals, st))


def _each(tz: dict, Ts, bodies, skip) -> tuple:
    """(exact, filter) per pose; skip[i] (P,) bool | None = body i's pairs guard_band proved > 0 (left 1e9)."""
    if tz["proxy"] != AP.GUARD_KIND:
        raise RuntimeError(f"[COLL_PROXY] the torso guard measures {AP.GUARD_KIND} bodies — got {tz['proxy']}")
    out, flt = np.full(Ts.shape[0], 1e9), np.full(Ts.shape[0], 1e9)
    for b, sk in zip(bodies, skip, strict=True):
        kp = slice(None) if sk is None else np.nonzero(~sk)[0]
        T = Ts[kp]
        if not len(T):
            continue
        fb = _body_filter_sd(tz, b, T)
        flt[kp] = np.minimum(flt[kp], fb)
        ix = np.nonzero(fb < NEAR_M)[0]
        if len(ix):
            Hs = [tz["torso"][tn] for tn in b["mask"]]
            fb[ix] = (
                np.min([_exact_cap(H, b, T[ix]) for H in Hs], 0)
                if b["kind"] == "cap"
                else _exact_hull(Hs, b, T[ix], tz["grids"][b["mask"]])
            )
        out[kp] = np.minimum(out[kp], fb)
    return out, flt


def torso_guard(ctx: dict, q_all, side: str, fingers, subdiv: int = 8, count=None) -> tuple[bool, str, float]:
    """Arm · palm · fingers ↔ torso with the exact ruler over the frames and the retiming chords (as
    collision.desk_penetration). fingers = that side's finger values (T, n) on q_all's time axis
    (hand_surface.guard_finger_q) — interpolated along the same chords; missing or another length dies.
    (ok, reason, min signed surface distance m of the pairs guard_band.torso_terms measured). Marker [TORSO_GUARD]
    where the filter (arm + palm) disagrees."""
    tz, q = ctx["torso"], np.asarray(q_all, float)
    if fingers is None:
        raise InputMissingError(
            f"[TORSO_HANDS] {side}: no finger values — the torso guard measures arm · palm · "
            "fingers together (hand_surface.guard_finger_q)"
        )
    fq = np.asarray(fingers, float)
    if fq.shape != (len(q), tz["hand"]["n"]):
        raise SystemExit(
            f"[TORSO_HANDS] {side}: finger values {fq.shape} ≠ (arm frames {len(q)}, fingers "
            f"{tz['hand']['n']}) — not the same time axis"
        )
    nj, qa = q.shape[1], np.concatenate([q, fq], 1)
    if subdiv > 1 and len(qa) > 1:
        w = np.linspace(0.0, 1.0, subdiv, endpoint=False)[1:]
        qa = np.concatenate(
            [
                qa,
                (qa[:-1, None, :] * (1.0 - w)[None, :, None] + qa[1:, None, :] * w[None, :, None]).reshape(
                    -1, qa.shape[1]
                ),
            ],
            0,
        )
    Ts = arm_link_transforms(ctx, qa[:, :nj])
    Tf = Ts[:, nj - 1][:, None] @ HS.finger_frames(tz["hand"], qa[:, nj:])[:, tz["fsel"]]
    ex_a, ex_f, filt = GB.torso_terms(tz, Ts, np.concatenate([Ts, Tf], 1), count)
    ex = min(ex_a, ex_f)
    ok_e, ok_f = ex >= -tol_m(), filt >= -tol_m()
    if ok_e != ok_f:
        print(
            f"[TORSO_GUARD] {side} filter≠guard filter={filt * 1000:.1f}mm exact={ex * 1000:.1f}mm "
            f"({'guard only fails' if ok_f else 'filter only fails'} · body "
            f"{'fingers' if ex_f < ex_a else 'arm·palm'})",
            flush=True,
        )
    return (True, "", ex) if ok_e else (False, f"{side}torso_surface{-ex * 1000:.1f}mm", ex)
