#!/usr/bin/env python3
"""GPU batch path of the arm search ruler (judge/boot._boot_gpu_batch) — the torch twins of collision.core.

The numpy functions of collision.core (arm_link_transforms · desk_penetration · desk_penetration_each) are the
canonical definitions. The `*_t` twins here compute **the same formulas** on torch float64 tensors (device of the
caller) with a leading batch axis, so the judge scores frames × candidates in one call instead of one numpy call per
candidate. Thresholds, sample counts, chord subdivision, the "first body / first frame wins" tie rules and the torso
merge (core.torso_shift_m) are reproduced 1:1. What is *not* guaranteed is bitwise equality of the floats (GPU matmul
accumulation order differs from numpy BLAS) — the judge verifies at the output level. Callers import this module
directly; core is reached at call time.
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.collision import arm_surface as AS
from dexengine.placement.collision import torso_torch as TST

#: desk_penetration_t reads the desk exactly only on the (item, pose) rows whose desk bound reaches the item's best
#: exact row (_desk_bodies_max_t); this margin keeps a row whose bound sits an ulp under its exact value.
DESK_UB_EPS_M = 1e-9

_TCTX_CACHE: dict = {}
_SERVER_KEEP = ("_TCTX_CACHE",)  # (ctx object, device) key — clip-independent


def torch_ctx(ctx: dict, dev) -> dict:
    """numpy FK/capsule context → torch float64 tensors on `dev` (cached per ctx object)."""
    import torch

    from dexengine.placement.collision import core as SCL

    key = (id(ctx), str(dev))
    hit = _TCTX_CACHE.get(key)
    if hit is not None and hit["_ctx"] is ctx:
        return hit
    f64 = {"dtype": torch.float64, "device": dev}
    nj = len(ctx["joints"])
    A = [np.eye(4)] + [np.asarray(ctx["A"][k], float) for k in range(1, nj)]
    axes = np.asarray(ctx["axes"], float)
    Ks, KKs = [], []
    for k in range(nj):
        a = axes[k] / np.linalg.norm(axes[k])
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Ks.append(K)
        KKs.append(K @ K)
    caps = ctx["caps"]
    ez = np.array([0.0, 0.0, 1.0])
    tc = {
        "_ctx": ctx,
        "nj": nj,
        "B0": torch.tensor(np.asarray(ctx["B0"], float), **f64),
        "A": torch.tensor(np.stack(A), **f64),  # (nj,4,4); A[0] unused (eye)
        "K": torch.tensor(np.stack(Ks), **f64),
        "KK": torch.tensor(np.stack(KKs), **f64),
        "I3": torch.eye(3, **f64),
        "cap_k": torch.tensor([int(c["k"]) for c in caps], dtype=torch.long, device=dev),
        "cap_p": torch.tensor(np.stack([np.asarray(c["p"], float) for c in caps]), **f64)
        if caps
        else torch.zeros((0, 3), **f64),
        "cap_ax": torch.tensor(np.stack([np.asarray(c["R"], float) @ ez for c in caps]), **f64)
        if caps
        else torch.zeros((0, 3), **f64),
        "cap_h": torch.tensor([float(c["L"]) / 2.0 for c in caps], **f64),
        "base": torch.tensor(np.asarray(ctx["base"], float), **f64),
        "ztop": float(ctx["ztop"]),
        "ts": torch.tensor(SCL._axis_ts(), **f64),
    }
    tc["surf"] = AS.torch_surface(ctx, dev)
    tc["torso"] = TST.torch_torso(ctx["torso"], dev)
    _TCTX_CACHE[key] = tc
    return tc


def arm_link_transforms_t(tc: dict, q):
    """q (..., nj) float64 tensor → link transforms (..., nj, 4, 4). Same recursion as
    arm_link_transforms: T_k = T_{k-1} · A_k · R(axis_k, q_k)."""
    import torch

    lead = q.shape[:-1]
    cur = tc["B0"].expand(*lead, 4, 4)
    out = []
    for k in range(tc["nj"]):
        if k > 0:
            cur = cur @ tc["A"][k]
        th = q[..., k]
        s = torch.sin(th)[..., None, None]
        c = torch.cos(th)[..., None, None]
        R = tc["I3"] + s * tc["K"][k] + (1 - c) * tc["KK"][k]
        M = torch.zeros(*lead, 4, 4, dtype=q.dtype, device=q.device)
        M[..., :3, :3] = R
        M[..., 3, 3] = 1.0
        cur = cur @ M
        out.append(cur)
    return torch.stack(out, -3)


def _cap_endpoints_t(tc: dict, Ts):
    """Ts (..., nj, 4, 4) → capsule endpoints a, b (..., nc, 3) and (ctr, axis) for sampling."""
    import torch

    Tk = Ts[..., tc["cap_k"], :, :]  # (..., nc, 4, 4)
    R, t = Tk[..., :3, :3], Tk[..., :3, 3]
    ctr = t + torch.einsum("...ij,...j->...i", R, tc["cap_p"])
    ax = torch.einsum("...ij,...j->...i", R, tc["cap_ax"])
    h = tc["cap_h"][..., None]
    return ctr - ax * h, ctr + ax * h, ctr, ax


def _base_t(tc: dict, z_off, q, n_lead: int):
    """base + [0,0,z_off] as a (N,1,..,3) tensor; z_off scalar or per-item (N,) tensor
    (the judge passes one dzm per candidate / per gate row)."""
    import torch

    if torch.is_tensor(z_off):
        z = z_off.to(dtype=q.dtype, device=q.device).reshape(-1)
        b = tc["base"].expand(z.shape[0], 3).clone()
        b[:, 2] = b[:, 2] + z
        return b.reshape(z.shape[0], *([1] * (n_lead - 1)), 3)
    return tc["base"] + torch.tensor([0.0, 0.0, float(z_off)], dtype=q.dtype, device=q.device)


def _first_argmax(x, dim):
    """index of the first maximum along dim (ties → smallest index), like numpy argmax."""
    import torch

    m = x.amax(dim, keepdim=True)
    n = x.shape[dim]
    idx = torch.arange(n, device=x.device).view([n if i == (dim % x.dim()) else 1 for i in range(x.dim())])
    return torch.where(x == m, idx, torch.full_like(idx, n)).amin(dim), m.squeeze(dim)


def _desk_ub_t(tc: dict, Ts, ctrb, axw, base, desk_xy):
    """Upper bound (…,) of the per-pose desk component max over bodies (AS.desk_pen_bodies_t) — prunes, never a value:
    a capsule ≤ ztop + r − its lower axis end (the samples lie on the axis), a hull ≤ ztop − its lowest vertex (the
    footprint-clipped hull lies inside the hull); −1e9 where the body's xy box (a capsule's grown by r) misses the
    footprint — no sample inside and no clip there, so the exact value is −1e9 too."""
    import torch

    from dexengine.placement.coords import core as VC

    (x0, x1), (y0, y1) = desk_xy if desk_xy is not None else VC.desk_bounds_scene(0.0, margin=0.0)
    st, ztop = tc["surf"], tc["ztop"]
    a, b = ctrb - axw * tc["cap_h"][:, None], ctrb + axw * tc["cap_h"][:, None]  # (…, nc, 3)
    lo, hi, r = torch.minimum(a, b), torch.maximum(a, b), st["r"]
    hit = (hi[..., 0] >= x0 - r) & (lo[..., 0] <= x1 + r) & (hi[..., 1] >= y0 - r) & (lo[..., 1] <= y1 + r)
    ub = torch.where(hit, ztop + r - lo[..., 2], torch.full_like(lo[..., 2], -1e9)).amax(-1)
    bt = base[..., 0, :] if base.dim() > 1 else base
    for k, V, _E, _EQ in st["hulls"]:
        Tk = Ts[..., k, :, :]
        W = torch.einsum("...ij,vj->...vi", Tk[..., :3, :3], V) + (Tk[..., :3, 3] + bt)[..., None, :]
        lo, hi = W.amin(-2), W.amax(-2)
        hit = (hi[..., 0] >= x0) & (lo[..., 0] <= x1) & (hi[..., 1] >= y0) & (lo[..., 1] <= y1)
        ub = torch.maximum(ub, torch.where(hit, ztop - lo[..., 2], torch.full_like(lo[..., 2], -1e9)))
    return ub


def _desk_bodies_max_t(tc: dict, Ts, ctrb, axw, base, desk_xy):
    """AS.desk_pen_bodies_t (N, T', nb) exact on every (item, pose) row whose desk bound (_desk_ub_t) reaches the item's
    best exact row and −inf elsewhere: the item max, its first body and its first frame are those of the full array
    (a pruned row ≤ its bound < that best ≤ the max — rows equal to the max are all kept). The best exact row first =
    the pose with the largest bound. Rows are independent (the gathered call is bit-equal to the full one —
    measured)."""
    import torch

    from dexengine.placement.coords import core as VC

    ts = tc["ts"]
    if VC.scene_mode() == "aerial":
        return AS.desk_pen_bodies_t(tc, Ts, ctrb, axw, base, ts, desk_xy)

    def rows(ii, jj):
        return AS.desk_pen_bodies_t(
            tc, Ts[ii, jj], ctrb[ii, jj], axw[ii, jj], base[ii, 0] if base.dim() > 1 else base, ts, desk_xy
        )

    N, Tp = Ts.shape[:2]
    ub = _desk_ub_t(tc, Ts, ctrb, axw, base, desk_xy)  # (N, T')
    ar = torch.arange(N, device=Ts.device)
    first = rows(ar, ub.argmax(1))  # (N, nb)
    ii, jj = torch.nonzero(ub + DESK_UB_EPS_M >= first.amax(-1)[:, None], as_tuple=True)
    out = torch.full((N, Tp, first.shape[-1]), -torch.inf, dtype=Ts.dtype, device=Ts.device)
    out[ii, jj] = rows(ii, jj)
    return out


def desk_penetration_t(
    tc: dict, q_all, z_off: float = 0.0, stride: int = 1, subdiv: int = 8, want_components: bool = False, desk_xy=None
):
    """Batched twin of desk_penetration. q_all: (N, T, nj) float64 tensor (or (T, nj) → N=1).
    Returns a list of N tuples with exactly desk_penetration's contract:
    (worst, frame, link) or (worst, frame, link, desk_m, torso_m) with want_components."""
    import torch

    from dexengine.placement.collision import core as SCL

    q = q_all if q_all.dim() == 3 else q_all[None]
    q = q[:, ::stride]
    N, Tq0, nj = q.shape
    if subdiv > 1 and Tq0 > 1:
        w = torch.tensor(np.linspace(0.0, 1.0, subdiv, endpoint=False)[1:], dtype=q.dtype, device=q.device)
        mids = (
            q[:, :-1, None, :] * (1.0 - w)[None, None, :, None] + q[:, 1:, None, :] * w[None, None, :, None]
        ).reshape(N, -1, nj)
        q = torch.cat([q, mids], 1)
    Ts = arm_link_transforms_t(tc, q)  # (N,T',nj,4,4)
    base = _base_t(tc, z_off, q, 3)  # (3,) or (N,1,1,3)
    _, _, ctr, axw = _cap_endpoints_t(tc, Ts)  # (N,T',nc,3)
    pfb = _desk_bodies_max_t(tc, Ts, ctr + base, axw, base, desk_xy)  # (N,T',nb) — numpy's first max row
    body_i, desk_only = _first_argmax(pfb.amax(1), 1)
    fi_desk, _ = _first_argmax(pfb[torch.arange(N, device=q.device), :, body_i], 1)
    fi_t, torso_only = _first_argmax(-TST.torso_filter_sd_t(tc["torso"], Ts), 1)
    links, shift = tc["surf"]["links"], SCL.torso_shift_m()
    d_np, bi, fd = desk_only.cpu().numpy(), body_i.cpu().numpy(), fi_desk.cpu().numpy()
    t_np, ft = torso_only.cpu().numpy(), fi_t.cpu().numpy()
    out = []
    for i in range(N):
        worst = (float(d_np[i]), SCL._frame_of(int(fd[i]), Tq0, subdiv, stride), links[int(bi[i])])
        if float(t_np[i]) + shift > worst[0]:
            worst = (float(t_np[i]) + shift, SCL._frame_of(int(ft[i]), Tq0, subdiv, stride), SCL.TORSO_LINK)
        out.append((*worst, float(d_np[i]), float(t_np[i])) if want_components else worst)
    return out


def desk_penetration_each_t(tc: dict, q_all, z_off: float = 0.0, desk_xy=None):
    """Batched twin of desk_penetration_each: q_all (..., nj) float64 tensor → (...) tensor."""
    import torch

    from dexengine.placement.collision import core as SCL

    Ts = arm_link_transforms_t(tc, q_all)
    base = _base_t(tc, z_off, q_all, q_all.dim())  # (3,) or (R,1,3)
    ts = tc["ts"]
    _, _, ctr, axw = _cap_endpoints_t(tc, Ts)
    out = AS.desk_pen_bodies_t(tc, Ts, ctr + base, axw, base, ts, desk_xy).amax(-1)
    return torch.maximum(out, -TST.torso_filter_sd_t(tc["torso"], Ts) + SCL.torso_shift_m())
