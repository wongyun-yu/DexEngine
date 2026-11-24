"""Torch twins of the torso filter (judge GPU batch path): the in-hull lower bound on one block (eager, or compiled once
per process on cuda), the torso record as torch tensors and the batched filter distance. The numpy filter in
collision.torso is the canonical definition; this module reads its block sizes and thresholds.
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.collision import arm_proxy as AP
from dexengine.placement.collision import sdf_grid as HG
from dexengine.placement.collision.torso import CHUNK, LB_EPS_M, SKIP_M

SPHERE_CHUNK = 65536  # poses per block of the torch filter's bounding-sphere grid read (poses × 3 — no sync inside)
_LB_FUSED: list = []
_SERVER_KEEP = ("_LB_FUSED",)  # the compiled bound


def in_hull_lb_t(tv, E, R, t):
    """torch twin of in_hull_lb on one block — elementwise, so compiled (lb_t on cuda) it is one fused reduction with
    no (C, V, K) array."""
    import torch

    nw = torch.einsum("tij,fj->tfi", R, E[:, :3])
    dw = E[:, 3][None] - (nw * t[:, None, :]).sum(-1)
    val = (
        tv[None, :, None, 0] * nw[:, None, :, 0]
        + tv[None, :, None, 1] * nw[:, None, :, 1]
        + tv[None, :, None, 2] * nw[:, None, :, 2]
        + dw[:, None, :]
    )
    return val.amax(-1).amin(-1)


def lb_t(tv, E, R, t):
    """The bound the torch filter reads: in_hull_lb_t compiled once per process on cuda (torch.compile · measured
    1.46 → 0.25 µs per pose · ~5–9 s compile), eager elsewhere — one formula, two schedules."""
    if not R.is_cuda:
        return in_hull_lb_t(tv, E, R, t)
    if not _LB_FUSED:
        import torch

        _LB_FUSED.append(torch.compile(in_hull_lb_t, dynamic=True))
        print("[TORSO_RULER] in-hull lower bound = torch.compile fused (cuda) — it only chooses rows", flush=True)
    return _LB_FUSED[0](tv, E, R, t)


def torch_torso(tz: dict, dev) -> dict:
    """The filter half of a torso record (build_torso) as torch float64 tensors on dev (bodies · grids · vertices)."""
    import torch

    f64 = {"dtype": torch.float64, "device": dev}
    gl = list(tz["grids"])
    return {
        "bodies": [
            {
                "k": b["k"],
                "pts": torch.tensor(np.asarray(b["pts"], float), **f64),
                "r": float(b["r"]),
                "gi": gl.index(b["mask"]),
                "ctr": torch.tensor(np.asarray(b["ctr"], float), **f64),
                "rad": float(b["rad"]),
                "eq": torch.tensor(b["hull"]["eq"], **f64) if b["kind"] == "hull" else None,
                "lb": torch.tensor(b["lb"], **f64) if b["kind"] == "hull" else None,
            }
            for b in tz["bodies"]
        ],
        "grids": [HG.to_torch(tz["grids"][k], dev) for k in gl],
        "px": tz["proxy"] == "capsule",
        "tverts": [torch.tensor(tz["tverts"][k], **f64) for k in gl],
    }


def torso_filter_sd_t(tt: dict, Ts):
    """torch twin of torso_filter_sd: Ts (..., nj, 4, 4) → (...). Per body: the sphere read over every pose, one
    nonzero for the near poses (grid value · in-hull bound in CHUNK blocks), one for the poses the bound does not
    clear (in-hull term) — a device sync per stage, not per block."""
    import torch

    lead, Tf = Ts.shape[:-3], Ts.reshape(-1, *Ts.shape[-3:])
    P = Tf.shape[0]
    out = torch.full((P,), 1e9, dtype=Ts.dtype, device=Ts.device)
    for b in tt["bodies"]:
        g, Tk = tt["grids"][b["gi"]], Tf[:, b["k"]]
        R, t = Tk[:, :3, :3], Tk[:, :3, 3]
        # centre product in CHUNK blocks (cuBLAS last bit changes with batch: equal ≤ 16384 poses, not 65536)
        C = AP.pose_block(len(b["pts"]), CHUNK) if tt["px"] else CHUNK
        cw = torch.cat([torch.einsum("tij,j->ti", R[i : i + C], b["ctr"]) for i in range(0, P, C)]) + t
        sd = torch.cat([HG.interp_t(g, cw[i : i + SPHERE_CHUNK]) for i in range(0, P, SPHERE_CHUNK)])
        sd = sd - b["rad"] - b["r"]
        ix = torch.nonzero(sd < SKIP_M).squeeze(-1)
        if ix.numel():
            Rn, tn = R.index_select(0, ix), t.index_select(0, ix)
            v = torch.cat(
                [
                    (
                        HG.interp_t(g, torch.einsum("tij,nj->tni", Rn[i : i + C], b["pts"]) + tn[i : i + C, None, :])
                        - b["r"]
                    ).amin(-1)
                    for i in range(0, len(ix), C)
                ]
            )
            if b["eq"] is not None:
                tv = tt["tverts"][b["gi"]]
                lb = torch.cat(
                    [lb_t(tv, b["lb"], Rn[i : i + CHUNK], tn[i : i + CHUNK]) for i in range(0, len(ix), CHUNK)]
                )
                j = torch.nonzero(lb <= v + LB_EPS_M).squeeze(-1)
                if j.numel():
                    Rj, tj = Rn.index_select(0, j), tn.index_select(0, j)
                    ih = []
                    for i in range(0, len(j), CHUNK):
                        pb = torch.einsum("tji,tmj->tmi", Rj[i : i + CHUNK], tv[None] - tj[i : i + CHUNK, None, :])
                        ih.append((pb @ b["eq"][:, :3].T + b["eq"][:, 3]).amax(-1).amin(-1))
                    v = v.index_copy(0, j, torch.minimum(v.index_select(0, j), torch.cat(ih)))
            sd = sd.index_copy(0, ix, v)
        out = torch.minimum(out, sd)
    return out.reshape(lead)
