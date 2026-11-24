"""Floating hand, joint wrist drive: the anchor-pose rule ``DEXENGINE_FH_ANCHOR=minmid`` (off by default).

Off = the anchor is the env's start reference wrist. 'minmid' = per clip and side, the anchor pose is chosen from the
whole reference wrist path of that clip (the same computation for any clip — no table):
  anchor y = the unit vector minimising max_t |y·ẑ_hand(t)| (it lowers the maximum middle angle |X| =
  asin(|ŷ·ẑ_hand|))
  · anchor x = the start hand x projected orthogonal to y · position = the start reference wrist.
The reset writes the virtual rotation start angles = YXZ(anchor⁻¹ ⊗ start hand).
Load-time gates (die): middle angle max after the rule > VJ_ANCHOR_MID_GUARD · prismatic extent per anchor axis >
VJ_ANCHOR_LIN_LIMIT_M − VJ_ANCHOR_EXTENT_MARGIN_M · unwrapped outer Y/Z max > VJ_ROT_OUTER_LIMIT.
The prismatic limit is ±2 m only when on (written at runtime after spawn — URDF and USD unchanged).
Measured middle-angle maximum before → after: taco 88.1 → 35.6° · oakink2 89.5 → 27.8 · grab 89.1 → 48.7 ·
arctic 88.5 → 41.2.

Markers: one `[FH_ANCHOR] <side> clip=<clip> …` line per clip and side, one
`[FH_ANCHOR] table rule=minmid clips=<K> rows <n>/<2K>` table line and `[FH_ANCHOR] reset <side> clip=<clip> frame=<f>`
reset lines (with `[FH_ANCHOR] <side> on rule=` from the action term). The run checks of the floating-hand jobs parse
them.

Env flag (read inside the Isaac process): DEXENGINE_FH_ANCHOR '' (default) | 'minmid' (needs DEXENGINE_FH_WRIST=joint).
"""

from __future__ import annotations

import os

from experiments.ablations.floating_hand.identity import MARK
from experiments.ablations.floating_hand.wrench import _quat_conj, _quat_mul
from experiments.ablations.floating_hand.wrist_joints import (
    VJ_LIN_LIMIT_M,
    VJ_ROT_OUTER_LIMIT,
    WRIST_DRIVE_ENV,
    euler_yxz,
    joint_drive,
)

ANCHOR_ENV = "DEXENGINE_FH_ANCHOR"
ANCHOR_RULES = ("minmid",)
VJ_ANCHOR_LIN_LIMIT_M = 2.0  # prismatic limit ± (m) when on — grab per-axis max 1.20 m
VJ_ANCHOR_EXTENT_MARGIN_M = 0.25  # prismatic extent gate = limit − this margin
VJ_ANCHOR_MID_GUARD = 1.0471975511965976  # 60° — middle-angle gate after the rule (rad)
ANCHOR_GRID_N = 4000  # candidate y directions (Fibonacci sphere grid, one hemisphere)


def anchor_rule() -> str:
    """The anchor rule — '' = off (default · start wrist) · 'minmid'. An unknown value, or 'minmid' without the joint
    drive, dies."""
    raw = os.environ.get(ANCHOR_ENV, "")
    if raw == "":
        return ""
    if raw not in ANCHOR_RULES:
        raise RuntimeError(f"{MARK} {ANCHOR_ENV}={raw!r} — only {'|'.join(ANCHOR_RULES)} (default unset = start wrist)")
    if not joint_drive():
        raise RuntimeError(
            f"{MARK} {ANCHOR_ENV}={raw} belongs to {WRIST_DRIVE_ENV}=joint — the wrench drive has no "
            "anchor to consume it"
        )
    return raw


def vj_lin_limit() -> float:
    """This process's virtual prismatic limit (m) — off = VJ_LIN_LIMIT_M (the URDF's) · on = VJ_ANCHOR_LIN_LIMIT_M."""
    return VJ_ANCHOR_LIN_LIMIT_M if anchor_rule() else VJ_LIN_LIMIT_M


def _np_qmul(a, b):
    import numpy as np  # noqa: PLC0415

    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def _np_rot(q, v):
    import numpy as np  # noqa: PLC0415

    u, w = q[..., 1:], q[..., :1]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _np_mat2q(R):
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
        return q / np.linalg.norm(q)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[1 + i] = s / 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def center_branch(u) -> int:
    """The multiple k ∈ {−1, 0, 1} of 2π to add to an unwrapped angle path u (rad) — the one with the smallest
    max |u + 2πk| (toward the middle of the limits)."""
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    u = np.asarray(u, dtype=np.float64)
    return min((-1, 0, 1), key=lambda m: float(np.abs(u + 2 * math.pi * m).max()))


def anchor_minmid(quats, pos) -> dict:
    """Rule minmid for one clip and side (reference wrist quats (T,4) wxyz · pos (T,3) in one world frame): the anchor
    pose and the gate values. Returns q (anchor wxyz) · mid_before (middle-angle max with the start-wrist anchor, rad)
    · mid_after · extent (prismatic extent per anchor axis (3,) m) · k (2π multiples of the outer Y · Z branches)
    · u ((T,2) unwrapped, centred outer Y · Z — the reset takes its start frame's value as the branch reference, RSI
    included) · argmax_mid (frame of the middle-angle max) · outer (|max| of the unwrapped outer Y/Z, start angle
    included). Deterministic (grid + fixed-seed refinement). Pure numpy."""
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    p = np.asarray(pos, dtype=np.float64)
    ez = np.tile([0.0, 0.0, 1.0], (len(q), 1))
    zs = _np_rot(q, ez)
    n = ANCHOR_GRID_N
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), math.pi * (1 + 5**0.5) * i
    G = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)
    G = G[G[:, 2] >= 0]  # ±y give the same anchor angle → one hemisphere
    y = G[int(np.argmin(np.abs(zs @ G.T).max(0)))]
    rng = np.random.default_rng(0)
    for step in (0.05, 0.02, 0.008, 0.003):  # local refinement (fixed seed)
        for _ in range(30):
            c = y + step * rng.normal(size=(64, 3))
            c /= np.linalg.norm(c, axis=-1, keepdims=True)
            sc = np.abs(zs @ c.T).max(0)
            k = int(np.argmin(sc))
            if sc[k] < np.abs(zs @ y).max():
                y = c[k]
    x0 = _np_rot(q[0], np.array([1.0, 0.0, 0.0]))
    x = x0 - (x0 @ y) * y
    if np.linalg.norm(x) < 1e-3:
        x = np.cross(y, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, y, np.cross(x, y)], -1)
    qa = _np_mat2q(R)
    y0 = _np_rot(q[0], np.array([0.0, 1.0, 0.0]))
    mid_before = float(np.arcsin(np.clip(np.abs(zs @ y0), 0, 1)).max())
    mid_after = float(np.arcsin(np.clip(np.abs(zs @ y), 0, 1)).max())
    pa = (p - p[0]) @ R  # components in anchor axes
    extent = pa.max(0) - pa.min(0)  # the extent covers every start frame (RSI included)
    rel = _np_qmul((qa * np.array([1, -1, -1, -1]))[None], q)
    w_, x_, y_, z_ = np.moveaxis(rel, -1, 0)
    ya = np.unwrap(np.arctan2(2 * (x_ * z_ + w_ * y_), 1 - 2 * (x_ * x_ + y_ * y_)))
    za = np.unwrap(np.arctan2(2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_)))
    # start branch of the outer Y · Z: the 2π multiple that centres the unwrapped path in the limits (for clips whose
    # start angle in (−π, π] turns more than once)
    ks = [center_branch(u) for u in (ya, za)]
    u = np.stack([ya + 2 * math.pi * ks[0], za + 2 * math.pi * ks[1]], -1)  # (T, 2) unwrapped, centred outer Y · Z
    outer = float(np.abs(u).max())
    bmid = np.arcsin(np.clip(zs @ y, -1, 1))
    return {
        "q": qa,
        "y": y,
        "mid_before": mid_before,
        "mid_after": mid_after,
        "extent": extent,
        "outer": outer,
        "k": (int(ks[0]), int(ks[1])),
        "u": u,
        "argmax_mid": int(np.abs(bmid).argmax()),
    }


def anchor_check(st: dict, clip: str, side: str) -> None:
    """The gates of rule minmid — a clip past one stops the load (nothing is clipped silently)."""
    import math  # noqa: PLC0415

    lim = VJ_ANCHOR_LIN_LIMIT_M - VJ_ANCHOR_EXTENT_MARGIN_M
    bad = []
    if st["mid_after"] > VJ_ANCHOR_MID_GUARD:
        bad.append(f"middle angle {math.degrees(st['mid_after']):.1f}° > {math.degrees(VJ_ANCHOR_MID_GUARD):.0f}°")
    if float(max(st["extent"])) > lim:
        bad.append(
            f"prismatic extent {float(max(st['extent'])):.2f} m > {lim:.2f} m (limit {VJ_ANCHOR_LIN_LIMIT_M} − "
            f"margin {VJ_ANCHOR_EXTENT_MARGIN_M})"
        )
    if st["outer"] > VJ_ROT_OUTER_LIMIT:
        bad.append(f"outer unwrapped angle {math.degrees(st['outer']):.0f}° > {math.degrees(VJ_ROT_OUTER_LIMIT):.0f}°")
    if bad:
        raise RuntimeError(f"[FH_ANCHOR] load refused {side} clip={clip}: " + " · ".join(bad))


def anchor_stamp(st: dict, clip: str, side: str) -> str:
    """One marker line per clip and side — axis · middle angle before → after · prismatic extent vs gate · outer
    angle.
    """
    import math  # noqa: PLC0415

    e = st["extent"]
    return (
        f"[FH_ANCHOR] {side} clip={clip} y=({st['y'][0]:+.3f},{st['y'][1]:+.3f},{st['y'][2]:+.3f}) middle "
        f"{math.degrees(st['mid_before']):.1f}° → {math.degrees(st['mid_after']):.1f}° · prismatic extent "
        f"({e[0]:.2f},{e[1]:.2f},{e[2]:.2f}) m / gate {VJ_ANCHOR_LIN_LIMIT_M - VJ_ANCHOR_EXTENT_MARGIN_M:.2f} · outer "
        f"{math.degrees(st['outer']):.0f}° · start branch {st['k']}"
    )


def anchor_start_angles(anchor_quat, start_quat, u_start):
    """Reset start virtual rotation angles (n,3) [Y, X, Z] — the outer Y · Z of YXZ(anchor⁻¹ ⊗ start hand) moved to
    the 2π branch nearest that frame's unwrapped reference u_start (n,2), so any start frame joins the reference path
    inside
    the limits (RSI included). Pure torch."""
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415

    e = euler_yxz(_quat_mul(_quat_conj(anchor_quat), start_quat))
    two_pi = 2.0 * math.pi
    ky = torch.round((u_start[:, 0] - e[:, 0]) / two_pi)
    kz = torch.round((u_start[:, 1] - e[:, 2]) / two_pi)
    return torch.stack([e[:, 0] + two_pi * ky, e[:, 1], e[:, 2] + two_pi * kz], dim=-1)


def anchor_table(tm, device) -> dict:
    """Anchor poses of every loaded clip (loader order) for both hands — {side: (clips, 4) wxyz ·
    'k': {side: (clips, 2)} · 'u': {side: (clips, max frames, 2) unwrapped outer Y · Z}}. Rule minmid · gates ·
    markers."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    ld = tm._loader
    names = list(ld.sequence_names)
    lens = ld.clip_lengths.cpu().numpy()
    out, n_ok = {"k": {}, "u": {}}, 0
    for side, P, Q in (
        ("right", tm._wrist_poses, tm._wrist_quats),
        ("left", tm._left_wrist_poses, tm._left_wrist_quats),
    ):
        if P is None or Q is None:
            raise RuntimeError(f"[FH_ANCHOR] {side}: no reference wrist table — the anchor rule cannot be computed")
        if Q.shape[0] != len(names):
            raise RuntimeError(f"[FH_ANCHOR] {side}: wrist table clips {Q.shape[0]} ≠ loader {len(names)}")
        Pn, Qn = P.detach().cpu().numpy(), Q.detach().cpu().numpy()
        qa = np.zeros((len(names), 4))
        kk = np.zeros((len(names), 2))
        uu = np.zeros((len(names), Q.shape[1], 2), dtype=np.float32)  # padded frames = the last value
        for i, nm in enumerate(names):
            T = int(lens[i])
            st = anchor_minmid(Qn[i, :T], Pn[i, :T])
            anchor_check(st, nm, side)
            print(anchor_stamp(st, nm, side), flush=True)
            qa[i] = st["q"]
            kk[i] = st["k"]
            uu[i, :T] = st["u"]
            uu[i, T:] = st["u"][-1]
            n_ok += 1
        out[side] = torch.tensor(qa, dtype=torch.float32, device=device)
        out["k"][side] = torch.tensor(kk, dtype=torch.float32, device=device)
        out["u"][side] = torch.tensor(uu, dtype=torch.float32, device=device)
    print(
        f"[FH_ANCHOR] table rule=minmid clips={len(names)} rows {n_ok}/{2 * len(names)} · "
        f"prismatic limit ±{VJ_ANCHOR_LIN_LIMIT_M} m · "
        f"middle gate 60°",
        flush=True,
    )
    return out


def anchor_reset_stamp(env, side, tm, cid, fr, e0) -> None:
    """Reset markers of the anchor rule — one line for each of the first 4 reset envs per side (clip · start frame ·
    start angles) and a running total every 8192 resets (share with start frame > 0)."""
    import math  # noqa: PLC0415

    cnt = getattr(env, "_fh_anchor_resets", None)
    if cnt is None:
        cnt = env._fh_anchor_resets = {"right": [0, 0], "left": [0, 0]}
    c = cnt[side]
    names = tm._loader.sequence_names
    for j in range(min(len(cid), max(0, 4 - c[0]))):
        a = [math.degrees(float(v)) for v in e0[j]]
        print(
            f"[FH_ANCHOR] reset {side} clip={names[int(cid[j])]} frame={int(fr[j])} start angles Y {a[0]:.1f} "
            f"X {a[1]:.1f} Z {a[2]:.1f}°",
            flush=True,
        )
    before = c[0]
    c[0] += len(cid)
    c[1] += int((fr > 0).sum())
    if c[0] // 8192 != before // 8192:
        print(f"[FH_ANCHOR] reset total {side} {c[0]} · start frame > 0 {c[1]}", flush=True)
