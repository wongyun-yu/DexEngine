"""S-float-settle in the training pin: a free main object the data statement marks `settle` is lowered onto the desk
or its own disc while its reference frame is before that object's first touch (the rest window).

Eligibility and the first-touch frame per clip and side come from the data statement (data/objects/float_settle,
collected once at scene build by `rl/env_setup/float_settle.build_float_settle` into `cfg.float_settle`); the amount
is the pin's own gap to the supports the lift reads (`object_pin.support_clearance`: the desk and this side's discs, at
the lifted pose), so lift and lower are one reading, symmetric. A non-finite gap (no support under the footprint) and a
negative one are not lowered. An eligible env whose gap exceeds scene_requirement.FLOAT_SETTLE_MAX_M raises: the data
statement drops such a clip, so reaching the pin means the two readings disagree. Without a statement (rule off, or a
scene built without it) nothing is lowered.

The gate is per call (frame < first touch): a reset inside the last SPAWN_HOLD frames before the first touch is held
lowered until the first-touch frame and at the reference pose after it, while the hand is already on the object.
"""

from __future__ import annotations

import torch

CFG_KEY = "float_settle"  # env cfg attribute: {side: [first-touch frame per loader clip, -1 = not eligible]}


def lower_amount(gap: torch.Tensor, frames: torch.Tensor, first_touch: torch.Tensor, cap: float | None = None):
    """(n,) m the pinned pose goes down: the gap where the env is eligible (frame < first touch ≥ 0) and the gap is
    finite and positive, else 0. An eligible gap above `cap` (default scene_requirement.FLOAT_SETTLE_MAX_M) raises."""
    if cap is None:
        from dexengine.data.objects.scene_requirement import FLOAT_SETTLE_MAX_M as cap  # noqa: PLC0415, N811
    ok = (first_touch >= 0) & (frames < first_touch) & torch.isfinite(gap) & (gap > 0)
    low = torch.where(ok, gap, torch.zeros_like(gap))
    if bool((low > cap).any()):
        j = int(torch.argmax(low))
        raise RuntimeError(
            f"[PIN_FEASIBLE] float settle: gap {float(low[j]) * 1000:.1f} mm > cap {cap * 1000:.0f} mm on an eligible "
            f"env (row {j}) — the data statement should have dropped this clip (float_settle.dropped)"
        )
    return low


def clearance(env, ids: torch.Tensor, side: str, obj, pos: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Support gap (n,) m of the `side` object at pose (pos, q): `object_pin.support_clearance` over the desk and this
    side's discs (the lift's supports). The one reading of the lowering and of judge gate D (object_pin.support_gap)."""
    from dexengine.rl.tasks.scene import object_pin as OP  # noqa: PLC0415 — the pin owns the geometry and supports

    P, valid, A = OP.object_geometry(env, obj)
    a = ids.long() % A
    desk, discs = OP.desk_box(env), OP.support_disc_rows(env, ids, side)
    if discs is not None:
        discs["on"] = discs["on"] & discs["own"]
    gap = torch.full((len(ids),), float("inf"), device=pos.device, dtype=pos.dtype)
    for s0 in range(0, len(ids), OP.PIN_CHUNK):
        sl = slice(s0, s0 + OP.PIN_CHUNK)
        pts = OP.quat_apply_wxyz(q[sl, None, :], P[a[sl]]) + pos[sl, None, :]
        dk = None if discs is None else {k: v[sl] for k, v in discs.items()}  # h is (n,K) too: column heights
        gap[sl] = OP.support_clearance(pts, valid[a[sl]], desk, dk)
    return gap


def first_touch(env, ids: torch.Tensor, side: str) -> torch.Tensor | None:
    """(n,) first-touch frame of `side`'s object per env (-1 = not lowered), None without a statement."""
    said = getattr(getattr(env, "cfg", None), CFG_KEY, None)
    if not said or side not in said:
        return None
    cache = env.__dict__.setdefault("_float_settle_t", {})
    if side not in cache:
        cache[side] = torch.tensor(said[side], dtype=torch.long, device=ids.device)
    t = cache[side]
    return t[env.trajectory_manager.clip_ids[ids].clamp(max=int(t.numel()) - 1)]


def lower(env, ids: torch.Tensor, frames: torch.Tensor, side: str, obj, pos: torch.Tensor, q: torch.Tensor):
    """(n,) m the pin lowers `side`'s object at its lifted pose (pos, q) — zeros without a statement (module doc)."""
    ft = first_touch(env, ids, side)
    if ft is None:
        return torch.zeros(len(ids), device=pos.device, dtype=pos.dtype)
    return lower_amount(clearance(env, ids, side, obj, pos, q), frames, ft)


def lowered_text(low: torch.Tensor) -> str:
    """The `[PIN_FEASIBLE] first pin` field: `lowered >0 n max x mm`."""
    mm = low.float() * 1000.0
    return f"lowered >0 {int((mm > 0).sum())} max {float(mm.max()) if mm.numel() else 0.0:.2f}"
