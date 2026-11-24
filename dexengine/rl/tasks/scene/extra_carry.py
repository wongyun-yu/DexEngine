"""Runtime of rule S-carried inside the extra slots: carried extras follow their carrier, not the world-recorded
trajectory.

Statement `cfg.extra_slot_carry` (writer `env_setup.extra_carry.carry_statement`; absent = no carried slot, this module
is never built and the slot path stays byte-identical): clip × slot None | {carrier_role, carrier_side, carrier_slot,
rel}. Every carried slot is kinematic and, at every step of the clip, stands at the **simulated** carrier pose × the
stated relative pose — the carrier's current sim pose (a main object's root, or the carrier slot's target of the same
step), never the world-recorded pose. A reset teleports to the reference start pose like every slot; the first step's
write puts it at the carrier. Quaternions are wxyz (this build's isaaclab.utils.math convention); the helpers
below are plain torch so CPU tests run without Isaac.
"""

from __future__ import annotations

import torch

MARK = "[EXTRA_CARRY]"


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(u, t, dim=-1)


def compose(pc: torch.Tensor, qc: torch.Tensor, rp: torch.Tensor, rq: torch.Tensor):
    """Carrier pose (pc, qc) × relative pose (rp, rq) → world pose of the carried body."""
    return pc + quat_apply(qc, rp), quat_mul(qc, rq)


def carrier_entity(scene_keys, side: str, shared: bool) -> str:
    """Scene entity of a main-object carrier: `object_<side>`; a shared object (side both, or the shared-object build)
    is the one prim `object_right`. Not in the scene raises."""
    name = "object_right" if shared or side == "both" else f"object_{side}"
    if name not in scene_keys:
        raise RuntimeError(
            f"{MARK} carrier main object {name!r} (side {side}) is not a scene entity {sorted(scene_keys)[:8]}"
        )
    return name


class CarryState:
    """Tensors of the statement for one env (built by ExtraSlots when `cfg.extra_slot_carry` exists)."""

    def __init__(self, env, K: int, clips: list):
        st = env.cfg.extra_slot_carry
        if len(st) != len(clips) or any(len(r) != K for r in st):
            raise RuntimeError(
                f"{MARK} statement shape {len(st)}×{[len(r) for r in st][:3]} ≠ clips {len(clips)} × K {K}"
            )
        dev, C = env.device, len(clips)
        self.on = torch.zeros(C, K, dtype=torch.bool, device=dev)
        self.main = torch.full((C, K), -1, dtype=torch.long, device=dev)
        self.slot = torch.full((C, K), -1, dtype=torch.long, device=dev)
        self.rp = torch.zeros(C, K, 3, device=dev)
        self.rq = torch.zeros(C, K, 4, device=dev)
        self.rq[..., 0] = 1.0
        keys, shared = list(env.scene.keys()), bool(getattr(env.cfg, "shared_object", False))
        self.mains: list = []
        for c, row in enumerate(st):
            for k, x in enumerate(row):
                if x is not None:
                    self._cell(c, k, x, keys, shared, clips, K)
        print(
            f"{MARK} attached for the whole clip: slots {int(self.on.sum())} · main carriers {self.mains} · slot "
            f"carriers {int((self.on & (self.slot >= 0)).sum())}",
            flush=True,
        )

    def _cell(self, c: int, k: int, x: dict, keys: list, shared: bool, clips: list, K: int) -> None:
        self.on[c, k] = True
        dev = self.rp.device
        self.rp[c, k] = torch.tensor(x["rel"][:3], device=dev)
        self.rq[c, k] = torch.tensor(x["rel"][3:7], device=dev)
        if x["carrier_role"] == "main":
            ent = carrier_entity(keys, str(x["carrier_side"]), shared)
            if ent not in self.mains:
                self.mains.append(ent)
            self.main[c, k] = self.mains.index(ent)
            return
        kc = int(x["carrier_slot"])
        if not 0 <= kc < K or kc == k:
            raise RuntimeError(f"{MARK} clip {clips[c]} slot {k}: carrier slot {kc} outside 0..{K - 1}")
        self.slot[c, k] = kc

    def carrier_pose(self, env, ids: torch.Tensor, ci: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor):
        """(n, K, 3) · (n, K, 4) world pose of every slot's carrier: a main object's simulated root, or the carrier
        slot's pose in (pos, quat) — the same step's targets."""
        n, K = pos.shape[0], pos.shape[1]
        pc, qc = pos.clone(), quat.clone()
        mi, si = self.main[ci], self.slot[ci]  # (n, K)
        for j, ent in enumerate(self.mains):
            d = env.scene[ent].data
            m = mi == j
            pc = torch.where(m[..., None], d.root_pos_w[ids][:, None, :].expand(n, K, 3), pc)
            qc = torch.where(m[..., None], d.root_quat_w[ids][:, None, :].expand(n, K, 4), qc)
        s = si.clamp(min=0)
        ps, qs = torch.gather(pos, 1, s[..., None].expand(n, K, 3)), torch.gather(quat, 1, s[..., None].expand(n, K, 4))
        sm = si >= 0
        return torch.where(sm[..., None], ps, pc), torch.where(sm[..., None], qs, qc)

    def apply(self, env, ids: torch.Tensor, ci: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor):
        """Targets of this step: carried slots at the simulated carrier × the relative pose, the others as given."""
        on = self.on[ci]
        pc, qc = self.carrier_pose(env, ids, ci, pos, quat)
        po, qo = compose(pc, qc, self.rp[ci], self.rq[ci])
        return torch.where(on[..., None], po, pos), torch.where(on[..., None], qo, quat)
