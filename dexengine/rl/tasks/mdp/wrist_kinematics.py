"""Single source for the link7 <-> virtual-wrist (hand_base) MOUNT transform.

The link7->hand_base mount is SIDE-DEPENDENT — right arm Rz(-90deg), mirrored left
arm Rz(+90deg) (URDF-proven) — plus a per-hand
translation (RH56F1 0.0555 / SharpA 0.0655). Both are resolved from mdp.naming. Every
runtime/eval link7<->wrist computation funnels through here so the constant can neither
drift nor be applied side-blind (the latter was the cause of the left-wrist 180deg bug). Routing the right side through
here is byte-identical to the
old inline Rz(-90)/0.0555 literals; the left side is corrected to Rz(+90).

NOT handled here (a DIFFERENT transform): the floating-hand root / trajectory raw->world
Rz(-90)+desk transform (TrajectoryManager.root_quat_rotated, _load_wrist_poses).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from dexengine.rl.tasks.mdp.naming import resolve_ee_to_wrist_rot, resolve_link7_to_wrist_offset


def side_from_body_name(name: str) -> str:
    """'left' if the EE/arm body name is a left body, else 'right'."""
    return "left" if "left" in name else "right"


def mount_tensors(asset, side: str, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (r_fwd_wxyz, r_inv_wxyz, t_offset) for the link7->hand_base mount of ``side``.

    r_fwd: hand_base_quat = link7_quat (x) r_fwd. r_inv = conjugate(r_fwd) (wrist->link7).
    t_offset: link7-frame translation to the virtual wrist.
    """
    fwd = list(resolve_ee_to_wrist_rot(asset, side))  # side-aware pair rotation
    r_fwd = torch.tensor(fwd, device=device)
    r_inv = torch.tensor([fwd[0], -fwd[1], -fwd[2], -fwd[3]], device=device)
    t_off = torch.tensor(list(resolve_link7_to_wrist_offset(asset, side)), device=device)
    return r_fwd, r_inv, t_off


def link7_to_virtual_wrist(
    link7_pos_w: torch.Tensor, link7_quat_w: torch.Tensor, r_fwd_wxyz: torch.Tensor, t_offset: torch.Tensor
):
    """Batched virtual-wrist (hand_base) world pose FROM link7 world pose.

    vw_quat = link7_quat (x) r_fwd ;  vw_pos = link7_pos + R(link7_quat) @ t_offset.
    Identical float ops to the legacy inline code (bit-parity preserved for the right).
    """
    n = link7_pos_w.shape[0]
    vw_quat = math_utils.quat_mul(link7_quat_w, r_fwd_wxyz.unsqueeze(0).expand(n, -1))
    rot = math_utils.matrix_from_quat(link7_quat_w)
    vw_pos = link7_pos_w + torch.bmm(rot, t_offset.view(1, 3, 1).expand(n, -1, -1)).squeeze(-1)
    return vw_pos, vw_quat


def virtual_wrist_to_link7(
    wrist_pos_w: torch.Tensor, wrist_quat_w: torch.Tensor, r_inv_wxyz: torch.Tensor, t_offset: torch.Tensor
):
    """Batched link7 world pose FROM a virtual-wrist (hand_base) target.

    link7_quat = wrist_quat (x) r_inv ;  link7_pos = wrist_pos - R(link7_quat) @ t_offset.
    Inverse of link7_to_virtual_wrist; identical float ops to the legacy inline code.
    """
    n = wrist_pos_w.shape[0]
    link7_quat = math_utils.quat_mul(wrist_quat_w, r_inv_wxyz.unsqueeze(0).expand(n, -1))
    rot = math_utils.matrix_from_quat(link7_quat)
    link7_pos = wrist_pos_w - torch.bmm(rot, t_offset.view(1, 3, 1).expand(n, -1, -1)).squeeze(-1)
    return link7_pos, link7_quat
