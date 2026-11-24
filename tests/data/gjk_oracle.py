"""Test oracle: exact convex-vs-convex depth by MuJoCo GJK/EPA (mj_geomDistance) on hand hulls and convex pieces, with
the interface of hand_clear.TriModel (place · depth · depth_poses). The tests read the triangle ruler against it on
bodies where both are exact (closed convex boxes)."""

from __future__ import annotations

import numpy as np


class GjkModel:
    """MuJoCo mocap model of convex meshes: hand hulls `hands` (geom frame), static pieces `pieces` (object frame)."""

    def __init__(self, hands: list, pieces: list):
        import mujoco as mj  # noqa: PLC0415

        meshes = [np.asarray(v, float) for v in [*hands, *pieces]]
        asset = "".join(
            f'<mesh name="m{i}" vertex="{" ".join(f"{x:.7f}" for x in v.ravel())}" inertia="shell"/>'
            for i, v in enumerate(meshes)
        )
        bodies = "".join(
            f'<body name="b{i}" mocap="true"><geom type="mesh" mesh="m{i}" contype="0" conaffinity="0"/></body>'
            for i in range(len(meshes))
        )
        xml = f'<mujoco><compiler boundmass="1e-6" boundinertia="1e-12"/><asset>{asset}</asset>'
        self.mj, self.m = mj, mj.MjModel.from_xml_string(xml + f"<worldbody>{bodies}</worldbody></mujoco>")
        self.d = mj.MjData(self.m)
        self.nh, self.np = len(hands), len(pieces)

    def place(self, hand_R, hand_t, piece_R, piece_t) -> None:
        """Pose every hand hull (R (n,3,3), t (n,3)) and every piece (one object pose)."""
        q = np.zeros(4)
        for i in range(self.nh):
            self.mj.mju_mat2Quat(q, np.asarray(hand_R[i], float).reshape(9))
            self.d.mocap_quat[i], self.d.mocap_pos[i] = q, hand_t[i]
        self.mj.mju_mat2Quat(q, np.asarray(piece_R, float).reshape(9))
        for k in range(self.nh, self.nh + self.np):
            self.d.mocap_quat[k], self.d.mocap_pos[k] = q, piece_t
        self.mj.mj_kinematics(self.m, self.d)

    def depth_poses(self, hand_R, hand_t, piece_R, piece_t, floor=0.0, hands=None) -> tuple[np.ndarray, np.ndarray]:
        """`place` + `depth(hands)` for each of n placements (hand_R (n,h,3,3) · hand_t (n,h,3) · piece_R (n,3,3) ·
        piece_t (n,3)) -> (depths (n,), deepest hand index (n,)); `floor` is TriModel's (unused
        here)."""
        out = []
        for p in range(len(piece_t)):
            self.place(hand_R[p], hand_t[p], piece_R[p], piece_t[p])
            out.append(self.depth(hands))
        return np.array([d for d, _a in out]), np.array([a for _d, a in out], int)

    def depth(self, hands=None) -> tuple[float, int]:
        """(largest penetration (m) over (hand hull, piece) pairs, the deepest hand index or -1)."""
        m, d, rad = self.m, self.d, self.m.geom_rbound
        best, arg = 0.0, -1
        for i in range(self.nh) if hands is None else hands:
            for j in range(self.nh, self.nh + self.np):
                if np.linalg.norm(d.geom_xpos[i] - d.geom_xpos[j]) > rad[i] + rad[j]:
                    continue
                dist = self.mj.mj_geomDistance(m, d, i, j, 0.002, None)
                if -dist > best:
                    best, arg = -dist, i
        return best, arg
