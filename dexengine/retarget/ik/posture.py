"""Posture prior as generalised forces on the finger joints.

Configuration comes only from `prior.build_prior_conf`; this module resolves joint names on the IK model, computes
the per-frame contact weights, the limit-barrier bands, and the force vector written into `qfrc_applied` before every
physics step. Terms whose weight is zero are skipped (they would add exact zeros).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from dexengine.retarget.ik.prior import PriorSchemaError


@dataclass
class FlexJoint:
    name: str
    qadr: int
    dadr: int
    sign: float
    neutral: float
    rest: float


@dataclass
class AbdJoint:
    name: str
    qadr: int
    dadr: int
    neutral: float
    band: tuple[float, float] | None = None  # limit-barrier edges (a, b), None = no barrier


@dataclass
class Finger:
    name: str
    side: str
    role: str
    contact_col: int
    mcp: FlexJoint
    pip: FlexJoint
    dip: FlexJoint  # aliases `pip` for a two-joint finger
    has_dip: bool
    abd: list[AbdJoint] = field(default_factory=list)


def _joint(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise PriorSchemaError(f"[PRIOR] prior joint {name!r} is not in the scene")
    return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def _flex(model: mujoco.MjModel, entry: dict) -> FlexJoint:
    qa, da = _joint(model, entry["joint"])
    return FlexJoint(entry["joint"], qa, da, float(entry["sign"]), float(entry["neutral"]), float(entry["rest"]))


class PosturePrior:
    """Resolved prior for one IK model and one clip."""

    def __init__(self, conf: dict, model: mujoco.MjModel, contact_raw: dict[str, np.ndarray]):
        self.conf = conf
        self.lam_syn = conf["lambda_synergy"]
        self.lam_mcp = conf["lambda_mcp_share"]
        self.lam_rest = conf["lambda_rest"]
        self.lam_abd = conf["lambda_abduction"]
        self.lam_hx = conf["lambda_hyperextension"]
        self.lam_lim = conf["lambda_limit_avoid"]
        self.lam_vel = conf["lambda_vel"]
        self.one_sided = conf["synergy_one_sided"]
        self.rho_dp = conf["defaults"]["dip_pip_ratio"]
        self.rho_pm = conf["defaults"]["pip_mcp_ratio"]
        self.slew_limit = conf["slew_rate_limit"]
        self.fingers: list[Finger] = []
        for f in conf["fingers"]:
            ch = f["chain"]
            mcp, pip = _flex(model, ch["mcp"]), _flex(model, ch["pip"])
            dip = _flex(model, ch["dip"]) if "dip" in ch else pip
            abd = []
            for a in f["abd"]:
                qa, da = _joint(model, a["joint"])
                abd.append(AbdJoint(a["joint"], qa, da, float(a["neutral"])))
            self.fingers.append(
                Finger(f["name"], f["side"], f["role"], int(f["contact_col"]), mcp, pip, dip, "dip" in ch, abd)
            )
        dofs: dict[int, int] = {}
        for fg in self.fingers:
            for j in [fg.mcp, fg.pip, fg.dip] + fg.abd:
                dofs[j.dadr] = j.qadr
        order = sorted(dofs)
        self.dof_adr = np.array(order, dtype=int)
        self.qpos_adr = np.array([dofs[d] for d in order], dtype=int)
        self.barrier_joints = self._barrier_bands(model)
        T = next(iter(contact_raw.values())).shape[0]
        #: (T, F) raw fingertip contact of each finger's own column; the contact weight is low on those frames
        self.contact_mask = np.zeros((T, len(self.fingers)), dtype=bool)
        for k, fg in enumerate(self.fingers):
            self.contact_mask[:, k] = contact_raw[fg.side][:, fg.contact_col] > 0.5
        self.weights = np.where(self.contact_mask, conf["contact_weight_scale"], 1.0)
        self._bar_idx = np.array([int(np.searchsorted(self.dof_adr, j.dadr)) for j in self.barrier_joints], dtype=int)
        self._bar_a = np.array([j.band[0] for j in self.barrier_joints])
        self._bar_b = np.array([j.band[1] for j in self.barrier_joints])

    # --- configuration-time pieces -------------------------------------------------------------------------------
    def _barrier_bands(self, model: mujoco.MjModel) -> list[AbdJoint]:
        if self.lam_lim <= 0:
            return []
        m = math.radians(self.conf["limit_margin_deg"])
        roles = set(self.conf["limit_avoid_roles"])
        out = []
        for fg in self.fingers:
            if fg.role not in roles:
                continue
            for j in fg.abd:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
                lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                mid = (lo + hi) / 2.0
                a = lo if j.neutral - lo < m else min(lo + m, mid)
                b = hi if hi - j.neutral < m else max(hi - m, mid)
                if a <= lo and b >= hi:
                    continue
                j.band = (a, b)
                out.append(j)
        if not out:
            raise PriorSchemaError(
                f"[PRIOR] lambda_limit_avoid={self.lam_lim} but no abduction joint of the roles "
                f"{sorted(roles)} gets a barrier band"
            )
        return out

    def apply_damping(self, model: mujoco.MjModel) -> None:
        """Raise the damping of every prior-controlled DOF to at least the prior's damping."""
        d = self.dof_adr
        model.dof_damping[d] = np.maximum(model.dof_damping[d], self.conf["damping"])

    def band_report(self) -> str:
        if self.lam_lim <= 0:
            return "[PRIOR] limit barrier off (lambda_limit_avoid=0)"
        bands = ", ".join(
            f"{j.name}[{math.degrees(j.band[0]):.2f},{math.degrees(j.band[1]):.2f}]deg" for j in self.barrier_joints
        )
        return (
            f"[PRIOR] limit barrier lambda={self.lam_lim} margin={self.conf['limit_margin_deg']}deg "
            f"roles={list(self.conf['limit_avoid_roles'])} joints={len(self.barrier_joints)}: {bands}"
        )

    # --- per-step forces -----------------------------------------------------------------------------------------
    def _phi(self, q: np.ndarray, j: FlexJoint) -> float:
        return j.sign * (q[j.qadr] - j.neutral)

    def _shaped_terms(self, q: np.ndarray, t: int, f: dict[int, float]) -> None:
        """Synergy, mcp share, hyperextension, rest and abduction springs (all weighted by the contact weight)."""
        for k, fg in enumerate(self.fingers):
            w = self.weights[t, k]
            p_mcp, p_pip, p_dip = self._phi(q, fg.mcp), self._phi(q, fg.pip), self._phi(q, fg.dip)
            rho_dp = self.rho_dp if fg.has_dip else 1.0
            if self.lam_syn > 0:
                r1 = p_dip - rho_dp * p_pip
                if self.one_sided:
                    r1 = max(r1, 0.0)
                f[fg.dip.dadr] += -self.lam_syn * w * r1 * fg.dip.sign
                f[fg.pip.dadr] += self.lam_syn * w * rho_dp * r1 * fg.pip.sign
            if self.lam_mcp > 0:
                r2 = p_pip - self.rho_pm * p_mcp
                if self.one_sided:
                    r2 = max(r2, 0.0)
                f[fg.pip.dadr] += -self.lam_mcp * w * r2 * fg.pip.sign
                f[fg.mcp.dadr] += self.lam_mcp * w * self.rho_pm * r2 * fg.mcp.sign
            if self.lam_hx > 0:
                f[fg.pip.dadr] += -self.lam_hx * min(p_pip, 0.0) * fg.pip.sign
                if fg.has_dip:
                    f[fg.dip.dadr] += -self.lam_hx * min(p_dip, 0.0) * fg.dip.sign
            if self.lam_rest > 0:
                for j in [fg.mcp, fg.pip, fg.dip] if fg.has_dip else [fg.mcp, fg.pip]:
                    f[j.dadr] += -self.lam_rest * w * (self._phi(q, j) - j.rest) * j.sign
            if self.lam_abd > 0:
                for j in fg.abd:
                    f[j.dadr] += -self.lam_abd * w * (q[j.qadr] - j.neutral)

    def forces(self, q: np.ndarray, t: int, q_prev: np.ndarray | None) -> np.ndarray:
        """Force vector over `dof_adr` for the current configuration, frame `t` and the previous accepted frame."""
        out = np.zeros(self.dof_adr.size)
        if self.lam_syn > 0 or self.lam_mcp > 0 or self.lam_hx > 0 or self.lam_rest > 0 or self.lam_abd > 0:
            f = {int(d): 0.0 for d in self.dof_adr}
            self._shaped_terms(q, t, f)
            out = np.array([f[int(d)] for d in self.dof_adr])
        if self._bar_idx.size:
            x = q[self.qpos_adr[self._bar_idx]]
            out[self._bar_idx] = out[self._bar_idx] + (-self.lam_lim) * (
                np.minimum(x - self._bar_a, 0.0) + np.maximum(x - self._bar_b, 0.0)
            )
        if q_prev is not None and self.lam_vel > 0:
            out = out + (-self.lam_vel) * (q[self.qpos_adr] - q_prev[self.qpos_adr])
        return out

    def write(self, data: mujoco.MjData, t: int, q_prev: np.ndarray | None) -> None:
        """Zero the prior DOFs of `qfrc_applied` and write this step's prior forces there (other DOFs untouched)."""
        data.qfrc_applied[self.dof_adr] = self.forces(data.qpos, t, q_prev)

    def slew(self, q: np.ndarray, q_prev: np.ndarray) -> None:
        """In place: limit each prior joint's step from the previous accepted frame to the slew limit."""
        if self.slew_limit <= 0:
            return
        a = self.qpos_adr
        q[a] = q_prev[a] + np.clip(q[a] - q_prev[a], -self.slew_limit, self.slew_limit)

    # --- diagnostics -----------------------------------------------------------------------------------------------
    def synergy_residual(self, qpos: np.ndarray) -> np.ndarray:
        """(rows, F) |phi_dip - rho_dp * phi_pip| of each finger on the given qpos rows."""
        out = np.zeros((qpos.shape[0], len(self.fingers)))
        for k, fg in enumerate(self.fingers):
            rho = self.rho_dp if fg.has_dip else 1.0
            p_pip = fg.pip.sign * (qpos[:, fg.pip.qadr] - fg.pip.neutral)
            p_dip = fg.dip.sign * (qpos[:, fg.dip.qadr] - fg.dip.neutral)
            out[:, k] = np.abs(p_dip - rho * p_pip)
        return out
