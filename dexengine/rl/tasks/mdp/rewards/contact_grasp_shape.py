"""Grasp-shape rewards with their diagnostics: object-surface contact, thumb opposition, pre-grasp palm orientation and
pinch aperture (split from contact_grasp, which keeps the approach and pre-grasp palm terms and the shared helpers)."""

from __future__ import annotations

import torch

from dexengine.rl.tasks.mdp.rewards.contact_base import surf_pts_for, tip_to_surface_dist
from dexengine.rl.tasks.mdp.rewards.contact_grasp import grasp_centroid_world, safe_normalize
from dexengine.rl.tasks.mdp.rewards.contact_match import ContactMatchBase


class RewardGraspSurface(ContactMatchBase):
    """H-A: pull the WILL-CONTACT fingertips onto the object SURFACE (non-thumb
    fingers up-weighted to drive >thumb+index engagement). Object-relative, NOT
    MANO-fingertip matching.
        reward(N,) = (Σ_f w_f·exp(-k·relu(d_f))) / Σ_f w_f , on reference-contact frames.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._k = float(cfg.params.get("surface_k", 40.0))
        self._nonthumb = float(cfg.params.get("nonthumb_scale", 1.5))
        self._sane = False

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        surface_k: float = 40.0,
        nonthumb_scale: float = 1.5,
    ) -> torch.Tensor:
        self._ensure_grasp_targets(env)  # lazily populate _will_contact
        d = self._robot_tip_dist(env).clamp(min=0.0)  # (N,5)
        prox = torch.exp(-self._k * d)  # (N,5)
        wc = self._will_contact[self._traj_mgr.clip_ids].float()  # (N,5)
        w = wc.clone()
        w[:, 1:] = w[:, 1:] * self._nonthumb
        r = (prox * w).sum(dim=-1) / w.sum(dim=-1).clamp(min=1.0)  # (N,)
        active = self._ref_for_frame(env).any(dim=-1).float()  # contact window
        if not self._sane:
            print(
                f"[RewardGraspSurface][sanity] k={self._k} nonthumb={self._nonthumb} "
                f"env0 d_cm={(d[0] * 100).round().tolist()} r={r[0].item():.3f} "
                f"active={active[0].item():.0f}"
            )
            self._sane = True
        return r * active


class RewardGraspOpposition(ContactMatchBase):
    """H-A: reward thumb-vs-finger OPPOSITION around the grasp centroid (a pinch/
    wrap, not a same-side poke), proximity-gated so hovering opposed fingers earn
    nothing.
        oppo = clip(-cos(thumb-c, mean(index,middle)-c), 0, 1)
        gate = sigmoid(kn(near-d_thumb))·sigmoid(kn(near-min(d_index,d_middle)))
        reward = oppo·gate , on reference-contact frames."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._near = float(cfg.params.get("oppo_near_m", 0.03))
        self._kn = float(cfg.params.get("oppo_near_k", 200.0))
        self._sane = False

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        oppo_near_m: float = 0.03,
    ) -> torch.Tensor:
        tips = self._tip_positions_w(env)  # (N,5,3) tip SITES
        c = grasp_centroid_world(self, env)  # (N,3)
        dt = safe_normalize(tips[:, 0] - c)
        df = safe_normalize(tips[:, 1:3].mean(dim=1) - c)
        oppo = (-(dt * df).sum(dim=-1)).clamp(0.0, 1.0)  # (N,)
        d = self._robot_tip_dist(env)  # (N,5)
        g = torch.sigmoid(self._kn * (self._near - d[:, 0])) * torch.sigmoid(
            self._kn * (self._near - d[:, 1:3].min(dim=1).values)
        )
        r = oppo * g * self._ref_for_frame(env).any(dim=-1).float()
        if not self._sane:
            print(
                f"[RewardGraspOpposition][sanity] near={self._near} env0 oppo={oppo[0].item():.2f} "
                f"gate={g[0].item():.2f} r={r[0].item():.3f}"
            )
            self._sane = True
        return r


class RewardPreGraspOrient(ContactMatchBase):
    """H-D: reward the palm being oriented so the fingers face the grasp region
    (so closing fingers WRAP the object, not poke from above). Convention-free:
    palm 'open' dir = normalize(mean_fingertip - palm); reward alignment with the
    palm->grasp-centroid direction. Active on PRE-contact (approach) frames."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._palm_id = None
        self._sane = False

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        from dexengine.rl.tasks.mdp.naming import resolve_palm_body_id

        hand = env.scene[self._hand_key]
        if self._palm_id is None:
            self._palm_id = resolve_palm_body_id(hand, self._side)
        palm = hand.data.body_link_pos_w[:, self._palm_id, :]  # (N,3)
        tips = self._tip_positions_w(env)  # (N,5,3) tip SITES
        open_dir = safe_normalize(tips.mean(dim=1) - palm)  # (N,3)
        to_obj = safe_normalize(grasp_centroid_world(self, env) - palm)
        align = (open_dir * to_obj).sum(dim=-1).clamp(0.0, 1.0)  # (N,)
        active = (~self._ref_for_frame(env).any(dim=-1)).float()  # pre-contact
        if not self._sane:
            print(f"[RewardPreGraspOrient][sanity] env0 align={align[0].item():.2f} active={active[0].item():.0f}")
            self._sane = True
        return align * active


class RewardDiagNFingersContact(ContactMatchBase):
    """Diag (×1e-9 → recover): number of fingertips in OBJECT contact this step.
    Grasp-quality metric (target >2: beyond thumb+index)."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        return self._robot_contact(env).float().sum(dim=-1)  # (N,) in [0,5]


class RewardDiagOppo(ContactMatchBase):
    """Diag (×1e-9): raw thumb-vs-finger opposition cos in [0,1] (no gate)."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        tips = self._tip_positions_w(env)  # (N,5,3) tip SITES
        c = grasp_centroid_world(self, env)
        dt = safe_normalize(tips[:, 0] - c)
        df = safe_normalize(tips[:, 1:3].mean(dim=1) - c)
        return (-(dt * df).sum(dim=-1)).clamp(0.0, 1.0)


class RewardGraspAperture(ContactMatchBase):
    """Wrist-pose lever: reward the object centroid lying in the PINCH APERTURE — near the
    midpoint of the thumb-tip <-> finger-tips segment — so the wrist is guided to a pose where
    CLOSING the fingers pinches the object (object BETWEEN thumb and the curling fingers), not
    poking it from above. General antipodal criterion; an alternative to pregrasp_orient (which
    rewards fingers pointing down at the object). Dense.
        m = 0.5*(thumb_tip + mean(index..little tips)); reward = exp(-k*||grasp_centroid - m||)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._k = float(cfg.params.get("aperture_k", 10.0))
        # RE-ANCHOR: target for the pinch midpoint.
        #   "mano"    = MANO witness grasp_centroid (the non-graspable poke spot).
        #   "surface" = nearest point on the OBJECT SURFACE (reachable; reward the pinch closing ON the object).
        self._target = str(cfg.params.get("aperture_target", "mano"))
        self._sane = False

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        aperture_k: float = 10.0,
        aperture_target: str = "mano",
    ) -> torch.Tensor:
        tips = self._tip_positions_w(env)  # (N,5,3) tip SITES
        m = 0.5 * (tips[:, 0] + tips[:, 1:5].mean(dim=1))  # (N,3) pinch midpoint
        if self._target == "surface":  # distance from pinch midpoint to nearest object-surface point
            obj = env.scene[self._object_key]
            d = tip_to_surface_dist(m.unsqueeze(1), obj, surf_pts_for(self, env)).squeeze(1).clamp(min=0.0)
        else:  # MANO witness grasp centroid
            d = torch.norm(grasp_centroid_world(self, env) - m, dim=-1)
        r = torch.exp(-self._k * d)
        if not self._sane:
            print(
                f"[RewardGraspAperture][sanity] k={self._k} target={self._target} "
                f"env0 aperture_dist_cm={d[0].item() * 100:.2f} r={r[0].item():.3f}"
            )
            self._sane = True
        return r


class RewardDiagApertureDist(ContactMatchBase):
    """Diag (×1e-9): object-centroid -> pinch-midpoint distance (m). Should FALL as the wrist learns
    to put the object between thumb and fingers."""

    def __call__(
        self,
        env,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
        aperture_target: str = "mano",
    ) -> torch.Tensor:
        tips = self._tip_positions_w(env)  # (N,5,3) tip SITES
        m = 0.5 * (tips[:, 0] + tips[:, 1:5].mean(dim=1))
        if aperture_target == "surface":
            return (
                tip_to_surface_dist(m.unsqueeze(1), env.scene[self._object_key], surf_pts_for(self, env))
                .squeeze(1)
                .clamp(min=0.0)
            )
        return torch.norm(grasp_centroid_world(self, env) - m, dim=-1)
