"""Contact-match base term (reference vs robot fingertip contact), its termination and its reward."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from dexengine.core.physics import GRAVITY_MPS2

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from dexengine.rl.tasks.mdp.rewards.contact_base import (
    OBJ_SENSOR_NAMES,
    _load_fingertip_contact_label,
    _load_grasp_targets,
    load_object_surface_points,
    resolve_tip_body_ids,
    resolve_tip_offsets_tensor,
    surf_pts_for,
    tip_sites_w,
    tip_to_surface_dist,
)


class ContactMatchBase(ManagerTermBase):
    """Shared machinery for the contact-match reward + its diagnostics.

    Resolves tip body ids + the PhysX ``{side}_hand_contact`` sensor + object
    surface points, and loads the new fingertip-region reference label.
    Subclasses call ``_ref_and_robot(env)`` to obtain aligned (N,5) bool
    tensors for the current frame.
    """

    #: The scene object a term reads when its params name none — read by __init__ and by the D16 idle-side sweep
    #: (env_setup.sides._term_object_binding), so the two cannot disagree.
    OBJECT_KEY_DEFAULT = "object_right"

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._side = cfg.params.get("side", "right")
        self._contact_source = cfg.params.get("contact_source", "mano")
        # Per-side contact annotation: contact_<source>_<side>.npz (both sides
        # suffixed; the right-implicit contact_<source>.npz was renamed).
        self._contact_source_eff = f"{self._contact_source}_{self._side}"
        self._F_thresh = float(cfg.params.get("F_threshold", 0.5))
        self._d_thresh = float(cfg.params.get("d_threshold", 0.01))
        self._hand_key = f"{self._side}_hand" if f"{self._side}_hand" in env.scene.keys() else "robot"  # noqa: SIM118  — InteractiveScene has no __contains__
        hand = env.scene[self._hand_key]
        self._tip_body_ids = resolve_tip_body_ids(hand, self._side)
        # Tip-site offsets (SharpA: ~2.6 cm DP-origin -> physical tip; None = origins,
        # byte-identical RH56F1 path). Resolved ONCE here; consumed by _tip_positions_w.
        self._tip_offsets = resolve_tip_offsets_tensor(hand, self._side, env.device)
        mesh_dir = cfg.params["mesh_dir"]
        # --multi_object: comma-list (LOADER clip order) -> per-clip padded cloud
        # (pads at +1e6 so a pad point is never the nearest); gathered by clip id.
        _dirs = [d for d in str(mesh_dir).split(",") if d]
        self._surf_multi = None
        if len(_dirs) > 1:
            _clouds = [load_object_surface_points(d, env.device) for d in _dirs]
            _maxp = max(c.shape[0] for c in _clouds)
            self._surf_multi = torch.full((len(_clouds), _maxp, 3), 1.0e6, device=env.device)
            for _i, _c in enumerate(_clouds):
                self._surf_multi[_i, : _c.shape[0]] = _c
            self._surf_local = _clouds[0]
        else:
            self._surf_local = load_object_surface_points(mesh_dir, env.device)
        self._object_key = cfg.params.get("object_key", self.OBJECT_KEY_DEFAULT)
        self._traj_mgr = env.trajectory_manager
        # slowdown_factor: stored on the trajectory manager (private attr set in __init__)
        self._slowdown = float(getattr(self._traj_mgr, "_slowdown_factor", 1.0))
        # Force-direction reward params (defaults match the reward; diags share them).
        self._force_cap = float(cfg.params.get("force_cap", 30.0))
        self._force_k = int(cfg.params.get("force_k", 3))
        self._force_eps = float(cfg.params.get("force_motion_eps", 0.003))  # 0.3 cm over k frames
        self._force_pre = int(cfg.params.get("force_window_pre", 3))
        self._force_post = int(cfg.params.get("force_window_post", 50))
        # New fingertip-region reference label, aligned to clip_ids/start_frames.
        # seq_dir param is the trajectory ROOT dir (containing <seq>/0/...);
        # we accept either an explicit data dir or fall back to the traj
        # manager's cfg.data_dir (TrajectoryManager stores self.cfg = cfg).
        data_dir = cfg.params.get("seq_dir", None)
        if data_dir is None:
            data_dir = self._traj_mgr.cfg.data_dir
        self._data_dir = data_dir
        self._ref_label = _load_fingertip_contact_label(
            data_dir, self._contact_source_eff, self._traj_mgr, env.device
        )  # (num_clips, max_length, 5) bool
        self._sensor_tip_ids: list[int] | None = None
        self._obj_sensors = None  # cached per-fingertip object-filtered sensors
        # Lazily-loaded witness_obj GT-target machinery (RewardGraspTargetSoft / Approach):
        self._grasp_targets = None  # (num_clips, 5, 3) object-local contact point per finger
        self._will_contact = None  # (num_clips, 5) bool: finger contacts at some point
        self._first_contact = None  # (num_clips,) int: first frame ANY finger contacts (force-window anchor)
        self._ref_start = None  # (num_clips, 5) int: per-finger first ref-contact frame (A1 per-finger window)
        self._ref_end = None  # (num_clips, 5) int: per-finger last ref-contact frame  (A1 per-finger window)
        self._force_sign_logged = False

    def _ref_for_frame(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Slowdown-aware (N,5) reference contact for the current frame.

        Indexing:
            af = floor(episode_length_buf / slowdown) + start_frames, clamped.
        Per-env clip selection via traj_mgr.clip_ids.
        """
        af = self._abs_frame(env)  # side-aware under the per-side clock
        af = af.clamp(0, self._ref_label.shape[1] - 1)
        return self._ref_label[self._traj_mgr.clip_ids, af]  # (N,5) bool

    def _robot_contact(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Live (N,5) robot per-tip OBJECT contact = object-filtered force > F_thresh.

        SINGLE, UNAMBIGUOUS PATH: the 5 per-fingertip OBJECT-filtered ContactSensors
        (``_robot_obj_force_mag``, filtering ``/Object/base_link``). NO distance gate,
        NO net force, NO fallback.

        Why object-filtered force only (contact-metric fix + strict cleanup): the old
        gate was ``net_force > F AND nearest-surface < 1cm``,
        doubly wrong — net force includes desk/self contact (false positives), and the
        nearest-surface distance was measured from the fingertip BODY ORIGIN (~1.8 cm
        inside the collision pad) so it never dropped below 1 cm even at hard contact.
        Measured: a thumb pressed a knife with 20.3 N object-filtered force
        yet robot_contact_frac read 0. Object-filtered force is geometrically correct
        and cleanly excludes desk/self, so it is the ONLY signal for the binary contact
        metric, its diagnostics, and the binary match reward.

        Net-force fallback REMOVED: if the per-fingertip object sensors are missing,
        ``_robot_force_contact`` raises RuntimeError. We deliberately do NOT silently
        fall back to net force (contaminated). Fix the scene config instead.
        """
        return self._robot_force_contact(env, self._F_thresh)  # (N,5) objforce > F; raises if sensors absent

    def _tip_positions_w(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """(N,5,3) WORLD fingertip SITE positions — THE single source for every
        object-distance quantity in this class. SharpA: DP origin + R·offset
        (physical pad tip); RH56F1/zero offsets: exactly the body origins."""
        return tip_sites_w(env.scene[self._hand_key], self._tip_body_ids, self._tip_offsets)

    def _robot_tip_dist(self, env: ManagerBasedRLEnv, tips_w: torch.Tensor | None = None) -> torch.Tensor:
        """Live (N,5) robot fingertip-SITE -> object surface distance (meters).

        Pure proximity — no force gate, no threshold; feeds the distance-based
        SOFT/surface/closure rewards and the d_cm diagnostics. ``tips_w`` is an
        internal override used only by the one-shot before/after sanity prints.
        """
        obj = env.scene[self._object_key]
        if tips_w is None:
            tips_w = self._tip_positions_w(env)  # (N,5,3) WORLD
        return tip_to_surface_dist(tips_w, obj, surf_pts_for(self, env))  # (N,5) meters

    # ------------------------------------------------------------------
    # witness_obj GT-target machinery (RewardGraspTargetSoft / RewardApproachSoft).
    # The GT-target is the per-finger ground-truth contact point on the OBJECT
    # (from the annotation's witness_obj), expressed OBJECT-LOCAL and re-anchored
    # to the LIVE object each step. This is NOT nearest-surface and NOT the
    # reference fingertip location — it is the specific grasp point on the object.
    # ------------------------------------------------------------------
    def _ensure_grasp_targets(self, env: ManagerBasedRLEnv) -> None:
        if self._grasp_targets is None:
            (self._grasp_targets, self._will_contact, self._first_contact, self._ref_start, self._ref_end) = (
                _load_grasp_targets(
                    self._data_dir, self._contact_source_eff, self._traj_mgr, env.device, side=self._side
                )
            )

    def _abs_frame(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """(N,) absolute reference frame index = floor(elen/slowdown)+start_frames — the SAME
        mapping _ref_for_frame uses. Drives the A1 per-finger closure windows.
        PER-SIDE CLOCK (reset_scheme=decoupled): this side's OWN clock (annotation
        windows follow the side's clip segment); disabled -> legacy shared mapping."""
        tm = self._traj_mgr
        if getattr(tm, "per_side_clock", False):
            # clamp=True: after an env reset draws a start at the clip's LAST frame
            # (RSI band hi ramps to 1.0), the side clock can sit one frame past the
            # end for a step before the completion-cycle snaps it (the elen>0 filter
            # skips freshly-reset envs) — unclamped indexing there is OOB (CUDA
            # device assert).
            return tm.side_abs_frames(self._side, env.episode_length_buf, clamp=True)
        return (env.episode_length_buf.float() / self._slowdown).floor().long() + self._traj_mgr.start_frames

    def _perfinger_windows(self, env: ManagerBasedRLEnv, margins: torch.Tensor):
        """Returns (af (N,1), will (N,5) bool, gfirst (N,1), rstart (N,5), hi (N,5)) for the A1
        per-finger closure logic. hi = ref_end - margin (per finger). Short/inverted windows
        (hi < ref_start) naturally yield no reward (the af<=hi test fails)."""
        self._ensure_grasp_targets(env)
        cid = self._traj_mgr.clip_ids
        af = self._abs_frame(env).unsqueeze(-1)  # (N,1)
        will = self._will_contact[cid]  # (N,5) bool
        gfirst = self._first_contact[cid].unsqueeze(-1)  # (N,1)
        rstart = self._ref_start[cid]  # (N,5)
        hi = self._ref_end[cid] - margins.unsqueeze(0)  # (N,5)
        return af, will, gfirst, rstart, hi

    def _robot_hand_force_vec(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """(N,3) HAND-applied force ON the object, WORLD frame.

        ``force_matrix_w`` reports the contact force ON the fingertip FROM the
        object (object-on-finger). By Newton's 3rd law the hand-on-object force
        is the NEGATIVE of the per-finger sum. (Sign asserted/logged once.)
        Reuses the 5 per-fingertip object-filtered sensors (table excluded).
        """
        if self._obj_sensors is None:
            _ = self._robot_obj_force_mag(env)  # resolves + caches self._obj_sensors (raises if absent)
        vecs = []
        for sensor in self._obj_sensors:
            fmat = sensor.data.force_matrix_w  # (N,1,F,3) object-on-finger | None
            if fmat is None:
                vecs.append(torch.zeros((env.num_envs, 3), device=env.device))
            else:
                vecs.append(fmat.sum(dim=(1, 2)))  # (N,3)
        obj_on_finger = torch.stack(vecs, dim=1).sum(dim=1)  # (N,3) sum over fingers
        F_hand_on_obj = -obj_on_finger  # Newton's 3rd law
        if not self._force_sign_logged:
            f0 = F_hand_on_obj[0].detach().cpu().numpy()
            print(
                f"[ForceDir][sign] F_hand_on_obj = -sum(force_matrix_w). env0 sample="
                f"[{f0[0]:.2f},{f0[1]:.2f},{f0[2]:.2f}]N (Fz<0 ⇒ hand pushes object DOWN). "
                f"force_matrix_w = object-on-finger; flipped to hand-on-object."
            )
            self._force_sign_logged = True
        return F_hand_on_obj

    def _force_quantities(self, env: ManagerBasedRLEnv) -> dict:
        """Shared per-step force-direction quantities, memoized on env by
        common_step_counter so the reward + its diagnostics compute it ONCE.

        proj    = F_hand_on_obj · F̂_req  (signed N)
        F_req/m = a_ref + g·ẑ  — the direction of the **force the reference requires**.
                  The old basis was the travel direction (d_ref = ref[t+k] - ref[t]), physically wrong because **while
                  decelerating the required force opposes the travel** (stopping a horizontal move). The mass only
                  enters the direction, so it cancels.
        gate2   = 0.5*(thumb_contact + index_contact)   (anti-poke soft scaling)
        window  = eff_frame in [first_contact - pre, first_contact + post]  (relative)
        motion  = ‖d_ref‖ > eps   (skip static frames; direction ill-defined)
        reward  = clip(0.5 + 0.5*proj/F_cap, 0, 1) * gate2 * window * motion  (UNWEIGHTED)
                  Option A (shifted, no-penalty): proj=-cap→0, 0→0.5, +cap→1. Gradient
                  even when misaligned (proj<0) so the policy can rotate a downward poke
                  into an aligned push; reward stays ≥0 (no penalty). The old positive-only
                  clip(proj/F_cap,0,1) gave zero reward AND zero gradient while proj<0,
                  making the term inert in the failure regime (reward share 0 %).
        """
        step = int(getattr(env, "common_step_counter", 0))
        # Per-SIDE cache key: bimanual right+left force_dir terms both evaluate in the same
        # step; a step-only cache let the 2nd-evaluated hand return the 1st hand's force q
        # (corrupting force_dir_left + its 5 diags). Key by side so each hand caches its own.
        _cache_attr = f"_force_q_cache_{self._side}"
        cache = getattr(env, _cache_attr, None)
        if cache is not None and cache[0] == step:
            return cache[1]
        self._ensure_grasp_targets(env)
        tm = self._traj_mgr
        elen = env.episode_length_buf
        F = self._robot_hand_force_vec(env)  # (N,3) hand->object, world
        ref_now = tm.get_reference_state(elen)[f"{self._side}_obj_pos"]
        ks = max(1, int(round(self._force_k * self._slowdown)))  # advance ~k effective frames
        ref_fut = tm.get_reference_state(elen + ks)[f"{self._side}_obj_pos"]
        ref_prv = tm.get_reference_state((elen - ks).clamp(min=0))[f"{self._side}_obj_pos"]
        # The gate is as before: only **frames where the reference moves** (static stretches excluded). Only the
        # direction changes below.
        d_travel = ref_fut - ref_now
        mag = torch.norm(d_travel, dim=-1)
        motion_ok = mag > self._force_eps
        # Direction of the required force: Newton F_hand = m·a − m·g_vec, g_vec = (0,0,−9.81) -> F_hand = m(a + 9.81 ẑ).
        # The mass is a positive scalar and cancels in the normalisation, so the direction is exact without knowing it.
        # Assumption: in this window (right after first contact) the object is off the desk and the hand contact force
        # is the only external force.
        dt_eff = max(1e-4, float(getattr(env, "step_dt", 1.0 / 60.0)) * ks)
        a_ref = (ref_fut - 2.0 * ref_now + ref_prv) / (dt_eff * dt_eff)  # (N,3) m/s^2
        d = a_ref.clone()
        d[:, 2] = d[:, 2] + GRAVITY_MPS2
        dhat = d / torch.norm(d, dim=-1).clamp(min=1e-6).unsqueeze(-1)
        proj = (F * dhat).sum(dim=-1)  # (N,) signed Newtons
        Fz = F[:, 2]
        eff = self._abs_frame(env)  # side-aware under the per-side clock
        fc = self._first_contact[tm.clip_ids]
        window_ok = (fc >= 0) & (eff >= fc - self._force_pre) & (eff <= fc + self._force_post)
        rc = self._robot_contact(env)  # (N,5) bool (object force)
        gate2 = 0.5 * (rc[:, 0].float() + rc[:, 1].float())  # thumb+index: 0/0.5/1.0
        reward = (
            torch.clamp(0.5 + 0.5 * proj / self._force_cap, 0.0, 1.0) * gate2 * window_ok.float() * motion_ok.float()
        )
        q = {
            "proj": proj,
            "Fz": Fz,
            "gate2": gate2,
            "window_ok": window_ok,
            "motion_ok": motion_ok,
            "reward": reward,
            "active": (window_ok & motion_ok & (gate2 > 0)),
            "ti_both": (rc[:, 0] & rc[:, 1]).float(),
        }
        setattr(env, _cache_attr, (step, q))
        return q

    def _robot_tip_to_grasp_target_dist(
        self, env: ManagerBasedRLEnv, tips_w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Live (N,5) robot fingertip-SITE -> per-finger GT contact target (meters).

        target_world = obj_pos_w + R_obj_w · c_local  (c_local = object-local
        witness_obj contact point). Object-relative (follows the LIVE object) and
        region-correct (the actual grasp point), unlike nearest-surface. ``tips_w``
        is an internal override used only by the one-shot before/after sanity prints.
        """
        from isaaclab.utils.math import quat_apply

        self._ensure_grasp_targets(env)
        obj = env.scene[self._object_key]
        c_local = self._grasp_targets[self._traj_mgr.clip_ids]  # (N,5,3) object-local
        N = c_local.shape[0]
        q = obj.data.root_quat_w.unsqueeze(1).expand(N, 5, 4).reshape(-1, 4)
        t_world = quat_apply(q, c_local.reshape(-1, 3)).reshape(N, 5, 3) + obj.data.root_pos_w.unsqueeze(
            1
        )  # (N,5,3) WORLD
        if tips_w is None:
            tips_w = self._tip_positions_w(env)  # (N,5,3) WORLD
        return torch.norm(tips_w - t_world, dim=-1)  # (N,5) meters

    def _will_contact_mask(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """(N,5) bool: per env, does each finger contact the object at SOME point
        in its clip (frame-independent). Used by the dense approach reward."""
        self._ensure_grasp_targets(env)
        return self._will_contact[self._traj_mgr.clip_ids]  # (N,5) bool

    def _robot_obj_force_mag(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """(N,5) per-fingertip OBJECT-filtered contact force magnitude (Newtons).

        Reads the 5 single-body per-fingertip contact sensors (``OBJ_SENSOR_NAMES``),
        each filtering ``/Object/base_link`` → ``force_matrix_w`` shape (N,1,n_filters,3).
        A single multi-body hand sensor cannot populate force_matrix_w in multi-env
        (PhysX filter-cardinality mismatch), so object-filtered force MUST come from
        these per-fingertip sensors. Order: thumb, index, middle, ring, little.
        """
        if self._obj_sensors is None:
            names = OBJ_SENSOR_NAMES[self._side]
            missing = [nm for nm in names if nm not in env.scene.keys()]  # noqa: SIM118  — InteractiveScene has no __contains__
            if missing:
                raise RuntimeError(
                    f"Object-filtered contact gate needs per-fingertip ContactSensors "
                    f"{names} but the scene is missing {missing}. Add 5 single-body "
                    f"ContactSensorCfgs (one per fingertip body, filter "
                    f"'{{ENV_REGEX_NS}}/Object/base_link') to this scene cfg — a "
                    f"single multi-body hand sensor cannot populate force_matrix_w in "
                    f"multi-env (PhysX filter cardinality)."
                )
            self._obj_sensors = [env.scene[nm] for nm in names]
        forces = []
        for sensor in self._obj_sensors:
            fmat = sensor.data.force_matrix_w  # (N,1,F,3) | None
            if fmat is None:
                forces.append(torch.zeros(env.num_envs, device=env.device))
            else:
                # sum contact-force vectors over body(1) + filter dims, then magnitude
                forces.append(fmat.sum(dim=(1, 2)).norm(dim=-1))  # (N,)
        return torch.stack(forces, dim=1)  # (N,5)

    def _robot_force_contact(self, env: ManagerBasedRLEnv, f_min: float) -> torch.Tensor:
        """Per-fingertip OBJECT-filtered contact gate: force_matrix_w > f_min. (N,5) bool."""
        return self._robot_obj_force_mag(env) > f_min

    def _ref_and_robot(self, env: ManagerBasedRLEnv):
        return self._ref_for_frame(env), self._robot_contact(env)


class HandContactMismatchSustained(ContactMatchBase):
    """Hand-level contact-mismatch termination on the NEW per-finger annotation.
    Terminate when (reference expects ANY fingertip contact) AND (robot has NO
    object contact) is sustained >= grace_steps. Robot contact = object-filtered
    fingertip force > force_threshold (distance gate deferred — see docs)."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._grace_steps = int(cfg.params.get("grace_steps", 50))
        self._f_min = float(cfg.params.get("force_threshold", 0.05))
        attr = f"_contact_mismatch_counter_{self._side}"
        if not hasattr(env, attr):
            setattr(env, attr, torch.zeros(env.num_envs, dtype=torch.int32, device=env.device))
        self._counter_attr = attr
        self._sanity_done = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        side="right",
        contact_source="mano",
        seq_dir="",
        mesh_dir="",
        force_threshold=0.05,
        grace_steps=50,
        object_key="object_right",
        snap_only: bool = False,
    ) -> torch.Tensor:
        ref_any = self._ref_for_frame(env).any(dim=-1)  # (N,)
        robot_any = self._robot_force_contact(env, self._f_min).any(dim=-1)  # (N,)
        mismatch = ref_any & ~robot_any
        counter = getattr(env, self._counter_attr)
        new_counter = torch.where(mismatch, counter + 1, torch.zeros_like(counter))
        new_counter = torch.where(env.episode_length_buf == 0, torch.zeros_like(new_counter), new_counter)
        setattr(env, self._counter_attr, new_counter)
        done = new_counter >= self._grace_steps
        # RESET SCHEME=decoupled: SNAP-ONLY — stash the side-fail mask
        # for the env's per-side snap instead of terminating. The side snap zeroes
        # this side's mismatch counter, re-granting the full grace window.
        if snap_only:
            # Post-snap grace (mirror of ObjectTrackingExceeded's): disarmed for
            # _pss_grace steps after this side's last snap.
            _last = getattr(env, "_pss_last_snap", None)
            if _last is not None:
                _g = int(getattr(env, "_pss_grace", 0))
                _i = 1 if self._side == "left" else 0
                done = done & ((int(env.common_step_counter) - _last[:, _i]) > _g)
            _attr = f"_side_fail_{self._side}"
            buf = getattr(env, _attr, None)
            if buf is None or buf.shape != done.shape:
                buf = torch.zeros_like(done)
            setattr(env, _attr, buf | done)
            done = torch.zeros_like(done)
        if not self._sanity_done:
            print(
                f"[HandContactMismatchSustained][sanity] side={self._side} source={self._contact_source} "
                f"grace={self._grace_steps} f_min={self._f_min:.3f}N env0: ref_any={ref_any[0].item()} "
                f"robot_any={robot_any[0].item()} done={done[0].item()}"
            )
            self._sanity_done = True
        return done


class RewardContactMatch(ContactMatchBase):
    """Per-finger POSITIVE contact-match reward (contact pilot).

    reward(N,) = sum over 5 fingers of (ref AND robot_contact).float()
    Range [0, 5]; the RewTerm weight (contact_reward_weight) scales it.
    Binary additive — each finger that is BOTH supposed-to-touch (ref) AND
    actually-touching (robot) earns +1.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sanity_done = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        side: str = "right",
        contact_source: str = "mano",
        seq_dir: str = "",
        mesh_dir: str = "",
        F_threshold: float = 0.5,
        d_threshold: float = 0.01,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        ref, robot = self._ref_and_robot(env)  # (N,5), (N,5) bool
        match = ref & robot  # (N,5)
        reward = match.float().sum(dim=-1)  # (N,) in [0,5]
        if not self._sanity_done:
            print(
                f"[RewardContactMatch][sanity] side={self._side} "
                f"source={self._contact_source} env0: "
                f"ref={ref[0].cpu().tolist()} robot={robot[0].cpu().tolist()} "
                f"match={match[0].cpu().tolist()} reward={reward[0].item():.2f}  "
                f"(F_thresh={self._F_thresh:.2f}N d_thresh={self._d_thresh * 100:.1f}cm "
                f"slowdown={self._slowdown:.2f})"
            )
            self._sanity_done = True
        return reward
