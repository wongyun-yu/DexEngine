"""Evaluation environment with the full-arm robot.

Subclasses ManagerBasedRLEnv to attach a TrajectoryManager and provide
scene aliases so that existing MDP terms (observations, rewards,
terminations) that reference "right_hand" / "left_hand" work transparently
with the full robot articulation.

The scene has a single "robot" articulation (the full bimanual robot).
After scene construction, we inject aliases:
  scene["right_hand"] -> scene["robot"]
  scene["left_hand"]  -> scene["robot"]

This avoids forking every MDP term for the eval config.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from dexengine.core.desk import DESK_TOP_SURFACE_Z
from dexengine.rl.swallow import note_once
from dexengine.rl.tasks.mdp import gravity_wrench as _gravity_wrench
from dexengine.rl.tasks.mdp import physx_props as _physx_props
from dexengine.rl.tasks.scene import split_artic as _split_artic


def active_viz_sides(scene) -> tuple:
    """Sides whose TASK OBJECT exists in the scene — the single source of truth for
    side-aware visualization (arrow markers AND GT fingertip
    overlays draw ONLY for active sides). The full_robot USD keeps BOTH arms in every
    build, so body/link presence cannot distinguish uni builds; task-object presence
    can: right-uni spawns object_right (legacy: 'object') only, left-uni object_left
    only, bimanual both. Falls back to both sides if no object entity exists."""
    keys = scene.keys()
    sides = tuple(
        s
        for s, present in (
            ("right", ("object_right" in keys) or ("object" in keys)),
            ("left", "object_left" in keys),
        )
        if present
    )
    return sides or ("right", "left")


class EvalEnv(ManagerBasedRLEnv):
    """Evaluation environment with the full-arm robot.

    Extends ManagerBasedRLEnv by:
    1. Creating scene aliases so "right_hand"/"left_hand" resolve to "robot".
    2. Attaching a TrajectoryManager for reference trajectory data.

    MDP functions access trajectory data via ``env.trajectory_manager``.
    The arm is driven by DifferentialIK (inside the action term); the RL
    policy outputs the finger positions (the hand's finger width).

    """

    # Dotted module that exports ``TrajectoryManager`` (resolved at init to keep this module light).
    _trajectory_manager_module: str = "dexengine.rl.tasks.mdp.trajectory.manager"

    def load_managers(self):
        """Create scene aliases, trajectory manager, then load all other managers."""
        # Inject aliases: MDP terms do env.scene["right_hand"] / env.scene["left_hand"]
        # but the eval scene only has "robot". We alias both to the same articulation.
        robot = self.scene["robot"]
        self.scene._articulations["right_hand"] = robot

        # GATED split articulation (DEXENGINE_BI_SPLIT_ARTIC=1): the LEFT arm+hand
        # are a SEPARATE PhysX articulation `robot_left` (see cfg/residual_scene.py).
        # Re-point the `left_hand` alias + the trajectory manager's left_hand to it so
        # the left action term (asset_name="left_hand") and left obs/reward auto-route
        # to the decoupled left articulation. UNSET -> both alias to `robot` (identical).
        _split = _split_artic.enabled()
        left_articulation = self.scene["robot_left"] if _split else robot
        self.scene._articulations["left_hand"] = left_articulation
        if _split:
            # DOF-count sanity check: robot (right+head) ~21, robot_left (left) ~19.
            try:
                _nr = robot.num_joints
                _nl = left_articulation.num_joints
            except Exception:  # noqa: BLE001 — debug markers are best effort; a failure leaves them off
                _nr = _nl = "n/a"
            print(
                "[EvalEnv] split-articulation: TWO articulations initialized. "
                f"robot DOF={_nr} (right arm + head + right hand), "
                f"robot_left DOF={_nl} (left arm + left hand). "
                "left_hand alias + trajectory manager left_hand -> 'robot_left'."
            )

        # Resolve dataset-specific TrajectoryManager at runtime to avoid
        # coupling the shared module to a specific dataset module.
        TrajectoryManager = importlib.import_module(self.__class__._trajectory_manager_module).TrajectoryManager

        # Create trajectory manager with the robot articulation
        self.trajectory_manager = TrajectoryManager(
            cfg=self.cfg.trajectory,
            device=self.device,
            num_envs=self.num_envs,
            right_hand=robot,
            left_hand=left_articulation,
        )
        # Cache of the robot articulation's centres of mass: Isaac Lab re-reads get_coms() (CPU) every physics step.
        # Nothing writes the robot's COMs, so the value never changes; an oracle compares it bit for bit with PhysX
        # periodically. Installed before the managers build their observations.
        _physx_props.install_articulation_com_cache(self, (robot, left_articulation))
        super().load_managers()

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments with trajectory-aware initialization."""
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )
        self.trajectory_manager.reset(env_ids_t)

        # Write default joint positions FIRST (arms straight-down 0, hands zero).
        # This must happen BEFORE super()._reset_idx() because the event manager
        # (called within super) runs reset_hands_to_trajectory which writes the
        # correct finger positions from the trajectory reference. If we write
        # defaults AFTER, we overwrite the trajectory-initialized fingers with zeros.
        robot = self.scene["robot"]
        init_joint_pos = robot.data.default_joint_pos[env_ids_t].clone()
        robot.write_joint_state_to_sim(
            init_joint_pos,
            torch.zeros_like(init_joint_pos),
            env_ids=env_ids_t,
        )

        # GATED split-articulation eval: the LEFT arm+hand live in the separate
        # `robot_left` articulation, so its default joint state must be written
        # independently (the `robot` write above only covers the right+head view).
        if _split_artic.enabled():
            robot_left = self.scene["robot_left"]
            init_left = robot_left.data.default_joint_pos[env_ids_t].clone()
            robot_left.write_joint_state_to_sim(
                init_left,
                torch.zeros_like(init_left),
                env_ids=env_ids_t,
            )

        # Now run parent reset: scene.reset(), event_manager.apply("reset"), etc.
        # reset_hands_to_trajectory (in the event manager) overwrites finger joints
        # with the trajectory reference, which is the correct initialization.
        super()._reset_idx(env_ids)

    def step(self, action):
        """Step with optional debug marker visualization."""
        # Pre-physics event (object gravity wrench): this spot, after every pose write (judge park, SPAWN_HOLD,
        # snaps), is its only caller.
        _gravity_wrench.apply(self)
        result = super().step(action)
        if not hasattr(self, "_debug_markers_attempted"):
            self._debug_markers_attempted = True
            self._setup_debug_markers()
        if self._markers_can_show():
            self._update_debug_markers()
        return result

    def _markers_can_show(self) -> bool:
        """True when a frame can show the debug markers: a GUI or an RTX sensor (Isaac Lab's own render condition in
        ``step``) or a gym render mode (the video wrapper renders through it). Otherwise the per-step update is skipped:
        it only reads state and writes the /Visuals point instancers (no physics prim), and costs two device-to-host
        syncs plus two USD writes per marker per step. Prints ``[MARKERS]`` once when it first skips."""
        if self.render_mode is not None or self.sim.has_gui() or self.sim.has_rtx_sensors():
            return True
        if not getattr(self, "_markers_skip_logged", False):
            self._markers_skip_logged = True
            print(
                "[MARKERS] debug marker updates skipped — no GUI, no RTX sensor, no render mode (nothing renders)",
                flush=True,
            )
        return False

    def _setup_debug_markers(self):
        """Create wrist markers for visual debugging (lazy init).

        Two RGB axis-frames are drawn — one at the GT wrist pose (MANO trajectory
        target) and one at the robot's CURRENT wrist pose. Both use the standard
        FRAME_MARKER_CFG (RGB axes from frame_prim.usd) at uniform scale, which
        preserves the USD's native length-to-thickness ratio. GT and CURRENT are
        distinguished by SIZE: CURRENT is 1.2× larger than GT.
        """
        if hasattr(self, "_debug_markers_ready"):
            return
        try:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import FRAME_MARKER_CFG

            self._global_side_aware_marker(FRAME_MARKER_CFG, VisualizationMarkers)

            if "left" in self._viz_sides:
                # LEFT-hand mirror markers. Same RGB-axis frames and size convention.
                cfg_lw_gt = FRAME_MARKER_CFG.copy()
                cfg_lw_gt.markers["frame"].scale = (0.0375, 0.0375, 0.0375)  # LEFT wrist TARGET
                self._marker_wrist_target_frame_left = VisualizationMarkers(
                    cfg_lw_gt.replace(prim_path="/Visuals/wrist_target_frame_left")
                )
                cfg_lw_cur = FRAME_MARKER_CFG.copy()
                cfg_lw_cur.markers["frame"].scale = (0.045, 0.045, 0.045)  # LEFT wrist CURRENT
                self._marker_wrist_current_frame_left = VisualizationMarkers(
                    cfg_lw_cur.replace(prim_path="/Visuals/wrist_current_frame_left")
                )
                cfg_lo_gt = FRAME_MARKER_CFG.copy()
                cfg_lo_gt.markers["frame"].scale = (0.0375, 0.0375, 0.0375)  # LEFT object TARGET
                self._marker_obj_target_frame_left = VisualizationMarkers(
                    cfg_lo_gt.replace(prim_path="/Visuals/obj_target_frame_left")
                )
                cfg_lo_cur = FRAME_MARKER_CFG.copy()
                cfg_lo_cur.markers["frame"].scale = (0.045, 0.045, 0.045)  # LEFT object CURRENT
                self._marker_obj_current_frame_left = VisualizationMarkers(
                    cfg_lo_cur.replace(prim_path="/Visuals/obj_current_frame_left")
                )

            self._debug_markers_ready = True
            print(
                f"[{type(self).__name__}] Wrist + OBJECT-base markers created for ACTIVE "
                f"side(s) {self._viz_sides} (side = its task object present in scene; the "
                f"parked opposite arm draws NO arrows). wrist GT=0.0375/CUR=0.045; object "
                f"TARGET/CURRENT same sizes — object frames = the exact poses "
                f"object_tracking's pos/rot error is computed between. All RGB axes."
            )
        except Exception as e:  # noqa: BLE001 — debug markers are best effort; a failure leaves them off
            self._debug_markers_ready = False
            print(f"[{type(self).__name__}] Could not create debug markers: {e}")

    def _global_side_aware_marker(self, FRAME_MARKER_CFG, VisualizationMarkers):
        """GLOBAL side-aware marker gating: create a side's arrow markers ONLY when that side is active (see
        active_viz_sides — module-level single source of truth, shared with the render fingertip overlay).
        """
        self._viz_sides = active_viz_sides(self.scene)

        self._marker_wrist_target_frame = None
        self._marker_wrist_current_frame = None
        self._marker_obj_target_frame = None
        self._marker_obj_current_frame = None
        self._marker_wrist_target_frame_left = None
        self._marker_wrist_current_frame_left = None
        self._marker_obj_target_frame_left = None
        self._marker_obj_current_frame_left = None

        if "right" in self._viz_sides:
            cfg_gt = FRAME_MARKER_CFG.copy()
            cfg_gt.markers["frame"].scale = (0.0375, 0.0375, 0.0375)  # GT (target) — smaller
            self._marker_wrist_target_frame = VisualizationMarkers(
                cfg_gt.replace(prim_path="/Visuals/wrist_target_frame")
            )

            cfg_cur = FRAME_MARKER_CFG.copy()
            cfg_cur.markers["frame"].scale = (0.045, 0.045, 0.045)  # CURRENT — 1.2× larger than GT
            self._marker_wrist_current_frame = VisualizationMarkers(
                cfg_cur.replace(prim_path="/Visuals/wrist_current_frame")
            )

            # OBJECT base frame markers — drawn at EXACTLY the two poses object_tracking
            # compares, so the pos/rot error is visually inspectable.
            # TARGET = reference object pose, CURRENT = sim object rigid-body root.
            cfg_obj_gt = FRAME_MARKER_CFG.copy()
            cfg_obj_gt.markers["frame"].scale = (
                0.0375,
                0.0375,
                0.0375,
            )  # object TARGET (reference) — same as wrist GT
            self._marker_obj_target_frame = VisualizationMarkers(
                cfg_obj_gt.replace(prim_path="/Visuals/obj_target_frame")
            )
            cfg_obj_cur = FRAME_MARKER_CFG.copy()
            cfg_obj_cur.markers["frame"].scale = (
                0.045,
                0.045,
                0.045,
            )  # object CURRENT (sim root) — same as wrist CURRENT
            self._marker_obj_current_frame = VisualizationMarkers(
                cfg_obj_cur.replace(prim_path="/Visuals/obj_current_frame")
            )

    def _update_debug_markers(self):
        """Update wrist/object frame markers (env 0 only) for the ACTIVE sides.

        Side-aware dispatch: each side's markers update only
        when that side is active (its task object in the scene — ``active_viz_sides``);
        the parked opposite arm draws nothing.
        """
        if not getattr(self, "_debug_markers_ready", False):
            return
        sides = getattr(self, "_viz_sides", ("right", "left"))
        if "right" in sides:
            self._update_debug_markers_right()
        if "left" in sides:
            self._update_debug_markers_left()

    def _update_debug_markers_right(self):
        """RIGHT wrist + object markers (env 0)."""
        import isaaclab.utils.math as math_utils  # noqa: F401

        robot = self.scene["robot"]

        # Find link7 body index (cache on first call)
        if not hasattr(self, "_debug_ee_idx"):
            try:
                # EE link name from the active ArmSpec (an arm-specific literal would silently kill the rby1 markers)
                from dexengine.core.arms import ARM_REGISTRY as _AR
                from dexengine.rl.tasks.mdp.naming import _active_arm_name as _aan

                ee_ids, _ = robot.find_bodies(_AR[_aan()].ee_body_name("right"))
                self._debug_ee_idx = ee_ids[0]
            except Exception:  # noqa: BLE001 — debug markers are best effort; a failure leaves them off
                self._debug_ee_idx = None
                return

        if self._debug_ee_idx is None:
            return

        # Current wrist from link7 via the SIDE-AWARE mount (right Rz(-90°)), funneled
        # through mdp.wrist_kinematics so the marker can never desync from the
        # action/obs/reward mount (the same Rz(-90)/0.0555 mount).
        from dexengine.rl.tasks.mdp.wrist_kinematics import link7_to_virtual_wrist, mount_tensors

        link7_pos = robot.data.body_pos_w[0:1, self._debug_ee_idx]
        link7_quat = robot.data.body_quat_w[0:1, self._debug_ee_idx]
        _r_fwd, _, _t_off = mount_tensors(robot, "right", self.device)
        wrist_pos, wrist_quat = link7_to_virtual_wrist(link7_pos, link7_quat, _r_fwd, _t_off)

        self._marker_wrist_current_frame.visualize(wrist_pos, wrist_quat)

        # Target wrist from trajectory reference
        try:
            traj_mgr = self.trajectory_manager
            # get_wrist_pos/quat_w expects (num_envs,) shape — pass full buf, select env 0
            all_target_pos = traj_mgr.get_wrist_pos_w(self.episode_length_buf)
            all_target_quat = traj_mgr.get_wrist_quat_w(self.episode_length_buf)
            target_pos = all_target_pos[0:1] + self.scene.env_origins[0:1]
            target_quat = all_target_quat[0:1]
            # _wrist_poses are stored with whatever traj_offset_z the env_cfg
            # set at load (eval env: 0.76, train_arm env: 0). For the marker
            # to always show the WORLD-frame target on top of the desk
            # surface, compensate when raw-z convention was loaded.
            traj_off = float(getattr(traj_mgr.cfg, "traj_offset_z", 0.0) or 0.0)
            if traj_off < 0.1:  # i.e. ~0 — raw replay-convention z
                desk_h = float(getattr(traj_mgr, "_desk_height", DESK_TOP_SURFACE_Z))
                target_pos = target_pos.clone()
                target_pos[..., 2] += desk_h
            self._marker_wrist_target_frame.visualize(target_pos, target_quat)
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as _exc:
            note_once("DEBUG_MARKERS right target", _exc)

        # OBJECT base frame markers (env 0): draw EXACTLY the poses object_tracking compares.
        # CURRENT = sim object rigid-body ROOT (obj.data.root_pos_w / root_quat_w); TARGET = the
        # reference object pose (ref[right_obj_pos] + env_origin, ref[right_obj_quat]). The drawn
        # gap between them IS pos_err = ||TARGET - CURRENT||; their angular offset IS rot_err.
        if getattr(self, "_marker_obj_current_frame", None) is not None:
            try:
                _ok = "object_right" if "object_right" in self.scene.keys() else "object"  # noqa: SIM118  — InteractiveScene has no __contains__
                obj = self.scene[_ok]
                self._marker_obj_current_frame.visualize(obj.data.root_pos_w[0:1], obj.data.root_quat_w[0:1])
                ref_state = self.trajectory_manager.get_reference_state(self.episode_length_buf)
                ref_obj_pos = ref_state["right_obj_pos"][0:1] + self.scene.env_origins[0:1]
                ref_obj_quat = ref_state["right_obj_quat"][0:1]
                self._marker_obj_target_frame.visualize(ref_obj_pos, ref_obj_quat)
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as _exc:
                note_once("DEBUG_MARKERS right object", _exc)

    def _update_debug_markers_left(self):
        """LEFT wrist + object mirror markers (env 0): the LEFT link7 body, the LEFT
        wrist reference (get_wrist_pos_w(side='left')) and the LEFT object
        (object_left / left_obj_*) — same construction as the right method.
        GATED split-articulation eval: the left arm's EE link lives in ``robot_left``,
        so read the left link + its body poses from there (falls back to ``robot``)."""
        robot = self.scene["robot"]
        left_robot = robot
        if _split_artic.enabled():
            try:
                left_robot = self.scene["robot_left"]
            except Exception:  # noqa: BLE001 — debug markers are best effort; a failure leaves them off
                left_robot = robot
        if not hasattr(self, "_debug_ee_idx_left"):
            # Resolve left link7 from the left articulation, falling back to the main
            # robot (non-split bimanual keeps both arms in `robot`). Cache the source so
            # the wrist read below uses the SAME articulation the index came from.
            self._debug_ee_idx_left = None
            self._debug_left_robot = left_robot
            from dexengine.core.arms import ARM_REGISTRY as _AR
            from dexengine.rl.tasks.mdp.naming import _active_arm_name as _aan

            _lee_name = _AR[_aan()].ee_body_name("left")
            for _src in (left_robot, robot):
                try:
                    lee_ids, _ = _src.find_bodies(_lee_name)
                    if lee_ids:
                        self._debug_ee_idx_left = lee_ids[0]
                        self._debug_left_robot = _src
                        break
                except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as _exc:
                    note_once("DEBUG_MARKERS left ee body", _exc)
            print(f"[{type(self).__name__}] LEFT marker link7 idx = {self._debug_ee_idx_left}")
        left_robot = getattr(self, "_debug_left_robot", left_robot)
        if getattr(self, "_debug_ee_idx_left", None) is not None:
            try:
                # LEFT current wrist via the SIDE-AWARE mount (left = Rz(+90°), the mirror
                # of the right's Rz(-90°)); same keystone funnel as the right marker so the
                # two can never diverge from the action/obs/reward mount.
                from dexengine.rl.tasks.mdp.wrist_kinematics import link7_to_virtual_wrist, mount_tensors

                l_link7_pos = left_robot.data.body_pos_w[0:1, self._debug_ee_idx_left]
                l_link7_quat = left_robot.data.body_quat_w[0:1, self._debug_ee_idx_left]
                _lr_fwd, _, _lt_off = mount_tensors(left_robot, "left", self.device)
                l_wrist_pos, l_wrist_quat = link7_to_virtual_wrist(l_link7_pos, l_link7_quat, _lr_fwd, _lt_off)
                self._marker_wrist_current_frame_left.visualize(l_wrist_pos, l_wrist_quat)
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as _exc:
                note_once("DEBUG_MARKERS left wrist", _exc)
            try:
                traj_mgr = self.trajectory_manager
                l_target_pos = (
                    traj_mgr.get_wrist_pos_w(self.episode_length_buf, side="left")[0:1] + self.scene.env_origins[0:1]
                )
                l_target_quat = traj_mgr.get_wrist_quat_w(self.episode_length_buf, side="left")[0:1]
                traj_off = float(getattr(traj_mgr.cfg, "traj_offset_z", 0.0) or 0.0)
                if traj_off < 0.1:  # raw replay-convention z → lift onto the desk for the WORLD-frame marker
                    desk_h = float(getattr(traj_mgr, "_desk_height", DESK_TOP_SURFACE_Z))
                    l_target_pos = l_target_pos.clone()
                    l_target_pos[..., 2] += desk_h
                self._marker_wrist_target_frame_left.visualize(l_target_pos, l_target_quat)
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as _exc:
                note_once("DEBUG_MARKERS left target", _exc)

        # LEFT object markers — INDEPENDENT of the left link7 lookup (mirror the right
        # object block). Previously nested inside the `_debug_ee_idx_left` gate, so when
        # left link7 didn't resolve BOTH left wrist AND left object markers froze at their
        # initial pose. Object markers only need object_left + the trajectory reference.
        if getattr(self, "_marker_obj_current_frame_left", None) is not None:
            try:
                obj_l = self.scene["object_left"]
                self._marker_obj_current_frame_left.visualize(obj_l.data.root_pos_w[0:1], obj_l.data.root_quat_w[0:1])
                ref_state_l = self.trajectory_manager.get_reference_state(self.episode_length_buf)
                ref_obj_pos_l = ref_state_l["left_obj_pos"][0:1] + self.scene.env_origins[0:1]
                ref_obj_quat_l = ref_state_l["left_obj_quat"][0:1]
                self._marker_obj_target_frame_left.visualize(ref_obj_pos_l, ref_obj_quat_l)
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as _exc:
                note_once("DEBUG_MARKERS left object", _exc)
