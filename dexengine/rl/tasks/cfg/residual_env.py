"""The residual training env cfg class: managers plus the PhysX and sim settings of __post_init__."""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from dexengine.core.desk import apply_desk_top as _apply_desk_top
from dexengine.core.desk import desk_size_mark as _desk_size_mark
from dexengine.core.desk import desk_top as _desk_top
from dexengine.core.desk import run_dataset as _desk_run_dataset
from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.cfg.residual import physx_shift
from dexengine.rl.tasks.scene import split_artic as _split_artic


def _residual_env_cfg(
    *,
    ArmResidualActionsCfg,
    ArmResidualEventCfg,
    ArmResidualSceneCfg,
    ArmResidualTerminationsCfg,
    left_present,
    max_depenetration_velocity,
    task_curriculum_cfg_cls,
    task_observations_cfg_cls,
    task_rewards_cfg_cls,
    train_arm_env_cfg_cls,
):
    """The residual training env cfg class (scene, actions, events and the PhysX/sim settings in __post_init__)."""

    @configclass
    class TrainArmResidualEnvCfg(train_arm_env_cfg_cls):
        """Arm-in-loop training: object manipulation with residual."""

        scene: ArmResidualSceneCfg = ArmResidualSceneCfg(num_envs=4096, env_spacing=2.5)
        observations: task_observations_cfg_cls = task_observations_cfg_cls()  # type: ignore[valid-type]
        actions: ArmResidualActionsCfg = ArmResidualActionsCfg()
        rewards: task_rewards_cfg_cls = task_rewards_cfg_cls()  # type: ignore[valid-type]
        terminations: ArmResidualTerminationsCfg = ArmResidualTerminationsCfg()
        events: ArmResidualEventCfg = ArmResidualEventCfg()
        curriculum: task_curriculum_cfg_cls = task_curriculum_cfg_cls()  # type: ignore[valid-type]

        # Runtime overrides
        object_usd_path: str | None = None

        # Bimanual SELECTIVE cross-hand-only self-collision (bimanual build only;
        # single-hand leaves it False = byte-identical). The env class
        # (ArmResidualTrainEnv.__init__) reads this to swap in BimanualSelfCollisionScene,
        # which (paired with enabled_self_collisions=True set in __post_init__) enables
        # ONLY left_hand<->right_hand collision and keeps inner-hand/inner-arm OFF.
        bimanual_self_collision_filter: bool = left_present

        def __post_init__(self):
            super().__post_init__()

            # The desk is fitted as one piece (collision top + visual USD + legs). Its centre x and size come from the
            # run dataset's top statement (dexengine.core.desk.desk_top, the function the judge and bake read); the env
            # checks it against the measurement (dexengine.rl.tasks.scene.desk_check).
            _desk_msg = _apply_desk_top(self.scene, _desk_top())
            print(_desk_size_mark(_desk_run_dataset()), flush=True)

            # ContactSensorCfg does NOT enable contact reporting — it REQUIRES it
            # (contact_sensor.py:265 collects only bodies that already carry
            # PhysxContactReportAPI, applied per-prim by the USD bake). The OBJECT spawn sets
            # activate_contact_sensors for its side. Do NOT enable it on the whole robot
            # spawn — that tracks contacts on all 41 bodies and blows up
            # mGpuContactPairsDev (10 GB+ OOM even at 64 envs). The importer does not
            # enable them already; this line is what turns them off.
            self.scene.robot.spawn.activate_contact_sensors = False
            print(
                f"[ArmResidual __post_init__] activate_contact_sensors = "
                f"{self.scene.robot.spawn.activate_contact_sensors}, {_desk_msg}"
            )

            _residual_env_post_init_physx_lite(self)

            _residual_env_post_init_collisions(self, left_present, max_depenetration_velocity)

    return (TrainArmResidualEnvCfg,)


def _residual_env_post_init_collisions(self, left_present, max_depenetration_velocity):
    """Bimanual physics correctness (bimanual-only; single-hand path is byte-identical — this whole block is gated off
    when bimanual=False). object<->object and hand<->other-object collide by default (no within- env filtering; an
    earlier filter that removed them was a physics- correctness BUG, now reverted). The remaining wanted pair is
    left_hand<->right_hand, but both hands are bodies of the SAME articulation whose self-collision is OFF
    (enabled_self_collisions=False). Blanket-enabling self-collision is HARMFUL: it also switches on every inner-hand
    (finger/palm) and inner-arm (link) self-pair, which the current convention deliberately keeps OFF. The correct
    fix is a SELECTIVE cross-hand-only self-collision filter (enable self-collision then exclude all self-pairs
    except right-hand x left-hand) — enabled. Inner-hand + inner-arm pairs stay OFF (current convention preserved).
    Buffers up-sized for the extra pairs (restored object/hand-cross + the cross-hand self-pairs).
    DEXENGINE_BI_SELFCOLL: self-collision on for bimanual runs. **Default 0 = off** (the same condition as uni):
    self-collision on was harmful (completion 1/20). =1 reproduces the old physics (the same gate as env.py).
    """
    if left_present and env_flags.get("DEXENGINE_BI_SELFCOLL") == "1":
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # Buffer exponent = the same read as the effective block (physx_shift).
        _shift_bi = physx_shift()
        self.sim.physx.gpu_temp_buffer_capacity = 2 ** (26 - _shift_bi)
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** (24 - _shift_bi)
        print(
            "[ArmResidual __post_init__] left_present (bimanual): "
            "enabled_self_collisions=True (+ cross-hand-only FilteredPairs via "
            "BimanualSelfCollisionScene), GPU buffers up-sized "
            f"(temp_buffer=2^{26 - _shift_bi}, "
            f"rigid_patch=2^{24 - _shift_bi}, shift={_shift_bi})."
        )

    # Override object with real USD if provided.
    # Articulated objects (rule D1) are not spawned here: the training entry rl.train.runner.train_cfg_overrides
    # overwrites the object slot wholesale, so the one owner is env_setup.build.build_object_entity_cfg.
    if self.object_usd_path:
        self.scene.object_right = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjectRight",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.object_usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=True,
                    max_linear_velocity=50.0,
                    max_angular_velocity=100.0,
                    max_depenetration_velocity=max_depenetration_velocity,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                    torsional_patch_radius=0.005,
                    min_torsional_patch_radius=0.001,
                ),
            ),
            # Pre-reset placeholder only: reset_object_to_trajectory overrides this
            # to the reference object pose each reset (~0.768 m, on the 0.76 m desk).
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.768),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    _residual_env_post_init_split_artic(self)


def _residual_env_post_init_physx_lite(self):
    """DEXENGINE_PHYSX_LITE step of TrainArmResidualEnvCfg.__post_init__ (statements unchanged)."""
    # DEXENGINE_PHYSX_LITE=1: small buffers for 1-few-env renders and checks. The eval tasks go through this
    # residual cfg, so changing only the arm-IK cfg would not take effect (a render process once asked for a
    # 10 GB mGpuContactPairsDev).
    if env_flags.get("DEXENGINE_PHYSX_LITE") == "1":
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**21
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_max_rigid_contact_count = 2**22
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
    else:
        # Lower the training-scale buffer exponents by k = physx_shift() (2^26 -> 2^(26-k), ...). There are
        # only 6 contact sensors per hand (12 for both), so the unshifted values over-allocate (10 GB); PhysX
        # warns explicitly on overflow, so a measured reduction is safe. Canonical 2 = VRAM 30.9 -> 19.3 GB
        # (32 clips) and 20.4 GB (256 clips), 0 overflows measured.
        _shift = physx_shift()
        self.sim.physx.gpu_found_lost_pairs_capacity = 2 ** (26 - _shift)
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** (26 - _shift)
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** (26 - _shift)
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** (27 - _shift)
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** (23 - _shift)
        # Memory knobs set in the process environment (unset = the
        # values above). DEXENGINE_PHYSX_CONTACT_CAP_LOG2 / DEXENGINE_PHYSX_PATCH_CAP_LOG2 set the contact /
        # patch stream caps as powers of two; past a cap PhysX reports a buffer overflow and drops the extra
        # contacts of that frame, so a cap is only lowered with headroom over the measured peak.
        # DEXENGINE_PHYSX_CAP_AUTO=1 picks the caps from the card size (explicit caps win):
        # < 40 GB -> contact 2^22, patch 2^19; >= 40 GB -> 2^23, 2^20.
        _cap_c = env_flags.get("DEXENGINE_PHYSX_CONTACT_CAP_LOG2")
        _cap_p = env_flags.get("DEXENGINE_PHYSX_PATCH_CAP_LOG2")
        if _cap_c:
            self.sim.physx.gpu_max_rigid_contact_count = 2 ** int(_cap_c)
        if _cap_p:
            self.sim.physx.gpu_max_rigid_patch_count = 2 ** int(_cap_p)
        if _cap_c or _cap_p:
            print(
                f"[PhysX caps] override: contact=2^{int(_cap_c) if _cap_c else 27 - _shift} "
                f"patch=2^{int(_cap_p) if _cap_p else 23 - _shift} (instead of the SHIFT={_shift} values)",
                flush=True,
            )
        if env_flags.get("DEXENGINE_PHYSX_CAP_AUTO") == "1" and not (_cap_c or _cap_p):
            try:
                import torch as _t_cap

                _vram_gb = _t_cap.cuda.get_device_properties(0).total_memory / 2**30
            except Exception:  # noqa: BLE001 — torch raises AssertionError or RuntimeError without a GPU; unknown VRAM reads 0
                _vram_gb = 0.0
            _cc, _pp = (23, 20) if _vram_gb >= 40 else (22, 19)
            self.sim.physx.gpu_max_rigid_contact_count = 2**_cc
            self.sim.physx.gpu_max_rigid_patch_count = 2**_pp
            print(f"[PhysX caps] auto: VRAM={_vram_gb:.0f}G → contact=2^{_cc} patch=2^{_pp}", flush=True)
        if _shift:
            print(
                f"[ArmResidual __post_init__] DEXENGINE_PHYSX_SHIFT={_shift} — "
                f"found_lost/aggregate=2^{26 - _shift} contact=2^{27 - _shift} patch=2^{23 - _shift}"
            )


def _residual_env_post_init_split_artic(self):
    """Split-articulation (bimanual) step of TrainArmResidualEnvCfg.__post_init__ (statements unchanged)."""
    # -----------------------------------------------------------------
    # Split articulation of the combo path: **two articulations, one per arm**
    # (DEXENGINE_BI_SPLIT_ARTIC=1). Unset skips this whole block -> **byte-identical** to the shared
    # articulation path.
    #
    # Why. Every collider of this asset is a USD instance proxy, so **no FilteredPairsAPI resolves onto an
    # articulation collider** (inside an articulation, or proxy rigid body -> link). Only a design without
    # filters remained: with the articulation split in two, the cross hands and arms collide as **ordinary
    # rigid bodies between articulations** (no authoring), and each articulation's
    # enabled_self_collisions=False switches the inside of each hand and arm off.
    #
    # **Two files**, so the left is spawned at a sibling prim (/RobotL); that is why the left contact sensor
    # paths go through _LEFT_NS.
    if _split_artic.bi_enabled():
        if env_flags.get("DEXENGINE_BI_SELFCOLL") == "1":
            raise RuntimeError(
                f"{_split_artic.BI_ENV}=1 and DEXENGINE_BI_SELFCOLL=1 cannot be combined. The split "
                "articulation assumes the self-collision flag permanently OFF (with SELFCOLL=1 the link "
                "FilteredPairs never reach PhysX and the insides of the hands and arms all collide). Mixing "
                "them makes it impossible to tell which physics is running."
            )

        _r_usd, _l_usd = _split_artic.split_usd_paths(self.scene.robot.spawn.usd_path)

        # ⚠ Order: copy **before** trimming. Trimming the right first would leave no left regexes in the
        #    copy and the left view would come out empty.
        robot_left = copy.deepcopy(self.scene.robot)

        # (a) Right articulation = torso + head + right arm + right hand. Only the asset changes; the left
        #     regexes and gain keys are dropped (Isaac Lab fails on a regex that matches nothing).
        self.scene.robot.spawn.usd_path = _r_usd
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        _drop_r = _split_artic.restrict_actuators_to_side(self.scene.robot, "right")
        _split_artic.restrict_init_state_to_side(self.scene.robot, "right")

        # (b) Left articulation = left arm + left hand: the same gains and materials, only the asset and
        #     prim path differ. The two root poses must be **the same value** (split_artic.mirror_root_pose).
        robot_left.prim_path = "{ENV_REGEX_NS}/" + _split_artic.LEFT_PRIM_LEAF
        robot_left.spawn.usd_path = _l_usd
        robot_left.spawn.articulation_props.enabled_self_collisions = False
        robot_left.articulation_root_prim_path = None  # the asset has a single root
        _drop_l = _split_artic.restrict_actuators_to_side(robot_left, "left")
        _split_artic.restrict_init_state_to_side(robot_left, "left")
        if not robot_left.actuators:
            raise RuntimeError(
                "split articulation: no left actuator group left — the side classification does not "
                "match this combo's joint naming "
                f"(dropped groups {_drop_l})."
            )
        self.scene.robot_left = robot_left

        print(
            f"[ArmResidual __post_init__] {_split_artic.BI_ENV}=1: one articulation per arm. "
            f"robot={_r_usd} (torso+head+right arm+right hand, groups {sorted(self.scene.robot.actuators)}"
            + (f"; dropped {_drop_r}" if _drop_r else "")
            + "), "
            f"robot_left={_l_usd} @ {robot_left.prim_path} "
            f"(left arm+left hand, groups {sorted(robot_left.actuators)}"
            + (f"; dropped {_drop_l}" if _drop_l else "")
            + "). "
            "Self-collision OFF on both, no FilteredPairs authoring — cross collision is ordinary rigid "
            "collision between articulations. ⚠ The torso belongs to the R articulation, so right arm/hand "
            "<-> torso stays self-collision (=OFF) (the known asymmetry of the split)."
        )
