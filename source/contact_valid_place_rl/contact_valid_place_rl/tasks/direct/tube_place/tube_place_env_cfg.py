"""Configuration for the tube placement direct RL environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from contact_valid_place_rl.assets import ASSET_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKControllerCfg

ROBOT_USD = ASSET_DIR / "xarm6_with_gripper_contact_refined.usd"
HOLDER_USD = ASSET_DIR / "4_50ml_conical_holder_contact_refined.usd"
VORTEXER_USD = ASSET_DIR / "vortexer_contact_refined.usd"
TUBE_USD = ASSET_DIR / "autobio_50ml_tube_contact_refined.usda"

TABLE_SIZE_M = (1.20, 0.90, 0.04)
TABLE_CENTER_M = (0.35, 0.0, 0.76)
TABLE_TOP_Z_M = TABLE_CENTER_M[2] + 0.5 * TABLE_SIZE_M[2]

ROBOT_ROOT_POS_M = (0.0, 0.0, TABLE_TOP_Z_M + 0.0501)
HOLDER_ROOT_POS_M = (0.43, -0.12, TABLE_TOP_Z_M + 0.004)
VORTEXER_ROOT_POS_M = (0.40, -0.01, TABLE_TOP_Z_M + 0.010)
VORTEXER_CENTER_LOCAL_XY_M = (0.060, 0.035)
VORTEXER_BODY_SIZE_M = (0.130, 0.110, 0.120)
VORTEXER_BODY_CENTER_M = (
    VORTEXER_ROOT_POS_M[0] + VORTEXER_CENTER_LOCAL_XY_M[0],
    VORTEXER_ROOT_POS_M[1] + VORTEXER_CENTER_LOCAL_XY_M[1],
    VORTEXER_ROOT_POS_M[2] + 0.070,
)
ROBOT_SCALE = (1.0, 1.0, 1.0)
HOLDER_SCALE = (0.00114, 0.00114, 0.001)
VORTEXER_SCALE = (0.001, 0.001, 0.001)
TUBE_SCALE = (1.0, 1.0, 1.0)

# The first holder cavity is offset from the holder asset origin. Its collision
# floor is at local z=0 and its XY center is (16.005, -16.205) asset units.
HOLDER_SLOT_0_CENTER_LOCAL = (16.005, -16.205)
TUBE_INITIAL_POS_M = (
    HOLDER_ROOT_POS_M[0] + HOLDER_SLOT_0_CENTER_LOCAL[0] * HOLDER_SCALE[0],
    HOLDER_ROOT_POS_M[1] + HOLDER_SLOT_0_CENTER_LOCAL[1] * HOLDER_SCALE[1],
    HOLDER_ROOT_POS_M[2],
)


@configclass
class TubePlaceEnvCfg(DirectRLEnvCfg):
    """Task configuration for final-stage tube placement into a vortexer.

    This first version is a release-residual task. The policy does not control
    the full xArm or gripper. Instead, it predicts a small correction to a
    nominal release pose: [dx, dy, dz].
    """

    # Environment timing.
    # The physics step is small because the task is contact-rich. The policy acts
    # every ``decimation`` physics steps.
    decimation: int = 2
    # The scripted approach reaches the RL release phase after roughly 7.7 s
    # and the complete physics rollout takes roughly 13.9 s.
    episode_length_s = 22.0

    # Spaces.
    # action = release residual [dx, dy, dz], normalized to [-1, 1].
    action_space = 3

    # obs = placement state(10) + previous action(3) + phase/contact state(5)
    observation_space = 18
    state_space = 0

    # Simulation.
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)

    # Scene.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # Controller.
    ik_controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
    )
    # Assets.
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER_M),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD), scale=ROBOT_SCALE, activate_contact_sensors=True),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=500.0,
                velocity_limit_sim=100.0,
                stiffness=3000.0,
                damping=60.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=5000.0,
                damping=50.0,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_ROOT_POS_M,
            joint_pos={
                "joint1": -0.30,
                "joint2": -0.5,
                "joint3": -1.0,
                "joint4": 0.0,
                "joint5": 1.5,
                "joint6": 0.0,
                "drive_joint": 0.0,
                "left_finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "right_finger_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
            },
        ),
    )

    holder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TubeHolder",
        spawn=sim_utils.UsdFileCfg(usd_path=str(HOLDER_USD), scale=HOLDER_SCALE, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=HOLDER_ROOT_POS_M),
    )

    vortexer: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vortexer",
        spawn=sim_utils.UsdFileCfg(usd_path=str(VORTEXER_USD), scale=VORTEXER_SCALE, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=VORTEXER_ROOT_POS_M),
    )

    # The refined USD disables the imported body collider and only keeps the
    # open-well proxies. This solid base makes the rendered machine impenetrable.
    vortexer_body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VortexerBodyCollision",
        spawn=sim_utils.CuboidCfg(
            size=VORTEXER_BODY_SIZE_M,
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=VORTEXER_BODY_CENTER_M),
    )

    tube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube",
        spawn=sim_utils.UsdFileCfg(usd_path=str(TUBE_USD), scale=TUBE_SCALE, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TUBE_INITIAL_POS_M),
    )
    left_finger_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/left_finger",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["/World/envs/env_.*/Tube"],
    )
    right_finger_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_finger",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["/World/envs/env_.*/Tube"],
    )
    tube_vortexer_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Tube",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Vortexer",
            "/World/envs/env_.*/VortexerBodyCollision",
        ],
    )

    # Demo reset pose.
    # Keep the tube deterministically seated in the holder every episode.
    holder_root_pos_m = HOLDER_ROOT_POS_M
    tube_reset_root_pos_m = TUBE_INITIAL_POS_M

    # Action scaling.
    # Convert normalized actions into meter/radian residuals. Keep these small
    # because the policy only fine-tunes the release pose.
    action_scale_xyz = (0.015, 0.015, 0.020)
    action_filter_alpha = 0.90

    # A hidden controller bias makes the residual policy necessary. It is only
    # enabled for training/evaluation; the nominal scripted demo disables it.
    enable_reset_randomization = True
    release_bias_randomization_m = (0.006, 0.006, 0.004)

    # Task geometry.
    # These describe the target support region on the vortexer, in the vortexer
    # local frame. They will be used by observations, rewards, and success checks.
    vortexer_center_local_xy = VORTEXER_CENTER_LOCAL_XY_M
    vortexer_top_from_root_m = 0.160
    vortexer_floor_from_root_m = 0.130
    vortexer_min_insertion_below_rim_m = 0.008
    vortexer_success_xy_tolerance_m = 0.010
    transfer_tube_xy_tolerance_m = 0.030
    carried_tcp_xy_compensation_m = (-0.006, 0.019)
    release_tcp_xy_correction_m = (0.0, 0.0)

    # Tube / grasp geometry.
    # The generated AutoBio tube USD has local z from 0 to about 0.126 m.
    tube_height_m = 0.126
    tube_initial_pos_m = TUBE_INITIAL_POS_M
    pregrasp_clearance_m = 0.025
    pregrasp_xy_offset_m = (-0.10, 0.03)
    grasp_xy_offset_m = (0.012, 0.010)
    lift_tcp_xy_correction_m = (-0.021, 0.005)
    top_grasp_quat_wxyz = (0.0, 0.988771, -0.149438, 0.0)
    grasp_height_from_tube_top_m = -0.015
    lift_to_grasp_z_offset_m = 0.250
    transfer_to_vortexer_z_offset_m = 0.160
    release_insertion_below_rim_m = 0.030
    retreat_from_release_m = 0.120
    release_clearance_m = 0.0
    pregrasp_align_steps = 90
    pregrasp_descent_steps = 180
    lift_motion_steps = 360
    transfer_motion_steps = 400
    release_align_steps = 240
    release_alignment_dwell_steps = 30
    release_motion_steps = 240
    retreat_motion_steps = 180
    open_dwell_steps = 120
    wrist_yaw_blend_steps = 180
    wrist_yaw_joint1_reference = -0.30

    # Event thresholds for phase transitions.
    ee_reached_threshold_m = 0.008
    pregrasp_reached_threshold_m = 0.025
    loaded_ee_reached_threshold_m = 0.040
    gripper_open_position = 0.0
    gripper_closed_position = 0.85
    gripper_grasp_position_threshold = 0.20
    bilateral_contact_dwell_steps = 30
    gripper_contact_preload = 0.08
    finger_contact_force_threshold_n = 0.05
    vortexer_support_force_threshold_n = 0.02
    tube_floor_tolerance_m = 0.001
    gripper_position_tolerance = 0.15
    gripper_release_position_threshold = 0.15
    grasp_pose_tolerance_m = 0.023
    held_tube_tolerance_m = 0.060
    lift_success_delta_m = 0.05
    tube_stable_lin_vel_threshold_mps = 0.05
    tube_stable_ang_vel_threshold_radps = 0.50
    phase_timeout_steps = 900
    pregrasp_reached_dwell_steps = 8
    grasp_loss_termination_steps = 15
    success_dwell_steps = 180

    # Reward scales.
    rew_xy_tracking = 2.0
    rew_inserted = 4.0
    rew_upright = 0.5
    rew_low_velocity = 1.0
    rew_success = 15.0
    rew_grasp_lost = -10.0
    rew_action_l2 = -0.02
    rew_action_rate_l2 = -0.10
    rew_xy_sigma_m = 0.015
    rew_insertion_xy_sigma_m = 0.010
    rew_lin_vel_sigma_mps = 0.05
    rew_ang_vel_sigma_radps = 0.50

    # Termination.
    fallen_height_threshold = TABLE_TOP_Z_M - 0.05
    max_xy_distance_from_vortexer = 0.20
    upright_success_threshold = 0.95
