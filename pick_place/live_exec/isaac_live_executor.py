#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from isaaclab.app import AppLauncher

from wetlab_benchmark.pick_place.live_exec.protocol import (
    ARM_TRAJECTORY_ACTION,
    CLOCK_TOPIC,
    EXECUTOR_CONTROL_TOPIC,
    EXECUTOR_STATUS_TOPIC,
    GRIPPER_COMMAND_TOPIC,
    JOINT_STATE_TOPIC,
    PHASE_ARM_DONE,
    PHASE_ARM_EXECUTING,
    PHASE_ERROR,
    PHASE_GRASP_ATTACHED,
    PHASE_GRASP_FAILED,
    PHASE_GRIPPER_CLOSING,
    PHASE_GRIPPER_OPEN,
    PHASE_GRIPPER_OPENING,
    PHASE_GRASP_LOST,
    PHASE_READY,
    PHASE_GRASP_SECURED,
    PHASE_LIFT_VERIFIED,
    PHASE_RELEASED,
    PHASE_SETTLED,
    camera_from_preset,
    ExecutorControl,
    ExecutorStatus,
)

parser = argparse.ArgumentParser(description="Live Isaac executor for the wet-lab benchmark.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task_mode", type=str.lower, default="sample", choices=["sample", "a", "b"])
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--trajectory_action", type=str, default=ARM_TRAJECTORY_ACTION)
parser.add_argument("--gripper_topic", type=str, default=GRIPPER_COMMAND_TOPIC)
parser.add_argument("--status_topic", type=str, default=EXECUTOR_STATUS_TOPIC)
parser.add_argument("--control_topic", type=str, default=EXECUTOR_CONTROL_TOPIC)
parser.add_argument("--joint_state_topic", type=str, default=JOINT_STATE_TOPIC)
parser.add_argument("--clock_topic", type=str, default=CLOCK_TOPIC)
parser.add_argument("--camera_preset", type=str, default="front_default")
parser.add_argument("--camera_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--camera_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument(
    "--record_video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Capture RGB frames and encode an mp4. Disable for faster grasp-only diagnostics.",
)
parser.add_argument("--video_fps", type=float, default=8.0)
parser.add_argument("--video_name", type=str, default="render.mp4")
parser.add_argument(
    "--state_trace_path",
    type=str,
    default="",
    help="Optional JSONL path for sampled robot/tube states recorded at the video capture cadence.",
)
parser.add_argument("--idle_exit_s", type=float, default=5.0)
parser.add_argument(
    "--initial_settle_timeout_s",
    type=float,
    default=3.0,
    help="Maximum simulated seconds to settle the initial scene before publishing ready.",
)
parser.add_argument(
    "--initial_settle_pose_window_s",
    type=float,
    default=0.25,
    help="Pose comparison window used while checking initial tube settle.",
)
parser.add_argument("--arm_drive_mode", type=str, choices=["state", "target"], default="target")
parser.add_argument("--arm_time_scale", type=float, default=10.0)
parser.add_argument("--arm_goal_tolerance_rad", type=float, default=0.05)
parser.add_argument("--arm_goal_relaxed_tolerance_rad", type=float, default=0.15)
parser.add_argument("--arm_goal_grace_s", type=float, default=20.0)
parser.add_argument("--physics_dt", type=float, default=0.0025)
parser.add_argument("--grasp_mode", type=str, choices=["physical", "fixed_joint"], default="physical")
parser.add_argument("--asset_profile", type=str, choices=["contact_refined", "imported"], default="contact_refined")
parser.add_argument(
    "--physical_gripper",
    type=str,
    choices=["baked", "imported", "proxy"],
    default="baked",
    help=(
        "Physical-mode gripper contact source. 'baked' uses the refined baked contact "
        "geometry on the real finger links. 'proxy' is deprecated and behaves the same as 'baked'."
    ),
)
parser.add_argument("--contact_force_threshold_n", type=float, default=0.05)
parser.add_argument("--attach_distance_m", type=float, default=0.02)
parser.add_argument("--require_grasp_contact", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--grasp_attach_timeout_s", type=float, default=1.5)
parser.add_argument("--gripper_physical_close_duration_s", type=float, default=2.0)
parser.add_argument("--gripper_open_duration_s", type=float, default=0.5)
parser.add_argument("--gripper_contact_overdrive_rad", type=float, default=0.08)
parser.add_argument("--gripper_contact_overdrive_ramp_s", type=float, default=0.8)
parser.add_argument("--gripper_contact_hold_s", type=float, default=1.0)
parser.add_argument("--gripper_contact_timeout_s", type=float, default=6.0)
parser.add_argument("--grasp_stable_lin_vel_tol_mps", type=float, default=0.012)
parser.add_argument("--grasp_stable_ang_vel_tol_radps", type=float, default=0.25)
parser.add_argument("--grasp_center_tol_m", type=float, default=0.010)
parser.add_argument(
    "--strict_grasp_stability",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Require low tube velocity before reporting grasp_secured. When false, "
        "a centered bilateral contact grasp may proceed and the following lift "
        "verification decides whether the physical grasp actually holds."
    ),
)
parser.add_argument(
    "--gripper_hold_effort",
    type=float,
    default=0.0,
    help="Feed-forward closing effort applied after bilateral contact in physical grasp mode.",
)
parser.add_argument(
    "--write_gripper_state",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Kinematically write gripper joint state while closing. Required for the Isaac 4.5 xArm gripper mimic/tendon import.",
)
parser.add_argument(
    "--gripper_state_write_mode",
    type=str,
    choices=["always", "until_contact", "until_secured", "never"],
    default="until_secured",
    help=(
        "Controls kinematic gripper joint writes when --write_gripper_state is enabled. "
        "'until_secured' uses state writes only to overcome import/mimic closure and seating issues, "
        "then lets the gripper actuator hold the secured grasp physically."
    ),
)
parser.add_argument(
    "--gripper_attached_write_s",
    type=float,
    default=0.0,
    help=(
        "Optional grace window after physical grasp secure during which the frozen "
        "secured gripper joint state is still written to sim. This stabilizes the "
        "imported mimic/tendon chain through the initial breakout lift."
    ),
)
parser.add_argument(
    "--gripper_attached_write_mode",
    type=str,
    choices=["captured", "closed_target"],
    default="captured",
    help=(
        "How to choose the optional post-secure gripper write target. "
        "'captured' freezes the secured joint state, while 'closed_target' "
        "writes the commanded fully-closed target during the grace window."
    ),
)
parser.add_argument("--lift_verify_min_delta_m", type=float, default=0.025)
parser.add_argument(
    "--lift_verify_settle_s",
    type=float,
    default=0.5,
    help="Additional settle window after the first lift trajectory completes before declaring lift verification failure.",
)
parser.add_argument(
    "--lift_verify_allow_single_contact",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Accept a previously secured physical grasp when the tube is lifted high, remains upright, "
        "and one finger still carries strong contact even if the other finger's contact bit drops."
    ),
)
parser.add_argument("--lift_verify_single_contact_min_delta_m", type=float, default=0.05)
parser.add_argument("--lift_verify_single_contact_min_force_n", type=float, default=20.0)
parser.add_argument("--lift_verify_single_contact_max_lin_vel_mps", type=float, default=0.35)
parser.add_argument("--lift_verify_single_contact_max_ang_vel_radps", type=float, default=3.0)
parser.add_argument("--proxy_gripper_open_inner_gap_m", type=float, default=0.075, help=argparse.SUPPRESS)
parser.add_argument("--proxy_gripper_closed_inner_gap_m", type=float, default=0.039, help=argparse.SUPPRESS)
parser.add_argument("--proxy_gripper_overdrive_m", type=float, default=0.0, help=argparse.SUPPRESS)
parser.add_argument("--proxy_gripper_pad_y_size_m", type=float, default=0.018, help=argparse.SUPPRESS)
parser.add_argument("--proxy_gripper_pad_center_tool_m", type=float, nargs=3, default=(0.0, 0.0, -0.037), help=argparse.SUPPRESS)
parser.add_argument("--proxy_gripper_disable_imported_collisions", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
parser.add_argument("--settle_timeout_s", type=float, default=6.0)
parser.add_argument("--settle_xy_tol_m", type=float, default=0.025)
parser.add_argument("--settle_z_tol_m", type=float, default=0.050)
parser.add_argument("--settle_lin_vel_tol_mps", type=float, default=0.05)
parser.add_argument("--settle_ang_vel_tol_radps", type=float, default=0.25)
parser.add_argument("--settle_pose_window_s", type=float, default=0.75)
parser.add_argument("--settle_pose_planar_jitter_m", type=float, default=0.003)
parser.add_argument("--settle_pose_z_jitter_m", type=float, default=0.004)
parser.add_argument("--upright_tol_deg", type=float, default=12.0)
parser.add_argument("--vortexer_settle_min_insertion_below_rim_m", type=float, default=0.008)
parser.add_argument("--seat_release_pose", action=argparse.BooleanOptionalAction, default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.record_video)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from PIL import Image
import omni.usd
import rclpy
import isaaclab.sim as sim_utils
from builtin_interfaces.msg import Time as TimeMsg
from control_msgs.action import FollowJointTrajectory
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdPhysics
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from wetlab_benchmark.pick_place.frames_to_mp4 import encode_png_sequence_to_mp4
from wetlab_benchmark.pick_place.live_exec.task_builder import (
    GRASP_TOOL_CLEARANCE_M,
    LiveTask,
    PoseWxyz,
    TUBE_CENTER_LOCAL_M,
    TUBE_SUPPORT_LOCAL_M,
    TUBE_TOP_LOCAL_M,
    TOP_GRASP_QUAT_WXYZ,
    build_live_task,
    holder_slot_support_pose,
    nominal_tool_pose_from_compensated_grasp,
    vortexer_support_pose,
)
from wetlab_benchmark.pick_place.runtime import (
    ASSET_PROFILE_CONTACT_REFINED,
    ASSET_PROFILE_IMPORTED,
    ContactRefinedPickPlaceSceneCfg,
    PickPlaceSceneCfg,
    SCENE_DOME_LIGHT_COLOR,
    SCENE_DOME_LIGHT_INTENSITY,
    create_simulation,
    initialize_pick_place_runtime,
)
from wetlab_benchmark.task_config import (
    FRAMES,
    GRIPPER_PAD_CONTACT_LOCAL_Y_M,
    GRIPPER_PAD_CONTACT_LOCAL_Z_M,
    IMPORTED_LAB_ASSETS,
    PICK_PLACE_RANDOMIZATION,
)
from wetlab_benchmark.task_objects import RobotAsset
from wetlab_benchmark.validation import _contact_any_for_filters, quat_apply


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_COMMAND_JOINT_NAMES = [FRAMES.left_finger_joint, FRAMES.right_finger_joint]
CONTROLLED_GRIPPER_JOINT_NAMES = [
    "drive_joint",
    FRAMES.left_finger_joint,
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    FRAMES.right_finger_joint,
    "right_inner_knuckle_joint",
]
GRIPPER_OPEN = [FRAMES.gripper_open_pos, FRAMES.gripper_open_pos]
GRIPPER_CLOSED = [FRAMES.gripper_closed_pos, FRAMES.gripper_closed_pos]
CONTROLLED_GRIPPER_OPEN = [FRAMES.gripper_open_pos] * len(CONTROLLED_GRIPPER_JOINT_NAMES)
CONTROLLED_GRIPPER_CLOSED = [FRAMES.gripper_closed_pos] * len(CONTROLLED_GRIPPER_JOINT_NAMES)
GRIPPER_MOVE_DURATION_S = 0.5
ATTACH_JOINT_PATH = "/World/envs/env_0/TubeGraspJoint"
EE_BODY_PRIM_PATH = f"/World/envs/env_0/Robot/{FRAMES.ee_body_name}"
TUBE_PRIM_PATH = "/World/envs/env_0/Tube"
LEFT_GRIPPER_CONTACT_FILTERS = (0, 2, 4)
RIGHT_GRIPPER_CONTACT_FILTERS = (1, 3, 5)
# Child collider contacts on a rigid finger link may be reported under either
# the authored child collider path or the parent rigid-body path in Isaac 4.5.
# Keep the refined child filter first, but also include the parent finger and
# legacy knuckle filters so the physical grasp detector does not miss valid
# contact coming back on the body-level slot.
LEFT_REFINED_GRIPPER_CONTACT_FILTERS = (0, 2, 4, 6, 8, 10, 12, 14, 16)
RIGHT_REFINED_GRIPPER_CONTACT_FILTERS = (1, 3, 5, 7, 9, 11, 13, 15, 17)
LEFT_PAD_LOCAL_M = (0.0, -GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)
RIGHT_PAD_LOCAL_M = (0.0, GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)
TUBE_GRIP_LOCAL_M = (
    TUBE_CENTER_LOCAL_M[0],
    TUBE_CENTER_LOCAL_M[1],
    TUBE_TOP_LOCAL_M[2] + GRASP_TOOL_CLEARANCE_M,
)
STATUS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


_BaseSceneCfg = ContactRefinedPickPlaceSceneCfg if args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED else PickPlaceSceneCfg


@configclass
class SceneCfgWithCamera(_BaseSceneCfg):
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=SCENE_DOME_LIGHT_INTENSITY,
            color=SCENE_DOME_LIGHT_COLOR,
        ),
    )
    obs_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ObsCamera",
        update_period=0,
        update_latest_camera_pose=False,
        height=720,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )


SceneCfg = SceneCfgWithCamera if args_cli.record_video else _BaseSceneCfg


def _pose_tensor(pose: PoseWxyz, *, device: str) -> torch.Tensor:
    return torch.tensor(
        [[pose.x, pose.y, pose.z, pose.qw, pose.qx, pose.qy, pose.qz]],
        device=device,
        dtype=torch.float32,
    )


def _expand_gripper_command(positions: list[float], *, device: str) -> torch.Tensor:
    if len(positions) != len(GRIPPER_COMMAND_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(GRIPPER_COMMAND_JOINT_NAMES)} gripper command positions, got {len(positions)}"
        )
    command_value = float(sum(positions) / len(positions))
    return torch.full(
        (1, len(CONTROLLED_GRIPPER_JOINT_NAMES)),
        command_value,
        device=device,
        dtype=torch.float32,
    )


def _write_rigid_pose(asset, pose: PoseWxyz, *, device: str, zero_velocity: bool = False) -> None:
    asset.write_root_pose_to_sim(_pose_tensor(pose, device=device))
    if zero_velocity:
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_apply_tuple(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    qvec = (0.0, v[0], v[1], v[2])
    rotated = _quat_mul(_quat_mul(q, qvec), _quat_conjugate(q))
    return (rotated[1], rotated[2], rotated[3])


def _pose_from_local_point(
    root_pose_w: PoseWxyz,
    local_xyz_m: tuple[float, float, float],
    local_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> PoseWxyz:
    offset = _quat_apply_tuple(root_pose_w.quat_wxyz, local_xyz_m)
    quat = _quat_mul(root_pose_w.quat_wxyz, local_quat_wxyz)
    return PoseWxyz(
        x=root_pose_w.x + offset[0],
        y=root_pose_w.y + offset[1],
        z=root_pose_w.z + offset[2],
        qw=quat[0],
        qx=quat[1],
        qy=quat[2],
        qz=quat[3],
    )


def _tube_support_pose_from_root(root_pose_w: PoseWxyz) -> PoseWxyz:
    return _pose_from_local_point(root_pose_w, TUBE_SUPPORT_LOCAL_M, TOP_GRASP_QUAT_WXYZ)


def _tube_root_pose_from_tool_grasp(tool_pose_w: PoseWxyz) -> PoseWxyz:
    nominal_tool_pose_w = nominal_tool_pose_from_compensated_grasp(tool_pose_w)
    tube_quat = _quat_mul(nominal_tool_pose_w.quat_wxyz, _quat_conjugate(TOP_GRASP_QUAT_WXYZ))
    tube_top_pos = (
        nominal_tool_pose_w.x,
        nominal_tool_pose_w.y,
        nominal_tool_pose_w.z - GRASP_TOOL_CLEARANCE_M,
    )
    tube_top_offset = _quat_apply_tuple(tube_quat, TUBE_TOP_LOCAL_M)
    return PoseWxyz(
        x=tube_top_pos[0] - tube_top_offset[0],
        y=tube_top_pos[1] - tube_top_offset[1],
        z=tube_top_pos[2] - tube_top_offset[2],
        qw=tube_quat[0],
        qx=tube_quat[1],
        qy=tube_quat[2],
        qz=tube_quat[3],
    )


def _tube_root_pose_from_support_pose(support_pose_w: PoseWxyz) -> PoseWxyz:
    tube_quat = _quat_mul(support_pose_w.quat_wxyz, _quat_conjugate(TOP_GRASP_QUAT_WXYZ))
    support_offset = _quat_apply_tuple(tube_quat, TUBE_SUPPORT_LOCAL_M)
    return PoseWxyz(
        x=support_pose_w.x - support_offset[0],
        y=support_pose_w.y - support_offset[1],
        z=support_pose_w.z - support_offset[2],
        qw=tube_quat[0],
        qx=tube_quat[1],
        qy=tube_quat[2],
        qz=tube_quat[3],
    )


def _pose_to_dict(pose: PoseWxyz) -> dict[str, float]:
    return pose.to_dict()


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _shortest_angular_distance(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = actual - target
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _interpolate_joint_positions_shortest(p0: torch.Tensor, p1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate revolute joints along the shortest angular path."""
    return p0 - _shortest_angular_distance(p0, p1) * alpha


def _unwrap_joint_target_near(reference: torch.Tensor, raw_target: torch.Tensor) -> torch.Tensor:
    """Choose the target angle representation closest to the reference angles."""
    return reference - _shortest_angular_distance(reference, raw_target)


def _unwrap_joint_trajectory_positions(
    initial_positions: torch.Tensor,
    ordered_positions: list[torch.Tensor],
) -> list[torch.Tensor]:
    if not ordered_positions:
        return []
    unwrapped: list[torch.Tensor] = []
    reference = initial_positions
    for raw_target in ordered_positions:
        target = _unwrap_joint_target_near(reference, raw_target)
        unwrapped.append(target)
        reference = target
    return unwrapped


def _midpoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _pose_world_point(pose_w: PoseWxyz, local_xyz_m: tuple[float, float, float]) -> tuple[float, float, float]:
    offset = _quat_apply_tuple(pose_w.quat_wxyz, local_xyz_m)
    return (pose_w.x + offset[0], pose_w.y + offset[1], pose_w.z + offset[2])


def _pose_local_point(pose_w: PoseWxyz, world_xyz_m: tuple[float, float, float]) -> tuple[float, float, float]:
    return _quat_apply_tuple(
        _quat_conjugate(pose_w.quat_wxyz),
        (
            world_xyz_m[0] - pose_w.x,
            world_xyz_m[1] - pose_w.y,
            world_xyz_m[2] - pose_w.z,
        ),
    )


def _pose_from_tensor(pose: torch.Tensor) -> PoseWxyz:
    return PoseWxyz(
        x=float(pose[0].item()),
        y=float(pose[1].item()),
        z=float(pose[2].item()),
        qw=float(pose[3].item()),
        qx=float(pose[4].item()),
        qy=float(pose[5].item()),
        qz=float(pose[6].item()),
    )


def _contact_max_for_filter(sensor, filter_index: int) -> float:
    forces = torch.nan_to_num(sensor.data.force_matrix_w)
    if forces.numel() == 0:
        return 0.0
    mags = torch.linalg.norm(forces, dim=-1)
    if mags.ndim == 0:
        return float(mags.item())
    if mags.ndim == 1:
        return float(torch.max(mags).item())
    if mags.ndim == 2:
        if filter_index >= mags.shape[1]:
            return 0.0
        return float(torch.max(mags[:, filter_index]).item())
    if filter_index >= mags.shape[-1]:
        return 0.0
    values = mags[..., filter_index]
    return float(torch.max(values).item())


def _contact_max_for_filters(sensor, filter_indices: tuple[int, ...]) -> float:
    if not filter_indices:
        return 0.0
    return max(_contact_max_for_filter(sensor, filter_index) for filter_index in filter_indices)


@dataclass
class TrajectoryGoalState:
    goal_handle: object
    joint_names: list[str]
    points: list[JointTrajectoryPoint]
    ordered_positions: list[torch.Tensor]
    initial_positions: torch.Tensor
    start_sim_time: float | None = None
    last_log_bucket: int = -1
    done_event: threading.Event = field(default_factory=threading.Event)
    result_error_code: int = 0
    result_error_string: str = ""


@dataclass
class GripperCommandState:
    target: torch.Tensor
    start_sim_time: float
    end_sim_time: float
    source: torch.Tensor
    mode: str
    leg_label: str | None
    contact_confirmed_at: float | None = None
    hold_target: torch.Tensor | None = None
    hold_source: torch.Tensor | None = None
    hold_start_sim_time: float | None = None
    hold_end_sim_time: float | None = None
    left_contact_seen: bool = False
    right_contact_seen: bool = False


class LiveIsaacExecutor(Node):
    def __init__(self, *, runtime, live_task: LiveTask, camera, frames_dir: Path | None, output_dir: Path) -> None:
        super().__init__("moveit_isaac_live_executor")
        self.runtime = runtime
        self.live_task = live_task
        self.camera = camera
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.video_path = output_dir / args_cli.video_name
        self.metadata_path = output_dir / "render_run.json"
        self.state_trace_path = Path(args_cli.state_trace_path).expanduser().resolve() if args_cli.state_trace_path else None
        self._state_trace_file = None
        self.status_history: list[dict[str, object]] = []
        self.summary_status: str | None = None

        arm_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
        arm_entity_cfg.resolve(runtime.scene)
        gripper_entity_cfg = SceneEntityCfg("robot", joint_names=CONTROLLED_GRIPPER_JOINT_NAMES)
        gripper_entity_cfg.resolve(runtime.scene)
        left_finger_cfg = SceneEntityCfg("robot", body_names=["left_finger"])
        left_finger_cfg.resolve(runtime.scene)
        right_finger_cfg = SceneEntityCfg("robot", body_names=["right_finger"])
        right_finger_cfg.resolve(runtime.scene)
        self.arm_joint_ids = arm_entity_cfg.joint_ids
        self.gripper_joint_ids = gripper_entity_cfg.joint_ids
        self.left_finger_body_id = left_finger_cfg.body_ids[0]
        self.right_finger_body_id = right_finger_cfg.body_ids[0]

        self.ee_tool_asset = RobotAsset(
            runtime.robot,
            ee_body_id=runtime.robot_entity_cfg.body_ids[0],
            ee_local_offset_m=FRAMES.ee_tool_offset_local_m,
        )
        self.ee_body_asset = RobotAsset(
            runtime.robot,
            ee_body_id=runtime.robot_entity_cfg.body_ids[0],
            ee_local_offset_m=(0.0, 0.0, 0.0),
        )

        self.arm_targets = runtime.robot.data.joint_pos[:, self.arm_joint_ids].clone()
        self.gripper_targets = torch.tensor([CONTROLLED_GRIPPER_OPEN], device=runtime.device, dtype=torch.float32)
        self.zero_arm_vel = torch.zeros_like(self.arm_targets)
        self.zero_gripper_vel = torch.zeros_like(self.gripper_targets)
        self.gripper_efforts = torch.zeros_like(self.gripper_targets)

        self._lock = threading.Lock()
        self._active_goal: TrajectoryGoalState | None = None
        self._gripper_command: GripperCommandState | None = None
        self._gripper_attached_write_target: torch.Tensor | None = None
        self._attached_since_sim_time: float | None = None
        self._attached = False
        self._release_active = False
        self._release_started_at = 0.0
        self._settle_history: list[tuple[float, float, float, float, float]] = []
        self._attach_deadline = 0.0
        self._last_leg_label: str | None = None
        self._last_stage: str | None = None
        self._grasped_tube_support_z: float | None = None
        self._lift_verify_started_at: float | None = None
        self._lift_verification_reported = False
        self._activity_started = False
        self._last_activity_sim_time = 0.0
        self._last_capture_step = -1
        self._last_lift_trace_sim_time = -math.inf
        self._lift_trace: list[dict[str, object]] = []
        self._frame_index = 0
        self._current_sim_time = float(runtime.sim.current_time)
        self._cached_arm_joint_positions = self.arm_targets.clone()
        self._cached_gripper_joint_positions = self.gripper_targets.clone()
        self._cached_tube_root_pose = live_task.initial_tube_root_w
        self._leg_destinations = {
            leg.label: (leg.dest_name, leg.dest_support_w)
            for leg in live_task.legs
        }
        if args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED:
            self.physical_gripper_mode = "baked"
        else:
            self.physical_gripper_mode = "imported"
        if args_cli.grasp_mode == "physical" and args_cli.physical_gripper == "proxy":
            self.get_logger().warning(
                "physical_gripper=proxy is deprecated; using baked finger-link contact geometry instead"
            )
        elif (
            args_cli.grasp_mode == "physical"
            and args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED
            and args_cli.physical_gripper != "baked"
        ):
            self.get_logger().warning(
                f"physical_gripper={args_cli.physical_gripper} is ignored for contact_refined assets; using baked finger-link contact geometry"
            )

        if args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED:
            self.left_gripper_contact_filters = LEFT_REFINED_GRIPPER_CONTACT_FILTERS
            self.right_gripper_contact_filters = RIGHT_REFINED_GRIPPER_CONTACT_FILTERS
        else:
            self.left_gripper_contact_filters = LEFT_GRIPPER_CONTACT_FILTERS
            self.right_gripper_contact_filters = RIGHT_GRIPPER_CONTACT_FILTERS

        self.status_pub = self.create_publisher(String, args_cli.status_topic, STATUS_QOS)
        self.joint_state_pub = self.create_publisher(JointState, args_cli.joint_state_topic, 20)
        self.clock_pub = self.create_publisher(Clock, args_cli.clock_topic, 20)
        self.create_subscription(JointTrajectory, args_cli.gripper_topic, self._on_gripper_command, 20)
        self.create_subscription(String, args_cli.control_topic, self._on_control, 20)
        self.create_timer(1.0, self._publish_ready_heartbeat)
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            args_cli.trajectory_action,
            execute_callback=self._execute_trajectory_goal,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )

        self.stage = omni.usd.get_context().get_stage()
        self.capture_stride = max(1, int(round((1.0 / max(args_cli.video_fps, 1.0)) / runtime.sim.get_physics_dt())))
        self.upright_cos_min = math.cos(math.radians(args_cli.upright_tol_deg))
        if self.state_trace_path is not None:
            self.state_trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_trace_file = self.state_trace_path.open("w", encoding="utf-8")

    def _step_sim_only(self) -> None:
        self.runtime.robot.set_joint_position_target(self.arm_targets, joint_ids=self.arm_joint_ids)
        self.runtime.robot.set_joint_position_target(self.gripper_targets, joint_ids=self.gripper_joint_ids)
        if args_cli.write_gripper_state and args_cli.gripper_state_write_mode != "never":
            self.runtime.robot.write_joint_state_to_sim(
                self.gripper_targets,
                self.zero_gripper_vel,
                joint_ids=self.gripper_joint_ids,
            )
        self.runtime.scene.write_data_to_sim()
        self.runtime.sim.step()
        self.runtime.scene.update(self.runtime.sim.get_physics_dt())
        self._cached_arm_joint_positions = self.runtime.robot.data.joint_pos[:, self.arm_joint_ids].clone()
        self._cached_gripper_joint_positions = self.runtime.robot.data.joint_pos[:, self.gripper_joint_ids].clone()
        self._cached_tube_root_pose = self.current_tube_root_pose()

    def settle_initial_scene(self) -> None:
        max_steps = max(1, int(round(args_cli.initial_settle_timeout_s / self.runtime.sim.get_physics_dt())))
        compare_window_steps = max(1, int(round(args_cli.initial_settle_pose_window_s / self.runtime.sim.get_physics_dt())))
        ref_pose = self.current_tube_root_pose()
        stable_windows = 0
        for step in range(max_steps):
            self._step_sim_only()
            if (step + 1) % compare_window_steps != 0:
                continue
            current = self.current_tube_root_pose()
            planar_delta = math.hypot(current.x - ref_pose.x, current.y - ref_pose.y)
            z_delta = abs(current.z - ref_pose.z)
            lin_speed = float(torch.linalg.norm(self.runtime.tube.data.root_lin_vel_w[0]).item())
            ang_speed = float(torch.linalg.norm(self.runtime.tube.data.root_ang_vel_w[0]).item())
            settled = (
                planar_delta <= args_cli.settle_pose_planar_jitter_m
                and z_delta <= args_cli.settle_pose_z_jitter_m
                and lin_speed <= args_cli.settle_lin_vel_tol_mps
                and ang_speed <= args_cli.settle_ang_vel_tol_radps
            )
            if settled:
                stable_windows += 1
                if stable_windows >= 2:
                    break
            else:
                stable_windows = 0
            ref_pose = current

    def _publish_status(self, status: ExecutorStatus) -> None:
        if status.leg_label is None and self._last_leg_label is not None:
            status = ExecutorStatus(
                phase=status.phase,
                ok=status.ok,
                attached=status.attached,
                tube_pose=status.tube_pose,
                reason=status.reason,
                leg_label=self._last_leg_label,
                stamp=status.stamp,
                extra=status.extra,
            )
        self.status_history.append(status.to_dict())
        msg = String()
        msg.data = status.to_json()
        self.status_pub.publish(msg)
        if not status.ok:
            self.summary_status = "failure"

    def _on_control(self, msg: String) -> None:
        try:
            control = ExecutorControl.from_json(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"Ignoring malformed control message: {exc}")
            return
        with self._lock:
            self._last_leg_label = control.leg_label
            self._last_stage = control.stage
            self._lift_verify_started_at = None
            self._lift_verification_reported = False

    def _publish_ready_heartbeat(self) -> None:
        if self._activity_started or self.summary_status is not None:
            return
        self._publish_status(
            ExecutorStatus(
                phase=PHASE_READY,
                ok=True,
                attached=self._attached,
                tube_pose=_pose_to_dict(self._cached_tube_root_pose),
            )
        )

    def _goal_callback(self, goal_request) -> GoalResponse:
        with self._lock:
            if self._active_goal is not None:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_trajectory_goal(self, goal_handle):
        with self._lock:
            current = self._cached_arm_joint_positions.clone()
            self._lift_verify_started_at = None
            self._lift_verification_reported = False
            ordered_positions = [
                self._reorder_joint_positions(goal_handle.request.trajectory.joint_names, point.positions)
                for point in goal_handle.request.trajectory.points
            ]
            self._active_goal = TrajectoryGoalState(
                goal_handle=goal_handle,
                joint_names=list(goal_handle.request.trajectory.joint_names),
                points=list(goal_handle.request.trajectory.points),
                ordered_positions=_unwrap_joint_trajectory_positions(current, ordered_positions),
                initial_positions=current,
            )
            active_goal = self._active_goal
            self._activity_started = True
        final_time = _point_time(active_goal.points[-1]) * max(args_cli.arm_time_scale, 1.0) if active_goal.points else 0.0
        self.get_logger().info(
            f"Accepted arm trajectory: points={len(active_goal.points)} final_time_s={final_time:.2f}"
        )
        active_goal.done_event.wait()
        result = FollowJointTrajectory.Result()
        result.error_code = int(active_goal.result_error_code)
        result.error_string = active_goal.result_error_string
        if result.error_code == 0:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        with self._lock:
            if self._active_goal is active_goal:
                self._active_goal = None
        return result

    def _on_gripper_command(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        positions = msg.points[-1].positions
        if len(positions) != len(GRIPPER_COMMAND_JOINT_NAMES):
            return
        target = _expand_gripper_command(list(positions), device=self.runtime.device)
        now = self._current_sim_time
        current = self._cached_gripper_joint_positions.clone()
        mode = "open" if float(sum(positions)) <= float(sum(GRIPPER_OPEN)) + 1.0e-6 else "close"
        duration = (
            max(args_cli.gripper_physical_close_duration_s, 0.05)
            if mode == "close" and args_cli.grasp_mode == "physical"
            else max(args_cli.gripper_open_duration_s, 0.05)
        )
        with self._lock:
            self._gripper_command = GripperCommandState(
                target=target,
                start_sim_time=now,
                end_sim_time=now + duration,
                source=current,
                mode=mode,
                leg_label=self._last_leg_label,
            )
            self._gripper_attached_write_target = None
            self._activity_started = True
            self._last_activity_sim_time = now
            if mode == "open":
                self._publish_status(ExecutorStatus(phase=PHASE_GRIPPER_OPENING, ok=True, attached=self._attached))
                if self._attached:
                    seated_root_pose = None
                    if args_cli.grasp_mode == "fixed_joint":
                        self._remove_fixed_attach_joint()
                    self._attached = False
                    self._attached_since_sim_time = None
                    self._gripper_attached_write_target = None
                    self._grasped_tube_support_z = None
                    if args_cli.grasp_mode == "fixed_joint" and args_cli.seat_release_pose:
                        _, target_support = self._candidate_support_pose()
                        seated_root_pose = _tube_root_pose_from_support_pose(target_support)
                        _write_rigid_pose(self.runtime.tube, seated_root_pose, device=self.runtime.device, zero_velocity=True)
                        self.runtime.scene.write_data_to_sim()
                        self.runtime.scene.update(0.0)
                        self._cached_tube_root_pose = seated_root_pose
                    self._release_active = True
                    self._release_started_at = now
                    self._settle_history.clear()
                    self._publish_status(
                        ExecutorStatus(
                            phase=PHASE_RELEASED,
                            ok=True,
                            attached=False,
                            tube_pose=_pose_to_dict(seated_root_pose or self.current_tube_root_pose()),
                            leg_label=self._last_leg_label,
                        )
                    )
            else:
                if args_cli.grasp_mode == "physical":
                    self._attach_deadline = now + max(
                        args_cli.gripper_contact_timeout_s,
                        duration + args_cli.gripper_contact_hold_s,
                    )
                else:
                    self._attach_deadline = now + args_cli.grasp_attach_timeout_s
                self._publish_status(ExecutorStatus(phase=PHASE_GRIPPER_CLOSING, ok=True, attached=self._attached))

    def current_tube_root_pose(self) -> PoseWxyz:
        pose = self.runtime.tube.data.root_pose_w[0]
        return PoseWxyz(
            x=float(pose[0].item()),
            y=float(pose[1].item()),
            z=float(pose[2].item()),
            qw=float(pose[3].item()),
            qx=float(pose[4].item()),
            qy=float(pose[5].item()),
            qz=float(pose[6].item()),
        )

    def current_tool_pose(self) -> PoseWxyz:
        pose = self.ee_tool_asset.ee_pose_w[0]
        return PoseWxyz(
            x=float(pose[0].item()),
            y=float(pose[1].item()),
            z=float(pose[2].item()),
            qw=float(pose[3].item()),
            qx=float(pose[4].item()),
            qy=float(pose[5].item()),
            qz=float(pose[6].item()),
        )

    def _publish_joint_state(self, sim_time: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        joint_pos = self.runtime.robot.data.joint_pos[0]
        joint_vel = self.runtime.robot.data.joint_vel[0]
        ids = list(self.arm_joint_ids)
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = [float(joint_pos[idx].item()) for idx in ids]
        msg.velocity = [float(joint_vel[idx].item()) for idx in ids]
        self.joint_state_pub.publish(msg)

    def _publish_clock(self, sim_time: float) -> None:
        msg = Clock()
        msg.clock = self._time_msg(sim_time)
        self.clock_pub.publish(msg)

    def _record_state_trace(self, sim_time: float, step_index: int) -> None:
        if self._state_trace_file is None:
            return
        sample = {
            "frame_index": self._frame_index,
            "step_index": step_index,
            "sim_time_s": sim_time,
            "arm_joint_pos": [float(value.item()) for value in self._cached_arm_joint_positions[0]],
            "gripper_joint_pos": [float(value.item()) for value in self._cached_gripper_joint_positions[0]],
            "tube_root_pose": _pose_to_dict(self._cached_tube_root_pose),
        }
        self._state_trace_file.write(json.dumps(sample) + "\n")

    @staticmethod
    def _time_msg(sim_time: float) -> TimeMsg:
        secs = int(sim_time)
        return TimeMsg(sec=secs, nanosec=int((sim_time - secs) * 1e9))

    def _apply_arm_goal(self, sim_time: float) -> None:
        goal = self._active_goal
        if goal is None:
            return
        if goal.start_sim_time is None:
            goal.start_sim_time = sim_time
            self._last_activity_sim_time = sim_time
            self._publish_status(ExecutorStatus(phase=PHASE_ARM_EXECUTING, ok=True, attached=self._attached))

        if not goal.points:
            goal.result_error_code = -1
            goal.result_error_string = "Empty arm trajectory"
            goal.done_event.set()
            return

        elapsed = max(sim_time - goal.start_sim_time, 0.0)
        sampled = self._sample_arm_positions(goal, elapsed, max(args_cli.arm_time_scale, 1.0))
        self.arm_targets[:, :] = sampled
        self._last_activity_sim_time = sim_time

    def _check_arm_goal_completion(self, sim_time: float) -> None:
        goal = self._active_goal
        if goal is None or goal.start_sim_time is None or not goal.points:
            return
        if goal.done_event.is_set():
            return

        elapsed = max(sim_time - goal.start_sim_time, 0.0)
        final_positions = goal.ordered_positions[-1]
        final_time = _point_time(goal.points[-1]) * max(args_cli.arm_time_scale, 1.0)
        actual = self._cached_arm_joint_positions
        max_error = float(torch.max(torch.abs(_shortest_angular_distance(actual, final_positions))).item())

        log_bucket = int(elapsed)
        if log_bucket != goal.last_log_bucket:
            goal.last_log_bucket = log_bucket
            self.get_logger().info(
                f"Arm tracking: elapsed_s={elapsed:.2f} final_s={final_time:.2f} max_err_rad={max_error:.4f}"
            )

        if elapsed < final_time:
            return

        if max_error <= args_cli.arm_goal_tolerance_rad:
            goal.result_error_code = 0
            goal.result_error_string = ""
            goal.done_event.set()
            self._publish_status(ExecutorStatus(phase=PHASE_ARM_DONE, ok=True, attached=self._attached))
            self._publish_lift_verification_if_needed(sim_time)
            return

        grace_s = max(args_cli.arm_goal_grace_s, 0.25 * max(final_time, 1.0))
        if elapsed < final_time + grace_s:
            self.arm_targets[:, :] = final_positions
            return

        if max_error <= args_cli.arm_goal_relaxed_tolerance_rad:
            self.get_logger().warning(
                f"Arm goal completed with relaxed tolerance: max_err_rad={max_error:.4f}"
            )
            goal.result_error_code = 0
            goal.result_error_string = (
                f"completed with relaxed tolerance max_err_rad={max_error:.4f}"
            )
            goal.done_event.set()
            self._publish_status(ExecutorStatus(phase=PHASE_ARM_DONE, ok=True, attached=self._attached))
            self._publish_lift_verification_if_needed(sim_time)
            return

        per_joint_error = torch.abs(_shortest_angular_distance(actual, final_positions))[0]
        joint_error_report = ", ".join(
            f"{joint_name}={float(error.item()):.4f}"
            for joint_name, error in zip(ARM_JOINT_NAMES, per_joint_error)
        )
        self.get_logger().error(
            f"Arm tracking timeout joint errors: {joint_error_report}"
        )

        goal.result_error_code = -4
        goal.result_error_string = (
            f"arm tracking timeout: final_s={final_time:.2f} elapsed_s={elapsed:.2f} max_err_rad={max_error:.4f}"
        )
        goal.done_event.set()
        self._publish_status(
            ExecutorStatus(
                phase=PHASE_ERROR,
                ok=False,
                attached=self._attached,
                leg_label=self._last_leg_label,
                tube_pose=_pose_to_dict(self._cached_tube_root_pose),
                reason=goal.result_error_string,
            )
        )

    def _sample_arm_positions(self, goal: TrajectoryGoalState, elapsed: float, time_scale: float) -> torch.Tensor:
        points = goal.points
        first_time = _point_time(points[0]) * time_scale
        if elapsed <= first_time:
            alpha = 0.0 if first_time <= 1.0e-6 else elapsed / first_time
            first = goal.ordered_positions[0]
            return _interpolate_joint_positions_shortest(goal.initial_positions, first, alpha)

        for index in range(len(points) - 1):
            t0 = _point_time(points[index]) * time_scale
            t1 = _point_time(points[index + 1]) * time_scale
            if elapsed <= t1:
                p0 = goal.ordered_positions[index]
                p1 = goal.ordered_positions[index + 1]
                alpha = 0.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
                return _interpolate_joint_positions_shortest(p0, p1, alpha)
        return goal.ordered_positions[-1]

    def _reorder_joint_positions(self, joint_names: list[str], positions: list[float]) -> torch.Tensor:
        tensor = torch.zeros((1, len(ARM_JOINT_NAMES)), device=self.runtime.device, dtype=torch.float32)
        for target_index, joint_name in enumerate(ARM_JOINT_NAMES):
            try:
                source_index = joint_names.index(joint_name)
            except ValueError as exc:
                raise RuntimeError(f"Trajectory missing joint {joint_name}") from exc
            tensor[0, target_index] = float(positions[source_index])
        return tensor

    def _apply_gripper_goal(self, sim_time: float) -> None:
        cmd = self._gripper_command
        if cmd is None:
            if self._attached and args_cli.grasp_mode == "physical":
                if self._gripper_attached_write_target is not None:
                    self.gripper_targets[:, :] = self._gripper_attached_write_target
                self.gripper_efforts.zero_()
                self.gripper_efforts[:, 0] = max(args_cli.gripper_hold_effort, 0.0)
            else:
                self.gripper_efforts.zero_()
            return
        self.gripper_efforts.zero_()
        if cmd.mode == "close" and args_cli.grasp_mode == "physical":
            if cmd.hold_target is not None:
                if cmd.hold_source is None or cmd.hold_start_sim_time is None or cmd.hold_end_sim_time is None:
                    self.gripper_targets[:, :] = cmd.hold_target
                    return
                alpha = min(
                    max(
                        (sim_time - cmd.hold_start_sim_time)
                        / max(cmd.hold_end_sim_time - cmd.hold_start_sim_time, 1.0e-6),
                        0.0,
                    ),
                    1.0,
                )
                self.gripper_targets[:, :] = cmd.hold_source * (1.0 - alpha) + cmd.hold_target * alpha
                return
            alpha = min(
                max((sim_time - cmd.start_sim_time) / max(cmd.end_sim_time - cmd.start_sim_time, 1.0e-6), 0.0),
                1.0,
            )
            self.gripper_targets[:, :] = cmd.source * (1.0 - alpha) + cmd.target * alpha
            return
        if sim_time >= cmd.end_sim_time:
            self.gripper_targets[:, :] = cmd.target
            if cmd.mode == "open":
                self.gripper_efforts.zero_()
                self._publish_status(ExecutorStatus(phase=PHASE_GRIPPER_OPEN, ok=True, attached=self._attached, leg_label=cmd.leg_label))
                self._gripper_command = None
            return
        alpha = (sim_time - cmd.start_sim_time) / max(cmd.end_sim_time - cmd.start_sim_time, 1.0e-6)
        self.gripper_targets[:, :] = cmd.source * (1.0 - alpha) + cmd.target * alpha

    def _finger_contacts(self) -> tuple[bool, bool]:
        left = bool(
            _contact_any_for_filters(self.runtime.tube_contacts, self.left_gripper_contact_filters, args_cli.contact_force_threshold_n)[0].item()
        )
        right = bool(
            _contact_any_for_filters(self.runtime.tube_contacts, self.right_gripper_contact_filters, args_cli.contact_force_threshold_n)[0].item()
        )
        return left, right

    def _finger_contact_details(self) -> dict[str, object]:
        left_contact, right_contact = self._finger_contacts()
        left_force = _contact_max_for_filters(self.runtime.tube_contacts, self.left_gripper_contact_filters)
        right_force = _contact_max_for_filters(self.runtime.tube_contacts, self.right_gripper_contact_filters)
        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_force_n": left_force,
            "right_force_n": right_force,
            "max_finger_force_n": max(left_force, right_force),
        }

    def _body_local_point_w(self, body_id: int, local_xyz_m: tuple[float, float, float]) -> tuple[float, float, float]:
        body_pose = _pose_from_tensor(self.runtime.robot.data.body_pose_w[0, body_id])
        return _pose_world_point(body_pose, local_xyz_m)

    def _grasp_geometry_extra(self) -> dict[str, object]:
        tube_root = self.current_tube_root_pose()
        tube_grip_center_w = _pose_world_point(tube_root, TUBE_GRIP_LOCAL_M)
        left_pad_w = self._body_local_point_w(self.left_finger_body_id, LEFT_PAD_LOCAL_M)
        right_pad_w = self._body_local_point_w(self.right_finger_body_id, RIGHT_PAD_LOCAL_M)
        pad_mid_w = _midpoint(left_pad_w, right_pad_w)
        tube_to_pad_mid_planar = math.hypot(
            tube_grip_center_w[0] - pad_mid_w[0],
            tube_grip_center_w[1] - pad_mid_w[1],
        )
        support = _tube_support_pose_from_root(tube_root)
        contact = self._finger_contact_details()
        up_axis = quat_apply(
            torch.tensor([[tube_root.qw, tube_root.qx, tube_root.qy, tube_root.qz]], device=self.runtime.device, dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device=self.runtime.device, dtype=torch.float32),
        )[0]
        return {
            **contact,
            "tube_root_pose": _pose_to_dict(tube_root),
            "tool_pose": _pose_to_dict(self.current_tool_pose()),
            "tube_support_pose": _pose_to_dict(support),
            "tube_grip_center_w": list(tube_grip_center_w),
            "left_pad_w": list(left_pad_w),
            "right_pad_w": list(right_pad_w),
            "pad_mid_w": list(pad_mid_w),
            "pad_gap_m": _distance(left_pad_w, right_pad_w),
            "physical_gripper_mode": self.physical_gripper_mode,
            "tube_to_pad_mid_m": _distance(tube_grip_center_w, pad_mid_w),
            "tube_to_pad_mid_planar_m": tube_to_pad_mid_planar,
            "tube_to_left_pad_m": _distance(tube_grip_center_w, left_pad_w),
            "tube_to_right_pad_m": _distance(tube_grip_center_w, right_pad_w),
            "tube_minus_pad_mid_w": [tube_grip_center_w[index] - pad_mid_w[index] for index in range(3)],
            "tube_up_axis_z": float(up_axis[2].item()),
            "tube_lin_speed_mps": float(torch.linalg.norm(self.runtime.tube.data.root_lin_vel_w[0]).item()),
            "tube_ang_speed_radps": float(torch.linalg.norm(self.runtime.tube.data.root_ang_vel_w[0]).item()),
            "gripper_joint_pos": [
                float(value.item()) for value in self.runtime.robot.data.joint_pos[0, self.gripper_joint_ids]
            ],
            "gripper_target": [
                float(value.item()) for value in self.gripper_targets[0]
            ],
        }

    def _record_lift_trace(self, sim_time: float) -> None:
        if args_cli.grasp_mode != "physical":
            return
        if self._last_stage is None or not self._last_stage.endswith(":lift"):
            return
        if not self._attached:
            return
        if sim_time - self._last_lift_trace_sim_time < 0.10:
            return
        self._last_lift_trace_sim_time = sim_time
        sample = {
            "sim_time_s": sim_time,
            **self._grasp_geometry_extra(),
        }
        self._lift_trace.append(sample)
        if len(self._lift_trace) > 120:
            self._lift_trace = self._lift_trace[-120:]

    def _check_grasp_state(self, sim_time: float) -> None:
        if args_cli.grasp_mode == "physical":
            self._check_physical_grasp(sim_time)
        else:
            self._check_for_fixed_joint_attach(sim_time)

    def _check_physical_grasp(self, sim_time: float) -> None:
        cmd = self._gripper_command
        if self._attached or cmd is None or cmd.mode != "close":
            return

        left_contact, right_contact = self._finger_contacts()
        bilateral_contact = left_contact and right_contact
        if bilateral_contact and cmd.contact_confirmed_at is None:
            hold_value = min(
                max(
                    float(cmd.target[0, 0].item()),
                    float(self.gripper_targets[0, 0].item()) + max(args_cli.gripper_contact_overdrive_rad, 0.0),
                ),
                FRAMES.gripper_closed_pos,
            )
            cmd.hold_target = torch.full_like(self.gripper_targets, hold_value)
            cmd.hold_source = self.gripper_targets.clone()
            cmd.hold_start_sim_time = sim_time
            cmd.hold_end_sim_time = sim_time + max(args_cli.gripper_contact_overdrive_ramp_s, 0.0)
            cmd.contact_confirmed_at = sim_time
            self.gripper_targets[:, :] = cmd.hold_target
            self.get_logger().info(
                f"{cmd.leg_label or 'grasp'}: bilateral tube contact observed; ramping compliant overdrive"
            )
            return

        if cmd.contact_confirmed_at is not None:
            hold_done_at = cmd.hold_end_sim_time if cmd.hold_end_sim_time is not None else cmd.contact_confirmed_at
            if sim_time - hold_done_at < max(args_cli.gripper_contact_hold_s, 0.0):
                return
            if bilateral_contact:
                current = self.current_tube_root_pose()
                support = _tube_support_pose_from_root(current)
                grasp_extra = self._grasp_geometry_extra()
                velocity_stable = (
                    float(grasp_extra["tube_lin_speed_mps"]) <= args_cli.grasp_stable_lin_vel_tol_mps
                    and float(grasp_extra["tube_ang_speed_radps"]) <= args_cli.grasp_stable_ang_vel_tol_radps
                    and float(grasp_extra["tube_up_axis_z"]) >= self.upright_cos_min
                )
                # For the vertical cap/shoulder grasp, the tube grip center can
                # sit materially above the fingertip contact-band midpoint in
                # world Z while still being well centered laterally between the
                # pads. Use planar XY centering here and let the later lift
                # verification decide whether the bilateral grasp actually
                # carries the tube.
                centered_contact = (
                    float(grasp_extra["tube_to_pad_mid_planar_m"]) <= args_cli.grasp_center_tol_m
                    and float(grasp_extra["tube_up_axis_z"]) >= self.upright_cos_min
                )
                stable = velocity_stable or (centered_contact and not args_cli.strict_grasp_stability)
                if not stable and sim_time < self._attach_deadline:
                    return
                if not stable:
                    self._publish_status(
                        ExecutorStatus(
                            phase=PHASE_GRASP_FAILED,
                            ok=False,
                            attached=False,
                            tube_pose=_pose_to_dict(current),
                            leg_label=cmd.leg_label,
                            reason="physical grasp did not stabilize before timeout",
                            extra=grasp_extra,
                        )
                    )
                    self._gripper_command = None
                    return
                self._attached = True
                self._attached_since_sim_time = sim_time
                if args_cli.gripper_attached_write_mode == "closed_target":
                    attached_target = cmd.hold_target if cmd.hold_target is not None else cmd.target
                    self._gripper_attached_write_target = torch.maximum(
                        self._cached_gripper_joint_positions,
                        attached_target.clone(),
                    )
                else:
                    self._gripper_attached_write_target = self._cached_gripper_joint_positions.clone()
                self._grasped_tube_support_z = support.z
                self._publish_status(
                    ExecutorStatus(
                        phase=PHASE_GRASP_SECURED,
                        ok=True,
                        attached=True,
                        tube_pose=_pose_to_dict(current),
                        leg_label=cmd.leg_label,
                        extra={
                            **grasp_extra,
                            "centered_contact": centered_contact,
                            "velocity_stable": velocity_stable,
                            "support_z_m": support.z,
                        },
                    )
                )
                self._last_activity_sim_time = sim_time
                self._gripper_command = None
                return
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_GRASP_FAILED,
                    ok=False,
                    attached=False,
                    tube_pose=_pose_to_dict(self.current_tube_root_pose()),
                    leg_label=cmd.leg_label,
                    reason="physical grasp contact was lost during hold",
                    extra=self._grasp_geometry_extra(),
                )
            )
            self._gripper_command = None
            return

        if sim_time >= self._attach_deadline:
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_GRASP_FAILED,
                    ok=False,
                    attached=False,
                    tube_pose=_pose_to_dict(self.current_tube_root_pose()),
                    leg_label=cmd.leg_label,
                    reason=(
                        "physical grasp validation failed before timeout: "
                        f"left_contact={left_contact} right_contact={right_contact}"
                    ),
                    extra=self._grasp_geometry_extra(),
                )
            )
            self._gripper_command = None

    def _check_for_fixed_joint_attach(self, sim_time: float) -> None:
        cmd = self._gripper_command
        if self._attached or cmd is None or cmd.mode != "close" or sim_time < cmd.end_sim_time:
            return

        ee_pose_w = self.ee_tool_asset.ee_pose_w[0]
        tube_grasp_pose_w = self.runtime.tube_asset.get_grasp_pose_w()[0]
        grasp_error = torch.linalg.norm(ee_pose_w[:3] - tube_grasp_pose_w[:3]).item()
        left_contact, right_contact = self._finger_contacts()
        contact = left_contact or right_contact
        contact_ok = (left_contact and right_contact) or (contact and not args_cli.require_grasp_contact)
        if contact_ok and grasp_error <= args_cli.attach_distance_m:
            snapped_root_pose = _tube_root_pose_from_tool_grasp(self.current_tool_pose())
            _write_rigid_pose(self.runtime.tube, snapped_root_pose, device=self.runtime.device, zero_velocity=True)
            self.runtime.scene.write_data_to_sim()
            self.runtime.scene.update(0.0)
            self._cached_tube_root_pose = snapped_root_pose
            self._create_fixed_attach_joint()
            self._attached = True
            self._attached_since_sim_time = sim_time
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_GRASP_ATTACHED,
                    ok=True,
                    attached=True,
                    tube_pose=_pose_to_dict(snapped_root_pose),
                    leg_label=cmd.leg_label,
                )
            )
            self._last_activity_sim_time = sim_time
            self._gripper_command = None
            return

        if sim_time >= self._attach_deadline:
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_GRASP_FAILED,
                    ok=False,
                    attached=False,
                    tube_pose=_pose_to_dict(self.current_tube_root_pose()),
                    leg_label=cmd.leg_label,
                    reason=(
                        f"grasp validation failed: contact={contact} require_contact={args_cli.require_grasp_contact} "
                        f"left_contact={left_contact} right_contact={right_contact} grasp_error={grasp_error:.4f}"
                    ),
                )
            )
            self._gripper_command = None

    def _publish_lift_verification_if_needed(self, sim_time: float) -> None:
        if args_cli.grasp_mode != "physical":
            return
        if self._last_stage is None or not self._last_stage.endswith(":lift"):
            return
        if self._lift_verification_reported:
            return
        if self._lift_verify_started_at is None:
            self._lift_verify_started_at = sim_time
        if not self._attached or self._grasped_tube_support_z is None:
            if sim_time - self._lift_verify_started_at < max(args_cli.lift_verify_settle_s, 0.0):
                return
            self._lift_verification_reported = True
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_GRASP_LOST,
                    ok=False,
                    attached=False,
                    tube_pose=_pose_to_dict(self.current_tube_root_pose()),
                    reason="lift finished without a secured physical grasp",
                )
            )
            return

        current = self.current_tube_root_pose()
        support = _tube_support_pose_from_root(current)
        lift_delta = support.z - self._grasped_tube_support_z
        contact = self._finger_contact_details()
        left_contact = bool(contact["left_contact"])
        right_contact = bool(contact["right_contact"])
        grasp_extra = self._grasp_geometry_extra()
        bilateral_contact_ok = lift_delta >= args_cli.lift_verify_min_delta_m and left_contact and right_contact
        single_contact_ok = (
            args_cli.lift_verify_allow_single_contact
            and lift_delta >= args_cli.lift_verify_single_contact_min_delta_m
            and (left_contact or right_contact)
            and float(grasp_extra["max_finger_force_n"]) >= args_cli.lift_verify_single_contact_min_force_n
            and float(grasp_extra["tube_up_axis_z"]) >= self.upright_cos_min
            and float(grasp_extra["tube_lin_speed_mps"]) <= args_cli.lift_verify_single_contact_max_lin_vel_mps
            and float(grasp_extra["tube_ang_speed_radps"]) <= args_cli.lift_verify_single_contact_max_ang_vel_radps
        )
        if bilateral_contact_ok or single_contact_ok:
            self._lift_verification_reported = True
            self._lift_verify_started_at = None
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_LIFT_VERIFIED,
                    ok=True,
                    attached=True,
                    tube_pose=_pose_to_dict(current),
                    extra={
                        "lift_delta_m": lift_delta,
                        "single_contact_verified": bool(single_contact_ok and not bilateral_contact_ok),
                        **grasp_extra,
                        "lift_trace": self._lift_trace[-20:],
                    },
                )
            )
            self._lift_trace.clear()
            self._last_activity_sim_time = sim_time
            return

        if sim_time - self._lift_verify_started_at < max(args_cli.lift_verify_settle_s, 0.0):
            return

        self._attached = False
        self._attached_since_sim_time = None
        self._gripper_attached_write_target = None
        self._lift_verification_reported = True
        self._lift_verify_started_at = None
        self._publish_status(
            ExecutorStatus(
                phase=PHASE_GRASP_LOST,
                ok=False,
                attached=False,
                tube_pose=_pose_to_dict(current),
                reason=(
                    f"physical lift verification failed: lift_delta_m={lift_delta:.4f} "
                    f"left_contact={left_contact} right_contact={right_contact}"
                ),
                extra={
                    "lift_delta_m": lift_delta,
                    **grasp_extra,
                    "lift_trace": self._lift_trace[-40:],
                },
            )
        )

    def _should_write_gripper_state(self, sim_time: float) -> bool:
        if not args_cli.write_gripper_state:
            return False
        if args_cli.gripper_state_write_mode == "always":
            return True
        if args_cli.gripper_state_write_mode == "never":
            return False
        if args_cli.grasp_mode == "physical":
            cmd = self._gripper_command
            if self._attached:
                if (
                    args_cli.gripper_attached_write_s > 0.0
                    and self._attached_since_sim_time is not None
                    and sim_time - self._attached_since_sim_time <= args_cli.gripper_attached_write_s
                ):
                    return True
                return False
            if (
                args_cli.gripper_state_write_mode == "until_contact"
                and cmd is not None
                and cmd.mode == "close"
                and cmd.contact_confirmed_at is not None
            ):
                return False
        return True

    def _candidate_support_pose(self) -> tuple[str, PoseWxyz]:
        if self._last_leg_label is not None and self._last_leg_label in self._leg_destinations:
            return self._leg_destinations[self._last_leg_label]
        current = self.current_tube_root_pose()
        support = _tube_support_pose_from_root(current)
        holder_support = holder_slot_support_pose(self.live_task.holder_pose_w, self.live_task.holder_slot_index)
        vortexer_support = vortexer_support_pose(self.live_task.vortexer_pose_w)
        holder_dist = math.hypot(support.x - holder_support.x, support.y - holder_support.y)
        vortexer_dist = math.hypot(support.x - vortexer_support.x, support.y - vortexer_support.y)
        if holder_dist <= vortexer_dist:
            return ("holder", holder_support)
        return ("vortexer", vortexer_support)

    def _check_for_settle(self, sim_time: float) -> None:
        if not self._release_active:
            return
        current = self.current_tube_root_pose()
        support = _tube_support_pose_from_root(current)
        target_name, target_support = self._candidate_support_pose()
        lin_speed = float(torch.linalg.norm(self.runtime.tube.data.root_lin_vel_w[0]).item())
        ang_speed = float(torch.linalg.norm(self.runtime.tube.data.root_ang_vel_w[0]).item())
        up_axis = quat_apply(
            torch.tensor([[current.qw, current.qx, current.qy, current.qz]], device=self.runtime.device, dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device=self.runtime.device, dtype=torch.float32),
        )[0]
        upright = float(up_axis[2].item()) >= self.upright_cos_min
        if target_name == "vortexer":
            support_local = _pose_local_point(self.live_task.vortexer_pose_w, support.pos)
            center_xy = IMPORTED_LAB_ASSETS.vortexer_support_zone_center_local_xy_m
            halfspan_xy = IMPORTED_LAB_ASSETS.vortexer_support_zone_halfspan_xy_m
            planar_dx = support_local[0] - center_xy[0]
            planar_dy = support_local[1] - center_xy[1]
            planar_error = math.hypot(planar_dx, planar_dy)
            inside_xy = abs(planar_dx) <= halfspan_xy[0] and abs(planar_dy) <= halfspan_xy[1]
            rim_z = self.live_task.vortexer_pose_w.z + IMPORTED_LAB_ASSETS.vortexer_top_from_root_m
            insertion_below_rim = rim_z - support.z
            z_error = max(args_cli.vortexer_settle_min_insertion_below_rim_m - insertion_below_rim, 0.0)
            position_ok = inside_xy and insertion_below_rim >= args_cli.vortexer_settle_min_insertion_below_rim_m
            settle_target_extra = {
                "target_name": target_name,
                "planar_error_m": planar_error,
                "support_z_error_m": z_error,
                "inside_cavity_xy": inside_xy,
                "insertion_below_rim_m": insertion_below_rim,
                "support_local_xy_m": [support_local[0], support_local[1]],
            }
        else:
            planar_error = math.hypot(support.x - target_support.x, support.y - target_support.y)
            z_error = abs(support.z - target_support.z)
            position_ok = planar_error <= args_cli.settle_xy_tol_m and z_error <= args_cli.settle_z_tol_m
            settle_target_extra = {
                "target_name": target_name,
                "planar_error_m": planar_error,
                "support_z_error_m": z_error,
            }
        self._settle_history.append((sim_time, support.x, support.y, support.z, float(up_axis[2].item())))
        cutoff = sim_time - args_cli.settle_pose_window_s
        self._settle_history = [sample for sample in self._settle_history if sample[0] >= cutoff]

        pose_stable = False
        if self._settle_history and (self._settle_history[-1][0] - self._settle_history[0][0]) >= args_cli.settle_pose_window_s:
            xs = [sample[1] for sample in self._settle_history]
            ys = [sample[2] for sample in self._settle_history]
            zs = [sample[3] for sample in self._settle_history]
            ups = [sample[4] for sample in self._settle_history]
            planar_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            z_span = max(zs) - min(zs)
            pose_stable = (
                planar_span <= args_cli.settle_pose_planar_jitter_m
                and z_span <= args_cli.settle_pose_z_jitter_m
                and min(ups) >= self.upright_cos_min
            )

        if position_ok and upright and (
            (lin_speed <= args_cli.settle_lin_vel_tol_mps and ang_speed <= args_cli.settle_ang_vel_tol_radps)
            or pose_stable
        ):
            self._release_active = False
            self._settle_history.clear()
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_SETTLED,
                    ok=True,
                    attached=False,
                    tube_pose=_pose_to_dict(current),
                    leg_label=self._last_leg_label,
                    extra=settle_target_extra,
                )
            )
            self._last_activity_sim_time = sim_time
            return

        if sim_time - self._release_started_at > args_cli.settle_timeout_s:
            self._release_active = False
            self._settle_history.clear()
            self._publish_status(
                ExecutorStatus(
                    phase=PHASE_ERROR,
                    ok=False,
                    attached=False,
                    tube_pose=_pose_to_dict(current),
                    leg_label=self._last_leg_label,
                    reason=(
                        f"tube did not settle: target={target_name} planar_error={planar_error:.4f} "
                        f"z_error={z_error:.4f} lin={lin_speed:.4f} ang={ang_speed:.4f} upright={upright}"
                    ),
                    extra={
                        "target_support_pose": _pose_to_dict(target_support),
                        "current_support_pose": _pose_to_dict(support),
                        **settle_target_extra,
                        "lin_speed_mps": lin_speed,
                        "ang_speed_radps": ang_speed,
                        "upright": upright,
                        "pose_stable": pose_stable,
                        "up_axis_z": float(up_axis[2].item()),
                    },
                )
            )

    def _create_fixed_attach_joint(self) -> None:
        self._remove_fixed_attach_joint()
        ee_pose = self.ee_body_asset.ee_pose_w[0]
        tube_pose = self.runtime.tube.data.root_pose_w[0]
        joint = UsdPhysics.FixedJoint.Define(self.stage, Sdf.Path(ATTACH_JOINT_PATH))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(EE_BODY_PRIM_PATH)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(TUBE_PRIM_PATH)])

        body0_quat = (float(ee_pose[3].item()), float(ee_pose[4].item()), float(ee_pose[5].item()), float(ee_pose[6].item()))
        body1_quat = (float(tube_pose[3].item()), float(tube_pose[4].item()), float(tube_pose[5].item()), float(tube_pose[6].item()))
        world_anchor = (float(tube_pose[0].item()), float(tube_pose[1].item()), float(tube_pose[2].item()))

        rel0 = _quat_apply_tuple(
            _quat_conjugate(body0_quat),
            (
                world_anchor[0] - float(ee_pose[0].item()),
                world_anchor[1] - float(ee_pose[1].item()),
                world_anchor[2] - float(ee_pose[2].item()),
            ),
        )
        rel_rot0 = _quat_mul(_quat_conjugate(body0_quat), body1_quat)

        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*rel0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_rot0[0], Gf.Vec3f(rel_rot0[1], rel_rot0[2], rel_rot0[3])))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        self.runtime.tube.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.runtime.device))

    def _remove_fixed_attach_joint(self) -> None:
        if self.stage.GetPrimAtPath(ATTACH_JOINT_PATH):
            self.stage.RemovePrim(ATTACH_JOINT_PATH)

    def step(self, sim_time: float, step_index: int) -> None:
        with self._lock:
            self._apply_arm_goal(sim_time)
            self._apply_gripper_goal(sim_time)
            self.runtime.robot.set_joint_position_target(self.arm_targets, joint_ids=self.arm_joint_ids)
            self.runtime.robot.set_joint_position_target(self.gripper_targets, joint_ids=self.gripper_joint_ids)
            self.runtime.robot.set_joint_effort_target(self.gripper_efforts, joint_ids=self.gripper_joint_ids)
            if self._should_write_gripper_state(sim_time):
                self.runtime.robot.write_joint_state_to_sim(
                    self.gripper_targets,
                    self.zero_gripper_vel,
                    joint_ids=self.gripper_joint_ids,
                )
            if args_cli.arm_drive_mode == "state":
                # The live executor owns robot execution. Writing the arm state gives
                # deterministic trajectory following while the tube/gripper contacts,
                # grasp constraint, release, and settling remain live in Isaac.
                self.runtime.robot.write_joint_state_to_sim(
                    self.arm_targets,
                    self.zero_arm_vel,
                    joint_ids=self.arm_joint_ids,
                )

        self.runtime.scene.write_data_to_sim()
        self.runtime.sim.step()
        dt = self.runtime.sim.get_physics_dt()
        self.runtime.scene.update(dt)

        with self._lock:
            self._current_sim_time = float(self.runtime.sim.current_time)
            self._cached_arm_joint_positions = self.runtime.robot.data.joint_pos[:, self.arm_joint_ids].clone()
            self._cached_gripper_joint_positions = self.runtime.robot.data.joint_pos[:, self.gripper_joint_ids].clone()
            self._cached_tube_root_pose = self.current_tube_root_pose()
            self._record_lift_trace(sim_time)
            self._check_grasp_state(sim_time)
            self._check_for_settle(sim_time)
            self._check_arm_goal_completion(sim_time)
            if self._active_goal is None:
                self._publish_lift_verification_if_needed(self._current_sim_time)

        self._publish_joint_state(sim_time)
        self._publish_clock(sim_time)
        if step_index % self.capture_stride == 0:
            self._record_state_trace(sim_time, step_index)

        if args_cli.record_video and self.camera is not None and self.frames_dir is not None and step_index % self.capture_stride == 0:
            self.runtime.sim.render()
            self.camera.update(self.runtime.sim.get_physics_dt())
            rgb = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
            Image.fromarray(rgb).save(str(self.frames_dir / f"frame_{self._frame_index:06d}.png"))
            self._frame_index += 1

    def should_exit(self, sim_time: float) -> bool:
        if self.summary_status == "failure":
            return True
        if self._activity_started and self._active_goal is None and self._gripper_command is None and not self._release_active:
            return (sim_time - self._last_activity_sim_time) >= args_cli.idle_exit_s
        return False

    def finalize(self, *, capture_wall_s: float) -> None:
        if self.summary_status is None:
            self.summary_status = "success"
        if args_cli.record_video and self.frames_dir is not None:
            encoded_frames = encode_png_sequence_to_mp4(
                frames_dir=self.frames_dir,
                output_path=self.video_path,
                fps=max(args_cli.video_fps, 1.0),
            )
            video_path = str(self.video_path)
        else:
            encoded_frames = 0
            video_path = None
        metadata = {
            "seed": self.live_task.seed,
            "task_mode": self.live_task.task_mode,
            "holder_slot_index": self.live_task.holder_slot_index,
            "camera_preset": args_cli.camera_preset,
            "record_video": bool(args_cli.record_video),
            "state_trace_path": str(self.state_trace_path) if self.state_trace_path is not None else None,
            "video_path": video_path,
            "summary_status": self.summary_status,
            "captured_frames": self._frame_index,
            "encoded_video_frames": encoded_frames,
            "encoded_video_fps": args_cli.video_fps,
            "capture_wall_s": capture_wall_s,
            "camera_eye": list(camera_cfg.eye),
            "camera_target": list(camera_cfg.target),
            "status_history": self.status_history,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2))
        if self._state_trace_file is not None:
            self._state_trace_file.close()
            self._state_trace_file = None
        print(f"[isaac_live_executor] summary_status={self.summary_status}")
        if video_path is not None:
            print(f"[isaac_live_executor] video={self.video_path}")
        print(f"[isaac_live_executor] metadata={self.metadata_path}")


def _point_time(point: JointTrajectoryPoint) -> float:
    return float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1.0e-9


camera_cfg = camera_from_preset(
    preset_name=args_cli.camera_preset,
    eye_override=tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else None,
    target_override=tuple(args_cli.camera_target) if args_cli.camera_target is not None else None,
)


def main() -> None:
    print("[isaac_live_executor] startup:begin", flush=True)
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames" if args_cli.record_video else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print("[isaac_live_executor] startup:build_live_task", flush=True)
    live_task = build_live_task(args_cli.seed, args_cli.task_mode)
    print("[isaac_live_executor] startup:create_simulation", flush=True)
    sim = create_simulation(
        device=args_cli.device,
        camera_eye=list(camera_cfg.eye),
        camera_target=list(camera_cfg.target),
        dt=args_cli.physics_dt,
        contact_physics=True,
    )
    print("[isaac_live_executor] startup:create_scene", flush=True)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.5))
    print("[isaac_live_executor] startup:scene_reset", flush=True)
    sim.reset()
    scene.reset()
    print("[isaac_live_executor] startup:initialize_runtime", flush=True)
    runtime = initialize_pick_place_runtime(
        sim=sim,
        scene=scene,
        seed=args_cli.seed,
        task_cfg=PICK_PLACE_RANDOMIZATION,
        asset_profile=args_cli.asset_profile,
    )
    camera = None
    if args_cli.record_video:
        print("[isaac_live_executor] startup:configure_camera", flush=True)
        camera = scene["obs_camera"]
        camera.set_world_poses_from_view(
            eyes=torch.tensor([camera_cfg.eye], device=runtime.device, dtype=torch.float32),
            targets=torch.tensor([camera_cfg.target], device=runtime.device, dtype=torch.float32),
        )

    print("[isaac_live_executor] startup:write_initial_poses", flush=True)
    _write_rigid_pose(runtime.robot, live_task.robot_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.holder, live_task.holder_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.vortexer, live_task.vortexer_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.tube, live_task.initial_tube_root_w, device=runtime.device, zero_velocity=True)

    print("[isaac_live_executor] startup:rclpy_init", flush=True)
    rclpy.init()
    print("[isaac_live_executor] startup:create_node", flush=True)
    node = LiveIsaacExecutor(runtime=runtime, live_task=live_task, camera=camera, frames_dir=frames_dir, output_dir=output_dir)
    print("[isaac_live_executor] startup:settle_initial", flush=True)
    node.settle_initial_scene()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("[isaac_live_executor] startup:publish_ready", flush=True)
    node._publish_status(ExecutorStatus(phase=PHASE_READY, ok=True, attached=False, tube_pose=_pose_to_dict(node.current_tube_root_pose())))
    print("[isaac_live_executor] startup:ready", flush=True)

    start_wall = time.monotonic()
    step_index = 0
    try:
        while simulation_app.is_running():
            sim_time = runtime.sim.current_time
            node.step(sim_time, step_index)
            step_index += 1
            if node.should_exit(sim_time):
                break
        node.finalize(capture_wall_s=time.monotonic() - start_wall)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        if getattr(args_cli, "headless", False):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
