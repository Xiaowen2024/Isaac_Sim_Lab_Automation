#!/usr/bin/env python3
"""
ROS 2 Humble + MoveIt 2 pick-place script for the wet-lab benchmark task.

This is a single-robot planner/executor that mirrors the benchmark combos:
- combo A: holder -> holder
- combo B: holder -> vortexer -> holder

It uses the shared benchmark sampler and geometry metadata instead of the old
hard-coded demo layout so seeds, holder slots, and support poses stay aligned
with the Isaac-Lab task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    # Support direct execution via `python path/to/pick_place_move_it.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import CollisionObject, Constraints, JointConstraint, MotionPlanRequest, MotionSequenceItem
from moveit_msgs.msg import RobotState as RobotStateMsg
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from moveit_msgs.srv import GetMotionSequence
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from wetlab_benchmark.task_config import FRAMES, IMPORTED_LAB_ASSETS, PICK_PLACE_RANDOMIZATION, PLACEMENT, SURFACE, THRESH

try:
    import yaml
except ImportError:  # pragma: no cover - optional on some hosts
    yaml = None


ARM_GROUP = "xarm6"
EE_LINK = "link_eef"
PLANNING_FRAME = "link_base"
GRIPPER_CMD_TOPIC = "/xarm_gripper_traj_controller/joint_trajectory"
SYNC_EVENT_TOPIC = "/wetlab_benchmark/moveit_events"
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = ["left_finger_joint", "right_finger_joint"]

GRIPPER_OPEN = [FRAMES.gripper_open_pos, FRAMES.gripper_open_pos]
GRIPPER_CLOSED = [FRAMES.gripper_closed_pos, FRAMES.gripper_closed_pos]
GRIPPER_MOVE_DURATION_S = 0.5
SCENE_UPDATE_WAIT_S = 0.5
PRE_SUPPORT_CLEARANCE_M = 0.065
LINEAR_SEGMENT_STEP_M = 0.02
TRANSFER_CLEARANCE_M = 0.18
PILZ_PIPELINE = "pilz_industrial_motion_planner"
PILZ_PTP_PLANNER_ID = "PTP"
PILZ_LIN_PLANNER_ID = "LIN"
SEQUENCE_SERVICE_NAME = "plan_sequence_path"
APPROACH_BLEND_RADIUS_M = 0.0
CARRY_BLEND_RADIUS_M = 0.0
SEQUENCE_PLAN_TIMEOUT_S = 20.0

TABLE_SIZE_M = (1.20, 0.90, 0.04)
TABLE_CENTER_WORLD_M = (0.25, 0.0, SURFACE.z - 0.02)

HOLDER_BOX_SIZE_M = (
    2.0 * (IMPORTED_LAB_ASSETS.holder_support_zone_halfspan_xy_m[0] + 0.028),
    2.0 * (IMPORTED_LAB_ASSETS.holder_support_zone_halfspan_xy_m[1] + 0.018),
    IMPORTED_LAB_ASSETS.holder_top_from_root_m - IMPORTED_LAB_ASSETS.holder_bottom_from_root_m,
)
HOLDER_BOX_CENTER_LOCAL_M = (
    IMPORTED_LAB_ASSETS.holder_support_zone_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.holder_support_zone_center_local_xy_m[1],
    0.5 * (IMPORTED_LAB_ASSETS.holder_top_from_root_m + IMPORTED_LAB_ASSETS.holder_bottom_from_root_m),
)

VORTEXER_BOX_SIZE_M = (0.13, 0.11, IMPORTED_LAB_ASSETS.vortexer_top_from_root_m - IMPORTED_LAB_ASSETS.vortexer_bottom_from_root_m)
VORTEXER_BOX_CENTER_LOCAL_M = (
    IMPORTED_LAB_ASSETS.vortexer_support_center_local_m[0],
    IMPORTED_LAB_ASSETS.vortexer_support_center_local_m[1],
    0.5 * (IMPORTED_LAB_ASSETS.vortexer_top_from_root_m + IMPORTED_LAB_ASSETS.vortexer_bottom_from_root_m),
)

TUBE_RADIUS_M = 0.018
TUBE_HEIGHT_M = IMPORTED_LAB_ASSETS.tube_top_from_root_m - IMPORTED_LAB_ASSETS.tube_bottom_from_root_m
TUBE_CENTER_LOCAL_M = (
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
    0.5 * (IMPORTED_LAB_ASSETS.tube_top_from_root_m + IMPORTED_LAB_ASSETS.tube_bottom_from_root_m),
)
TUBE_SUPPORT_LOCAL_M = (
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
    IMPORTED_LAB_ASSETS.tube_bottom_from_root_m,
)
TOP_GRASP_QUAT_WXYZ = FRAMES.ee_top_grasp_quat_wxyz


@dataclass(frozen=True)
class PoseWxyz:
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        return (self.qw, self.qx, self.qy, self.qz)


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_apply(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    qvec = (0.0, v[0], v[1], v[2])
    rotated = _quat_mul(_quat_mul(q, qvec), _quat_conjugate(q))
    return (rotated[1], rotated[2], rotated[3])


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _pose_from_local_point(
    root_pose_w: PoseWxyz,
    local_xyz_m: tuple[float, float, float],
    local_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> PoseWxyz:
    offset = _quat_apply(root_pose_w.quat_wxyz, local_xyz_m)
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


def _pose_in_robot_frame(world_pose: PoseWxyz, robot_pose_w: PoseWxyz) -> PoseWxyz:
    robot_quat_inv = _quat_conjugate(robot_pose_w.quat_wxyz)
    rel_pos = _quat_apply(
        robot_quat_inv,
        (
            world_pose.x - robot_pose_w.x,
            world_pose.y - robot_pose_w.y,
            world_pose.z - robot_pose_w.z,
        ),
    )
    rel_quat = _quat_mul(robot_quat_inv, world_pose.quat_wxyz)
    return PoseWxyz(
        x=rel_pos[0],
        y=rel_pos[1],
        z=rel_pos[2],
        qw=rel_quat[0],
        qx=rel_quat[1],
        qy=rel_quat[2],
        qz=rel_quat[3],
    )


def _root_pose_from_support_point(
    support_pos_w: tuple[float, float, float],
    fixture_quat_w: tuple[float, float, float, float],
    local_support_point_m: tuple[float, float, float],
) -> PoseWxyz:
    local_support = _quat_apply(fixture_quat_w, local_support_point_m)
    return PoseWxyz(
        x=support_pos_w[0] - local_support[0],
        y=support_pos_w[1] - local_support[1],
        z=support_pos_w[2] - local_support[2],
        qw=fixture_quat_w[0],
        qx=fixture_quat_w[1],
        qy=fixture_quat_w[2],
        qz=fixture_quat_w[3],
    )


def _offset_world_z(pose: PoseWxyz, dz: float) -> PoseWxyz:
    return PoseWxyz(
        x=pose.x,
        y=pose.y,
        z=pose.z + dz,
        qw=pose.qw,
        qx=pose.qx,
        qy=pose.qy,
        qz=pose.qz,
    )


def _planar_distance(a: PoseWxyz, b: PoseWxyz) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _sample_fixture_pose(fixture_cfg, rng: random.Random) -> PoseWxyz:
    x = rng.uniform(*fixture_cfg.planar_bounds.x)
    y = rng.uniform(*fixture_cfg.planar_bounds.y)
    yaw = rng.uniform(*fixture_cfg.planar_bounds.yaw)
    yaw_quat = _yaw_to_quat_wxyz(yaw)
    quat = _quat_mul(yaw_quat, fixture_cfg.base_quat_wxyz)
    z = PICK_PLACE_RANDOMIZATION.surfaces[fixture_cfg.surface].z + fixture_cfg.root_height_from_surface_m
    return PoseWxyz(x=x, y=y, z=z, qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])


def _sample_fixture_layout(rng: random.Random) -> dict[str, PoseWxyz]:
    for _ in range(512):
        placed = {
            fixture_cfg.name: _sample_fixture_pose(fixture_cfg, rng)
            for fixture_cfg in PICK_PLACE_RANDOMIZATION.fixtures
        }
        valid = True
        for fixture_cfg in PICK_PLACE_RANDOMIZATION.fixtures:
            pose = placed[fixture_cfg.name]
            for constraint in fixture_cfg.min_distance_from:
                other_pose = placed[constraint.other_asset]
                planar_dist = _planar_distance(pose, other_pose)
                if planar_dist < constraint.min_distance_m:
                    valid = False
                    break
                if constraint.max_distance_m is not None and planar_dist > constraint.max_distance_m:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return placed
    raise RuntimeError("Failed to sample a valid robot/holder/vortexer layout after 512 attempts")


def _holder_slot_support_pose(holder_pose_w: PoseWxyz, slot_index: int) -> PoseWxyz:
    return _pose_from_local_point(
        holder_pose_w,
        IMPORTED_LAB_ASSETS.holder_slot_centers_local_m[slot_index],
        TOP_GRASP_QUAT_WXYZ,
    )


def _vortexer_support_pose(vortexer_pose_w: PoseWxyz) -> PoseWxyz:
    return _pose_from_local_point(
        vortexer_pose_w,
        IMPORTED_LAB_ASSETS.vortexer_support_center_local_m,
        TOP_GRASP_QUAT_WXYZ,
    )


def _tube_center_pose(tube_root_pose_w: PoseWxyz) -> PoseWxyz:
    return _pose_from_local_point(tube_root_pose_w, TUBE_CENTER_LOCAL_M)


def _ee_pose_at_tube_top(support_pose_w: PoseWxyz) -> PoseWxyz:
    return _offset_world_z(support_pose_w, TUBE_HEIGHT_M)


def _lookup_holder_slot_points() -> tuple[tuple[float, float], ...]:
    for fixture_cfg in PICK_PLACE_RANDOMIZATION.fixtures:
        if fixture_cfg.name != "holder":
            continue
        for zone in fixture_cfg.support_zones:
            if zone.name == "tube_support":
                return zone.discrete_local_xy_points_m
    raise RuntimeError("Holder tube_support zone was not found in task config")


def _pose_label(mode: int) -> str:
    return "A(holder->holder)" if mode == 0 else "B(holder->vortexer->holder)"


def _build_moveit_config_dict(args: argparse.Namespace) -> dict:
    try:
        from ament_index_python.packages import get_package_share_directory
        from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
    except ImportError as exc:
        raise RuntimeError(
            "uf_ros_lib.moveit_configs_builder is unavailable. "
            "Source the xArm ROS workspace before running this script."
        ) from exc

    moveit_config = MoveItConfigsBuilder(
        context=None,
        controllers_name=args.moveit_controllers_name,
        dof=args.robot_dof,
        robot_type=args.robot_type,
        ros2_control_plugin=args.ros2_control_plugin,
    ).to_moveit_configs().to_dict()

    pipeline_names = moveit_config.get("planning_pipelines", [])
    if isinstance(pipeline_names, str):
        pipeline_names = [pipeline_names]
    if not pipeline_names:
        pipeline_names = [moveit_config.get("default_planning_pipeline", "ompl")]
    if yaml is not None:
        pilz_config_dir = (
            Path(get_package_share_directory("xarm_moveit_config"))
            / "config"
            / "moveit_configs"
        )
        pilz_config_path = pilz_config_dir / "pilz_industrial_motion_planner_planning.yaml"
        if pilz_config_path.exists():
            with pilz_config_path.open("r", encoding="utf-8") as stream:
                pilz_config = yaml.safe_load(stream) or {}
            moveit_config[PILZ_PIPELINE] = pilz_config
            if PILZ_PIPELINE not in pipeline_names:
                pipeline_names.append(PILZ_PIPELINE)
        pilz_cartesian_limits_path = pilz_config_dir / "pilz_cartesian_limits.yaml"
        if pilz_cartesian_limits_path.exists():
            with pilz_cartesian_limits_path.open("r", encoding="utf-8") as stream:
                pilz_cartesian_limits = yaml.safe_load(stream) or {}
            moveit_config.setdefault("robot_description_planning", {})
            moveit_config["robot_description_planning"].update(pilz_cartesian_limits)

    # Humble MoveItCpp expects the pipeline list under planning_pipelines.pipeline_names
    # plus a small MoveItCpp-specific block for the planning scene monitor and default
    # plan request parameters.
    moveit_config["planning_pipelines"] = {"pipeline_names": list(pipeline_names)}
    moveit_config["planning_scene_monitor_options"] = {
        "name": "planning_scene_monitor",
        "robot_description": "robot_description",
        "joint_state_topic": "/joint_states",
        "attached_collision_object_topic": "/attached_collision_object",
        "publish_planning_scene_topic": "/planning_scene",
        "monitored_planning_scene_topic": "/monitored_planning_scene",
        "wait_for_initial_state_timeout": 10.0,
    }
    moveit_config["plan_request_params"] = {
        "planning_attempts": 1,
        "planning_pipeline": moveit_config.get("default_planning_pipeline", pipeline_names[0]),
        "max_velocity_scaling_factor": 1.0,
        "max_acceleration_scaling_factor": 1.0,
    }
    return moveit_config


@dataclass(frozen=True)
class TaskPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    @classmethod
    def from_pose_wxyz(cls, pose_wxyz: PoseWxyz) -> "TaskPose":
        return cls(
            x=pose_wxyz.x,
            y=pose_wxyz.y,
            z=pose_wxyz.z,
            qx=pose_wxyz.qx,
            qy=pose_wxyz.qy,
            qz=pose_wxyz.qz,
            qw=pose_wxyz.qw,
        )

    def to_pose_stamped(self, frame_id: str) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = self.z
        msg.pose.orientation.x = self.qx
        msg.pose.orientation.y = self.qy
        msg.pose.orientation.z = self.qz
        msg.pose.orientation.w = self.qw
        return msg


def _interpolate_task_pose(start: TaskPose, goal: TaskPose, alpha: float) -> TaskPose:
    start_quat = np.asarray((start.qx, start.qy, start.qz, start.qw), dtype=np.float64)
    goal_quat = np.asarray((goal.qx, goal.qy, goal.qz, goal.qw), dtype=np.float64)
    if float(np.dot(start_quat, goal_quat)) < 0.0:
        goal_quat = -goal_quat
    blended_quat = start_quat * (1.0 - alpha) + goal_quat * alpha
    quat_norm = float(np.linalg.norm(blended_quat))
    if quat_norm > 1.0e-9:
        blended_quat /= quat_norm
    else:
        blended_quat = goal_quat

    return TaskPose(
        x=start.x + (goal.x - start.x) * alpha,
        y=start.y + (goal.y - start.y) * alpha,
        z=start.z + (goal.z - start.z) * alpha,
        qx=float(blended_quat[0]),
        qy=float(blended_quat[1]),
        qz=float(blended_quat[2]),
        qw=float(blended_quat[3]),
    )


@dataclass(frozen=True)
class PrimitiveSpec:
    object_id: str
    primitive_type: int
    dimensions: tuple[float, ...]
    pose: TaskPose


@dataclass(frozen=True)
class LegPlan:
    label: str
    pregrasp: TaskPose
    grasp: TaskPose
    lift: TaskPose
    transit: TaskPose
    preplace: TaskPose
    place: TaskPose
    retreat: TaskPose
    placed_tube: PrimitiveSpec


@dataclass(frozen=True)
class PlannedTask:
    seed: int
    task_mode: int
    holder_slot_index: int
    scene_objects: tuple[PrimitiveSpec, ...]
    initial_tube: PrimitiveSpec
    legs: tuple[LegPlan, ...]


def _duration_msg_to_seconds(duration_msg: Duration) -> float:
    return float(duration_msg.sec) + float(duration_msg.nanosec) * 1.0e-9


def _serialize_joint_trajectory(msg: JointTrajectory) -> dict[str, object]:
    return {
        "joint_names": list(msg.joint_names),
        "points": [
            {
                "positions": [float(value) for value in point.positions],
                "velocities": [float(value) for value in point.velocities],
                "accelerations": [float(value) for value in point.accelerations],
                "time_from_start_s": _duration_msg_to_seconds(point.time_from_start),
            }
            for point in msg.points
        ],
    }


def _serialize_moveit_trajectory(trajectory) -> dict[str, object]:
    try:
        robot_msg = trajectory.get_robot_trajectory_msg()
    except TypeError:
        robot_msg = RobotTrajectoryMsg()
        trajectory.get_robot_trajectory_msg(robot_msg)
    return _serialize_joint_trajectory(robot_msg.joint_trajectory)


def _task_mode_from_arg(mode_arg: str, rng: random.Random) -> int:
    if mode_arg == "a":
        return 0
    if mode_arg == "b":
        return 1
    return rng.choices(population=[0, 1], weights=list(PLACEMENT.task_combo_probs), k=1)[0]


def _build_planned_task(seed: int, mode_arg: str) -> PlannedTask:
    rng = random.Random(seed)

    fixture_poses = _sample_fixture_layout(rng)
    robot_pose_w = fixture_poses["robot"]
    holder_pose_w = fixture_poses["holder"]
    vortexer_pose_w = fixture_poses["vortexer"]

    holder_slot_index = rng.randrange(len(_lookup_holder_slot_points()))
    task_mode = _task_mode_from_arg(mode_arg, rng)

    holder_support_w = _holder_slot_support_pose(holder_pose_w, holder_slot_index)
    vortexer_support_w = _vortexer_support_pose(vortexer_pose_w)
    initial_tube_root_w = _root_pose_from_support_point(
        holder_support_w.pos,
        holder_pose_w.quat_wxyz,
        TUBE_SUPPORT_LOCAL_M,
    )
    initial_tube_center_w = _tube_center_pose(initial_tube_root_w)

    def ee_task_pose(world_pose_w: PoseWxyz) -> TaskPose:
        return TaskPose.from_pose_wxyz(_pose_in_robot_frame(world_pose_w, robot_pose_w))

    def tube_collision_at_support(support_pose_w: PoseWxyz, fixture_pose_w: PoseWxyz) -> PrimitiveSpec:
        tube_root_w = _root_pose_from_support_point(
            support_pose_w.pos,
            fixture_pose_w.quat_wxyz,
            TUBE_SUPPORT_LOCAL_M,
        )
        tube_center_w = _tube_center_pose(tube_root_w)
        return PrimitiveSpec(
            object_id="tube",
            primitive_type=SolidPrimitive.CYLINDER,
            dimensions=(TUBE_HEIGHT_M, TUBE_RADIUS_M),
            pose=TaskPose.from_pose_wxyz(_pose_in_robot_frame(tube_center_w, robot_pose_w)),
        )

    def make_leg(
        *,
        label: str,
        source_support_w: PoseWxyz,
        dest_support_w: PoseWxyz,
        dest_fixture_w: PoseWxyz,
    ) -> LegPlan:
        grasp_w = _ee_pose_at_tube_top(source_support_w)
        pregrasp_w = _offset_world_z(grasp_w, PRE_SUPPORT_CLEARANCE_M)
        lift_w = _offset_world_z(grasp_w, THRESH.lift_height_m)

        place_w = _ee_pose_at_tube_top(dest_support_w)
        preplace_w = _offset_world_z(place_w, PRE_SUPPORT_CLEARANCE_M)
        retreat_w = _offset_world_z(place_w, THRESH.retreat_height_m)
        transit_height_w = max(lift_w.z, preplace_w.z, place_w.z + TRANSFER_CLEARANCE_M)
        transit_w = PoseWxyz(
            x=place_w.x,
            y=place_w.y,
            z=transit_height_w,
            qw=place_w.qw,
            qx=place_w.qx,
            qy=place_w.qy,
            qz=place_w.qz,
        )

        return LegPlan(
            label=label,
            pregrasp=ee_task_pose(pregrasp_w),
            grasp=ee_task_pose(grasp_w),
            lift=ee_task_pose(lift_w),
            transit=ee_task_pose(transit_w),
            preplace=ee_task_pose(preplace_w),
            place=ee_task_pose(place_w),
            retreat=ee_task_pose(retreat_w),
            placed_tube=tube_collision_at_support(dest_support_w, dest_fixture_w),
        )

    table_pose_w = PoseWxyz(x=TABLE_CENTER_WORLD_M[0], y=TABLE_CENTER_WORLD_M[1], z=TABLE_CENTER_WORLD_M[2], qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    holder_box_pose_w = _pose_from_local_point(holder_pose_w, HOLDER_BOX_CENTER_LOCAL_M)
    vortexer_box_pose_w = _pose_from_local_point(vortexer_pose_w, VORTEXER_BOX_CENTER_LOCAL_M)

    scene_objects = (
        PrimitiveSpec(
            object_id="table",
            primitive_type=SolidPrimitive.BOX,
            dimensions=TABLE_SIZE_M,
            pose=TaskPose.from_pose_wxyz(_pose_in_robot_frame(table_pose_w, robot_pose_w)),
        ),
        PrimitiveSpec(
            object_id="holder",
            primitive_type=SolidPrimitive.BOX,
            dimensions=HOLDER_BOX_SIZE_M,
            pose=TaskPose.from_pose_wxyz(_pose_in_robot_frame(holder_box_pose_w, robot_pose_w)),
        ),
        PrimitiveSpec(
            object_id="vortexer",
            primitive_type=SolidPrimitive.BOX,
            dimensions=VORTEXER_BOX_SIZE_M,
            pose=TaskPose.from_pose_wxyz(_pose_in_robot_frame(vortexer_box_pose_w, robot_pose_w)),
        ),
    )
    initial_tube = PrimitiveSpec(
        object_id="tube",
        primitive_type=SolidPrimitive.CYLINDER,
        dimensions=(TUBE_HEIGHT_M, TUBE_RADIUS_M),
        pose=TaskPose.from_pose_wxyz(_pose_in_robot_frame(initial_tube_center_w, robot_pose_w)),
    )

    legs: list[LegPlan] = []
    if task_mode == 0:
        legs.append(
            make_leg(
                label="leg_0",
                source_support_w=holder_support_w,
                dest_support_w=holder_support_w,
                dest_fixture_w=holder_pose_w,
            )
        )
    else:
        legs.append(
            make_leg(
                label="leg_0",
                source_support_w=holder_support_w,
                dest_support_w=vortexer_support_w,
                dest_fixture_w=vortexer_pose_w,
            )
        )
        legs.append(
            make_leg(
                label="leg_1",
                source_support_w=vortexer_support_w,
                dest_support_w=holder_support_w,
                dest_fixture_w=holder_pose_w,
            )
        )

    return PlannedTask(
        seed=seed,
        task_mode=task_mode,
        holder_slot_index=holder_slot_index,
        scene_objects=scene_objects,
        initial_tube=initial_tube,
        legs=tuple(legs),
    )


class XArm6MoveItPickPlace(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("xarm6_moveit_pick_place")
        self.args = args
        self.moveit = MoveItPy(node_name="moveit_py", config_dict=_build_moveit_config_dict(args))
        self.arm = self.moveit.get_planning_component(args.arm_group)
        self.scene_monitor = self.moveit.get_planning_scene_monitor()
        self.robot_model = self.moveit.get_robot_model()
        self.plan_params_ptp = self._make_plan_request_params(
            planning_pipeline=PILZ_PIPELINE,
            planner_id=PILZ_PTP_PLANNER_ID,
        )
        self.plan_params_lin = self._make_plan_request_params(
            planning_pipeline=PILZ_PIPELINE,
            planner_id=PILZ_LIN_PLANNER_ID,
        )
        self.plan_params_ompl = self._make_plan_request_params(
            planning_pipeline="ompl",
            planner_id="",
        )
        self.sequence_client = self.create_client(GetMotionSequence, SEQUENCE_SERVICE_NAME)
        self.gripper_pub = self.create_publisher(JointTrajectory, args.gripper_topic, 10)
        self.sync_event_pub = self.create_publisher(String, args.sync_events_topic, 10)
        self.task = _build_planned_task(seed=args.seed, mode_arg=args.task_mode)
        self.current_stage: str = "init"
        self.completed_stages: list[str] = []

        self.get_logger().info(
            f"MoveIt ready: group='{args.arm_group}' ee_link='{args.ee_link}' "
            f"task={_pose_label(self.task.task_mode)} seed={self.task.seed} "
            f"holder_slot={self.task.holder_slot_index}"
        )

    def run(self) -> None:
        try:
            self._wait_for_sync_subscriber()
            self._build_planning_scene(self.task.scene_objects, self.task.initial_tube)
            self._open_gripper(label="initial_open")
            self._sleep(1.0)

            for index, leg in enumerate(self.task.legs):
                self.get_logger().info(f"Executing {leg.label} ({index + 1}/{len(self.task.legs)})")
                self._do_leg(leg)

            self.get_logger().info("MoveIt task complete.")
            self._log_debug_summary(status="success")
        except Exception:
            self._log_debug_summary(status="failure")
            raise

    def _do_leg(self, leg: LegPlan) -> None:
        # Remove the tube from the free-space collision scene while it is being
        # grasped and later carried. This is a simpler fallback than attached
        # objects and lets the final descent into grasp pass through the target
        # object instead of treating it as a hard obstacle.
        self._remove_object("tube", label=f"{leg.label}:remove_tube_collision")
        try:
            self._move_pose_sequence(
                [
                    (leg.pregrasp, PILZ_PTP_PLANNER_ID, APPROACH_BLEND_RADIUS_M),
                    (leg.grasp, PILZ_LIN_PLANNER_ID, 0.0),
                ],
                label=f"{leg.label}:pickup_approach",
            )
        except RuntimeError as exc:
            self.get_logger().warning(
                f"[pick_place_move_it] stage={leg.label}:pickup_approach sequence_failed_falling_back={exc}"
            )
            self._move_ptp_pose(leg.pregrasp, f"{leg.label}:pregrasp")
            self._move_linear_pose(leg.pregrasp, leg.grasp, f"{leg.label}:grasp")
        self._close_gripper(label=f"{leg.label}:close_gripper")
        self._sleep(0.8)

        try:
            self._move_pose_sequence(
                [
                    (leg.lift, PILZ_LIN_PLANNER_ID, CARRY_BLEND_RADIUS_M),
                    (leg.transit, PILZ_LIN_PLANNER_ID, CARRY_BLEND_RADIUS_M),
                    (leg.preplace, PILZ_LIN_PLANNER_ID, CARRY_BLEND_RADIUS_M),
                    (leg.place, PILZ_LIN_PLANNER_ID, 0.0),
                ],
                label=f"{leg.label}:carry_place",
            )
        except RuntimeError as exc:
            self.get_logger().warning(
                f"[pick_place_move_it] stage={leg.label}:carry_place sequence_failed_falling_back={exc}"
            )
            self._move_linear_pose(leg.grasp, leg.lift, f"{leg.label}:lift")
            self._move_linear_pose(leg.lift, leg.transit, f"{leg.label}:transit")
            self._move_linear_pose(leg.transit, leg.preplace, f"{leg.label}:preplace")
            self._move_linear_pose(leg.preplace, leg.place, f"{leg.label}:place")

        self._open_gripper(label=f"{leg.label}:open_gripper")
        self._sleep(0.8)
        self._move_linear_pose(leg.place, leg.retreat, f"{leg.label}:retreat")
        self._apply_primitive(leg.placed_tube, label=f"{leg.label}:add_tube_collision")

    def _move_to_pose(
        self,
        goal: TaskPose,
        label: str,
        *,
        plan_params: PlanRequestParameters | None = None,
    ) -> None:
        self._log_stage_start(
            label,
            details=(
                f"goal_pos=({goal.x:.3f}, {goal.y:.3f}, {goal.z:.3f}) "
                f"goal_quat_xyzw=({goal.qx:.3f}, {goal.qy:.3f}, {goal.qz:.3f}, {goal.qw:.3f})"
            ),
        )
        plan_start = time.perf_counter()
        self.arm.set_start_state_to_current_state()
        seeded_goal_state = self._goal_robot_state_from_current(goal)
        if seeded_goal_state is None:
            self.arm.set_goal_state(
                pose_stamped_msg=goal.to_pose_stamped(self.args.planning_frame),
                pose_link=self.args.ee_link,
            )
        else:
            self.arm.set_goal_state(robot_state=seeded_goal_state)

        plan_result = self.arm.plan(plan_params)
        if not plan_result:
            self.get_logger().error(f"[pick_place_move_it] stage={label} status=PLAN_FAILED")
            raise RuntimeError(f"Planning failed for {label}")

        plan_elapsed = time.perf_counter() - plan_start
        self.get_logger().info(f"[pick_place_move_it] stage={label} status=PLAN_OK plan_s={plan_elapsed:.3f}")
        exec_start_wall = time.time()
        self._publish_trajectory_event(
            label=label,
            trajectory=plan_result.trajectory,
            execution_start_time=exec_start_wall,
        )
        exec_start = time.perf_counter()
        self._execute_trajectory(plan_result.trajectory)
        exec_elapsed = time.perf_counter() - exec_start
        self._log_stage_success(label, details=f"plan_s={plan_elapsed:.3f} exec_s={exec_elapsed:.3f}")

    def _make_plan_request_params(self, *, planning_pipeline: str, planner_id: str) -> PlanRequestParameters | None:
        try:
            params = PlanRequestParameters(self.moveit)
        except Exception as exc:  # pragma: no cover - runtime binding dependent
            self.get_logger().warning(
                f"[pick_place_move_it] could not create plan params for {planning_pipeline}/{planner_id}: {exc}"
            )
            return None
        params.planning_pipeline = planning_pipeline
        params.planner_id = planner_id
        params.planning_attempts = 1
        params.planning_time = 2.0
        params.max_velocity_scaling_factor = 1.0
        params.max_acceleration_scaling_factor = 1.0
        return params

    def _move_ptp_pose(self, goal: TaskPose, label: str) -> None:
        if self.plan_params_ptp is not None:
            try:
                self._move_to_pose(goal, label, plan_params=self.plan_params_ptp)
                return
            except RuntimeError:
                self.get_logger().warning(f"[pick_place_move_it] stage={label} pilz_ptp_failed_falling_back=ompl")
        self._move_to_pose(goal, label, plan_params=self.plan_params_ompl)

    def _move_linear_pose(self, start: TaskPose, goal: TaskPose, label: str) -> None:
        if self.plan_params_lin is not None:
            try:
                self._move_to_pose(goal, label, plan_params=self.plan_params_lin)
                return
            except RuntimeError:
                self.get_logger().warning(f"[pick_place_move_it] stage={label} pilz_lin_failed_falling_back=ompl")
        self._move_segmented_pose(start, goal, label, plan_params=self.plan_params_ompl)

    def _move_pose_sequence(
        self,
        steps: list[tuple[TaskPose, str, float]],
        *,
        label: str,
    ) -> None:
        if not steps:
            return
        if self.sequence_client is None or not self.sequence_client.wait_for_service(timeout_sec=0.5):
            raise RuntimeError(f"Sequence service '{SEQUENCE_SERVICE_NAME}' is unavailable")

        self.arm.set_start_state_to_current_state()
        start_state = self.arm.get_start_state()
        if start_state is None:
            raise RuntimeError(f"Sequence planning could not read the current start state for {label}")

        current_positions = self._joint_group_positions(start_state, self.args.arm_group)
        if current_positions is None:
            raise RuntimeError(f"Sequence planning could not read joint positions for {label}")

        request = GetMotionSequence.Request()
        request.request.items = []
        seed_positions = current_positions
        for index, (goal, planner_id, blend_radius) in enumerate(steps):
            goal_state = self._goal_robot_state_from_positions(seed_positions, goal)
            if goal_state is None:
                raise RuntimeError(f"Sequence IK failed for {label} item {index}")
            goal_positions = self._joint_group_positions(goal_state, self.args.arm_group)
            if goal_positions is None:
                raise RuntimeError(f"Sequence goal state was incomplete for {label} item {index}")

            item = MotionSequenceItem()
            item.req = self._motion_plan_request_for_positions(
                goal_positions=goal_positions,
                planner_id=planner_id,
            )
            if index == 0:
                item.req.start_state = self._robot_state_msg_from_positions(current_positions)
            item.blend_radius = float(blend_radius)
            request.request.items.append(item)
            seed_positions = goal_positions

        self._log_stage_start(label, details=f"items={len(request.request.items)} service={SEQUENCE_SERVICE_NAME}")
        plan_start = time.perf_counter()
        future = self.sequence_client.call_async(request)
        timeout_s = SEQUENCE_PLAN_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"Sequence planning timed out for {label}")

        response = future.result()
        if response is None:
            raise RuntimeError(f"Sequence planning returned no response for {label}")
        if int(response.response.error_code.val) != 1:
            raise RuntimeError(
                f"Sequence planning failed for {label} with MoveIt error {int(response.response.error_code.val)}"
            )
        if not response.response.planned_trajectories:
            raise RuntimeError(f"Sequence planning returned no trajectories for {label}")

        plan_elapsed = time.perf_counter() - plan_start
        self.get_logger().info(
            f"[pick_place_move_it] stage={label} status=PLAN_OK plan_s={plan_elapsed:.3f} "
            f"trajectories={len(response.response.planned_trajectories)}"
        )

        exec_start = time.perf_counter()
        for traj_index, trajectory_msg in enumerate(response.response.planned_trajectories, start=1):
            trajectory = self._robot_trajectory_from_msg(start_state, trajectory_msg)
            exec_start_wall = time.time()
            trajectory_label = label if len(response.response.planned_trajectories) == 1 else f"{label}:traj_{traj_index:02d}"
            self._publish_trajectory_event(
                label=trajectory_label,
                trajectory=trajectory,
                execution_start_time=exec_start_wall,
            )
            self._execute_trajectory(trajectory)
        exec_elapsed = time.perf_counter() - exec_start
        self._log_stage_success(label, details=f"plan_s={plan_elapsed:.3f} exec_s={exec_elapsed:.3f}")

    def _robot_state_msg_from_positions(self, joint_positions: np.ndarray) -> RobotStateMsg:
        msg = RobotStateMsg()
        msg.joint_state = JointState()
        msg.joint_state.name = list(ARM_JOINT_NAMES)
        msg.joint_state.position = [float(value) for value in joint_positions]
        return msg

    def _motion_plan_request_for_positions(
        self,
        *,
        goal_positions: np.ndarray,
        planner_id: str,
    ) -> MotionPlanRequest:
        req = MotionPlanRequest()
        req.pipeline_id = PILZ_PIPELINE
        req.planner_id = planner_id
        req.group_name = self.args.arm_group
        req.num_planning_attempts = 1
        req.allowed_planning_time = 2.0
        req.max_velocity_scaling_factor = 1.0
        req.max_acceleration_scaling_factor = 1.0
        constraints = Constraints()
        constraints.joint_constraints = []
        for name, position in zip(ARM_JOINT_NAMES, goal_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = float(position)
            joint_constraint.tolerance_above = 1.0e-4
            joint_constraint.tolerance_below = 1.0e-4
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        req.goal_constraints = [constraints]
        return req

    @staticmethod
    def _joint_group_positions(robot_state: RobotState, group_name: str) -> np.ndarray | None:
        try:
            positions = robot_state.get_joint_group_positions(group_name)
        except RuntimeError:
            return None
        if len(positions) == 0:
            return None
        return np.asarray(positions, dtype=np.float64)

    def _robot_trajectory_from_msg(self, start_state: RobotState, trajectory_msg: RobotTrajectoryMsg) -> RobotTrajectory:
        trajectory = RobotTrajectory(self.robot_model)
        trajectory.set_robot_trajectory_msg(start_state, trajectory_msg)
        return trajectory

    def _goal_robot_state_from_current(self, goal: TaskPose) -> RobotState | None:
        start_state = self.arm.get_start_state()
        if start_state is None:
            return None

        current_group_positions = self._joint_group_positions(start_state, self.args.arm_group)
        if current_group_positions is None:
            return None
        return self._goal_robot_state_from_positions(current_group_positions, goal)

    def _goal_robot_state_from_positions(self, seed_positions: np.ndarray, goal: TaskPose) -> RobotState | None:
        goal_state = RobotState(self.robot_model)
        goal_state.set_joint_group_positions(self.args.arm_group, seed_positions)
        solved = goal_state.set_from_ik(
            self.args.arm_group,
            goal.to_pose_stamped(self.args.planning_frame).pose,
            self.args.ee_link,
            0.2,
        )
        if not solved:
            return None
        goal_state.update()
        return goal_state

    def _move_segmented_pose(
        self,
        start: TaskPose,
        goal: TaskPose,
        label: str,
        *,
        plan_params: PlanRequestParameters | None = None,
    ) -> None:
        distance_m = math.dist((start.x, start.y, start.z), (goal.x, goal.y, goal.z))
        segment_count = max(1, int(math.ceil(distance_m / LINEAR_SEGMENT_STEP_M)))
        for segment_index in range(1, segment_count + 1):
            waypoint = _interpolate_task_pose(start, goal, segment_index / segment_count)
            segment_label = label if segment_count == 1 else f"{label}:seg_{segment_index:02d}"
            self._move_to_pose(waypoint, segment_label, plan_params=plan_params)

    def _execute_trajectory(self, trajectory) -> None:
        try:
            self.moveit.execute(self.args.arm_group, trajectory, blocking=True)
        except TypeError:
            # Newer MoveItPy releases accept the trajectory directly and expose
            # controller selection as an extra keyword argument.
            self.moveit.execute(trajectory, blocking=True, controllers=[])

    def _build_planning_scene(self, scene_objects: Iterable[PrimitiveSpec], initial_tube: PrimitiveSpec) -> None:
        self.get_logger().info("Adding planning scene collision objects")
        for primitive in scene_objects:
            self._apply_primitive(primitive, label=f"scene:add:{primitive.object_id}")
        self._apply_primitive(initial_tube, label="scene:add:tube")

    def _apply_primitive(self, primitive: PrimitiveSpec, label: str) -> None:
        self._log_stage_start(
            label,
            details=(
                f"object={primitive.object_id} type={primitive.primitive_type} "
                f"pos=({primitive.pose.x:.3f}, {primitive.pose.y:.3f}, {primitive.pose.z:.3f})"
            ),
        )
        msg = CollisionObject()
        msg.header.frame_id = self.args.planning_frame
        msg.id = primitive.object_id

        shape = SolidPrimitive()
        shape.type = primitive.primitive_type
        shape.dimensions = list(primitive.dimensions)

        msg.primitives.append(shape)
        msg.primitive_poses.append(primitive.pose.to_pose_stamped(self.args.planning_frame).pose)
        msg.operation = CollisionObject.ADD

        with self.scene_monitor.read_write() as scene_rw:
            scene_rw.apply_collision_object(msg)
        self._sleep(SCENE_UPDATE_WAIT_S)
        self._log_stage_success(label)

    def _remove_object(self, object_id: str, label: str) -> None:
        self._log_stage_start(label, details=f"object={object_id}")
        msg = CollisionObject()
        msg.header.frame_id = self.args.planning_frame
        msg.id = object_id
        msg.operation = CollisionObject.REMOVE
        with self.scene_monitor.read_write() as scene_rw:
            scene_rw.apply_collision_object(msg)
        self._sleep(SCENE_UPDATE_WAIT_S)
        self._log_stage_success(label)

    def _open_gripper(self, label: str) -> None:
        self._log_stage_start(label, details="action=open")
        self._publish_gripper(GRIPPER_OPEN)
        self._log_stage_success(label)

    def _close_gripper(self, label: str) -> None:
        self._log_stage_start(label, details="action=close")
        self._publish_gripper(GRIPPER_CLOSED)
        self._log_stage_success(label)

    def _publish_gripper(self, positions: list[float]) -> None:
        msg = JointTrajectory()
        msg.joint_names = GRIPPER_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        secs = int(GRIPPER_MOVE_DURATION_S)
        nsecs = int((GRIPPER_MOVE_DURATION_S - secs) * 1e9)
        point.time_from_start = Duration(sec=secs, nanosec=nsecs)
        msg.points = [point]
        self.gripper_pub.publish(msg)

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)

    def _wait_for_sync_subscriber(self) -> None:
        timeout_s = max(0.0, float(self.args.wait_for_sync_subscriber_timeout_s))
        if timeout_s <= 0.0:
            return

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.sync_event_pub.get_subscription_count() > 0:
                self.get_logger().info(
                    f"[pick_place_move_it] sync subscriber ready on {self.args.sync_events_topic}"
                )
                return
            time.sleep(0.1)

        raise RuntimeError(
            f"No sync subscriber connected to {self.args.sync_events_topic} within {timeout_s:.1f}s"
        )

    def _publish_sync_event(
        self,
        *,
        kind: str,
        label: str,
        status: str,
        details: str | None = None,
        event_time: float | None = None,
        extra_payload: dict[str, object] | None = None,
    ) -> None:
        if self.sync_event_pub is None:
            return
        payload = {
            "kind": kind,
            "label": label,
            "status": status,
            "details": details,
            "seed": self.task.seed,
            "task_mode": self.task.task_mode,
            "holder_slot_index": self.task.holder_slot_index,
            "time": time.time() if event_time is None else float(event_time),
        }
        if extra_payload:
            payload.update(extra_payload)
        msg = String()
        msg.data = json.dumps(payload)
        self.sync_event_pub.publish(msg)

    def _publish_trajectory_event(self, *, label: str, trajectory, execution_start_time: float) -> None:
        serialized = _serialize_moveit_trajectory(trajectory)
        details = f"points={len(serialized['points'])}"
        self._publish_sync_event(
            kind="trajectory",
            label=label,
            status="READY",
            details=details,
            event_time=execution_start_time,
            extra_payload={
                "execution_start_time": execution_start_time,
                "trajectory": serialized,
            },
        )

    def _log_stage_start(self, label: str, details: str | None = None) -> None:
        self.current_stage = label
        suffix = f" {details}" if details else ""
        self._publish_sync_event(kind="stage", label=label, status="START", details=details)
        self.get_logger().info(f"[pick_place_move_it] stage={label} status=START{suffix}")

    def _log_stage_success(self, label: str, details: str | None = None) -> None:
        if not self.completed_stages or self.completed_stages[-1] != label:
            self.completed_stages.append(label)
        suffix = f" {details}" if details else ""
        self._publish_sync_event(kind="stage", label=label, status="OK", details=details)
        self.get_logger().info(f"[pick_place_move_it] stage={label} status=OK{suffix}")

    def _log_debug_summary(self, status: str) -> None:
        completed = ",".join(self.completed_stages) if self.completed_stages else "none"
        self._publish_sync_event(kind="summary", label="task", status=status, details=completed)
        self.get_logger().info(
            f"[pick_place_move_it] summary status={status} "
            f"task={_pose_label(self.task.task_mode)} seed={self.task.seed} "
            f"holder_slot={self.task.holder_slot_index} "
            f"last_stage={self.current_stage} completed={completed}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoveIt wet-lab pick-place runner for xArm6.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task_mode", type=str.lower, default="sample", choices=["sample", "a", "b"])
    parser.add_argument("--arm_group", type=str, default=ARM_GROUP)
    parser.add_argument("--ee_link", type=str, default=EE_LINK)
    parser.add_argument("--planning_frame", type=str, default=PLANNING_FRAME)
    parser.add_argument("--gripper_topic", type=str, default=GRIPPER_CMD_TOPIC)
    parser.add_argument("--sync_events_topic", type=str, default=SYNC_EVENT_TOPIC)
    parser.add_argument("--robot_type", type=str, default="xarm")
    parser.add_argument("--robot_dof", type=int, default=6)
    parser.add_argument("--moveit_controllers_name", type=str, default="fake_controllers")
    parser.add_argument("--ros2_control_plugin", type=str, default="uf_robot_hardware/UFRobotFakeSystemHardware")
    parser.add_argument("--wait_for_sync_subscriber_timeout_s", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = XArm6MoveItPickPlace(args)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run()
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
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
