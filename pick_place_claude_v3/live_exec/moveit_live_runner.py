#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, Constraints, JointConstraint, MotionPlanRequest, MotionSequenceItem
from moveit_msgs.msg import RobotState as RobotStateMsg
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from moveit_msgs.srv import GetMotionSequence
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from wetlab_benchmark.pick_place_claude_v3.live_exec.protocol import (
    ARM_TRAJECTORY_ACTION,
    EXECUTOR_STATUS_TOPIC,
    EXECUTOR_CONTROL_TOPIC,
    GRIPPER_COMMAND_TOPIC,
    JOINT_STATE_TOPIC,
    PHASE_ERROR,
    PHASE_GRASP_ATTACHED,
    PHASE_GRASP_FAILED,
    PHASE_GRASP_LOST,
    PHASE_GRASP_SECURED,
    PHASE_GRIPPER_OPEN,
    PHASE_LIFT_VERIFIED,
    PHASE_READY,
    PHASE_RELEASED,
    PHASE_SETTLED,
    ExecutorControl,
    ExecutorStatus,
)
from wetlab_benchmark.pick_place_claude_v3.live_exec.task_builder import (
    GRASP_TOOL_CLEARANCE_M,
    LiveLegPlan,
    LiveTask,
    PICKUP_GRASP_BIAS_LOCAL_M,
    PoseWxyz,
    PrimitiveSpec,
    TUBE_CENTER_LOCAL_M,
    build_leg_plan,
    build_live_task,
    ee_body_pose_from_tool_pose,
    tube_center_pose,
    tube_ee_grasp_pose,
    tube_world_primitive,
)
from wetlab_benchmark.task_config import FRAMES

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ARM_GROUP = "xarm6"
EE_LINK = "link_eef"
PLANNING_FRAME = "link_base"
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = [FRAMES.left_finger_joint, FRAMES.right_finger_joint]

GRIPPER_OPEN = [FRAMES.gripper_open_pos, FRAMES.gripper_open_pos]
GRIPPER_CLOSED = [FRAMES.gripper_closed_pos, FRAMES.gripper_closed_pos]
GRIPPER_TOUCH_LINKS = [
    EE_LINK,
    FRAMES.ee_body_name,
    "left_finger",
    "right_finger",
]
FIXTURE_COLLISION_OBJECT_IDS = (
    "holder_wall_left",
    "holder_wall_right",
    "holder_wall_front",
    "holder_wall_back",
    "vortexer_wall_left",
    "vortexer_wall_right",
    "vortexer_wall_front",
    "vortexer_wall_back",
)
FIXTURE_COLLISION_OBJECT_IDS_BY_PREFIX = {
    "holder": tuple(object_id for object_id in FIXTURE_COLLISION_OBJECT_IDS if object_id.startswith("holder_")),
    "vortexer": tuple(object_id for object_id in FIXTURE_COLLISION_OBJECT_IDS if object_id.startswith("vortexer_")),
}
GRIPPER_MOVE_DURATION_S = 0.5
SCENE_UPDATE_WAIT_S = 0.3
PILZ_PIPELINE = "pilz_industrial_motion_planner"
PILZ_PTP_PLANNER_ID = "PTP"
PILZ_LIN_PLANNER_ID = "LIN"
SEQUENCE_SERVICE_NAME = "plan_sequence_path"
SEQUENCE_PLAN_TIMEOUT_S = 20.0
EXECUTOR_READY_TIMEOUT_S = 120.0
EXECUTOR_PHASE_TIMEOUT_S = 90.0
STATUS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _build_moveit_config_dict(args: argparse.Namespace) -> dict:
    try:
        from ament_index_python.packages import get_package_share_directory
        from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
    except ImportError as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "uf_ros_lib.moveit_configs_builder is unavailable. Source the xArm ROS workspace before running this script."
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
        pilz_config_dir = Path(get_package_share_directory("xarm_moveit_config")) / "config" / "moveit_configs"
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

    moveit_config["planning_pipelines"] = {"pipeline_names": list(pipeline_names)}
    moveit_config["planning_scene_monitor_options"] = {
        "name": "planning_scene_monitor",
        "robot_description": "robot_description",
        "joint_state_topic": JOINT_STATE_TOPIC,
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


def _duration_to_seconds(duration_msg) -> float:
    return float(duration_msg.sec) + float(duration_msg.nanosec) * 1.0e-9


def _serialize_moveit_trajectory(trajectory) -> JointTrajectory:
    try:
        robot_msg = trajectory.get_robot_trajectory_msg()
    except TypeError:
        robot_msg = RobotTrajectoryMsg()
        trajectory.get_robot_trajectory_msg(robot_msg)
    return robot_msg.joint_trajectory


def _joint_trajectory_to_dict(trajectory: JointTrajectory) -> dict:
    return {
        "joint_names": list(trajectory.joint_names),
        "points": [
            {
                "positions": list(point.positions),
                "velocities": list(point.velocities),
                "accelerations": list(point.accelerations),
                "effort": list(point.effort),
                "time_from_start": {
                    "sec": int(point.time_from_start.sec),
                    "nanosec": int(point.time_from_start.nanosec),
                },
            }
            for point in trajectory.points
        ],
    }


def _joint_trajectory_from_dict(payload: dict) -> JointTrajectory:
    msg = JointTrajectory()
    msg.joint_names = list(payload.get("joint_names", []))
    for point_payload in payload.get("points", []):
        point = JointTrajectoryPoint()
        point.positions = list(point_payload.get("positions", []))
        point.velocities = list(point_payload.get("velocities", []))
        point.accelerations = list(point_payload.get("accelerations", []))
        point.effort = list(point_payload.get("effort", []))
        time_payload = point_payload.get("time_from_start", {})
        point.time_from_start.sec = int(time_payload.get("sec", 0))
        point.time_from_start.nanosec = int(time_payload.get("nanosec", 0))
        msg.points.append(point)
    return msg


def _scale_joint_trajectory_timing(trajectory: JointTrajectory, scale: float) -> JointTrajectory:
    if scale <= 0.0:
        raise ValueError(f"trajectory timing scale must be positive, got {scale}")
    if math.isclose(scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return trajectory

    scaled = JointTrajectory()
    scaled.joint_names = list(trajectory.joint_names)
    for point in trajectory.points:
        scaled_point = JointTrajectoryPoint()
        scaled_point.positions = list(point.positions)
        scaled_point.velocities = [value / scale for value in point.velocities]
        accel_scale = scale * scale
        scaled_point.accelerations = [value / accel_scale for value in point.accelerations]
        scaled_point.effort = list(point.effort)
        time_s = _duration_to_seconds(point.time_from_start) * scale
        secs = int(time_s)
        scaled_point.time_from_start.sec = secs
        scaled_point.time_from_start.nanosec = int(round((time_s - secs) * 1.0e9))
        if scaled_point.time_from_start.nanosec >= 1_000_000_000:
            scaled_point.time_from_start.sec += 1
            scaled_point.time_from_start.nanosec -= 1_000_000_000
        scaled.points.append(scaled_point)
    return scaled


def _cache_safe_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    return safe or "trajectory"


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


def _relative_pose(parent_w: PoseWxyz, child_w: PoseWxyz) -> PoseWxyz:
    parent_inv = _quat_conjugate(parent_w.quat_wxyz)
    rel_pos = _quat_apply(
        parent_inv,
        (
            child_w.x - parent_w.x,
            child_w.y - parent_w.y,
            child_w.z - parent_w.z,
        ),
    )
    rel_quat = _quat_mul(parent_inv, child_w.quat_wxyz)
    return PoseWxyz(
        x=rel_pos[0],
        y=rel_pos[1],
        z=rel_pos[2],
        qw=rel_quat[0],
        qx=rel_quat[1],
        qy=rel_quat[2],
        qz=rel_quat[3],
    )


def _pose_to_msg(pose_wxyz: PoseWxyz) -> Pose:
    msg = Pose()
    msg.position.x = pose_wxyz.x
    msg.position.y = pose_wxyz.y
    msg.position.z = pose_wxyz.z
    msg.orientation.w = pose_wxyz.qw
    msg.orientation.x = pose_wxyz.qx
    msg.orientation.y = pose_wxyz.qy
    msg.orientation.z = pose_wxyz.qz
    return msg


class LiveMoveItRunner(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("moveit_live_runner")
        self.args = args
        self.task: LiveTask = build_live_task(args.seed, args.task_mode)
        self.moveit = MoveItPy(node_name="moveit_live_runner_cpp", config_dict=_build_moveit_config_dict(args))
        self.arm = self.moveit.get_planning_component(args.arm_group)
        self.scene_monitor = self.moveit.get_planning_scene_monitor()
        self.robot_model = self.moveit.get_robot_model()

        self.plan_params_ptp = self._make_plan_request_params(PILZ_PIPELINE, PILZ_PTP_PLANNER_ID)
        self.plan_params_lin = self._make_plan_request_params(PILZ_PIPELINE, PILZ_LIN_PLANNER_ID)
        self.plan_params_ompl = self._make_plan_request_params("ompl", "")

        self.sequence_client = self.create_client(GetMotionSequence, SEQUENCE_SERVICE_NAME)
        self.arm_client = ActionClient(self, FollowJointTrajectory, args.trajectory_action)
        self.gripper_pub = self.create_publisher(JointTrajectory, args.gripper_topic, 10)
        self.attached_pub = self.create_publisher(AttachedCollisionObject, "/attached_collision_object", 10)
        self.control_pub = self.create_publisher(String, args.control_topic, 20)
        self.status_sub = self.create_subscription(String, args.status_topic, self._on_status, STATUS_QOS)

        self._status_lock = threading.Condition()
        self._status_history: list[ExecutorStatus] = []
        self.current_stage = "init"
        self.completed_stages: list[str] = []
        self.trajectory_cache_dir = (
            Path(args.trajectory_cache_dir).expanduser().resolve() if args.trajectory_cache_dir else None
        )
        if self.trajectory_cache_dir is not None and self._trajectory_cache_can_write():
            self.trajectory_cache_dir.mkdir(parents=True, exist_ok=True)

    def _on_status(self, msg: String) -> None:
        try:
            status = ExecutorStatus.from_json(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"Ignoring malformed executor status: {exc}")
            return
        with self._status_lock:
            self._status_history.append(status)
            self._status_lock.notify_all()

    def _wait_for_status(
        self,
        *,
        since_stamp: float,
        phases: set[str],
        timeout_s: float,
        leg_label: str | None = None,
    ) -> ExecutorStatus:
        deadline = time.monotonic() + timeout_s
        with self._status_lock:
            while True:
                for status in reversed(self._status_history):
                    if (
                        status.stamp >= since_stamp
                        and status.phase in phases
                        and (leg_label is None or status.leg_label == leg_label)
                    ):
                        return status
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(f"Timed out waiting for executor phases {sorted(phases)}")
                self._status_lock.wait(timeout=remaining)

    def _wait_for_executor_ready(self) -> None:
        self.get_logger().info("Waiting for Isaac executor trajectory action server")
        if not self.arm_client.wait_for_server(timeout_sec=self.args.executor_ready_timeout_s):
            raise RuntimeError(f"Trajectory action server {self.args.trajectory_action} is unavailable")
        self.get_logger().info("Isaac executor trajectory action server is ready")

    def _wait_for_initial_tube_pose(self) -> PoseWxyz:
        ready = self._wait_for_status(
            since_stamp=0.0,
            phases={PHASE_READY, PHASE_ERROR},
            timeout_s=self.args.executor_ready_timeout_s,
        )
        if ready.phase == PHASE_ERROR or not ready.ok:
            raise RuntimeError(ready.reason or "Isaac executor reported startup failure")
        if ready.tube_pose is None:
            self.get_logger().warning("Isaac ready status did not include tube_pose; using nominal seeded pose")
            return self.task.initial_tube_root_w
        tube_root_w = PoseWxyz.from_dict(ready.tube_pose)
        self.get_logger().info(
            "Using Isaac-settled initial tube pose "
            f"x={tube_root_w.x:.4f} y={tube_root_w.y:.4f} z={tube_root_w.z:.4f}"
        )
        return tube_root_w

    def _trajectory_cache_can_read(self) -> bool:
        return self.trajectory_cache_dir is not None and self.args.trajectory_cache_mode in {"read", "readwrite"}

    def _trajectory_cache_can_write(self) -> bool:
        return self.trajectory_cache_dir is not None and self.args.trajectory_cache_mode in {"write", "readwrite"}

    def _trajectory_cache_path(self, label: str) -> Path:
        if self.trajectory_cache_dir is None:
            raise RuntimeError("trajectory cache dir is not configured")
        return self.trajectory_cache_dir / f"{_cache_safe_name(label)}.json"

    def _sequence_cache_path(self, sequence_label: str) -> Path:
        if self.trajectory_cache_dir is None:
            raise RuntimeError("trajectory cache dir is not configured")
        return self.trajectory_cache_dir / f"{_cache_safe_name(sequence_label)}.sequence.json"

    def _load_cached_trajectory(self, label: str) -> JointTrajectory | None:
        if not self._trajectory_cache_can_read():
            return None
        path = self._trajectory_cache_path(label)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.get_logger().info(f"{label}: replaying cached trajectory from {path}")
        return _joint_trajectory_from_dict(payload)

    def _store_cached_trajectory(self, label: str, trajectory_msg: JointTrajectory) -> None:
        if not self._trajectory_cache_can_write():
            return
        path = self._trajectory_cache_path(label)
        path.write_text(json.dumps(_joint_trajectory_to_dict(trajectory_msg), indent=2), encoding="utf-8")

    def _load_sequence_manifest(self, sequence_label: str) -> list[str] | None:
        if not self._trajectory_cache_can_read():
            return None
        path = self._sequence_cache_path(sequence_label)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = payload.get("trajectory_labels")
        if not isinstance(labels, list) or not labels:
            return None
        self.get_logger().info(f"{sequence_label}: replaying cached sequence manifest from {path}")
        return [str(label) for label in labels]

    def _store_sequence_manifest(self, sequence_label: str, trajectory_labels: list[str]) -> None:
        if not self._trajectory_cache_can_write():
            return
        path = self._sequence_cache_path(sequence_label)
        payload = {
            "sequence_label": sequence_label,
            "trajectory_labels": list(trajectory_labels),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run(self) -> None:
        current_tube_root_w = self.task.initial_tube_root_w
        current_fixture_name = "holder"
        try:
            self._wait_for_executor_ready()
            current_tube_root_w = self._wait_for_initial_tube_pose()
            self._build_planning_scene(self.task.scene_objects, tube_root_w=current_tube_root_w)
            self._open_gripper("initial_open")
            max_legs = self.args.max_legs if self.args.max_legs and self.args.max_legs > 0 else len(self.task.legs)
            for leg_spec in self.task.legs[:max_legs]:
                self.get_logger().info(f"Executing {leg_spec.label}")
                pickup_grasp_bias_local_m = (
                    self.args.pickup_grasp_bias_x_m + self.args.vortexer_pickup_grasp_bias_x_m,
                    self.args.pickup_grasp_bias_y_m + self.args.vortexer_pickup_grasp_bias_y_m,
                    self.args.pickup_grasp_bias_z_m + self.args.vortexer_pickup_grasp_bias_z_m,
                ) if current_fixture_name == "vortexer" else (
                    self.args.pickup_grasp_bias_x_m,
                    self.args.pickup_grasp_bias_y_m,
                    self.args.pickup_grasp_bias_z_m,
                )
                leg = build_leg_plan(
                    label=leg_spec.label,
                    dest_name=leg_spec.dest_name,
                    robot_pose_w=self.task.robot_pose_w,
                    source_tube_root_w=current_tube_root_w,
                    dest_support_w=leg_spec.dest_support_w,
                    dest_fixture_w=leg_spec.dest_fixture_w,
                    grasp_tool_clearance_m=self.args.grasp_tool_clearance_m,
                    verify_lift_height_m=self.args.verify_lift_height_m,
                    place_tool_clearance_m=(
                        self.args.vortexer_place_clearance_m if leg_spec.dest_name == "vortexer" else None
                    ),
                    pickup_grasp_bias_local_m=pickup_grasp_bias_local_m,
                    place_bias_local_m=(
                        (
                            self.args.vortexer_place_bias_x_m,
                            self.args.vortexer_place_bias_y_m,
                            self.args.vortexer_place_bias_z_m,
                        )
                        if leg_spec.dest_name == "vortexer"
                        else (0.0, 0.0, 0.0)
                    ),
                )
                settled = self._do_leg(
                    leg,
                    source_tube_root_w=current_tube_root_w,
                    source_fixture_name=current_fixture_name,
                )
                if not settled.ok or settled.tube_pose is None:
                    raise RuntimeError(settled.reason or f"{leg.label}: settle failed")
                current_tube_root_w = PoseWxyz.from_dict(settled.tube_pose)
                current_fixture_name = leg_spec.dest_name
                self._apply_world_primitive(
                    tube_world_primitive(tube_root_w=current_tube_root_w, robot_pose_w=self.task.robot_pose_w),
                    label=f"{leg.label}:add_world_tube",
                )
            self.get_logger().info("Live MoveIt task complete.")
        except Exception:
            self.get_logger().error(f"Live MoveIt task failed:\n{traceback.format_exc()}")
            raise

    def _do_leg(
        self,
        leg: LiveLegPlan,
        *,
        source_tube_root_w: PoseWxyz,
        source_fixture_name: str,
    ) -> ExecutorStatus:
        self._execute_pose_steps(
            [
                # The cap-grasp branch is less tolerant of the aggressive Pilz
                # PTP shoulder motion here; use OMPL for pregrasp so the arm can
                # choose a smoother reachable approach before the final linear
                # grasp stroke.
                (leg.pregrasp, self.plan_params_ompl, f"{leg.label}:pregrasp"),
                (leg.grasp, self.plan_params_lin, f"{leg.label}:grasp"),
            ],
            sequence_label=f"{leg.label}:pickup_approach",
        )
        # Plan this while the gripper is still open. After Isaac reports a
        # physical grasp, sending the lift immediately avoids a long planning
        # gap where the imported gripper can relax and lose contact.
        preplanned_lift = self._load_cached_trajectory(f"{leg.label}:lift")
        if preplanned_lift is None:
            preplanned_lift = self._plan_to_pose(
                leg.lift,
                label=f"{leg.label}:lift_preplan",
                plan_params=self.plan_params_lin,
            )

        close_stamp = self._close_gripper(label=f"{leg.label}:close_gripper")
        grasp_success_phase = PHASE_GRASP_ATTACHED if self.args.grasp_mode == "fixed_joint" else PHASE_GRASP_SECURED
        grasp_status = self._wait_for_status(
            since_stamp=close_stamp,
            phases={grasp_success_phase, PHASE_GRASP_FAILED, PHASE_ERROR},
            timeout_s=self.args.executor_phase_timeout_s,
            leg_label=leg.label,
        )
        if grasp_status.phase != grasp_success_phase or not grasp_status.ok:
            raise RuntimeError(grasp_status.reason or f"{leg.label}: physical grasp failed")

        if self.args.grasp_mode == "physical" and self.args.post_grasp_settle_s > 0.0:
            settle_deadline = time.monotonic() + self.args.post_grasp_settle_s
            while time.monotonic() < settle_deadline:
                try:
                    settle_status = self._wait_for_status(
                        since_stamp=grasp_status.stamp,
                        phases={PHASE_GRASP_LOST, PHASE_ERROR},
                        timeout_s=min(0.1, settle_deadline - time.monotonic()),
                        leg_label=leg.label,
                    )
                except RuntimeError:
                    continue
                raise RuntimeError(settle_status.reason or f"{leg.label}: physical grasp was lost during settle")

        planning_tube_root_w = (
            PoseWxyz.from_dict(grasp_status.tube_pose) if grasp_status.tube_pose is not None else source_tube_root_w
        )
        grasp_w = ee_body_pose_from_tool_pose(tube_ee_grasp_pose(planning_tube_root_w))
        self._attach_tube_to_ee(
            source_tube_root_w=planning_tube_root_w,
            grasp_w=grasp_w,
            label=f"{leg.label}:attach_tube",
        )
        removed_fixture_object_ids: tuple[str, ...] = ()
        should_remove_source_fixture = self.args.grasp_mode == "physical" or self.args.remove_fixture_collision_during_carry
        if should_remove_source_fixture:
            removed_fixture_object_ids = FIXTURE_COLLISION_OBJECT_IDS_BY_PREFIX.get(source_fixture_name, ())
            if removed_fixture_object_ids:
                self._remove_fixture_collision_objects(
                    object_ids=removed_fixture_object_ids,
                    label=f"{leg.label}:remove_{source_fixture_name}_walls_for_carry",
                )

        self._send_arm_trajectory(preplanned_lift, label=f"{leg.label}:lift")
        if self.args.grasp_mode == "physical":
            lift_status = self._wait_for_status(
                since_stamp=close_stamp,
                phases={PHASE_LIFT_VERIFIED, PHASE_GRASP_LOST, PHASE_ERROR},
                timeout_s=self.args.executor_phase_timeout_s,
                leg_label=leg.label,
            )
            if lift_status.phase != PHASE_LIFT_VERIFIED or not lift_status.ok:
                raise RuntimeError(lift_status.reason or f"{leg.label}: physical lift verification failed")

        self._execute_pose_steps(
            [
                (leg.transit, self.plan_params_lin, f"{leg.label}:transit"),
                (leg.preplace, self.plan_params_lin, f"{leg.label}:preplace"),
                (leg.place, self.plan_params_lin, f"{leg.label}:place"),
            ],
            sequence_label=f"{leg.label}:carry_place",
        )

        open_stamp = self._open_gripper(label=f"{leg.label}:open_gripper")
        released = self._wait_for_status(
            since_stamp=open_stamp,
            phases={PHASE_RELEASED, PHASE_SETTLED, PHASE_ERROR},
            timeout_s=self.args.executor_phase_timeout_s,
            leg_label=leg.label,
        )
        if not released.ok:
            raise RuntimeError(released.reason or f"{leg.label}: release failed")

        self._detach_tube_from_ee(label=f"{leg.label}:detach_tube")
        settled = released
        retreat_steps = [(leg.retreat, self.plan_params_lin, f"{leg.label}:retreat")]
        if leg.dest_name == "vortexer":
            # For cavity placement, retreat immediately after opening so the tube
            # can settle under gravity without the open jaws hovering around it.
            self._execute_pose_steps(retreat_steps, sequence_label=f"{leg.label}:retreat_sequence")
            if released.phase != PHASE_SETTLED:
                settled = self._wait_for_status(
                    since_stamp=open_stamp,
                    phases={PHASE_SETTLED, PHASE_ERROR},
                    timeout_s=self.args.executor_phase_timeout_s,
                    leg_label=leg.label,
                )
                if not settled.ok:
                    raise RuntimeError(settled.reason or f"{leg.label}: settle failed")
        else:
            if released.phase != PHASE_SETTLED:
                settled = self._wait_for_status(
                    since_stamp=open_stamp,
                    phases={PHASE_SETTLED, PHASE_ERROR},
                    timeout_s=self.args.executor_phase_timeout_s,
                    leg_label=leg.label,
                )
                if not settled.ok:
                    raise RuntimeError(settled.reason or f"{leg.label}: settle failed")
            self._execute_pose_steps(retreat_steps, sequence_label=f"{leg.label}:retreat_sequence")
        if removed_fixture_object_ids:
            self._restore_fixture_collision_objects(
                object_ids=removed_fixture_object_ids,
                label=f"{leg.label}:restore_{source_fixture_name}_walls",
            )
        return settled

    def _execute_pose_steps(
        self,
        steps: list[tuple[object, PlanRequestParameters | None, str]],
        *,
        sequence_label: str,
    ) -> None:
        cached_sequence = self._load_sequence_manifest(sequence_label)
        if cached_sequence is not None:
            for cached_label in cached_sequence:
                cached_trajectory = self._load_cached_trajectory(cached_label)
                if cached_trajectory is None:
                    raise RuntimeError(f"{sequence_label}: missing cached trajectory for {cached_label}")
                self._send_arm_trajectory(cached_trajectory, label=cached_label)
            return
        use_sequence_service = (
            self.args.use_sequence_service
            and self.sequence_client.wait_for_service(timeout_sec=self.args.sequence_service_wait_s)
        )
        self.get_logger().info(f"{sequence_label}: use_sequence_service={use_sequence_service}")
        if use_sequence_service:
            try:
                self._execute_sequence_steps(steps, label=sequence_label)
                return
            except Exception as exc:
                self.get_logger().warning(f"{sequence_label}: sequence planning failed, falling back to staged planning: {exc}")
        executed_labels: list[str] = []
        for goal, params, label in steps:
            effective_params = params if params is not None else self.plan_params_ompl
            trajectory = self._plan_to_pose(goal, label=label, plan_params=effective_params)
            self._send_arm_trajectory(trajectory, label=label)
            executed_labels.append(label)
        self._store_sequence_manifest(sequence_label, executed_labels)

    def _execute_sequence_steps(
        self,
        steps: list[tuple[object, PlanRequestParameters | None, str]],
        *,
        label: str,
    ) -> None:
        self.arm.set_start_state_to_current_state()
        start_state = self.arm.get_start_state()
        if start_state is None:
            raise RuntimeError(f"{label}: start state unavailable")

        current_positions = self._joint_group_positions(start_state, self.args.arm_group)
        if current_positions is None:
            raise RuntimeError(f"{label}: current joint positions unavailable")

        request = GetMotionSequence.Request()
        request.request.items = []
        seed_positions = current_positions
        for index, (goal, params, _) in enumerate(steps):
            planner_id = (params.planner_id if params is not None else "") or PILZ_LIN_PLANNER_ID
            goal_state = self._goal_robot_state_from_positions(seed_positions, goal)
            if goal_state is None:
                raise RuntimeError(f"{label}: IK failed for item {index}")
            goal_positions = self._joint_group_positions(goal_state, self.args.arm_group)
            if goal_positions is None:
                raise RuntimeError(f"{label}: goal positions unavailable for item {index}")
            item = MotionSequenceItem()
            item.req = self._motion_plan_request_for_positions(goal_positions=goal_positions, planner_id=planner_id)
            if index == 0:
                item.req.start_state = self._robot_state_msg_from_positions(current_positions)
            item.blend_radius = 0.0
            request.request.items.append(item)
            seed_positions = goal_positions

        future = self.sequence_client.call_async(request)
        deadline = time.monotonic() + SEQUENCE_PLAN_TIMEOUT_S
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"{label}: sequence planning timed out")
        response = future.result()
        if response is None or int(response.response.error_code.val) != 1:
            raise RuntimeError(f"{label}: sequence planning failed")
        executed_labels: list[str] = []
        for traj_index, trajectory_msg in enumerate(response.response.planned_trajectories, start=1):
            trajectory = self._robot_trajectory_from_msg(start_state, trajectory_msg)
            traj_label = label if len(response.response.planned_trajectories) == 1 else f"{label}:traj_{traj_index:02d}"
            self._send_arm_trajectory(trajectory, label=traj_label)
            executed_labels.append(traj_label)
        self._store_sequence_manifest(label, executed_labels)

    def _plan_to_pose(self, goal, *, label: str, plan_params: PlanRequestParameters | None):
        self.current_stage = label
        self.get_logger().info(f"{label}: planning begin pipeline={getattr(plan_params, 'planning_pipeline', 'default')}")
        self.arm.set_start_state_to_current_state()
        seeded_goal_state = self._goal_robot_state_from_current(goal)
        if seeded_goal_state is None:
            self.arm.set_goal_state(pose_stamped_msg=goal.to_pose_stamped(self.args.planning_frame), pose_link=self.args.ee_link)
        else:
            self.arm.set_goal_state(robot_state=seeded_goal_state)
        plan_result = self.arm.plan(plan_params)
        if not plan_result:
            raise RuntimeError(f"{label}: planning failed")
        self.get_logger().info(f"{label}: planning done")
        return plan_result.trajectory

    def _send_arm_trajectory(self, trajectory, *, label: str) -> None:
        self.get_logger().info(f"{label}: sending trajectory")
        self._publish_control(stage=label, leg_label=self._leg_label_from_stage(label))
        goal = FollowJointTrajectory.Goal()
        if isinstance(trajectory, JointTrajectory):
            goal.trajectory = trajectory
        else:
            goal.trajectory = _serialize_moveit_trajectory(trajectory)
        goal.trajectory = _scale_joint_trajectory_timing(goal.trajectory, self.args.arm_trajectory_time_scale)
        self._store_cached_trajectory(label, goal.trajectory)
        send_future = self.arm_client.send_goal_async(goal)
        while not send_future.done():
            time.sleep(0.01)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"{label}: trajectory goal rejected")
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
        result = result_future.result()
        if result is None:
            raise RuntimeError(f"{label}: no action result returned")
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"{label}: action status={result.status}")
        if int(result.result.error_code) != 0:
            raise RuntimeError(f"{label}: controller error_code={int(result.result.error_code)} {result.result.error_string}")
        self.get_logger().info(f"{label}: trajectory complete")
        self.completed_stages.append(label)

    @staticmethod
    def _leg_label_from_stage(stage: str) -> str | None:
        if ":" not in stage:
            return None
        return stage.split(":", 1)[0]

    def _publish_control(self, *, stage: str, leg_label: str | None) -> float:
        msg = String()
        stamp = time.time()
        msg.data = ExecutorControl(stage=stage, leg_label=leg_label, stamp=stamp).to_json()
        self.control_pub.publish(msg)
        return stamp

    def _make_plan_request_params(self, planning_pipeline: str, planner_id: str) -> PlanRequestParameters | None:
        try:
            params = PlanRequestParameters(self.moveit)
        except Exception:
            return None
        params.planning_pipeline = planning_pipeline
        params.planner_id = planner_id
        params.planning_attempts = 1
        params.planning_time = 2.0
        params.max_velocity_scaling_factor = 1.0
        params.max_acceleration_scaling_factor = 1.0
        return params

    @staticmethod
    def _joint_group_positions(robot_state: RobotState, group_name: str) -> np.ndarray | None:
        try:
            positions = robot_state.get_joint_group_positions(group_name)
        except RuntimeError:
            return None
        if len(positions) == 0:
            return None
        return np.asarray(positions, dtype=np.float64)

    def _goal_robot_state_from_current(self, goal) -> RobotState | None:
        start_state = self.arm.get_start_state()
        if start_state is None:
            return None
        current_positions = self._joint_group_positions(start_state, self.args.arm_group)
        if current_positions is None:
            return None
        return self._goal_robot_state_from_positions(current_positions, goal)

    def _goal_robot_state_from_positions(self, seed_positions: np.ndarray, goal) -> RobotState | None:
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

    def _robot_state_msg_from_positions(self, joint_positions: np.ndarray) -> RobotStateMsg:
        msg = RobotStateMsg()
        msg.joint_state = JointState()
        msg.joint_state.name = list(ARM_JOINT_NAMES)
        msg.joint_state.position = [float(value) for value in joint_positions]
        return msg

    def _motion_plan_request_for_positions(self, *, goal_positions: np.ndarray, planner_id: str) -> MotionPlanRequest:
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

    def _robot_trajectory_from_msg(self, start_state: RobotState, trajectory_msg: RobotTrajectoryMsg) -> RobotTrajectory:
        trajectory = RobotTrajectory(self.robot_model)
        trajectory.set_robot_trajectory_msg(start_state, trajectory_msg)
        return trajectory

    def _build_planning_scene(self, scene_objects: tuple[PrimitiveSpec, ...], *, tube_root_w: PoseWxyz) -> None:
        for primitive in scene_objects:
            self._apply_world_primitive(primitive, label=f"scene:add:{primitive.object_id}")
        self._apply_world_primitive(
            tube_world_primitive(tube_root_w=tube_root_w, robot_pose_w=self.task.robot_pose_w),
            label="scene:add:tube",
        )

    def _apply_world_primitive(self, primitive: PrimitiveSpec, *, label: str) -> None:
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
        self.completed_stages.append(label)

    def _remove_world_object(self, object_id: str, *, label: str) -> None:
        msg = CollisionObject()
        msg.header.frame_id = self.args.planning_frame
        msg.id = object_id
        msg.operation = CollisionObject.REMOVE
        with self.scene_monitor.read_write() as scene_rw:
            scene_rw.apply_collision_object(msg)
        self._sleep(SCENE_UPDATE_WAIT_S)
        self.completed_stages.append(label)

    def _remove_fixture_collision_objects(self, *, object_ids: tuple[str, ...] = FIXTURE_COLLISION_OBJECT_IDS, label: str) -> None:
        for object_id in object_ids:
            self._remove_world_object(object_id, label=f"{label}:{object_id}")

    def _restore_fixture_collision_objects(self, *, object_ids: tuple[str, ...] = FIXTURE_COLLISION_OBJECT_IDS, label: str) -> None:
        primitives_by_id = {primitive.object_id: primitive for primitive in self.task.scene_objects}
        for object_id in object_ids:
            primitive = primitives_by_id.get(object_id)
            if primitive is not None:
                self._apply_world_primitive(primitive, label=f"{label}:{object_id}")

    def _attach_tube_to_ee(self, *, source_tube_root_w: PoseWxyz, grasp_w: PoseWxyz, label: str) -> None:
        tube_center_w = tube_center_pose(source_tube_root_w)
        relative_center = _relative_pose(grasp_w, tube_center_w)
        tube_primitive = tube_world_primitive(tube_root_w=source_tube_root_w, robot_pose_w=self.task.robot_pose_w)
        self._remove_world_object("tube", label=f"{label}:remove_world")
        attached = AttachedCollisionObject()
        attached.link_name = self.args.ee_link
        attached.touch_links = list(GRIPPER_TOUCH_LINKS)
        attached.object.header.frame_id = self.args.ee_link
        attached.object.id = "tube"
        shape = SolidPrimitive()
        shape.type = SolidPrimitive.CYLINDER
        shape.dimensions = list(tube_primitive.dimensions)
        attached.object.primitives.append(shape)
        attached.object.primitive_poses.append(_pose_to_msg(relative_center))
        attached.object.operation = CollisionObject.ADD
        self.attached_pub.publish(attached)
        self._sleep(SCENE_UPDATE_WAIT_S)
        self.completed_stages.append(label)

    def _detach_tube_from_ee(self, *, label: str) -> None:
        attached = AttachedCollisionObject()
        attached.link_name = self.args.ee_link
        attached.object.id = "tube"
        attached.object.operation = CollisionObject.REMOVE
        self.attached_pub.publish(attached)
        self._sleep(SCENE_UPDATE_WAIT_S)
        self.completed_stages.append(label)

    def _publish_gripper(self, positions: list[float], *, label: str) -> float:
        msg = JointTrajectory()
        msg.joint_names = GRIPPER_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        secs = int(GRIPPER_MOVE_DURATION_S)
        point.time_from_start.sec = secs
        point.time_from_start.nanosec = int((GRIPPER_MOVE_DURATION_S - secs) * 1e9)
        msg.points = [point]
        stage_stamp = self._publish_control(stage=label, leg_label=self._leg_label_from_stage(label))
        stamp = time.time()
        self.gripper_pub.publish(msg)
        return max(stage_stamp, stamp)

    def _open_gripper(self, label: str) -> float:
        stamp = self._publish_gripper(GRIPPER_OPEN, label=label)
        status = self._wait_for_status(
            since_stamp=stamp,
            phases={PHASE_GRIPPER_OPEN, PHASE_ERROR},
            timeout_s=self.args.executor_phase_timeout_s,
            leg_label=self._leg_label_from_stage(label),
        )
        if not status.ok:
            raise RuntimeError(status.reason or f"{label}: open failed")
        self.completed_stages.append(label)
        return stamp

    def _close_gripper(self, label: str) -> float:
        stamp = self._publish_gripper(GRIPPER_CLOSED, label=label)
        self.completed_stages.append(label)
        return stamp

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live MoveIt runner for the wet-lab benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task_mode", type=str.lower, default="sample", choices=["sample", "a", "b"])
    parser.add_argument("--max_legs", type=int, default=0, help="Limit executed legs for targeted debugging; 0 runs all legs.")
    parser.add_argument("--arm_group", type=str, default=ARM_GROUP)
    parser.add_argument("--ee_link", type=str, default=EE_LINK)
    parser.add_argument("--planning_frame", type=str, default=PLANNING_FRAME)
    parser.add_argument("--trajectory_action", type=str, default=ARM_TRAJECTORY_ACTION)
    parser.add_argument("--gripper_topic", type=str, default=GRIPPER_COMMAND_TOPIC)
    parser.add_argument("--status_topic", type=str, default=EXECUTOR_STATUS_TOPIC)
    parser.add_argument("--control_topic", type=str, default=EXECUTOR_CONTROL_TOPIC)
    parser.add_argument("--joint_state_topic", type=str, default=JOINT_STATE_TOPIC)
    parser.add_argument("--executor_ready_timeout_s", type=float, default=EXECUTOR_READY_TIMEOUT_S)
    parser.add_argument("--executor_phase_timeout_s", type=float, default=EXECUTOR_PHASE_TIMEOUT_S)
    parser.add_argument("--use_sequence_service", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sequence_service_wait_s", type=float, default=2.0)
    parser.add_argument("--grasp_mode", type=str, choices=["physical", "fixed_joint"], default="physical")
    parser.add_argument("--grasp_tool_clearance_m", type=float, default=GRASP_TOOL_CLEARANCE_M)
    parser.add_argument("--verify_lift_height_m", type=float, default=0.03)
    parser.add_argument("--vortexer_place_clearance_m", type=float, default=None)
    parser.add_argument("--arm_trajectory_time_scale", type=float, default=1.0)
    parser.add_argument("--pickup_grasp_bias_x_m", type=float, default=PICKUP_GRASP_BIAS_LOCAL_M[0])
    parser.add_argument("--pickup_grasp_bias_y_m", type=float, default=PICKUP_GRASP_BIAS_LOCAL_M[1])
    parser.add_argument("--pickup_grasp_bias_z_m", type=float, default=PICKUP_GRASP_BIAS_LOCAL_M[2])
    parser.add_argument("--vortexer_pickup_grasp_bias_x_m", type=float, default=0.0)
    parser.add_argument("--vortexer_pickup_grasp_bias_y_m", type=float, default=0.0)
    parser.add_argument("--vortexer_pickup_grasp_bias_z_m", type=float, default=0.0)
    parser.add_argument("--vortexer_place_bias_x_m", type=float, default=0.0)
    parser.add_argument("--vortexer_place_bias_y_m", type=float, default=0.0)
    parser.add_argument("--vortexer_place_bias_z_m", type=float, default=0.0)
    parser.add_argument(
        "--post_grasp_settle_s",
        type=float,
        default=1.0,
        help="Optional dwell after Isaac reports physical grasp secure, before sending the first lift trajectory.",
    )
    parser.add_argument("--trajectory_cache_dir", type=str, default="")
    parser.add_argument(
        "--trajectory_cache_mode",
        type=str,
        choices=["off", "write", "read", "readwrite"],
        default="off",
        help="Optional on-disk cache for executed arm trajectories and sequence manifests.",
    )
    parser.add_argument("--remove_fixture_collision_during_carry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--moveit_controllers_name", type=str, default="fake_controllers")
    parser.add_argument("--robot_dof", type=int, default=6)
    parser.add_argument("--robot_type", type=str, default="xarm")
    parser.add_argument("--ros2_control_plugin", type=str, default="uf_robot_hardware/UFRobotFakeSystemHardware")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = LiveMoveItRunner(args)
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
