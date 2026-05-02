#!/usr/bin/env python3
"""
Mirror the MoveIt pick-place execution inside Isaac Sim and save RGB frames.

This script does not run the Isaac-side FSM. Instead it:
- spawns the wet-lab scene in Isaac with the same seeded layout as MoveIt
- subscribes to MoveIt's `/joint_states` to mirror arm/gripper motion
- subscribes to MoveIt stage events to attach/release the tube deterministically
- writes RGB frames to disk for dataset generation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mirror MoveIt execution inside Isaac Sim and save RGB frames.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task_mode", type=str.lower, default="sample", choices=["sample", "a", "b"])
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--joint_state_topic", type=str, default="/joint_states")
parser.add_argument("--sync_events_topic", type=str, default="/wetlab_benchmark/moveit_events")
parser.add_argument("--timeout_s", type=float, default=120.0)
parser.add_argument("--frame_stride", type=int, default=1)
parser.add_argument("--post_summary_frames", type=int, default=15)
parser.add_argument("--video_fps", type=float, default=8.0)
parser.add_argument("--video_name", type=str, default="render.mp4")
parser.add_argument("--debug_materials", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--replay_lead_s", type=float, default=1.0)
parser.add_argument("--attach_distance_m", type=float, default=0.035)
parser.add_argument("--release_distance_m", type=float, default=0.045)
parser.add_argument("--camera_eye", type=float, nargs=3, default=(0.28, 0.96, 1.35), metavar=("X", "Y", "Z"))
parser.add_argument("--camera_target", type=float, nargs=3, default=(0.28, 0.0, 0.82), metavar=("X", "Y", "Z"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from PIL import Image
import rclpy
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from wetlab_benchmark.pick_place_claude_v4.pick_place_move_it import (
    GRIPPER_CLOSED,
    GRIPPER_MOVE_DURATION_S,
    GRIPPER_OPEN,
    PoseWxyz,
    TOP_GRASP_QUAT_WXYZ,
    TUBE_SUPPORT_LOCAL_M,
    _ee_pose_at_tube_top,
    _holder_slot_support_pose,
    _lookup_holder_slot_points,
    _quat_conjugate,
    _root_pose_from_support_point,
    _sample_fixture_layout,
    _task_mode_from_arg,
    _vortexer_support_pose,
)
from wetlab_benchmark.pick_place_claude_v4.frames_to_mp4 import encode_png_sequence_to_mp4
from wetlab_benchmark.pick_place_claude_v4.runtime import (
    PickPlaceSceneCfg,
    SCENE_DOME_LIGHT_COLOR,
    SCENE_DOME_LIGHT_INTENSITY,
    create_simulation,
    initialize_pick_place_runtime,
)
from wetlab_benchmark.task_objects import RobotAsset
from wetlab_benchmark.randomization import quat_apply, quat_mul
from wetlab_benchmark.task_config import FRAMES, IMPORTED_LAB_ASSETS, PICK_PLACE_RANDOMIZATION


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = [FRAMES.left_finger_joint, FRAMES.right_finger_joint]
TUBE_GRASP_LOCAL_M = (
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
    IMPORTED_LAB_ASSETS.tube_top_from_root_m,
)
VORTEXER_SETTLE_DROP_M = 0.055
SETTLE_DURATION_S = 0.45


@configclass
class SceneCfgWithCamera(PickPlaceSceneCfg):
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
        offset=CameraCfg.OffsetCfg(pos=args_cli.camera_eye, rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )


@dataclass(frozen=True)
class RenderLeg:
    label: str
    grasp_w: PoseWxyz
    place_w: PoseWxyz
    placed_tube_root_w: PoseWxyz
    settled_tube_root_w: PoseWxyz


@dataclass(frozen=True)
class RenderTask:
    seed: int
    task_mode: int
    holder_slot_index: int
    robot_pose_w: PoseWxyz
    holder_pose_w: PoseWxyz
    vortexer_pose_w: PoseWxyz
    initial_tube_root_w: PoseWxyz
    legs: tuple[RenderLeg, ...]


def _build_render_task(seed: int, mode_arg: str) -> RenderTask:
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

    def make_leg(
        *,
        label: str,
        source_support_w: PoseWxyz,
        dest_support_w: PoseWxyz,
        dest_fixture_w: PoseWxyz,
        settle_drop_m: float = 0.0,
    ) -> RenderLeg:
        placed_tube_root_w = _root_pose_from_support_point(
            dest_support_w.pos,
            dest_fixture_w.quat_wxyz,
            TUBE_SUPPORT_LOCAL_M,
        )
        return RenderLeg(
            label=label,
            grasp_w=_ee_pose_at_tube_top(source_support_w),
            place_w=_ee_pose_at_tube_top(dest_support_w),
            placed_tube_root_w=placed_tube_root_w,
            settled_tube_root_w=PoseWxyz(
                x=placed_tube_root_w.x,
                y=placed_tube_root_w.y,
                z=placed_tube_root_w.z - settle_drop_m,
                qw=placed_tube_root_w.qw,
                qx=placed_tube_root_w.qx,
                qy=placed_tube_root_w.qy,
                qz=placed_tube_root_w.qz,
            ),
        )

    legs: list[RenderLeg] = []
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
                settle_drop_m=VORTEXER_SETTLE_DROP_M,
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

    return RenderTask(
        seed=seed,
        task_mode=task_mode,
        holder_slot_index=holder_slot_index,
        robot_pose_w=robot_pose_w,
        holder_pose_w=holder_pose_w,
        vortexer_pose_w=vortexer_pose_w,
        initial_tube_root_w=initial_tube_root_w,
        legs=tuple(legs),
    )


def _pose_tensor(pose: PoseWxyz, *, device: str) -> torch.Tensor:
    return torch.tensor(
        [[pose.x, pose.y, pose.z, pose.qw, pose.qx, pose.qy, pose.qz]],
        device=device,
        dtype=torch.float32,
    )


def _bind_debug_material(
    prim_path: str,
    material_name: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.65,
    metallic: float = 0.0,
) -> None:
    material_path = f"/World/Looks/{material_name}"
    material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness, metallic=metallic)
    try:
        material_cfg.func(material_path, material_cfg)
    except ValueError:
        pass
    sim_utils.bind_visual_material(prim_path, material_path)


def _bind_component_material_recursive(
    prim_path: str,
    material_name: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.65,
    metallic: float = 0.0,
) -> None:
    import omni.usd
    from pxr import UsdGeom

    def _iter_descendants(root_prim):
        stack = [child for child in root_prim.GetChildren() if child.IsValid()]
        while stack:
            current = stack.pop()
            yield current
            children = [child for child in current.GetChildren() if child.IsValid()]
            stack.extend(reversed(children))

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        return

    gprim_paths: list[str] = []
    for descendant in _iter_descendants(prim):
        if descendant.IsA(UsdGeom.Gprim):
            gprim_paths.append(descendant.GetPath().pathString)

    if not gprim_paths and prim.IsA(UsdGeom.Gprim):
        gprim_paths.append(prim.GetPath().pathString)

    if not gprim_paths:
        _bind_debug_material(
            prim_path,
            material_name,
            color,
            roughness=roughness,
            metallic=metallic,
        )
        return

    for gprim_path in gprim_paths:
        _bind_debug_material(
            gprim_path,
            material_name,
            color,
            roughness=roughness,
            metallic=metallic,
        )


def _apply_debug_materials() -> None:
    # Keep the arm neutral and make the gripper/tube high-contrast so the pick
    # interaction remains readable even in sparse offline captures.
    _bind_component_material_recursive(
        "/World/envs/env_0/Robot",
        "RenderRobotBody",
        (0.78, 0.80, 0.82),
        roughness=0.42,
        metallic=0.30,
    )
    _bind_component_material_recursive(
        "/World/envs/env_0/Robot/left_finger",
        "RenderLeftFinger",
        (0.12, 0.64, 0.86),
        roughness=0.30,
        metallic=0.18,
    )
    _bind_component_material_recursive(
        "/World/envs/env_0/Robot/right_finger",
        "RenderRightFinger",
        (0.12, 0.64, 0.86),
        roughness=0.30,
        metallic=0.18,
    )
    _bind_component_material_recursive(
        "/World/envs/env_0/Robot/left_outer_knuckle",
        "RenderLeftKnuckle",
        (0.18, 0.26, 0.34),
        roughness=0.55,
        metallic=0.15,
    )
    _bind_component_material_recursive(
        "/World/envs/env_0/Robot/right_outer_knuckle",
        "RenderRightKnuckle",
        (0.18, 0.26, 0.34),
        roughness=0.55,
        metallic=0.15,
    )
    _bind_component_material_recursive(
        "/World/envs/env_0/Tube",
        "RenderTube",
        (0.90, 0.18, 0.14),
        roughness=0.36,
        metallic=0.04,
    )
    _bind_component_material_recursive(
        "/World/envs/env_0/TubeHolder",
        "RenderHolder",
        (0.22, 0.60, 0.30),
        roughness=0.72,
        metallic=0.06,
    )
    _bind_component_material_recursive(
        "/World/envs/env_0/Vortexer",
        "RenderVortexer",
        (0.30, 0.32, 0.36),
        roughness=0.44,
        metallic=0.26,
    )


def _write_rigid_pose(asset, pose: PoseWxyz, *, device: str, zero_velocity: bool = False) -> None:
    world_pose = _pose_tensor(pose, device=device)
    asset.write_root_pose_to_sim(world_pose)
    if zero_velocity:
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


def _leg_index_from_label(label: str) -> int | None:
    match = re.match(r"leg_(\d+):", label)
    return int(match.group(1)) if match else None


def _ee_pose_distance(ee_pose_w: torch.Tensor, goal_w: PoseWxyz) -> float:
    goal = torch.tensor([[goal_w.x, goal_w.y, goal_w.z]], device=ee_pose_w.device, dtype=ee_pose_w.dtype)
    return float(torch.linalg.norm(ee_pose_w[:, 0:3] - goal, dim=-1)[0].item())


def _tube_root_from_ee_pose(ee_pose_w: torch.Tensor) -> torch.Tensor:
    top_grasp_inv = torch.tensor(_quat_conjugate(TOP_GRASP_QUAT_WXYZ), device=ee_pose_w.device, dtype=ee_pose_w.dtype).repeat(
        ee_pose_w.shape[0], 1
    )
    local_grasp = torch.tensor(TUBE_GRASP_LOCAL_M, device=ee_pose_w.device, dtype=ee_pose_w.dtype).repeat(ee_pose_w.shape[0], 1)
    tube_quat = quat_mul(ee_pose_w[:, 3:7], top_grasp_inv)
    root_pos = ee_pose_w[:, 0:3] - quat_apply(tube_quat, local_grasp)
    return torch.cat((root_pos, tube_quat), dim=-1)


class MoveItMirrorNode(Node):
    def __init__(self, *, joint_state_topic: str, event_topic: str) -> None:
        super().__init__("moveit_isaac_render_bridge")
        self._lock = threading.Lock()
        self._latest_joint_positions: dict[str, float] = {}
        self._events: list[dict] = []
        self._summary_status: str | None = None
        self._joint_state_count = 0

        self.create_subscription(JointState, joint_state_topic, self._on_joint_state, 50)
        self.create_subscription(String, event_topic, self._on_event, 50)

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            for name, position in zip(msg.name, msg.position):
                self._latest_joint_positions[name] = float(position)
            self._joint_state_count += 1

    def _on_event(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning(f"Ignoring malformed event payload: {msg.data!r}")
            return

        label = str(payload.get("label", ""))
        status = str(payload.get("status", ""))
        kind = str(payload.get("kind", ""))

        with self._lock:
            self._events.append(payload)
            if kind == "summary":
                self._summary_status = status

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "joint_positions": dict(self._latest_joint_positions),
                "events": list(self._events),
                "summary_status": self._summary_status,
                "joint_state_count": self._joint_state_count,
            }

    def export_events(self) -> list[dict]:
        with self._lock:
            return list(self._events)


def _sorted_events(events: list[dict]) -> list[dict]:
    return sorted(events, key=lambda event: float(event.get("time", 0.0)))


def _first_motion_time(events: list[dict]) -> float | None:
    for event in _sorted_events(events):
        if event.get("kind") == "trajectory":
            return float(event.get("execution_start_time", event.get("time", 0.0)))
    for event in _sorted_events(events):
        if event.get("kind") != "stage" or event.get("status") != "START":
            continue
        label = str(event.get("label", ""))
        if label.endswith(":pregrasp"):
            return float(event["time"])
    return None


def _summary_time(events: list[dict], fallback_time: float) -> float:
    for event in reversed(_sorted_events(events)):
        if event.get("kind") == "summary":
            return float(event.get("time", fallback_time))
    return fallback_time


def _build_leg_windows(events: list[dict]) -> dict[int, dict[str, float]]:
    leg_windows: dict[int, dict[str, float]] = {}
    for event in _sorted_events(events):
        if event.get("kind") != "stage" or event.get("status") != "START":
            continue
        label = str(event.get("label", ""))
        leg_index = _leg_index_from_label(label)
        if leg_index is None:
            continue
        window = leg_windows.setdefault(leg_index, {})
        if label.endswith(":close_gripper"):
            window["attach_time"] = float(event["time"]) + GRIPPER_MOVE_DURATION_S
        elif label.endswith(":open_gripper"):
            window["release_time"] = float(event["time"]) + GRIPPER_MOVE_DURATION_S
    return leg_windows


def _build_arm_trajectory_samples(*, events: list[dict], device: str) -> tuple[list[float], torch.Tensor]:
    samples: list[tuple[float, list[float]]] = []
    for event in _sorted_events(events):
        if event.get("kind") != "trajectory":
            continue
        trajectory = event.get("trajectory") or {}
        joint_names = [str(name) for name in trajectory.get("joint_names", [])]
        if not joint_names:
            continue
        try:
            arm_joint_indices = [joint_names.index(name) for name in ARM_JOINT_NAMES]
        except ValueError:
            continue

        start_time = float(event.get("execution_start_time", event.get("time", 0.0)))
        for point in trajectory.get("points", []):
            positions = list(point.get("positions", []))
            if len(positions) != len(joint_names):
                continue
            samples.append(
                (
                    start_time + float(point.get("time_from_start_s", 0.0)),
                    [float(positions[index]) for index in arm_joint_indices],
                )
            )

    if not samples:
        raise RuntimeError("No MoveIt trajectory events were recorded from the execution run.")

    samples.sort(key=lambda item: item[0])
    sample_times: list[float] = []
    sample_positions: list[list[float]] = []
    for sample_time, sample_position in samples:
        if sample_times and abs(sample_time - sample_times[-1]) < 1.0e-6:
            sample_positions[-1] = sample_position
            continue
        sample_times.append(sample_time)
        sample_positions.append(sample_position)

    return (
        sample_times,
        torch.tensor(sample_positions, device=device, dtype=torch.float32),
    )


def _build_gripper_action_events(*, events: list[dict], device: str) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for event in _sorted_events(events):
        if event.get("kind") != "stage" or event.get("status") != "START":
            continue
        label = str(event.get("label", ""))
        if label == "initial_open" or label.endswith(":open_gripper"):
            target = torch.tensor(GRIPPER_OPEN, device=device, dtype=torch.float32)
        elif label.endswith(":close_gripper"):
            target = torch.tensor(GRIPPER_CLOSED, device=device, dtype=torch.float32)
        else:
            continue
        actions.append({"time": float(event.get("time", 0.0)), "target": target})
    return actions


def _gripper_state_for_time(query_t: float, *, actions: list[dict[str, object]], device: str) -> torch.Tensor:
    state = torch.tensor(GRIPPER_OPEN, device=device, dtype=torch.float32)
    for action in actions:
        start_time = float(action["time"])
        target = action["target"]
        if query_t < start_time:
            return state
        end_time = start_time + GRIPPER_MOVE_DURATION_S
        if query_t < end_time:
            alpha = (query_t - start_time) / GRIPPER_MOVE_DURATION_S
            return state * (1.0 - alpha) + target * alpha
        state = target
    return state


def _placed_tube_pose_for_time(
    query_t: float,
    *,
    render_task: RenderTask,
    leg_windows: dict[int, dict[str, float]],
) -> PoseWxyz:
    pose = render_task.initial_tube_root_w
    for leg_index, leg in enumerate(render_task.legs):
        release_time = leg_windows.get(leg_index, {}).get("release_time")
        if release_time is None or query_t < release_time:
            continue
        settle_end_time = release_time + SETTLE_DURATION_S
        if query_t >= settle_end_time:
            pose = leg.settled_tube_root_w
            continue
        alpha = max(0.0, min((query_t - release_time) / SETTLE_DURATION_S, 1.0))
        pose = PoseWxyz(
            x=leg.placed_tube_root_w.x + (leg.settled_tube_root_w.x - leg.placed_tube_root_w.x) * alpha,
            y=leg.placed_tube_root_w.y + (leg.settled_tube_root_w.y - leg.placed_tube_root_w.y) * alpha,
            z=leg.placed_tube_root_w.z + (leg.settled_tube_root_w.z - leg.placed_tube_root_w.z) * alpha,
            qw=leg.placed_tube_root_w.qw,
            qx=leg.placed_tube_root_w.qx,
            qy=leg.placed_tube_root_w.qy,
            qz=leg.placed_tube_root_w.qz,
        )
    return pose


def _attached_leg_for_time(query_t: float, *, leg_windows: dict[int, dict[str, float]]) -> int | None:
    for leg_index in sorted(leg_windows):
        attach_time = leg_windows[leg_index].get("attach_time")
        release_time = leg_windows[leg_index].get("release_time")
        if attach_time is None:
            continue
        if release_time is None and query_t >= attach_time:
            return leg_index
        if release_time is not None and attach_time <= query_t < release_time:
            return leg_index
    return None


def _interpolate_joint_positions(
    *,
    sample_times: list[float],
    sample_values: torch.Tensor,
    query_t: float,
    sample_index: int,
) -> tuple[torch.Tensor, int]:
    if query_t <= sample_times[0]:
        return sample_values[0], 0
    while sample_index + 1 < len(sample_times) and sample_times[sample_index + 1] < query_t:
        sample_index += 1
    if sample_index + 1 >= len(sample_times):
        return sample_values[-1], len(sample_times) - 1

    t0 = sample_times[sample_index]
    t1 = sample_times[sample_index + 1]
    if t1 <= t0:
        return sample_values[sample_index + 1], sample_index + 1

    alpha = (query_t - t0) / (t1 - t0)
    interpolated = sample_values[sample_index] * (1.0 - alpha) + sample_values[sample_index + 1] * alpha
    return interpolated, sample_index


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    render_task = _build_render_task(args_cli.seed, args_cli.task_mode)
    sim = create_simulation(
        device=args_cli.device,
        camera_eye=list(args_cli.camera_eye),
        camera_target=list(args_cli.camera_target),
    )
    scene = InteractiveScene(SceneCfgWithCamera(num_envs=1, env_spacing=2.5))

    sim.reset()
    scene.reset()
    runtime = initialize_pick_place_runtime(sim=sim, scene=scene, seed=args_cli.seed, task_cfg=PICK_PLACE_RANDOMIZATION)
    camera = scene["obs_camera"]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([args_cli.camera_eye], device=runtime.device, dtype=torch.float32),
        targets=torch.tensor([args_cli.camera_target], device=runtime.device, dtype=torch.float32),
    )

    # Override the randomized layout with the exact same seeded fixture poses that
    # the MoveIt runner uses.
    _write_rigid_pose(runtime.robot, render_task.robot_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.holder, render_task.holder_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.vortexer, render_task.vortexer_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.tube, render_task.initial_tube_root_w, device=runtime.device, zero_velocity=True)
    if args_cli.debug_materials:
        _apply_debug_materials()

    arm_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_entity_cfg.resolve(runtime.scene)
    gripper_entity_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_entity_cfg.resolve(runtime.scene)

    arm_joint_ids = arm_entity_cfg.joint_ids
    gripper_joint_ids = gripper_entity_cfg.joint_ids

    arm_targets = torch.zeros((1, len(ARM_JOINT_NAMES)), device=runtime.device, dtype=torch.float32)
    gripper_targets = torch.tensor([GRIPPER_OPEN], device=runtime.device, dtype=torch.float32)
    arm_velocities = torch.zeros_like(arm_targets)
    gripper_velocities = torch.zeros_like(gripper_targets)
    ee_tool_asset = RobotAsset(
        runtime.robot,
        ee_body_id=runtime.robot_entity_cfg.body_ids[0],
        ee_local_offset_m=FRAMES.ee_tool_offset_local_m,
    )

    rclpy.init()
    ros_node = MoveItMirrorNode(joint_state_topic=args_cli.joint_state_topic, event_topic=args_cli.sync_events_topic)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    summary_status: str | None = None
    record_start_time = time.monotonic()
    dt = runtime.sim.get_physics_dt()
    video_path = output_dir / args_cli.video_name

    metadata = {
        "seed": render_task.seed,
        "task_mode": render_task.task_mode,
        "holder_slot_index": render_task.holder_slot_index,
        "joint_state_topic": args_cli.joint_state_topic,
        "sync_events_topic": args_cli.sync_events_topic,
        "camera_eye": list(args_cli.camera_eye),
        "camera_target": list(args_cli.camera_target),
        "frame_stride": args_cli.frame_stride,
        "video_fps": args_cli.video_fps,
        "video_path": None,
        "attach_distance_m": args_cli.attach_distance_m,
        "release_distance_m": args_cli.release_distance_m,
        "robot_pose_w": asdict(render_task.robot_pose_w),
        "holder_pose_w": asdict(render_task.holder_pose_w),
        "vortexer_pose_w": asdict(render_task.vortexer_pose_w),
        "initial_tube_root_w": asdict(render_task.initial_tube_root_w),
        "legs": [
            {
                "label": leg.label,
                "grasp_w": asdict(leg.grasp_w),
                "place_w": asdict(leg.place_w),
                "placed_tube_root_w": asdict(leg.placed_tube_root_w),
                "settled_tube_root_w": asdict(leg.settled_tube_root_w),
            }
            for leg in render_task.legs
        ],
    }

    try:
        # Stage 1: record the MoveIt execution timeline from ROS.
        while True:
            snapshot = ros_node.snapshot()
            summary_status = snapshot["summary_status"] or summary_status
            if summary_status is not None:
                break
            if time.monotonic() - record_start_time > args_cli.timeout_s:
                summary_status = summary_status or "timeout"
                break
            time.sleep(0.05)

        capture_wall_s = time.monotonic() - record_start_time
        events = ros_node.export_events()
        sample_times, arm_samples = _build_arm_trajectory_samples(events=events, device=runtime.device)
        gripper_actions = _build_gripper_action_events(events=events, device=runtime.device)
        leg_windows = _build_leg_windows(events)

        first_motion_time = _first_motion_time(events) or sample_times[0]
        replay_start_time = first_motion_time - args_cli.replay_lead_s
        replay_end_time = max(_summary_time(events, sample_times[-1]), sample_times[-1], replay_start_time)
        encode_fps = args_cli.video_fps if args_cli.video_fps > 0 else 12.0
        replay_duration_s = max(replay_end_time - replay_start_time, 1.0 / encode_fps)
        replay_frame_count = max(int(math.ceil(replay_duration_s * encode_fps)) + args_cli.post_summary_frames, 1)
        replay_dt = 1.0 / encode_fps
        first_attach_time = min(
            (float(window["attach_time"]) for window in leg_windows.values() if "attach_time" in window),
            default=None,
        )

        frame_index = 0
        sample_index = 0
        zero_root_velocity = torch.zeros(1, 6, device=runtime.device)

        # Reset the scene before the offline replay pass so the video is not
        # tied to the slow live render loop or the live physics timeline.
        _write_rigid_pose(runtime.robot, render_task.robot_pose_w, device=runtime.device)
        _write_rigid_pose(runtime.holder, render_task.holder_pose_w, device=runtime.device)
        _write_rigid_pose(runtime.vortexer, render_task.vortexer_pose_w, device=runtime.device)
        _write_rigid_pose(runtime.tube, render_task.initial_tube_root_w, device=runtime.device, zero_velocity=True)

        for replay_frame_idx in range(replay_frame_count):
            replay_t = min(replay_start_time + replay_frame_idx * replay_dt, replay_end_time)
            arm_state, sample_index = _interpolate_joint_positions(
                sample_times=sample_times,
                sample_values=arm_samples,
                query_t=replay_t,
                sample_index=sample_index,
            )
            gripper_state = _gripper_state_for_time(
                replay_t,
                actions=gripper_actions,
                device=runtime.device,
            )

            arm_targets[0] = arm_state
            gripper_targets[0] = gripper_state

            runtime.robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
            runtime.robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
            runtime.robot.write_joint_state_to_sim(arm_targets, arm_velocities, joint_ids=arm_joint_ids)
            runtime.robot.write_joint_state_to_sim(gripper_targets, gripper_velocities, joint_ids=gripper_joint_ids)

            runtime.scene.write_data_to_sim()
            runtime.sim.step()
            runtime.scene.update(dt)

            attached_leg_index = _attached_leg_for_time(replay_t, leg_windows=leg_windows)
            if attached_leg_index is None:
                if first_attach_time is None or replay_t < first_attach_time:
                    _write_rigid_pose(
                        runtime.tube,
                        render_task.initial_tube_root_w,
                        device=runtime.device,
                        zero_velocity=True,
                    )
                else:
                    _write_rigid_pose(
                        runtime.tube,
                        _placed_tube_pose_for_time(
                            replay_t,
                            render_task=render_task,
                            leg_windows=leg_windows,
                        ),
                        device=runtime.device,
                        zero_velocity=True,
                    )
            else:
                runtime.tube.write_root_pose_to_sim(_tube_root_from_ee_pose(ee_tool_asset.ee_pose_w))
                runtime.tube.write_root_velocity_to_sim(zero_root_velocity)

            runtime.sim.render()
            camera.update(dt)

            if replay_frame_idx % args_cli.frame_stride == 0:
                rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
                frame_path = frames_dir / f"frame_{frame_index:06d}.png"
                Image.fromarray(rgb).save(str(frame_path))
                frame_index += 1

        metadata["summary_status"] = summary_status
        metadata["captured_frames"] = frame_index
        metadata["capture_wall_s"] = capture_wall_s
        metadata["joint_state_count"] = ros_node.snapshot()["joint_state_count"]
        metadata["recorded_trajectory_segments"] = sum(1 for event in events if event.get("kind") == "trajectory")
        metadata["recorded_arm_samples"] = len(sample_times)
        metadata["events"] = events
        metadata["replay_start_time"] = replay_start_time
        metadata["replay_end_time"] = replay_end_time
        metadata["replay_duration_s"] = replay_duration_s
        metadata["encoded_video_fps"] = encode_fps
        metadata["video_path"] = str(video_path)
        encoded_frames = encode_png_sequence_to_mp4(
            frames_dir=frames_dir,
            output_path=video_path,
            fps=encode_fps,
        )
        metadata["encoded_video_frames"] = encoded_frames
        metadata_path = output_dir / "render_run.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"[pick_place_moveit_render] frames_dir={frames_dir}")
        print(f"[pick_place_moveit_render] metadata={metadata_path}")
        print(f"[pick_place_moveit_render] video={video_path}")
        print(f"[pick_place_moveit_render] video_fps={encode_fps:.3f}")
        print(f"[pick_place_moveit_render] summary_status={summary_status}")
        print(f"[pick_place_moveit_render] frames={frame_index}")
    finally:
        executor.shutdown()
        ros_node.destroy_node()
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
