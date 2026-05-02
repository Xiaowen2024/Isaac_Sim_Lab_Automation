#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Physical gripper/tube contact probe for live execution assets.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task_mode", type=str.lower, default="b", choices=["a", "b", "sample"])
parser.add_argument("--asset_profile", type=str, choices=["contact_refined", "imported"], default="contact_refined")
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--physics_dt", type=float, default=0.0025)
parser.add_argument("--contact_force_threshold_n", type=float, default=0.05)
parser.add_argument("--approach_steps", type=int, default=900)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--close_steps", type=int, default=900)
parser.add_argument("--lift_steps", type=int, default=900)
parser.add_argument("--lift_delta_m", type=float, default=0.03)
parser.add_argument("--lift_method", type=str, choices=["arm_ik", "robot_root_pose"], default="arm_ik")
parser.add_argument("--contact_overdrive_rad", type=float, default=0.02)
parser.add_argument("--contact_overdrive_ramp_steps", type=int, default=160)
parser.add_argument("--support_mode", type=str, choices=["holder_slot", "vortexer", "flat_table"], default="holder_slot")
parser.add_argument("--holder_slot_index", type=int, default=1)
parser.add_argument("--release_after_lift", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--release_steps", type=int, default=120)
parser.add_argument("--post_release_settle_steps", type=int, default=360)
parser.add_argument("--settle_xy_tol_m", type=float, default=0.025)
parser.add_argument("--settle_z_tol_m", type=float, default=0.05)
parser.add_argument("--upright_tol_deg", type=float, default=12.0)
parser.add_argument("--spawn_alignment", type=str, choices=["open_pad_mid", "closed_pad_mid"], default="closed_pad_mid")
parser.add_argument("--closed_alignment_steps", type=int, default=40)
parser.add_argument("--spawn_offset_x_m", type=float, default=0.0)
parser.add_argument("--spawn_offset_y_m", type=float, default=0.0)
parser.add_argument("--spawn_offset_z_m", type=float, default=0.0)
parser.add_argument("--recenter_after_spawn", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--progress_json", type=str, default=None)
parser.add_argument(
    "--write_gripper_state",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Kinematically write gripper joint state while closing. Required for Isaac 4.5 xArm gripper mimic stability.",
)
parser.add_argument(
    "--close_app",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Call simulation_app.close() before process exit. Disabled by default because Isaac 4.5 can hang here in headless probes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from wetlab_benchmark.pick_place_claude.live_exec.task_builder import (
    GRASP_TOOL_CLEARANCE_M,
    PoseWxyz,
    TOP_GRASP_QUAT_WXYZ,
    TUBE_CENTER_LOCAL_M,
    TUBE_SUPPORT_LOCAL_M,
    TUBE_TOP_LOCAL_M,
)
from wetlab_benchmark.pick_place_claude.runtime import (
    ASSET_PROFILE_CONTACT_REFINED,
    CONTACT_PHYSICS_DT,
    create_pick_place_runtime,
)
from wetlab_benchmark.task_config import (
    FRAMES,
    GRIPPER_PAD_CONTACT_LOCAL_Y_M,
    GRIPPER_PAD_CONTACT_LOCAL_Z_M,
    IMPORTED_LAB_ASSETS,
    PICK_PLACE_RANDOMIZATION,
)
from wetlab_benchmark.task_objects import RobotAsset
from wetlab_benchmark.validation import _contact_any_for_filters


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
CONTROLLED_GRIPPER_JOINT_NAMES = [
    "drive_joint",
    FRAMES.left_finger_joint,
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    FRAMES.right_finger_joint,
    "right_inner_knuckle_joint",
]
IMPORTED_LEFT_FILTERS = (0, 2, 4)
IMPORTED_RIGHT_FILTERS = (1, 3, 5)
# Option A: 10 refined finger paths (pad + v_groove*2 + x_stop*2 per side, alternating
# left/right) followed by 6 imported finger paths. Even = left, odd = right.
REFINED_LEFT_FILTERS = (0, 2, 4, 6, 8, 10, 12, 14)
REFINED_RIGHT_FILTERS = (1, 3, 5, 7, 9, 11, 13, 15)
TABLE_THICKNESS_M = 0.04
HOLDER_ROOT_HEIGHT_FROM_SURFACE_M = next(
    fixture.root_height_from_surface_m for fixture in PICK_PLACE_RANDOMIZATION.fixtures if fixture.name == "holder"
)
VORTEXER_ROOT_HEIGHT_FROM_SURFACE_M = next(
    fixture.root_height_from_surface_m for fixture in PICK_PLACE_RANDOMIZATION.fixtures if fixture.name == "vortexer"
)
LEFT_PAD_LOCAL_M = (0.0, -GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)
RIGHT_PAD_LOCAL_M = (0.0, GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)
TUBE_PROBE_GRIP_LOCAL_M = (
    TUBE_CENTER_LOCAL_M[0],
    TUBE_CENTER_LOCAL_M[1],
    TUBE_TOP_LOCAL_M[2] + GRASP_TOOL_CLEARANCE_M,
)


class ProbeProgress:
    def __init__(self, path: Path):
        self.path = path
        self.started_at = time.monotonic()
        self.events: list[dict] = []

    def mark(self, phase: str, extra: dict | None = None) -> None:
        event = {
            "phase": phase,
            "elapsed_s": round(time.monotonic() - self.started_at, 3),
        }
        if extra:
            event.update(extra)
        self.events.append(event)
        payload = {
            "last": event,
            "events": self.events,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[contact_probe] {event}", flush=True)


def _task_pose_to_tensor(task_pose, *, device: str) -> torch.Tensor:
    return torch.tensor(
        [[task_pose.x, task_pose.y, task_pose.z, task_pose.qw, task_pose.qx, task_pose.qy, task_pose.qz]],
        device=device,
        dtype=torch.float32,
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


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_apply(q, v):
    return _quat_mul(_quat_mul(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))[1:4]


def _quat_conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def _tube_root_pose_from_tool_pose(tool_pose_w: PoseWxyz, *, grasp_tool_clearance_m: float) -> PoseWxyz:
    # task_builder defines grasp_tool_w = tube_top_grasp_w offset along world Z.
    tube_top_pos_w = (
        tool_pose_w.x,
        tool_pose_w.y,
        tool_pose_w.z - grasp_tool_clearance_m,
    )
    tube_root_quat = _quat_mul(tool_pose_w.quat_wxyz, _quat_conjugate(TOP_GRASP_QUAT_WXYZ))
    top_offset_w = _quat_apply(tube_root_quat, TUBE_TOP_LOCAL_M)
    return PoseWxyz(
        x=tube_top_pos_w[0] - top_offset_w[0],
        y=tube_top_pos_w[1] - top_offset_w[1],
        z=tube_top_pos_w[2] - top_offset_w[2],
        qw=tube_root_quat[0],
        qx=tube_root_quat[1],
        qy=tube_root_quat[2],
        qz=tube_root_quat[3],
    )


def _tube_root_pose_from_grip_center(grip_center_w: tuple[float, float, float], tool_quat_wxyz) -> PoseWxyz:
    tube_root_quat = _quat_mul(tool_quat_wxyz, _quat_conjugate(TOP_GRASP_QUAT_WXYZ))
    grip_offset_w = _quat_apply(tube_root_quat, TUBE_PROBE_GRIP_LOCAL_M)
    return PoseWxyz(
        x=grip_center_w[0] - grip_offset_w[0],
        y=grip_center_w[1] - grip_offset_w[1],
        z=grip_center_w[2] - grip_offset_w[2],
        qw=tube_root_quat[0],
        qx=tube_root_quat[1],
        qy=tube_root_quat[2],
        qz=tube_root_quat[3],
    )


def _pose_world_point(pose_w: PoseWxyz, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    offset = _quat_apply(pose_w.quat_wxyz, local_xyz)
    return (pose_w.x + offset[0], pose_w.y + offset[1], pose_w.z + offset[2])


def _body_local_point_w(runtime, body_id: int, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return _pose_world_point(_pose_from_tensor(runtime.robot.data.body_pose_w[0, body_id]), local_xyz)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _midpoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _write_pose(asset, pose, *, device: str, zero_velocity: bool = False) -> None:
    tensor = torch.tensor(
        [[pose.x, pose.y, pose.z, pose.qw, pose.qx, pose.qy, pose.qz]],
        device=device,
        dtype=torch.float32,
    )
    asset.write_root_pose_to_sim(tensor)
    if zero_velocity:
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


def _root_pose_from_local_point(
    point_w: tuple[float, float, float],
    root_quat_wxyz: tuple[float, float, float, float],
    local_point_m: tuple[float, float, float],
) -> PoseWxyz:
    offset_w = _quat_apply(root_quat_wxyz, local_point_m)
    return PoseWxyz(
        x=point_w[0] - offset_w[0],
        y=point_w[1] - offset_w[1],
        z=point_w[2] - offset_w[2],
        qw=root_quat_wxyz[0],
        qx=root_quat_wxyz[1],
        qy=root_quat_wxyz[2],
        qz=root_quat_wxyz[3],
    )


def _place_support(runtime, support_pos_w: tuple[float, float, float]) -> dict:
    if args_cli.support_mode == "holder_slot":
        slot_index = max(0, min(args_cli.holder_slot_index, len(IMPORTED_LAB_ASSETS.holder_slot_centers_local_m) - 1))
        holder_pose = _root_pose_from_local_point(
            support_pos_w,
            IMPORTED_LAB_ASSETS.upright_quat_wxyz,
            IMPORTED_LAB_ASSETS.holder_slot_centers_local_m[slot_index],
        )
        surface_z = holder_pose.z - HOLDER_ROOT_HEIGHT_FROM_SURFACE_M
        table_pose = PoseWxyz(
            x=holder_pose.x,
            y=holder_pose.y,
            z=surface_z - 0.5 * TABLE_THICKNESS_M,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        )
        _write_pose(runtime.scene["table"], table_pose, device=runtime.device)
        _write_pose(runtime.holder, holder_pose, device=runtime.device)
        return {
            "support_mode": args_cli.support_mode,
            "holder_slot_index": slot_index,
            "holder_pose": holder_pose.to_dict(),
            "target_support_pos_w": support_pos_w,
        }

    if args_cli.support_mode == "vortexer":
        vortexer_pose = _root_pose_from_local_point(
            support_pos_w,
            IMPORTED_LAB_ASSETS.upright_quat_wxyz,
            IMPORTED_LAB_ASSETS.vortexer_support_center_local_m,
        )
        surface_z = vortexer_pose.z - VORTEXER_ROOT_HEIGHT_FROM_SURFACE_M
        table_pose = PoseWxyz(
            x=vortexer_pose.x,
            y=vortexer_pose.y,
            z=surface_z - 0.5 * TABLE_THICKNESS_M,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        )
        _write_pose(runtime.scene["table"], table_pose, device=runtime.device)
        _write_pose(runtime.vortexer, vortexer_pose, device=runtime.device)
        return {
            "support_mode": args_cli.support_mode,
            "vortexer_pose": vortexer_pose.to_dict(),
            "target_support_pos_w": support_pos_w,
        }

    table_pose = PoseWxyz(
        x=support_pos_w[0],
        y=support_pos_w[1],
        z=support_pos_w[2] - 0.5 * TABLE_THICKNESS_M,
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
    )
    _write_pose(runtime.scene["table"], table_pose, device=runtime.device)
    return {"support_mode": args_cli.support_mode, "target_support_pos_w": support_pos_w}


def _step(runtime, arm_target, gripper_target, *, arm_ids, gripper_ids) -> None:
    runtime.robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
    runtime.robot.set_joint_position_target(gripper_target, joint_ids=gripper_ids)
    if args_cli.write_gripper_state:
        runtime.robot.write_joint_state_to_sim(
            gripper_target,
            torch.zeros_like(gripper_target),
            joint_ids=gripper_ids,
        )
    runtime.scene.write_data_to_sim()
    runtime.sim.step()
    runtime.scene.update(runtime.sim.get_physics_dt())


def _contacts(runtime, *, left_filters, right_filters, threshold: float) -> tuple[bool, bool, float]:
    forces = torch.nan_to_num(runtime.tube_contacts.data.force_matrix_w)
    max_force = float(torch.max(torch.linalg.norm(forces, dim=-1)).item()) if forces.numel() else 0.0
    left = bool(_contact_any_for_filters(runtime.tube_contacts, left_filters, threshold)[0].item())
    right = bool(_contact_any_for_filters(runtime.tube_contacts, right_filters, threshold)[0].item())
    return left, right, max_force


def _tube_support_pose_w(tube_root_pose_w: PoseWxyz) -> tuple[float, float, float]:
    return _pose_world_point(tube_root_pose_w, TUBE_SUPPORT_LOCAL_M)


def _tube_up_axis_z(tube_root_pose_w: PoseWxyz) -> float:
    return float(_quat_apply(tube_root_pose_w.quat_wxyz, (0.0, 0.0, 1.0))[2])


def _drive_body_to_task_pose(runtime, target_task_pose, *, arm_ids, gripper_ids, gripper_target, steps: int) -> float:
    controller = runtime.controller
    robot = runtime.robot
    robot_entity_cfg = runtime.robot_entity_cfg
    ee_body = RobotAsset(robot, ee_body_id=runtime.robot_entity_cfg.body_ids[0], ee_local_offset_m=(0.0, 0.0, 0.0))
    target_w = _task_pose_to_tensor(target_task_pose, device=runtime.device)
    max_err = math.inf
    for _ in range(max(steps, 1)):
        jacobian = robot.root_physx_view.get_jacobians()[:, runtime.ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        root_pose_w = robot.data.root_pose_w
        ee_pose_w = ee_body.ee_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            target_w[:, 0:3],
            target_w[:, 3:7],
        )
        controller.set_command(target_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        joint_pos_des = joint_pos + torch.clamp(joint_pos_des - joint_pos, -0.02, 0.02)
        _step(runtime, joint_pos_des, gripper_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        max_err = float(torch.linalg.norm(ee_pose_w[:, 0:3] - target_w[:, 0:3], dim=-1)[0].item())
    return max_err


def main() -> int:
    output_path = Path(args_cli.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args_cli.progress_json).expanduser().resolve() if args_cli.progress_json else output_path.with_suffix(".progress.json")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress = ProbeProgress(progress_path)
    progress.mark(
        "main:start",
        {
            "asset_profile": args_cli.asset_profile,
            "support_mode": args_cli.support_mode,
            "settle_steps": args_cli.settle_steps,
            "close_steps": args_cli.close_steps,
            "lift_steps": args_cli.lift_steps,
        },
    )

    try:
        runtime = create_pick_place_runtime(
            num_envs=1,
            seed=args_cli.seed,
            device=args_cli.device,
            camera_eye=[1.38, 0.24, 1.08],
            camera_target=[0.35, 0.0, 0.58],
            dt=args_cli.physics_dt or CONTACT_PHYSICS_DT,
            contact_physics=True,
            task_cfg=PICK_PLACE_RANDOMIZATION,
            asset_profile=args_cli.asset_profile,
            progress_callback=progress.mark,
        )
    except Exception as exc:
        progress.mark("runtime:error", {"error": repr(exc)})
        raise
    progress.mark("runtime:done")

    progress.mark("resolve_entities:start")
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(runtime.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=CONTROLLED_GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(runtime.scene)
    left_finger_cfg = SceneEntityCfg("robot", body_names=["left_finger"])
    left_finger_cfg.resolve(runtime.scene)
    right_finger_cfg = SceneEntityCfg("robot", body_names=["right_finger"])
    right_finger_cfg.resolve(runtime.scene)
    progress.mark("resolve_entities:done")
    arm_ids = arm_cfg.joint_ids
    gripper_ids = gripper_cfg.joint_ids
    left_finger_body_id = left_finger_cfg.body_ids[0]
    right_finger_body_id = right_finger_cfg.body_ids[0]

    open_target = torch.zeros((1, len(CONTROLLED_GRIPPER_JOINT_NAMES)), device=runtime.device)
    close_target = torch.full((1, len(CONTROLLED_GRIPPER_JOINT_NAMES)), FRAMES.gripper_closed_pos, device=runtime.device)
    left_filters = REFINED_LEFT_FILTERS if args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED else IMPORTED_LEFT_FILTERS
    right_filters = REFINED_RIGHT_FILTERS if args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED else IMPORTED_RIGHT_FILTERS

    arm_hold = runtime.robot.data.joint_pos[:, arm_ids].clone()
    settle_steps = max(args_cli.settle_steps, 1)
    progress.mark("initial_settle:start", {"steps": settle_steps})
    for step in range(settle_steps):
        _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        if step == 0 or step + 1 == settle_steps or (step + 1) % max(settle_steps // 5, 1) == 0:
            progress.mark("initial_settle:step", {"step": step + 1, "steps": settle_steps})
    progress.mark("initial_settle:done")

    progress.mark("compute_spawn_pose:start")
    tool_asset = RobotAsset(
        runtime.robot,
        ee_body_id=runtime.robot_entity_cfg.body_ids[0],
        ee_local_offset_m=FRAMES.ee_tool_offset_local_m,
    )
    tool_pose_w = _pose_from_tensor(tool_asset.ee_pose_w[0])
    left_pad_open_w = _body_local_point_w(runtime, left_finger_body_id, LEFT_PAD_LOCAL_M)
    right_pad_open_w = _body_local_point_w(runtime, right_finger_body_id, RIGHT_PAD_LOCAL_M)
    pad_mid_open_w = _midpoint(left_pad_open_w, right_pad_open_w)
    alignment_pad_mid_w = pad_mid_open_w
    closed_pad_mid_w = None
    closed_pad_shift_w = (0.0, 0.0, 0.0)
    if args_cli.spawn_alignment == "closed_pad_mid":
        alignment_steps = max(args_cli.closed_alignment_steps, 1)
        progress.mark("closed_alignment:start", {"steps": alignment_steps})
        for _ in range(alignment_steps):
            _step(runtime, arm_hold, close_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        left_pad_closed_w = _body_local_point_w(runtime, left_finger_body_id, LEFT_PAD_LOCAL_M)
        right_pad_closed_w = _body_local_point_w(runtime, right_finger_body_id, RIGHT_PAD_LOCAL_M)
        closed_pad_mid_w = _midpoint(left_pad_closed_w, right_pad_closed_w)
        closed_pad_shift_w = tuple(closed_pad_mid_w[i] - pad_mid_open_w[i] for i in range(3))
        for _ in range(alignment_steps):
            _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        alignment_pad_mid_w = closed_pad_mid_w
        progress.mark(
            "closed_alignment:done",
            {
                "closed_pad_mid_w": closed_pad_mid_w,
                "closed_pad_shift_w": closed_pad_shift_w,
            },
        )
    spawn_offset_w = (
        float(args_cli.spawn_offset_x_m),
        float(args_cli.spawn_offset_y_m),
        float(args_cli.spawn_offset_z_m),
    )
    alignment_pad_mid_w = tuple(alignment_pad_mid_w[i] + spawn_offset_w[i] for i in range(3))
    tube_root_pose_w = _tube_root_pose_from_grip_center(alignment_pad_mid_w, tool_pose_w.quat_wxyz)
    tube_support_offset_w = _quat_apply(tube_root_pose_w.quat_wxyz, TUBE_SUPPORT_LOCAL_M)
    tube_support_pos_w = (
        tube_root_pose_w.x + tube_support_offset_w[0],
        tube_root_pose_w.y + tube_support_offset_w[1],
        tube_root_pose_w.z + tube_support_offset_w[2],
    )
    progress.mark("compute_spawn_pose:done")

    progress.mark("place_support:start")
    support_info = _place_support(runtime, tube_support_pos_w)
    progress.mark("place_support:done", support_info)

    progress.mark("write_tube:start")
    _write_pose(runtime.tube, tube_root_pose_w, device=runtime.device, zero_velocity=True)
    progress.mark("write_tube:done")

    spawn_settle_steps = max(args_cli.settle_steps // 2, 1)
    progress.mark("spawn_settle:start", {"steps": spawn_settle_steps})
    for step in range(spawn_settle_steps):
        _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        if step == 0 or step + 1 == spawn_settle_steps or (step + 1) % max(spawn_settle_steps // 5, 1) == 0:
            progress.mark("spawn_settle:step", {"step": step + 1, "steps": spawn_settle_steps})
    progress.mark("spawn_settle:done")

    progress.mark("measure_spawn:start")
    tube_root_after_spawn = _pose_from_tensor(runtime.tube.data.root_pose_w[0])
    tube_grip_center_after_spawn = _pose_world_point(tube_root_after_spawn, TUBE_PROBE_GRIP_LOCAL_M)
    left_pad_after_spawn_w = _body_local_point_w(runtime, left_finger_body_id, LEFT_PAD_LOCAL_M)
    right_pad_after_spawn_w = _body_local_point_w(runtime, right_finger_body_id, RIGHT_PAD_LOCAL_M)
    pad_mid_after_spawn_w = _midpoint(left_pad_after_spawn_w, right_pad_after_spawn_w)
    approach_err = _distance(alignment_pad_mid_w, tube_grip_center_after_spawn)
    approach_err_open = _distance(pad_mid_after_spawn_w, tube_grip_center_after_spawn)
    progress.mark(
        "measure_spawn:done",
        {"approach_err_m": approach_err, "approach_err_open_pad_m": approach_err_open},
    )
    recenter_offset_w = (0.0, 0.0, 0.0)
    if args_cli.recenter_after_spawn:
        recenter_offset_w = tuple(tube_grip_center_after_spawn[i] - alignment_pad_mid_w[i] for i in range(3))
        robot_root = _pose_from_tensor(runtime.robot.data.root_pose_w[0])
        _write_pose(
            runtime.robot,
            PoseWxyz(
                x=robot_root.x + recenter_offset_w[0],
                y=robot_root.y + recenter_offset_w[1],
                z=robot_root.z + recenter_offset_w[2],
                qw=robot_root.qw,
                qx=robot_root.qx,
                qy=robot_root.qy,
                qz=robot_root.qz,
            ),
            device=runtime.device,
        )
        for _ in range(max(args_cli.settle_steps // 2, 1)):
            _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        left_pad_after_spawn_w = _body_local_point_w(runtime, left_finger_body_id, LEFT_PAD_LOCAL_M)
        right_pad_after_spawn_w = _body_local_point_w(runtime, right_finger_body_id, RIGHT_PAD_LOCAL_M)
        pad_mid_after_spawn_w = _midpoint(left_pad_after_spawn_w, right_pad_after_spawn_w)
        alignment_pad_mid_w = tuple(alignment_pad_mid_w[i] + recenter_offset_w[i] for i in range(3))
        tube_root_after_spawn = _pose_from_tensor(runtime.tube.data.root_pose_w[0])
        tube_grip_center_after_spawn = _pose_world_point(tube_root_after_spawn, TUBE_PROBE_GRIP_LOCAL_M)
        approach_err = _distance(alignment_pad_mid_w, tube_grip_center_after_spawn)
        approach_err_open = _distance(pad_mid_after_spawn_w, tube_grip_center_after_spawn)
        arm_hold = runtime.robot.data.joint_pos[:, arm_ids].clone()
        progress.mark(
            "recenter_after_spawn:done",
            {
                "recenter_offset_w": recenter_offset_w,
                "approach_err_m": approach_err,
                "approach_err_open_pad_m": approach_err_open,
            },
        )
    arm_hold = runtime.robot.data.joint_pos[:, arm_ids].clone()
    contact_seen = {"left": False, "right": False}
    max_force = 0.0
    contact_hold_target = None
    contact_hold_source = None
    contact_hold_start_step = 0
    close_steps = max(args_cli.close_steps, 1)
    progress.mark("close:start", {"steps": close_steps})
    for step in range(close_steps):
        if contact_hold_target is None:
            alpha = min((step + 1) / close_steps, 1.0)
            gripper_target = open_target * (1.0 - alpha) + close_target * alpha
        else:
            ramp_steps = max(args_cli.contact_overdrive_ramp_steps, 1)
            ramp_alpha = min(max((step - contact_hold_start_step + 1) / ramp_steps, 0.0), 1.0)
            gripper_target = contact_hold_source * (1.0 - ramp_alpha) + contact_hold_target * ramp_alpha
        _step(runtime, arm_hold, gripper_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        left, right, force = _contacts(runtime, left_filters=left_filters, right_filters=right_filters, threshold=args_cli.contact_force_threshold_n)
        contact_seen["left"] = contact_seen["left"] or left
        contact_seen["right"] = contact_seen["right"] or right
        max_force = max(max_force, force)
        if left and right and contact_hold_target is None:
            hold_value = min(float(gripper_target[0, 0].item()) + max(args_cli.contact_overdrive_rad, 0.0), FRAMES.gripper_closed_pos)
            contact_hold_target = torch.full_like(close_target, hold_value)
            contact_hold_source = gripper_target.clone()
            contact_hold_start_step = step
            progress.mark(
                "close:bilateral_contact",
                {
                    "step": step + 1,
                    "hold_value": hold_value,
                    "ramp_steps": max(args_cli.contact_overdrive_ramp_steps, 1),
                    "max_force_n": max_force,
                },
            )
        if step == 0 or step + 1 == close_steps or (step + 1) % max(close_steps // 10, 1) == 0:
            progress.mark(
                "close:step",
                {
                    "step": step + 1,
                    "steps": close_steps,
                    "left_contact_seen": contact_seen["left"],
                    "right_contact_seen": contact_seen["right"],
                    "max_force_n": max_force,
                },
            )
    progress.mark(
        "close:done",
        {
            "left_contact_seen": contact_seen["left"],
            "right_contact_seen": contact_seen["right"],
            "has_hold_target": contact_hold_target is not None,
            "max_force_n": max_force,
        },
    )

    support_z_before = float(runtime.tube.data.root_pose_w[0, 2].item())
    robot_root_start = _pose_from_tensor(runtime.robot.data.root_pose_w[0])
    lift_gripper_target = contact_hold_target if contact_hold_target is not None else close_target
    lift_steps = max(args_cli.lift_steps, 1)
    progress.mark("lift:start", {"steps": lift_steps})
    if args_cli.lift_method == "arm_ik":
        ee_body_pose = _pose_from_tensor(runtime.robot.data.body_pose_w[0, runtime.robot_entity_cfg.body_ids[0]])
        lift_target = PoseWxyz(
            x=ee_body_pose.x,
            y=ee_body_pose.y,
            z=ee_body_pose.z + args_cli.lift_delta_m,
            qw=ee_body_pose.qw,
            qx=ee_body_pose.qx,
            qy=ee_body_pose.qy,
            qz=ee_body_pose.qz,
        )
        lift_err = _drive_body_to_task_pose(
            runtime,
            lift_target,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            gripper_target=lift_gripper_target,
            steps=lift_steps,
        )
        progress.mark("lift:arm_ik_done", {"steps": lift_steps, "ee_position_err_m": lift_err})
    else:
        for step in range(lift_steps):
            alpha = min((step + 1) / lift_steps, 1.0)
            lifted_robot_root = PoseWxyz(
                x=robot_root_start.x,
                y=robot_root_start.y,
                z=robot_root_start.z + args_cli.lift_delta_m * alpha,
                qw=robot_root_start.qw,
                qx=robot_root_start.qx,
                qy=robot_root_start.qy,
                qz=robot_root_start.qz,
            )
            _write_pose(runtime.robot, lifted_robot_root, device=runtime.device)
            _step(runtime, arm_hold, lift_gripper_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
            if step == 0 or step + 1 == lift_steps or (step + 1) % max(lift_steps // 5, 1) == 0:
                progress.mark("lift:step", {"step": step + 1, "steps": lift_steps})
        lift_err = abs((robot_root_start.z + args_cli.lift_delta_m) - _pose_from_tensor(runtime.robot.data.root_pose_w[0]).z)
    progress.mark("lift:done")

    progress.mark("measure_final:start")
    support_z_after = float(runtime.tube.data.root_pose_w[0, 2].item())
    lift_delta = support_z_after - support_z_before
    left_now, right_now, force_now = _contacts(
        runtime,
        left_filters=left_filters,
        right_filters=right_filters,
        threshold=args_cli.contact_force_threshold_n,
    )
    left_pad_final_w = _body_local_point_w(runtime, left_finger_body_id, LEFT_PAD_LOCAL_M)
    right_pad_final_w = _body_local_point_w(runtime, right_finger_body_id, RIGHT_PAD_LOCAL_M)
    tube_root_final = _pose_from_tensor(runtime.tube.data.root_pose_w[0])
    tube_grip_center_final_w = _pose_world_point(tube_root_final, TUBE_PROBE_GRIP_LOCAL_M)
    max_force = max(max_force, force_now)
    grasp_ok = bool(contact_hold_target is not None and left_now and right_now and lift_delta >= args_cli.lift_delta_m * 0.8)
    reason = "ok" if grasp_ok else (
        f"left_contact_seen={contact_seen['left']} right_contact_seen={contact_seen['right']} "
        f"lift_delta_m={lift_delta:.4f}"
    )
    release_result = None
    if args_cli.release_after_lift:
        release_steps = max(args_cli.release_steps, 1)
        progress.mark("release:start", {"steps": release_steps})
        for step in range(release_steps):
            alpha = min((step + 1) / release_steps, 1.0)
            gripper_target = lift_gripper_target * (1.0 - alpha) + open_target * alpha
            _step(runtime, arm_hold, gripper_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
            if step == 0 or step + 1 == release_steps or (step + 1) % max(release_steps // 5, 1) == 0:
                progress.mark("release:step", {"step": step + 1, "steps": release_steps})
        progress.mark("release:done")

        settle_steps = max(args_cli.post_release_settle_steps, 1)
        progress.mark("post_release_settle:start", {"steps": settle_steps})
        for step in range(settle_steps):
            _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
            if step == 0 or step + 1 == settle_steps or (step + 1) % max(settle_steps // 5, 1) == 0:
                progress.mark("post_release_settle:step", {"step": step + 1, "steps": settle_steps})
        progress.mark("post_release_settle:done")

        tube_root_release = _pose_from_tensor(runtime.tube.data.root_pose_w[0])
        support_pos_release = _tube_support_pose_w(tube_root_release)
        target_support_pos_w = tuple(float(value) for value in support_info["target_support_pos_w"])
        planar_error = math.hypot(
            support_pos_release[0] - target_support_pos_w[0],
            support_pos_release[1] - target_support_pos_w[1],
        )
        z_error = abs(support_pos_release[2] - target_support_pos_w[2])
        up_axis_z_release = _tube_up_axis_z(tube_root_release)
        upright_cos_min = math.cos(math.radians(args_cli.upright_tol_deg))
        release_ok = bool(
            planar_error <= args_cli.settle_xy_tol_m
            and z_error <= args_cli.settle_z_tol_m
            and up_axis_z_release >= upright_cos_min
        )
        release_result = {
            "ok": release_ok,
            "support_pos_w": support_pos_release,
            "target_support_pos_w": target_support_pos_w,
            "planar_error_m": planar_error,
            "support_z_error_m": z_error,
            "tube_up_axis_z": up_axis_z_release,
        }
        if not release_ok:
            reason = (
                f"release_planar_error_m={planar_error:.4f} "
                f"release_z_error_m={z_error:.4f} "
                f"release_up_axis_z={up_axis_z_release:.4f}"
            )
    ok = grasp_ok and (release_result is None or release_result["ok"])
    payload = {
        "ok": ok,
        "reason": reason,
        "asset_profile": args_cli.asset_profile,
        "probe_mode": "spawn_tube_in_open_gripper",
        **support_info,
        "lift_method": args_cli.lift_method,
        "approach_err_m": approach_err,
        "approach_err_open_pad_m": approach_err_open,
        "spawn_alignment": args_cli.spawn_alignment,
        "spawn_offset_w": spawn_offset_w,
        "closed_pad_mid_w": closed_pad_mid_w,
        "closed_pad_shift_w": closed_pad_shift_w,
        "recenter_after_spawn": args_cli.recenter_after_spawn,
        "recenter_offset_w": recenter_offset_w,
        "lift_err_m": lift_err,
        "left_contact_seen": contact_seen["left"],
        "right_contact_seen": contact_seen["right"],
        "left_contact_final": left_now,
        "right_contact_final": right_now,
        "contact_hold_target": contact_hold_target.detach().cpu().tolist()[0] if contact_hold_target is not None else None,
        "left_pad_open_w": left_pad_open_w,
        "right_pad_open_w": right_pad_open_w,
        "pad_gap_open_m": _distance(left_pad_open_w, right_pad_open_w),
        "left_pad_final_w": left_pad_final_w,
        "right_pad_final_w": right_pad_final_w,
        "pad_gap_final_m": _distance(left_pad_final_w, right_pad_final_w),
        "tube_grip_center_after_spawn_w": tube_grip_center_after_spawn,
        "tube_grip_center_final_w": tube_grip_center_final_w,
        "tube_to_left_pad_final_m": _distance(tube_grip_center_final_w, left_pad_final_w),
        "tube_to_right_pad_final_m": _distance(tube_grip_center_final_w, right_pad_final_w),
        "max_contact_force_n": max_force,
        "spawned_tube_root_pose": tube_root_pose_w.to_dict(),
        "tube_root_z_before": support_z_before,
        "tube_root_z_after": support_z_after,
        "tube_root_lift_delta_m": lift_delta,
        "grasp_ok": grasp_ok,
        "release_after_lift": args_cli.release_after_lift,
        "release_result": release_result,
    }
    progress.mark("measure_final:done", {"ok": ok, "reason": reason})
    progress.mark("write_output:start")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    progress.mark("write_output:done", {"ok": ok})
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if args_cli.close_app:
            simulation_app.close()
