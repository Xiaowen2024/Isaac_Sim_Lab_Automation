#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep physical gripper/tube contact offsets in Isaac.")
parser.add_argument("--asset_profile", type=str, choices=["contact_refined", "imported"], default="contact_refined")
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--physics_dt", type=float, default=0.0025)
parser.add_argument("--contact_force_threshold_n", type=float, default=0.05)
parser.add_argument("--close_steps", type=int, default=220)
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--lift_steps", type=int, default=80)
parser.add_argument("--lift_delta_m", type=float, default=0.03)
parser.add_argument("--contact_overdrive_rad", type=float, default=0.02)
parser.add_argument("--spawn_alignment", type=str, choices=["open_pad_mid", "closed_pad_mid"], default="closed_pad_mid")
parser.add_argument("--closed_alignment_steps", type=int, default=40)
parser.add_argument(
    "--write_gripper_state",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Kinematically write gripper joint state while closing. Required for Isaac 4.5 xArm gripper mimic stability.",
)
parser.add_argument("--support_mode", type=str, choices=["holder_slot", "flat_table"], default="holder_slot")
parser.add_argument("--holder_slot_index", type=int, default=1)
parser.add_argument("--lateral_offsets_m", type=str, default="-0.030,-0.020,-0.010,0.000,0.010,0.020,0.030")
parser.add_argument("--vertical_offsets_m", type=str, default="-0.015,0.000,0.015")
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

from wetlab_benchmark.pick_place_claude_v4.live_exec.task_builder import (
    GRASP_TOOL_CLEARANCE_M,
    PoseWxyz,
    TOP_GRASP_QUAT_WXYZ,
    TUBE_CENTER_LOCAL_M,
    TUBE_SUPPORT_LOCAL_M,
    TUBE_TOP_LOCAL_M,
)
from wetlab_benchmark.pick_place_claude_v4.runtime import (
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
REFINED_LEFT_FILTERS = (0, 2, 4, 6, 8)
REFINED_RIGHT_FILTERS = (1, 3, 5, 7, 9)
IMPORTED_LEFT_FILTERS = (0, 2, 4)
IMPORTED_RIGHT_FILTERS = (1, 3, 5)
TABLE_THICKNESS_M = 0.04
HOLDER_ROOT_HEIGHT_FROM_SURFACE_M = next(
    fixture.root_height_from_surface_m for fixture in PICK_PLACE_RANDOMIZATION.fixtures if fixture.name == "holder"
)
LEFT_PAD_LOCAL_M = (0.0, -GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)
RIGHT_PAD_LOCAL_M = (0.0, GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)
TUBE_PROBE_GRIP_LOCAL_M = (
    TUBE_CENTER_LOCAL_M[0],
    TUBE_CENTER_LOCAL_M[1],
    TUBE_TOP_LOCAL_M[2] + GRASP_TOOL_CLEARANCE_M,
)


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


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


def _quat_conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def _quat_apply(q, v):
    return _quat_mul(_quat_mul(q, (0.0, *v)), _quat_conjugate(q))[1:4]


def _pose_world_point(pose_w: PoseWxyz, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    offset = _quat_apply(pose_w.quat_wxyz, local_xyz)
    return (pose_w.x + offset[0], pose_w.y + offset[1], pose_w.z + offset[2])


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
    return {"support_mode": args_cli.support_mode}


def _body_local_point_w(runtime, body_id: int, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return _pose_world_point(_pose_from_tensor(runtime.robot.data.body_pose_w[0, body_id]), local_xyz)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v, s: float):
    return (v[0] * s, v[1] * s, v[2] * s)


def _norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v):
    length = max(_norm(v), 1.0e-9)
    return (v[0] / length, v[1] / length, v[2] / length)


def _midpoint(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _distance(a, b) -> float:
    return _norm(_sub(a, b))


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


def _write_pose(asset, pose: PoseWxyz, *, device: str, zero_velocity: bool = False) -> None:
    tensor = torch.tensor([[pose.x, pose.y, pose.z, pose.qw, pose.qx, pose.qy, pose.qz]], device=device, dtype=torch.float32)
    asset.write_root_pose_to_sim(tensor)
    if zero_velocity:
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


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


def _trial(runtime, *, params: dict, ids: dict, targets: dict, filters: dict, base: dict) -> dict:
    arm_ids = ids["arm"]
    gripper_ids = ids["gripper"]
    arm_hold = runtime.robot.data.joint_pos[:, arm_ids].clone()
    open_target = targets["open"]
    close_target = targets["close"]

    _write_pose(runtime.robot, base["robot_root"], device=runtime.device)
    for _ in range(max(args_cli.settle_steps, 1)):
        _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)

    left_open = _body_local_point_w(runtime, ids["left_finger_body"], LEFT_PAD_LOCAL_M)
    right_open = _body_local_point_w(runtime, ids["right_finger_body"], RIGHT_PAD_LOCAL_M)
    alignment_mid = _midpoint(left_open, right_open)
    pad_axis = _unit(_sub(right_open, left_open))
    if args_cli.spawn_alignment == "closed_pad_mid":
        alignment_steps = max(args_cli.closed_alignment_steps, 1)
        for _ in range(alignment_steps):
            _step(runtime, arm_hold, close_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        left_closed = _body_local_point_w(runtime, ids["left_finger_body"], LEFT_PAD_LOCAL_M)
        right_closed = _body_local_point_w(runtime, ids["right_finger_body"], RIGHT_PAD_LOCAL_M)
        alignment_mid = _midpoint(left_closed, right_closed)
        pad_axis = _unit(_sub(right_closed, left_closed))
        for _ in range(alignment_steps):
            _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
    grip_center = _add(
        _add(alignment_mid, _scale(pad_axis, params["lateral_offset_m"])),
        (0.0, 0.0, params["vertical_offset_m"]),
    )
    tube_root = _tube_root_pose_from_grip_center(grip_center, base["tool_quat_wxyz"])
    support_offset = _quat_apply(tube_root.quat_wxyz, TUBE_SUPPORT_LOCAL_M)
    support_pos = (tube_root.x + support_offset[0], tube_root.y + support_offset[1], tube_root.z + support_offset[2])
    support_info = _place_support(runtime, support_pos)
    _write_pose(runtime.tube, tube_root, device=runtime.device, zero_velocity=True)
    for _ in range(max(args_cli.settle_steps, 1)):
        _step(runtime, arm_hold, open_target, arm_ids=arm_ids, gripper_ids=gripper_ids)

    first_left = None
    first_right = None
    first_bilateral = None
    max_force = 0.0
    hold_target = None
    for step in range(max(args_cli.close_steps, 1)):
        if hold_target is None:
            scalar = FRAMES.gripper_closed_pos * min((step + 1) / max(args_cli.close_steps, 1), 1.0)
            gripper_target = torch.full_like(close_target, scalar)
        else:
            gripper_target = hold_target
            scalar = float(hold_target[0, 0].item())
        _step(runtime, arm_hold, gripper_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        left, right, force = _contacts(
            runtime,
            left_filters=filters["left"],
            right_filters=filters["right"],
            threshold=args_cli.contact_force_threshold_n,
        )
        max_force = max(max_force, force)
        if left and first_left is None:
            first_left = scalar
        if right and first_right is None:
            first_right = scalar
        if left and right and first_bilateral is None:
            first_bilateral = scalar
            hold_value = min(scalar + max(args_cli.contact_overdrive_rad, 0.0), FRAMES.gripper_closed_pos)
            hold_target = torch.full_like(close_target, hold_value)

    z_before = float(runtime.tube.data.root_pose_w[0, 2].item())
    if hold_target is not None:
        for step in range(max(args_cli.lift_steps, 1)):
            alpha = min((step + 1) / max(args_cli.lift_steps, 1), 1.0)
            lifted_root = PoseWxyz(
                x=base["robot_root"].x,
                y=base["robot_root"].y,
                z=base["robot_root"].z + args_cli.lift_delta_m * alpha,
                qw=base["robot_root"].qw,
                qx=base["robot_root"].qx,
                qy=base["robot_root"].qy,
                qz=base["robot_root"].qz,
            )
            _write_pose(runtime.robot, lifted_root, device=runtime.device)
            _step(runtime, arm_hold, hold_target, arm_ids=arm_ids, gripper_ids=gripper_ids)
    z_after = float(runtime.tube.data.root_pose_w[0, 2].item())
    left_final, right_final, force_final = _contacts(
        runtime,
        left_filters=filters["left"],
        right_filters=filters["right"],
        threshold=args_cli.contact_force_threshold_n,
    )
    max_force = max(max_force, force_final)
    left_final_w = _body_local_point_w(runtime, ids["left_finger_body"], LEFT_PAD_LOCAL_M)
    right_final_w = _body_local_point_w(runtime, ids["right_finger_body"], RIGHT_PAD_LOCAL_M)
    tube_final = _pose_from_tensor(runtime.tube.data.root_pose_w[0])
    tube_grip_final = _pose_world_point(tube_final, TUBE_PROBE_GRIP_LOCAL_M)
    lift_delta = z_after - z_before
    ok = bool(first_bilateral is not None and lift_delta >= args_cli.lift_delta_m * 0.8)

    return {
        **params,
        **support_info,
        "ok": ok,
        "first_left_target": first_left,
        "first_right_target": first_right,
        "first_bilateral_target": first_bilateral,
        "hold_target": float(hold_target[0, 0].item()) if hold_target is not None else None,
        "left_final": left_final,
        "right_final": right_final,
        "max_contact_force_n": max_force,
        "tube_root_lift_delta_m": lift_delta,
        "pad_gap_open_m": _distance(left_open, right_open),
        "pad_gap_final_m": _distance(left_final_w, right_final_w),
        "tube_to_left_pad_final_m": _distance(tube_grip_final, left_final_w),
        "tube_to_right_pad_final_m": _distance(tube_grip_final, right_final_w),
    }


def main() -> int:
    output_path = Path(args_cli.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = create_pick_place_runtime(
        num_envs=1,
        seed=0,
        device=args_cli.device,
        camera_eye=[1.38, 0.24, 1.08],
        camera_target=[0.35, 0.0, 0.58],
        dt=args_cli.physics_dt or CONTACT_PHYSICS_DT,
        contact_physics=True,
        task_cfg=PICK_PLACE_RANDOMIZATION,
        asset_profile=args_cli.asset_profile,
    )
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(runtime.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=CONTROLLED_GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(runtime.scene)
    left_finger_cfg = SceneEntityCfg("robot", body_names=["left_finger"])
    left_finger_cfg.resolve(runtime.scene)
    right_finger_cfg = SceneEntityCfg("robot", body_names=["right_finger"])
    right_finger_cfg.resolve(runtime.scene)

    tool_asset = RobotAsset(runtime.robot, ee_body_id=runtime.robot_entity_cfg.body_ids[0], ee_local_offset_m=FRAMES.ee_tool_offset_local_m)
    base = {
        "robot_root": _pose_from_tensor(runtime.robot.data.root_pose_w[0]),
        "tool_quat_wxyz": _pose_from_tensor(tool_asset.ee_pose_w[0]).quat_wxyz,
    }
    ids = {
        "arm": arm_cfg.joint_ids,
        "gripper": gripper_cfg.joint_ids,
        "left_finger_body": left_finger_cfg.body_ids[0],
        "right_finger_body": right_finger_cfg.body_ids[0],
    }
    targets = {
        "open": torch.zeros((1, len(CONTROLLED_GRIPPER_JOINT_NAMES)), device=runtime.device),
        "close": torch.full((1, len(CONTROLLED_GRIPPER_JOINT_NAMES)), FRAMES.gripper_closed_pos, device=runtime.device),
    }
    filters = {
        "left": REFINED_LEFT_FILTERS if args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED else IMPORTED_LEFT_FILTERS,
        "right": REFINED_RIGHT_FILTERS if args_cli.asset_profile == ASSET_PROFILE_CONTACT_REFINED else IMPORTED_RIGHT_FILTERS,
    }

    results = []
    for z_offset in _parse_float_list(args_cli.vertical_offsets_m):
        for lateral_offset in _parse_float_list(args_cli.lateral_offsets_m):
            result = _trial(
                runtime,
                params={"lateral_offset_m": lateral_offset, "vertical_offset_m": z_offset},
                ids=ids,
                targets=targets,
                filters=filters,
                base=base,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    ok_results = [result for result in results if result["ok"]]
    best = None
    if ok_results:
        best = max(ok_results, key=lambda item: item["tube_root_lift_delta_m"])
    payload = {
        "ok": best is not None,
        "best": best,
        "asset_profile": args_cli.asset_profile,
        "support_mode": args_cli.support_mode,
        "holder_slot_index": args_cli.holder_slot_index,
        "pad_local_left_m": LEFT_PAD_LOCAL_M,
        "pad_local_right_m": RIGHT_PAD_LOCAL_M,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if best is not None else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if args_cli.close_app:
            simulation_app.close()
