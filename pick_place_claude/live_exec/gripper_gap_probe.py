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

parser = argparse.ArgumentParser(description="Measure refined gripper pad motion across close targets.")
parser.add_argument("--asset_profile", type=str, choices=["contact_refined", "imported"], default="contact_refined")
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--physics_dt", type=float, default=0.0025)
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--samples", type=int, default=18)
parser.add_argument(
    "--control_mode",
    type=str,
    choices=["drive", "finger_pair", "all", "right_negative_all", "left_negative_all", "finger_pair_opposed"],
    default="drive",
)
parser.add_argument("--write_gripper_state", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.managers import SceneEntityCfg

from wetlab_benchmark.pick_place_claude.runtime import CONTACT_PHYSICS_DT, create_pick_place_runtime
from wetlab_benchmark.task_config import FRAMES, GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M, PICK_PLACE_RANDOMIZATION


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
CONTROLLED_GRIPPER_JOINT_NAMES = [
    "drive_joint",
    FRAMES.left_finger_joint,
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    FRAMES.right_finger_joint,
    "right_inner_knuckle_joint",
]
FINGER_PAIR_JOINT_NAMES = [FRAMES.left_finger_joint, FRAMES.right_finger_joint]
DRIVE_JOINT_NAMES = ["drive_joint"]
LEFT_PAD_LOCAL_M = (0.0, -GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)
RIGHT_PAD_LOCAL_M = (0.0, GRIPPER_PAD_CONTACT_LOCAL_Y_M, GRIPPER_PAD_CONTACT_LOCAL_Z_M)


def _pose_from_tensor(pose: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    return (
        (float(pose[0].item()), float(pose[1].item()), float(pose[2].item())),
        (float(pose[3].item()), float(pose[4].item()), float(pose[5].item()), float(pose[6].item())),
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


def _body_local_point_w(runtime, body_id: int, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    pos, quat = _pose_from_tensor(runtime.robot.data.body_pose_w[0, body_id])
    offset = _quat_apply(quat, local_xyz)
    return (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])


def _distance(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


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
    if args_cli.control_mode == "drive":
        control_joint_names = DRIVE_JOINT_NAMES
    elif args_cli.control_mode == "finger_pair":
        control_joint_names = FINGER_PAIR_JOINT_NAMES
    elif args_cli.control_mode == "finger_pair_opposed":
        control_joint_names = FINGER_PAIR_JOINT_NAMES
    else:
        control_joint_names = CONTROLLED_GRIPPER_JOINT_NAMES

    gripper_cfg = SceneEntityCfg("robot", joint_names=control_joint_names)
    gripper_cfg.resolve(runtime.scene)
    all_gripper_cfg = SceneEntityCfg("robot", joint_names=CONTROLLED_GRIPPER_JOINT_NAMES)
    all_gripper_cfg.resolve(runtime.scene)
    left_finger_cfg = SceneEntityCfg("robot", body_names=["left_finger"])
    left_finger_cfg.resolve(runtime.scene)
    right_finger_cfg = SceneEntityCfg("robot", body_names=["right_finger"])
    right_finger_cfg.resolve(runtime.scene)

    arm_ids = arm_cfg.joint_ids
    gripper_ids = gripper_cfg.joint_ids
    all_gripper_ids = all_gripper_cfg.joint_ids
    left_body = left_finger_cfg.body_ids[0]
    right_body = right_finger_cfg.body_ids[0]
    arm_hold = runtime.robot.data.joint_pos[:, arm_ids].clone()

    results = []
    sample_count = max(args_cli.samples, 2)
    for sample in range(sample_count + 1):
        target_scalar = FRAMES.gripper_closed_pos * sample / sample_count
        if args_cli.control_mode == "right_negative_all":
            values = [target_scalar, target_scalar, target_scalar, -target_scalar, -target_scalar, -target_scalar]
        elif args_cli.control_mode == "left_negative_all":
            values = [-target_scalar, -target_scalar, -target_scalar, target_scalar, target_scalar, target_scalar]
        elif args_cli.control_mode == "finger_pair_opposed":
            values = [target_scalar, -target_scalar]
        else:
            values = [target_scalar] * len(control_joint_names)
        target = torch.tensor([values], device=runtime.device, dtype=torch.float32)
        for _ in range(max(args_cli.settle_steps, 1)):
            _step(runtime, arm_hold, target, arm_ids=arm_ids, gripper_ids=gripper_ids)
        left_w = _body_local_point_w(runtime, left_body, LEFT_PAD_LOCAL_M)
        right_w = _body_local_point_w(runtime, right_body, RIGHT_PAD_LOCAL_M)
        joint_pos = runtime.robot.data.joint_pos[:, all_gripper_ids].detach().cpu().tolist()[0]
        results.append(
            {
                "target_scalar": target_scalar,
                "control_mode": args_cli.control_mode,
                "control_joint_names": control_joint_names,
                "joint_positions": dict(zip(CONTROLLED_GRIPPER_JOINT_NAMES, joint_pos, strict=True)),
                "left_pad_w": left_w,
                "right_pad_w": right_w,
                "pad_center_gap_m": _distance(left_w, right_w),
            }
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    output_path.write_text(
        json.dumps(
            {
                "asset_profile": args_cli.asset_profile,
                "left_pad_local_m": LEFT_PAD_LOCAL_M,
                "right_pad_local_m": RIGHT_PAD_LOCAL_M,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
