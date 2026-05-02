#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render a successful no-video physical trace without replaying physics.")
parser.add_argument("--input_dir", type=str, required=True, help="Run directory containing render_run.json and state trace JSONL.")
parser.add_argument("--output_dir", type=str, default="", help="Optional output directory; defaults to <input_dir>/trace_render.")
parser.add_argument("--trace_path", type=str, default="", help="Optional explicit state trace JSONL path.")
parser.add_argument("--asset_profile", type=str, choices=["contact_refined", "imported"], default="contact_refined")
parser.add_argument("--camera_preset", type=str, default="")
parser.add_argument("--camera_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--camera_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--video_fps", type=float, default=8.0)
parser.add_argument("--video_name", type=str, default="render_trace.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from wetlab_benchmark.pick_place_claude_v4.frames_to_mp4 import encode_png_sequence_to_mp4
from wetlab_benchmark.pick_place_claude_v4.live_exec.protocol import camera_from_preset
from wetlab_benchmark.pick_place_claude_v4.live_exec.task_builder import LiveTask, PoseWxyz, build_live_task
from wetlab_benchmark.pick_place_claude_v4.runtime import (
    ASSET_PROFILE_CONTACT_REFINED,
    ContactRefinedPickPlaceSceneCfg,
    PickPlaceSceneCfg,
    SCENE_DOME_LIGHT_COLOR,
    SCENE_DOME_LIGHT_INTENSITY,
    create_simulation,
    initialize_pick_place_runtime,
)
from wetlab_benchmark.task_config import PICK_PLACE_RANDOMIZATION


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
CONTROLLED_GRIPPER_JOINT_NAMES = [
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
]


def _pose_tensor(pose: PoseWxyz, *, device: str) -> torch.Tensor:
    return torch.tensor(
        [[pose.x, pose.y, pose.z, pose.qw, pose.qx, pose.qy, pose.qz]],
        device=device,
        dtype=torch.float32,
    )


def _write_rigid_pose(asset, pose: PoseWxyz, *, device: str, zero_velocity: bool = False) -> None:
    asset.write_root_pose_to_sim(_pose_tensor(pose, device=device))
    if zero_velocity:
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


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


def _load_trace(trace_path: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if not samples:
        raise RuntimeError(f"No samples found in trace {trace_path}")
    return samples


def main() -> None:
    input_dir = Path(args_cli.input_dir).expanduser().resolve()
    metadata_path = input_dir / "render_run.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    trace_path = (
        Path(args_cli.trace_path).expanduser().resolve()
        if args_cli.trace_path
        else (
            Path(metadata["state_trace_path"]).expanduser().resolve()
            if metadata.get("state_trace_path")
            else input_dir / "state_trace.jsonl"
        )
    )
    if not trace_path.exists():
        raise RuntimeError(f"Missing state trace: {trace_path}")

    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else input_dir / "trace_render"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    seed = int(metadata["seed"])
    task_mode = str(metadata["task_mode"])
    camera_preset = args_cli.camera_preset or str(metadata.get("camera_preset") or "side_wide")
    camera_cfg = camera_from_preset(
        preset_name=camera_preset,
        eye_override=tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else None,
        target_override=tuple(args_cli.camera_target) if args_cli.camera_target is not None else None,
    )
    live_task: LiveTask = build_live_task(seed, task_mode)

    sim = create_simulation(
        device=args_cli.device,
        camera_eye=list(camera_cfg.eye),
        camera_target=list(camera_cfg.target),
        dt=0.0025,
        contact_physics=True,
    )
    scene = InteractiveScene(SceneCfgWithCamera(num_envs=1, env_spacing=2.5))
    sim.reset()
    scene.reset()
    runtime = initialize_pick_place_runtime(
        sim=sim,
        scene=scene,
        seed=seed,
        task_cfg=PICK_PLACE_RANDOMIZATION,
        asset_profile=args_cli.asset_profile,
    )
    camera = scene["obs_camera"]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([camera_cfg.eye], device=runtime.device, dtype=torch.float32),
        targets=torch.tensor([camera_cfg.target], device=runtime.device, dtype=torch.float32),
    )

    _write_rigid_pose(runtime.robot, live_task.robot_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.holder, live_task.holder_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.vortexer, live_task.vortexer_pose_w, device=runtime.device)
    _write_rigid_pose(runtime.tube, live_task.initial_tube_root_w, device=runtime.device, zero_velocity=True)
    scene.write_data_to_sim()
    scene.update(0.0)

    arm_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_entity_cfg.resolve(scene)
    gripper_entity_cfg = SceneEntityCfg("robot", joint_names=CONTROLLED_GRIPPER_JOINT_NAMES)
    gripper_entity_cfg.resolve(scene)
    arm_joint_ids = arm_entity_cfg.joint_ids
    gripper_joint_ids = gripper_entity_cfg.joint_ids
    zero_arm_vel = torch.zeros((1, len(arm_joint_ids)), device=runtime.device, dtype=torch.float32)
    zero_gripper_vel = torch.zeros((1, len(gripper_joint_ids)), device=runtime.device, dtype=torch.float32)

    samples = _load_trace(trace_path)
    for frame_index, sample in enumerate(samples):
        arm_pos = torch.tensor([sample["arm_joint_pos"]], device=runtime.device, dtype=torch.float32)
        gripper_pos = torch.tensor([sample["gripper_joint_pos"]], device=runtime.device, dtype=torch.float32)
        tube_pose = PoseWxyz.from_dict(sample["tube_root_pose"])

        runtime.robot.write_joint_state_to_sim(arm_pos, zero_arm_vel, joint_ids=arm_joint_ids)
        runtime.robot.write_joint_state_to_sim(gripper_pos, zero_gripper_vel, joint_ids=gripper_joint_ids)
        runtime.robot.set_joint_position_target(arm_pos, joint_ids=arm_joint_ids)
        runtime.robot.set_joint_position_target(gripper_pos, joint_ids=gripper_joint_ids)
        _write_rigid_pose(runtime.tube, tube_pose, device=runtime.device, zero_velocity=True)
        scene.write_data_to_sim()
        scene.update(0.0)
        sim.render()
        camera.update(0.0)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
        Image.fromarray(rgb).save(str(frames_dir / f"frame_{frame_index:06d}.png"))

    video_path = output_dir / args_cli.video_name
    encoded_frames = encode_png_sequence_to_mp4(
        frames_dir=frames_dir,
        output_path=video_path,
        fps=max(args_cli.video_fps, 1.0),
    )
    replay_metadata = {
        "source_run_dir": str(input_dir),
        "source_trace_path": str(trace_path),
        "seed": seed,
        "task_mode": task_mode,
        "camera_preset": camera_preset,
        "video_path": str(video_path),
        "captured_frames": len(samples),
        "encoded_video_frames": encoded_frames,
        "encoded_video_fps": args_cli.video_fps,
    }
    (output_dir / "render_replay.json").write_text(json.dumps(replay_metadata, indent=2), encoding="utf-8")
    print(f"[render_state_trace] video={video_path}")
    print(f"[render_state_trace] metadata={output_dir / 'render_replay.json'}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        error_dir = None
        try:
            error_dir = (
                Path(args_cli.output_dir).expanduser().resolve()
                if args_cli.output_dir
                else Path(args_cli.input_dir).expanduser().resolve()
            )
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "render_trace_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        raise
    finally:
        if getattr(args_cli, "headless", False):
            sys.stdout.flush()
            sys.stderr.flush()
            import os

            os._exit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
