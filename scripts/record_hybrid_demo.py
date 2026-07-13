#!/usr/bin/env python3
"""Record a physics-driven rollout of the hybrid tube placement controller."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import torch

from isaaclab.app import AppLauncher

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a physics-driven tube placement demo.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--render_every", type=int, default=2)
    parser.add_argument("--video_dir", type=str, default="outputs/demo_videos")
    parser.add_argument("--camera_view", choices=("close", "wide"), default="close")
    parser.add_argument("--checkpoint", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from contact_valid_place_rl.tasks.direct.tube_place.tube_place_env import PHASE_SETTLE, TubePlaceEnv
from contact_valid_place_rl.tasks.direct.tube_place.agents.rsl_rl_ppo_cfg import TubePlacePPORunnerCfg
from contact_valid_place_rl.tasks.direct.tube_place.tube_place_env_cfg import TubePlaceEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    cfg = TubePlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.enable_reset_randomization = args_cli.checkpoint is not None
    rollout_seconds = args_cli.steps * cfg.sim.dt * cfg.decimation
    cfg.episode_length_s = max(cfg.episode_length_s, rollout_seconds + 2.0)

    env = TubePlaceEnv(cfg, render_mode="rgb_array")
    camera_views = {
        "close": ((0.76, -0.50, 1.08), (0.43, 0.01, 0.88)),
        "wide": ((1.70, -1.60, 1.65), (0.08, 0.00, 0.98)),
    }
    camera_eye, camera_target = camera_views[args_cli.camera_view]
    env.sim.set_camera_view(eye=camera_eye, target=camera_target)
    out_dir = Path(args_cli.video_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "rl-video-episode-0.mp4"

    obs, info = env.reset()
    actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    policy = None
    wrapped_env = None
    if args_cli.checkpoint is not None:
        agent_cfg = TubePlacePPORunnerCfg()
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(Path(args_cli.checkpoint).expanduser().resolve()))
        policy = runner.get_inference_policy(device=env.device)
        obs = wrapped_env.get_observations()
    finger_joint_ids = env.robot.find_joints([".*finger.*"])[0]
    previous_phase = int(env.phase[0].item())
    settle_frames = 0 
    fps = round(1.0 / (cfg.sim.dt * cfg.decimation * args_cli.render_every))
    print(f"reset_ok policy_shape={obs['policy'].shape} fps={fps}", flush=True)

    with imageio.get_writer(video_path, fps=fps) as writer, torch.inference_mode():
        for step in range(args_cli.steps):
            if policy is None:
                obs, rew, terminated, truncated, info = env.step(actions)
            else:
                actions = policy(obs)
                obs, rew, dones, info = wrapped_env.step(actions)
                terminated = dones
                truncated = torch.zeros_like(dones)
            phase = int(env.phase[0].item())

            if step % args_cli.render_every == 0:
                writer.append_data(env.render())

            if phase != previous_phase or step % 120 == 0:
                ee_pos = env.robot.data.body_pose_w[0, env.end_effector_body_id, :3]
                tube_pos = env.tube.data.root_pos_w[0]
                gripper_pos = env.robot.data.joint_pos[0, env.gripper_joint_ids]
                finger_pos = env.robot.data.joint_pos[0, finger_joint_ids]
                left_force, right_force = env._finger_tube_contact_forces()
                left_net = torch.linalg.norm(env.left_finger_contact.data.net_forces_w[0]).amax()
                right_net = torch.linalg.norm(env.right_finger_contact.data.net_forces_w[0]).amax()
                support_force = env._tube_vortexer_contact_force()[0]
                print(
                    f"step={step} phase={phase} reward={float(rew[0].item()):.4f} "
                    f"ee={ee_pos.tolist()} tube={tube_pos.tolist()} gripper={gripper_pos.tolist()} "
                    f"finger_joints={finger_pos.tolist()} "
                    f"finger_force=({float(left_force[0].item()):.3f},{float(right_force[0].item()):.3f}) "
                    f"finger_net=({float(left_net.item()):.3f},{float(right_net.item()):.3f}) "
                    f"vortexer_force={float(support_force.item()):.3f} "
                    f"held={bool(env._tube_held()[0].item())} inserted={bool(env._tube_inserted()[0].item())}",
                    flush=True,
                )
                previous_phase = phase

            if phase == PHASE_SETTLE and bool((env._tube_inserted() & env._tube_stable())[0].item()):
                settle_frames += 1
                if settle_frames >= 120:
                    print(f"physics_demo_success step={step}", flush=True)
                    break
            else:
                settle_frames = 0

            if bool(terminated[0].item()) or bool(truncated[0].item()):
                print(f"physics_demo_failed step={step} phase={phase}", flush=True)
                break

    env.close()
    print(f"video_path={video_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
