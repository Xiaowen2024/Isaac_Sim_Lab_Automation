#!/usr/bin/env python3
"""Run controller-independent tube drop probes against vortexer colliders."""

from __future__ import annotations

import argparse

import torch

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settle_steps", type=int, default=720)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from contact_valid_place_rl.tasks.direct.tube_place.tube_place_env import PHASE_RL_RELEASE, TubePlaceEnv
from contact_valid_place_rl.tasks.direct.tube_place.tube_place_env_cfg import TubePlaceEnvCfg


def set_tube_pose(env: TubePlaceEnv, xyz: tuple[float, float, float]) -> None:
    state = env.tube.data.default_root_state[:1].clone()
    state[:, :3] = torch.tensor(xyz, device=env.device)
    state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
    state[:, 7:] = 0.0
    env.tube.write_root_state_to_sim(state)


def settle(env: TubePlaceEnv) -> tuple[list[float], float, float]:
    for _ in range(args_cli.settle_steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    pos = env.tube.data.root_pos_w[0]
    lin_speed = torch.linalg.norm(env.tube.data.root_lin_vel_w[0])
    ang_speed = torch.linalg.norm(env.tube.data.root_ang_vel_w[0])
    return pos.tolist(), float(lin_speed.item()), float(ang_speed.item())


def main() -> None:
    cfg = TubePlaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60.0
    cfg.enable_reset_randomization = False
    env = TubePlaceEnv(cfg)
    env.reset()

    robot_state = env.robot.data.default_root_state[:1].clone()
    robot_state[:, :3] = torch.tensor((-2.0, -2.0, 0.8), device=env.device)
    env.robot.write_root_state_to_sim(robot_state)

    center_x = cfg.vortexer_body.init_state.pos[0]
    center_y = cfg.vortexer_body.init_state.pos[1]
    drop_z = cfg.vortexer.init_state.pos[2] + cfg.vortexer_top_from_root_m + 0.02
    probes = {
        "well_center": (center_x, center_y, drop_z),
        "well_rim": (center_x + 0.028, center_y, drop_z),
        "solid_body": (center_x - 0.050, center_y, drop_z),
        "off_target_table": (center_x - 0.250, center_y + 0.100, drop_z),
    }

    results = {}
    for name, start in probes.items():
        set_tube_pose(env, start)
        pos, lin_speed, ang_speed = settle(env)
        env.phase[:] = PHASE_RL_RELEASE
        reward = float(env._get_rewards()[0].item())
        inserted = bool(env._tube_inserted()[0].item())
        supported = bool(env._tube_supported_by_vortexer()[0].item())
        results[name] = (reward, inserted, supported)
        print(
            f"probe={name} start={start} final={pos} "
            f"lin_speed={lin_speed:.6f} ang_speed={ang_speed:.6f} "
            f"reward={reward:.4f} inserted={inserted} supported={supported}",
            flush=True,
        )

    center_reward, center_inserted, center_supported = results["well_center"]
    body_reward, body_inserted, _ = results["solid_body"]
    table_reward, table_inserted, table_supported = results["off_target_table"]
    if not center_inserted or not center_supported or center_reward < cfg.rew_success:
        raise RuntimeError("center probe did not produce contact-valid insertion reward")
    if table_inserted or table_supported or table_reward > 0.1:
        raise RuntimeError("off-target table probe still receives placement reward")
    if body_inserted or body_reward > 0.1:
        raise RuntimeError("solid vortexer body probe still receives placement reward")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
