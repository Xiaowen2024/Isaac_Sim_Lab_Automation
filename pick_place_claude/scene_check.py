"""Load a batched wet-lab scene and randomize the main fixture poses.

This is the first script to make work. It only checks:
1. the stage loads,
2. assets spawn under Isaac Lab,
3. randomized placements are collision-reasonable,
4. robot Jacobians and body frames are readable.
"""

import argparse
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage validation for the wet-lab benchmark.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tube_source", type=str, default="all", choices=["all", "holder", "vortexer", "table"])
parser.add_argument("--debug_metrics", action="store_true", help="Print per-env placement and contact diagnostics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from wetlab_benchmark.pick_place_claude.runtime import create_pick_place_runtime
from wetlab_benchmark.task_config import PICK_PLACE_RANDOMIZATION, SCENE_CHECK, TUBE_SOURCE_NAMES, with_tube_source
from wetlab_benchmark.validation import (
    _pose_in_env,
    _support_point_in_env,
    quat_apply,
    quat_conjugate,
    validate_task_reachability,
    validate_task_scene,
)


def _contact_force_summary(sensor) -> torch.Tensor:
    forces = torch.nan_to_num(sensor.data.force_matrix_w)
    mags = torch.linalg.norm(forces, dim=-1)
    if mags.ndim == 1:
        return mags.unsqueeze(-1)
    if mags.ndim == 2:
        return mags
    reduce_dims = tuple(range(1, mags.ndim - 1))
    if len(reduce_dims) == 0:
        return mags
    return torch.amax(mags, dim=reduce_dims)


def _print_debug_metrics(runtime) -> None:
    device = runtime.tube.data.root_pose_w.device
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).repeat(runtime.scene.num_envs, 1)

    tube_pos_env, tube_quat_w = _pose_in_env(runtime.tube, runtime.scene.env_origins)
    holder_pos_env, holder_quat_w = _pose_in_env(runtime.holder, runtime.scene.env_origins)
    vortexer_pos_env, _ = _pose_in_env(runtime.vortexer, runtime.scene.env_origins)
    robot_pos_env, _ = _pose_in_env(runtime.robot, runtime.scene.env_origins)

    sampled_tube = runtime.layout["poses"]["tube"]
    placement_idx = runtime.layout["metadata"]["tube_placement_index"].to(device)
    holder_slot_idx = runtime.layout["metadata"].get("tube_holder_point_index")
    if holder_slot_idx is not None:
        holder_slot_idx = holder_slot_idx.to(device)

    tube_cfg = next(cfg for cfg in runtime.task_cfg.objects if cfg.name == "tube")
    holder_cfg = next(cfg for cfg in runtime.task_cfg.fixtures if cfg.name == "holder")
    holder_zone = next(zone for zone in holder_cfg.support_zones if zone.name == "tube_support")

    support_point_env = _support_point_in_env(tube_pos_env, tube_quat_w, tube_cfg.local_support_point_m)
    sampled_drift = torch.linalg.norm(tube_pos_env - sampled_tube.pos.to(device), dim=-1)
    up_axis = quat_apply(tube_quat_w, z_axis)
    local_offset = quat_apply(quat_conjugate(holder_quat_w), support_point_env - holder_pos_env)
    local_xy = local_offset[:, :2]
    expected_support_z = (
        runtime.task_cfg.surfaces[holder_cfg.surface].z
        + holder_zone.support_height_from_surface_m
        + tube_cfg.root_height_from_support_m
    )

    tube_contact_summary = _contact_force_summary(runtime.tube_contacts)
    holder_contact_summary = _contact_force_summary(runtime.holder_contacts)
    robot_holder_dist = torch.linalg.norm(robot_pos_env[:, :2] - holder_pos_env[:, :2], dim=-1)
    vortexer_holder_dist = torch.linalg.norm(vortexer_pos_env[:, :2] - holder_pos_env[:, :2], dim=-1)

    holder_points = torch.tensor(
        [(point[0], point[1]) for point in holder_zone.discrete_local_xy_points_m],
        device=device,
        dtype=local_xy.dtype,
    )

    print("[scene_check] debug_metrics begin")
    for env_id in range(runtime.scene.num_envs):
        placement_name = TUBE_SOURCE_NAMES[int(placement_idx[env_id].item())]
        slot_index = int(holder_slot_idx[env_id].item()) if holder_slot_idx is not None else -1
        target_xy = None
        planar_error = None
        if placement_name == "holder" and 0 <= slot_index < holder_points.shape[0]:
            target_xy = holder_points[slot_index]
            planar_error = torch.linalg.norm(local_xy[env_id] - target_xy).item()

        print(
            "[scene_check][env={}] placement={} slot={} drift_m={:.5f} up_z={:.5f} support_z={:.5f} expected_support_z={:.5f} support_z_err={:.5f}".format(
                env_id,
                placement_name,
                slot_index,
                sampled_drift[env_id].item(),
                up_axis[env_id, 2].item(),
                support_point_env[env_id, 2].item(),
                expected_support_z,
                abs(support_point_env[env_id, 2].item() - expected_support_z),
            )
        )
        print(
            "[scene_check][env={}] tube_pos={} sampled_tube_pos={}".format(
                env_id,
                [round(v, 5) for v in tube_pos_env[env_id].tolist()],
                [round(v, 5) for v in sampled_tube.pos[env_id].tolist()],
            )
        )
        print(
            "[scene_check][env={}] holder_pos={} holder_local_support_xyz={}".format(
                env_id,
                [round(v, 5) for v in holder_pos_env[env_id].tolist()],
                [round(v, 5) for v in local_offset[env_id].tolist()],
            )
        )
        if target_xy is not None:
            print(
                "[scene_check][env={}] holder_target_xy={} planar_err={:.5f}".format(
                    env_id,
                    [round(v, 5) for v in target_xy.tolist()],
                    planar_error,
                )
            )
        print(
            "[scene_check][env={}] tube_contacts[left_finger,right_finger,holder,vortexer]={} holder_contacts[left_finger,right_finger,tube]={}".format(
                env_id,
                [round(v, 6) for v in tube_contact_summary[env_id].tolist()],
                [round(v, 6) for v in holder_contact_summary[env_id].tolist()],
            )
        )
        print(
            "[scene_check][env={}] planar_dist(robot,holder)={:.5f} planar_dist(holder,vortexer)={:.5f}".format(
                env_id,
                robot_holder_dist[env_id].item(),
                vortexer_holder_dist[env_id].item(),
            )
        )
    print("[scene_check] debug_metrics end")


def main():
    task_cfg = with_tube_source(PICK_PLACE_RANDOMIZATION, args_cli.tube_source)
    runtime = create_pick_place_runtime(
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=args_cli.device,
        camera_eye=[2.6, 2.0, 2.2],
        camera_target=[0.4, 0.0, 0.8],
        task_cfg=task_cfg,
    )

    for _ in range(SCENE_CHECK.settle_steps):
        runtime.scene.write_data_to_sim()
        runtime.sim.step()
        runtime.scene.update(runtime.sim.get_physics_dt())

    failures = validate_task_scene(
        scene=runtime.scene,
        assets=runtime.assets,
        layout=runtime.layout,
        task_cfg=runtime.task_cfg,
        check_cfg=SCENE_CHECK,
        sensors=runtime.sensors,
    )
    failures.extend(validate_task_reachability(runtime=runtime))

    print("[scene_check] stage loaded and validated")
    print(f"[scene_check] num_envs={runtime.scene.num_envs}")
    print(f"[scene_check] robot_root_shape={tuple(runtime.robot.data.root_pose_w.shape)}")
    print(f"[scene_check] tube_root_shape={tuple(runtime.tube.data.root_pose_w.shape)}")
    placement_index = runtime.layout["metadata"]["tube_placement_index"]
    source_counts = torch.bincount(placement_index.cpu(), minlength=3).tolist()
    print(f"[scene_check] tube_sources(holder,vortexer,table)={source_counts}")
    holder_slot_index = runtime.layout["metadata"].get("tube_holder_point_index")
    if holder_slot_index is not None:
        valid_slot_index = holder_slot_index[holder_slot_index >= 0].cpu()
        if valid_slot_index.numel() > 0:
            slot_counts = torch.bincount(valid_slot_index, minlength=4).tolist()
            print(f"[scene_check] holder_slot_counts={slot_counts}")
    if args_cli.debug_metrics:
        _print_debug_metrics(runtime)
    if failures:
        print("[scene_check] validation failed")
        for failure in failures:
            print(f"[scene_check] failure: {failure}")
        raise RuntimeError("scene_check validation failed")
    print("[scene_check] validation passed")

if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        # Isaac Sim 4.5 can hang in Kit shutdown after a successful headless run,
        # which blocks shell seed loops from advancing to the next invocation.
        # For one-shot batch validation jobs, flush output and terminate directly.
        if getattr(args_cli, "headless", False):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
