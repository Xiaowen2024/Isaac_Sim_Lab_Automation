"""Finite-state-machine for a wet-lab pick-and-place task.

This script provides:
- batched scene creation
- random initialisation
- differential IK control loop
- full task state-machine with correct per-env phase transitions
- per-env gripper control (open / close)
- success / timeout / drop accounting

--- What you still need to connect ---
1. Confirm FRAMES.gripper_open_pos / gripper_closed_pos match your URDF joint limits
   (check with: grep -A 8 'name="left_finger_joint"' xarm6_with_gripper.urdf | grep limit)
2. Replace placeholder grasp / place poses with USD-authored frames once the assets have them.
"""

import argparse
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wet-lab pick-place state machine.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task_mode", type=str.lower, default="sample", choices=["sample", "a", "b"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# SceneEntityCfg lets us resolve joint / body IDs from the scene by name.
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from wetlab_benchmark.pick_place_claude_v4.runtime import create_pick_place_runtime
from wetlab_benchmark.pick_place_claude_v4.task_policy import Phase, PickPlaceTaskPolicy
# FRAMES carries joint/body name strings (ee_body_name, left_finger_joint, …).
from wetlab_benchmark.task_config import FRAMES, PLACEMENT, SCENE_CHECK, THRESH


def pose_error_norm(cur_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
    """L2 distance between current and goal position across a batch of envs."""
    return torch.linalg.norm(goal_pos - cur_pos, dim=-1)


def command_gripper(
    robot,
    gripper_joint_ids: list[int],
    closed_mask: torch.Tensor,
    device: str,
) -> None:
    """Send open/close position targets to the xArm6 finger joints, per env.

    Args:
        robot:            Isaac Lab Articulation object for the robot.
        gripper_joint_ids: Joint index list resolved from SceneEntityCfg for the
                           two finger joints (left_finger_joint, right_finger_joint).
        closed_mask:      Bool tensor [num_envs] — True → close, False → open.
        device:           Torch device string.

    TODO: If you are using a custom attach/detach helper (e.g. for a suction cup
          or rigid-body weld), replace the set_joint_position_target call below
          with your helper and ignore gripper_joint_ids.
    """
    num_envs = robot.num_instances
    num_joints = len(gripper_joint_ids)

    # Build per-env target tensor: open position by default …
    pos = torch.full((num_envs, num_joints), FRAMES.gripper_open_pos, device=device)
    # … then override the envs that should be closed.
    pos[closed_mask] = FRAMES.gripper_closed_pos

    robot.set_joint_position_target(pos, joint_ids=gripper_joint_ids)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    res = q.clone()
    res[..., 1:] = -res[..., 1:]
    return res


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = torch.cat([torch.zeros(v.shape[:-1] + (1,), device=v.device, dtype=v.dtype), v], dim=-1)
    return quat_mul(quat_mul(q, qvec), quat_conjugate(q))[..., 1:]


def skew_matrix(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    return torch.stack(
        (
            torch.stack((zeros, -v[:, 2], v[:, 1]), dim=-1),
            torch.stack((v[:, 2], zeros, -v[:, 0]), dim=-1),
            torch.stack((-v[:, 1], v[:, 0], zeros), dim=-1),
        ),
        dim=1,
    )


def support_point_w(asset, local_support_point_m: tuple[float, float, float]) -> torch.Tensor:
    local_support = torch.tensor(
        local_support_point_m,
        device=asset.data.root_pose_w.device,
        dtype=asset.data.root_pose_w.dtype,
    ).repeat(asset.data.root_pose_w.shape[0], 1)
    return asset.data.root_pose_w[:, 0:3] + quat_apply(asset.data.root_pose_w[:, 3:7], local_support)


def contact_any_for_filters(sensor, filter_indices: tuple[int, ...], force_threshold_n: float) -> torch.Tensor:
    forces = torch.nan_to_num(sensor.data.force_matrix_w)
    mags = torch.linalg.norm(forces, dim=-1)
    if mags.ndim == 1:
        hits = mags.unsqueeze(-1)
    elif mags.ndim == 2:
        hits = mags
    else:
        reduce_dims = tuple(range(1, mags.ndim - 1))
        hits = torch.amax(mags, dim=reduce_dims) if len(reduce_dims) > 0 else mags
    return torch.any(hits[:, filter_indices] > force_threshold_n, dim=-1)


def assign_task_mode(*, num_envs: int, device: str, mode_arg: str) -> torch.Tensor:
    if mode_arg == "a":
        return torch.zeros(num_envs, dtype=torch.long, device=device)
    if mode_arg == "b":
        return torch.ones(num_envs, dtype=torch.long, device=device)
    combo_probs = torch.tensor(PLACEMENT.task_combo_probs, device=device)
    return torch.multinomial(combo_probs.expand(num_envs, -1), num_samples=1).squeeze(-1)


def evaluate_final_success(
    *,
    runtime,
    policy: PickPlaceTaskPolicy,
    task_mode: torch.Tensor,
    task_step: torch.Tensor,
    holder_slot_index: torch.Tensor | None,
    phase: torch.Tensor,
    gripper_joint_ids: list[int],
    tube_local_support_point_m: tuple[float, float, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = runtime.device
    num_envs = runtime.num_envs

    final_goals = policy.build_support_goals_w(
        runtime.holder_asset,
        runtime.vortexer_asset,
        task_mode,
        task_step,
        holder_slot_index,
    )
    target_place_pos_w = final_goals["place"][:, 0:3]
    tube_support_pos_w = support_point_w(runtime.tube, tube_local_support_point_m)

    support_pos_err = torch.linalg.norm(tube_support_pos_w - target_place_pos_w, dim=-1)
    support_xy_err = torch.linalg.norm(tube_support_pos_w[:, :2] - target_place_pos_w[:, :2], dim=-1)
    support_z_err = torch.abs(tube_support_pos_w[:, 2] - target_place_pos_w[:, 2])

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).repeat(num_envs, 1)
    tube_up_axis = quat_apply(runtime.tube.data.root_pose_w[:, 3:7], z_axis)
    upright_ok = tube_up_axis[:, 2] >= math.cos(SCENE_CHECK.upright_tilt_tol_rad)

    no_drop = ~runtime.tube_asset.is_dropped(THRESH.drop_height_m)
    released_ok = ~contact_any_for_filters(
        runtime.tube_contacts,
        (0, 1),
        SCENE_CHECK.contact_force_threshold_n,
    ).to(device)

    gripper_pos = runtime.robot.data.joint_pos[:, gripper_joint_ids]
    gripper_open_ok = torch.all(
        torch.abs(gripper_pos - FRAMES.gripper_open_pos) < 0.10,
        dim=-1,
    )

    placed_ok = (support_xy_err <= SCENE_CHECK.pos_tol_m) & (support_z_err <= SCENE_CHECK.z_tol_m)
    phase_done_ok = phase == int(Phase.DONE)

    success = phase_done_ok & no_drop & upright_ok & released_ok & gripper_open_ok & placed_ok
    diagnostics = {
        "phase_done_ok": phase_done_ok,
        "no_drop": no_drop,
        "upright_ok": upright_ok,
        "released_ok": released_ok,
        "gripper_open_ok": gripper_open_ok,
        "placed_ok": placed_ok,
        "support_xy_err": support_xy_err,
        "support_z_err": support_z_err,
        "support_pos_err": support_pos_err,
    }
    return success, diagnostics


def main():
    runtime = create_pick_place_runtime(
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=args_cli.device,
        camera_eye=[2.4, 2.0, 2.0],
        camera_target=[0.35, 0.0, 0.85],
    )

    robot = runtime.robot
    robot_asset = runtime.robot_asset
    controller = runtime.controller
    tube_asset = runtime.tube_asset
    holder_asset = runtime.holder_asset
    vortexer_asset = runtime.vortexer_asset
    robot_entity_cfg = runtime.robot_entity_cfg
    ee_jacobi_idx = runtime.ee_jacobi_idx

    # -----------------------------------------------------------------------
    # Gripper entity config
    # -----------------------------------------------------------------------
    # We need a separate SceneEntityCfg for the gripper finger joints so that
    # we can send position targets to them independently of the arm joints.
    # TODO: if your xArm6 USD uses different joint names, update FRAMES in
    #       task_config.py (left_finger_joint / right_finger_joint).
    gripper_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=[FRAMES.left_finger_joint, FRAMES.right_finger_joint],
    )
    gripper_entity_cfg.resolve(runtime.scene)
    gripper_joint_ids: list[int] = gripper_entity_cfg.joint_ids

    policy = PickPlaceTaskPolicy(
        lift_height_m=THRESH.lift_height_m,
        retreat_height_m=THRESH.retreat_height_m,
    )
    tube_cfg = next(cfg for cfg in runtime.task_cfg.objects if cfg.name == "tube")
    holder_slot_index = runtime.layout["metadata"].get("tube_holder_point_index")
    if holder_slot_index is not None:
        holder_slot_index = holder_slot_index.to(runtime.device)

    # -----------------------------------------------------------------------
    # Per-env task combo assignment
    # -----------------------------------------------------------------------
    # task_mode == 0: combo A — pick from holder, place back in holder.
    # task_mode == 1: combo B — pick from holder, place on vortexer, return to holder.
    task_mode = assign_task_mode(
        num_envs=runtime.num_envs,
        device=runtime.device,
        mode_arg=args_cli.task_mode,
    )

    # task_step tracks where we are within the combo:
    #   0 = first pick-place leg (always holder → target)
    #   1 = second pick-place leg (vortexer → holder, combo B only)
    task_step = torch.zeros(runtime.num_envs, dtype=torch.long, device=runtime.device)

    # Per-env phase, step counter, terminal flags.
    phase = torch.full(
        (runtime.num_envs,), int(Phase.APPROACH), dtype=torch.long, device=runtime.device
    )
    phase_steps = torch.zeros(runtime.num_envs, dtype=torch.long, device=runtime.device)
    done = torch.zeros(runtime.num_envs, dtype=torch.bool, device=runtime.device)
    success = torch.zeros(runtime.num_envs, dtype=torch.bool, device=runtime.device)
    max_phase_reached = phase.clone()

    # Reset the IK controller before the first step so it starts from a clean state.
    controller.reset()

    for sim_step in range(THRESH.max_episode_steps):

        # -----------------------------------------------------------------------
        # 1. Read current robot / EE state
        # -----------------------------------------------------------------------
        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
        ]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pose_w = robot_asset.ee_pose_w          # [N, 7]  world-frame EE pose
        root_pose_w = robot.data.root_pose_w       # [N, 7]  world-frame robot base

        # Convert EE pose from world frame to robot base frame for the IK controller.
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],   ee_pose_w[:, 3:7],
        )
        offset_local = torch.tensor(
            FRAMES.ee_control_offset_local_m,
            device=runtime.device,
            dtype=ee_pos_b.dtype,
        ).repeat(runtime.num_envs, 1)
        offset_b = quat_apply(ee_quat_b, offset_local)
        jacobian = jacobian.clone()
        jacobian[:, 0:3, :] = jacobian[:, 0:3, :] - torch.bmm(skew_matrix(offset_b), jacobian[:, 3:6, :])

        # -----------------------------------------------------------------------
        # 2. Query task policy for the current target pose (world frame)
        # -----------------------------------------------------------------------
        target_w = policy.target_pose_w(
            phase,
            tube_asset=tube_asset,
            holder_asset=holder_asset,
            vortexer_asset=vortexer_asset,
            task_mode=task_mode,
            task_step=task_step,
            holder_slot_index=holder_slot_index,
            num_envs=runtime.num_envs,
            device=runtime.device,
        )
        inactive = done | (phase == int(Phase.DONE)) | (phase == int(Phase.FAILED))
        if bool(torch.any(inactive)):
            target_w[inactive] = ee_pose_w[inactive]

        # Convert target from world frame to robot base frame.
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_w[:, 0:3],    target_w[:, 3:7],
        )
        # -----------------------------------------------------------------------
        # 3. Run differential IK and send joint targets
        # -----------------------------------------------------------------------
        # IsaacLab now requires the current EE orientation even for position-mode IK commands.
        controller.set_command(target_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # Clamp per-joint delta so each step stays within the linear IK regime.
        # Without this the arm spirals: a 0.10m phase-transition jump overshoots
        # in joint space and the nonlinear Jacobian drives the EE away.
        MAX_JOINT_DELTA_RAD = 0.02
        joint_pos_des = joint_pos + torch.clamp(
            joint_pos_des - joint_pos, -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD
        )
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # -----------------------------------------------------------------------
        # 4. Gripper control — per-env, not global
        # -----------------------------------------------------------------------
        # gripper_closed_mask returns True for phases where the finger should stay closed
        # (CLOSE_GRIPPER, LIFT, TRANSFER, PLACE).  All other phases keep it open.
        closed_mask = policy.gripper_closed_mask(phase)
        command_gripper(robot, gripper_joint_ids, closed_mask, runtime.device)

        # -----------------------------------------------------------------------
        # 5. Phase transition logic
        # -----------------------------------------------------------------------
        # Only count steps for envs still running (terminal envs have no meaningful phase).
        phase_steps[~done] += 1

        # Distance from current EE world position to the current target position.
        ee_goal_dist = pose_error_norm(ee_pose_w[:, 0:3], target_w[:, 0:3])
        reach_tol = torch.full((runtime.num_envs,), THRESH.ee_pos_tol_m, device=runtime.device)
        reach_tol[phase == int(Phase.APPROACH)] = THRESH.approach_pos_tol_m
        reached = ee_goal_dist < reach_tol
        
        if sim_step % 100 == 0:
            pos_err_w = target_w[:, 0:3] - ee_pose_w[:, 0:3]
            print(
                "[pick_place_sm][env=0] "
                f"step={sim_step} phase={phase[0].item()} ee_dist={ee_goal_dist[0].item():.4f} "
                f"err_xyz=({pos_err_w[0, 0].item():.4f}, {pos_err_w[0, 1].item():.4f}, {pos_err_w[0, 2].item():.4f}) "
                f"ee_pos=({ee_pose_w[0, 0].item():.4f}, {ee_pose_w[0, 1].item():.4f}, {ee_pose_w[0, 2].item():.4f}) "
                f"target_pos=({target_w[0, 0].item():.4f}, {target_w[0, 1].item():.4f}, {target_w[0, 2].item():.4f})"
            )

        # `advance` is True for envs that reached their waypoint and are not done yet.
        advance = reached & (~done)

        # Pre-compute ALL masks using the CURRENT phase before any mutations.
        # This prevents an env that just entered CLOSE_GRIPPER or OPEN_GRIPPER (via a
        # reach-based transition) from immediately firing the time-based settle/open
        # mask in the same step because phase_steps hasn't been reset yet.
        approach_advance  = (phase == int(Phase.APPROACH))      & advance
        pregrasp_advance  = (phase == int(Phase.PREGRASP))      & advance
        lift_advance      = (phase == int(Phase.LIFT))          & advance
        transfer_advance  = (phase == int(Phase.TRANSFER))      & advance
        place_advance     = (phase == int(Phase.PLACE))         & advance
        retreat_advance   = (phase == int(Phase.RETREAT))       & advance
        settle_mask       = (phase == int(Phase.CLOSE_GRIPPER)) & (phase_steps > THRESH.grasp_settle_steps)
        open_mask         = (phase == int(Phase.OPEN_GRIPPER))  & (phase_steps > THRESH.grasp_settle_steps)

        # Apply reach-based transitions (all masks pre-computed above, no inter-step
        # contamination between phases).
        phase[approach_advance] = int(Phase.PREGRASP)
        phase[pregrasp_advance] = int(Phase.CLOSE_GRIPPER)
        phase[lift_advance]     = int(Phase.TRANSFER)
        phase[transfer_advance] = int(Phase.PLACE)
        phase[place_advance]    = int(Phase.OPEN_GRIPPER)

        # RETREAT is the end of one pick-place leg.
        # Combo A (mode 0) always finishes here.
        # Combo B (mode 1) step 0: still need a second leg (vortexer → holder).
        # Combo B (mode 1) step 1: fully done.
        retreat_done     = retreat_advance & ((task_mode == 0) | (task_step == 1))
        retreat_continue = retreat_advance & (task_mode == 1) & (task_step == 0)

        phase[retreat_done]     = int(Phase.DONE)
        # Reset to APPROACH for the second leg; increment task_step so the policy
        # now targets the holder as the place destination.
        phase[retreat_continue] = int(Phase.APPROACH)
        task_step[retreat_continue] = 1

        # Apply time-based transitions (using masks computed before phase mutations).
        phase[settle_mask] = int(Phase.LIFT)
        phase[open_mask]   = int(Phase.RETREAT)

        # Collect all envs that just changed phase so we can:
        #   (a) reset their per-env step counter, and
        #   (b) reset the IK controller so it starts fresh from the new waypoint.
        transitioned = advance | settle_mask | open_mask
        phase_steps[transitioned] = 0

        # Reset IK controller state for envs that just transitioned.
        # Without this, the controller retains the previous phase's error integral /
        # pseudo-inverse, which causes a jerky motion on the first step of the new phase.
        if bool(torch.any(transitioned)):
            controller.reset(env_ids=transitioned.nonzero(as_tuple=False).squeeze(-1))

        # -----------------------------------------------------------------------
        # 6. Failure detection (tube dropped below table surface)
        # -----------------------------------------------------------------------
        dropped = tube_asset.is_dropped(THRESH.drop_height_m)
        failed = dropped & (~done)
        phase[failed] = int(Phase.FAILED)

        done |= (phase == int(Phase.DONE)) | (phase == int(Phase.FAILED))
        success |= phase == int(Phase.DONE)
        max_phase_reached = torch.maximum(max_phase_reached, phase.clamp_max(int(Phase.DONE)))

        # -----------------------------------------------------------------------
        # 7. Step the simulation
        # -----------------------------------------------------------------------
        runtime.scene.write_data_to_sim()
        runtime.sim.step()
        runtime.scene.update(runtime.sim.get_physics_dt())

        # Render the viewer frame (no-op in headless mode).
        # Without this call the GUI window freezes when running with a display.
        if not args_cli.headless:
            runtime.sim.render()

        if torch.all(done):
            break

    timed_out = ~done
    if bool(torch.any(timed_out)):
        phase[timed_out] = int(Phase.FAILED)
        done[timed_out] = True

    success, diagnostics = evaluate_final_success(
        runtime=runtime,
        policy=policy,
        task_mode=task_mode,
        task_step=task_step,
        holder_slot_index=holder_slot_index,
        phase=phase,
        gripper_joint_ids=gripper_joint_ids,
        tube_local_support_point_m=tube_cfg.local_support_point_m,
    )
    phase[done & ~success] = int(Phase.FAILED)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    combo_a = (task_mode == 0).sum().item()
    combo_b = (task_mode == 1).sum().item()
    combo_a_success = (success & (task_mode == 0)).sum().item()
    combo_b_success = (success & (task_mode == 1)).sum().item()
    failed_count = int((phase == int(Phase.FAILED)).sum().item())

    print(f"[pick_place_sm] steps={sim_step + 1}")
    print(f"[pick_place_sm] success={int(success.sum().item())}/{runtime.num_envs}")
    print(f"[pick_place_sm] failed={failed_count}/{runtime.num_envs}")
    print(f"[pick_place_sm] combo_A(holder->holder): {int(combo_a_success)}/{int(combo_a)}")
    print(f"[pick_place_sm] combo_B(holder->vortexer->holder): {int(combo_b_success)}/{int(combo_b)}")
    print(f"[pick_place_sm] task_mode_arg={args_cli.task_mode}")
    phase_labels = (
        "APPROACH",
        "PREGRASP",
        "CLOSE_GRIPPER",
        "LIFT",
        "TRANSFER",
        "PLACE",
        "OPEN_GRIPPER",
        "RETREAT",
        "DONE",
        "FAILED",
    )
    phase_counts = [int((phase == idx).sum().item()) for idx in range(len(phase_labels))]
    print(
        "[pick_place_sm] phase_counts "
        + " ".join(f"{label}={count}" for label, count in zip(phase_labels, phase_counts))
    )
    max_phase_counts = [int((max_phase_reached == idx).sum().item()) for idx in range(len(phase_labels) - 1)]
    print(
        "[pick_place_sm] max_phase_reached "
        + " ".join(f"{label}={count}" for label, count in zip(phase_labels[:-1], max_phase_counts))
    )
    print(
        "[pick_place_sm] final_checks "
        f"phase_done={int(diagnostics['phase_done_ok'].sum().item())}/{runtime.num_envs} "
        f"placed={int(diagnostics['placed_ok'].sum().item())}/{runtime.num_envs} "
        f"released={int(diagnostics['released_ok'].sum().item())}/{runtime.num_envs} "
        f"upright={int(diagnostics['upright_ok'].sum().item())}/{runtime.num_envs} "
        f"gripper_open={int(diagnostics['gripper_open_ok'].sum().item())}/{runtime.num_envs} "
        f"no_drop={int(diagnostics['no_drop'].sum().item())}/{runtime.num_envs}"
    )

    return 0 if int(success.sum().item()) == runtime.num_envs else 1


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
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
