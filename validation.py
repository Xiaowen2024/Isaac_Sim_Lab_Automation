from __future__ import annotations

import math

import torch

from wetlab_benchmark.task_config import (
    FixtureRandomizationCfg,
    ObjectRandomizationCfg,
    REACHABILITY,
    ReachabilityCfg,
    SceneCheckCfg,
    SurfacePlacementOptionCfg,
    TaskRandomizationCfg,
    THRESH,
    ZonePlacementOptionCfg,
)


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


def _pose_in_env(asset, env_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pose_w = asset.data.root_pose_w
    pos_env = pose_w[:, 0:3] - env_origins.to(pose_w.device)
    quat_w = pose_w[:, 3:7]
    return pos_env, quat_w


def _support_point_in_env(
    pos_env: torch.Tensor,
    quat_w: torch.Tensor,
    local_support_point_m: tuple[float, float, float],
) -> torch.Tensor:
    local_support_point = torch.tensor(
        local_support_point_m,
        device=pos_env.device,
        dtype=pos_env.dtype,
    ).repeat(pos_env.shape[0], 1)
    return pos_env + quat_apply(quat_w, local_support_point)


def _contact_any_for_filter(sensor, filter_index: int, force_threshold_n: float) -> torch.Tensor:
    forces = torch.nan_to_num(sensor.data.force_matrix_w)
    mags = torch.linalg.norm(forces, dim=-1)
    if mags.ndim == 1:
        return mags > force_threshold_n
    if mags.ndim == 2:
        return mags[:, filter_index] > force_threshold_n
    hits = mags[..., filter_index] > force_threshold_n
    reduce_dims = tuple(range(1, hits.ndim))
    return torch.any(hits, dim=reduce_dims)


def _contact_any_for_filters(sensor, filter_indices: tuple[int, ...], force_threshold_n: float) -> torch.Tensor:
    hits = [_contact_any_for_filter(sensor, filter_index, force_threshold_n) for filter_index in filter_indices]
    if len(hits) == 1:
        return hits[0]
    return torch.any(torch.stack(hits, dim=0), dim=0)


def _support_zone_map(task_cfg: TaskRandomizationCfg) -> dict[str, dict[str, object]]:
    zones: dict[str, dict[str, object]] = {}
    for fixture in task_cfg.fixtures:
        zones[fixture.name] = {zone.name: zone for zone in fixture.support_zones}
    return zones


def validate_task_scene(
    *,
    scene,
    assets: dict[str, object],
    layout: dict,
    task_cfg: TaskRandomizationCfg,
    check_cfg: SceneCheckCfg,
    sensors: dict[str, object] | None = None,
) -> list[str]:
    failures: list[str] = []
    fixture_cfgs: dict[str, FixtureRandomizationCfg] = {cfg.name: cfg for cfg in task_cfg.fixtures}
    object_cfgs: dict[str, ObjectRandomizationCfg] = {cfg.name: cfg for cfg in task_cfg.objects}
    zone_map = _support_zone_map(task_cfg)
    actual: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        asset_name: _pose_in_env(asset, scene.env_origins) for asset_name, asset in assets.items()
    }

    sampled = layout["poses"]
    metadata = layout["metadata"]

    def add_failure(message: str):
        failures.append(message)

    # scene.device may return "cpu" even when PhysX runs on CUDA; derive the true
    # simulation device from the asset tensors themselves.
    _device = next(iter(actual.values()))[0].device
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=_device, dtype=torch.float32).repeat(scene.num_envs, 1)
    upright_cos_min = math.cos(check_cfg.upright_tilt_tol_rad)

    # Fixtures: bounds, height, upright, drift.
    for fixture_cfg in task_cfg.fixtures:
        pos_env, quat_w = actual[fixture_cfg.name]
        sampled_pose = sampled[fixture_cfg.name]
        bounds = fixture_cfg.planar_bounds
        surface_z = task_cfg.surfaces[fixture_cfg.surface].z
        expected_z = surface_z + fixture_cfg.root_height_from_surface_m

        if torch.any(pos_env[:, 0] < bounds.x[0] - check_cfg.pos_tol_m) or torch.any(pos_env[:, 0] > bounds.x[1] + check_cfg.pos_tol_m):
            add_failure(f"{fixture_cfg.name}: x out of configured bounds")
        if torch.any(pos_env[:, 1] < bounds.y[0] - check_cfg.pos_tol_m) or torch.any(pos_env[:, 1] > bounds.y[1] + check_cfg.pos_tol_m):
            add_failure(f"{fixture_cfg.name}: y out of configured bounds")
        if torch.any(torch.abs(pos_env[:, 2] - expected_z) > check_cfg.z_tol_m):
            add_failure(f"{fixture_cfg.name}: z does not match support height; tune root_height_from_surface_m")
        up_axis = quat_apply(quat_w, z_axis)
        if torch.any(up_axis[:, 2] < upright_cos_min):
            add_failure(f"{fixture_cfg.name}: not upright after settling")
        if torch.any(torch.linalg.norm(pos_env - sampled_pose.pos.to(_device), dim=-1) > check_cfg.pos_tol_m):
            add_failure(f"{fixture_cfg.name}: drifted from sampled pose after settling")

    # Fixture pair separations.
    for fixture_cfg in task_cfg.fixtures:
        pos_env, _ = actual[fixture_cfg.name]
        for constraint in fixture_cfg.min_distance_from:
            other_pos, _ = actual[constraint.other_asset]
            planar_dist = torch.linalg.norm(pos_env[:, :2] - other_pos[:, :2], dim=-1)
            if torch.any(planar_dist < constraint.min_distance_m - check_cfg.pos_tol_m):
                add_failure(
                    f"{fixture_cfg.name}: planar clearance to {constraint.other_asset} below minimum {constraint.min_distance_m:.3f} m"
                )
            if constraint.max_distance_m is not None and torch.any(planar_dist > constraint.max_distance_m + check_cfg.pos_tol_m):
                add_failure(
                    f"{fixture_cfg.name}: planar distance to {constraint.other_asset} exceeds maximum {constraint.max_distance_m:.3f} m"
                )

    # Objects: placement-specific support height, bounds, upright, drift.
    for object_cfg in task_cfg.objects:
        pos_env, quat_w = actual[object_cfg.name]
        sampled_pose = sampled[object_cfg.name]
        placement_idx = metadata[f"{object_cfg.name}_placement_index"].to(_device)
        support_point_env = _support_point_in_env(pos_env, quat_w, object_cfg.local_support_point_m)
        up_axis = quat_apply(quat_w, z_axis)
        if torch.any(up_axis[:, 2] < upright_cos_min):
            add_failure(f"{object_cfg.name}: not upright after settling")
        if torch.any(torch.linalg.norm(pos_env - sampled_pose.pos.to(_device), dim=-1) > check_cfg.pos_tol_m):
            add_failure(f"{object_cfg.name}: drifted from sampled pose after settling")

        for option_i, option_cfg in enumerate(object_cfg.placement_options):
            mask = placement_idx == option_i
            if not bool(torch.any(mask)):
                continue

            if isinstance(option_cfg, SurfacePlacementOptionCfg):
                bounds = option_cfg.planar_bounds
                surface_z = task_cfg.surfaces[option_cfg.surface].z + option_cfg.support_height_from_surface_m
                expected_z = surface_z + object_cfg.root_height_from_support_m
                if torch.any(support_point_env[mask, 0] < bounds.x[0] - check_cfg.pos_tol_m) or torch.any(
                    support_point_env[mask, 0] > bounds.x[1] + check_cfg.pos_tol_m
                ):
                    add_failure(f"{object_cfg.name}:{option_cfg.name}: x out of configured bounds")
                if torch.any(support_point_env[mask, 1] < bounds.y[0] - check_cfg.pos_tol_m) or torch.any(
                    support_point_env[mask, 1] > bounds.y[1] + check_cfg.pos_tol_m
                ):
                    add_failure(f"{object_cfg.name}:{option_cfg.name}: y out of configured bounds")
                if torch.any(torch.abs(support_point_env[mask, 2] - expected_z) > check_cfg.z_tol_m):
                    add_failure(f"{object_cfg.name}:{option_cfg.name}: support height mismatch")
                for constraint in option_cfg.min_distance_from:
                    other_pos, _ = actual[constraint.other_asset]
                    planar_dist = torch.linalg.norm(support_point_env[mask, :2] - other_pos[mask, :2], dim=-1)
                    if torch.any(planar_dist < constraint.min_distance_m - check_cfg.pos_tol_m):
                        add_failure(
                            f"{object_cfg.name}:{option_cfg.name}: planar clearance to {constraint.other_asset} below minimum"
                        )
                    if constraint.max_distance_m is not None and torch.any(
                        planar_dist > constraint.max_distance_m + check_cfg.pos_tol_m
                    ):
                        add_failure(
                            f"{object_cfg.name}:{option_cfg.name}: planar distance to {constraint.other_asset} exceeds maximum"
                        )
            elif isinstance(option_cfg, ZonePlacementOptionCfg):
                parent_pos, parent_quat = actual[option_cfg.parent_asset]
                parent_cfg = fixture_cfgs[option_cfg.parent_asset]
                zone_cfg = zone_map[option_cfg.parent_asset][option_cfg.zone_name]
                local_offset = quat_apply(
                    quat_conjugate(parent_quat[mask]),
                    support_point_env[mask] - parent_pos[mask],
                )
                local_xy = local_offset[:, :2]
                point_key = f"{object_cfg.name}_{option_cfg.name}_point_index"
                point_index = metadata.get(point_key)
                if point_index is not None:
                    point_index = point_index.to(_device)
                if zone_cfg.discrete_local_xy_points_m and point_index is not None:
                    selected_index = point_index[mask]
                    points = torch.tensor(
                        zone_cfg.discrete_local_xy_points_m,
                        device=_device,
                        dtype=local_xy.dtype,
                    )
                    if torch.any(selected_index < 0) or torch.any(selected_index >= points.shape[0]):
                        add_failure(f"{object_cfg.name}:{option_cfg.name}: invalid support point index metadata")
                    else:
                        target_xy = points[selected_index]
                        planar_error = torch.linalg.norm(local_xy - target_xy, dim=-1)
                        if torch.any(planar_error > check_cfg.pos_tol_m):
                            add_failure(f"{object_cfg.name}:{option_cfg.name}: not on the selected support point")
                else:
                    if torch.any(
                        torch.abs(local_xy[:, 0] - zone_cfg.local_xy_center_m[0])
                        > zone_cfg.local_xy_halfspan_m[0] + check_cfg.pos_tol_m
                    ):
                        add_failure(f"{object_cfg.name}:{option_cfg.name}: local x outside support zone")
                    if torch.any(
                        torch.abs(local_xy[:, 1] - zone_cfg.local_xy_center_m[1])
                        > zone_cfg.local_xy_halfspan_m[1] + check_cfg.pos_tol_m
                    ):
                        add_failure(f"{object_cfg.name}:{option_cfg.name}: local y outside support zone")
                expected_z = (
                    task_cfg.surfaces[parent_cfg.surface].z
                    + zone_cfg.support_height_from_surface_m
                    + object_cfg.root_height_from_support_m
                )
                if torch.any(torch.abs(support_point_env[mask, 2] - expected_z) > check_cfg.z_tol_m):
                    add_failure(f"{object_cfg.name}:{option_cfg.name}: support height mismatch")
            else:
                add_failure(f"{object_cfg.name}: unsupported placement option type {type(option_cfg)}")

    # Forbidden contacts using filtered contact sensors.
    if sensors is not None:
        tube_placement = metadata.get("tube_placement_index")
        if tube_placement is not None:
            tube_placement = tube_placement.to(_device)
        tube_contacts = sensors.get("tube_contacts")
        holder_contacts = sensors.get("holder_contacts")

        if tube_contacts is not None:
            # Filter order is [left_finger, right_finger, holder, vortexer].
            tube_robot = _contact_any_for_filters(tube_contacts, (0, 1), check_cfg.contact_force_threshold_n).to(_device)
            if torch.any(tube_robot):
                add_failure("tube has forbidden contact with robot")

            tube_holder = _contact_any_for_filters(tube_contacts, (2,), check_cfg.contact_force_threshold_n).to(_device)
            tube_vortexer = _contact_any_for_filters(tube_contacts, (3,), check_cfg.contact_force_threshold_n).to(_device)
            if tube_placement is not None:
                forbidden_tube_holder = tube_holder & (tube_placement != 0)
                forbidden_tube_vortexer = tube_vortexer & (tube_placement != 1)
                if torch.any(forbidden_tube_holder):
                    add_failure("tube has forbidden contact with holder outside holder-supported placement")
                if torch.any(forbidden_tube_vortexer):
                    add_failure("tube has forbidden contact with vortexer outside vortexer-supported placement")

        if holder_contacts is not None:
            # Filter order is [left_finger, right_finger, tube].
            holder_robot = _contact_any_for_filters(holder_contacts, (0, 1), check_cfg.contact_force_threshold_n).to(_device)
            if torch.any(holder_robot):
                add_failure("holder has forbidden contact with robot")

    return failures


def validate_task_reachability(
    *,
    runtime,
    reach_cfg: ReachabilityCfg = REACHABILITY,
) -> list[str]:
    """Validate that the policy waypoints stay inside a conservative robot-base workspace."""
    from isaaclab.utils.math import subtract_frame_transforms

    from wetlab_benchmark.pick_place.task_policy import PickPlaceTaskPolicy

    failures: list[str] = []
    policy = PickPlaceTaskPolicy(
        lift_height_m=THRESH.lift_height_m,
        retreat_height_m=THRESH.retreat_height_m,
    )

    zeros = torch.zeros(runtime.num_envs, dtype=torch.long, device=runtime.device)
    ones = torch.ones(runtime.num_envs, dtype=torch.long, device=runtime.device)
    holder_slot_index = runtime.layout["metadata"].get("tube_holder_point_index")
    if holder_slot_index is not None:
        holder_slot_index = holder_slot_index.to(runtime.device)

    holder_goals = policy.build_goals_w(
        runtime.tube_asset,
        runtime.holder_asset,
        runtime.vortexer_asset,
        task_mode=zeros,
        task_step=zeros,
        holder_slot_index=holder_slot_index,
    )
    vortexer_goals = policy.build_goals_w(
        runtime.tube_asset,
        runtime.holder_asset,
        runtime.vortexer_asset,
        task_mode=ones,
        task_step=zeros,
        holder_slot_index=holder_slot_index,
    )

    goal_sets = {
        "pick:pregrasp": holder_goals["pregrasp"],
        "pick:grasp": holder_goals["grasp"],
        "pick:lift": holder_goals["lift"],
        "holder:preplace": holder_goals["preplace"],
        "holder:place": holder_goals["place"],
        "holder:retreat": holder_goals["retreat"],
        "vortexer:preplace": vortexer_goals["preplace"],
        "vortexer:place": vortexer_goals["place"],
        "vortexer:retreat": vortexer_goals["retreat"],
    }

    root_pose_w = runtime.robot.data.root_pose_w
    for goal_name, goal_w in goal_sets.items():
        goal_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            goal_w[:, 0:3],
            goal_w[:, 3:7],
        )
        planar_dist = torch.linalg.norm(goal_pos_b[:, :2], dim=-1)
        height_rel_base = goal_pos_b[:, 2]
        if torch.any(planar_dist < reach_cfg.min_goal_planar_distance_m - reach_cfg.tolerance_m):
            failures.append(
                f"reachability:{goal_name}: planar distance from robot base below minimum {reach_cfg.min_goal_planar_distance_m:.3f} m"
            )
        if torch.any(planar_dist > reach_cfg.max_goal_planar_distance_m + reach_cfg.tolerance_m):
            failures.append(
                f"reachability:{goal_name}: planar distance from robot base exceeds maximum {reach_cfg.max_goal_planar_distance_m:.3f} m"
            )
        if torch.any(height_rel_base < reach_cfg.min_goal_height_rel_base_m - reach_cfg.tolerance_m):
            failures.append(
                f"reachability:{goal_name}: height relative to robot base below minimum {reach_cfg.min_goal_height_rel_base_m:.3f} m"
            )
        if torch.any(height_rel_base > reach_cfg.max_goal_height_rel_base_m + reach_cfg.tolerance_m):
            failures.append(
                f"reachability:{goal_name}: height relative to robot base exceeds maximum {reach_cfg.max_goal_height_rel_base_m:.3f} m"
            )

    return failures
