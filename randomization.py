from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.assets import RigidObject

from wetlab_benchmark.task_config import (
    FixtureRandomizationCfg,
    ObjectRandomizationCfg,
    PICK_PLACE_RANDOMIZATION,
    PlanarPoseBoundsCfg,
    SurfacePlacementOptionCfg,
    TaskRandomizationCfg,
    ZonePlacementOptionCfg,
)

@dataclass
class AssetPoseSample:
    pos: torch.Tensor
    quat: torch.Tensor
    yaw: torch.Tensor


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat


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


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    res = q.clone()
    res[..., 1:] = -res[..., 1:]
    return res


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = torch.cat([torch.zeros(v.shape[:-1] + (1,), device=v.device, dtype=v.dtype), v], dim=-1)
    return quat_mul(quat_mul(q, qvec), quat_conjugate(q))[..., 1:]


def _make_quat(
    *,
    num_envs: int,
    device: str,
    yaw: torch.Tensor,
    base_quat_wxyz: tuple[float, float, float, float],
) -> torch.Tensor:
    yaw_quat = yaw_to_quat(yaw)
    base_quat = torch.tensor(base_quat_wxyz, device=device, dtype=yaw.dtype).repeat(num_envs, 1)
    return quat_mul(yaw_quat, base_quat)


def _sample_planar_bounds(num_envs: int, bounds: PlanarPoseBoundsCfg, device: str):
    x = torch.empty(num_envs, device=device).uniform_(*bounds.x)
    y = torch.empty(num_envs, device=device).uniform_(*bounds.y)
    yaw = torch.empty(num_envs, device=device).uniform_(*bounds.yaw)
    return x, y, yaw


def _pairwise_planar_dist(candidate_xy: torch.Tensor, other_pos: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(candidate_xy - other_pos[:, :2], dim=-1)


def _sample_fixture_pose_raw(
    fixture_cfg: FixtureRandomizationCfg,
    surfaces: dict[str, object],
    num_envs: int,
    device: str,
) -> AssetPoseSample:
    surface_z = surfaces[fixture_cfg.surface].z

    x, y, yaw = _sample_planar_bounds(num_envs, fixture_cfg.planar_bounds, device)
    pos = torch.stack(
        [x, y, torch.full_like(x, surface_z + fixture_cfg.root_height_from_surface_m)],
        dim=-1,
    )

    quat = _make_quat(
        num_envs=num_envs,
        device=device,
        yaw=yaw,
        base_quat_wxyz=fixture_cfg.base_quat_wxyz,
    )
    return AssetPoseSample(pos=pos, quat=quat, yaw=yaw)


def _assign_masked_pose(dst: AssetPoseSample, src: AssetPoseSample, mask: torch.Tensor) -> None:
    if not bool(torch.any(mask)):
        return
    dst.pos[mask] = src.pos[mask]
    dst.quat[mask] = src.quat[mask]
    dst.yaw[mask] = src.yaw[mask]


def _fixture_pair_min_distance(
    fixture_a: str,
    fixture_b: str,
    fixture_cfgs: dict[str, FixtureRandomizationCfg],
) -> float:
    min_distance = 0.0
    for constraint in fixture_cfgs[fixture_a].min_distance_from:
        if constraint.other_asset == fixture_b:
            min_distance = max(min_distance, constraint.min_distance_m)
    for constraint in fixture_cfgs[fixture_b].min_distance_from:
        if constraint.other_asset == fixture_a:
            min_distance = max(min_distance, constraint.min_distance_m)
    return min_distance


def _fixture_pair_max_distance(
    fixture_a: str,
    fixture_b: str,
    fixture_cfgs: dict[str, FixtureRandomizationCfg],
) -> float | None:
    max_distances: list[float] = []
    for constraint in fixture_cfgs[fixture_a].min_distance_from:
        if constraint.other_asset == fixture_b and constraint.max_distance_m is not None:
            max_distances.append(constraint.max_distance_m)
    for constraint in fixture_cfgs[fixture_b].min_distance_from:
        if constraint.other_asset == fixture_a and constraint.max_distance_m is not None:
            max_distances.append(constraint.max_distance_m)
    if not max_distances:
        return None
    return min(max_distances)


def _sample_fixture_poses(
    task_cfg: TaskRandomizationCfg,
    *,
    num_envs: int,
    device: str,
) -> dict[str, AssetPoseSample]:
    fixture_cfgs = {cfg.name: cfg for cfg in task_cfg.fixtures}
    placed = {
        cfg.name: _sample_fixture_pose_raw(cfg, task_cfg.surfaces, num_envs, device)
        for cfg in task_cfg.fixtures
    }

    for _ in range(128):
        invalid = {
            cfg.name: torch.zeros(num_envs, dtype=torch.bool, device=device)
            for cfg in task_cfg.fixtures
        }
        any_invalid = False

        for idx_a, cfg_a in enumerate(task_cfg.fixtures):
            for cfg_b in task_cfg.fixtures[idx_a + 1 :]:
                min_distance = _fixture_pair_min_distance(cfg_a.name, cfg_b.name, fixture_cfgs)
                max_distance = _fixture_pair_max_distance(cfg_a.name, cfg_b.name, fixture_cfgs)
                if min_distance <= 0.0 and max_distance is None:
                    continue
                if min_distance <= 0.0:
                    min_violation = torch.zeros(num_envs, dtype=torch.bool, device=device)
                else:
                    planar_dist = _pairwise_planar_dist(placed[cfg_a.name].pos[:, :2], placed[cfg_b.name].pos)
                    min_violation = planar_dist < min_distance
                if max_distance is None:
                    max_violation = torch.zeros(num_envs, dtype=torch.bool, device=device)
                else:
                    if min_distance <= 0.0:
                        planar_dist = _pairwise_planar_dist(placed[cfg_a.name].pos[:, :2], placed[cfg_b.name].pos)
                    max_violation = planar_dist > max_distance
                violation = min_violation | max_violation
                if bool(torch.any(violation)):
                    invalid[cfg_a.name] |= violation
                    invalid[cfg_b.name] |= violation
                    any_invalid = True

        if not any_invalid:
            return placed

        for cfg in task_cfg.fixtures:
            mask = invalid[cfg.name]
            if bool(torch.any(mask)):
                resample = _sample_fixture_pose_raw(cfg, task_cfg.surfaces, num_envs, device)
                _assign_masked_pose(placed[cfg.name], resample, mask)

    unresolved = []
    for idx_a, cfg_a in enumerate(task_cfg.fixtures):
        for cfg_b in task_cfg.fixtures[idx_a + 1 :]:
            min_distance = _fixture_pair_min_distance(cfg_a.name, cfg_b.name, fixture_cfgs)
            max_distance = _fixture_pair_max_distance(cfg_a.name, cfg_b.name, fixture_cfgs)
            if min_distance <= 0.0 and max_distance is None:
                continue
            planar_dist = _pairwise_planar_dist(placed[cfg_a.name].pos[:, :2], placed[cfg_b.name].pos)
            too_close = planar_dist < min_distance if min_distance > 0.0 else torch.zeros_like(planar_dist, dtype=torch.bool)
            too_far = planar_dist > max_distance if max_distance is not None else torch.zeros_like(planar_dist, dtype=torch.bool)
            if bool(torch.any(too_close | too_far)):
                unresolved.append(f"{cfg_a.name}<->{cfg_b.name}")
    raise RuntimeError(f"Failed to sample non-overlapping fixture layout for: {', '.join(unresolved)}")


def _lookup_zone(parent_cfg: FixtureRandomizationCfg, zone_name: str):
    for zone in parent_cfg.support_zones:
        if zone.name == zone_name:
            return zone
    raise KeyError(f"Zone '{zone_name}' not found in fixture '{parent_cfg.name}'")


def _sample_surface_object_pose(
    object_cfg: ObjectRandomizationCfg,
    option_cfg: SurfacePlacementOptionCfg,
    surfaces: dict[str, object],
    placed: dict[str, AssetPoseSample],
    num_envs: int,
    device: str,
) -> AssetPoseSample:
    surface_z = surfaces[option_cfg.surface].z + option_cfg.support_height_from_surface_m
    x, y, yaw = _sample_planar_bounds(num_envs, option_cfg.planar_bounds, device)
    quat = _make_quat(
        num_envs=num_envs,
        device=device,
        yaw=yaw,
        base_quat_wxyz=object_cfg.base_quat_wxyz,
    )
    support_pos = torch.stack(
        [x, y, torch.full_like(x, surface_z + object_cfg.root_height_from_support_m)],
        dim=-1,
    )
    local_support_point = torch.tensor(
        object_cfg.local_support_point_m,
        device=device,
        dtype=quat.dtype,
    ).repeat(num_envs, 1)
    pos = support_pos - quat_apply(quat, local_support_point)
    valid = torch.ones(num_envs, dtype=torch.bool, device=device)
    for constraint in option_cfg.min_distance_from:
        dist = _pairwise_planar_dist(support_pos[:, :2], placed[constraint.other_asset].pos)
        valid &= dist > constraint.min_distance_m
        if constraint.max_distance_m is not None:
            valid &= dist < constraint.max_distance_m

    for _ in range(128):
        if bool(torch.all(valid)):
            break
        rx, ry, ryaw = _sample_planar_bounds(num_envs, option_cfg.planar_bounds, device)
        yaw[~valid] = ryaw[~valid]
        quat[~valid] = _make_quat(
            num_envs=num_envs,
            device=device,
            yaw=yaw,
            base_quat_wxyz=object_cfg.base_quat_wxyz,
        )[~valid]
        support_pos[~valid, 0] = rx[~valid]
        support_pos[~valid, 1] = ry[~valid]
        pos[~valid] = support_pos[~valid] - quat_apply(quat[~valid], local_support_point[~valid])
        valid = torch.ones(num_envs, dtype=torch.bool, device=device)
        for constraint in option_cfg.min_distance_from:
            dist = _pairwise_planar_dist(support_pos[:, :2], placed[constraint.other_asset].pos)
            valid &= dist > constraint.min_distance_m
            if constraint.max_distance_m is not None:
                valid &= dist < constraint.max_distance_m
    return AssetPoseSample(pos=pos, quat=quat, yaw=yaw)


def _sample_zone_object_pose(
    object_cfg: ObjectRandomizationCfg,
    option_cfg: ZonePlacementOptionCfg,
    fixture_cfgs: dict[str, FixtureRandomizationCfg],
    placed: dict[str, AssetPoseSample],
    surfaces: dict[str, object],
    num_envs: int,
    device: str,
) -> AssetPoseSample:
    parent_pose = placed[option_cfg.parent_asset]
    parent_cfg = fixture_cfgs[option_cfg.parent_asset]
    zone = _lookup_zone(parent_cfg, option_cfg.zone_name)
    surface_z = surfaces[parent_cfg.surface].z + zone.support_height_from_surface_m

    point_index = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    if zone.discrete_local_xy_points_m:
        points = torch.tensor(zone.discrete_local_xy_points_m, device=device, dtype=parent_pose.pos.dtype)
        point_index = torch.randint(points.shape[0], (num_envs,), device=device)
        local_xy = points[point_index]
    else:
        local_xy = torch.empty(num_envs, 2, device=device)
        local_xy[:, 0].uniform_(
            zone.local_xy_center_m[0] - zone.local_xy_halfspan_m[0],
            zone.local_xy_center_m[0] + zone.local_xy_halfspan_m[0],
        )
        local_xy[:, 1].uniform_(
            zone.local_xy_center_m[1] - zone.local_xy_halfspan_m[1],
            zone.local_xy_center_m[1] + zone.local_xy_halfspan_m[1],
        )
    rel_yaw = torch.empty(num_envs, device=device).uniform_(*zone.yaw_range_rel)

    cos_yaw = torch.cos(parent_pose.yaw)
    sin_yaw = torch.sin(parent_pose.yaw)
    rotated_x = cos_yaw * local_xy[:, 0] - sin_yaw * local_xy[:, 1]
    rotated_y = sin_yaw * local_xy[:, 0] + cos_yaw * local_xy[:, 1]

    support_pos = torch.zeros(num_envs, 3, device=device)
    support_pos[:, 0] = parent_pose.pos[:, 0] + rotated_x
    support_pos[:, 1] = parent_pose.pos[:, 1] + rotated_y
    support_pos[:, 2] = surface_z + object_cfg.root_height_from_support_m
    yaw = parent_pose.yaw + rel_yaw if zone.inherit_parent_yaw else rel_yaw
    quat = _make_quat(
        num_envs=num_envs,
        device=device,
        yaw=yaw,
        base_quat_wxyz=object_cfg.base_quat_wxyz,
    )
    local_support_point = torch.tensor(
        object_cfg.local_support_point_m,
        device=device,
        dtype=quat.dtype,
    ).repeat(num_envs, 1)
    pos = support_pos - quat_apply(quat, local_support_point)
    return AssetPoseSample(pos=pos, quat=quat, yaw=yaw), point_index


def sample_task_layout(task_cfg: TaskRandomizationCfg, *, num_envs: int, device: str):
    fixture_cfgs = {cfg.name: cfg for cfg in task_cfg.fixtures}
    placed: dict[str, AssetPoseSample] = _sample_fixture_poses(task_cfg, num_envs=num_envs, device=device)
    metadata: dict[str, torch.Tensor] = {}

    for object_cfg in task_cfg.objects:
        probs = torch.tensor(object_cfg.placement_option_probs, device=device, dtype=torch.float32)
        option_idx = torch.multinomial(probs.expand(num_envs, -1), num_samples=1).squeeze(-1)
        pos = torch.zeros(num_envs, 3, device=device)
        quat = torch.zeros(num_envs, 4, device=device)
        yaw = torch.zeros(num_envs, device=device)

        for i, option_cfg in enumerate(object_cfg.placement_options):
            mask = option_idx == i
            if not bool(torch.any(mask)):
                continue

            if isinstance(option_cfg, SurfacePlacementOptionCfg):
                sample = _sample_surface_object_pose(object_cfg, option_cfg, task_cfg.surfaces, placed, num_envs, device)
            elif isinstance(option_cfg, ZonePlacementOptionCfg):
                sample, point_index = _sample_zone_object_pose(
                    object_cfg,
                    option_cfg,
                    fixture_cfgs,
                    placed,
                    task_cfg.surfaces,
                    num_envs,
                    device,
                )
                point_key = f"{object_cfg.name}_{option_cfg.name}_point_index"
                if point_key not in metadata:
                    metadata[point_key] = torch.full((num_envs,), -1, device=device, dtype=torch.long)
                metadata[point_key][mask] = point_index[mask]
            else:
                raise TypeError(f"Unsupported placement option: {type(option_cfg)}")

            pos[mask] = sample.pos[mask]
            quat[mask] = sample.quat[mask]
            yaw[mask] = sample.yaw[mask]

        placed[object_cfg.name] = AssetPoseSample(pos=pos, quat=quat, yaw=yaw)
        metadata[f"{object_cfg.name}_placement_index"] = option_idx

    return placed, metadata


def apply_task_randomization(scene, assets: dict[str, object], task_cfg: TaskRandomizationCfg):
    placed, metadata = sample_task_layout(task_cfg, num_envs=scene.num_envs, device=scene.device)
    env_origins = scene.env_origins
    object_cfgs = {cfg.name: cfg for cfg in task_cfg.objects}

    for asset_name, asset in assets.items():
        sample = placed[asset_name]
        spawn_pos = sample.pos.clone()
        object_cfg = object_cfgs.get(asset_name)
        if object_cfg is not None and object_cfg.spawn_height_offset_m != 0.0:
            spawn_pos[:, 2] += object_cfg.spawn_height_offset_m
        world_pose = torch.cat([spawn_pos + env_origins, sample.quat], dim=-1)
        asset.write_root_pose_to_sim(world_pose)
        # Zero velocity only for dynamic (non-kinematic) rigid bodies.
        # Kinematic RigidObjects and Articulations reject PhysX velocity writes.
        if isinstance(asset, RigidObject) and not asset.cfg.spawn.rigid_props.kinematic_enabled:
            zero_vel = torch.zeros(scene.num_envs, 6, device=scene.device)
            asset.write_root_velocity_to_sim(zero_vel)

    return {
        "poses": placed,
        "metadata": metadata,
    }


def apply_scene_randomization(scene, robot, holder, vortexer, tube):
    """Backward-compatible wrapper for the current pick-place task."""
    return apply_task_randomization(
        scene,
        {
            "robot": robot,
            "holder": holder,
            "vortexer": vortexer,
            "tube": tube,
        },
        PICK_PLACE_RANDOMIZATION,
    )
