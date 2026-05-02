import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union


BENCHMARK_ROOT = Path(__file__).resolve().parent
HOLDER_XY_SCALE_FACTOR = 1.14

RAW_HOLDER_SLOT_CENTERS_LOCAL_M = (
    (0.016005, -0.016205, 0.000),
    (0.053005, -0.016205, 0.000),
    (0.090005, -0.016205, 0.000),
    (0.127005, -0.016205, 0.000),
)
RAW_HOLDER_SUPPORT_ZONE_CENTER_LOCAL_XY_M = (0.071505, -0.016205)
RAW_HOLDER_SUPPORT_ZONE_HALFSPAN_XY_M = (0.0555, 0.0110)


def _scale_xy(point: tuple[float, float], scale: float = HOLDER_XY_SCALE_FACTOR) -> tuple[float, float]:
    return (point[0] * scale, point[1] * scale)


def _scale_xy_xyz(point: tuple[float, float, float], scale: float = HOLDER_XY_SCALE_FACTOR) -> tuple[float, float, float]:
    return (point[0] * scale, point[1] * scale, point[2])


def _resolve_candidate(*relative_paths: str) -> Path:
    """Return the first existing candidate path, else the first candidate as default."""
    candidates = [BENCHMARK_ROOT / rel for rel in relative_paths]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_override_path(env_var: str) -> Path | None:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = BENCHMARK_ROOT / candidate
    return candidate


@dataclass(frozen=True)
class AssetPaths:
    xarm6_urdf: Path = _resolve_candidate(
        "basic_assets/xarm6_with_gripper.urdf",
        "basic_assets/xarm6.urdf",
        "xarm_ros2/xarm6.urdf",
        "assets/xarm6.urdf",
    )
    xarm6_usd: Path = _resolve_override_path("WETLAB_XARM6_USD_OVERRIDE") or _resolve_candidate(
        "basic_assets/xarm6_with_gripper.usd",
        "basic_assets/xarm6_with_gripper.usda",
        "basic_assets/xarm6.usd",
        "basic_assets/xarm6.usda",
        "xarm_ros2/xarm6.usd",
        "xarm_ros2/xarm6.usda",
        "xarm_ros2/xarm6/xarm6.usd",
        "xarm_ros2/xarm6/xarm6.usda",
        "assets/xarm6.usd",
    )
    vortexer_usd: Path = _resolve_candidate(
        "basic_assets/Vortexer_rev.usd",
        "basic_assets/vortexer.usd",
        "assets/vortexer.usd",
    )
    tube_holder_combo_usd: Path = _resolve_candidate(
        "basic_assets/4_50ml_Conical_Holder.usd",
        "basic_assets/tube_holder_combo.usd",
        "assets/tube_holder_combo.usd",
    )
    tube_usd: Path = _resolve_candidate(
        "basic_assets/50ml Conical EP Tube.usd",
        "basic_assets/tube.usd",
        "assets/tube.usd",
    )


@dataclass(frozen=True)
class TaskFrames:
    ee_body_name: str = "xarm_gripper_base_link"
    ee_control_offset_local_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_tool_offset_local_m: tuple[float, float, float] = (0.0, 0.0, 0.172)
    arm_joint_regex: str = "joint[1-6]"
    left_finger_joint: str = "left_finger_joint"
    right_finger_joint: str = "right_finger_joint"
    tube_grasp_frame: str = "grasp_frame"
    holder_slot_prefix: str = "slot_"
    # Placeholder wrist orientation for a vertical top grasp, matching the
    # MoveIt scaffold's RPY(pi, 0, 0) assumption.
    ee_top_grasp_quat_wxyz: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    # xArm gripper finger joint angles (revolute, radians).
    # Verify against: grep -A 8 'name="left_finger_joint"' xarm6_with_gripper.urdf | grep limit
    # Standard xArm gripper: lower=0.0 (open), upper=0.85 (closed).
    gripper_open_pos: float = 0.0
    gripper_closed_pos: float = 0.85


# Contact reference for the real G1 / AG1002 fingertip pad. This is not the
# raw steel finger inner wall; it is the nominal center of the effective
# fingertip contact band, which sits slightly inward and higher toward the tip.
GRIPPER_PAD_CONTACT_LOCAL_Y_M = 0.0115
GRIPPER_PAD_CONTACT_LOCAL_Z_M = 0.050


@dataclass(frozen=True)
class PlanarPoseBoundsCfg:
    x: tuple[float, float]
    y: tuple[float, float]
    yaw: tuple[float, float]


@dataclass(frozen=True)
class SurfaceCfg:
    z: float


@dataclass(frozen=True)
class DistanceConstraintCfg:
    other_asset: str
    min_distance_m: float
    max_distance_m: float | None = None


@dataclass(frozen=True)
class SupportZoneCfg:
    name: str
    local_xy_center_m: tuple[float, float] = (0.0, 0.0)
    local_xy_halfspan_m: tuple[float, float] = (0.0, 0.0)
    discrete_local_xy_points_m: tuple[tuple[float, float], ...] = ()
    support_height_from_surface_m: float = 0.0
    yaw_range_rel: tuple[float, float] = (-3.14159, 3.14159)
    inherit_parent_yaw: bool = True


@dataclass(frozen=True)
class FixtureRandomizationCfg:
    name: str
    surface: str
    planar_bounds: PlanarPoseBoundsCfg
    root_height_from_surface_m: float = 0.0
    base_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    min_distance_from: tuple[DistanceConstraintCfg, ...] = ()
    support_zones: tuple[SupportZoneCfg, ...] = ()


@dataclass(frozen=True)
class SurfacePlacementOptionCfg:
    name: str
    surface: str
    planar_bounds: PlanarPoseBoundsCfg
    support_height_from_surface_m: float = 0.0
    min_distance_from: tuple[DistanceConstraintCfg, ...] = ()


@dataclass(frozen=True)
class ZonePlacementOptionCfg:
    name: str
    parent_asset: str
    zone_name: str


PlacementOptionCfg = Union[SurfacePlacementOptionCfg, ZonePlacementOptionCfg]


@dataclass(frozen=True)
class ObjectRandomizationCfg:
    name: str
    root_height_from_support_m: float = 0.0
    spawn_height_offset_m: float = 0.0
    local_support_point_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    placement_options: tuple[PlacementOptionCfg, ...] = ()
    placement_option_probs: tuple[float, ...] = ()


@dataclass(frozen=True)
class TaskRandomizationCfg:
    surfaces: dict[str, SurfaceCfg]
    fixtures: tuple[FixtureRandomizationCfg, ...]
    objects: tuple[ObjectRandomizationCfg, ...]


@dataclass(frozen=True)
class ImportedLabAssetGeometry:
    # These three assets were inspected from the local USDs in basic_assets/.
    # They report Y-up metadata, but their actual height extends along +Z already,
    # so we keep the spawn rotation as identity and only normalize scale/offsets.
    # USDs store geometry in mm; main Isaac Lab stage is metersPerUnit=1.0, so
    # the effective scale is 1mm × 0.001 = 0.001 m/unit.
    scale: tuple[float, float, float] = (0.001, 0.001, 0.001)
    holder_scale: tuple[float, float, float] = (
        0.001 * HOLDER_XY_SCALE_FACTOR,
        0.001 * HOLDER_XY_SCALE_FACTOR,
        0.001,
    )
    upright_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    holder_bottom_from_root_m: float = 0.004
    holder_top_from_root_m: float = 0.055
    holder_slot_centers_local_m: tuple[tuple[float, float, float], ...] = (
        # Slot centers are open hollows whose support floor is at local z = 0.
        # The holder is widened in x/y only, so keep z unchanged and scale the
        # authored slot coordinates to match the spawned geometry.
        tuple(_scale_xy_xyz(point) for point in RAW_HOLDER_SLOT_CENTERS_LOCAL_M)
    )
    holder_support_zone_center_local_xy_m: tuple[float, float] = _scale_xy(RAW_HOLDER_SUPPORT_ZONE_CENTER_LOCAL_XY_M)
    holder_support_zone_halfspan_xy_m: tuple[float, float] = _scale_xy(RAW_HOLDER_SUPPORT_ZONE_HALFSPAN_XY_M)
    # The current tube USD is already authored with its visible bottom face at
    # local z = 0 and its top at local z = 117 mm. Keep the support/grasp math
    # aligned to that visible mesh so the tube does not appear to hover above
    # fixtures while actually resting on hidden collision geometry below it.
    tube_bottom_from_root_m: float = 0.0
    tube_top_from_root_m: float = 0.117
    tube_center_local_xy_m: tuple[float, float] = (0.01775, 0.017748)
    vortexer_bottom_from_root_m: float = 0.010
    vortexer_top_from_root_m: float = 0.160
    # The raw mesh has a tiny raised center boss at z = 145 mm, but the visible
    # cup floor that actually supports the tube body is the broader annulus at
    # about z = 130 mm. Target the annular floor so the tube seats inside the
    # hollow instead of appearing to balance on the center boss.
    vortexer_support_center_local_m: tuple[float, float, float] = (0.060, 0.035, 0.130)
    vortexer_support_zone_center_local_xy_m: tuple[float, float] = (0.060, 0.035)
    vortexer_support_zone_halfspan_xy_m: tuple[float, float] = (0.02161, 0.02162)


@dataclass(frozen=True)
class SupportSurface:
    z: float = 0.78
    robot_root_height_from_table_m: float = 0.0
    holder_root_height_from_table_m: float = 0.004
    vortexer_root_height_from_table_m: float = 0.010
    tube_root_height_from_support_m: float = 0.0
    # Scene initialization needs a deterministic seated pose, not a dynamic drop.
    # Spawning directly on the support avoids slot-edge bounce and keeps
    # scene_check focused on frame / collision correctness rather than rigid-body settling.
    tube_spawn_height_above_support_m: float = 0.0
    # After widening the holder in x/y, keep the tube slightly higher so the
    # occupied slot reads as seated rather than visibly intersecting the walls.
    holder_tube_support_height_from_table_m: float = 0.003
    # Vortexer visible well floor: local z = 130 mm, plus the 10 mm fixture root.
    vortexer_tube_support_height_from_table_m: float = 0.140


@dataclass(frozen=True)
class PlacementHeuristics:
    robot_bounds: PlanarPoseBoundsCfg = PlanarPoseBoundsCfg(
        x=(-0.10, 0.10),
        y=(-0.10, 0.10),
        yaw=(-0.35, 0.35),
    )
    vortexer_bounds: PlanarPoseBoundsCfg = PlanarPoseBoundsCfg(
        x=(0.30, 0.45),
        y=(-0.25, 0.25),
        yaw=(-0.40, 0.40),
    )
    holder_bounds: PlanarPoseBoundsCfg = PlanarPoseBoundsCfg(
        x=(0.30, 0.55),
        y=(-0.20, 0.20),
        yaw=(-0.50, 0.50),
    )
    tube_table_bounds: PlanarPoseBoundsCfg = PlanarPoseBoundsCfg(
        x=(0.22, 0.60),
        y=(-0.28, 0.28),
        yaw=(-3.14159, 3.14159),
    )
    min_fixture_separation_m: float = 0.18
    min_robot_fixture_separation_m: float = 0.20
    max_robot_holder_distance_m: float = 0.58
    max_robot_vortexer_distance_m: float = 0.58
    max_robot_table_tube_distance_m: float = 0.58
    min_table_tube_fixture_clearance_m: float = 0.10
    holder_support_xy_halfspan_m: tuple[float, float] = _scale_xy(RAW_HOLDER_SUPPORT_ZONE_HALFSPAN_XY_M)
    vortexer_support_xy_halfspan_m: tuple[float, float] = (0.02161, 0.02162)
    # Tube always starts in the holder — the two combos differ only in where it stops.
    tube_source_probs: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # Combo 0: holder → holder (simple pick-and-return).
    # Combo 1: holder → vortexer → holder (intermediate vortex stop).
    task_combo_probs: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True)
class TaskThresholds:
    approach_pos_tol_m: float = 0.025
    ee_pos_tol_m: float = 0.03
    ee_rot_tol_rad: float = 0.08
    grasp_settle_steps: int = 25
    lift_height_m: float = 0.10
    retreat_height_m: float = 0.12
    max_episode_steps: int = 3000
    drop_height_m: float = 0.74


@dataclass(frozen=True)
class ReachabilityCfg:
    # Conservative robot-base-frame workspace band for task waypoints.
    # This is not a full IK proof, but it filters layouts whose policy goals are
    # obviously outside the xArm's intended working envelope.
    tolerance_m: float = 0.02
    min_goal_planar_distance_m: float = 0.08
    max_goal_planar_distance_m: float = 0.58
    min_goal_height_rel_base_m: float = -0.02
    max_goal_height_rel_base_m: float = 0.32


@dataclass(frozen=True)
class SceneCheckCfg:
    settle_steps: int = 300
    pos_tol_m: float = 0.02
    z_tol_m: float = 0.015
    upright_tilt_tol_rad: float = 0.25
    contact_force_threshold_n: float = 1.0e-4


ASSETS = AssetPaths()
FRAMES = TaskFrames()
IMPORTED_LAB_ASSETS = ImportedLabAssetGeometry()
SURFACE = SupportSurface()
PLACEMENT = PlacementHeuristics()
THRESH = TaskThresholds()
REACHABILITY = ReachabilityCfg()
SCENE_CHECK = SceneCheckCfg()

TUBE_SOURCE_NAMES = ("holder", "vortexer", "table")


def with_tube_source(task_cfg: TaskRandomizationCfg, tube_source: str) -> TaskRandomizationCfg:
    if tube_source == "all":
        return task_cfg
    if tube_source not in TUBE_SOURCE_NAMES:
        raise ValueError(f"Unsupported tube source '{tube_source}'. Expected one of: all, {', '.join(TUBE_SOURCE_NAMES)}")

    source_index = TUBE_SOURCE_NAMES.index(tube_source)
    updated_objects = []
    for object_cfg in task_cfg.objects:
        if object_cfg.name != "tube":
            updated_objects.append(object_cfg)
            continue
        probs = tuple(1.0 if idx == source_index else 0.0 for idx in range(len(object_cfg.placement_options)))
        updated_objects.append(replace(object_cfg, placement_option_probs=probs))
    return replace(task_cfg, objects=tuple(updated_objects))

PICK_PLACE_RANDOMIZATION = TaskRandomizationCfg(
    surfaces={
        "table": SurfaceCfg(z=SURFACE.z),
    },
    fixtures=(
        FixtureRandomizationCfg(
            name="robot",
            surface="table",
            planar_bounds=PLACEMENT.robot_bounds,
            root_height_from_surface_m=SURFACE.robot_root_height_from_table_m,
            min_distance_from=(
                DistanceConstraintCfg(
                    "holder",
                    PLACEMENT.min_robot_fixture_separation_m,
                    PLACEMENT.max_robot_holder_distance_m,
                ),
                DistanceConstraintCfg(
                    "vortexer",
                    PLACEMENT.min_robot_fixture_separation_m,
                    PLACEMENT.max_robot_vortexer_distance_m,
                ),
            ),
        ),
        FixtureRandomizationCfg(
            name="holder",
            surface="table",
            planar_bounds=PLACEMENT.holder_bounds,
            root_height_from_surface_m=SURFACE.holder_root_height_from_table_m,
            base_quat_wxyz=IMPORTED_LAB_ASSETS.upright_quat_wxyz,
            min_distance_from=(
                DistanceConstraintCfg("vortexer", PLACEMENT.min_fixture_separation_m),
            ),
            support_zones=(
                SupportZoneCfg(
                    name="tube_support",
                    local_xy_center_m=IMPORTED_LAB_ASSETS.holder_support_zone_center_local_xy_m,
                    local_xy_halfspan_m=PLACEMENT.holder_support_xy_halfspan_m,
                    discrete_local_xy_points_m=tuple(
                        (slot[0], slot[1]) for slot in IMPORTED_LAB_ASSETS.holder_slot_centers_local_m
                    ),
                    support_height_from_surface_m=SURFACE.holder_tube_support_height_from_table_m,
                    yaw_range_rel=(0.0, 0.0),
                ),
            ),
        ),
        FixtureRandomizationCfg(
            name="vortexer",
            surface="table",
            planar_bounds=PLACEMENT.vortexer_bounds,
            root_height_from_surface_m=SURFACE.vortexer_root_height_from_table_m,
            base_quat_wxyz=IMPORTED_LAB_ASSETS.upright_quat_wxyz,
            support_zones=(
                SupportZoneCfg(
                    name="tube_support",
                    local_xy_center_m=IMPORTED_LAB_ASSETS.vortexer_support_zone_center_local_xy_m,
                    local_xy_halfspan_m=PLACEMENT.vortexer_support_xy_halfspan_m,
                    discrete_local_xy_points_m=(
                        (
                            IMPORTED_LAB_ASSETS.vortexer_support_center_local_m[0],
                            IMPORTED_LAB_ASSETS.vortexer_support_center_local_m[1],
                        ),
                    ),
                    support_height_from_surface_m=SURFACE.vortexer_tube_support_height_from_table_m,
                    yaw_range_rel=(0.0, 0.0),
                ),
            ),
        ),
    ),
    objects=(
        ObjectRandomizationCfg(
            name="tube",
            placement_option_probs=PLACEMENT.tube_source_probs,
            root_height_from_support_m=SURFACE.tube_root_height_from_support_m,
            spawn_height_offset_m=SURFACE.tube_spawn_height_above_support_m,
            local_support_point_m=(
                IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
                IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
                IMPORTED_LAB_ASSETS.tube_bottom_from_root_m,
            ),
            base_quat_wxyz=IMPORTED_LAB_ASSETS.upright_quat_wxyz,
            placement_options=(
                ZonePlacementOptionCfg(name="holder", parent_asset="holder", zone_name="tube_support"),
                ZonePlacementOptionCfg(name="vortexer", parent_asset="vortexer", zone_name="tube_support"),
                SurfacePlacementOptionCfg(
                    name="table",
                    surface="table",
                    planar_bounds=PLACEMENT.tube_table_bounds,
                    support_height_from_surface_m=0.0,
                    min_distance_from=(
                        DistanceConstraintCfg("robot", 0.0, PLACEMENT.max_robot_table_tube_distance_m),
                        DistanceConstraintCfg("holder", PLACEMENT.min_table_tube_fixture_clearance_m),
                        DistanceConstraintCfg("vortexer", PLACEMENT.min_table_tube_fixture_clearance_m),
                    ),
                ),
            ),
        ),
    ),
)
