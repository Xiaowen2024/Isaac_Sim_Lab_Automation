from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

from wetlab_benchmark.task_config import FRAMES, IMPORTED_LAB_ASSETS, PICK_PLACE_RANDOMIZATION, PLACEMENT, SURFACE, THRESH


PRE_SUPPORT_CLEARANCE_M = 0.065
TRANSFER_CLEARANCE_M = 0.18
VORTEXER_PREPLACE_CLEARANCE_M = 0.030
VORTEXER_TRANSFER_CLEARANCE_M = 0.090
VORTEXER_RELEASE_INSERTION_BELOW_RIM_M = 0.024

TABLE_SIZE_M = (1.20, 0.90, 0.04)
TABLE_CENTER_WORLD_M = (0.25, 0.0, SURFACE.z - 0.02)

HOLDER_BOX_SIZE_M = (
    2.0 * (IMPORTED_LAB_ASSETS.holder_support_zone_halfspan_xy_m[0] + 0.028),
    2.0 * (IMPORTED_LAB_ASSETS.holder_support_zone_halfspan_xy_m[1] + 0.018),
    IMPORTED_LAB_ASSETS.holder_top_from_root_m - IMPORTED_LAB_ASSETS.holder_bottom_from_root_m,
)
HOLDER_BOX_CENTER_LOCAL_M = (
    IMPORTED_LAB_ASSETS.holder_support_zone_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.holder_support_zone_center_local_xy_m[1],
    0.5 * (IMPORTED_LAB_ASSETS.holder_top_from_root_m + IMPORTED_LAB_ASSETS.holder_bottom_from_root_m),
)
VORTEXER_BOX_SIZE_M = (
    0.13,
    0.11,
    IMPORTED_LAB_ASSETS.vortexer_top_from_root_m - IMPORTED_LAB_ASSETS.vortexer_bottom_from_root_m,
)
VORTEXER_BOX_CENTER_LOCAL_M = (
    IMPORTED_LAB_ASSETS.vortexer_support_center_local_m[0],
    IMPORTED_LAB_ASSETS.vortexer_support_center_local_m[1],
    0.5 * (IMPORTED_LAB_ASSETS.vortexer_top_from_root_m + IMPORTED_LAB_ASSETS.vortexer_bottom_from_root_m),
)

TUBE_RADIUS_M = 0.018
TUBE_HEIGHT_M = IMPORTED_LAB_ASSETS.tube_top_from_root_m - IMPORTED_LAB_ASSETS.tube_bottom_from_root_m
TUBE_CENTER_LOCAL_M = (
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
    0.5 * (IMPORTED_LAB_ASSETS.tube_top_from_root_m + IMPORTED_LAB_ASSETS.tube_bottom_from_root_m),
)
TUBE_SUPPORT_LOCAL_M = (
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
    IMPORTED_LAB_ASSETS.tube_bottom_from_root_m,
)
TUBE_TOP_LOCAL_M = (
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
    IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
    IMPORTED_LAB_ASSETS.tube_top_from_root_m,
)
TOP_GRASP_QUAT_WXYZ = FRAMES.ee_top_grasp_quat_wxyz
# Target the rigid cap/shoulder band instead of the thinner upper tube wall.
# Keep the grasp shallow, but not so shallow that the tube never meets the
# fingertip pad during closure. This is a midpoint between the old deeper tube
# wall grasp and the failed ultra-shallow cap-only branch.
GRASP_TOOL_CLEARANCE_M = -0.007
PHYSICAL_VERIFY_LIFT_HEIGHT_M = 0.03
# The closed contact pads sit almost directly below the reported TCP in local
# gripper coordinates. Keep this geometric compensation separate from the
# pickup-only lateral bias below, which exists to cancel the real sideways tube
# shove observed during physical closure on the imported xArm gripper.
GRIPPER_CLOSED_PAD_CENTER_SHIFT_LOCAL_M = (0.0, 0.0, -0.030)
GRIPPER_GRASP_TOOL_COMPENSATION_LOCAL_M = tuple(-value for value in GRIPPER_CLOSED_PAD_CENTER_SHIFT_LOCAL_M)
# Keep the top grasp centered while the finger contact geometry is narrowed
# toward a more realistic fingertip pad band.
PICKUP_GRASP_BIAS_LOCAL_M = (0.0, 0.0, 0.0)
EE_TOOL_OFFSET_LOCAL_M = FRAMES.ee_tool_offset_local_m
# The imported support-zone metadata is a placement tolerance, not a free-space
# cavity.  The holder Y halfspan is smaller than the tube radius, which makes
# MoveIt think an intentionally seated/attached tube starts inside a solid wall.
# Keep the real Isaac contact assets authoritative, but give the planner a
# coarse open cavity with enough clearance for insertion/extraction.
PLANNING_CAVITY_HALFSPAN_MIN_M = (0.035, 0.035)


@dataclass(frozen=True)
class PoseWxyz:
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        return (self.qw, self.qx, self.qy, self.qz)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qw": self.qw,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "PoseWxyz":
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            z=float(payload["z"]),
            qw=float(payload["qw"]),
            qx=float(payload["qx"]),
            qy=float(payload["qy"]),
            qz=float(payload["qz"]),
        )


@dataclass(frozen=True)
class TaskPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    @classmethod
    def from_pose_wxyz(cls, pose_wxyz: PoseWxyz) -> "TaskPose":
        return cls(
            x=pose_wxyz.x,
            y=pose_wxyz.y,
            z=pose_wxyz.z,
            qx=pose_wxyz.qx,
            qy=pose_wxyz.qy,
            qz=pose_wxyz.qz,
            qw=pose_wxyz.qw,
        )

    def to_pose_stamped(self, frame_id: str) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = self.z
        msg.pose.orientation.x = self.qx
        msg.pose.orientation.y = self.qy
        msg.pose.orientation.z = self.qz
        msg.pose.orientation.w = self.qw
        return msg


@dataclass(frozen=True)
class PrimitiveSpec:
    object_id: str
    primitive_type: int
    dimensions: tuple[float, ...]
    pose: TaskPose


@dataclass(frozen=True)
class LegSpec:
    label: str
    dest_name: str
    dest_support_w: PoseWxyz
    dest_fixture_w: PoseWxyz


@dataclass(frozen=True)
class LiveLegPlan:
    label: str
    dest_name: str
    pregrasp: TaskPose
    grasp: TaskPose
    lift: TaskPose
    transit: TaskPose
    preplace: TaskPose
    place: TaskPose
    retreat: TaskPose


@dataclass(frozen=True)
class LiveTask:
    seed: int
    task_mode: int
    holder_slot_index: int
    robot_pose_w: PoseWxyz
    holder_pose_w: PoseWxyz
    vortexer_pose_w: PoseWxyz
    initial_tube_root_w: PoseWxyz
    scene_objects: tuple[PrimitiveSpec, ...]
    legs: tuple[LegSpec, ...]


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_apply(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    qvec = (0.0, v[0], v[1], v[2])
    rotated = _quat_mul(_quat_mul(q, qvec), _quat_conjugate(q))
    return (rotated[1], rotated[2], rotated[3])


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _top_grasp_quat_wxyz() -> tuple[float, float, float, float]:
    yaw_deg = float(os.environ.get("WETLAB_TOP_GRASP_TOOL_YAW_DEG", "0.0"))
    if abs(yaw_deg) <= 1.0e-9:
        return TOP_GRASP_QUAT_WXYZ
    return _quat_mul(TOP_GRASP_QUAT_WXYZ, _yaw_to_quat_wxyz(math.radians(yaw_deg)))


def _pose_from_local_point(
    root_pose_w: PoseWxyz,
    local_xyz_m: tuple[float, float, float],
    local_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> PoseWxyz:
    offset = _quat_apply(root_pose_w.quat_wxyz, local_xyz_m)
    quat = _quat_mul(root_pose_w.quat_wxyz, local_quat_wxyz)
    return PoseWxyz(
        x=root_pose_w.x + offset[0],
        y=root_pose_w.y + offset[1],
        z=root_pose_w.z + offset[2],
        qw=quat[0],
        qx=quat[1],
        qy=quat[2],
        qz=quat[3],
    )


def _pose_in_robot_frame(world_pose: PoseWxyz, robot_pose_w: PoseWxyz) -> PoseWxyz:
    robot_quat_inv = _quat_conjugate(robot_pose_w.quat_wxyz)
    rel_pos = _quat_apply(
        robot_quat_inv,
        (
            world_pose.x - robot_pose_w.x,
            world_pose.y - robot_pose_w.y,
            world_pose.z - robot_pose_w.z,
        ),
    )
    rel_quat = _quat_mul(robot_quat_inv, world_pose.quat_wxyz)
    return PoseWxyz(
        x=rel_pos[0],
        y=rel_pos[1],
        z=rel_pos[2],
        qw=rel_quat[0],
        qx=rel_quat[1],
        qy=rel_quat[2],
        qz=rel_quat[3],
    )


def _root_pose_from_support_point(
    support_pos_w: tuple[float, float, float],
    fixture_quat_w: tuple[float, float, float, float],
    local_support_point_m: tuple[float, float, float],
) -> PoseWxyz:
    local_support = _quat_apply(fixture_quat_w, local_support_point_m)
    return PoseWxyz(
        x=support_pos_w[0] - local_support[0],
        y=support_pos_w[1] - local_support[1],
        z=support_pos_w[2] - local_support[2],
        qw=fixture_quat_w[0],
        qx=fixture_quat_w[1],
        qy=fixture_quat_w[2],
        qz=fixture_quat_w[3],
    )


def _offset_world_z(pose: PoseWxyz, dz: float) -> PoseWxyz:
    return PoseWxyz(
        x=pose.x,
        y=pose.y,
        z=pose.z + dz,
        qw=pose.qw,
        qx=pose.qx,
        qy=pose.qy,
        qz=pose.qz,
    )


def _planar_distance(a: PoseWxyz, b: PoseWxyz) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _sample_fixture_pose(fixture_cfg, rng: random.Random) -> PoseWxyz:
    x = rng.uniform(*fixture_cfg.planar_bounds.x)
    y = rng.uniform(*fixture_cfg.planar_bounds.y)
    yaw = rng.uniform(*fixture_cfg.planar_bounds.yaw)
    yaw_quat = _yaw_to_quat_wxyz(yaw)
    quat = _quat_mul(yaw_quat, fixture_cfg.base_quat_wxyz)
    z = PICK_PLACE_RANDOMIZATION.surfaces[fixture_cfg.surface].z + fixture_cfg.root_height_from_surface_m
    return PoseWxyz(x=x, y=y, z=z, qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])


def _sample_fixture_layout(rng: random.Random) -> dict[str, PoseWxyz]:
    for _ in range(512):
        placed = {
            fixture_cfg.name: _sample_fixture_pose(fixture_cfg, rng)
            for fixture_cfg in PICK_PLACE_RANDOMIZATION.fixtures
        }
        valid = True
        for fixture_cfg in PICK_PLACE_RANDOMIZATION.fixtures:
            pose = placed[fixture_cfg.name]
            for constraint in fixture_cfg.min_distance_from:
                other_pose = placed[constraint.other_asset]
                planar_dist = _planar_distance(pose, other_pose)
                if planar_dist < constraint.min_distance_m:
                    valid = False
                    break
                if constraint.max_distance_m is not None and planar_dist > constraint.max_distance_m:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return placed
    raise RuntimeError("Failed to sample a valid robot/holder/vortexer layout after 512 attempts")


def holder_slot_support_pose(holder_pose_w: PoseWxyz, slot_index: int) -> PoseWxyz:
    return _pose_from_local_point(
        holder_pose_w,
        IMPORTED_LAB_ASSETS.holder_slot_centers_local_m[slot_index],
        _top_grasp_quat_wxyz(),
    )


def vortexer_support_pose(vortexer_pose_w: PoseWxyz) -> PoseWxyz:
    return _pose_from_local_point(
        vortexer_pose_w,
        IMPORTED_LAB_ASSETS.vortexer_support_center_local_m,
        _top_grasp_quat_wxyz(),
    )


def vortexer_release_pose(vortexer_pose_w: PoseWxyz) -> PoseWxyz:
    return _pose_from_local_point(
        vortexer_pose_w,
        (
            IMPORTED_LAB_ASSETS.vortexer_support_zone_center_local_xy_m[0],
            IMPORTED_LAB_ASSETS.vortexer_support_zone_center_local_xy_m[1],
            IMPORTED_LAB_ASSETS.vortexer_top_from_root_m - VORTEXER_RELEASE_INSERTION_BELOW_RIM_M,
        ),
        _top_grasp_quat_wxyz(),
    )


def tube_center_pose(tube_root_pose_w: PoseWxyz) -> PoseWxyz:
    return _pose_from_local_point(tube_root_pose_w, TUBE_CENTER_LOCAL_M)


def tube_top_grasp_pose(tube_root_pose_w: PoseWxyz) -> PoseWxyz:
    return _pose_from_local_point(tube_root_pose_w, TUBE_TOP_LOCAL_M, _top_grasp_quat_wxyz())


def _offset_world_xyz(pose: PoseWxyz, offset_xyz_m: tuple[float, float, float]) -> PoseWxyz:
    return PoseWxyz(
        x=pose.x + offset_xyz_m[0],
        y=pose.y + offset_xyz_m[1],
        z=pose.z + offset_xyz_m[2],
        qw=pose.qw,
        qx=pose.qx,
        qy=pose.qy,
        qz=pose.qz,
    )


def _offset_local_xyz(pose: PoseWxyz, offset_xyz_m: tuple[float, float, float]) -> PoseWxyz:
    offset_world = _quat_apply(pose.quat_wxyz, offset_xyz_m)
    return _offset_world_xyz(pose, offset_world)


def compensate_tool_pose_for_closed_gripper(tool_pose_w: PoseWxyz) -> PoseWxyz:
    return _offset_local_xyz(tool_pose_w, GRIPPER_GRASP_TOOL_COMPENSATION_LOCAL_M)


def nominal_tool_pose_from_compensated_grasp(tool_pose_w: PoseWxyz) -> PoseWxyz:
    return _offset_local_xyz(tool_pose_w, GRIPPER_CLOSED_PAD_CENTER_SHIFT_LOCAL_M)


def tube_ee_grasp_pose(tube_root_pose_w: PoseWxyz) -> PoseWxyz:
    return compensate_tool_pose_for_closed_gripper(
        _offset_world_z(tube_top_grasp_pose(tube_root_pose_w), GRASP_TOOL_CLEARANCE_M)
    )


def ee_body_pose_from_tool_pose(tool_pose_w: PoseWxyz) -> PoseWxyz:
    tool_offset_w = _quat_apply(tool_pose_w.quat_wxyz, EE_TOOL_OFFSET_LOCAL_M)
    return PoseWxyz(
        x=tool_pose_w.x - tool_offset_w[0],
        y=tool_pose_w.y - tool_offset_w[1],
        z=tool_pose_w.z - tool_offset_w[2],
        qw=tool_pose_w.qw,
        qx=tool_pose_w.qx,
        qy=tool_pose_w.qy,
        qz=tool_pose_w.qz,
    )


def _lookup_holder_slot_points() -> tuple[tuple[float, float], ...]:
    for fixture_cfg in PICK_PLACE_RANDOMIZATION.fixtures:
        if fixture_cfg.name != "holder":
            continue
        for zone in fixture_cfg.support_zones:
            if zone.name == "tube_support":
                return zone.discrete_local_xy_points_m
    raise RuntimeError("Holder tube_support zone was not found in task config")


def _task_mode_from_arg(mode_arg: str, rng: random.Random) -> int:
    if mode_arg == "a":
        return 0
    if mode_arg == "b":
        return 1
    return rng.choices(population=[0, 1], weights=list(PLACEMENT.task_combo_probs), k=1)[0]


def _box_primitive(object_id: str, world_pose_w: PoseWxyz, dims: tuple[float, float, float], robot_pose_w: PoseWxyz) -> PrimitiveSpec:
    return PrimitiveSpec(
        object_id=object_id,
        primitive_type=SolidPrimitive.BOX,
        dimensions=dims,
        pose=TaskPose.from_pose_wxyz(_pose_in_robot_frame(world_pose_w, robot_pose_w)),
    )


def _open_top_wall_primitives(
    *,
    object_prefix: str,
    fixture_pose_w: PoseWxyz,
    robot_pose_w: PoseWxyz,
    outer_size_m: tuple[float, float, float],
    outer_center_local_m: tuple[float, float, float],
    cavity_center_local_xy_m: tuple[float, float],
    cavity_halfspan_xy_m: tuple[float, float],
) -> list[PrimitiveSpec]:
    outer_half_x = 0.5 * outer_size_m[0]
    outer_half_y = 0.5 * outer_size_m[1]
    outer_half_z = 0.5 * outer_size_m[2]
    outer_min_x = outer_center_local_m[0] - outer_half_x
    outer_max_x = outer_center_local_m[0] + outer_half_x
    outer_min_y = outer_center_local_m[1] - outer_half_y
    outer_max_y = outer_center_local_m[1] + outer_half_y
    cavity_min_x = cavity_center_local_xy_m[0] - cavity_halfspan_xy_m[0]
    cavity_max_x = cavity_center_local_xy_m[0] + cavity_halfspan_xy_m[0]
    cavity_min_y = cavity_center_local_xy_m[1] - cavity_halfspan_xy_m[1]
    cavity_max_y = cavity_center_local_xy_m[1] + cavity_halfspan_xy_m[1]

    left_thickness = max(cavity_min_x - outer_min_x, 0.004)
    right_thickness = max(outer_max_x - cavity_max_x, 0.004)
    front_thickness = max(cavity_min_y - outer_min_y, 0.004)
    back_thickness = max(outer_max_y - cavity_max_y, 0.004)

    wall_center_z = outer_center_local_m[2]
    side_span_y = max(outer_size_m[1], 0.010)
    inner_span_x = max(cavity_max_x - cavity_min_x, 0.010)
    walls_local = [
        (
            f"{object_prefix}_wall_left",
            (left_thickness, side_span_y, outer_size_m[2]),
            (outer_min_x + 0.5 * left_thickness, outer_center_local_m[1], wall_center_z),
        ),
        (
            f"{object_prefix}_wall_right",
            (right_thickness, side_span_y, outer_size_m[2]),
            (cavity_max_x + 0.5 * right_thickness, outer_center_local_m[1], wall_center_z),
        ),
        (
            f"{object_prefix}_wall_front",
            (inner_span_x, front_thickness, outer_size_m[2]),
            (cavity_center_local_xy_m[0], outer_min_y + 0.5 * front_thickness, wall_center_z),
        ),
        (
            f"{object_prefix}_wall_back",
            (inner_span_x, back_thickness, outer_size_m[2]),
            (cavity_center_local_xy_m[0], cavity_max_y + 0.5 * back_thickness, wall_center_z),
        ),
    ]

    return [
        _box_primitive(
            object_id,
            _pose_from_local_point(fixture_pose_w, center_local),
            dims,
            robot_pose_w,
        )
        for object_id, dims, center_local in walls_local
    ]


def _planning_cavity_halfspan(cavity_halfspan_xy_m: tuple[float, float]) -> tuple[float, float]:
    return (
        max(cavity_halfspan_xy_m[0], PLANNING_CAVITY_HALFSPAN_MIN_M[0]),
        max(cavity_halfspan_xy_m[1], PLANNING_CAVITY_HALFSPAN_MIN_M[1]),
    )


def scene_primitives(
    *,
    robot_pose_w: PoseWxyz,
    holder_pose_w: PoseWxyz,
    vortexer_pose_w: PoseWxyz,
) -> tuple[PrimitiveSpec, ...]:
    table_pose_w = PoseWxyz(
        x=TABLE_CENTER_WORLD_M[0],
        y=TABLE_CENTER_WORLD_M[1],
        z=TABLE_CENTER_WORLD_M[2],
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
    )
    primitives = [
        _box_primitive("table", table_pose_w, TABLE_SIZE_M, robot_pose_w),
    ]
    primitives.extend(
        _open_top_wall_primitives(
            object_prefix="holder",
            fixture_pose_w=holder_pose_w,
            robot_pose_w=robot_pose_w,
            outer_size_m=HOLDER_BOX_SIZE_M,
            outer_center_local_m=HOLDER_BOX_CENTER_LOCAL_M,
            cavity_center_local_xy_m=IMPORTED_LAB_ASSETS.holder_support_zone_center_local_xy_m,
            cavity_halfspan_xy_m=_planning_cavity_halfspan(IMPORTED_LAB_ASSETS.holder_support_zone_halfspan_xy_m),
        )
    )
    primitives.extend(
        _open_top_wall_primitives(
            object_prefix="vortexer",
            fixture_pose_w=vortexer_pose_w,
            robot_pose_w=robot_pose_w,
            outer_size_m=VORTEXER_BOX_SIZE_M,
            outer_center_local_m=VORTEXER_BOX_CENTER_LOCAL_M,
            cavity_center_local_xy_m=IMPORTED_LAB_ASSETS.vortexer_support_zone_center_local_xy_m,
            cavity_halfspan_xy_m=_planning_cavity_halfspan(IMPORTED_LAB_ASSETS.vortexer_support_zone_halfspan_xy_m),
        )
    )
    return tuple(primitives)


def tube_world_primitive(*, tube_root_w: PoseWxyz, robot_pose_w: PoseWxyz, object_id: str = "tube") -> PrimitiveSpec:
    center_w = tube_center_pose(tube_root_w)
    return PrimitiveSpec(
        object_id=object_id,
        primitive_type=SolidPrimitive.CYLINDER,
        dimensions=(TUBE_HEIGHT_M, TUBE_RADIUS_M),
        pose=TaskPose.from_pose_wxyz(_pose_in_robot_frame(center_w, robot_pose_w)),
    )


def build_leg_plan(
    *,
    label: str,
    dest_name: str,
    robot_pose_w: PoseWxyz,
    source_tube_root_w: PoseWxyz,
    dest_support_w: PoseWxyz,
    dest_fixture_w: PoseWxyz | None = None,
    grasp_tool_clearance_m: float = GRASP_TOOL_CLEARANCE_M,
    verify_lift_height_m: float = PHYSICAL_VERIFY_LIFT_HEIGHT_M,
    place_tool_clearance_m: float | None = None,
    pickup_grasp_bias_local_m: tuple[float, float, float] = PICKUP_GRASP_BIAS_LOCAL_M,
    place_bias_local_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LiveLegPlan:
    pickup_contact_nominal_tool_w = _offset_world_z(tube_top_grasp_pose(source_tube_root_w), grasp_tool_clearance_m)
    pickup_contact_tool_w = _offset_local_xyz(
        pickup_contact_nominal_tool_w,
        pickup_grasp_bias_local_m,
    )
    # Keep the hover pose on the nominal pickup axis so the approach remains
    # planner-friendly, then slide into the biased retention pocket only during
    # the final grasp stroke.
    grasp_tool_w = compensate_tool_pose_for_closed_gripper(pickup_contact_tool_w)
    pregrasp_tool_w = _offset_world_z(
        compensate_tool_pose_for_closed_gripper(pickup_contact_nominal_tool_w),
        PRE_SUPPORT_CLEARANCE_M,
    )
    # Keep the physical verification lift configurable so breakout from the
    # source support can be tuned independently from the later carry path.
    lift_tool_w = _offset_world_z(grasp_tool_w, verify_lift_height_m)

    is_vortexer_place = dest_name == "vortexer" and dest_fixture_w is not None
    if is_vortexer_place:
        release_reference_w = vortexer_release_pose(dest_fixture_w)
    else:
        release_reference_w = dest_support_w
    biased_dest_support_w = _offset_local_xyz(release_reference_w, place_bias_local_m)
    final_place_tool_clearance_m = (
        grasp_tool_clearance_m if place_tool_clearance_m is None else place_tool_clearance_m
    )
    place_tool_w = compensate_tool_pose_for_closed_gripper(
        _offset_world_z(biased_dest_support_w, TUBE_HEIGHT_M + final_place_tool_clearance_m)
    )
    preplace_clearance_m = VORTEXER_PREPLACE_CLEARANCE_M if is_vortexer_place else PRE_SUPPORT_CLEARANCE_M
    transfer_clearance_m = VORTEXER_TRANSFER_CLEARANCE_M if is_vortexer_place else TRANSFER_CLEARANCE_M
    preplace_tool_w = _offset_world_z(place_tool_w, preplace_clearance_m)
    retreat_tool_w = _offset_world_z(place_tool_w, THRESH.retreat_height_m)
    transit_height_w = max(lift_tool_w.z, preplace_tool_w.z, place_tool_w.z + transfer_clearance_m)
    transit_tool_w = PoseWxyz(
        x=place_tool_w.x,
        y=place_tool_w.y,
        z=transit_height_w,
        qw=place_tool_w.qw,
        qx=place_tool_w.qx,
        qy=place_tool_w.qy,
        qz=place_tool_w.qz,
    )
    return LiveLegPlan(
        label=label,
        dest_name=dest_name,
        pregrasp=TaskPose.from_pose_wxyz(_pose_in_robot_frame(ee_body_pose_from_tool_pose(pregrasp_tool_w), robot_pose_w)),
        grasp=TaskPose.from_pose_wxyz(_pose_in_robot_frame(ee_body_pose_from_tool_pose(grasp_tool_w), robot_pose_w)),
        lift=TaskPose.from_pose_wxyz(_pose_in_robot_frame(ee_body_pose_from_tool_pose(lift_tool_w), robot_pose_w)),
        transit=TaskPose.from_pose_wxyz(_pose_in_robot_frame(ee_body_pose_from_tool_pose(transit_tool_w), robot_pose_w)),
        preplace=TaskPose.from_pose_wxyz(_pose_in_robot_frame(ee_body_pose_from_tool_pose(preplace_tool_w), robot_pose_w)),
        place=TaskPose.from_pose_wxyz(_pose_in_robot_frame(ee_body_pose_from_tool_pose(place_tool_w), robot_pose_w)),
        retreat=TaskPose.from_pose_wxyz(_pose_in_robot_frame(ee_body_pose_from_tool_pose(retreat_tool_w), robot_pose_w)),
    )


def build_live_task(seed: int, mode_arg: str) -> LiveTask:
    rng = random.Random(seed)
    fixture_poses = _sample_fixture_layout(rng)
    robot_pose_w = fixture_poses["robot"]
    holder_pose_w = fixture_poses["holder"]
    vortexer_pose_w = fixture_poses["vortexer"]

    holder_slot_index = rng.randrange(len(_lookup_holder_slot_points()))
    task_mode = _task_mode_from_arg(mode_arg, rng)

    holder_support_w = holder_slot_support_pose(holder_pose_w, holder_slot_index)
    vortexer_support_w = vortexer_support_pose(vortexer_pose_w)
    initial_tube_root_w = _root_pose_from_support_point(
        holder_support_w.pos,
        holder_pose_w.quat_wxyz,
        TUBE_SUPPORT_LOCAL_M,
    )

    legs: list[LegSpec] = []
    if task_mode == 0:
        legs.append(
            LegSpec(
                label="leg_0",
                dest_name="holder",
                dest_support_w=holder_support_w,
                dest_fixture_w=holder_pose_w,
            )
        )
    else:
        legs.append(
            LegSpec(
                label="leg_0",
                dest_name="vortexer",
                dest_support_w=vortexer_support_w,
                dest_fixture_w=vortexer_pose_w,
            )
        )
        legs.append(
            LegSpec(
                label="leg_1",
                dest_name="holder",
                dest_support_w=holder_support_w,
                dest_fixture_w=holder_pose_w,
            )
        )

    return LiveTask(
        seed=seed,
        task_mode=task_mode,
        holder_slot_index=holder_slot_index,
        robot_pose_w=robot_pose_w,
        holder_pose_w=holder_pose_w,
        vortexer_pose_w=vortexer_pose_w,
        initial_tube_root_w=initial_tube_root_w,
        scene_objects=scene_primitives(
            robot_pose_w=robot_pose_w,
            holder_pose_w=holder_pose_w,
            vortexer_pose_w=vortexer_pose_w,
        ),
        legs=tuple(legs),
    )
