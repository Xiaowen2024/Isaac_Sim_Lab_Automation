#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Legacy Isaac-only proxy-pad physical tube grasp probe. "
        "The baked-contact live-exec path no longer uses runtime proxy pad bodies."
    )
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--physics_dt", type=float, default=0.0025)
parser.add_argument("--support_xy", type=float, nargs=2, default=(0.42, 0.08))
parser.add_argument("--inner_gap_m", type=float, default=0.039)
parser.add_argument("--open_inner_gap_m", type=float, default=0.075)
parser.add_argument("--pad_y_size_m", type=float, default=0.018)
parser.add_argument("--pad_center_below_tube_top_m", type=float, default=0.007)
parser.add_argument("--contact_force_threshold_n", type=float, default=0.03)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--close_steps", type=int, default=420)
parser.add_argument("--hold_steps", type=int, default=120)
parser.add_argument("--lift_steps", type=int, default=520)
parser.add_argument("--lift_delta_m", type=float, default=0.03)
parser.add_argument("--min_lift_delta_m", type=float, default=0.020)
parser.add_argument("--carry_dx_m", type=float, default=0.0)
parser.add_argument("--carry_dy_m", type=float, default=0.0)
parser.add_argument("--carry_steps", type=int, default=0)
parser.add_argument("--min_carry_xy_delta_m", type=float, default=0.04)
parser.add_argument("--pad_quat_mode", choices=["top_grasp", "identity"], default="top_grasp")
parser.add_argument(
    "--close_app",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Call simulation_app.close() before exit. Disabled by default because Isaac 4.5 can hang on close.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from wetlab_benchmark.pick_place_claude.runtime import (
    ASSET_PROFILE_CONTACT_REFINED,
    CONTACT_PHYSICS_DT,
    create_pick_place_runtime,
)
from wetlab_benchmark.task_config import FRAMES, IMPORTED_LAB_ASSETS, PICK_PLACE_RANDOMIZATION, SURFACE
from wetlab_benchmark.validation import _contact_any_for_filters


LEFT_PROXY_FILTERS = (2,)
RIGHT_PROXY_FILTERS = (3,)
TABLE_THICKNESS_M = 0.04
TOP_GRASP_QUAT_WXYZ = FRAMES.ee_top_grasp_quat_wxyz
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


def _pose_world_point(pose_w: PoseWxyz, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    offset = _quat_apply(pose_w.quat_wxyz, local_xyz)
    return (pose_w.x + offset[0], pose_w.y + offset[1], pose_w.z + offset[2])


def _pose_tensor(pos, quat, *, device: str) -> torch.Tensor:
    return torch.tensor([[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]], device=device, dtype=torch.float32)


def _write_pose(asset, pose: PoseWxyz, *, device: str, zero_velocity: bool = False) -> None:
    asset.write_root_pose_to_sim(_pose_tensor(pose.pos, pose.quat_wxyz, device=device))
    if zero_velocity:
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


def _root_pose_from_local_point(
    point_w: tuple[float, float, float],
    root_quat_wxyz: tuple[float, float, float, float],
    local_point_m: tuple[float, float, float],
) -> PoseWxyz:
    offset_w = _quat_apply(root_quat_wxyz, local_point_m)
    return PoseWxyz(
        x=point_w[0] - offset_w[0],
        y=point_w[1] - offset_w[1],
        z=point_w[2] - offset_w[2],
        qw=root_quat_wxyz[0],
        qx=root_quat_wxyz[1],
        qy=root_quat_wxyz[2],
        qz=root_quat_wxyz[3],
    )


def _step(runtime) -> None:
    runtime.scene.write_data_to_sim()
    runtime.sim.step()
    runtime.scene.update(runtime.sim.get_physics_dt())


def _contacts(runtime, threshold: float) -> tuple[bool, bool, float]:
    forces = torch.nan_to_num(runtime.tube_contacts.data.force_matrix_w)
    max_force = float(torch.max(torch.linalg.norm(forces, dim=-1)).item()) if forces.numel() else 0.0
    left = bool(_contact_any_for_filters(runtime.tube_contacts, LEFT_PROXY_FILTERS, threshold)[0].item())
    right = bool(_contact_any_for_filters(runtime.tube_contacts, RIGHT_PROXY_FILTERS, threshold)[0].item())
    return left, right, max_force


def _write_proxy_pads(
    runtime,
    *,
    center_xy: tuple[float, float],
    center_z: float,
    inner_gap_m: float,
    pad_y_size_m: float,
    pad_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center_distance = inner_gap_m + pad_y_size_m
    left = (center_xy[0], center_xy[1] - 0.5 * center_distance, center_z)
    right = (center_xy[0], center_xy[1] + 0.5 * center_distance, center_z)
    runtime.proxy_left_gripper_pad.write_root_pose_to_sim(_pose_tensor(left, pad_quat, device=runtime.device))
    runtime.proxy_right_gripper_pad.write_root_pose_to_sim(_pose_tensor(right, pad_quat, device=runtime.device))
    return left, right


def _progress(phase: str, extra: dict | None = None) -> None:
    payload = {"phase": phase, "elapsed_s": round(time.monotonic() - _progress.started_at, 3)}
    if extra:
        payload.update(extra)
    print(f"[proxy_grasp_probe] {json.dumps(payload, sort_keys=True)}", flush=True)


_progress.started_at = time.monotonic()


def main() -> int:
    output_path = Path(args_cli.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _progress("runtime:start")
    runtime = create_pick_place_runtime(
        num_envs=1,
        seed=args_cli.seed,
        device=args_cli.device,
        camera_eye=[1.0, 0.4, 0.9],
        camera_target=[0.35, 0.05, 0.55],
        dt=args_cli.physics_dt or CONTACT_PHYSICS_DT,
        contact_physics=True,
        task_cfg=PICK_PLACE_RANDOMIZATION,
        asset_profile=ASSET_PROFILE_CONTACT_REFINED,
        progress_callback=lambda phase, extra=None: _progress(phase, extra),
    )
    proxy_left_pad = getattr(runtime, "proxy_left_gripper_pad", None)
    proxy_right_pad = getattr(runtime, "proxy_right_gripper_pad", None)
    if proxy_left_pad is None or proxy_right_pad is None:
        raise RuntimeError(
            "contact_refined runtime no longer creates proxy pad rigid objects; "
            "this legacy probe must be replaced with a baked-contact gripper probe"
        )

    _progress("place_assets:start")
    table_pose = PoseWxyz(
        x=args_cli.support_xy[0],
        y=args_cli.support_xy[1],
        z=SURFACE.z - 0.5 * TABLE_THICKNESS_M,
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
    )
    _write_pose(runtime.scene["table"], table_pose, device=runtime.device)
    _write_pose(
        runtime.holder,
        PoseWxyz(x=1.0, y=0.45, z=SURFACE.z, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        device=runtime.device,
    )
    _write_pose(
        runtime.vortexer,
        PoseWxyz(x=1.0, y=-0.45, z=SURFACE.z, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        device=runtime.device,
    )
    tube_quat = IMPORTED_LAB_ASSETS.upright_quat_wxyz
    tube_support_w = (args_cli.support_xy[0], args_cli.support_xy[1], SURFACE.z)
    tube_pose = _root_pose_from_local_point(tube_support_w, tube_quat, TUBE_SUPPORT_LOCAL_M)
    _write_pose(runtime.tube, tube_pose, device=runtime.device, zero_velocity=True)
    tube_top_w = _pose_world_point(tube_pose, TUBE_TOP_LOCAL_M)
    tube_center_w = _pose_world_point(tube_pose, TUBE_CENTER_LOCAL_M)
    pad_center_z = tube_top_w[2] - args_cli.pad_center_below_tube_top_m
    pad_quat = TOP_GRASP_QUAT_WXYZ if args_cli.pad_quat_mode == "top_grasp" else (1.0, 0.0, 0.0, 0.0)
    prev_left, prev_right = _write_proxy_pads(
        runtime,
        center_xy=(tube_center_w[0], tube_center_w[1]),
        center_z=pad_center_z,
        inner_gap_m=args_cli.open_inner_gap_m,
        pad_y_size_m=args_cli.pad_y_size_m,
        pad_quat=pad_quat,
    )
    for _ in range(max(args_cli.settle_steps, 1)):
        _step(runtime)
    _progress("place_assets:done", {"tube_top_w": tube_top_w, "pad_center_z": pad_center_z})

    contact_seen = {"left": False, "right": False}
    max_force = 0.0
    _progress("close:start", {"steps": args_cli.close_steps})
    for step in range(max(args_cli.close_steps, 1)):
        alpha = min((step + 1) / max(args_cli.close_steps, 1), 1.0)
        inner_gap = args_cli.open_inner_gap_m * (1.0 - alpha) + args_cli.inner_gap_m * alpha
        prev_left, prev_right = _write_proxy_pads(
            runtime,
            center_xy=(tube_center_w[0], tube_center_w[1]),
            center_z=pad_center_z,
            inner_gap_m=inner_gap,
            pad_y_size_m=args_cli.pad_y_size_m,
            pad_quat=pad_quat,
        )
        _step(runtime)
        left, right, force = _contacts(runtime, args_cli.contact_force_threshold_n)
        contact_seen["left"] = contact_seen["left"] or left
        contact_seen["right"] = contact_seen["right"] or right
        max_force = max(max_force, force)
        if step == 0 or step + 1 == args_cli.close_steps or (step + 1) % max(args_cli.close_steps // 5, 1) == 0:
            _progress(
                "close:step",
                {
                    "step": step + 1,
                    "inner_gap_m": inner_gap,
                    "left_contact": left,
                    "right_contact": right,
                    "max_force_n": max_force,
                },
            )

    _progress("hold:start", {"steps": args_cli.hold_steps})
    for _ in range(max(args_cli.hold_steps, 1)):
        prev_left, prev_right = _write_proxy_pads(
            runtime,
            center_xy=(tube_center_w[0], tube_center_w[1]),
            center_z=pad_center_z,
            inner_gap_m=args_cli.inner_gap_m,
            pad_y_size_m=args_cli.pad_y_size_m,
            pad_quat=pad_quat,
        )
        _step(runtime)
        left, right, force = _contacts(runtime, args_cli.contact_force_threshold_n)
        contact_seen["left"] = contact_seen["left"] or left
        contact_seen["right"] = contact_seen["right"] or right
        max_force = max(max_force, force)

    tube_z_before = float(runtime.tube.data.root_pose_w[0, 2].item())
    tube_xy_before_lift = tuple(float(value.item()) for value in runtime.tube.data.root_pose_w[0, 0:2])
    _progress("lift:start", {"steps": args_cli.lift_steps, "tube_z_before": tube_z_before})
    for step in range(max(args_cli.lift_steps, 1)):
        alpha = min((step + 1) / max(args_cli.lift_steps, 1), 1.0)
        lift_z = pad_center_z + args_cli.lift_delta_m * alpha
        prev_left, prev_right = _write_proxy_pads(
            runtime,
            center_xy=(tube_center_w[0], tube_center_w[1]),
            center_z=lift_z,
            inner_gap_m=args_cli.inner_gap_m,
            pad_y_size_m=args_cli.pad_y_size_m,
            pad_quat=pad_quat,
        )
        _step(runtime)
        left, right, force = _contacts(runtime, args_cli.contact_force_threshold_n)
        max_force = max(max_force, force)
        if step == 0 or step + 1 == args_cli.lift_steps or (step + 1) % max(args_cli.lift_steps // 5, 1) == 0:
            _progress(
                "lift:step",
                {
                    "step": step + 1,
                    "left_contact": left,
                    "right_contact": right,
                    "tube_z": float(runtime.tube.data.root_pose_w[0, 2].item()),
                },
            )

    carry_requested = abs(args_cli.carry_dx_m) + abs(args_cli.carry_dy_m) > 0.0 and args_cli.carry_steps > 0
    if carry_requested:
        _progress(
            "carry:start",
            {"steps": args_cli.carry_steps, "carry_dx_m": args_cli.carry_dx_m, "carry_dy_m": args_cli.carry_dy_m},
        )
        carry_start_xy = (tube_center_w[0], tube_center_w[1])
        carry_end_xy = (tube_center_w[0] + args_cli.carry_dx_m, tube_center_w[1] + args_cli.carry_dy_m)
        lift_z = pad_center_z + args_cli.lift_delta_m
        for step in range(max(args_cli.carry_steps, 1)):
            alpha = min((step + 1) / max(args_cli.carry_steps, 1), 1.0)
            center_xy = (
                carry_start_xy[0] * (1.0 - alpha) + carry_end_xy[0] * alpha,
                carry_start_xy[1] * (1.0 - alpha) + carry_end_xy[1] * alpha,
            )
            prev_left, prev_right = _write_proxy_pads(
                runtime,
                center_xy=center_xy,
                center_z=lift_z,
                inner_gap_m=args_cli.inner_gap_m,
                pad_y_size_m=args_cli.pad_y_size_m,
                pad_quat=pad_quat,
            )
            _step(runtime)
            if step == 0 or step + 1 == args_cli.carry_steps or (step + 1) % max(args_cli.carry_steps // 5, 1) == 0:
                tube_xy_now = tuple(float(value.item()) for value in runtime.tube.data.root_pose_w[0, 0:2])
                _progress(
                    "carry:step",
                    {
                        "step": step + 1,
                        "tube_xy": tube_xy_now,
                        "tube_xy_delta_m": math.hypot(
                            tube_xy_now[0] - tube_xy_before_lift[0],
                            tube_xy_now[1] - tube_xy_before_lift[1],
                        ),
                    },
                )

    left_final, right_final, force_final = _contacts(runtime, args_cli.contact_force_threshold_n)
    max_force = max(max_force, force_final)
    tube_z_after = float(runtime.tube.data.root_pose_w[0, 2].item())
    lift_delta = tube_z_after - tube_z_before
    tube_xy_after = tuple(float(value.item()) for value in runtime.tube.data.root_pose_w[0, 0:2])
    carry_xy_delta = math.hypot(tube_xy_after[0] - tube_xy_before_lift[0], tube_xy_after[1] - tube_xy_before_lift[1])
    tube_pose_final = runtime.tube.data.root_pose_w[0].detach().cpu().tolist()
    tube_quat_final = tube_pose_final[3:7]
    tube_up = _quat_apply(tuple(tube_quat_final), (0.0, 0.0, 1.0))
    proxy_geometry_closed = args_cli.inner_gap_m <= 0.044
    ok = bool(
        proxy_geometry_closed
        and lift_delta >= args_cli.min_lift_delta_m
        and (not carry_requested or carry_xy_delta >= args_cli.min_carry_xy_delta_m)
        and tube_up[2] >= math.cos(math.radians(18.0))
    )
    reason = "ok" if ok else (
        f"left_seen={contact_seen['left']} right_seen={contact_seen['right']} "
        f"left_final={left_final} right_final={right_final} lift_delta_m={lift_delta:.4f} up_z={tube_up[2]:.3f}"
    )
    payload = {
        "ok": ok,
        "reason": reason,
        "inner_gap_m": args_cli.inner_gap_m,
        "open_inner_gap_m": args_cli.open_inner_gap_m,
        "pad_y_size_m": args_cli.pad_y_size_m,
        "pad_center_below_tube_top_m": args_cli.pad_center_below_tube_top_m,
        "pad_quat_mode": args_cli.pad_quat_mode,
        "left_contact_seen": contact_seen["left"],
        "right_contact_seen": contact_seen["right"],
        "left_contact_final": left_final,
        "right_contact_final": right_final,
        "proxy_geometry_closed": proxy_geometry_closed,
        "contact_sensor_note": "Isaac contact force matrix can remain zero for proxy pad child colliders; lift_delta is the physical gate.",
        "max_contact_force_n": max_force,
        "tube_root_z_before": tube_z_before,
        "tube_root_z_after": tube_z_after,
        "tube_root_lift_delta_m": lift_delta,
        "carry_requested": carry_requested,
        "carry_dx_m": args_cli.carry_dx_m,
        "carry_dy_m": args_cli.carry_dy_m,
        "tube_root_xy_delta_m": carry_xy_delta,
        "tube_up_axis_z": tube_up[2],
        "tube_root_pose_final": tube_pose_final,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if args_cli.close_app:
            simulation_app.close()
