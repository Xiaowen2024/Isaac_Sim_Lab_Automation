"""Save a simple top-down debug image of one randomized wet-lab scene."""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plot one randomized wet-lab scene layout.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--settle_steps", type=int, default=100)
parser.add_argument("--tube_source", type=str, default="all", choices=["all", "holder", "vortexer", "table"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from PIL import Image, ImageDraw

from wetlab_benchmark.pick_place_claude_v4.runtime import create_pick_place_runtime
from wetlab_benchmark.task_config import IMPORTED_LAB_ASSETS, PICK_PLACE_RANDOMIZATION, SCENE_CHECK, with_tube_source
from wetlab_benchmark.validation import validate_task_scene


WORLD_X = (-0.3, 0.9)
WORLD_Y = (-0.5, 0.5)
CANVAS_W = 1200
CANVAS_H = 900


def yaw_from_quat_wxyz(quat: list[float]) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_px(x: float, y: float) -> tuple[float, float]:
    u = (x - WORLD_X[0]) / (WORLD_X[1] - WORLD_X[0]) * CANVAS_W
    v = CANVAS_H - (y - WORLD_Y[0]) / (WORLD_Y[1] - WORLD_Y[0]) * CANVAS_H
    return u, v


def draw_oriented_box(
    draw: ImageDraw.ImageDraw,
    root_xy,
    size_xy,
    yaw: float,
    outline,
    width=4,
    local_center_offset_xy=(0.0, 0.0),
):
    """Draw an axis-aligned box whose geometry may be asymmetric about its USD root.

    root_xy:               world XY of the asset root.
    size_xy:               full extent (width, height) of the footprint in local frame.
    local_center_offset_xy: offset from USD root to the geometric centre of the box,
                            expressed in the asset's LOCAL frame (pre-rotation).
                            Use this when the USD origin is not at the mesh centre.
    """
    rx, ry = root_xy
    sx, sy = size_xy[0] * 0.5, size_xy[1] * 0.5
    ox, oy = local_center_offset_xy
    # Corners in local frame (relative to geometric centre, which is offset from root).
    corners = [(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)]
    pts = []
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    for lx, ly in corners:
        # Shift by centre offset then rotate into world frame.
        local_x = lx + ox
        local_y = ly + oy
        x = rx + cos_yaw * local_x - sin_yaw * local_y
        y = ry + sin_yaw * local_x + cos_yaw * local_y
        pts.append(world_to_px(x, y))
    draw.line(pts + [pts[0]], fill=outline, width=width)


def draw_circle(draw: ImageDraw.ImageDraw, center_xy, radius_m: float, outline, width=4):
    cx, cy = world_to_px(*center_xy)
    rx = radius_m / (WORLD_X[1] - WORLD_X[0]) * CANVAS_W
    ry = radius_m / (WORLD_Y[1] - WORLD_Y[0]) * CANVAS_H
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=outline, width=width)


def add_label(draw: ImageDraw.ImageDraw, pos_xy, text: str, fill):
    x, y = world_to_px(*pos_xy)
    draw.text((x + 6, y + 6), text, fill=fill)


def main():
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = with_tube_source(PICK_PLACE_RANDOMIZATION, args_cli.tube_source)

    runtime = create_pick_place_runtime(
        num_envs=1,
        seed=args_cli.seed,
        device=args_cli.device,
        camera_eye=[2.4, 2.0, 2.0],
        camera_target=[0.35, 0.0, 0.85],
        task_cfg=task_cfg,
    )

    for _ in range(args_cli.settle_steps):
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

    robot_pose = runtime.robot.data.root_pose_w[0].cpu().tolist()
    holder_pose = runtime.holder.data.root_pose_w[0].cpu().tolist()
    vortexer_pose = runtime.vortexer.data.root_pose_w[0].cpu().tolist()
    tube_pose = runtime.tube.data.root_pose_w[0].cpu().tolist()

    image = Image.new("RGB", (CANVAS_W, CANVAS_H), color=(248, 248, 244))
    draw = ImageDraw.Draw(image)

    # Table bounds
    draw.rectangle((0, 0, CANVAS_W - 1, CANVAS_H - 1), outline=(120, 120, 120), width=3)

    # Robot as a circle at base.
    draw_circle(draw, (robot_pose[0], robot_pose[1]), 0.08, outline=(30, 90, 180), width=5)
    add_label(draw, (robot_pose[0], robot_pose[1]), "robot", fill=(30, 90, 180))

    # Holder and vortexer footprints from inspected metric dimensions.
    # Holder STL bounds from root (world metres, scale=0.1, metersPerUnit=0.01):
    #   X: -0.0025 to +0.1455  →  centre at +0.0715, half-width = 0.074
    #   Y: -0.0479 to +0.0155  →  centre at -0.0162, half-height = 0.0317
    # Pass local_center_offset_xy so the rectangle reflects the actual asymmetric mesh.
    holder_yaw = yaw_from_quat_wxyz(holder_pose[3:7])
    draw_oriented_box(
        draw,
        (holder_pose[0], holder_pose[1]),
        (0.148, 0.06339),
        holder_yaw,
        outline=(20, 120, 60),
        width=5,
        local_center_offset_xy=(0.0715, -0.0162),
    )
    # Label at the geometric centre of the holder (rotated offset from root).
    holder_label_x = holder_pose[0] + math.cos(holder_yaw) * 0.0715 - math.sin(holder_yaw) * -0.0162
    holder_label_y = holder_pose[1] + math.sin(holder_yaw) * 0.0715 + math.cos(holder_yaw) * -0.0162
    add_label(draw, (holder_label_x, holder_label_y), "holder", fill=(20, 120, 60))

    draw_oriented_box(
        draw,
        (vortexer_pose[0], vortexer_pose[1]),
        (0.120, 0.17540),
        yaw_from_quat_wxyz(vortexer_pose[3:7]),
        outline=(160, 70, 20),
        width=5,
    )
    add_label(draw, (vortexer_pose[0], vortexer_pose[1]), "vortexer", fill=(160, 70, 20))

    draw_circle(
        draw,
        (tube_pose[0], tube_pose[1]),
        IMPORTED_LAB_ASSETS.scale[0] * 17.75 / 100.0,
        outline=(180, 20, 20),
        width=5,
    )
    add_label(draw, (tube_pose[0], tube_pose[1]), "tube", fill=(180, 20, 20))

    image_path = output_dir / f"layout_seed_{args_cli.seed:03d}.png"
    image.save(image_path)

    metadata = {
        "seed": args_cli.seed,
        "tube_source": args_cli.tube_source,
        "tube_placement_index": runtime.layout["metadata"]["tube_placement_index"].cpu().tolist(),
        "tube_holder_point_index": runtime.layout["metadata"].get("tube_holder_point_index", torch.tensor([-1])).cpu().tolist(),
        "failures": failures,
        "robot_root_pose_w": robot_pose,
        "holder_root_pose_w": holder_pose,
        "vortexer_root_pose_w": vortexer_pose,
        "tube_root_pose_w": tube_pose,
    }
    metadata_path = output_dir / f"layout_seed_{args_cli.seed:03d}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"[plot_random_scene_layout] image={image_path}")
    print(f"[plot_random_scene_layout] metadata={metadata_path}")
    print(f"[plot_random_scene_layout] failures={len(failures)}")
    for failure in failures:
        print(f"[plot_random_scene_layout] failure: {failure}")


if __name__ == "__main__":
    main()
    simulation_app.close()
