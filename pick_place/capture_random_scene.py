"""Render one randomized wet-lab scene for visual inspection.

This is only for scene-generation debugging:
- no solver
- no FSM execution
- capture the settled randomized initialization
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture one randomized wet-lab scene.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--settle_steps", type=int, default=300)
parser.add_argument("--tube_source", type=str, default="all", choices=["all", "holder", "vortexer", "table"])
parser.add_argument("--camera_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--camera_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--debug_materials", action="store_true", help="Apply flat-color debug materials instead of the USD-authored appearance.")
parser.add_argument("--hide_robot", action="store_true", help="Hide the robot prim before capture for occlusion debugging.")
parser.add_argument("--hide_holder", action="store_true", help="Hide the holder prim before capture for occlusion debugging.")
parser.add_argument("--hide_vortexer", action="store_true", help="Hide the vortexer prim before capture for occlusion debugging.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from wetlab_benchmark.pick_place.runtime import (
    PickPlaceSceneCfg,
    SCENE_DOME_LIGHT_COLOR,
    SCENE_DOME_LIGHT_INTENSITY,
    create_simulation,
    initialize_pick_place_runtime,
)
from wetlab_benchmark.task_config import PICK_PLACE_RANDOMIZATION, SCENE_CHECK, TUBE_SOURCE_NAMES, with_tube_source
from wetlab_benchmark.validation import validate_task_reachability, validate_task_scene


# Move the eye onto a principal table axis so the tabletop reads aligned in the
# frame instead of diagonally rotated, while keeping a slightly wider view.
DEFAULT_CAMERA_EYE = (0.28, 0.96, 1.35)
DEFAULT_CAMERA_TARGET = (0.28, 0.0, 0.82)
RENDER_WARMUP_STEPS = 20


@configclass
class SceneCfgWithCamera(PickPlaceSceneCfg):
    """Adds a fixed world-frame observation camera to the pick-place scene."""

    light = AssetBaseCfg(
        prim_path="/World/Light",
        # Keep the backdrop meaningfully darker than the white robot so renders
        # remain legible even with USD-authored materials.
        spawn=sim_utils.DomeLightCfg(
            intensity=SCENE_DOME_LIGHT_INTENSITY,
            color=SCENE_DOME_LIGHT_COLOR,
        ),
    )

    obs_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ObsCamera",
        update_period=0,
        update_latest_camera_pose=False,
        height=720,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=DEFAULT_CAMERA_EYE, rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )


def _hide_prim_if_present(prim_path: str) -> bool:
    """Hide a stage prim for debug renders if it exists."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        return False
    UsdGeom.Imageable(prim).MakeInvisible()
    return True


def _bind_debug_material(
    prim_path: str,
    material_name: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.65,
    metallic: float = 0.0,
) -> None:
    """Override visuals with a simple preview material so debug captures are readable."""
    material_path = f"/World/Looks/{material_name}"
    material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness, metallic=metallic)
    try:
        material_cfg.func(material_path, material_cfg)
    except ValueError:
        pass
    sim_utils.bind_visual_material(prim_path, material_path)


def _bind_component_material_recursive(
    prim_path: str,
    material_name: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.65,
    metallic: float = 0.0,
) -> None:
    """Bind a material to every visual descendant under a subtree."""
    import omni.usd
    from pxr import UsdGeom

    def _iter_descendants(root_prim):
        stack = [child for child in root_prim.GetChildren() if child.IsValid()]
        while stack:
            current = stack.pop()
            yield current
            children = [child for child in current.GetChildren() if child.IsValid()]
            stack.extend(reversed(children))

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        return

    gprim_paths: list[str] = []
    for descendant in _iter_descendants(prim):
        if descendant.IsA(UsdGeom.Gprim):
            gprim_paths.append(descendant.GetPath().pathString)

    if not gprim_paths and prim.IsA(UsdGeom.Gprim):
        gprim_paths.append(prim.GetPath().pathString)

    if not gprim_paths:
        _bind_debug_material(
            prim_path,
            material_name,
            color,
            roughness=roughness,
            metallic=metallic,
        )
        return

    for gprim_path in gprim_paths:
        _bind_debug_material(
            gprim_path,
            material_name,
            color,
            roughness=roughness,
            metallic=metallic,
        )


def _partition_descendants(prim_path: str, *, max_depth: int = 3) -> list[str]:
    """Return a useful set of child subtree roots for component-wise coloring."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        return []

    current = prim
    for _ in range(max_depth):
        children = [child for child in current.GetChildren() if child.IsValid()]
        if len(children) >= 2:
            return [child.GetPath().pathString for child in children]
        if len(children) != 1:
            return []
        current = children[0]
    return [child.GetPath().pathString for child in current.GetChildren() if child.IsValid()]


def _component_name(prim_path: str) -> str:
    return prim_path.rsplit("/", 1)[-1].lower()


def _bind_component_palette(
    root_prim_path: str,
    material_prefix: str,
    fallback_palette: tuple[dict[str, object], ...],
    named_specs: tuple[dict[str, object], ...] = (),
) -> None:
    """Apply distinct materials to child component subtrees below a root prim."""
    component_paths = _partition_descendants(root_prim_path)
    if not component_paths:
        spec = fallback_palette[0]
        _bind_component_material_recursive(
            root_prim_path,
            f"{material_prefix}_root",
            spec["color"],
            roughness=spec.get("roughness", 0.65),
            metallic=spec.get("metallic", 0.0),
        )
        return

    for index, component_path in enumerate(component_paths):
        name = _component_name(component_path)
        spec = None
        for candidate in named_specs:
            if any(fragment in name for fragment in candidate["matches"]):
                spec = candidate
                break
        if spec is None:
            spec = fallback_palette[index % len(fallback_palette)]
        _bind_component_material_recursive(
            component_path,
            f"{material_prefix}_{index}",
            spec["color"],
            roughness=spec.get("roughness", 0.65),
            metallic=spec.get("metallic", 0.0),
        )


def _apply_debug_materials() -> None:
    robot_palette = (
        {"color": (0.78, 0.80, 0.82), "roughness": 0.40, "metallic": 0.35},
        {"color": (0.60, 0.67, 0.75), "roughness": 0.32, "metallic": 0.45},
        {"color": (0.54, 0.61, 0.68), "roughness": 0.38, "metallic": 0.42},
        {"color": (0.72, 0.74, 0.77), "roughness": 0.34, "metallic": 0.30},
        {"color": (0.64, 0.70, 0.73), "roughness": 0.36, "metallic": 0.32},
        {"color": (0.70, 0.72, 0.76), "roughness": 0.28, "metallic": 0.55},
    )
    robot_named_specs = (
        {"matches": ("finger",), "color": (0.14, 0.15, 0.18), "roughness": 0.82, "metallic": 0.04},
        {"matches": ("knuckle", "gripper"), "color": (0.56, 0.60, 0.65), "roughness": 0.46, "metallic": 0.28},
        {"matches": ("base",), "color": (0.48, 0.52, 0.57), "roughness": 0.34, "metallic": 0.55},
    )
    vortexer_palette = (
        {"color": (0.24, 0.28, 0.33), "roughness": 0.30, "metallic": 0.72},
        {"color": (0.80, 0.24, 0.18), "roughness": 0.52, "metallic": 0.10},
        {"color": (0.88, 0.79, 0.18), "roughness": 0.62, "metallic": 0.05},
        {"color": (0.16, 0.50, 0.62), "roughness": 0.34, "metallic": 0.55},
        {"color": (0.70, 0.72, 0.76), "roughness": 0.18, "metallic": 0.92},
        {"color": (0.12, 0.12, 0.13), "roughness": 0.86, "metallic": 0.02},
    )
    vortexer_named_specs = (
        {"matches": ("base", "body", "housing"), "color": (0.22, 0.26, 0.31), "roughness": 0.32, "metallic": 0.75},
        {"matches": ("button", "switch", "knob"), "color": (0.82, 0.33, 0.12), "roughness": 0.48, "metallic": 0.12},
        {"matches": ("pad", "support", "plate", "top"), "color": (0.74, 0.76, 0.78), "roughness": 0.26, "metallic": 0.88},
    )

    _bind_debug_material("/World/envs/env_0/Table", "DebugTable", (0.20, 0.25, 0.32), roughness=0.94, metallic=0.0)
    _bind_debug_material("/World/envs/env_0/TubeHolder", "DebugHolder", (0.20, 0.62, 0.34), roughness=0.74, metallic=0.04)
    _bind_component_material_recursive(
        "/World/envs/env_0/Robot",
        "DebugRobotBaseWash",
        (0.76, 0.78, 0.80),
        roughness=0.42,
        metallic=0.30,
    )
    _bind_component_palette("/World/envs/env_0/Robot", "DebugRobot", robot_palette, robot_named_specs)
    _bind_component_palette("/World/envs/env_0/Vortexer", "DebugVortexer", vortexer_palette, vortexer_named_specs)
    _bind_debug_material("/World/envs/env_0/Tube", "DebugTube", (0.82, 0.12, 0.10), roughness=0.38, metallic=0.06)


def _reseat_holder_supported_tube_for_capture(runtime, camera) -> bool:
    """Restore the sampled holder-supported pose before the final capture.

    The capture script is for visual review, so prefer a deterministic seated
    pose over tiny dynamic drift accumulated during long warmup / settle loops.
    """
    holder_source_index = TUBE_SOURCE_NAMES.index("holder")
    placement_index = runtime.layout["metadata"]["tube_placement_index"]
    if placement_index.numel() == 0 or int(placement_index[0].item()) != holder_source_index:
        return False

    sample = runtime.layout["poses"]["tube"]
    world_pose = torch.cat([sample.pos + runtime.scene.env_origins, sample.quat], dim=-1)
    runtime.tube.write_root_pose_to_sim(world_pose)
    runtime.tube.write_root_velocity_to_sim(torch.zeros(runtime.scene.num_envs, 6, device=runtime.device))

    dt = runtime.sim.get_physics_dt()
    for _ in range(2):
        runtime.scene.update(dt)
        runtime.sim.render()
        camera.update(dt)
    return True


def main():
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = with_tube_source(PICK_PLACE_RANDOMIZATION, args_cli.tube_source)
    camera_eye = tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else DEFAULT_CAMERA_EYE
    camera_target = tuple(args_cli.camera_target) if args_cli.camera_target is not None else DEFAULT_CAMERA_TARGET

    sim = create_simulation(
        device=args_cli.device,
        camera_eye=list(camera_eye),
        camera_target=list(camera_target),
    )
    scene = InteractiveScene(SceneCfgWithCamera(num_envs=1, env_spacing=2.5))

    sim.reset()
    scene.reset()

    runtime = initialize_pick_place_runtime(sim=sim, scene=scene, seed=args_cli.seed, task_cfg=task_cfg)
    camera = scene["obs_camera"]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([camera_eye], device=runtime.device, dtype=torch.float32),
        targets=torch.tensor([camera_target], device=runtime.device, dtype=torch.float32),
    )

    hidden_robot = False
    hidden_holder = False
    hidden_vortexer = False
    if args_cli.hide_robot:
        hidden_robot = _hide_prim_if_present("/World/envs/env_0/Robot")
        print(f"[capture_random_scene] hide_robot={hidden_robot}")
    if args_cli.hide_holder:
        hidden_holder = _hide_prim_if_present("/World/envs/env_0/TubeHolder")
        print(f"[capture_random_scene] hide_holder={hidden_holder}")
    if args_cli.hide_vortexer:
        hidden_vortexer = _hide_prim_if_present("/World/envs/env_0/Vortexer")
        print(f"[capture_random_scene] hide_vortexer={hidden_vortexer}")
    if args_cli.debug_materials:
        _apply_debug_materials()

    # Physics settle without rendering (fast path — no RTX calls).
    for _ in range(args_cli.settle_steps):
        runtime.scene.write_data_to_sim()
        runtime.sim.step()
        runtime.scene.update(runtime.sim.get_physics_dt())

    # Prime the RTX pipeline: step physics to flush positions to USD, then render.
    for _ in range(RENDER_WARMUP_STEPS):
        runtime.scene.write_data_to_sim()
        runtime.sim.step()
        runtime.sim.render()
        runtime.scene.update(runtime.sim.get_physics_dt())
        camera.update(runtime.sim.get_physics_dt())

    holder_tube_reseated = _reseat_holder_supported_tube_for_capture(runtime, camera)

    scene_failures = validate_task_scene(
        scene=runtime.scene,
        assets=runtime.assets,
        layout=runtime.layout,
        task_cfg=runtime.task_cfg,
        check_cfg=SCENE_CHECK,
        sensors=runtime.sensors,
    )
    reachability_failures = validate_task_reachability(runtime=runtime)
    failures = scene_failures + reachability_failures

    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    image_path = output_dir / f"seed_{args_cli.seed:03d}.png"
    Image.fromarray(rgb).save(str(image_path))

    metadata = {
        "seed": args_cli.seed,
        "tube_source": args_cli.tube_source,
        "hide_robot": hidden_robot,
        "hide_holder": hidden_holder,
        "hide_vortexer": hidden_vortexer,
        "debug_materials": args_cli.debug_materials,
        "holder_tube_reseated_for_capture": holder_tube_reseated,
        "tube_placement_index": runtime.layout["metadata"]["tube_placement_index"].cpu().tolist(),
        "tube_holder_point_index": runtime.layout["metadata"].get("tube_holder_point_index", torch.tensor([-1])).cpu().tolist(),
        "failures": failures,
        "scene_failures": scene_failures,
        "reachability_failures": reachability_failures,
        "camera_eye": list(camera_eye),
        "camera_target": list(camera_target),
        "rgb_min": int(rgb.min()),
        "rgb_max": int(rgb.max()),
        "robot_root_pose_w": runtime.robot.data.root_pose_w.cpu().tolist(),
        "tube_root_pose_w": runtime.tube.data.root_pose_w.cpu().tolist(),
        "holder_root_pose_w": runtime.holder.data.root_pose_w.cpu().tolist(),
        "vortexer_root_pose_w": runtime.vortexer.data.root_pose_w.cpu().tolist(),
    }
    metadata_path = output_dir / f"seed_{args_cli.seed:03d}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"[capture_random_scene] image={image_path}")
    print(f"[capture_random_scene] metadata={metadata_path}")
    print(f"[capture_random_scene] failures={len(failures)}")
    for failure in failures:
        print(f"[capture_random_scene] failure: {failure}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        # Isaac Sim 4.5 can hang in Kit shutdown after a successful headless run.
        # For one-shot render captures, flush logs and terminate directly so batch
        # jobs can advance to the next seed without blocking on shutdown.
        if getattr(args_cli, "headless", False):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
