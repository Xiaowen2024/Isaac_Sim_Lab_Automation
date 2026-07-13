#!/usr/bin/env python3
"""Run the canonical contact-valid placement scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the contact-valid placement scene.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_spacing", type=float, default=2.5)
    parser.add_argument("--sim_steps", type=int, default=600)
    parser.add_argument("--physics_dt", type=float, default=1.0 / 240.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


ROBOT_USD = ASSET_DIR / "xarm6_with_gripper_contact_refined.usd"
HOLDER_USD = ASSET_DIR / "4_50ml_conical_holder_contact_refined.usd"
VORTEXER_USD = ASSET_DIR / "vortexer_contact_refined.usd"
TUBE_USD = ASSET_DIR / "autobio_50ml_tube_contact_refined.usda"

TABLE_SIZE_M = (1.20, 0.90, 0.04)
TABLE_CENTER_M = (0.35, 0.0, 0.76)
TABLE_TOP_Z_M = TABLE_CENTER_M[2] + 0.5 * TABLE_SIZE_M[2]

ROBOT_ROOT_POS_M = (0.0, 0.0, TABLE_TOP_Z_M + 0.0501)
HOLDER_ROOT_POS_M = (0.43, -0.12, TABLE_TOP_Z_M + 0.004)
VORTEXER_ROOT_POS_M = (0.43, 0.14, TABLE_TOP_Z_M + 0.010)
TUBE_INITIAL_POS_M = (0.43, -0.12, TABLE_TOP_Z_M + 0.14)

ROBOT_SCALE = (1.0, 1.0, 1.0)
HOLDER_SCALE = (0.00114, 0.00114, 0.001)
VORTEXER_SCALE = (0.001, 0.001, 0.001)
TUBE_SCALE = (1.0, 1.0, 1.0)


def require_assets() -> None:
    missing = [path for path in (ROBOT_USD, HOLDER_USD, VORTEXER_USD, TUBE_USD) if not path.exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required scene assets:\n{missing_list}")


@configclass
class ContactValidPlacementSceneCfg(InteractiveSceneCfg):
    """Scene assets for the contact-valid placement task."""

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.55,
                dynamic_friction=0.45,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER_M),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD), scale=ROBOT_SCALE, activate_contact_sensors=True),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=400.0,
                damping=40.0,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_ROOT_POS_M,
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "joint5": 1.5708,
                "joint6": 0.0,
                "drive_joint": 0.0,
                "left_finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "right_finger_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
            },
        ),
    )

    holder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TubeHolder",
        spawn=sim_utils.UsdFileCfg(usd_path=str(HOLDER_USD), scale=HOLDER_SCALE, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=HOLDER_ROOT_POS_M),
    )

    vortexer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vortexer",
        spawn=sim_utils.UsdFileCfg(usd_path=str(VORTEXER_USD), scale=VORTEXER_SCALE, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=VORTEXER_ROOT_POS_M),
    )

    tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube",
        spawn=sim_utils.UsdFileCfg(usd_path=str(TUBE_USD), scale=TUBE_SCALE, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TUBE_INITIAL_POS_M),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.78, 0.82)),
    )


def main() -> None:
    require_assets()

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.05, 0.55, 1.15), target=(0.35, 0.02, 0.82))

    scene_cfg = ContactValidPlacementSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    print("[contact_valid_place_rl] Scene loaded.")
    print(f"[contact_valid_place_rl] Assets: {ASSET_DIR}")

    for _ in range(args_cli.sim_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    print("[contact_valid_place_rl] Simulation completed.")
    simulation_app.close()


if __name__ == "__main__":
    main()
