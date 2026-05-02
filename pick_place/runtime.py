from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import carb
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from wetlab_benchmark.randomization import apply_task_randomization
from wetlab_benchmark.task_config import ASSETS, FRAMES, IMPORTED_LAB_ASSETS, PICK_PLACE_RANDOMIZATION, SURFACE, TaskRandomizationCfg
from wetlab_benchmark.task_objects import HolderAsset, RobotAsset, TubeAsset, VortexerAsset

SCENE_DOME_LIGHT_INTENSITY = 650.0
SCENE_DOME_LIGHT_COLOR = (0.70, 0.75, 0.82)
ARM_JOINT_STIFFNESS = 30000.0
ARM_JOINT_DAMPING = 1200.0
ARM_JOINT_EFFORT_LIMIT = 300.0
ARM_JOINT_VELOCITY_LIMIT = 4.0
# Keep the imported xArm gripper compliant enough for contact-rich closure, but
# give it enough authority to hold a secured grasp without relying on continuous
# kinematic state writing.
GRIPPER_JOINT_STIFFNESS = 7000.0
GRIPPER_JOINT_DAMPING = 300.0
GRIPPER_JOINT_EFFORT_LIMIT = 75.0
IK_DLS_LAMBDA = 0.002
CONTACT_PHYSICS_DT = 0.0025
CONTACT_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0)
TUBE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    linear_damping=0.15,
    angular_damping=0.15,
    max_depenetration_velocity=0.25,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=8,
)
FIXTURE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=8,
)
TABLE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.55,
    dynamic_friction=0.45,
    restitution=0.0,
    friction_combine_mode="average",
    restitution_combine_mode="min",
)
FIXTURE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.20,
    dynamic_friction=0.10,
    restitution=0.0,
    friction_combine_mode="average",
    restitution_combine_mode="min",
)
TUBE_GLASS_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    # Rubber-on-glass can sustain a high static coefficient in dry contact.
    # Use max-combine so the fingertip rubber dominates the pair instead of
    # averaging back down to fixture-like plastic friction.
    static_friction=0.85,
    dynamic_friction=0.65,
    restitution=0.02,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)
FINGER_RUBBER_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.2,
    dynamic_friction=1.7,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
    compliant_contact_stiffness=6000.0,
    compliant_contact_damping=180.0,
)
ASSET_PROFILE_IMPORTED = "imported"
ASSET_PROFILE_CONTACT_REFINED = "contact_refined"
CONTACT_ASSET_DIR = Path(__file__).resolve().parents[1] / "contact_assets"
CONTACT_REFINED_XARM6_USD = CONTACT_ASSET_DIR / "xarm6_with_gripper_contact_refined.usd"
CONTACT_REFINED_TUBE_USD = CONTACT_ASSET_DIR / "50ml_conical_ep_tube_contact_refined.usd"
CONTACT_REFINED_HOLDER_USD = CONTACT_ASSET_DIR / "4_50ml_conical_holder_contact_refined.usd"
CONTACT_REFINED_VORTEXER_USD = CONTACT_ASSET_DIR / "vortexer_contact_refined.usd"
REFINED_FINGER_CONTACT_PATHS = [
    "{ENV_REGEX_NS}/Robot/left_finger/contact_pad",
    "{ENV_REGEX_NS}/Robot/right_finger/contact_pad",
    "{ENV_REGEX_NS}/Robot/left_finger/contact_shelf",
    "{ENV_REGEX_NS}/Robot/right_finger/contact_shelf",
    "{ENV_REGEX_NS}/Robot/left_finger/contact_v_groove_neg",
    "{ENV_REGEX_NS}/Robot/right_finger/contact_v_groove_neg",
    "{ENV_REGEX_NS}/Robot/left_finger/contact_v_groove_pos",
    "{ENV_REGEX_NS}/Robot/right_finger/contact_v_groove_pos",
    "{ENV_REGEX_NS}/Robot/left_finger/contact_x_stop_neg",
    "{ENV_REGEX_NS}/Robot/right_finger/contact_x_stop_neg",
    "{ENV_REGEX_NS}/Robot/left_finger/contact_x_stop_pos",
    "{ENV_REGEX_NS}/Robot/right_finger/contact_x_stop_pos",
]


def _robot_cfg(*, usd_path: Path) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path), activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
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
            }
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[FRAMES.arm_joint_regex],
                stiffness=ARM_JOINT_STIFFNESS,
                damping=ARM_JOINT_DAMPING,
                effort_limit_sim=ARM_JOINT_EFFORT_LIMIT,
                velocity_limit_sim=ARM_JOINT_VELOCITY_LIMIT,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[
                    "drive_joint",
                    "left_finger_joint",
                    "left_inner_knuckle_joint",
                    "right_outer_knuckle_joint",
                    "right_finger_joint",
                    "right_inner_knuckle_joint",
                ],
                stiffness=GRIPPER_JOINT_STIFFNESS,
                damping=GRIPPER_JOINT_DAMPING,
                effort_limit_sim=GRIPPER_JOINT_EFFORT_LIMIT,
            ),
        },
    )


def _tube_cfg(*, usd_path: Path) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            scale=IMPORTED_LAB_ASSETS.scale,
            activate_contact_sensors=True,
            rigid_props=TUBE_RIGID_PROPS,
            collision_props=CONTACT_COLLISION_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.008),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=IMPORTED_LAB_ASSETS.upright_quat_wxyz),
    )


def _tube_contact_cfg(*, include_refined_pads: bool) -> ContactSensorCfg:
    filters = []
    if include_refined_pads:
        filters.extend(REFINED_FINGER_CONTACT_PATHS)
    filters.extend(
        [
            "{ENV_REGEX_NS}/Robot/left_finger",
            "{ENV_REGEX_NS}/Robot/right_finger",
            "{ENV_REGEX_NS}/Robot/left_inner_knuckle",
            "{ENV_REGEX_NS}/Robot/right_inner_knuckle",
            "{ENV_REGEX_NS}/Robot/left_outer_knuckle",
            "{ENV_REGEX_NS}/Robot/right_outer_knuckle",
            "{ENV_REGEX_NS}/TubeHolder",
            "{ENV_REGEX_NS}/Vortexer",
        ]
    )
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Tube",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=filters,
    )


def _holder_contact_cfg(*, include_refined_pads: bool) -> ContactSensorCfg:
    filters = []
    if include_refined_pads:
        filters.extend(REFINED_FINGER_CONTACT_PATHS)
    filters.extend(
        [
            "{ENV_REGEX_NS}/Robot/left_finger",
            "{ENV_REGEX_NS}/Robot/right_finger",
            "{ENV_REGEX_NS}/Robot/left_inner_knuckle",
            "{ENV_REGEX_NS}/Robot/right_inner_knuckle",
            "{ENV_REGEX_NS}/Robot/left_outer_knuckle",
            "{ENV_REGEX_NS}/Robot/right_outer_knuckle",
            "{ENV_REGEX_NS}/Tube",
        ]
    )
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TubeHolder",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=filters,
    )


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.90, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=TABLE_PHYSICS_MATERIAL,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, SURFACE.z - 0.02)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=SCENE_DOME_LIGHT_INTENSITY,
            color=SCENE_DOME_LIGHT_COLOR,
        ),
    )
    robot = _robot_cfg(usd_path=ASSETS.xarm6_usd)
    vortexer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vortexer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS.vortexer_usd),
            scale=IMPORTED_LAB_ASSETS.scale,
            activate_contact_sensors=True,
            rigid_props=FIXTURE_RIGID_PROPS,
            collision_props=CONTACT_COLLISION_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=IMPORTED_LAB_ASSETS.upright_quat_wxyz),
    )
    holder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TubeHolder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS.tube_holder_combo_usd),
            scale=IMPORTED_LAB_ASSETS.holder_scale,
            activate_contact_sensors=True,
            rigid_props=FIXTURE_RIGID_PROPS,
            collision_props=CONTACT_COLLISION_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=IMPORTED_LAB_ASSETS.upright_quat_wxyz),
    )
    tube = _tube_cfg(usd_path=ASSETS.tube_usd)
    tube_contacts = _tube_contact_cfg(include_refined_pads=False)
    holder_contacts = _holder_contact_cfg(include_refined_pads=False)


@configclass
class ContactRefinedPickPlaceSceneCfg(PickPlaceSceneCfg):
    robot = _robot_cfg(usd_path=CONTACT_REFINED_XARM6_USD)
    vortexer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vortexer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CONTACT_REFINED_VORTEXER_USD),
            scale=IMPORTED_LAB_ASSETS.scale,
            activate_contact_sensors=True,
            rigid_props=FIXTURE_RIGID_PROPS,
            collision_props=CONTACT_COLLISION_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=IMPORTED_LAB_ASSETS.upright_quat_wxyz),
    )
    holder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TubeHolder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CONTACT_REFINED_HOLDER_USD),
            scale=IMPORTED_LAB_ASSETS.holder_scale,
            activate_contact_sensors=True,
            rigid_props=FIXTURE_RIGID_PROPS,
            collision_props=CONTACT_COLLISION_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=IMPORTED_LAB_ASSETS.upright_quat_wxyz),
    )
    tube = _tube_cfg(usd_path=CONTACT_REFINED_TUBE_USD)
    tube_contacts = _tube_contact_cfg(include_refined_pads=True)
    holder_contacts = _holder_contact_cfg(include_refined_pads=True)


@dataclass
class PickPlaceRuntime:
    sim: sim_utils.SimulationContext
    scene: InteractiveScene
    task_cfg: TaskRandomizationCfg
    robot: object
    holder: object
    tube: object
    vortexer: object
    tube_contacts: object
    holder_contacts: object
    assets: dict[str, object]
    sensors: dict[str, object]
    robot_asset: RobotAsset
    tube_asset: TubeAsset
    holder_asset: HolderAsset
    vortexer_asset: VortexerAsset
    controller: DifferentialIKController
    robot_entity_cfg: SceneEntityCfg
    ee_jacobi_idx: int
    layout: dict
    device: str
    num_envs: int


def _contact_physx_cfg():
    return sim_utils.PhysxCfg(
        solver_type=1,
        min_position_iteration_count=8,
        min_velocity_iteration_count=2,
        enable_ccd=True,
        bounce_threshold_velocity=0.05,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.006,
    )


def create_simulation(
    *,
    device: str,
    camera_eye: list[float],
    camera_target: list[float],
    dt: float = 0.01,
    contact_physics: bool = False,
    progress_callback=None,
):
    if progress_callback is not None:
        progress_callback("create_simulation:start")
    sim_cfg_kwargs = {"dt": dt, "device": device}
    if contact_physics:
        sim_cfg_kwargs["physx"] = _contact_physx_cfg()
        sim_cfg_kwargs["physics_material"] = FIXTURE_PHYSICS_MATERIAL
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(**sim_cfg_kwargs))
    if device.startswith("cuda"):
        # Isaac Sim 4.5's direct GPU API mode (`/physics/suppressReadback=true`) rejects dynamic
        # rigid-body pose/velocity writes during randomized scene initialization. The wet-lab
        # benchmark relies on those writes for tube placement, so disable that mode up-front.
        carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
    sim.set_camera_view(camera_eye, camera_target)
    if progress_callback is not None:
        progress_callback("create_simulation:done")
    return sim


def apply_contact_physics_materials(*, scene, asset_profile: str = ASSET_PROFILE_IMPORTED) -> None:
    """Bind task-specific physics materials to imported USD collision prims."""
    material_specs = {
        "/World/PhysicsMaterials/TablePlastic": TABLE_PHYSICS_MATERIAL,
        "/World/PhysicsMaterials/FixturePlastic": FIXTURE_PHYSICS_MATERIAL,
        "/World/PhysicsMaterials/TubeGlass": TUBE_GLASS_PHYSICS_MATERIAL,
        "/World/PhysicsMaterials/FingerRubber": FINGER_RUBBER_PHYSICS_MATERIAL,
    }
    for material_path, material_cfg in material_specs.items():
        material_cfg.func(material_path, material_cfg)

    for env_id in range(scene.num_envs):
        env_path = f"/World/envs/env_{env_id}"
        bindings = [
            (f"{env_path}/Table", "/World/PhysicsMaterials/TablePlastic"),
            (f"{env_path}/TubeHolder", "/World/PhysicsMaterials/FixturePlastic"),
            (f"{env_path}/Vortexer", "/World/PhysicsMaterials/FixturePlastic"),
            (f"{env_path}/Tube", "/World/PhysicsMaterials/TubeGlass"),
        ]
        if asset_profile == ASSET_PROFILE_CONTACT_REFINED:
            bindings.extend(
                [
                    (prim_path.replace("{ENV_REGEX_NS}", env_path), "/World/PhysicsMaterials/FingerRubber")
                    for prim_path in REFINED_FINGER_CONTACT_PATHS
                ]
            )
        else:
            bindings.extend(
                [
                    (f"{env_path}/Robot/left_finger", "/World/PhysicsMaterials/FingerRubber"),
                    (f"{env_path}/Robot/right_finger", "/World/PhysicsMaterials/FingerRubber"),
                    (f"{env_path}/Robot/left_inner_knuckle", "/World/PhysicsMaterials/FingerRubber"),
                    (f"{env_path}/Robot/right_inner_knuckle", "/World/PhysicsMaterials/FingerRubber"),
                    (f"{env_path}/Robot/left_outer_knuckle", "/World/PhysicsMaterials/FingerRubber"),
                    (f"{env_path}/Robot/right_outer_knuckle", "/World/PhysicsMaterials/FingerRubber"),
                ]
            )
        for prim_path, material_path in bindings:
            try:
                sim_utils.bind_physics_material(prim_path, material_path)
            except ValueError as exc:
                carb.log_warn(f"Unable to bind physics material {material_path} to {prim_path}: {exc}")


def _scene_cfg_class(asset_profile: str):
    if asset_profile == ASSET_PROFILE_IMPORTED:
        return PickPlaceSceneCfg
    if asset_profile == ASSET_PROFILE_CONTACT_REFINED:
        missing_assets = [
            path
            for path in (
                CONTACT_REFINED_XARM6_USD,
                CONTACT_REFINED_TUBE_USD,
                CONTACT_REFINED_HOLDER_USD,
                CONTACT_REFINED_VORTEXER_USD,
            )
            if not path.exists()
        ]
        if missing_assets:
            raise FileNotFoundError(
                "Contact-refined assets are missing. Run "
                "`/home/ubuntu/IsaacLab/_isaac_sim/python.sh "
                "wetlab_benchmark/pick_place/live_exec/build_contact_refined_assets.py --force` first. "
                f"Missing: {', '.join(str(path) for path in missing_assets)}"
            )
        return ContactRefinedPickPlaceSceneCfg
    raise ValueError(f"Unsupported asset_profile '{asset_profile}'")


def create_scene(
    *,
    sim,
    num_envs: int,
    env_spacing: float = 2.5,
    asset_profile: str = ASSET_PROFILE_IMPORTED,
    progress_callback=None,
):
    if progress_callback is not None:
        progress_callback("create_scene:resolve_cfg")
    scene_cfg_class = _scene_cfg_class(asset_profile)
    if progress_callback is not None:
        progress_callback("create_scene:construct")
    scene = InteractiveScene(scene_cfg_class(num_envs=num_envs, env_spacing=env_spacing))
    if progress_callback is not None:
        progress_callback("create_scene:sim_reset")
    sim.reset()
    if progress_callback is not None:
        progress_callback("create_scene:scene_reset")
    scene.reset()
    if progress_callback is not None:
        progress_callback("create_scene:done")
    return scene


def initialize_pick_place_runtime(
    *,
    sim,
    scene,
    seed: int,
    task_cfg: TaskRandomizationCfg = PICK_PLACE_RANDOMIZATION,
    asset_profile: str = ASSET_PROFILE_IMPORTED,
    progress_callback=None,
) -> PickPlaceRuntime:
    if progress_callback is not None:
        progress_callback("initialize:start")
    torch.manual_seed(seed)

    robot = scene["robot"]
    holder = scene["holder"]
    tube = scene["tube"]
    vortexer = scene["vortexer"]
    tube_contacts = scene["tube_contacts"]
    holder_contacts = scene["holder_contacts"]
    assets = {
        "robot": robot,
        "holder": holder,
        "vortexer": vortexer,
        "tube": tube,
    }
    sensors = {
        "tube_contacts": tube_contacts,
        "holder_contacts": holder_contacts,
    }

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[FRAMES.arm_joint_regex], body_names=[FRAMES.ee_body_name])
    if progress_callback is not None:
        progress_callback("initialize:resolve_robot_entity")
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    if progress_callback is not None:
        progress_callback("initialize:create_ik_controller")
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": IK_DLS_LAMBDA},
        ),
        num_envs=scene.num_envs,
        device=sim.device,
    )

    pre_support_clearance_m = 0.065
    tube_cfg = next(cfg for cfg in task_cfg.objects if cfg.name == "tube")

    if progress_callback is not None:
        progress_callback("initialize:randomize_layout")
    layout = apply_task_randomization(scene, assets, task_cfg)
    if progress_callback is not None:
        progress_callback("initialize:first_scene_write")
    scene.write_data_to_sim()
    if progress_callback is not None:
        progress_callback("initialize:first_sim_step")
    sim.step()
    if progress_callback is not None:
        progress_callback("initialize:first_scene_update")
    scene.update(sim.get_physics_dt())
    if progress_callback is not None:
        progress_callback("initialize:bind_contact_materials")
    apply_contact_physics_materials(scene=scene, asset_profile=asset_profile)
    if progress_callback is not None:
        progress_callback("initialize:final_scene_write")
    scene.write_data_to_sim()

    if progress_callback is not None:
        progress_callback("initialize:build_runtime_dataclass")
    return PickPlaceRuntime(
        sim=sim,
        scene=scene,
        task_cfg=task_cfg,
        robot=robot,
        holder=holder,
        tube=tube,
        vortexer=vortexer,
        tube_contacts=tube_contacts,
        holder_contacts=holder_contacts,
        assets=assets,
        sensors=sensors,
        robot_asset=RobotAsset(
            robot,
            ee_body_id=robot_entity_cfg.body_ids[0],
            ee_local_offset_m=FRAMES.ee_control_offset_local_m,
        ),
        tube_asset=TubeAsset(
            tube,
            local_grasp_point_m=(
                IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0],
                IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1],
                IMPORTED_LAB_ASSETS.tube_top_from_root_m,
            ),
            local_support_point_m=tube_cfg.local_support_point_m,
            default_grasp_quat=FRAMES.ee_top_grasp_quat_wxyz,
        ),
        holder_asset=HolderAsset(
            holder,
            slot_centers_local_m=IMPORTED_LAB_ASSETS.holder_slot_centers_local_m,
            default_slot_index=1,
            preplace_height_offset_m=pre_support_clearance_m,
            default_place_quat=FRAMES.ee_top_grasp_quat_wxyz,
        ),
        vortexer_asset=VortexerAsset(
            vortexer,
            support_center_local_m=IMPORTED_LAB_ASSETS.vortexer_support_center_local_m,
            presupport_height_offset_m=pre_support_clearance_m,
            default_support_quat=FRAMES.ee_top_grasp_quat_wxyz,
        ),
        controller=controller,
        robot_entity_cfg=robot_entity_cfg,
        ee_jacobi_idx=ee_jacobi_idx,
        layout=layout,
        device=sim.device,
        num_envs=scene.num_envs,
    )


def create_pick_place_runtime(
    *,
    num_envs: int,
    seed: int,
    device: str,
    camera_eye: list[float],
    camera_target: list[float],
    env_spacing: float = 2.5,
    dt: float = 0.01,
    contact_physics: bool = False,
    task_cfg: TaskRandomizationCfg = PICK_PLACE_RANDOMIZATION,
    asset_profile: str = ASSET_PROFILE_IMPORTED,
    progress_callback=None,
) -> PickPlaceRuntime:
    if progress_callback is not None:
        progress_callback("runtime:create_simulation")
    sim = create_simulation(
        device=device,
        camera_eye=camera_eye,
        camera_target=camera_target,
        dt=dt,
        contact_physics=contact_physics,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback("runtime:create_scene")
    scene = create_scene(
        sim=sim,
        num_envs=num_envs,
        env_spacing=env_spacing,
        asset_profile=asset_profile,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback("runtime:initialize")
    return initialize_pick_place_runtime(
        sim=sim,
        scene=scene,
        seed=seed,
        task_cfg=task_cfg,
        asset_profile=asset_profile,
        progress_callback=progress_callback,
    )
