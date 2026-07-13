"""Direct RL environment for tube placement.

This file is intentionally a scaffold for now.

Design direction:

- Scripted IK controls the large motion phases.
- The policy only controls the final release residual.
- The gripper follows fixed open/close rules.
"""

from __future__ import annotations

import torch
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade
from isaaclab.controllers import DifferentialIKController
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, subtract_frame_transforms
from contact_valid_place_rl.tasks.direct.tube_place.tube_place_env_cfg import TubePlaceEnvCfg

PHASE_PREGRASP = 0
PHASE_GRASP = 1
PHASE_CLOSE = 2
PHASE_LIFT = 3
PHASE_TRANSFER = 4
PHASE_RL_RELEASE = 5
PHASE_OPEN = 6
PHASE_SETTLE = 7


class TubePlaceEnv(DirectRLEnv):
    def __init__(self, cfg: TubePlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.filtered_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.release_pose_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.phase_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.bilateral_contact_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.grasp_contact_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gripper_hold_target = torch.full(
            (self.num_envs, 1), self.cfg.gripper_closed_position, device=self.device
        )
        self.release_alignment_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.release_descent_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.release_descent_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_loss_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.success_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.arm_joint_ids = self.robot.find_joints(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])[0]
        self.gripper_joint_ids = self.robot.find_joints(["drive_joint"])[0]
        self.end_effector_body_id = self.robot.find_bodies("link_tcp")[0][0]
        self.jacobian_body_indices = torch.full(
            (self.num_envs,), self.end_effector_body_id - 1, dtype=torch.long, device=self.device
        )
        self.ik_controller = DifferentialIKController(self.cfg.ik_controller, num_envs=self.num_envs, device=self.device)
        self.pregrasp_start_pose = self.robot.data.body_pose_w[:, self.end_effector_body_id].clone()
        self.capture_pregrasp_start = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.pregrasp_reached_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)


    def _setup_scene(self):
        robot_cfg = self.cfg.robot.replace(prim_path="/World/envs/env_.*/Robot")
        table_cfg = self.cfg.table.replace(prim_path="/World/envs/env_.*/Table")
        holder_cfg = self.cfg.holder.replace(prim_path="/World/envs/env_.*/TubeHolder")
        vortexer_cfg = self.cfg.vortexer.replace(prim_path="/World/envs/env_.*/Vortexer")
        vortexer_body_cfg = self.cfg.vortexer_body.replace(prim_path="/World/envs/env_.*/VortexerBodyCollision")
        tube_cfg = self.cfg.tube.replace(prim_path="/World/envs/env_.*/Tube")

        self.robot = Articulation(robot_cfg)
        self.holder = RigidObject(holder_cfg)
        self.vortexer = RigidObject(vortexer_cfg)
        self.vortexer_body = RigidObject(vortexer_body_cfg)
        self.tube = RigidObject(tube_cfg)
        self.table = RigidObject(table_cfg)
        self.left_finger_contact = ContactSensor(self.cfg.left_finger_contact)
        self.right_finger_contact = ContactSensor(self.cfg.right_finger_contact)
        self.tube_vortexer_contact = ContactSensor(self.cfg.tube_vortexer_contact)

        self.scene.clone_environments(copy_from_source=False)
        self._stabilize_gripper_mimic_joints()
        self._apply_render_materials()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["holder"] = self.holder
        self.scene.rigid_objects["vortexer"] = self.vortexer
        self.scene.rigid_objects["vortexer_body"] = self.vortexer_body
        self.scene.rigid_objects["tube"] = self.tube
        self.scene.sensors["left_finger_contact"] = self.left_finger_contact
        self.scene.sensors["right_finger_contact"] = self.right_finger_contact
        self.scene.sensors["tube_vortexer_contact"] = self.tube_vortexer_contact

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_render_materials(self) -> None:
        """Bind high-contrast PBR materials without changing collision geometry."""

        stage = omni.usd.get_context().get_stage()

        def make_material(
            name: str,
            color: tuple[float, float, float],
            *,
            roughness: float,
            metallic: float = 0.0,
            opacity: float = 1.0,
        ) -> UsdShade.Material:
            material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material

        def bind_recursive(root_path: str, material: UsdShade.Material) -> None:
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                return
            descendants = list(root.GetChildren())
            for prim in descendants:
                descendants.extend(prim.GetChildren())
                if prim.IsA(UsdGeom.Gprim) and prim.GetAttribute("visibility").Get() != "invisible":
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            if root.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(root).Bind(material)

        tube_glass_material = make_material(
            "RenderTubeGlass", (0.58, 0.61, 0.64), roughness=0.30, opacity=1.0
        )
        tube_cap_material = make_material("RenderTubeCap", (0.12, 0.14, 0.16), roughness=0.34)

        for env_id in range(self.num_envs):
            env_path = f"/World/envs/env_{env_id}"
            bind_recursive(f"{env_path}/Tube/Visuals/body_visual", tube_glass_material)
            bind_recursive(f"{env_path}/Tube/Visuals/cap_visual", tube_cap_material)

    def _stabilize_gripper_mimic_joints(self) -> None:
        """Critically damp the underdamped mimic joints from the imported gripper USD."""

        stage = omni.usd.get_context().get_stage()
        follower_names = {
            "left_finger_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint",
            "right_inner_knuckle_joint",
        }
        for prim in stage.Traverse():
            if prim.GetName() not in follower_names or "/Robot/" not in str(prim.GetPath()):
                continue
            prim.GetAttribute("physxMimicJoint:rotX:naturalFrequency").Set(500.0)
            prim.GetAttribute("physxMimicJoint:rotX:dampingRatio").Set(1.0)

        # The authored open gap is narrower than this 50 ml tube. Shift the
        # thin pad and its matching visuals together; disable auxiliary shelves
        # and stops so a lift can only be supported by bilateral pad friction.
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/Robot/left_finger/" in path:
                side_sign = -1.0
            elif "/Robot/right_finger/" in path:
                side_sign = 1.0
            else:
                continue
            name = prim.GetName()
            if name.startswith("contact_") and name != "contact_pad":
                collision_enabled = prim.GetAttribute("physics:collisionEnabled")
                if collision_enabled:
                    collision_enabled.Set(False)
                continue
            if name in {"contact_pad", "visual_contact_pad"}:
                translate = prim.GetAttribute("xformOp:translate")
                if translate:
                    translate.Set((0.0, side_sign * 0.010, 0.0275))
                scale = prim.GetAttribute("xformOp:scale")
                if scale:
                    scale.Set((0.028, 0.004, 0.025))

    def _compute_pregrasp_pose(self):
        """Return the end-effector target pose above the tube.

        Shape: [num_envs, 7] = [x, y, z, qw, qx, qy, qz].
        """

        tube_root_pos = self._tube_reset_pos_w()
        pregrasp_z_from_root = self.cfg.tube_height_m + self.cfg.pregrasp_clearance_m
        pregrasp_pos = self._offset_world_z(tube_root_pos, pregrasp_z_from_root)
        pregrasp_pos[:, 0] += self.cfg.pregrasp_xy_offset_m[0]
        pregrasp_pos[:, 1] += self.cfg.pregrasp_xy_offset_m[1]
        return self._make_top_grasp_pose(pregrasp_pos)

    def _compute_grasp_pose(self):
        """Return the end-effector target pose at the tube cap.

        Shape: [num_envs, 7] = [x, y, z, qw, qx, qy, qz].
        """

        tube_root_pos = self._tube_reset_pos_w()
        grasp_z_from_root = self.cfg.tube_height_m + self.cfg.grasp_height_from_tube_top_m
        grasp_pos = self._offset_world_z(tube_root_pos, grasp_z_from_root)
        return self._make_top_grasp_pose(grasp_pos)

    def _compute_close_pose(self):
        """Return the compensated TCP pose that centers the loaded fingers."""

        close_pose = self._compute_grasp_pose()
        close_pose[:, 0] += self.cfg.grasp_xy_offset_m[0]
        close_pose[:, 1] += self.cfg.grasp_xy_offset_m[1]
        return close_pose

    def _compute_lift_pose(self):
        """Return the end-effector target pose after lifting the grasped tube.

        Shape: [num_envs, 7] = [x, y, z, qw, qx, qy, qz].
        """

        grasp_pose = self._compute_close_pose()
        lift_pose = grasp_pose.clone()
        lift_pose[:, 0] += self.cfg.lift_tcp_xy_correction_m[0]
        lift_pose[:, 1] += self.cfg.lift_tcp_xy_correction_m[1]
        lift_pose[:, 2] += self.cfg.lift_to_grasp_z_offset_m
        return lift_pose

    def _compute_transfer_pose(self):
        """Return a safe transfer pose above the vortexer rim.

        Shape: [num_envs, 7] = [x, y, z, qw, qx, qy, qz].
        """

        vortexer_root_pos = self.vortexer.data.root_pos_w
        transfer_pos = self._vortexer_center_pos_w()
        transfer_pos[:, 0] += self.cfg.carried_tcp_xy_compensation_m[0]
        transfer_pos[:, 1] += self.cfg.carried_tcp_xy_compensation_m[1]
        transfer_pos[:, 2] = (
            vortexer_root_pos[:, 2]
            + self.cfg.vortexer_top_from_root_m
            + self.cfg.transfer_to_vortexer_z_offset_m
        )
        return self._make_top_grasp_pose(transfer_pos)

    def _compute_nominal_release_pose(self):
        """Return the nominal release pose over the vortexer.

        Shape: [num_envs, 7] = [x, y, z, qw, qx, qy, qz].
        """

        vortexer_root_pos = self.vortexer.data.root_pos_w
        release_pos = self._vortexer_center_pos_w()
        release_pos[:, 0] += self.cfg.carried_tcp_xy_compensation_m[0]
        release_pos[:, 1] += self.cfg.carried_tcp_xy_compensation_m[1]
        release_pos[:, 0] += self.cfg.release_tcp_xy_correction_m[0]
        release_pos[:, 1] += self.cfg.release_tcp_xy_correction_m[1]
        grasp_z_from_root = self.cfg.tube_height_m + self.cfg.grasp_height_from_tube_top_m
        release_pos[:, 2] = (
            vortexer_root_pos[:, 2]
            + self.cfg.vortexer_top_from_root_m
            - self.cfg.release_insertion_below_rim_m
            + grasp_z_from_root
        )
        return self._make_top_grasp_pose(release_pos)

    def _compute_release_approach_pose(self):
        """Return the release XY target while retaining transfer clearance."""

        approach_pose = self._compute_corrected_release_pose()
        approach_pose[:, 2] = self._compute_transfer_pose()[:, 2]
        return approach_pose

    def _compute_retreat_pose(self):
        """Return a collision-free pose above the released tube."""

        retreat_pose = self._compute_corrected_release_pose()
        retreat_pose[:, 2] += self.cfg.retreat_from_release_m
        return retreat_pose

    def _compute_release_clear_pose(self):
        """Return a small upward motion that clears fingers from the well."""

        clear_pose = self._compute_corrected_release_pose()
        clear_pose[:, 2] += self.cfg.release_clearance_m
        return clear_pose

    def _compute_corrected_release_pose(self):
        """Return the release pose after applying the policy residual.

        Shape: [num_envs, 7] = [x, y, z, qw, qx, qy, qz].
        """

        release_pose = self._compute_nominal_release_pose()
        action_scale_xyz = torch.tensor(
            self.cfg.action_scale_xyz,
            device=self.device,
            dtype=release_pose.dtype,
        )
        release_pose[:, :3] += self.release_pose_bias
        release_pose[:, :3] += self.filtered_actions[:, :3] * action_scale_xyz
        return release_pose

    def _interpolate_pose(self, start_pose: torch.Tensor, end_pose: torch.Tensor, duration_steps: int) -> torch.Tensor:
        """Smoothly interpolate between phase waypoints using phase-local time."""

        alpha = torch.clamp(self.phase_step.float() / float(duration_steps), 0.0, 1.0)
        alpha = alpha.square() * (3.0 - 2.0 * alpha)
        return start_pose + alpha.unsqueeze(-1) * (end_pose - start_pose)

    def _interpolate_pose_window(
        self, start_pose: torch.Tensor, end_pose: torch.Tensor, start_step: int, duration_steps: int
    ) -> torch.Tensor:
        """Smoothly interpolate within a phase-local step window."""

        alpha = torch.clamp(
            (self.phase_step.float() - float(start_step)) / float(duration_steps), 0.0, 1.0
        )
        alpha = alpha.square() * (3.0 - 2.0 * alpha)
        return start_pose + alpha.unsqueeze(-1) * (end_pose - start_pose)

    @staticmethod
    def _interpolate_pose_progress(
        start_pose: torch.Tensor, end_pose: torch.Tensor, progress: torch.Tensor, duration_steps: int
    ) -> torch.Tensor:
        """Smoothly interpolate from an explicit per-environment progress counter."""

        alpha = torch.clamp(progress.float() / float(duration_steps), 0.0, 1.0)
        alpha = alpha.square() * (3.0 - 2.0 * alpha)
        return start_pose + alpha.unsqueeze(-1) * (end_pose - start_pose)

    def _make_top_grasp_pose(self, position_w: torch.Tensor) -> torch.Tensor:
        """Create a batched top-grasp pose from world-frame positions."""

        quat_w = torch.tensor(
            self.cfg.top_grasp_quat_wxyz,
            device=self.device,
            dtype=position_w.dtype,
        ).repeat(self.num_envs, 1)
        return torch.cat((position_w, quat_w), dim=-1)

    def _offset_world_z(self, position_w: torch.Tensor, z_offset_m: float) -> torch.Tensor:
        """Offset a batched world-frame position along world z."""

        offset = torch.tensor(
            [0.0, 0.0, z_offset_m],
            device=self.device,
            dtype=position_w.dtype,
        )
        return position_w + offset

    def _tube_reset_pos_w(self) -> torch.Tensor:
        """Return the deterministic seated tube root position in world frame."""

        reset_pos = torch.as_tensor(
            self.cfg.tube_reset_root_pos_m,
            device=self.device,
            dtype=self.tube.data.root_pos_w.dtype,
        ).repeat(self.num_envs, 1)
        return reset_pos + self.scene.env_origins

    def _world_up(self, dtype: torch.dtype) -> torch.Tensor:
        """Return one world-up vector per environment."""

        return torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=dtype).repeat(self.num_envs, 1)

    def _vortexer_center_pos_w(self) -> torch.Tensor:
        """Return the vortexer target center position in world frame.

        First version assumes the vortexer has no yaw rotation, so local XY can
        be added directly to world XY. If the vortexer is randomized in yaw later,
        rotate this local offset by the vortexer orientation first.
        """

        vortexer_root_pos = self.vortexer.data.root_pos_w
        center_pos = vortexer_root_pos.clone()
        center_pos[:, 0] += self.cfg.vortexer_center_local_xy[0]
        center_pos[:, 1] += self.cfg.vortexer_center_local_xy[1]
        return center_pos

    def _ee_reached(self, pose: torch.Tensor, threshold_m: float | None = None) -> torch.Tensor:
        """Return whether the end-effector reached a target pose.

        Only position is checked in the first version. Orientation tolerance can
        be added later after the IK loop is working.
        """

        ee_pose_w = self.robot.data.body_pose_w[:, self.end_effector_body_id]
        ee_pos_w = ee_pose_w[:, :3]
        pos_error = torch.linalg.norm(ee_pos_w - pose[:, :3], dim=-1)
        threshold = self.cfg.ee_reached_threshold_m if threshold_m is None else threshold_m
        return pos_error < threshold

    def _gripper_closed(self) -> torch.Tensor:
        """Return whether the driver reached a contact-limited grasp position."""

        gripper_pos = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        return torch.min(gripper_pos, dim=-1).values > self.cfg.gripper_grasp_position_threshold

    def _gripper_open(self) -> torch.Tensor:
        """Return whether the gripper joints are near the open command."""

        gripper_pos = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        target = torch.full_like(gripper_pos, self.cfg.gripper_open_position)
        error = torch.max(torch.abs(gripper_pos - target), dim=-1).values
        return error < self.cfg.gripper_position_tolerance

    def _gripper_released(self) -> torch.Tensor:
        """Return whether the jaws opened enough to stop retaining the tube."""

        gripper_pos = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        return torch.max(gripper_pos, dim=-1).values < self.cfg.gripper_release_position_threshold

    def _bilateral_tube_contact(self) -> torch.Tensor:
        """Return whether both finger pads exert force on the tube."""

        left_force, right_force = self._finger_tube_contact_forces()
        threshold = self.cfg.finger_contact_force_threshold_n
        return (left_force > threshold) & (right_force > threshold)

    def _finger_tube_contact_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return maximum filtered tube-contact force for each finger."""

        left_force = torch.linalg.norm(self.left_finger_contact.data.force_matrix_w, dim=-1).amax(dim=(1, 2))
        right_force = torch.linalg.norm(self.right_finger_contact.data.force_matrix_w, dim=-1).amax(dim=(1, 2))
        return left_force, right_force

    def _tube_at_grasp_pose(self) -> torch.Tensor:
        """Return whether the TCP surrounds the tube's cap before closing."""

        ee_pos_w = self.robot.data.body_pose_w[:, self.end_effector_body_id, :3]
        tube_root_pos_w = self.tube.data.root_pos_w
        grasp_z = self.cfg.tube_height_m + self.cfg.grasp_height_from_tube_top_m
        expected_ee_pos_w = self._offset_world_z(tube_root_pos_w, grasp_z)
        return torch.linalg.norm(ee_pos_w - expected_ee_pos_w, dim=-1) < self.cfg.grasp_pose_tolerance_m

    def _tube_supported_by_vortexer(self) -> torch.Tensor:
        """Return whether the tube has measurable contact with the vortexer."""

        return self._tube_vortexer_contact_force() > self.cfg.vortexer_support_force_threshold_n

    def _tube_vortexer_contact_force(self) -> torch.Tensor:
        """Return the maximum filtered tube-to-vortexer contact force."""

        force = torch.linalg.norm(self.tube_vortexer_contact.data.force_matrix_w, dim=-1).amax(dim=(1, 2))
        return force

    def _tube_held(self) -> torch.Tensor:
        """Return whether the tube is still following the closed gripper."""

        ee_pos_w = self.robot.data.body_pose_w[:, self.end_effector_body_id, :3]
        grasp_z = self.cfg.tube_height_m + self.cfg.grasp_height_from_tube_top_m
        expected_root_pos_w = self._offset_world_z(ee_pos_w, -grasp_z)
        return torch.linalg.norm(self.tube.data.root_pos_w - expected_root_pos_w, dim=-1) < self.cfg.held_tube_tolerance_m

    def _tube_lifted(self) -> torch.Tensor:
        """Return whether the tube has been lifted from its reset height."""

        lift_delta = self.tube.data.root_pos_w[:, 2] - self.cfg.tube_initial_pos_m[2]
        return lift_delta > self.cfg.lift_success_delta_m

    def _tube_stable(self) -> torch.Tensor:
        """Return whether the tube is moving slowly enough to count as settled."""

        lin_speed = torch.linalg.norm(self.tube.data.root_lin_vel_w, dim=-1)
        # Axial spin is irrelevant for an approximately rotationally symmetric tube.
        ang_speed = torch.linalg.norm(self.tube.data.root_ang_vel_w[:, :2], dim=-1)
        return (
            (lin_speed < self.cfg.tube_stable_lin_vel_threshold_mps)
            & (ang_speed < self.cfg.tube_stable_ang_vel_threshold_radps)
        )

    def _tube_inside_vortexer_xy(self) -> torch.Tensor:
        """Return whether the tube is horizontally close to the vortexer center."""

        tube_xy = self.tube.data.root_pos_w[:, :2]
        target_xy = self._vortexer_center_pos_w()[:, :2]
        xy_error = torch.linalg.norm(tube_xy - target_xy, dim=-1)
        return xy_error < self.cfg.vortexer_success_xy_tolerance_m

    def _tube_aligned_for_release(self) -> torch.Tensor:
        """Return whether the carried tube is close enough to descend into the well."""

        tube_xy = self.tube.data.root_pos_w[:, :2]
        target_xy = self._vortexer_center_pos_w()[:, :2]
        return torch.linalg.norm(tube_xy - target_xy, dim=-1) < self.cfg.transfer_tube_xy_tolerance_m

    def _tube_inserted(self) -> torch.Tensor:
        """Return whether the tube is upright and physically below the well rim."""

        rim_z_w = self.vortexer.data.root_pos_w[:, 2] + self.cfg.vortexer_top_from_root_m
        floor_z_w = self.vortexer.data.root_pos_w[:, 2] + self.cfg.vortexer_floor_from_root_m
        insertion_depth = rim_z_w - self.tube.data.root_pos_w[:, 2]
        tube_up_axis_w = quat_apply(
            self.tube.data.root_pose_w[:, 3:7],
            self._world_up(self.tube.data.root_pos_w.dtype),
        )
        return (
            self._tube_inside_vortexer_xy()
            & (insertion_depth > self.cfg.vortexer_min_insertion_below_rim_m)
            & (self.tube.data.root_pos_w[:, 2] >= floor_z_w - self.cfg.tube_floor_tolerance_m)
            & (tube_up_axis_w[:, 2] > self.cfg.upright_success_threshold)
        )

    def _phase_timed_out(self) -> torch.Tensor:
        """Return whether the current phase has exceeded its safety timeout."""

        return self.phase_step > self.cfg.phase_timeout_steps

    def _advance_phase(self, env_ids: torch.Tensor):
        """Advance the phase for the given environment IDs."""

        if env_ids.dtype == torch.bool:
            env_ids = env_ids.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        self.phase[env_ids] += 1
        self.phase_step[env_ids] = 0

    def _update_phase(self) -> None:
        """Advance each environment when its current phase event is complete."""

        self.phase_step += 1

        pregrasp_reached = (self.phase == PHASE_PREGRASP) & self._ee_reached(
            self._compute_pregrasp_pose(), self.cfg.pregrasp_reached_threshold_m
        )
        self.pregrasp_reached_step = torch.where(
            pregrasp_reached,
            self.pregrasp_reached_step + 1,
            torch.zeros_like(self.pregrasp_reached_step),
        )
        pregrasp_done = self.pregrasp_reached_step >= self.cfg.pregrasp_reached_dwell_steps
        grasp_done = (self.phase == PHASE_GRASP) & self._ee_reached(
            self._compute_grasp_pose(), self.cfg.grasp_pose_tolerance_m
        )
        stable_close_contact = (
            (self.phase == PHASE_CLOSE)
            & self._gripper_closed()
            & self._bilateral_tube_contact()
            & self._tube_held()
        )
        new_contact = stable_close_contact & ~self.grasp_contact_latched
        if torch.any(new_contact):
            self.gripper_hold_target[new_contact] = self.robot.data.joint_pos[new_contact][
                :, self.gripper_joint_ids
            ] + self.cfg.gripper_contact_preload
            self.gripper_hold_target[new_contact].clamp_(max=self.cfg.gripper_closed_position)
            self.grasp_contact_latched[new_contact] = True
        self.bilateral_contact_step = torch.where(
            stable_close_contact,
            self.bilateral_contact_step + 1,
            torch.zeros_like(self.bilateral_contact_step),
        )
        close_done = stable_close_contact & (
            self.bilateral_contact_step >= self.cfg.bilateral_contact_dwell_steps
        )
        release_aligned = (
            (self.phase == PHASE_RL_RELEASE)
            & ~self.release_descent_started
            & self._tube_inside_vortexer_xy()
            & self._bilateral_tube_contact()
            & self._tube_held()
        )
        self.release_alignment_step = torch.where(
            release_aligned,
            self.release_alignment_step + 1,
            torch.zeros_like(self.release_alignment_step),
        )
        self.release_descent_started |= (
            self.release_alignment_step >= self.cfg.release_alignment_dwell_steps
        )
        self.release_descent_step = torch.where(
            self.release_descent_started,
            self.release_descent_step + 1,
            torch.zeros_like(self.release_descent_step),
        )
        lift_done = (
            (self.phase == PHASE_LIFT)
            & (self.phase_step >= self.cfg.lift_motion_steps)
            & self._ee_reached(self._compute_lift_pose(), self.cfg.loaded_ee_reached_threshold_m)
            & self._tube_lifted()
            & self._tube_held()
        )
        transfer_done = (
            (self.phase == PHASE_TRANSFER)
            & (self.phase_step >= self.cfg.transfer_motion_steps)
            & self._ee_reached(self._compute_transfer_pose(), self.cfg.loaded_ee_reached_threshold_m)
            & self._tube_held()
            & self._bilateral_tube_contact()
            & self._tube_aligned_for_release()
        )
        release_done = (
            (self.phase == PHASE_RL_RELEASE)
            & (self.release_descent_step >= self.cfg.release_motion_steps)
            & self._ee_reached(self._compute_corrected_release_pose(), self.cfg.loaded_ee_reached_threshold_m)
            & self._tube_held()
            & self._bilateral_tube_contact()
            & self._tube_inside_vortexer_xy()
            & self._tube_supported_by_vortexer()
        )
        open_done = (
            (self.phase == PHASE_OPEN)
            & (self.phase_step >= self.cfg.open_dwell_steps)
            & self._gripper_released()
            & self._tube_supported_by_vortexer()
        )
        done = (
            pregrasp_done
            | grasp_done
            | close_done
            | lift_done
            | transfer_done
            | release_done
            | open_done
        )
        self._advance_phase(done)


    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-physics step for the environment."""
        if torch.any(self.capture_pregrasp_start):
            self.pregrasp_start_pose[self.capture_pregrasp_start] = self.robot.data.body_pose_w[
                self.capture_pregrasp_start, self.end_effector_body_id
            ]
            self.capture_pregrasp_start[self.capture_pregrasp_start] = False

        self.previous_actions = self.actions.clone()
        self.actions = torch.clamp(actions.clone(), -1.0, 1.0)

        release_mask = self.phase == PHASE_RL_RELEASE
        alpha = self.cfg.action_filter_alpha
        self.filtered_actions[release_mask] = (
            alpha * self.filtered_actions[release_mask]
            + (1.0 - alpha) * self.actions[release_mask]
        )
        self.filtered_actions[~release_mask] = 0.0

    def _apply_action(self):
        target_pose_w = self._target_pose_for_current_phase()

        root_pose_w = self.robot.data.root_pose_w
        ee_pose_w = self.robot.data.body_pose_w[:, self.end_effector_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pose_w[:, 0:3], target_pose_w[:, 3:7]
        )
        jacobians_w = self.robot.root_physx_view.get_jacobians()
        env_indices = torch.arange(self.num_envs, device=self.device)
        jacobian_w = jacobians_w[env_indices, self.jacobian_body_indices, :, :][:, :, self.arm_joint_ids]
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_b[:, 3:, :])

        self.ik_controller.set_command(target_pos_b, ee_pos_b, ee_quat_b)
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        transfer_alpha = torch.clamp(
            self.phase_step.float() / float(self.cfg.wrist_yaw_blend_steps), 0.0, 1.0
        )
        transfer_alpha = transfer_alpha.square() * (3.0 - 2.0 * transfer_alpha)
        transfer_alpha = torch.where(self.phase >= PHASE_TRANSFER, transfer_alpha, torch.zeros_like(transfer_alpha))
        wrist_yaw_des = joint_pos_des[:, 0] - self.cfg.wrist_yaw_joint1_reference
        joint_pos_des[:, 5] = torch.lerp(joint_pos_des[:, 5], wrist_yaw_des, transfer_alpha)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.arm_joint_ids)

        gripper_target = torch.full(
            (self.num_envs, len(self.gripper_joint_ids)),
            self.cfg.gripper_open_position,
            device=self.device,
            dtype=joint_pos.dtype,
        )
        closed_mask = (self.phase >= PHASE_CLOSE) & (self.phase <= PHASE_RL_RELEASE)
        gripper_target[closed_mask] = self.cfg.gripper_closed_position
        hold_mask = closed_mask & self.grasp_contact_latched
        gripper_target[hold_mask] = self.gripper_hold_target[hold_mask]
        self.robot.set_joint_position_target(gripper_target, joint_ids=self.gripper_joint_ids)

    def _target_pose_for_current_phase(self) -> torch.Tensor:
        """Return the batched IK target pose for each environment's phase."""

        pregrasp_pose = self._compute_pregrasp_pose()
        pregrasp_above = pregrasp_pose.clone()
        pregrasp_above[:, 2] = self.pregrasp_start_pose[:, 2]
        pregrasp_align_target = self._interpolate_pose(
            self.pregrasp_start_pose, pregrasp_above, self.cfg.pregrasp_align_steps
        )
        pregrasp_descent_target = self._interpolate_pose_window(
            pregrasp_above,
            pregrasp_pose,
            self.cfg.pregrasp_align_steps,
            self.cfg.pregrasp_descent_steps,
        )
        pregrasp_target = torch.where(
            (self.phase_step < self.cfg.pregrasp_align_steps).unsqueeze(-1),
            pregrasp_align_target,
            pregrasp_descent_target,
        )
        target_pose = pregrasp_target.clone()

        lift_target = self._interpolate_pose(
            self._compute_close_pose(), self._compute_lift_pose(), self.cfg.lift_motion_steps
        )
        transfer_target = self._interpolate_pose(
            self._compute_lift_pose(), self._compute_transfer_pose(), self.cfg.transfer_motion_steps
        )
        release_approach = self._compute_release_approach_pose()
        release_align_target = self._interpolate_pose(
            self._compute_transfer_pose(), release_approach, self.cfg.release_align_steps
        )
        release_descent_target = self._interpolate_pose_progress(
            release_approach,
            self._compute_corrected_release_pose(),
            self.release_descent_step,
            self.cfg.release_motion_steps,
        )
        release_target = torch.where(
            self.release_descent_started.unsqueeze(-1), release_descent_target, release_align_target
        )
        open_target = self._interpolate_pose(
            self._compute_corrected_release_pose(), self._compute_release_clear_pose(), self.cfg.open_dwell_steps
        )
        retreat_target = self._interpolate_pose(
            self._compute_release_clear_pose(), self._compute_retreat_pose(), self.cfg.retreat_motion_steps
        )
        phase_targets = (
            (PHASE_PREGRASP, pregrasp_target),
            (PHASE_GRASP, self._compute_grasp_pose()),
            (PHASE_CLOSE, self._compute_close_pose()),
            (PHASE_LIFT, lift_target),
            (PHASE_TRANSFER, transfer_target),
            (PHASE_RL_RELEASE, release_target),
            (PHASE_OPEN, open_target),
            (PHASE_SETTLE, retreat_target),
        )

        for phase_id, pose in phase_targets:
            mask = self.phase == phase_id
            if torch.any(mask):
                target_pose[mask] = pose[mask]

        return target_pose

    def _get_observations(self):
        """Get the observations for the environment."""
        tube_root_pos_w = self.tube.data.root_pos_w
        vortexer_center_xy_w = self._vortexer_center_pos_w()[:, :2]
        vortexer_rim_z_w = self.vortexer.data.root_pos_w[:, 2] + self.cfg.vortexer_top_from_root_m

        tube_up_axis_w = quat_apply(
            self.tube.data.root_pose_w[:, 3:7],
            self._world_up(tube_root_pos_w.dtype),
        )

        obs = torch.cat(
            (
                tube_root_pos_w[:, :2] - vortexer_center_xy_w,
                (tube_root_pos_w[:, 2:3] - vortexer_rim_z_w.unsqueeze(-1)),
                tube_up_axis_w[:, 2:3],
                self.tube.data.root_lin_vel_w,
                self.tube.data.root_ang_vel_w,
                self.previous_actions,
                self.phase.float().unsqueeze(-1) / float(PHASE_SETTLE),
                self._bilateral_tube_contact().float().unsqueeze(-1),
                self._tube_held().float().unsqueeze(-1),
                self._tube_vortexer_contact_force().clamp(max=1.0).unsqueeze(-1),
                self.release_descent_started.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Reward contact-valid placement and reject stable off-target drops."""
        tube_root_pos_w = self.tube.data.root_pos_w
        vortexer_center_xy_w = self._vortexer_center_pos_w()[:, :2]
        vortexer_rim_z_w = self.vortexer.data.root_pos_w[:, 2] + self.cfg.vortexer_top_from_root_m
        release_or_later = self.phase >= PHASE_RL_RELEASE

        xy_error = torch.linalg.norm(tube_root_pos_w[:, :2] - vortexer_center_xy_w, dim=-1)
        xy_reward = torch.exp(-torch.square(xy_error / self.cfg.rew_xy_sigma_m))
        insertion_xy_gate = torch.exp(
            -torch.square(xy_error / self.cfg.rew_insertion_xy_sigma_m)
        )

        insertion_depth = torch.clamp(vortexer_rim_z_w - tube_root_pos_w[:, 2], min=0.0)
        depth_reward = torch.clamp(
            insertion_depth / self.cfg.vortexer_min_insertion_below_rim_m,
            min=0.0,
            max=1.0,
        )
        inserted_reward = depth_reward * insertion_xy_gate

        tube_up_axis_w = quat_apply(
            self.tube.data.root_pose_w[:, 3:7],
            self._world_up(tube_root_pos_w.dtype),
        )
        upright_reward = torch.clamp(tube_up_axis_w[:, 2], min=0.0, max=1.0) * insertion_xy_gate

        lin_speed = torch.linalg.norm(self.tube.data.root_lin_vel_w, dim=-1)
        ang_speed = torch.linalg.norm(self.tube.data.root_ang_vel_w[:, :2], dim=-1)
        supported = self._tube_supported_by_vortexer()
        low_velocity_reward = (
            torch.exp(-lin_speed / self.cfg.rew_lin_vel_sigma_mps)
            * torch.exp(-ang_speed / self.cfg.rew_ang_vel_sigma_radps)
            * supported.float()
            * insertion_xy_gate
        )

        physical_success = self._tube_inserted() & supported & self._tube_stable()
        grasp_lost = (
            (self.phase == PHASE_RL_RELEASE)
            & ~self._tube_held()
            & ~supported
        )
        action_l2 = torch.sum(self.actions.square(), dim=-1)
        action_rate_l2 = torch.sum(
            torch.square(self.actions - self.previous_actions), dim=-1
        )

        dense_reward = release_or_later.float() * (
            self.cfg.rew_xy_tracking * xy_reward
            + self.cfg.rew_inserted * inserted_reward
            + self.cfg.rew_upright * upright_reward
            + self.cfg.rew_low_velocity * low_velocity_reward
            + self.cfg.rew_action_l2 * action_l2
            + self.cfg.rew_action_rate_l2 * action_rate_l2
        )
        control_dt = self.cfg.sim.dt * self.cfg.decimation
        reward = dense_reward * control_dt
        new_success = physical_success & (self.success_step == 0)
        new_grasp_loss = grasp_lost & (self.grasp_loss_step == 0)
        reward += self.cfg.rew_success * new_success.float()
        reward += self.cfg.rew_grasp_lost * new_grasp_loss.float()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_phase()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        tube_height_w = self.tube.data.root_pos_w[:, 2]
        phase_failed = self._phase_timed_out() & (self.phase < PHASE_SETTLE)
        supported = self._tube_supported_by_vortexer()
        grasp_loss_condition = (
            (self.phase == PHASE_RL_RELEASE)
            & ~self._tube_held()
            & ~supported
        )
        self.grasp_loss_step = torch.where(
            grasp_loss_condition,
            self.grasp_loss_step + 1,
            torch.zeros_like(self.grasp_loss_step),
        )
        physical_success = self._tube_inserted() & supported & self._tube_stable()
        self.success_step = torch.where(
            physical_success,
            self.success_step + 1,
            torch.zeros_like(self.success_step),
        )

        grasp_failed = self.grasp_loss_step >= self.cfg.grasp_loss_termination_steps
        success = self.success_step >= self.cfg.success_dwell_steps
        fallen = tube_height_w < self.cfg.fallen_height_threshold
        terminated = fallen | phase_failed | grasp_failed | success

        xy_error = torch.linalg.norm(
            self.tube.data.root_pos_w[:, :2] - self._vortexer_center_pos_w()[:, :2], dim=-1
        )
        self.extras["log"] = {
            "Task/success": success.float().mean(),
            "Task/physical_success": physical_success.float().mean(),
            "Task/grasp_failed": grasp_failed.float().mean(),
            "Task/xy_error_m": xy_error.mean(),
            "Task/support": supported.float().mean(),
            "Task/inserted": self._tube_inserted().float().mean(),
        }
        return terminated, time_out

    def _reset_idx(self, env_ids): 
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.ik_controller.reset(env_ids)
        
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        robot_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)

        robot_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        robot_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)

        # Reset fixed scene objects to their default poses in each environment.
        for asset in (self.table, self.holder, self.vortexer, self.vortexer_body):
            root_state = asset.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            asset.write_root_state_to_sim(root_state, env_ids=env_ids)

        tube_root_state = self.tube.data.default_root_state[env_ids].clone()
        tube_root_state[:, :3] = torch.as_tensor(
            self.cfg.tube_reset_root_pos_m,
            device=self.device,
            dtype=tube_root_state.dtype,
        )
        tube_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.tube.write_root_state_to_sim(tube_root_state, env_ids=env_ids)

        # Clear action buffers and phase state.
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.release_pose_bias[env_ids] = 0.0
        if self.cfg.enable_reset_randomization:
            bias_limit = torch.as_tensor(
                self.cfg.release_bias_randomization_m,
                device=self.device,
                dtype=self.release_pose_bias.dtype,
            )
            self.release_pose_bias[env_ids] = (
                2.0 * torch.rand((len(env_ids), 3), device=self.device) - 1.0
            ) * bias_limit
        self.phase[env_ids] = PHASE_PREGRASP
        self.phase_step[env_ids] = 0
        self.bilateral_contact_step[env_ids] = 0
        self.grasp_contact_latched[env_ids] = False
        self.gripper_hold_target[env_ids] = self.cfg.gripper_closed_position
        self.release_alignment_step[env_ids] = 0
        self.release_descent_step[env_ids] = 0
        self.release_descent_started[env_ids] = False
        self.grasp_loss_step[env_ids] = 0
        self.success_step[env_ids] = 0
        self.capture_pregrasp_start[env_ids] = True
        self.pregrasp_reached_step[env_ids] = 0
            
