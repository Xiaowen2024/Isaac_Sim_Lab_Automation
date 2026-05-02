from __future__ import annotations
import torch


def _pose_tensor(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([pos, quat], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    res = q.clone()
    res[..., 1:] = -res[..., 1:]
    return res


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = torch.cat([torch.zeros(v.shape[:-1] + (1,), device=v.device, dtype=v.dtype), v], dim=-1)
    return _quat_mul(_quat_mul(q, qvec), _quat_conjugate(q))[..., 1:]


class RootPoseAsset:
    """Small semantic wrapper around an Isaac Lab rigid object or articulation root pose."""

    def __init__(self, sim_object):
        self.sim_object = sim_object

    @property
    def root_pose_w(self) -> torch.Tensor:
        return self.sim_object.data.root_pose_w

    @property
    def pos_w(self) -> torch.Tensor:
        return self.root_pose_w[:, 0:3]

    @property
    def quat_w(self) -> torch.Tensor:
        return self.root_pose_w[:, 3:7]

    def _constant_quat(self, quat_wxyz: tuple[float, float, float, float]) -> torch.Tensor:
        quat = torch.tensor(quat_wxyz, device=self.root_pose_w.device, dtype=self.root_pose_w.dtype)
        return quat.repeat(self.root_pose_w.shape[0], 1)

    def _pose_with_vertical_offset(
        self,
        z_offset_m: float,
        quat_wxyz: tuple[float, float, float, float],
    ) -> torch.Tensor:
        pos = self.pos_w.clone()
        pos[:, 2] += z_offset_m
        return _pose_tensor(pos, self._constant_quat(quat_wxyz))

    def _broadcast_local_point(self, local_xyz_m: tuple[float, float, float]) -> torch.Tensor:
        point = torch.tensor(local_xyz_m, device=self.root_pose_w.device, dtype=self.root_pose_w.dtype)
        return point.repeat(self.root_pose_w.shape[0], 1)

    def _pose_from_local_point(
        self,
        local_xyz_m: tuple[float, float, float],
        local_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        vertical_offset_m: float = 0.0,
    ) -> torch.Tensor:
        local_point = self._broadcast_local_point(local_xyz_m)
        local_point[:, 2] += vertical_offset_m
        world_pos = self.pos_w + _quat_apply(self.quat_w, local_point)
        world_quat = _quat_mul(self.quat_w, self._constant_quat(local_quat_wxyz))
        return _pose_tensor(world_pos, world_quat)


class TubeAsset(RootPoseAsset):
    """Semantic helper around the simulated tube rigid object."""

    def __init__(
        self,
        rigid_object,
        *,
        local_grasp_point_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_support_point_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pregrasp_height_offset_m: float = 0.10,
        default_grasp_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        super().__init__(rigid_object)
        self.local_grasp_point_m = local_grasp_point_m
        self.local_support_point_m = local_support_point_m
        self.pregrasp_height_offset_m = pregrasp_height_offset_m
        self.default_grasp_quat = default_grasp_quat

    def get_grasp_pose_w(self) -> torch.Tensor:
        return self._pose_from_local_point(self.local_grasp_point_m, self.default_grasp_quat)

    def get_pregrasp_pose_w(self) -> torch.Tensor:
        return self._pose_from_local_point(
            self.local_grasp_point_m,
            self.default_grasp_quat,
            vertical_offset_m=self.pregrasp_height_offset_m,
        )

    def get_lift_pose_w(self, lift_height_m: float) -> torch.Tensor:
        return self._pose_from_local_point(
            self.local_grasp_point_m,
            self.default_grasp_quat,
            vertical_offset_m=lift_height_m,
        )

    def get_carried_grasp_pose_w(self, support_pose_w: torch.Tensor) -> torch.Tensor:
        local_offset = torch.tensor(
            (
                self.local_grasp_point_m[0] - self.local_support_point_m[0],
                self.local_grasp_point_m[1] - self.local_support_point_m[1],
                self.local_grasp_point_m[2] - self.local_support_point_m[2],
            ),
            device=support_pose_w.device,
            dtype=support_pose_w.dtype,
        ).repeat(support_pose_w.shape[0], 1)
        world_pos = support_pose_w[:, 0:3] + _quat_apply(support_pose_w[:, 3:7], local_offset)
        return _pose_tensor(world_pos, support_pose_w[:, 3:7].clone())

    def is_dropped(self, min_z: float) -> torch.Tensor:
        return self.pos_w[:, 2] < min_z


class SupportAsset(RootPoseAsset):
    """Semantic helper for assets that support the tube on a top surface."""

    def __init__(
        self,
        rigid_object,
        *,
        local_support_point_m: tuple[float, float, float],
        presupport_height_offset_m: float,
        default_support_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        super().__init__(rigid_object)
        self.local_support_point_m = local_support_point_m
        self.presupport_height_offset_m = presupport_height_offset_m
        self.default_support_quat = default_support_quat

    def get_support_pose_w(self) -> torch.Tensor:
        return self._pose_from_local_point(self.local_support_point_m, self.default_support_quat)

    def get_presupport_pose_w(self) -> torch.Tensor:
        return self._pose_from_local_point(
            self.local_support_point_m,
            self.default_support_quat,
            vertical_offset_m=self.presupport_height_offset_m,
        )

    def get_retreat_pose_w(self, retreat_height_m: float) -> torch.Tensor:
        return self._pose_from_local_point(
            self.local_support_point_m,
            self.default_support_quat,
            vertical_offset_m=retreat_height_m,
        )


class HolderAsset(SupportAsset):
    """Semantic helper for placeholder place / retreat targets on the holder."""

    def __init__(
        self,
        rigid_object,
        *,
        slot_centers_local_m: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
        default_slot_index: int = 0,
        preplace_height_offset_m: float = 0.12,
        default_place_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        self.slot_centers_local_m = slot_centers_local_m
        self.default_slot_index = default_slot_index
        super().__init__(
            rigid_object,
            local_support_point_m=slot_centers_local_m[default_slot_index],
            presupport_height_offset_m=preplace_height_offset_m,
            default_support_quat=default_place_quat,
        )

    def get_place_pose_w(self, slot_index: int | torch.Tensor | None = None) -> torch.Tensor:
        if slot_index is None:
            return self.get_support_pose_w()
        if isinstance(slot_index, torch.Tensor):
            local_points = torch.tensor(
                [self.slot_centers_local_m[int(idx)] for idx in slot_index.tolist()],
                device=self.root_pose_w.device,
                dtype=self.root_pose_w.dtype,
            )
            world_pos = self.pos_w + _quat_apply(self.quat_w, local_points)
            world_quat = _quat_mul(self.quat_w, self._constant_quat(self.default_support_quat))
            return _pose_tensor(world_pos, world_quat)
        return self._pose_from_local_point(self.slot_centers_local_m[slot_index], self.default_support_quat)

    def get_preplace_pose_w(self, slot_index: int | torch.Tensor | None = None) -> torch.Tensor:
        if slot_index is None:
            return self.get_presupport_pose_w()
        if isinstance(slot_index, torch.Tensor):
            local_points = torch.tensor(
                [self.slot_centers_local_m[int(idx)] for idx in slot_index.tolist()],
                device=self.root_pose_w.device,
                dtype=self.root_pose_w.dtype,
            )
            local_points[:, 2] += self.presupport_height_offset_m
            world_pos = self.pos_w + _quat_apply(self.quat_w, local_points)
            world_quat = _quat_mul(self.quat_w, self._constant_quat(self.default_support_quat))
            return _pose_tensor(world_pos, world_quat)
        return self._pose_from_local_point(
            self.slot_centers_local_m[slot_index],
            self.default_support_quat,
            vertical_offset_m=self.presupport_height_offset_m,
        )


class VortexerAsset(SupportAsset):
    """Semantic helper for tasks that place the tube onto the vortexer support."""

    def __init__(
        self,
        rigid_object,
        *,
        support_center_local_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        presupport_height_offset_m: float = 0.12,
        default_support_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        super().__init__(
            rigid_object,
            local_support_point_m=support_center_local_m,
            presupport_height_offset_m=presupport_height_offset_m,
            default_support_quat=default_support_quat,
        )

    def get_tube_support_pose_w(self) -> torch.Tensor:
        return self.get_support_pose_w()

    def get_tube_presupport_pose_w(self) -> torch.Tensor:
        return self.get_presupport_pose_w()


class RobotAsset(RootPoseAsset):
    """Semantic helper for robot root and end-effector poses."""

    def __init__(
        self,
        articulation,
        *,
        ee_body_id: int,
        ee_local_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__(articulation)
        self.articulation = articulation
        self.ee_body_id = ee_body_id
        self.ee_local_offset_m = ee_local_offset_m

    @property
    def ee_body_pose_w(self) -> torch.Tensor:
        return self.articulation.data.body_pose_w[:, self.ee_body_id]

    @property
    def ee_pose_w(self) -> torch.Tensor:
        body_pose_w = self.ee_body_pose_w
        if self.ee_local_offset_m == (0.0, 0.0, 0.0):
            return body_pose_w
        local_offset = torch.tensor(
            self.ee_local_offset_m,
            device=body_pose_w.device,
            dtype=body_pose_w.dtype,
        ).repeat(body_pose_w.shape[0], 1)
        world_pos = body_pose_w[:, 0:3] + _quat_apply(body_pose_w[:, 3:7], local_offset)
        return _pose_tensor(world_pos, body_pose_w[:, 3:7])

    @property
    def ee_pos_w(self) -> torch.Tensor:
        return self.ee_pose_w[:, 0:3]

    @property
    def ee_quat_w(self) -> torch.Tensor:
        return self.ee_pose_w[:, 3:7]

    def get_vertical_approach_pose_w(self, target_pose_w: torch.Tensor, z_offset_m: float) -> torch.Tensor:
        pos = target_pose_w[:, 0:3].clone()
        pos[:, 2] += z_offset_m
        return _pose_tensor(pos, target_pose_w[:, 3:7])
