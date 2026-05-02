from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from wetlab_benchmark.task_config import FRAMES
from wetlab_benchmark.task_objects import HolderAsset, TubeAsset, VortexerAsset


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


def _shift_pose_origin_local(pose_w: torch.Tensor, local_offset_m: tuple[float, float, float]) -> torch.Tensor:
    offset = torch.tensor(local_offset_m, device=pose_w.device, dtype=pose_w.dtype).repeat(pose_w.shape[0], 1)
    shifted = pose_w.clone()
    shifted[:, 0:3] = shifted[:, 0:3] + _quat_apply(shifted[:, 3:7], offset)
    return shifted


class Phase(IntEnum):
    APPROACH = 0
    PREGRASP = 1
    CLOSE_GRIPPER = 2
    LIFT = 3
    TRANSFER = 4
    PLACE = 5
    OPEN_GRIPPER = 6
    RETREAT = 7
    DONE = 8
    FAILED = 9


# Maps each phase to the goal dict key used as the IK target for that phase.
PHASE_GOAL_NAMES = {
    Phase.APPROACH:      "pregrasp",
    Phase.PREGRASP:      "grasp",
    Phase.CLOSE_GRIPPER: "grasp",    # stay at grasp while fingers close
    Phase.LIFT:          "lift",
    Phase.TRANSFER:      "preplace",
    Phase.PLACE:         "place",
    Phase.OPEN_GRIPPER:  "place",    # stay at place while fingers open
    Phase.RETREAT:       "retreat",
}


@dataclass(frozen=True)
class PickPlaceTaskPolicy:
    lift_height_m: float
    retreat_height_m: float

    def build_pick_goals_w(
        self,
        tube_asset: TubeAsset,
        holder_asset: HolderAsset,
        vortexer_asset: VortexerAsset,
        task_mode: torch.Tensor,
        task_step: torch.Tensor,
        holder_slot_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Use the intended support location as the pick anchor instead of the live
        # tube pose so the controller does not chase the object if it gets nudged
        # during approach.
        source_support = holder_asset.get_place_pose_w(holder_slot_index).clone()
        use_vortexer_source = (task_mode == 1) & (task_step == 1)
        if bool(torch.any(use_vortexer_source)):
            source_support[use_vortexer_source] = vortexer_asset.get_support_pose_w()[use_vortexer_source]

        grasp = tube_asset.get_carried_grasp_pose_w(source_support)
        pregrasp = grasp.clone()
        pregrasp[:, 2] += tube_asset.pregrasp_height_offset_m
        lift = grasp.clone()
        lift[:, 2] += self.lift_height_m
        return {
            "pregrasp": pregrasp,
            "grasp": grasp,
            "lift": lift,
        }

    def build_support_goals_w(
        self,
        holder_asset: HolderAsset,
        vortexer_asset: VortexerAsset,
        task_mode: torch.Tensor,
        task_step: torch.Tensor,
        holder_slot_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        preplace = holder_asset.get_preplace_pose_w(holder_slot_index).clone()
        place    = holder_asset.get_place_pose_w(holder_slot_index).clone()
        retreat  = holder_asset.get_retreat_pose_w(self.retreat_height_m).clone()

        use_vortexer = (task_mode == 1) & (task_step == 0)
        if bool(torch.any(use_vortexer)):
            preplace[use_vortexer] = vortexer_asset.get_presupport_pose_w()[use_vortexer]
            place[use_vortexer]    = vortexer_asset.get_support_pose_w()[use_vortexer]
            retreat[use_vortexer]  = vortexer_asset.get_retreat_pose_w(self.retreat_height_m)[use_vortexer]

        return {
            "preplace": preplace,
            "place":    place,
            "retreat":  retreat,
        }

    def build_goals_w(
        self,
        tube_asset: TubeAsset,
        holder_asset: HolderAsset,
        vortexer_asset: VortexerAsset,
        task_mode: torch.Tensor,
        task_step: torch.Tensor,
        holder_slot_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-env end-effector goals for every named waypoint."""
        pick_goals = self.build_pick_goals_w(
            tube_asset,
            holder_asset,
            vortexer_asset,
            task_mode,
            task_step,
            holder_slot_index,
        )
        pregrasp = pick_goals["pregrasp"]
        grasp = pick_goals["grasp"]
        lift = pick_goals["lift"]

        support_goals = self.build_support_goals_w(
            holder_asset,
            vortexer_asset,
            task_mode,
            task_step,
            holder_slot_index,
        )
        preplace = tube_asset.get_carried_grasp_pose_w(support_goals["preplace"])
        place    = tube_asset.get_carried_grasp_pose_w(support_goals["place"])
        retreat  = tube_asset.get_carried_grasp_pose_w(support_goals["retreat"])

        # Policy goals are authored for the tool/TCP point. The controller acts on
        # the gripper-base body, so shift every goal into that body frame using the
        # desired fixed top-grasp orientation, not the current wrist orientation.
        body_offset = tuple(-value for value in FRAMES.ee_tool_offset_local_m)
        pregrasp = _shift_pose_origin_local(pregrasp, body_offset)
        grasp = _shift_pose_origin_local(grasp, body_offset)
        lift = _shift_pose_origin_local(lift, body_offset)
        preplace = _shift_pose_origin_local(preplace, body_offset)
        place = _shift_pose_origin_local(place, body_offset)
        retreat = _shift_pose_origin_local(retreat, body_offset)

        return {
            "pregrasp": pregrasp,
            "grasp":    grasp,
            "lift":     lift,
            "preplace": preplace,
            "place":    place,
            "retreat":  retreat,
        }

    def target_pose_w(
        self,
        phase: torch.Tensor,
        *,
        tube_asset: TubeAsset,
        holder_asset: HolderAsset,
        vortexer_asset: VortexerAsset,
        task_mode: torch.Tensor,
        task_step: torch.Tensor,
        holder_slot_index: torch.Tensor | None = None,
        num_envs: int,
        device: str,
    ) -> torch.Tensor:
        goals_w = self.build_goals_w(
            tube_asset, holder_asset, vortexer_asset, task_mode, task_step, holder_slot_index
        )
        target_w = torch.zeros(num_envs, 7, device=device)
        for phase_enum, goal_name in PHASE_GOAL_NAMES.items():
            mask = phase == int(phase_enum)
            if bool(torch.any(mask)):
                target_w[mask] = goals_w[goal_name][mask]
        return target_w

    def gripper_closed_mask(self, phase: torch.Tensor) -> torch.Tensor:
        return (
            (phase == int(Phase.CLOSE_GRIPPER))
            | (phase == int(Phase.LIFT))
            | (phase == int(Phase.TRANSFER))
            | (phase == int(Phase.PLACE))
        )
