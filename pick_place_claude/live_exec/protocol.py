from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


ARM_TRAJECTORY_ACTION = "/xarm6_traj_controller/follow_joint_trajectory"
GRIPPER_COMMAND_TOPIC = "/wetlab_benchmark/live_exec/gripper_cmd"
EXECUTOR_STATUS_TOPIC = "/wetlab_benchmark/live_exec/status"
EXECUTOR_CONTROL_TOPIC = "/wetlab_benchmark/live_exec/control"
JOINT_STATE_TOPIC = "/joint_states"
CLOCK_TOPIC = "/clock"

PHASE_READY = "ready"
PHASE_ARM_EXECUTING = "arm_executing"
PHASE_ARM_DONE = "arm_done"
PHASE_GRIPPER_OPENING = "gripper_opening"
PHASE_GRIPPER_OPEN = "gripper_open"
PHASE_GRIPPER_CLOSING = "gripper_closing"
PHASE_GRASP_ATTACHED = "grasp_attached"
PHASE_GRASP_SECURED = "grasp_secured"
PHASE_GRASP_FAILED = "grasp_failed"
PHASE_LIFT_VERIFIED = "lift_verified"
PHASE_GRASP_LOST = "grasp_lost"
PHASE_RELEASED = "released"
PHASE_SETTLED = "settled"
PHASE_ERROR = "error"


@dataclass(frozen=True)
class CameraPreset:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]


CAMERA_PRESETS: dict[str, CameraPreset] = {
    "front_default": CameraPreset(eye=(0.28, 0.96, 1.35), target=(0.28, 0.0, 0.82)),
    "side_wide": CameraPreset(eye=(1.38, 0.24, 1.08), target=(0.35, 0.0, 0.58)),
    "top_oblique": CameraPreset(eye=(0.90, 0.72, 1.72), target=(0.36, -0.02, 0.58)),
}


@dataclass
class ExecutorStatus:
    phase: str
    ok: bool
    attached: bool
    tube_pose: dict[str, float] | None = None
    reason: str | None = None
    leg_label: str | None = None
    stamp: float = field(default_factory=time.time)
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase": self.phase,
            "ok": bool(self.ok),
            "attached": bool(self.attached),
            "stamp": float(self.stamp),
        }
        if self.tube_pose is not None:
            payload["tube_pose"] = dict(self.tube_pose)
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.leg_label is not None:
            payload["leg_label"] = self.leg_label
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "ExecutorStatus":
        payload = json.loads(data)
        return cls(
            phase=str(payload["phase"]),
            ok=bool(payload["ok"]),
            attached=bool(payload["attached"]),
            tube_pose=payload.get("tube_pose"),
            reason=payload.get("reason"),
            leg_label=payload.get("leg_label"),
            stamp=float(payload.get("stamp", time.time())),
            extra=payload.get("extra"),
        )


@dataclass
class ExecutorControl:
    stage: str
    leg_label: str | None = None
    stamp: float = field(default_factory=time.time)
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "stamp": float(self.stamp),
        }
        if self.leg_label is not None:
            payload["leg_label"] = self.leg_label
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "ExecutorControl":
        payload = json.loads(data)
        return cls(
            stage=str(payload["stage"]),
            leg_label=payload.get("leg_label"),
            stamp=float(payload.get("stamp", time.time())),
            extra=payload.get("extra"),
        )


def camera_from_preset(
    *,
    preset_name: str,
    eye_override: tuple[float, float, float] | None,
    target_override: tuple[float, float, float] | None,
) -> CameraPreset:
    preset = CAMERA_PRESETS[preset_name]
    return CameraPreset(
        eye=tuple(eye_override) if eye_override is not None else preset.eye,
        target=tuple(target_override) if target_override is not None else preset.target,
    )
