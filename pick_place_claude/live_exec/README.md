# Live MoveIt-to-Isaac Execution

This folder contains the new live-execution path for the wet-lab pick/place benchmark.

The legacy scripts remain unchanged:
- `wetlab_benchmark/pick_place/pick_place_move_it.py`
- `wetlab_benchmark/pick_place/pick_place_moveit_render.py`

The live path changes the ownership model:
- `moveit_live_runner.py` plans with MoveIt and sends trajectories/gripper commands over ROS 2.
- `isaac_live_executor.py` owns the robot execution, tube physics, attach/detach, and rendering inside Isaac Sim.
- Physical training mode is the default: the tube is carried only if contact and friction actually hold it.

## Files

- `task_builder.py`
  - Builds the seeded layout and task legs for modes `a` and `b`.
  - Uses decomposed open-top obstacle walls for the holder and vortexer, instead of one solid box.
- `protocol.py`
  - Shared topic names, camera presets, and JSON message helpers.
- `moveit_live_runner.py`
  - MoveIt-side planner and arm/gripper client.
  - Uses `AttachedCollisionObject` for the carried tube.
- `isaac_live_executor.py`
  - Isaac-side live execution backend.
  - Publishes `/joint_states` and `/clock`.
  - Serves `FollowJointTrajectory`.
  - Validates bilateral gripper/tube contact, verifies lift, and records frames during live sim.
  - Keeps `--grasp_mode fixed_joint` only as a debug fallback for the old snap-attach behavior.

## ROS Interfaces

Standard ROS interfaces:
- `/joint_states`
- `/clock`
- `/xarm6_traj_controller/follow_joint_trajectory`

Benchmark-private JSON channels:
- `/wetlab_benchmark/live_exec/status`
- `/wetlab_benchmark/live_exec/control`
- `/wetlab_benchmark/live_exec/gripper_cmd`

## Launch Order

Start Isaac first. Start MoveIt second.

### Terminal 1: Isaac Executor

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/wetlab_benchmark/moveit_py_ws/install/setup.bash
cd /home/ubuntu/wetlab_benchmark
/home/ubuntu/IsaacLab/isaaclab.sh -p /home/ubuntu/wetlab_benchmark/wetlab_benchmark/pick_place/live_exec/isaac_live_executor.py \
  --headless \
  --seed 0 \
  --task_mode b \
  --camera_preset side_wide \
  --output_dir /home/ubuntu/wetlab_benchmark/live_exec_task_b \
  --grasp_mode physical \
  --arm_drive_mode target \
  --no-seat_release_pose
```

### Terminal 2: MoveIt Runner

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/wetlab_benchmark/xarm_runtime_ws/install/setup.bash
source /home/ubuntu/wetlab_benchmark/moveit_py_ws/install/setup.bash
export AMENT_PREFIX_PATH=/home/ubuntu/wetlab_benchmark/xarm_runtime_ws/install/xarm_moveit_config:$AMENT_PREFIX_PATH
export CMAKE_PREFIX_PATH=/home/ubuntu/wetlab_benchmark/xarm_runtime_ws/install/xarm_moveit_config:$CMAKE_PREFIX_PATH
python3 /home/ubuntu/wetlab_benchmark/wetlab_benchmark/pick_place/live_exec/moveit_live_runner.py \
  --seed 0 \
  --task_mode b \
  --grasp_mode physical
```

## Output

The executor writes:
- `frames/frame_XXXXXX.png`
- `render.mp4`
- `render_run.json`

## Failure Semantics

The run aborts if:
- Isaac never becomes ready.
- The arm action goal is rejected or returns a controller error.
- Bilateral contact is not detected during close.
- The tube does not rise with the gripper during lift verification.
- Tube settle validation fails after release.

The status topic publishes JSON with:
- `phase`
- `ok`
- `attached`
- `tube_pose`
- `leg_label`
- `reason` on failure

## Current Scope

Supported:
- single environment
- single xArm6 robot
- task modes `a` and `b`
- live rendering with named camera presets

Current limitations:
- no custom ROS message package; benchmark-specific coordination stays on JSON `std_msgs/String`
- the MoveIt runner assumes the xArm MoveIt config and joint names used by this repo
- imported gripper collision/material prims are still USD-dependent; if physical grasp fails with no contact, the next step is collider refinement rather than re-enabling snap attach
