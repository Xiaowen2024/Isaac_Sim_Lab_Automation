# Wet Lab Benchmark Starter

This starter package is organized around one immediate goal:
make a deterministic, batched Isaac Lab unit test for a single wet-lab pick-place task before bringing ROS 2 and MoveIt into the control loop.

## Recommended build order

1. Freeze the assets and frames.
   - Convert `xarm6.urdf` to a clean USD once.
   - Ensure the vortexer, tube holder combo, and tube all have stable prim names and collision meshes.
   - Add one grasp frame on the tube and one place frame on each slot in the holder.

2. Solve the task inside Isaac Lab first.
   - Use a standalone script and Isaac Lab's `InteractiveScene`.
   - Use differential IK for the arm.
   - Use a simple gripper command plus attach/detach fallback if contact grasp is not stable yet.
   - Run batched random initializations and compute pass rate.

3. Turn it into a benchmark.
   - Define success metrics.
   - Log seed, initial poses, completion time, smoothness, and failure stage.
   - Require 100 percent success over a bounded randomized distribution before integrating planners.

4. Add ROS 2 + MoveIt only after the simulator-side task is stable.
   - ROS 2 / MoveIt should be used for planner comparison and system integration.
   - They should not be the first dependency in the unit test because they make debugging slower.

## Minimal project structure

`task_config.py`
Shared asset paths, frame names, and randomization bounds.

`scripts/scene_check.py`
Loads all assets, randomizes object poses, and validates the stage.

`scripts/pick_place_sm.py`
A standalone finite-state-machine controller for:
approach -> pregrasp -> grasp -> lift -> transfer -> place -> release -> retreat

`scripts/run_batch_eval.py`
Runs many seeds and reports pass rate and timing.

## Success criteria for the unit test

- `success`: tube ends inside the target holder slot and is released
- `no_drop`: tube never falls below the table threshold
- `collision_ok`: no forbidden collisions with holder walls or vortexer body
- `smoothness_ok`: joint velocity and acceleration stay under chosen limits
- `timeout_ok`: full sequence finishes within a fixed horizon

## Randomization scope for now

Randomize only what matters for the first benchmark:

- xArm6 base planar pose `(x, y, yaw)` within a reachable work envelope
- vortexer planar pose `(x, y, yaw)`
- tube-holder combo planar pose `(x, y, yaw)`
- selected holder slot index

Keep the following fixed initially:

- table height
- robot base z
- tube geometry
- grasp frame transform relative to the tube
- place frame transform relative to each slot

## Design choice

For the first version, use a scripted state machine, not MoveIt, for execution.
This gives you:

- a stronger lower-bound benchmark
- easier reproducibility under random seeds
- simpler debugging of collisions, frames, and grasp logic

After this is reliable, add:

1. `MoveIt plan + execute`
2. `MoveIt fallback on scripted recovery`
3. `Arena ranking` across policies and planners

## Running order

Assuming your Isaac Lab environment is active:

```bash
./isaaclab.sh -p wetlab_benchmark/scripts/scene_check.py --num_envs 8 --headless
./isaaclab.sh -p wetlab_benchmark/scripts/pick_place_sm.py --num_envs 8 --headless
./isaaclab.sh -p wetlab_benchmark/scripts/run_batch_eval.py --num_envs 32 --num_trials 200 --headless
```

## What you still need to fill in

- real asset file paths in `task_config.py`
- xArm end-effector body name
- arm joint name regex
- gripper joint names or attach/detach helper
- holder slot transforms
- exact tube grasp transform
- forbidden collision filters

