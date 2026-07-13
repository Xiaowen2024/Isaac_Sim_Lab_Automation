# Contact-Valid Place RL

This repository now tracks a cleaner Isaac Lab version of the wet-lab tube placement project. It is a simplified follow-up to the first `Isaac_Sim_Lab_Automation` attempt: instead of carrying over the large live-execution, ROS/MoveIt, replay, and calibration script stack, the new direction keeps only the assets and task ideas needed for a focused Isaac Lab training demo.

The current clean workspace is:

```text
/Users/bytedance/Desktop/hw2/contact_valid_place_rl
```

## Motivation

The first automation branch showed the right task direction, but it became difficult to debug because motion execution, asset conversion, contact refinement, gripper control, trace replay, and placement logic were all mixed together. The new version separates the problem into:

- contact-valid assets
- scripted phase-based motion
- RL only for the final release refinement
- a standard Isaac Lab direct-RL package layout

This makes the project easier to explain, test, and extend.

## Active Assets

The clean scene uses four assets under `contact_valid_place_rl/assets/`:

- `xarm6_with_gripper_contact_refined.usd`
- `4_50ml_conical_holder_contact_refined.usd`
- `vortexer_contact_refined.usd`
- `autobio_50ml_tube_contact_refined.usda`

The xArm, holder, and vortexer come from the contact-refined assets generated during the first automation attempt. The tube is replaced with a simplified AutoBio-inspired 50ml tube asset with explicit body, sleeve, and cap/collar collision primitives. The old refined tube USD is intentionally not used.

## Phase-Based Planning

The planned task is not full end-to-end RL. Instead, the robot behavior is decomposed into phases:

```text
PREGRASP -> GRASP -> CLOSE -> LIFT -> TRANSFER -> RL_RELEASE -> OPEN -> SETTLE
```

The scripted phases use IK or fixed rules:

- `PREGRASP`: move the gripper above the tube
- `GRASP`: move to the tube cap/grasp height
- `CLOSE`: close the gripper with a fixed command
- `LIFT`: lift the tube from the holder
- `TRANSFER`: move above the vortexer
- `OPEN`: open the gripper with a fixed command
- `SETTLE`: wait for the tube to settle

RL is used only during `RL_RELEASE`, where the policy predicts a small residual offset around the nominal release pose:

```text
action = [dx, dy, dz, dyaw]
```

This keeps large-scale motion simple and lets RL focus on the contact-sensitive part of the task: where exactly to release the tube so it settles into the vortexer.

## Isaac Lab Training

The clean workspace is structured as an Isaac Lab extension:

```text
contact_valid_place_rl/
├── assets/
├── scripts/
└── source/contact_valid_place_rl/
```

The direct-RL task scaffold lives at:

```text
source/contact_valid_place_rl/contact_valid_place_rl/tasks/direct/tube_place/
```

Training is intended to use Isaac Lab's direct workflow with RSL-RL PPO:

- environment base: `DirectRLEnv`
- runner: RSL-RL on-policy runner
- algorithm: PPO
- action: release residual `[dx, dy, dz, dyaw]`
- observation: tube-vortexer relative state, tube velocities, upright signal, previous action
- reward: centered placement, insertion below rim, uprightness, low velocity, and action regularization

The training setup is still being scaffolded, but the package structure and task config are organized to match standard Isaac Lab projects.

## Scene Checks

The canonical scene smoke-test script is:

```bash
cd /Users/bytedance/Desktop/hw2/contact_valid_place_rl
/path/to/IsaacLab/isaaclab.sh -p scripts/run_scene.py
```

If Isaac Lab or a GPU is unavailable, asset placement can still be inspected offline:

```bash
python scripts/inspect_scene_assets.py
```

The offline inspection script opens the USD files with `pxr`, reports default prims, approximate bounds, planned root poses, scales, and each asset's bottom height relative to the table top.

## Starting The Task

There are two ways to start the project, depending on whether the goal is scene validation or RL training.

### Without Training

Use this path when checking that assets load, scale correctly, and sit on the table. This does not start PPO and does not require the Gym task registration to be complete.

```bash
cd /Users/bytedance/Desktop/hw2/contact_valid_place_rl
/path/to/IsaacLab/isaaclab.sh -p scripts/run_scene.py
```

For a headless smoke test:

```bash
cd /Users/bytedance/Desktop/hw2/contact_valid_place_rl
/path/to/IsaacLab/isaaclab.sh -p scripts/run_scene.py --headless --sim_steps 300
```

If Isaac Lab is not installed locally, the offline asset check can still run with a Python environment that has `pxr`:

```bash
cd /Users/bytedance/Desktop/hw2/contact_valid_place_rl
python scripts/inspect_scene_assets.py
```

### With Training

Training uses the Isaac Lab direct RL workflow with RSL-RL PPO. First install the extension into the Isaac Lab Python environment:

```bash
cd /Users/bytedance/Desktop/hw2/contact_valid_place_rl/source/contact_valid_place_rl
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e .
```

Then launch RSL-RL training:

```bash
cd /Users/bytedance/Desktop/hw2/contact_valid_place_rl
/path/to/IsaacLab/isaaclab.sh -p /path/to/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task ContactValid-TubePlace-Direct-v0 \
  --headless
```

For a shorter debug run:

```bash
cd /Users/bytedance/Desktop/hw2/contact_valid_place_rl
/path/to/IsaacLab/isaaclab.sh -p /path/to/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task ContactValid-TubePlace-Direct-v0 \
  --headless \
  --num_envs 16 \
  --max_iterations 100
```

Training will only work after the direct environment, PPO config, and Gym registration are fully implemented under:

```text
source/contact_valid_place_rl/contact_valid_place_rl/tasks/direct/tube_place/
```
