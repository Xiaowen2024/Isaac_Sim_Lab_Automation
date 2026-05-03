# Wet Lab Benchmark

This repository contains the Isaac Sim + MoveIt wet-lab pick/place benchmark, including the live execution path and the trace replay flow used to render the reference video:

- `remote_runs/live_exec_officialbase_leg0_vortex_lowdrop_try2/trace_render/render_trace_h264.mp4`

That reference replay corresponds to:

- `seed = 0`
- `task_mode = b` in the launch scripts, `task_mode = 1` in the saved metadata
- `holder_slot_index = 1`
- `camera_preset = side_wide`

## Canonical code path

The current benchmark implementation lives under:

- `wetlab_benchmark/pick_place/live_exec/isaac_live_executor.py`
- `wetlab_benchmark/pick_place/live_exec/moveit_live_runner.py`
- `wetlab_benchmark/pick_place/live_exec/render_state_trace.py`
- `wetlab_benchmark/pick_place/live_exec/task_builder.py`

The older `pick_place_claude*` directories are historical snapshots. Use the `pick_place/live_exec` path for the canonical live-exec workflow.

## Replaying the reference trace

The reference MP4 is produced by replaying the saved state trace from the run directory:

```bash
./isaaclab.sh -p wetlab_benchmark/pick_place/live_exec/render_state_trace.py \
  --input_dir wetlab_benchmark/remote_runs/live_exec_officialbase_leg0_vortex_lowdrop_try2 \
  --asset_profile contact_refined \
  --camera_preset side_wide
```

The replay script reads:

- `render_run.json`
- `state_trace.jsonl`

and writes the rendered frames and video to `trace_render/` by default.

## Live execution

The live executor and MoveIt runner are launched separately. The executor owns the Isaac Sim side of the task; the runner plans and sends trajectories over ROS 2.

Executor:

```bash
./isaaclab.sh -p wetlab_benchmark/pick_place/live_exec/isaac_live_executor.py \
  --headless \
  --seed 0 \
  --task_mode b \
  --camera_preset side_wide \
  --output_dir /home/ubuntu/wetlab_benchmark/live_exec_task_b \
  --grasp_mode physical \
  --arm_drive_mode target \
  --no-seat_release_pose
```

Runner:

```bash
python3 wetlab_benchmark/pick_place/live_exec/moveit_live_runner.py \
  --seed 0 \
  --task_mode b \
  --grasp_mode physical
```

## Output artifacts

Typical run outputs include:

- `render_run.json`
- `state_trace.jsonl`
- `trace_render/render_trace_h264.mp4`
- `trace_render/render_replay.json`
- `frames/frame_*.png` for live execution runs

## Notes

- `task_mode b` is the vortexer-first layout used by the reference trace.
- The replay script supports alternative camera presets and optional visual overrides for debugging.
- The repository already contains multiple implementation snapshots; the live-exec folder above is the one to use for the current benchmark state.
