"""Batch evaluation harness for the wet-lab pick-place unit test."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run many randomized wet-lab trials.")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--script", type=str, default="wetlab_benchmark/pick_place_claude_v3/pick_place_sm.py")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_mode", type=str.lower, default="sample", choices=["sample", "a", "b"])
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    passed = 0
    for seed in range(args.num_trials):
        cmd = [
            "./isaaclab.sh",
            "-p",
            str(script_path),
            "--num_envs",
            str(args.num_envs),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--task_mode",
            args.task_mode,
        ]
        if args.headless:
            cmd.append("--headless")
        result = subprocess.run(cmd, capture_output=True, text=True)
        ok = result.returncode == 0 and "[pick_place_sm] success=" in result.stdout and "failed=0" in result.stdout
        passed += int(ok)
        print(f"[trial {seed:03d}] pass={ok}")
        if not ok:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)

    print(f"[batch_eval] passed_trials={passed}/{args.num_trials}")
    print(f"[batch_eval] trial_pass_rate={passed / max(args.num_trials, 1):.3f}")


if __name__ == "__main__":
    main()
