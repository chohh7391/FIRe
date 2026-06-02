from __future__ import annotations

import argparse

from fire_core.residual_rl.features import (
    make_residual_action_features,
    make_residual_observation_features,
)
from fire_core.residual_rl.learner import run_residual_learner
from lerobot_robot_fr3 import create_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run a LeRobot SAC learner for FIRe residual RL.",
    )
    parser.add_argument("--task", default="forge-peg_insert")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output_dir", default="outputs/fire_residual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage_device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--online_steps", type=int, default=1000000)
    parser.add_argument("--online_buffer_capacity", type=int, default=100000)
    parser.add_argument("--online_step_before_learning", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = create_task(args.task)
    action_features = make_residual_action_features(task.action_features)
    run_residual_learner(
        obs_features=make_residual_observation_features(
            task.observation_features,
            task.action_features,
        ),
        action_features=action_features,
        task=args.task,
        fps=args.fps,
        output_dir=args.output_dir,
        device=args.device,
        storage_device=args.storage_device,
        batch_size=args.batch_size,
        online_steps=args.online_steps,
        online_buffer_capacity=args.online_buffer_capacity,
        online_step_before_learning=args.online_step_before_learning,
    )


if __name__ == "__main__":
    main()
