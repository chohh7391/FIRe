from __future__ import annotations

import argparse
from typing import Optional

from fire_core.residual_rl.actor import (
    AsyncVLAProvider,
    ResidualVLAProvider,
    ZeroVLAProvider,
    run_residual_actor,
)
from fire_core.residual_rl.features import make_residual_action_features
from fire_core.utils import total_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run a FIRe FR3 residual RL actor.",
    )
    parser.add_argument("--task", default="forge-peg_insert")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output_dir", default="outputs/fire_residual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage_device", default="cpu")
    parser.add_argument("--learner_host", default="127.0.0.1")
    parser.add_argument("--learner_port", type=int, default=50051)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--online_steps", type=int, default=1000000)
    parser.add_argument("--online_buffer_capacity", type=int, default=100000)
    parser.add_argument("--online_step_before_learning", type=int, default=100)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--test", action="store_true", help="Use zero VLA actions without a VLA server.")
    parser.add_argument("--teleop_device", choices=["keyboard"], default=None)
    parser.add_argument("--use_cameras", action="store_true")
    parser.add_argument("--use_ft_sensor", action="store_true")
    parser.add_argument("--use_sim_time", action="store_true")
    parser.add_argument("--vla", choices=["zero", "gr00t", "pi05"], default=None)
    parser.add_argument("--vla_chunk_size", type=int, default=16)
    parser.add_argument("--host", choices=["localhost", "163.180.160.225"], default=None)
    parser.add_argument("--port", type=int, choices=[5555, 5556, 5557, 5558, 5559], default=None)
    return parser.parse_args()


def build_vla_provider(args: argparse.Namespace) -> Optional[ResidualVLAProvider]:
    if args.test:
        return ZeroVLAProvider()
    if args.vla is None:
        return None
    if args.vla == "zero":
        return ZeroVLAProvider()
    if args.host is None or args.port is None:
        raise ValueError("--host and --port are required when --vla is set.")

    from fire_core.vla_clients import AsyncGr00tInferenceClient, AsyncPi05InferenceClient

    if args.vla == "gr00t":
        client = AsyncGr00tInferenceClient(host=args.host, port=args.port)
    else:
        host = "127.0.0.1" if args.host == "localhost" else args.host
        client = AsyncPi05InferenceClient(host=host, port=args.port)
    return AsyncVLAProvider(client, chunk_size=args.vla_chunk_size)


def main() -> None:
    args = parse_args()

    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot
    from lerobot_teleoperator_keyboard import FIReKeyboardTeleop, FIReKeyboardTeleopConfig

    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time,
        is_relative=False,
        rotation_type="quaternion",
        use_cameras=args.use_cameras,
        use_ft_sensor=args.use_ft_sensor,
    )
    robot = FR3Robot(config, task_name=args.task)
    teleop = None
    if args.teleop_device == "keyboard":
        teleop = FIReKeyboardTeleop(
            FIReKeyboardTeleopConfig(
                action_dim=total_dim(make_residual_action_features(robot.task.action_features)),
            ),
        )
    run_residual_actor(
        robot=robot,
        task=args.task,
        fps=args.fps,
        output_dir=args.output_dir,
        device=args.device,
        storage_device=args.storage_device,
        learner_host=args.learner_host,
        learner_port=args.learner_port,
        online_steps=args.online_steps,
        online_buffer_capacity=args.online_buffer_capacity,
        online_step_before_learning=args.online_step_before_learning,
        batch_size=args.batch_size,
        residual_scale=args.residual_scale,
        vla_provider=build_vla_provider(args),
        teleop=teleop,
    )


if __name__ == "__main__":
    main()
