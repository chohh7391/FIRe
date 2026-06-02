from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

from fire_core.residual_rl.action import combine_vla_and_residual, make_robot_action
from fire_core.residual_rl.actor import AsyncVLAProvider, ResidualVLAProvider, ZeroVLAProvider
from fire_core.residual_rl.features import (
    action_to_numpy,
    make_residual_action_features,
    make_residual_observation_features,
    make_state_batch,
    pad_action,
    task_arm_action_dim,
)
from fire_core.utils import total_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", default="forge-peg_insert")
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to a SAC pretrained_model directory, checkpoint dir, or training output dir.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--test", action="store_true", help="Use zero VLA actions without a VLA server.")
    parser.add_argument("--use_cameras", action="store_true")
    parser.add_argument("--use_ft_sensor", action="store_true")
    parser.add_argument("--use_sim_time", action="store_true")
    parser.add_argument("--vla", choices=["gr00t", "pi05"], default=None)
    parser.add_argument("--vla_chunk_size", type=int, default=16)
    parser.add_argument("--host", choices=["localhost", "163.180.160.225"], default=None)
    parser.add_argument("--port", type=int, choices=[5555, 5556, 5557, 5558, 5559], default=None)
    parser.add_argument("--teleop_device", choices=["keyboard"], default=None)
    return parser.parse_args()


def resolve_policy_dir(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "config.json").is_file():
        return root
    if (root / PRETRAINED_MODEL_DIR / "config.json").is_file():
        return root / PRETRAINED_MODEL_DIR
    last = root / "checkpoints" / "last" / PRETRAINED_MODEL_DIR
    if (last / "config.json").is_file():
        return last
    candidates = sorted(root.glob(f"checkpoints/*/{PRETRAINED_MODEL_DIR}/config.json"))
    if candidates:
        return candidates[-1].parent
    raise FileNotFoundError(f"Could not resolve SAC pretrained_model directory from {root}")


def build_vla_provider(args: argparse.Namespace) -> ResidualVLAProvider:
    if args.test:
        return ZeroVLAProvider()
    if args.vla is None:
        return ResidualVLAProvider()
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
    from lerobot_robot_fr3.fr3 import FR3Robot, TeleopAction
    from lerobot_teleoperator_keyboard import FIReKeyboardTeleop, FIReKeyboardTeleopConfig

    policy_dir = resolve_policy_dir(args.policy)
    policy = SACPolicy.from_pretrained(policy_dir).to(args.device).eval()

    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time,
        is_relative=False,
        rotation_type="quaternion",
        use_cameras=args.use_cameras,
        use_ft_sensor=args.use_ft_sensor,
    )
    robot = FR3Robot(config, task_name=args.task)
    obs_features = make_residual_observation_features(
        robot.task.observation_features,
        robot.task.action_features,
    )
    action_dim = total_dim(make_residual_action_features(robot.task.action_features))
    robot_arm_dim = task_arm_action_dim(robot.task)
    vla_provider = build_vla_provider(args)

    teleop: Optional[FIReKeyboardTeleop] = None
    if args.teleop_device == "keyboard":
        teleop = FIReKeyboardTeleop(
            FIReKeyboardTeleopConfig(
                action_dim=action_dim,
            )
        )

    print(f"[INFO] Loading residual SAC policy from {policy_dir}")
    print("[INFO] Connecting to robot ...")
    robot.connect()
    if teleop is not None:
        teleop.connect()
    vla_provider.reset(robot, action_dim)

    dt = 1.0 / args.control_hz
    try:
        for _ in range(args.episode_length):
            t0 = time.perf_counter()
            obs = robot.get_observation()
            batch = make_state_batch(obs, obs_features, device=args.device)
            with torch.no_grad():
                residual_tensor = policy.select_action(batch=batch)
            residual_action = action_to_numpy(residual_tensor, action_dim) * float(args.residual_scale)
            vla_action = vla_provider.get_action(robot, action_dim)

            is_intervention = False
            if teleop is not None:
                teleop_action, is_intervention = teleop.get_intervention_action()
                if is_intervention:
                    residual_action = teleop_action
                    print(
                        f"[TELEOP] key={teleop.last_key} residual_action={teleop_action}",
                        flush=True,
                    )

            final_action = combine_vla_and_residual(
                vla_action=vla_action,
                residual_action=residual_action,
                action_dim=action_dim,
            )
            robot_action = make_robot_action(pad_action(final_action, robot_arm_dim))
            if is_intervention:
                robot.send_teleop_action(
                    TeleopAction(
                        action=robot_action,
                        action_space="task_space",
                        is_relative=True,
                    )
                )
            else:
                robot.send_action(robot_action)

            sleep_t = dt - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        if teleop is not None:
            teleop.disconnect()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    main()
