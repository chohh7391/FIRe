from __future__ import annotations

import argparse
import ctypes
import time
from multiprocessing import RawValue
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation

from fire_core.checkpoints import resolve_checkpoint_path
from fire_core.utils import total_dim
from fire_core.inference import start_inference_process
from fire_core.strategies import LiveInferenceStrategy
from fire_core.loop import run_control_loop


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wxyz_to_rotation(q: np.ndarray) -> Rotation:
    """[qw, qx, qy, qz] → scipy Rotation (internal XYZW)."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _rotation_to_wxyz(r: Rotation) -> np.ndarray:
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--task",         default="forge-peg_insert")
    p.add_argument("--control_hz",   type=float, default=15.0)
    p.add_argument("--use_cameras",  action="store_true")
    p.add_argument("--use_ft_sensor", action="store_true")
    p.add_argument("--use_sim_time", action="store_true")
    p.add_argument("--episode_length", type=int, default=88)
    p.add_argument("--lerobot_repo_id", default=None)
    p.add_argument("--lerobot_root",    default=None)
    p.add_argument("--lerobot_task",    default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--last_episode",
        nargs="?", const=-1, type=int, default=None,
        help=(
            "After recording, encode deferred GR00T videos up to this episode "
            "index. Omit value to encode through the latest episode."
        ),
    )
    p.add_argument("--encoder_threads", type=int, default=None)

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--teleop", choices=["inverse3"], default=None,
        help="Use a teleop device for demonstration recording.",
    )

    # ── Model mode ────────────────────────────────────────────────────────────
    ckpt = p.add_mutually_exclusive_group()
    ckpt.add_argument("--checkpoint")
    ckpt.add_argument("--hf_checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--vla", type=str, default=None, choices=["gr00t", "pi05"],
        help="Recording backend (required for model mode; optional for teleop).",
    )

    # ── Inverse3 teleop options ───────────────────────────────────────────────
    inv3 = p.add_argument_group("Inverse3 Teleop")
    inv3.add_argument("--inv3_port",       default="/dev/inverse3_left")
    inv3.add_argument("--versegrip_port",  default="/dev/versegrip_left")
    inv3.add_argument("--position_scale",  type=float, default=1.0)
    inv3.add_argument("--rotation_scale",  type=float, default=1.0)
    inv3.add_argument("--enable_button",   type=int, default=0)

    args = p.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if args.last_episode is not None:
        if args.lerobot_root is None:
            p.error("--last_episode requires --lerobot_root")
        if args.last_episode < -1:
            p.error("--last_episode must be >= 0")

    if args.teleop is None:
        # Model mode: checkpoint + vla required
        no_ckpt = args.checkpoint is None and args.hf_checkpoint is None
        if no_ckpt:
            if args.last_episode is not None:
                return args
            p.error("Model mode requires --checkpoint or --hf_checkpoint")
        if args.vla is None:
            p.error("Model mode requires --vla")
        if args.last_episode is not None and args.vla != "gr00t":
            p.error("--last_episode is only supported with --vla gr00t")
        if args.resume and args.vla != "gr00t":
            p.error("--resume is only supported with --vla gr00t")
    else:
        # Teleop mode: checkpoint not needed
        if args.checkpoint is not None or args.hf_checkpoint is not None:
            print("[WARN] --checkpoint/--hf_checkpoint ignored in teleop mode.")

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Shared: recorder factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_recorder(args: argparse.Namespace, robot: Any) -> Any | None:
    if args.vla is None:
        return None
    if args.vla == "gr00t":
        from fire_core.recorders import GR00TRecorder
        return GR00TRecorder(
            robot=robot,
            repo_id=args.lerobot_repo_id,
            root=args.lerobot_root,
            task=args.task,
            task_text=args.lerobot_task,
            fps=int(round(args.control_hz)),
            resume=args.resume,
            defer_video_encoding=True,
        )
    if args.vla == "pi05":
        from fire_core.recorders import PI05Recorder
        return PI05Recorder(
            robot=robot,
            repo_id=args.lerobot_repo_id,
            root=args.lerobot_root,
            task=args.task,
            task_text=args.lerobot_task,
            fps=int(round(args.control_hz)),
        )
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Encode helper
# ──────────────────────────────────────────────────────────────────────────────

def encode_deferred_videos(
    args: argparse.Namespace, root: Path | None = None
) -> None:
    from fire_core.recorders.gr00t_exporter import GR00TExporter
    last_episode = None if args.last_episode == -1 else args.last_episode
    GR00TExporter.encode_deferred_videos(
        root=root or Path(args.lerobot_root),
        last_episode=last_episode,
        vcodec="h264",
        encoder_threads=args.encoder_threads,
        remove_images=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Teleop control loop
# ──────────────────────────────────────────────────────────────────────────────

def run_teleop_loop(robot, teleop, recorder, *, control_hz: float, max_steps: int) -> None:
    from lerobot_robot_fr3.fr3 import TeleopAction

    dt = 1.0 / control_hz
    prev_enabled = False
    robot_home_pos = robot.robot_state_manager.ee_pos.copy()
    robot_home_quat = robot.robot_state_manager.ee_quat.copy()

    print(f"\n[INFO] Teleop loop @ {control_hz} Hz  (hold enable button to control)")
    print("[INFO] Press Ctrl+C to stop and save.")

    step = 0
    try:
        while max_steps <= 0 or step < max_steps:
            t0 = time.time()

            action = teleop.get_action()
            enabled = bool(action["inv3.enabled"].item())

            # Rising edge: re-anchor robot home so robot stays put on engage
            if enabled and not prev_enabled:
                robot_home_pos = robot.robot_state_manager.ee_pos.copy()
                robot_home_quat = robot.robot_state_manager.ee_quat.copy()
                print(f"\n[INFO] Teleop engaged — home: {robot_home_pos}")

            prev_enabled = enabled

            if not enabled:
                sleep_t = dt - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
                continue

            # Compute absolute EEF target
            target_pos = robot_home_pos + action["inv3.pos"]
            delta_rot = _wxyz_to_rotation(action["inv3.rot"])
            home_rot = _wxyz_to_rotation(robot_home_quat)
            target_rot = delta_rot * home_rot
            target_quat = _rotation_to_wxyz(target_rot)
            arm_action = np.concatenate(
                [target_pos, target_quat]
            ).astype(np.float32)

            robot.send_teleop_action(TeleopAction(
                action={
                    "arm_actions": arm_action,
                    "gripper_actions": np.array([], dtype=np.float32),
                },
                action_space="task_space",
                is_relative=False,
            ))

            if recorder is not None:
                recorder.record(
                    arm_action=arm_action,
                    gripper_action=np.array([-1.0], dtype=np.float32),
                )

            step += 1
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[INFO] Teleop stopped by user.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Encode-only ───────────────────────────────────────────────────────────
    encode_only = args.checkpoint is None and args.hf_checkpoint is None
    if args.last_episode is not None and encode_only and args.teleop is None:
        encode_deferred_videos(args)
        return

    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time,
        is_relative=False,
        rotation_type="quaternion",
        use_cameras=args.use_cameras,
        use_ft_sensor=args.use_ft_sensor,
    )
    robot = FR3Robot(config, task_name=args.task)

    # ── Teleop mode ───────────────────────────────────────────────────────────
    if args.teleop == "inverse3":
        from lerobot_teleoperator_inverse3 import (
            Inverse3Teleop, Inverse3TeleopConfig,
        )

        teleop_cfg = Inverse3TeleopConfig(
            inverse3_port=args.inv3_port,
            versegrip_port=args.versegrip_port,
            position_scale=args.position_scale,
            rotation_scale=args.rotation_scale,
            enable_button=args.enable_button,
        )
        teleop = Inverse3Teleop(teleop_cfg)

        if args.vla == "gr00t" and not args.use_cameras:
            print(
                "[WARN] Recording GR00T dataset without --use_cameras. "
                "The dataset will contain zero-filled fallback images."
            )

        print("[INFO] Connecting to robot ...")
        robot.connect()

        print("[INFO] Connecting to Inverse3 ...")
        teleop.connect()

        recorder = _build_recorder(args, robot)
        encode_root: Path | None = None
        try:
            run_teleop_loop(
                robot, teleop, recorder,
                control_hz=args.control_hz,
                max_steps=args.episode_length,
            )
        finally:
            try:
                if recorder:
                    recorder.save()
                    if args.last_episode is not None:
                        encode_root = recorder.output_root
            finally:
                teleop.disconnect()
                robot.disconnect()
                print("[INFO] Disconnected.")

        if encode_root is not None:
            encode_deferred_videos(args, root=encode_root)
        return

    # ── Model mode ────────────────────────────────────────────────────────────
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        hf_checkpoint=args.hf_checkpoint,
    )

    obs_features    = robot.task.observation_features
    action_features = robot.task.action_features
    obs_dim    = total_dim(obs_features)
    action_dim = total_dim(action_features)
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    if args.vla == "gr00t" and not args.use_cameras:
        print(
            "[WARN] Recording GR00T dataset without --use_cameras. "
            "The dataset will contain zero-filled fallback images."
        )

    obs_shm    = torch.zeros(1, obs_dim).share_memory_()
    action_shm = torch.zeros(1, action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)

    recorder = _build_recorder(args, robot)

    infer_handle = start_inference_process(
        obs_shm, action_shm, obs_flag, action_flag,
        checkpoint=checkpoint_path,
        cfg_path=robot.task.model_cfg_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device,
    )
    infer_handle.wait_ready(action_flag)

    print("[INFO] Connecting to robot ...")
    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] {e}")
        infer_handle.shutdown()
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect.")
        infer_handle.shutdown()
        return

    obs_buf = obs_shm.numpy()[0]
    strategy = LiveInferenceStrategy(
        robot, obs_buf, obs_features,
        obs_flag, action_flag, action_shm, action_dim,
    )

    encode_root = None
    try:
        run_control_loop(
            robot, strategy,
            control_hz=args.control_hz,
            max_steps=args.episode_length,
            recorder=recorder,
        )
    finally:
        try:
            if recorder:
                recorder.save()
                if args.last_episode is not None:
                    encode_root = recorder.output_root
        finally:
            infer_handle.shutdown()
            cv2.destroyAllWindows()
            robot.disconnect()
            print("[INFO] Robot disconnected.")

    if encode_root is not None:
        encode_deferred_videos(args, root=encode_root)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
