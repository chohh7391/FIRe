from __future__ import annotations

import argparse
import ctypes
from multiprocessing import RawValue
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from fire_core.utils import total_dim
from fire_core.inference import start_inference_process
from fire_core.strategies import LiveInferenceStrategy
from fire_core.loop import run_control_loop


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",           default="forge-peg_insert")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--control_hz",     type=float, default=15.0)
    p.add_argument("--use_cameras",    action="store_true")
    p.add_argument("--use_ft_sensor",  action="store_true")
    p.add_argument("--use_sim_time",   action="store_true")
    p.add_argument("--vla", type=str, default=None, choices=["gr00t", "pi05"])
    p.add_argument("--episode_length", type=int, default=300)
    p.add_argument("--lerobot_repo_id", default=None, help="Enable LeRobot recording with this repo id.")
    p.add_argument("--lerobot_root",    default=None, help="Local root directory for the LeRobot dataset.")
    p.add_argument("--lerobot_task",    default=None, help="Natural-language task saved in the dataset.")

    args = p.parse_args()
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    args = parse_args()

    # ── Robot 객체 (connect 전에 task features 접근) ───────────────────────────
    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time,
        is_relative=False,
        rotation_type="quaternion",
        use_cameras=args.use_cameras,
        use_ft_sensor=args.use_ft_sensor,
    )
    robot = FR3Robot(config, task_name=args.task)

    obs_features    = robot.task.observation_features
    log_features    = robot.task.log_features
    action_features = robot.task.action_features
    obs_dim    = total_dim(obs_features)
    action_dim = total_dim(action_features)
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    # ── Shared memory ─────────────────────────────────────────────────────────
    obs_shm    = torch.zeros(1, obs_dim).share_memory_()
    action_shm = torch.zeros(1, action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)

    # ── VLA client ────────────────────────────────────────────────────────────
    recorder = None
    if args.vla == "gr00t":
        from fire_core.recorders import GR00TRecorder
        recorder = GR00TRecorder(
            robot=robot,
            repo_id=args.lerobot_repo_id,
            root=args.lerobot_root,
            task=args.task,
            task_text=args.lerobot_task,
            fps=int(round(args.control_hz)),
        )
    elif args.vla == "pi05":
        from fire_core.recorders import PI05Recorder
        recorder = PI05Recorder(
            robot=robot,
            repo_id=args.lerobot_repo_id,
            root=args.lerobot_root,
            task=args.task,
            task_text=args.lerobot_task,
            fps=int(round(args.control_hz)),
        )

    # ── Inference process ─────────────────────────────────────────────────────
    infer_handle = start_inference_process(
        obs_shm, action_shm, obs_flag, action_flag,
        checkpoint=args.checkpoint,
        cfg_path=robot.task.model_cfg_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device,
    )
    infer_handle.wait_ready(action_flag)

    # ── Robot connect ─────────────────────────────────────────────────────────
    print("[INFO] Connecting to robot ...")
    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] {e}")
        if infer_handle:
            infer_handle.shutdown()
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect.")
        if infer_handle:
            infer_handle.shutdown()
        return

    obs_buf = obs_shm.numpy()[0]

    strategy = LiveInferenceStrategy(
        robot, obs_buf, obs_features,
        obs_flag, action_flag, action_shm, action_dim,
    )

    # ── Control loop ──────────────────────────────────────────────────────────
    try:
        run_control_loop(robot, strategy, control_hz=args.control_hz, max_steps=args.episode_length, recorder=recorder)
    finally:
        if recorder:
            recorder.save()
        if infer_handle:
            infer_handle.shutdown()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()