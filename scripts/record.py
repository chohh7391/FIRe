from __future__ import annotations

import argparse
import ctypes
from multiprocessing import RawValue
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.multiprocessing as mp

from fire_core.checkpoints import resolve_checkpoint_path
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
    checkpoint_group = p.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument("--checkpoint", help="Local checkpoint path.")
    checkpoint_group.add_argument(
        "--hf_checkpoint",
        help="Hugging Face checkpoint path, e.g. user/repo or user/repo/path/to/model.pth.",
    )
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--control_hz",     type=float, default=15.0)
    p.add_argument("--use_cameras",    action="store_true")
    p.add_argument("--use_ft_sensor",  action="store_true")
    p.add_argument("--use_sim_time",   action="store_true")
    p.add_argument("--vla", type=str, default=None, choices=["gr00t", "pi05"])
    p.add_argument("--episode_length", type=int, default=88)
    p.add_argument("--lerobot_repo_id", default=None, help="Optional HF repo id for metadata.")
    p.add_argument("--lerobot_root",    default=None, help="Local root directory for the LeRobot dataset.")
    p.add_argument("--lerobot_task",    default=None, help="Natural-language task saved in the dataset.")
    p.add_argument("--resume",          action="store_true", help="Append a new episode to an existing dataset root.")
    p.add_argument(
        "--last_episode",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help=(
            "After recording, encode deferred GR00T videos from episode 0 through this inclusive "
            "episode index. Omit the value to encode through the latest episode in the root. "
            "If used without a checkpoint, runs encode-only mode."
        ),
    )
    p.add_argument(
        "--encoder_threads",
        type=int,
        default=None,
        help="Optional number of threads per video encoder when encoding deferred videos.",
    )

    args = p.parse_args()
    if args.last_episode is not None:
        if args.lerobot_root is None:
            p.error("--last_episode requires --lerobot_root")
        if args.last_episode < -1:
            p.error("--last_episode must be >= 0, or omit the value to encode through the latest episode")

    if args.checkpoint is None and args.hf_checkpoint is None:
        if args.last_episode is not None:
            return args
        p.error("scripts/record.py requires --checkpoint or --hf_checkpoint unless --last_episode is used")
    if args.vla is None:
        p.error("scripts/record.py requires a recording backend via --vla")
    if args.last_episode is not None and args.vla != "gr00t":
        p.error("--last_episode is currently implemented for --vla gr00t only")
    if args.resume and args.vla != "gr00t":
        p.error("--resume is currently implemented for --vla gr00t only")
    return args


def encode_deferred_videos(args: argparse.Namespace, root: Path | None = None) -> None:
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
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    encode_only = args.checkpoint is None and args.hf_checkpoint is None
    if args.last_episode is not None and encode_only:
        encode_deferred_videos(args)
        return

    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        hf_checkpoint=args.hf_checkpoint,
    )

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

    if args.vla == "gr00t" and not args.use_cameras:
        print(
            "[WARN] Recording GR00T dataset without --use_cameras. "
            "The dataset will contain zero-filled fallback images."
        )

    # ── Shared memory ─────────────────────────────────────────────────────────
    obs_shm    = torch.zeros(1, obs_dim).share_memory_()
    action_shm = torch.zeros(1, action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)

    # ── VLA client ────────────────────────────────────────────────────────────
    recorder: Any | None = None
    if args.vla == "gr00t":
        from fire_core.recorders import GR00TRecorder
        recorder = GR00TRecorder(
            robot=robot,
            repo_id=args.lerobot_repo_id,
            root=args.lerobot_root,
            task=args.task,
            task_text=args.lerobot_task,
            fps=int(round(args.control_hz)),
            resume=args.resume,
            defer_video_encoding=True,
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
        checkpoint=checkpoint_path,
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
    encode_root: Path | None = None
    try:
        run_control_loop(robot, strategy, control_hz=args.control_hz, max_steps=args.episode_length, recorder=recorder)
    finally:
        try:
            if recorder:
                recorder.save()
                if args.last_episode is not None:
                    encode_root = recorder.output_root
        finally:
            if infer_handle:
                infer_handle.shutdown()
            cv2.destroyAllWindows()
            robot.disconnect()
            print("[INFO] Robot disconnected.")

    if encode_root is not None:
        encode_deferred_videos(args, root=encode_root)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
