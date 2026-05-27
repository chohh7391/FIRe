"""play.py

Franka FR3 inference runner.

Architecture: Fully Decoupled 3-Loop (Sense → Plan → Act)
----------------------------------------------------------
  Observation loop  : robot I/O를 계속 수집해 SharedState에 넣음.
  RL inference loop : obs를 subprocess에 trigger (non-blocking).
                      별도 poll thread가 action을 SharedState에 씀.
  VLA inference loop: control loop의 Event를 받아 다음 chunk를 비동기 요청.
  Control loop      : 정확한 Hz로 latest action을 꺼내 로봇에 전송.
                      inference latency에 무관 — 절대 blocking 없음.
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
import threading
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from lerobot_robot_fr3.utils.runner.utils import (
    Features, total_dim, flat_keys, flatten,
    obs_to_indexed, flat_array_to_indexed, write_obs_shm, feature_slice,
)
from runners.utils import SharedState
from runners.rl_runner import RLRunner
from runners.vla_runner import VLARunner


# ──────────────────────────────────────────────────────────────────────────────
# Observation loop
# ──────────────────────────────────────────────────────────────────────────────

def observation_loop(robot, shared_state: SharedState) -> None:
    print("[OBS-LOOP] Started.")
    while shared_state.is_running:
        obs_dict = robot.get_observation()
        shared_state.update_obs(obs_dict)


# ──────────────────────────────────────────────────────────────────────────────
# RL inference loop
# ──────────────────────────────────────────────────────────────────────────────

def rl_inference_loop(
    rl_runner: RLRunner,
    obs_features: Features,
    shared_state: SharedState,
) -> None:
    """obs가 업데이트될 때마다 subprocess에 trigger — non-blocking.

    결과는 rl_runner.start_action_poll()이 생성한 poll thread가
    SharedState.update_rl_action()으로 채워줌.
    """
    print("[RL-LOOP] Started.")
    obs_buf = np.zeros(rl_runner.obs_dim, dtype=np.float32)
    prev_obs_id: Optional[int] = None

    while shared_state.is_running:
        obs_dict = shared_state.get_obs()
        if obs_dict is None:
            time.sleep(0.001)
            continue

        # 동일한 obs_dict 객체면 skip (새 obs가 아직 안 왔음)
        curr_id = id(obs_dict)
        if curr_id == prev_obs_id:
            time.sleep(0.001)
            continue
        prev_obs_id = curr_id

        write_obs_shm(obs_dict, obs_buf, obs_features)
        rl_runner.trigger(obs_buf)  # non-blocking: flag 세우고 즉시 반환


# ──────────────────────────────────────────────────────────────────────────────
# VLA inference loop
# ──────────────────────────────────────────────────────────────────────────────

def vla_inference_loop(
    vla_runner: VLARunner,
    robot,
    shared_state: SharedState,
) -> None:
    print("[VLA-LOOP] Started.")

    # 첫 chunk 동기 수신
    chunk = vla_runner.get_action_sync(robot.get_vla_observation())
    shared_state.update_vla_chunk(chunk)

    while shared_state.is_running:
        # control loop가 Event를 set할 때까지 blocking (CPU 소모 없음)
        shared_state.vla_request_event.wait()
        if not shared_state.is_running:
            break
        shared_state.vla_request_event.clear()

        try:
            vla_runner.request_action(robot.get_vla_observation())
            chunk = vla_runner.get_result()
            shared_state.update_vla_chunk(chunk)
        except Exception as e:
            print(f"[WARN] VLA inference failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Control loop
# ──────────────────────────────────────────────────────────────────────────────

def control_loop(
    robot,
    shared_state: SharedState,
    control_hz: float,
    vla_chunk_size: int,
    logger: Optional["StepLogger"],
) -> None:
    dt = 1.0 / control_hz

    try:
        # Warm-up: 첫 RL action이 준비될 때까지 대기
        print("[CONTROL-LOOP] Waiting for first RL action ...")
        while shared_state.is_running and not shared_state.has_first_rl_action():
            time.sleep(0.01)

        print(f"[CONTROL-LOOP] Running @ {control_hz} Hz  (Ctrl+C to stop)")

        step_idx = 0
        while shared_state.is_running:
            t_start = time.time()

            # 1. Latest obs & actions 획득 (non-blocking)
            obs_dict = shared_state.get_obs()
            rl_action, vla_chunk = shared_state.get_actions()

            if obs_dict is None or rl_action is None:
                time.sleep(0.001)
                continue

            # 2. RL + VLA 합산
            combined = rl_action.copy()
            if vla_chunk is not None:
                chunk_idx = step_idx % vla_chunk_size
                combined += vla_chunk[chunk_idx]

                # chunk 중간 시점에 다음 chunk 비동기 요청
                if chunk_idx == vla_chunk_size // 2:
                    shared_state.vla_request_event.set()

            action_dict = {
                "arm_actions":     combined,
                "gripper_actions": np.array([-1.0], dtype=np.float32),
            }

            # 3. 전송
            processed_action = robot.send_action(action_dict)

            # 4. 로깅
            if logger:
                logger.record(obs_dict, combined, processed_action)

            # 5. Hz 유지
            step_idx += 1
            sleep_t = dt - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[INFO] Control loop stopped by user.")
        shared_state.is_running = False


# ──────────────────────────────────────────────────────────────────────────────
# Replay loop
# ──────────────────────────────────────────────────────────────────────────────

def run_replay_loop(
    robot,
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
    obs_features: Features,
    log_features: Features,
    logger: Optional["StepLogger"],
) -> None:
    df = pd.read_csv(args.replay)
    print(f"[INFO] Replaying {len(df)} steps from {args.replay}")

    rl_runner: Optional[RLRunner] = None
    if args.raw:
        rl_runner = RLRunner(obs_dim, action_dim, args.checkpoint, args.cfg, args.device)
        rl_runner.start()

    dt = 1.0 / args.control_hz

    if args.raw and len(df) > 0:
        first_obs_np = df.iloc[0][[f"obs_{i}" for i in range(obs_dim)]].to_numpy(dtype=np.float32)
        _sync_initial_ema(robot, first_obs_np, obs_features)

    try:
        for step_idx in range(len(df)):
            t_start = time.time()
            row = df.iloc[step_idx]

            if args.pose:
                pose_cols = (
                    flat_keys({"target_pos":  log_features["target_pos"]})
                    + flat_keys({"target_quat": log_features["target_quat"]})
                )
                arm_action = row[pose_cols].to_numpy(dtype=np.float32)
                action_dict = {"arm_actions": arm_action, "gripper_actions": np.array([-1.0], dtype=np.float32)}
                processed = robot.send_processed_action(action_dict)
            else:
                obs_np = row[[f"obs_{i}" for i in range(obs_dim)]].to_numpy(dtype=np.float32)
                # replay raw는 동기 infer 사용 (단일 스레드, 순서 보장 필요)
                rl_runner.obs_shm[0].copy_(__import__("torch").from_numpy(obs_np))
                rl_runner.obs_flag.value = 1
                while rl_runner.action_flag.value == 0:
                    pass
                rl_runner.action_flag.value = 0
                arm_action = rl_runner.action_shm.numpy().flatten()[:action_dim].copy()

                action_dict = {"arm_actions": arm_action, "gripper_actions": np.array([-1.0], dtype=np.float32)}
                obs_dict = robot.get_observation()
                processed = robot.send_action(action_dict)

                if logger:
                    logger.record(obs_dict, arm_action, processed)

            print(f"\r[REPLAY] {step_idx + 1}/{len(df)}", end="")
            sleep_t = dt - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[INFO] Replay stopped by user.")
    finally:
        if rl_runner:
            rl_runner.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

class StepLogger:
    def __init__(self, robot, obs_features: Features, log_features: Features) -> None:
        self._robot = robot
        self._obs_features = obs_features
        self._log_features = log_features
        self._buffer: List[dict] = []

    def record(
        self,
        obs_dict: dict,
        policy_action: np.ndarray,
        processed_action: dict,
    ) -> None:
        rec: dict = {}
        rec.update(obs_to_indexed(obs_dict, self._obs_features))
        rec.update(flat_array_to_indexed("policy_action", policy_action))
        rec.update(flat_array_to_indexed("normalized_action", policy_action))

        task_action = getattr(self._robot.task, "action", None)
        if task_action is not None:
            rec.update(flat_array_to_indexed("ema_action", task_action))
            rec.update(flat_array_to_indexed("raw_action", task_action))

        proc_arm = processed_action.get("processed_arm_action")
        proc_grip = processed_action.get("processed_gripper_action")
        if proc_arm is not None:
            rec.update(flat_array_to_indexed("processed_arm_action", proc_arm))
        if proc_grip is not None:
            rec.update(flat_array_to_indexed("processed_gripper_action", proc_grip))

        rec.update(flatten(self._robot.task.get_log(), self._log_features))
        self._buffer.append(rec)

    def save(self, save_dir: str) -> None:
        if not self._buffer:
            print("[INFO] Logger buffer empty — nothing to save.")
            return
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"collected_data_{ts}.csv")
        pd.DataFrame(self._buffer).to_csv(path, index=False)
        print(f"[INFO] Saved {len(self._buffer)} steps → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# EMA sync helper
# ──────────────────────────────────────────────────────────────────────────────

def _sync_initial_ema(robot, obs_np: np.ndarray, obs_features: Features) -> None:
    prev_slice = feature_slice(obs_features, "prev_actions")
    if prev_slice is None:
        print("[WARN] EMA sync skipped: prev_actions not in obs_features.")
        return

    task = getattr(robot, "task", None)
    if task is None:
        return

    prev_actions = obs_np[prev_slice].astype(np.float32, copy=True)
    synced = False

    for action_name, prev_name in (("action", "prev_action"), ("actions", "prev_actions")):
        if not hasattr(task, action_name):
            continue
        action_arr = getattr(task, action_name)
        npa = np.asarray(action_arr, dtype=np.float32).copy()
        npa.reshape(-1)[:min(npa.size, prev_actions.size)] = prev_actions[:min(npa.size, prev_actions.size)]
        setattr(task, action_name, npa.reshape(np.asarray(action_arr).shape))

        if hasattr(task, prev_name):
            prev_arr = getattr(task, prev_name)
            npp = np.asarray(prev_arr, dtype=np.float32).copy()
            npp.reshape(-1)[:min(npp.size, prev_actions.size)] = prev_actions[:min(npp.size, prev_actions.size)]
            setattr(task, prev_name, npp.reshape(np.asarray(prev_arr).shape))
        synced = True

    if synced:
        print("[INFO] EMA state synced from obs_0.prev_actions.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",           default="peg_insert")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--cfg",            required=True)
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--control_hz",     type=float, default=15.0)
    p.add_argument("--replay",         default=None, metavar="CSV")
    p.add_argument("--save_path",      default=None)
    p.add_argument("--use_cameras",    action="store_true")
    p.add_argument("--use_ft_sensor",  action="store_true")
    p.add_argument("--use_sim_time",   action="store_true")
    p.add_argument("--vla",            type=str, default=None, choices=["gr00t", "pi05"])
    p.add_argument("--vla_chunk_size", type=int, default=16)
    p.add_argument("--host",           type=str, default=None)
    p.add_argument("--port",           type=int, default=None)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--raw",  action="store_true", help="Replay: CSV obs → RL inference")
    mode.add_argument("--pose", action="store_true", help="Replay: CSV target pose → direct send")

    args = p.parse_args()
    if (args.raw or args.pose) and not args.replay:
        p.error("--raw / --pose requires --replay")
    if args.replay and not (args.raw or args.pose):
        p.error("--replay requires --raw or --pose")
    if args.vla and (args.host is None or args.port is None):
        p.error("--vla requires --host and --port")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    args = parse_args()

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
    obs_dim         = total_dim(obs_features)
    action_dim      = total_dim(robot.task.action_features)
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    print("[INFO] Connecting to robot ...")
    robot.connect()
    if not robot.is_connected:
        print("[ERROR] Failed to connect.")
        return

    logger = StepLogger(robot, obs_features, log_features) if args.save_path else None

    # ── Replay ────────────────────────────────────────────────────────────────
    if args.replay:
        try:
            run_replay_loop(robot, args, obs_dim, action_dim, obs_features, log_features, logger)
        finally:
            if logger:
                logger.save(args.save_path)
            robot.disconnect()
        return

    # ── Live inference ────────────────────────────────────────────────────────
    shared_state = SharedState()

    rl_runner = RLRunner(obs_dim, action_dim, args.checkpoint, args.cfg, args.device)
    rl_runner.start()
    rl_runner.start_action_poll(shared_state)  # action_flag → SharedState bridge

    vla_runner: Optional[VLARunner] = None
    if args.vla:
        vla_runner = VLARunner(args.vla, args.host, args.port, action_dim)
        vla_runner.start()

    threads = [
        threading.Thread(
            target=observation_loop,
            args=(robot, shared_state),
            daemon=True,
        ),
        threading.Thread(
            target=rl_inference_loop,
            args=(rl_runner, obs_features, shared_state),
            daemon=True,
        ),
    ]
    if vla_runner:
        threads.append(threading.Thread(
            target=vla_inference_loop,
            args=(vla_runner, robot, shared_state),
            daemon=True,
        ))

    for t in threads:
        t.start()

    try:
        control_loop(robot, shared_state, args.control_hz, args.vla_chunk_size, logger)
    finally:
        print("\n[INFO] Shutting down ...")
        shared_state.is_running = False
        shared_state.vla_request_event.set()  # vla loop의 blocking wait 해제

        if logger:
            logger.save(args.save_path)

        rl_runner.stop()
        if vla_runner:
            vla_runner.stop()

        for t in threads:
            t.join(timeout=1.0)

        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()