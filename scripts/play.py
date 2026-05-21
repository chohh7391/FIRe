"""vla_runner.py

Franka FR3 VLA inference runner.

Step ordering — mirrors Isaac Lab DirectRLEnv
---------------------------------------------
Isaac Lab:
  reset()  →  obs_0  (zero action initialized)
  loop:
    action_t = policy(obs_t)     # deterministic policy output
    pre_physics_step(action_t)   # buffer action
    apply_action()               # send buffered action → robot/sim
    sim.step()                   # physics
    obs_t+1 = get_observations() # collect next obs

This runner:
  connect()  →  obs_0  (zero action initialized)
  loop:
    action_t = infer(obs_t)      # deterministic policy output
    send_action(action_t)        # apply buffered action first  ← pre_physics + apply_action
    obs_t+1 = get_observation()  # collect obs after physics    ← get_observations
    action_t+1 = infer(obs_t+1)  # compute next action (async)  ← policy

Architecture
------------
Main Process (A)  ←→  Inference Process (B)
  - ROS2 / robot I/O         - rl-games agent (GPU)
  - control loop @ N Hz      - spin-wait on obs_flag
  - CSV logging              - writes action to shared mem

Communication: shared torch tensors + RawValue spin-flags
  obs_flag  : 0=idle  1=obs_ready
  action_flag: 0=idle  1=action_ready  2=warmup_done
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import RawValue
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp


# ──────────────────────────────────────────────────────────────────────────────
# Feature utilities
# ──────────────────────────────────────────────────────────────────────────────

Features = Dict[str, Tuple[int, ...]]


def total_dim(features: Features) -> int:
    return sum(int(np.prod(s)) for s in features.values())


def flat_keys(features: Features) -> List[str]:
    keys: List[str] = []
    for name, shape in features.items():
        for i in range(int(np.prod(shape))):
            keys.append(f"{name}_{i}")
    return keys


def flatten(data: Dict[str, np.ndarray], features: Features) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, shape in features.items():
        arr = np.asarray(data[key], dtype=np.float32).ravel()
        dim = int(np.prod(shape))
        for i in range(dim):
            out[f"{key}_{i}"] = float(arr[i])
    return out


def obs_to_indexed(obs: Dict[str, np.ndarray], features: Features) -> Dict[str, float]:
    """obs_features 순서대로 obs_0 ~ obs_N 형태로 직렬화."""
    out: Dict[str, float] = {}
    idx = 0
    for key, shape in features.items():
        arr = np.asarray(obs[key], dtype=np.float32).ravel()
        for v in arr[: int(np.prod(shape))]:
            out[f"obs_{idx}"] = float(v)
            idx += 1
    return out


def flat_array_to_indexed(prefix: str, values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).ravel()
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(arr)}


def write_obs_shm(obs: dict, buf: np.ndarray, features: Features) -> None:
    """shared memory에 obs를 직접 slice-write (GIL 최소화)."""
    offset = 0
    for key, shape in features.items():
        dim = int(np.prod(shape))
        buf[offset : offset + dim] = np.asarray(obs[key], dtype=np.float32).ravel()[:dim]
        offset += dim


def feature_slice(features: Features, name: str) -> Optional[slice]:
    offset = 0
    for key, shape in features.items():
        dim = int(np.prod(shape))
        if key == name:
            return slice(offset, offset + dim)
        offset += dim
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Inference process (Process B)
# ──────────────────────────────────────────────────────────────────────────────

def _build_dummy_env(obs_dim: int, action_dim: int, device: torch.device, clip_actions: float):
    """rl-games가 요구하는 최소 env 인터페이스."""

    class _DummyEnv:
        observation_space = type("S", (), {"shape": (obs_dim,)})()
        action_space = type("S", (), {
            "shape": (action_dim,),
            "high":  np.full(action_dim, clip_actions, dtype=np.float32),
            "low":  np.full(action_dim, -clip_actions, dtype=np.float32),
        })()
        num_envs = 1

        def reset(self):
            return torch.zeros(1, obs_dim, device=device)

        def step(self, _):
            return torch.zeros(1, obs_dim, device=device), torch.zeros(1), \
                   torch.zeros(1, dtype=torch.bool), {}

    return _DummyEnv()


def run_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,
    action_flag: RawValue,
    stop_event: mp.Event,
    *,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device_str: str,
    warmup_steps: int = 50,
) -> None:
    """Inference-only subprocess: ROS 없음, GIL 경합 없음."""
    import yaml
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["params"].update({
        "load_checkpoint": True,
        "load_path": checkpoint,
    })
    cfg["params"]["config"].update({
        "device": str(device),
        "device_name": str(device),
        "num_actors": 1,
    })

    clip_actions = float(cfg["params"]["env"].get("clip_actions", 1.0))
    dummy_env = _build_dummy_env(obs_dim, action_dim, device, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda *_, **__: dummy_env)
    env_configurations.register("rlgpu", {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **__: dummy_env,
    })

    runner = Runner()
    runner.load(cfg)
    # FIRe's rl-games fork enables CUDA matmul TF32 in Runner.__init__ for speed.
    # The sim environment keeps PyTorch's default matmul precision, so restore
    # those defaults for action comparisons.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    agent = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()
    dummy_obs = torch.zeros(1, obs_dim, device=device)
    _ = agent.get_batch_size(dummy_obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    model_device = next(agent.model.parameters()).device

    print("[INFER] Ready.")
    action_flag.value = 2  # signal: warm-up done

    # ── Inference loop (spin-wait) ────────────────────────────────────────────
    while not stop_event.is_set():
        if obs_flag.value != 1:
            continue
        obs_flag.value = 0

        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs_shm.to(device)).to(model_device)
            act   = agent.get_action(obs_t, is_deterministic=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        action_shm.copy_(act.cpu())
        action_flag.value = 1

    print("[INFER] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Control strategies (Strategy pattern)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    obs_dict: Optional[dict]       # policy_action을 만든 observation dict
    action_dict: dict              # 이번 스텝에 send할 raw policy action
    policy_action: Optional[np.ndarray] = None
    obs_array: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
    processed_action: Optional[dict] = None


class ControlStrategy(ABC):
    """Isaac Lab DirectRLEnv와 동일한 action-first 스텝 순서를 구현하는 전략.

    reset()에서 obs_0 inference를 미리 걸고, step()은 pending obs_t로부터
    계산된 action_t를 반환한다. action_t 전송 이후 after_action_sent()가
    obs_t+1을 준비하고 다음 inference를 트리거한다.
    """

    @abstractmethod
    def reset(self) -> dict:
        """첫 inference 입력을 준비하고 내부 action 버퍼를 초기화."""
        ...

    @abstractmethod
    def step(self, step_idx: int) -> Optional[StepResult]:
        """None 반환 시 루프 종료."""
        ...

    def after_action_sent(self, step_idx: int, result: StepResult) -> StepResult:
        """Action 전송 이후 다음 inference 입력을 준비한다."""
        return result


def _zero_action(action_dim: int) -> dict:
    return {
        "arm_actions":     np.zeros(action_dim, dtype=np.float32),
        "gripper_actions": np.array([-1.0], dtype=np.float32),
    }


class LiveInferenceStrategy(ControlStrategy):
    """실시간: action apply → obs 수집 → inference (Isaac Lab 순서)."""

    def __init__(self, robot, obs_buf: np.ndarray, obs_features: Features,
                 obs_flag: RawValue, action_flag: RawValue,
                 action_shm: torch.Tensor, action_dim: int):
        self._robot = robot
        self._obs_buf = obs_buf
        self._obs_features = obs_features
        self._obs_flag = obs_flag
        self._action_flag = action_flag
        self._action_shm = action_shm
        self._action_dim = action_dim
        self._buffered_action: dict = _zero_action(action_dim)
        self._pending_obs_dict: Optional[dict] = None

    def _wait_inference(self) -> np.ndarray:
        while self._action_flag.value == 0:
            pass
        self._action_flag.value = 0
        return self._action_shm.numpy().flatten()[: self._action_dim].copy()

    def _trigger_inference(self, obs_dict: dict) -> None:
        write_obs_shm(obs_dict, self._obs_buf, self._obs_features)
        self._obs_flag.value = 1

    def reset(self) -> dict:
        self._buffered_action = _zero_action(self._action_dim)
        obs_dict = self._robot.get_observation()
        self._pending_obs_dict = obs_dict
        # 첫 obs로 첫 inference를 미리 트리거 -> step(0)에서 바로 action 사용
        self._trigger_inference(obs_dict)
        return obs_dict

    def step(self, step_idx: int) -> Optional[StepResult]:
        # 1. pending obs_t로 계산된 inference 결과를 회수해 action_t 구성
        arm_action = self._wait_inference()
        self._buffered_action = {
            "arm_actions":     arm_action,
            "gripper_actions": np.array([-1.0], dtype=np.float32),
        }

        return StepResult(
            obs_dict=self._pending_obs_dict,
            action_dict=self._buffered_action,
            policy_action=arm_action,
        )

    def after_action_sent(self, step_idx: int, result: StepResult) -> StepResult:
        # action_t 전송 이후 obs_{t+1}을 수집하고 다음 inference를 준비
        obs_dict = self._robot.get_observation()
        self._pending_obs_dict = obs_dict
        self._trigger_inference(obs_dict)
        return result


class ReplayRawStrategy(ControlStrategy):
    """Replay --raw: CSV obs → inference → action (Isaac Lab 순서)."""

    def __init__(self, replay_data: pd.DataFrame, obs_buf: np.ndarray,
                 obs_flag: RawValue, action_flag: RawValue,
                 action_shm: torch.Tensor, obs_dim: int, action_dim: int,
                 robot=None, obs_features: Optional[Features] = None):
        self._data = replay_data
        self._obs_buf = obs_buf
        self._obs_flag = obs_flag
        self._action_flag = action_flag
        self._action_shm = action_shm
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._buffered_action: dict = _zero_action(action_dim)
        self._robot = robot
        self._obs_features = obs_features
        self._pending_obs_np: Optional[np.ndarray] = None

    def _load_obs_row(self, step_idx: int) -> np.ndarray:
        row = self._data.iloc[step_idx]
        return row[[f"obs_{i}" for i in range(self._obs_dim)]].to_numpy(dtype=np.float32)

    def _row_metadata(self, step_idx: int) -> dict:
        row = self._data.iloc[step_idx]
        metadata = {}
        for col in ("episode", "step"):
            if col in self._data.columns:
                value = row[col]
                metadata[col] = int(value) if float(value).is_integer() else float(value)
        return metadata

    def _trigger_inference(self, obs_np: np.ndarray) -> None:
        self._obs_buf[:] = obs_np
        self._obs_flag.value = 1

    def _wait_inference(self) -> np.ndarray:
        while self._action_flag.value == 0:
            pass
        self._action_flag.value = 0
        return self._action_shm.numpy().flatten()[: self._action_dim].copy()

    def _sync_initial_action_from_obs(self, obs_np: np.ndarray) -> None:
        if self._robot is None or self._obs_features is None:
            return

        prev_slice = feature_slice(self._obs_features, "prev_actions")
        if prev_slice is None:
            print("[WARN] Replay raw initial action sync skipped: prev_actions not in obs_features.")
            return

        task = getattr(self._robot, "task", None)
        if task is None:
            print("[WARN] Replay raw initial action sync skipped: robot has no task.")
            return

        prev_actions = obs_np[prev_slice].astype(np.float32, copy=True)

        synced = False
        for action_name, prev_name in (("action", "prev_action"), ("actions", "prev_actions")):
            if not hasattr(task, action_name):
                continue

            action_arr = getattr(task, action_name)
            dim = min(np.asarray(action_arr).size, prev_actions.size)
            next_action = np.asarray(action_arr, dtype=np.float32).copy()
            next_action.reshape(-1)[:dim] = prev_actions[:dim]
            setattr(task, action_name, next_action.reshape(np.asarray(action_arr).shape))

            if hasattr(task, prev_name):
                prev_arr = getattr(task, prev_name)
                next_prev = np.asarray(prev_arr, dtype=np.float32).copy()
                prev_dim = min(next_prev.size, prev_actions.size)
                next_prev.reshape(-1)[:prev_dim] = prev_actions[:prev_dim]
                setattr(task, prev_name, next_prev.reshape(np.asarray(prev_arr).shape))

            synced = True

        if synced:
            print("[INFO] Replay raw initial EMA state synced from obs_0.prev_actions.")
        else:
            print("[WARN] Replay raw initial action sync skipped: task has no action buffer.")

    def reset(self) -> dict:
        self._buffered_action = _zero_action(self._action_dim)
        # 첫 obs로 첫 inference를 미리 트리거
        self._pending_obs_np = self._load_obs_row(0)
        self._sync_initial_action_from_obs(self._pending_obs_np)
        self._trigger_inference(self._pending_obs_np)
        return {}

    def step(self, step_idx: int) -> Optional[StepResult]:
        if step_idx >= len(self._data):
            print("\n[INFO] Replay CSV finished.")
            return None

        # 1. row step_idx의 obs_t로 계산된 inference 결과를 회수
        arm_action = self._wait_inference()
        self._buffered_action = {
            "arm_actions":     arm_action,
            "gripper_actions": np.array([-1.0], dtype=np.float32),
        }

        print(f"\r[REPLAY-RAW] {step_idx + 1}/{len(self._data)}", end="")
        return StepResult(
            obs_dict=None,
            obs_array=self._pending_obs_np,
            action_dict=self._buffered_action,
            policy_action=arm_action,
            metadata=self._row_metadata(step_idx),
        )

    def after_action_sent(self, step_idx: int, result: StepResult) -> StepResult:
        next_idx = step_idx + 1
        if next_idx >= len(self._data):
            return result

        self._pending_obs_np = self._load_obs_row(next_idx)
        self._trigger_inference(self._pending_obs_np)
        return result


class ReplayPoseStrategy(ControlStrategy):
    """Replay --pose: CSV target pose를 직접 action으로 전송 (inference 없음)."""

    def __init__(self, replay_data: pd.DataFrame, pose_cols: List[str], action_dim: int):
        self._data = replay_data
        self._pose_cols = pose_cols
        self._action_dim = action_dim
        self._buffered_action: dict = _zero_action(action_dim)

    def reset(self) -> dict:
        self._buffered_action = _zero_action(self._action_dim)
        return {}

    def step(self, step_idx: int) -> Optional[StepResult]:
        if step_idx >= len(self._data):
            print("\n[INFO] Replay CSV finished.")
            return None

        # 1. 버퍼된 action 전송
        sent_action = self._buffered_action.copy()

        # 2. 다음 스텝 target pose를 버퍼에 준비
        next_idx = step_idx + 1
        if next_idx < len(self._data):
            row = self._data.iloc[next_idx]
            target_pose = row[self._pose_cols].to_numpy(dtype=np.float32)
            self._buffered_action = {
                "arm_actions":     target_pose,
                "gripper_actions": np.array([-1.0], dtype=np.float32),
            }
        else:
            self._buffered_action = _zero_action(self._action_dim)

        print(f"\r[REPLAY-POSE] {step_idx + 1}/{len(self._data)}", end="")
        return StepResult(obs_dict=None, action_dict=sent_action)


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

class StepLogger:
    """제어 루프에서 분리된 로깅 책임."""

    def __init__(self, robot, obs_features: Features, log_features: Features):
        self._robot = robot
        self._obs_features = obs_features
        self._log_features = log_features
        self._buffer: List[dict] = []

    def record(self, result: StepResult) -> None:
        record = dict(result.metadata or {})

        if result.obs_array is not None:
            record.update(flat_array_to_indexed("obs", result.obs_array))
        else:
            obs = result.obs_dict if result.obs_dict is not None else self._robot.get_observation()
            record.update(obs_to_indexed(obs, self._obs_features))

        policy_action = result.policy_action
        if policy_action is None and result.action_dict:
            policy_action = result.action_dict.get("arm_actions")
        if policy_action is not None:
            record.update(flat_array_to_indexed("policy_action", policy_action))
            # Legacy alias: old sim CSVs used normalized_action for policy output.
            record.update(flat_array_to_indexed("normalized_action", policy_action))

        task_action = getattr(self._robot.task, "action", None)
        if task_action is not None:
            record.update(flat_array_to_indexed("ema_action", task_action))
            # Legacy alias: old sim CSVs used raw_action for EMA-smoothed action.
            record.update(flat_array_to_indexed("raw_action", task_action))

        if result.processed_action is not None:
            processed_arm = result.processed_action.get("processed_arm_action")
            processed_gripper = result.processed_action.get("processed_gripper_action")
            if processed_arm is not None:
                record.update(flat_array_to_indexed("processed_arm_action", processed_arm))
            if processed_gripper is not None:
                record.update(flat_array_to_indexed("processed_gripper_action", processed_gripper))

        record.update(flatten(self._robot.task.get_log(), self._log_features))
        self._buffer.append(record)

    def save(self, save_dir: str) -> None:
        if not self._buffer:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"collected_data_{ts}.csv")
        pd.DataFrame(self._buffer).to_csv(path, index=False)
        print(f"\n[INFO] Saved {len(self._buffer)} steps → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference process manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceProcessHandle:
    proc: mp.Process
    stop_event: mp.Event

    def wait_ready(self, action_flag: RawValue) -> None:
        print("[INFO] Waiting for inference warm-up ...")
        while action_flag.value != 2:
            time.sleep(0.05)
        action_flag.value = 0
        print("[INFO] Inference process ready.")

    def shutdown(self) -> None:
        self.stop_event.set()
        self.proc.join(timeout=3)


def start_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,
    action_flag: RawValue,
    *,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device: str,
) -> InferenceProcessHandle:
    stop_event = mp.Event()
    proc = mp.Process(
        target=run_inference_process,
        args=(obs_shm, action_shm, obs_flag, action_flag, stop_event),
        kwargs=dict(
            checkpoint=checkpoint,
            cfg_path=cfg_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device_str=device,
        ),
        daemon=True,
    )
    proc.start()
    return InferenceProcessHandle(proc=proc, stop_event=stop_event)


# ──────────────────────────────────────────────────────────────────────────────
# Control loop
# ──────────────────────────────────────────────────────────────────────────────

def run_control_loop(
    robot,
    strategy: ControlStrategy,
    *,
    control_hz: float,
    logger: Optional[StepLogger] = None,
) -> None:
    """Isaac Lab DirectRLEnv.step()과 동일한 action-first 순서로 제어 루프 실행.

    타임라인:
        reset()  → obs_0 수집, 첫 inference 트리거
        t=0: action_0 = wait_inference()  →  send(action_0)  →  obs_1, trigger infer(obs_1)
        t=1: action_1 = wait_inference()  →  send(action_1)  →  obs_2, trigger infer(obs_2)
        ...

    send / wait 순서가 Isaac Lab의 apply_action → sim.step → get_observations와 1:1 대응.
    """
    dt = 1.0 / control_hz
    print(f"\n[INFO] Control loop @ {control_hz} Hz  (Ctrl+C to stop)")

    strategy.reset()

    step_idx = 0
    try:
        while True:
            t_start = time.time()

            result = strategy.step(step_idx)
            if result is None:
                break

            # strategy.step()이 반환한 action_dict를 실제로 로봇에 전송
            # (pose 모드만 send_processed_action, 나머지는 send_action)
            if isinstance(strategy, ReplayPoseStrategy):
                if result.action_dict:
                    result.processed_action = robot.send_processed_action(result.action_dict)
            else:
                result.processed_action = robot.send_action(result.action_dict)

            if logger is not None:
                logger.record(result)

            result = strategy.after_action_sent(step_idx, result)

            sleep_t = dt - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

            step_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")


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

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--raw",  action="store_true", help="Replay: CSV obs → inference")
    mode.add_argument("--pose", action="store_true", help="Replay: CSV target pose → direct send")

    args = p.parse_args()
    if (args.raw or args.pose) and not args.replay:
        p.error("--raw / --pose requires --replay")
    if args.replay and not (args.raw or args.pose):
        p.error("--replay requires --raw or --pose")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    args = parse_args()
    needs_inference = not (args.replay and args.pose)

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
    action_dim = total_dim({"arm_actions": action_features["arm_actions"]})
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    # ── Shared memory ─────────────────────────────────────────────────────────
    obs_shm    = torch.zeros(1, obs_dim).share_memory_()
    action_shm = torch.zeros(1, action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)

    # ── Inference process ─────────────────────────────────────────────────────
    infer_handle: Optional[InferenceProcessHandle] = None
    if needs_inference:
        infer_handle = start_inference_process(
            obs_shm, action_shm, obs_flag, action_flag,
            checkpoint=args.checkpoint,
            cfg_path=args.cfg,
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

    # ── CSV 로드 & 컬럼 검증 ──────────────────────────────────────────────────
    replay_data: Optional[pd.DataFrame] = None
    pose_cols: List[str] = []

    if args.replay:
        replay_data = pd.read_csv(args.replay)
        print(f"[INFO] Loaded {len(replay_data)} steps from {args.replay}")

        if args.raw:
            required = [f"obs_{i}" for i in range(obs_dim)]
        else:
            pose_cols = (
                flat_keys({"target_pos":  log_features["target_pos"]})
                + flat_keys({"target_quat": log_features["target_quat"]})
            )
            required = pose_cols

        missing = [c for c in required if c not in replay_data.columns]
        if missing:
            print(f"[ERROR] CSV missing columns: {missing}")
            if infer_handle:
                infer_handle.shutdown()
            robot.disconnect()
            return

    # ── Strategy 선택 ─────────────────────────────────────────────────────────
    obs_buf = obs_shm.numpy()[0]

    if args.replay and args.raw:
        strategy: ControlStrategy = ReplayRawStrategy(
            replay_data, obs_buf, obs_flag, action_flag,
            action_shm, obs_dim, action_dim,
            robot=robot, obs_features=obs_features,
        )
    elif args.replay and args.pose:
        strategy = ReplayPoseStrategy(replay_data, pose_cols, action_dim)
    else:
        strategy = LiveInferenceStrategy(
            robot, obs_buf, obs_features,
            obs_flag, action_flag, action_shm, action_dim,
        )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = StepLogger(robot, obs_features, log_features) if args.save_path else None

    # ── Control loop ──────────────────────────────────────────────────────────
    try:
        run_control_loop(robot, strategy, control_hz=args.control_hz, logger=logger)
    finally:
        if logger:
            logger.save(args.save_path)
        if infer_handle:
            infer_handle.shutdown()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
