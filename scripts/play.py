import argparse
import ctypes
import time
import torch
import torch.multiprocessing as mp
from multiprocessing import RawValue
from typing import Dict, Tuple, List
import numpy as np
import cv2
import pandas as pd
import datetime
import os


# =================================================================
# Feature helpers
# observation_features / log_features 같은 Dict[str, Tuple[int, ...]]
# 정의를 받아서 평탄화/컬럼명 생성/shared memory write를 일반화한다.
# =================================================================
def features_total_dim(features: Dict[str, Tuple[int, ...]]) -> int:
    """features의 모든 항목 dim 합."""
    return sum(int(np.prod(shape)) for shape in features.values())


def features_to_flat_keys(features: Dict[str, Tuple[int, ...]]) -> List[str]:
    """features 정의로부터 평탄화된 컬럼 이름 리스트를 만든다.
    예) {"ee_pos": (3,)} → ["ee_pos_0", "ee_pos_1", "ee_pos_2"]
    """
    keys: List[str] = []
    for name, shape in features.items():
        dim = int(np.prod(shape))
        for i in range(dim):
            keys.append(f"{name}_{i}")
    return keys


def flatten_features(
    data_dict: Dict[str, np.ndarray],
    features: Dict[str, Tuple[int, ...]],
) -> Dict[str, float]:
    """features 정의에 맞춰 data_dict를 평탄화한 dict로 변환."""
    flat: Dict[str, float] = {}
    for key, shape in features.items():
        arr = np.asarray(data_dict[key], dtype=np.float32).ravel()
        dim = int(np.prod(shape))
        if arr.size < dim:
            raise ValueError(
                f"'{key}' has {arr.size} elements, expected {dim}"
            )
        for i in range(dim):
            flat[f"{key}_{i}"] = float(arr[i])
    return flat


def flatten_obs_to_indexed(
    obs_dict: Dict[str, np.ndarray],
    obs_features: Dict[str, Tuple[int, ...]],
) -> Dict[str, float]:
    """obs_features 순서대로 obs_dict를 이어 붙여 obs_0~N 형태로 반환.
    --raw replay에서 읽는 컬럼명과 일치하도록 obs_? 인덱스로 저장한다."""
    flat: Dict[str, float] = {}
    idx = 0
    for key, shape in obs_features.items():
        arr = np.asarray(obs_dict[key], dtype=np.float32).ravel()
        dim = int(np.prod(shape))
        for i in range(dim):
            flat[f"obs_{idx}"] = float(arr[i])
            idx += 1
    return flat


def write_obs_to_shm(
    obs_dict: dict,
    obs_buf: np.ndarray,
    obs_features: Dict[str, Tuple[int, ...]],
) -> None:
    """obs_features 정의 순서대로 obs_dict 값을 shared memory에 직접 쓴다.
    torch 변환 없이 순수 numpy slice write → GIL 최소화."""
    offset = 0
    for key, shape in obs_features.items():
        dim = int(np.prod(shape))
        val = np.asarray(obs_dict[key], dtype=np.float32).ravel()
        obs_buf[offset:offset + dim] = val[:dim]
        offset += dim


# =================================================================
# Inference 프로세스 (프로세스 B)
# ROS 스레드 없음 → GIL 경합 없음
# spin-wait 플래그로 OS 스케줄링 레이턴시 제거
# =================================================================
def run_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,       # 0=idle, 1=obs ready
    action_flag: RawValue,    # 0=idle, 1=action ready, 2=warm-up done
    stop_event: mp.Event,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device_str: str,
):
    import yaml
    import numpy as np
    import torch
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    with open(cfg_path, "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["device"] = str(device)
    agent_cfg["params"]["config"]["num_actors"] = 1

    class DummyEnv:
        def __init__(self):
            self.observation_space = type("S", (), {"shape": (obs_dim,)})()
            self.action_space = type("S", (), {
                "shape": (action_dim,),
                "high": np.ones(action_dim, dtype=np.float32),
                "low": -np.ones(action_dim, dtype=np.float32),
            })()
            self.num_envs = 1
        def reset(self): return torch.zeros(1, obs_dim, device=device)
        def step(self, _): return torch.zeros(1, obs_dim, device=device), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}

    dummy_env = DummyEnv()
    vecenv.register("IsaacRlgWrapper", lambda _config_name, _num_actors, **_kw: dummy_env)
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_kw: dummy_env})

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()
    if agent.is_rnn:
        agent.init_rnn()

    model_device = next(agent.model.parameters()).device

    # Warm-up
    print("[INFER PROC] Warming up...")
    dummy = torch.ones(1, obs_dim, device=device)
    obs_t = agent.obs_to_torch(dummy).to(model_device)
    with torch.inference_mode():
        for _ in range(50):
            agent.get_action(obs_t, is_deterministic=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[INFER PROC] Ready. model_device={model_device}")
    action_flag.value = 2  # warm-up 완료 신호

    # Inference 루프 (spin-wait)
    while not stop_event.is_set():
        # obs가 준비될 때까지 spin
        if obs_flag.value == 0:
            continue
        obs_flag.value = 0

        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs_shm.to(device)).to(model_device)
            rl_action = agent.get_action(obs_t, is_deterministic=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        action_shm.copy_(rl_action.cpu())
        action_flag.value = 1  # action 준비 완료

    print("[INFER PROC] Stopped.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="peg_insert")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg",        type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--visualize",  action="store_true", default=False)
    parser.add_argument("--use_sim_time", action="store_true")
    parser.add_argument("--replay",     type=str, default=None, help="Path to replay CSV file")
    parser.add_argument("--use_cameras", action="store_true", help="Can use camera sensors to save resources")
    parser.add_argument("--use_ft_sensor", action="store_true", help="Can use FT sensor")

    # Replay 모드 선택 (--replay와 함께 사용; 둘 중 하나만 지정 가능)
    replay_mode = parser.add_mutually_exclusive_group()
    replay_mode.add_argument(
        "--raw",
        action="store_true",
        help="Replay 모드: CSV의 obs를 사용해 model inference → send_action(raw=True)",
    )
    replay_mode.add_argument(
        "--pose",
        action="store_true",
        help="Replay 모드: CSV의 target_pos/quat을 직접 사용 → send_action(raw=False)",
    )

    parser.add_argument("--save_path", type=str, help="Path to save collected data")
    args = parser.parse_args()

    # --raw / --pose 는 --replay와 함께만 의미가 있음.
    if (args.raw or args.pose) and not args.replay:
        parser.error("--raw / --pose 는 --replay 와 함께 사용해야 합니다.")
    if args.replay and not (args.raw or args.pose):
        parser.error("--replay 사용 시 --raw 또는 --pose 중 하나를 반드시 지정해야 합니다.")

    return args


def load_replay_obs(replay_data: pd.DataFrame, step_idx: int, obs_dim: int) -> np.ndarray:
    """CSV에서 obs_0 ~ obs_{obs_dim-1} 컬럼을 읽어 numpy array로 반환."""
    row = replay_data.iloc[step_idx]
    obs_cols = [f"obs_{i}" for i in range(obs_dim)]

    missing = [c for c in obs_cols if c not in replay_data.columns]
    if missing:
        raise KeyError(f"[REPLAY] CSV에 다음 컬럼이 없습니다: {missing}")

    return row[obs_cols].to_numpy(dtype=np.float32)


def load_replay_target_pose(
    replay_data: pd.DataFrame,
    step_idx: int,
    pose_cols: List[str],
) -> np.ndarray:
    """CSV에서 target_pos_* + target_quat_* 컬럼(총 7개)을 numpy array로 반환.

    반환 shape: (7,) → [pos_0, pos_1, pos_2, quat_0, quat_1, quat_2, quat_3]
    quat 순서는 task의 log_features 정의(quat = [w, x, y, z])를 따른다.
    """
    row = replay_data.iloc[step_idx]
    missing = [c for c in pose_cols if c not in replay_data.columns]
    if missing:
        raise KeyError(f"[REPLAY] CSV에 다음 컬럼이 없습니다: {missing}")

    return row[pose_cols].to_numpy(dtype=np.float32)


def main():
    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    args = parse_args()

    # --pose 모드는 inference 프로세스가 필요 없다.
    needs_inference = not (args.replay and args.pose)

    # =================================================================
    # 1. Robot 객체 생성 → connect() 전에 task features로 dim 계산
    #    (task는 __init__에서 생성되므로 connect() 없이도 접근 가능)
    # =================================================================
    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time, is_relative=False, rotation_type="quaternion",
        use_cameras=args.use_cameras, use_ft_sensor=args.use_ft_sensor
    )
    robot = FR3Robot(config, task_name=args.task)

    obs_features   = robot.task.observation_features
    action_features = robot.task.action_features
    log_features   = robot.task.log_features

    obs_dim    = features_total_dim(obs_features)
    # action_features는 arm_actions + gripper_actions를 포함하므로 arm만 사용
    action_dim = features_total_dim({"arm_actions": action_features["arm_actions"]})

    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim} (from task.features)")

    # =================================================================
    # 2. Shared memory + spin-wait 플래그 + Inference 프로세스 시작
    # =================================================================
    obs_shm    = torch.zeros(1, obs_dim).share_memory_()
    action_shm = torch.zeros(1, action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)
    stop_event  = mp.Event()

    infer_proc = None
    if needs_inference:
        infer_proc = mp.Process(
            target=run_inference_process,
            args=(obs_shm, action_shm, obs_flag, action_flag, stop_event,
                  args.checkpoint, args.cfg, obs_dim, action_dim, args.device),
            daemon=True,
        )
        infer_proc.start()

        print("[INFO] Waiting for inference process to warm up...")
        while action_flag.value != 2:
            time.sleep(0.1)
        action_flag.value = 0
        print("[INFO] Inference process ready.")
    else:
        print("[INFO] --replay --pose 모드: inference 프로세스를 시작하지 않습니다.")

    # =================================================================
    # 3. 로봇 연결
    # =================================================================
    print("[INFO] Connecting to Robot...")

    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        stop_event.set()
        if infer_proc is not None:
            infer_proc.join(timeout=3)
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        stop_event.set()
        if infer_proc is not None:
            infer_proc.join(timeout=3)
        return

    # =================================================================
    # 4. CSV 데이터 로드 및 컬럼 검증
    # =================================================================
    replay_data = None
    # --pose에서 사용할 target pose 컬럼명 (log_features 기반)
    pose_cols_for_replay: List[str] = (
        features_to_flat_keys({"target_pos": log_features["target_pos"]})
        + features_to_flat_keys({"target_quat": log_features["target_quat"]})
    ) if ("target_pos" in log_features and "target_quat" in log_features) else []

    if args.replay:
        print(f"[INFO] Loading CSV: {args.replay}")
        replay_data = pd.read_csv(args.replay)
        print(f"[INFO] Loaded {len(replay_data)} steps from CSV.")

        if args.raw:
            # obs 컬럼은 observation_features 평탄화 키와 동일
            expected_cols = [f"obs_{i}" for i in range(obs_dim)]
            mode_label = f"obs_0 ~ obs_{obs_dim - 1}"
        else:  # args.pose
            if not pose_cols_for_replay:
                print(
                    "[ERROR] --pose 모드는 task.log_features에 "
                    "'target_pos'와 'target_quat'가 필요합니다."
                )
                stop_event.set()
                if infer_proc is not None:
                    infer_proc.join(timeout=3)
                return
            expected_cols = pose_cols_for_replay
            mode_label = ", ".join(expected_cols)

        missing_cols = [c for c in expected_cols if c not in replay_data.columns]
        if missing_cols:
            print(f"[ERROR] CSV에 필요한 컬럼이 없습니다: {missing_cols}")
            stop_event.set()
            if infer_proc is not None:
                infer_proc.join(timeout=3)
            return
        print(f"[INFO] {mode_label} 컬럼 확인 완료.")

    # =================================================================
    # 4. Control Loop 및 데이터 로깅 준비
    # =================================================================
    dt = 1.0 / args.control_hz
    step_idx = 0
    obs_buf = obs_shm.numpy()[0]

    log_data: List[dict] = []

    print(f"\n[INFO] Starting control loop at {args.control_hz} Hz. Ctrl+C to stop.")

    try:
        while True:
            t_start = time.time()

            # --- A. Observation ---
            t0 = time.perf_counter()

            if args.replay and args.raw:
                # Replay --raw: CSV의 obs_0~obs_N을 직접 shared memory에 씀
                if step_idx >= len(replay_data):
                    print("\n[INFO] Replay CSV finished.")
                    break

                obs_buf[:] = load_replay_obs(replay_data, step_idx, obs_dim)
                obs_dict = None
                print(f"[REPLAY-RAW] step {step_idx + 1}/{len(replay_data)}", end='\r')

            elif args.replay and args.pose:
                # Replay --pose: obs 없음. inference도 안 함.
                if step_idx >= len(replay_data):
                    print("\n[INFO] Replay CSV finished.")
                    break

                obs_dict = None
                print(f"[REPLAY-POSE] step {step_idx + 1}/{len(replay_data)}", end='\r')

            else:
                # 실시간 모드: 로봇에서 observation 수집
                obs_dict = robot.get_observation()

            t1 = time.perf_counter()

            # --- B. Obs → shared memory (실시간 모드만) ---
            if not args.replay:
                write_obs_to_shm(obs_dict, obs_buf, obs_features)
            # replay --raw 모드는 A 단계에서 obs_buf에 직접 썼으므로 skip
            # replay --pose 모드는 obs / inference 자체가 없으므로 skip
            t2 = time.perf_counter()

            # --- C. Action 결정 ---
            if args.replay and args.pose:
                target_pose_np = load_replay_target_pose(
                    replay_data, step_idx, pose_cols_for_replay
                )
                action_np = None
            else:
                obs_flag.value = 1
                while action_flag.value == 0:
                    pass
                action_flag.value = 0
                action_np = action_shm.numpy().flatten()
                target_pose_np = None
            t3 = time.perf_counter()

            # --- D. Action 전송 ---
            if args.replay and args.pose:
                action_dict = {
                    "arm_actions": target_pose_np.astype(np.float32),
                    "gripper_actions": np.array([-1.0], dtype=np.float32),
                }
                # directly controlled by target pose from csv
                processed_action_dict = robot.send_processed_action(action_dict)
            else:
                arm_action_np = action_np[:action_dim].copy()
                action_dict = {
                    "arm_actions": arm_action_np,
                    "gripper_actions": np.array([-1.0], dtype=np.float32),
                }
                # controlled by action related to observation from csv
                processed_action_dict = robot.send_action(action_dict)
            t4 = time.perf_counter()

            # --- E. 데이터 로깅 (features 기반 자동화) ---
            if args.save_path:
                # obs 부분: 실시간 모드면 이미 받은 obs_dict 재사용,
                #          replay 모드면 실제 로봇 상태로 새로 받아온다.
                obs_for_log = obs_dict if obs_dict is not None else robot.get_observation()
                record = flatten_obs_to_indexed(obs_for_log, obs_features)

                # log 부분: task.get_log() 에 위임
                log_dict = robot.task.get_log()
                record.update(flatten_features(log_dict, log_features))

                log_data.append(record)

            # --- F. 주기 유지 ---
            elapsed = time.time() - t_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            step_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        if log_data and args.save_path:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(args.save_path, f"collected_data_{current_time}.csv")
            df = pd.DataFrame(log_data)
            df.to_csv(save_path, index=False)
            print(f"\n[INFO] Successfully saved {len(df)} steps to {save_path}")

        stop_event.set()
        if infer_proc is not None:
            infer_proc.join(timeout=3)
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()
        print("[INFO] Robot disconnected safely.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()