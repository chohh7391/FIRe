import argparse
import ctypes
import time
import torch
import torch.multiprocessing as mp
from multiprocessing import RawValue
import numpy as np
import cv2
import pandas as pd
import datetime
import os

OBS_KEYS_AND_DIM = [
    ("fingertip_pos_rel_fixed", 3),
    ("fingertip_quat", 4),
    ("ee_linvel", 3),
    ("ee_angvel", 3),
    ("prev_actions", 6),
]


def write_obs_to_shm(obs_dict: dict, obs_buf: np.ndarray) -> None:
    """obs_dict를 shared memory numpy buffer에 직접 씀.
    torch 변환 없이 순수 numpy slice write → GIL 최소화."""
    offset = 0
    for key, dim in OBS_KEYS_AND_DIM:
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg",        type=str, required=True)
    parser.add_argument("--obs_dim",    type=int, default=19)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--visualize",  action="store_true", default=False)
    parser.add_argument("--use_sim_time", action="store_true")
    parser.add_argument("--replay",     type=str, default=None, help="Path to replay CSV file")

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


def load_replay_target_pose(replay_data: pd.DataFrame, step_idx: int) -> np.ndarray:
    """CSV에서 target_pos_x/y/z + target_quat_w/x/y/z 컬럼 (총 7개) 을 numpy array로 반환.

    반환 shape: (7,) → [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    """
    row = replay_data.iloc[step_idx]
    pose_cols = [
        "target_pos_x", "target_pos_y", "target_pos_z",
        "target_quat_w", "target_quat_x", "target_quat_y", "target_quat_z",
    ]
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
    # 1. Shared memory + spin-wait 플래그 + Inference 프로세스 시작
    # =================================================================
    obs_shm    = torch.zeros(1, args.obs_dim).share_memory_()
    action_shm = torch.zeros(1, args.action_dim).share_memory_()
    obs_flag    = RawValue(ctypes.c_int32, 0)
    action_flag = RawValue(ctypes.c_int32, 0)
    stop_event  = mp.Event()

    infer_proc = None
    if needs_inference:
        infer_proc = mp.Process(
            target=run_inference_process,
            args=(obs_shm, action_shm, obs_flag, action_flag, stop_event,
                  args.checkpoint, args.cfg, args.obs_dim, args.action_dim, args.device),
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
    # 2. 로봇 연결
    # =================================================================
    print("[INFO] Connecting to Robot...")
    config = FR3RobotConfig(use_sim_time=args.use_sim_time, is_relative=False, rotation_type="quaternion", arm_action_dim=6)
    robot = FR3Robot(config)

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
    # 3. CSV 데이터 로드 및 컬럼 검증
    # =================================================================
    replay_data = None
    if args.replay:
        print(f"[INFO] Loading CSV: {args.replay}")
        replay_data = pd.read_csv(args.replay)
        print(f"[INFO] Loaded {len(replay_data)} steps from CSV.")

        if args.raw:
            # --raw: obs_0 ~ obs_{obs_dim-1} 컬럼 검증
            expected_cols = [f"obs_{i}" for i in range(args.obs_dim)]
            mode_label = f"obs_0 ~ obs_{args.obs_dim - 1}"
        else:  # args.pose
            # --pose: target_pos / target_quat 컬럼 검증
            expected_cols = [
                "target_pos_x", "target_pos_y", "target_pos_z",
                "target_quat_w", "target_quat_x", "target_quat_y", "target_quat_z",
            ]
            mode_label = "target_pos_*, target_quat_*"

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

    log_data = []

    print(f"\n[INFO] Starting control loop at {args.control_hz} Hz. Ctrl+C or 'q' to stop.")

    try:
        while True:
            t_start = time.time()

            # --- A. Observation ---
            t0 = time.perf_counter()

            if args.replay and args.raw:
                # -------------------------------------------------------
                # Replay --raw 모드: CSV의 obs_0~obs_N을 직접 shared memory에 씀
                # write_obs_to_shm을 거치지 않고 바로 obs_buf에 복사
                # -------------------------------------------------------
                if step_idx >= len(replay_data):
                    print("\n[INFO] Replay CSV finished.")
                    break

                obs_buf[:] = load_replay_obs(replay_data, step_idx, args.obs_dim)
                obs_dict = None  # replay --raw 모드에서는 obs_dict 불필요
                print(f"[REPLAY-RAW] step {step_idx + 1}/{len(replay_data)}", end='\r')

            elif args.replay and args.pose:
                # -------------------------------------------------------
                # Replay --pose 모드: obs 없음. inference도 안 함.
                # CSV의 target_pos/quat을 그대로 send_action에 전달한다.
                # -------------------------------------------------------
                if step_idx >= len(replay_data):
                    print("\n[INFO] Replay CSV finished.")
                    break

                obs_dict = None
                print(f"[REPLAY-POSE] step {step_idx + 1}/{len(replay_data)}", end='\r')

            else:
                # -------------------------------------------------------
                # 실시간 모드: 로봇에서 observation 수집
                # -------------------------------------------------------
                obs_dict = robot.get_observation()

            t1 = time.perf_counter()

            # # 시각화 / VLA state 저장용 (replay 모드에서도 실제 로봇 상태 기록)
            # vla_obs_dict = robot.get_vla_observation()

            # --- B. Obs → shared memory (실시간 모드만 write_obs_to_shm 사용) ---
            if not args.replay:
                write_obs_to_shm(obs_dict, obs_buf)
            # replay --raw 모드는 이미 A 단계에서 obs_buf에 직접 썼으므로 skip
            # replay --pose 모드는 obs / inference 자체가 없으므로 skip
            t2 = time.perf_counter()

            # --- C. Action 결정 ---
            #   * 실시간 / replay --raw: inference 프로세스를 통해 action 계산
            #   * replay --pose       : CSV에서 target_pos/quat을 직접 로드
            if args.replay and args.pose:
                # CSV의 target pose를 그대로 사용. inference 우회.
                target_pose_np = load_replay_target_pose(replay_data, step_idx)
                action_np = None  # 아래 D 단계에서 분기 처리
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
                    "gripper_actions": np.array([0.0], dtype=np.float32),
                }
                robot.send_action(action_dict, is_raw_action=False)
            else:
                # 실시간 / replay --raw 공통: model이 만든 raw 7-vec을 처리해 보낸다.
                arm_action_np = action_np[:6].copy()
                action_dict = {
                    "arm_actions": arm_action_np,
                    "gripper_actions": np.array([0.0], dtype=np.float32),
                }
                robot.send_action(action_dict)
            t4 = time.perf_counter()

            # --- 데이터 로깅 ---
            if args.save_path:
                record_dict = robot.get_observation()
                record = {
                    "fingertip_pos_rel_fixed_x": record_dict["fingertip_pos_rel_fixed"][0],
                    "fingertip_pos_rel_fixed_y": record_dict["fingertip_pos_rel_fixed"][1],
                    "fingertip_pos_rel_fixed_z": record_dict["fingertip_pos_rel_fixed"][2],
                    "fingertip_quat_w": record_dict["fingertip_quat"][0],
                    "fingertip_quat_x": record_dict["fingertip_quat"][1],
                    "fingertip_quat_y": record_dict["fingertip_quat"][2],
                    "fingertip_quat_z": record_dict["fingertip_quat"][3],
                    "ee_linvel_x": record_dict["ee_linvel"][0],
                    "ee_linvel_y": record_dict["ee_linvel"][1],
                    "ee_linvel_z": record_dict["ee_linvel"][2],
                    "ee_angvel_x": record_dict["ee_angvel"][0],
                    "ee_angvel_y": record_dict["ee_angvel"][1],
                    "ee_angvel_z": record_dict["ee_angvel"][2],
                    "prev_actions_0": record_dict["prev_actions"][0],
                    "prev_actions_1": record_dict["prev_actions"][1],
                    "prev_actions_2": record_dict["prev_actions"][2],
                    "prev_actions_3": record_dict["prev_actions"][3],
                    "prev_actions_4": record_dict["prev_actions"][4],
                    "prev_actions_5": record_dict["prev_actions"][5],

                    "ee_pos_x": robot.robot_state_manager.ee_pos[0],
                    "ee_pos_y": robot.robot_state_manager.ee_pos[1],
                    "ee_pos_z": robot.robot_state_manager.ee_pos[2],
                    "ee_quat_w": robot.robot_state_manager.ee_quat[0],
                    "ee_quat_x": robot.robot_state_manager.ee_quat[1],
                    "ee_quat_y": robot.robot_state_manager.ee_quat[2],
                    "ee_quat_z": robot.robot_state_manager.ee_quat[3],

                    "target_pos_x": robot.processed_arm_action[0],
                    "target_pos_y": robot.processed_arm_action[1],
                    "target_pos_z": robot.processed_arm_action[2],
                    "target_quat_w": robot.processed_arm_action[6],
                    "target_quat_x": robot.processed_arm_action[3],
                    "target_quat_y": robot.processed_arm_action[4],
                    "target_quat_z": robot.processed_arm_action[5],
                }

                log_data.append(record)

            # # VLA State
            # if "state.eef_position" in vla_obs_dict:
            #     for i in range(3): record[f"vla/state.eef_position_{i}"] = float(vla_obs_dict["state.eef_position"][i])
            # if "state.eef_quaternion" in vla_obs_dict:
            #     for i in range(4): record[f"vla/state.eef_quaternion_{i}"] = float(vla_obs_dict["state.eef_quaternion"][i])
            # if "state.gripper_qpos" in vla_obs_dict:
            #     for i in range(2): record[f"vla/state.gripper_qpos_{i}"] = float(vla_obs_dict["state.gripper_qpos"][i])

            # # --- E. 카메라 시각화 ---
            # if args.visualize:
            #     panels = []
            #     for key in ["wrist", "left", "right"]:
            #         obs_key = f"video.{key}_view"
            #         if obs_key in vla_obs_dict:
            #             frame = vla_obs_dict[obs_key]
            #             if frame.ndim == 4: frame = frame[0]
            #             bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #             label = key
            #         else:
            #             bgr = np.zeros((config.cameras[key].height, config.cameras[key].width, 3), dtype=np.uint8)
            #             label = f"{key} (no frame)"
            #         cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            #         panels.append(bgr)

            #     cv2.imshow("FR3 Multi-Camera Viewer", np.hstack(panels))
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         print("\n[INFO] 'q' pressed. Exiting.")
            #         break

            # --- F. 주기 유지 ---
            elapsed = time.time() - t_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            print(
                f"[TIMING] obs={1000*(t1-t0):.1f}ms | build={1000*(t2-t1):.1f}ms | "
                f"infer={1000*(t3-t2):.1f}ms | send={1000*(t4-t3):.1f}ms | total={1000*(t4-t0):.1f}ms",
                end='\r'
            )

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