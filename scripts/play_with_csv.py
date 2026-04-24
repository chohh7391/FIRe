import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot


def parse_args():
    parser = argparse.ArgumentParser(description="Replay actions from CSV on Real Robot (Franka)")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--hz",       type=float, default=15.0, help="Control loop rate (Hz)")
    parser.add_argument("--log",      action="store_true",      help="Enable logging")
    parser.add_argument("--log_dir",  type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--use_sim_time", action="store_true", help="Use simulation time")
    return parser.parse_args()


def flatten_obs(obs_dict: dict, prefix: str = "") -> dict:
    """obs_dict의 ndarray를 컬럼별로 flatten해서 {key_0, key_1, ...} dict 반환."""
    row = {}
    for k, v in obs_dict.items():
        v = np.array(v).flatten()
        col_prefix = f"{prefix}{k}" if prefix else k
        if v.size == 1:
            row[col_prefix] = v[0]
        else:
            for i, val in enumerate(v):
                row[f"{col_prefix}_{i}"] = val
    return row


def main():
    args = parse_args()

    # =================================================================
    # 1. CSV 데이터 로드
    # =================================================================
    print(f"[INFO] Loading CSV data from: {args.csv_path}")
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    action_cols = [f"processed_action_{i}" for i in range(6)]
    if not all(col in df.columns for col in action_cols):
        print(f"[ERROR] CSV must contain columns: {action_cols}")
        return

    # =================================================================
    # 2. 로봇 연결
    # =================================================================
    print("[INFO] Connecting to Robot...")
    # config = FR3RobotConfig(use_sim_time=args.use_sim_time, is_relative=False, rotation_type="axis_angle", arm_action_dim=6)
    config = FR3RobotConfig(use_sim_time=args.use_sim_time, is_relative=False, rotation_type="quaternion", arm_action_dim=7)
    robot  = FR3Robot(config)
    robot.connect()

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return

    # =================================================================
    # 3. 로그 준비
    # =================================================================
    log_rows = []   # 루프 중 메모리에 누적, 종료 시 한 번에 저장
    if args.log:
        log_dir  = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path  = log_dir / f"replay_log_{timestamp}.csv"
        print(f"[INFO] Logging enabled → {log_path}")

    # =================================================================
    # 4. 제어 루프
    # =================================================================
    dt = 1.0 / args.hz
    total_steps = len(df)
    print(f"\n[INFO] Starting CSV Playback! (Steps: {total_steps}, Rate: {args.hz} Hz)")

    try:
        for index, row in df.iterrows():
            t_start = time.time()

            # --- A. Observation ---
            obs_dict     = robot.get_observation()
            vla_obs_dict = robot.get_vla_observation()

            # --- B. Action 구성 ---
            arm_action_np = np.array([
                row["raw_action_0"],
                row["raw_action_1"],
                row["raw_action_2"],
                row["raw_action_3"],
                row["raw_action_4"],
                row["raw_action_5"],
            ], dtype=np.float32)

            # arm_action_np = np.array([
            #     row["processed_action_0"],
            #     row["processed_action_1"],
            #     row["processed_action_2"],
            #     row["processed_action_3"],
            #     row["processed_action_4"],
            #     row["processed_action_5"],
            # ], dtype=np.float32)

            gripper_action = np.array([-1.0], dtype=np.float32)

            action_dict = {
                "arm_actions": arm_action_np,
                "success_prediction": np.ones(1, dtype=np.float32),
                "gripper_actions": gripper_action
            }
            robot.send_action(action_dict, is_raw_action=True)

            # --- C. 로그 기록 ---
            if args.log:
                log_row = {"step": index, "timestamp": time.time()}

                # action
                for i, v in enumerate(arm_action_np):
                    log_row[f"action_{i}"] = v
                log_row["gripper_action"] = gripper_action[0]

                # obs (카메라 제외, 수치 데이터만)
                log_row.update(flatten_obs(obs_dict,     prefix="obs/"))
                # vla_obs에서 카메라 프레임은 용량이 크므로 수치 state만 저장
                vla_numeric = {k: v for k, v in vla_obs_dict.items()
                               if not k.startswith("video.")}
                log_row.update(flatten_obs(vla_numeric, prefix="vla/"))

                log_rows.append(log_row)

            # --- D. 진행 출력 ---
            elapsed = time.time() - t_start
            sleep_t = dt - elapsed
            time.sleep(max(0.0, sleep_t))
            elapsed = time.time() - t_start
            print(f"[INFO] Step {index}/{total_steps}  loop={elapsed*1000:.1f}ms")

            if index == 50:
                if args.log and log_rows:
                    pd.DataFrame(log_rows).to_csv(log_path, index=False)
                    print(f"[INFO] Log saved → {log_path}  ({len(log_rows)} steps)")
                break

        print("\n[INFO] CSV Playback Finished.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        # 로그 저장 (정상 종료 / Ctrl+C / 에러 모두 저장)
        if args.log and log_rows:
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
            print(f"[INFO] Log saved → {log_path}  ({len(log_rows)} steps)")

        robot.disconnect()
        print("[INFO] Robot disconnected safely.")


if __name__ == "__main__":
    main()