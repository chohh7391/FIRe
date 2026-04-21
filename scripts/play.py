import time
import argparse
import numpy as np
import pandas as pd

# [2] 우리가 만든 로봇 통신 라이브러리
from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot

def parse_args():
    parser = argparse.ArgumentParser(description="Replay actions from CSV on Real Robot (Franka)")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--hz", type=float, default=12.0, help="Control loop rate (Hz)")
    return parser.parse_args()

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

    # 필수 컬럼(action_0 ~ action_6)이 있는지 검증
    action_cols = [f"action_{i}" for i in range(6)]
    if not all(col in df.columns for col in action_cols):
        print(f"[ERROR] CSV file must contain columns: {action_cols}")
        return

    # =================================================================
    # 2. 로봇 (ROS 2) 초기화
    # =================================================================
    print("[INFO] Connecting to Robot...")
    
    # 설정: 상대 제어(Velocity/Delta), axis_angle, 6자유도 액션
    config = FR3RobotConfig(is_relative=True, rotation_type="axis_angle", arm_action_dim=6)
    robot = FR3Robot(config)
    robot.connect()

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return

    # =================================================================
    # 3. CSV 데이터 기반 제어 루프
    # =================================================================
    sleep_time = 1.0 / args.hz
    total_steps = len(df)

    print(f"\n🚀 [INFO] Starting CSV Playback! (Total Steps: {total_steps}, Rate: {args.hz} Hz)")
    print("   - Press Ctrl+C to stop.")

    try:
        # iterrows()를 사용하여 한 줄씩 읽어서 실행
        for index, row in df.iterrows():
            t_start = time.time()

            # --- A. CSV에서 Action 값 추출 ---
            # action_0 ~ action_5: Arm Action (6-DoF)
            arm_action = np.array([
                row["action_0"],
                row["action_1"],
                row["action_2"] + 0.1, 
                row["action_3"],
                row["action_4"],
                row["action_5"]
            ], dtype=np.float32)

            # # test
            # arm_action[:] = 0.0
            # arm_action[0] = 0.1
            
            # action_6: Gripper Action (1-DoF)
            gripper_action = [-1.0]

            # --- B. C++ 액션 서버로 전송 ---
            action_dict = {
                "arm_actions": np.array([arm_action], dtype=np.float32),
                "gripper_actions": np.array(gripper_action, dtype=np.float32)
            }
            robot.send_action(action_dict)
            
            # 진행 상태 출력 (매 50 스텝마다)
            if index % 50 == 0:
                print(f"[INFO] Playing step {index}/{total_steps} ...")

            # --- C. 제어 주기 (Hz) 맞추기 ---
            elapsed_loop = time.time() - t_start
            time.sleep(max(0.0, sleep_time - elapsed_loop))

        print("\n[INFO] 🏁 CSV Playback Finished successfully!")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by User.")
    finally:
        robot.disconnect()
        print("[INFO] Robot Disconnected Safely.")

if __name__ == "__main__":
    main()