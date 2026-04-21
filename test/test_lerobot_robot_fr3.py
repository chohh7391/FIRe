import cv2
import numpy as np
from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot


def main():
    print("[INFO] Connecting to Robot...")
    # 시각화를 위해 arm_action_dim 등 설정 유지
    config = FR3RobotConfig(is_relative=True, rotation_type="axis_angle", arm_action_dim=6)
    robot = FR3Robot(config)
    
    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return
    
    print("[INFO] Starting Control & Visualization Loop. Press 'q' to quit.")
    
    try:
        while True:
            # 1. Observation 데이터 획득
            # robot_state_manager를 통해 얻는 수치 정보
            obs_dict = robot.get_observation()
            print("=========== observation ===========")
            for obs_key, obs_value in obs_dict.items():
                print(f"{obs_key}: {obs_value.shape}")

            vla_obs_dict = robot.get_vla_observation()
            print("=========== vla observation ===========")
            for obs_key, obs_value in vla_obs_dict.items():
                print(f"{obs_key}: {obs_value.shape}")

            print(f"ee_pos: {obs_dict['fingertip_pos']}")
            print(f"ee_quat: {obs_dict['fingertip_quat']}")
            print(f"ee_linvel: {obs_dict['ee_linvel']}")
            print(f"ee_angvel: {obs_dict['ee_angvel']}")

            # 2. 시각화 패널 생성 (제공해주신 뷰어 로직 활용)
            panels = []
            cam_keys = ["wrist", "left", "right"]
            
            for key in cam_keys:
                obs_key = f"video.{key}_view"
                if obs_key in vla_obs_dict:
                    # lerobot 카메라는 보통 (1, H, W, 3) 또는 (H, W, 3) RGB를 반환함
                    frame = vla_obs_dict[obs_key]
                    if frame.ndim == 4:
                        frame = frame[0]
                    
                    # RGB to BGR 변환 (OpenCV 시각화용)
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    label = key
                else:
                    # 프레임이 없을 경우 검은 화면
                    bgr = np.zeros((config.cameras[key].height, config.cameras[key].width, 3), dtype=np.uint8)
                    label = f"{key} (no frame)"

                # 카메라 이름 텍스트 삽입
                cv2.putText(bgr, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                panels.append(bgr)

            # 3. 가로로 이어 붙여서 출력
            if panels:
                combined = np.hstack(panels)
                cv2.imshow("FR3 Robot Multi-Camera Viewer", combined)

            # 4. 로봇 제어 명령 전송 (Random Action 예시)
            action_dict = {
                "arm_actions": np.random.random((6,)),
                "gripper_actions": np.random.random((1,)),
            }
            robot.send_action(action_dict)

            # 5. 종료 조건 및 루프 속도 조절
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by User.")
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")
    finally:
        # 안전한 종료 처리
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Robot Disconnected Safely.")


if __name__ == "__main__":
    main()