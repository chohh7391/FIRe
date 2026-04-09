from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot
import numpy as np


def main():
    print("[INFO] Connecting to Robot...")
    config = FR3RobotConfig(is_relative=True, rotation_type="axis_angle", arm_action_dim=6)
    robot = FR3Robot(config)
    robot.connect()

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return
    
    try:
        while True:

            # get observation
            obs_dict = robot.get_observation()
            print(f"obs_dict: {obs_dict}")

            # control robot
            action_dict= {
                "arm_action": np.random.random((6,)),
                "gripper_action": np.random.random((1,)),
            }
            robot.send_action(action_dict)
        
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by User.")
    finally:
        robot.disconnect()
        print("[INFO] Robot Disconnected Safely.")


if __name__ == "__main__":
    main()