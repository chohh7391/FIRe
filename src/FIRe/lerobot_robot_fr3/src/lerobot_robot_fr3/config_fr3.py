from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
from typing import Dict
from lerobot_ft_sensor.configuration_ft_sensor import FTSensorConfig
import os

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")


@dataclass
class FR3RobotConfig(RobotConfig):
    ros_domain_id: int = 0
    
    # --- ROS 2 토픽 및 액션 서버 이름 ---
    observation_topic: str = "/vla/observation/ee_pose"
    action_topic: str = "/vla/action/ee_pose"
    vla_action_server: str = "/controller_action_server/vla_controller"
    
    # --- VLA Action Goal 설정 ---
    model_name: str = "lerobot_vla_policy"
    control_mode: str = "effort"  # C++ 서버 환경에 맞게 "effort" 또는 "position"
    action_space: str = "task"    # Task space 제어
    
    # --- VLA Chunk 설정 ---
    is_relative: bool = False
    rotation_type: str = "axis_angle"  # "axis_angle", "euler", "quaternion", "rotation6d"
    arm_action_dim: int = 6
    
    ft_sensor: FTSensorConfig = FTSensorConfig(
        config_path="/home/home/FIRe/src/FIRe/lerobot_ft_sensor/src/lerobot_ft_sensor/config/bota_binary.json"
    )
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": ZMQCameraConfig(
                server_address="127.0.0.1", port=5555, camera_name="wrist", timeout_ms=5000,
                width=640, height=480, fps=30
            ),
            "left":  ZMQCameraConfig(
                server_address="127.0.0.1", port=5555, camera_name="left", timeout_ms=5000,
                width=640, height=480, fps=30
            ),
            "right": ZMQCameraConfig(
                server_address="127.0.0.1", port=5555, camera_name="right", timeout_ms=5000,
                width=640, height=480, fps=30
            )
        }
    )