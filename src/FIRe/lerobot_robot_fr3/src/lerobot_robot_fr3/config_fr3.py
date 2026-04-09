from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig


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
    rotation_type: str = "rotation6d"  # "axis_angle", "euler", "quaternion", "rotation6d"
    arm_action_dim: int = 9  
    
    # cameras: dict[str, CameraConfig] = field(
    #     default_factory={
    #         "wrist": ZMQCameraConfig(server_address=SAM3_HOST, port=5600, camera_name="wrist"),
    #         "left":  ZMQCameraConfig(server_address=SAM3_HOST, port=5601, camera_name="left"),
    #         "right": ZMQCameraConfig(server_address=SAM3_HOST, port=5602, camera_name="right"),
    #     }
    # )