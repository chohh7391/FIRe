import threading
import time
from typing import Any, Dict
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Header, Float64MultiArray

try:
    from cho_interfaces.action import VisionLanguageAction
    from cho_interfaces.msg import ActionChunk
except ImportError:
    raise ImportError(
        "cho_interfaces 패키지를 찾을 수 없습니다. "
        "ROS 2 워크스페이스를 source 했는지 확인하세요."
    )

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from .config_fr3 import FR3RobotConfig


class FR3Robot(Robot):
    config_class = FR3RobotConfig
    name = "fr3_vla"

    def __init__(self, config: FR3RobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self.node = None
        self.executor_thread = None
        self._is_connected = False
        
        # Action Client 관련 변수
        self._action_client = None
        self._goal_handle = None
        self._goal_accepted = False
        
        # EE Pose 관측치
        self._latest_ee_pose = []
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('lerobot_fr3_vla_client')

        # 1. Pub/Sub 생성
        self.sub_ee_pose = self.node.create_subscription(
            Float64MultiArray, self.config.observation_topic, self._ee_pose_callback, 10
        )
        self.pub_action_chunk = self.node.create_publisher(
            ActionChunk, self.config.action_topic, 10
        )

        # 2. Action Client 생성
        self._action_client = ActionClient(
            self.node, VisionLanguageAction, self.config.vla_action_server
        )

        # 3. 백그라운드 ROS 2 스레드 시작 (Action 통신을 위해 spin 필요)
        self.executor_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.executor_thread.start()

        # 4. Action Server 접속 대기
        print(f"[{self.name}] Waiting for VLA Action Server: {self.config.vla_action_server}...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            raise ConnectionError("VLA Action Server is not available.")

        # 5. Goal 전송 및 수락(Accept) 대기
        self._send_vla_goal()

        # 6. 첫 EE Pose 데이터가 들어올 때까지 대기
        print(f"[{self.name}] Waiting for first EE pose observation...")
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if len(self._latest_ee_pose) > 0:
                    break
            time.sleep(0.1)
        else:
            print("Warning: Did not receive EE pose within timeout.")

        # 카메라 연결
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        print(f"[{self.name}] VLA Client Connected and Ready!")

    def _send_vla_goal(self):
        """Action Server에 제어 권한(Goal)을 요청합니다."""
        goal_msg = VisionLanguageAction.Goal()
        goal_msg.model_name = self.config.model_name
        goal_msg.control_mode = self.config.control_mode

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_callback)

        # Goal이 수락될 때까지 블로킹 대기 (최대 5초)
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._goal_accepted:
                return
            time.sleep(0.1)
        
        raise ConnectionError("Goal was not accepted by the VLA Action Server.")

    def _goal_response_callback(self, future):
        """Goal 전송에 대한 서버의 응답을 처리합니다."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('VLA Goal rejected by server!')
            self._goal_accepted = False
            return

        self.node.get_logger().info('VLA Goal accepted! Robot control started.')
        self._goal_handle = goal_handle
        self._goal_accepted = True

    def _spin_ros(self):
        rclpy.spin(self.node)

    def _ee_pose_callback(self, msg: Float64MultiArray):
        with self._lock:
            self._latest_ee_pose = list(msg.data)

    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        obs_dict = {}
        with self._lock:
            obs_dict["ee_pose"] = np.array(self._latest_ee_pose, dtype=np.float32)

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        # Goal이 수락되지 않았으면 명령을 쏘지 않음
        if not self._goal_accepted:
            self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
            return action

        arm_action_np = np.array(action["arm_action"]).reshape(-1)
        gripper_action_np = np.array(action.get("gripper_action", [])).reshape(-1)
        chunk_size = len(arm_action_np) // self.config.arm_action_dim

        msg = ActionChunk()
        # 헤더 시간 및 프레임 명시 (Tester 참고)
        msg.header = Header(stamp=self.node.get_clock().now().to_msg(), frame_id="base_link")
        msg.action_space = self.config.action_space
        msg.relative = self.config.is_relative
        msg.rotation_type = self.config.rotation_type
        msg.chunk_size = chunk_size
        
        msg.arm_action = arm_action_np.tolist()
        msg.gripper_action = gripper_action_np.tolist()

        self.pub_action_chunk.publish(msg)
        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        for cam in self.cameras.values():
            cam.disconnect()

        # 제어 안전 종료: 활성화된 Goal이 있다면 Cancel 요청 전송
        if self._goal_handle is not None and self._goal_accepted:
            print(f"[{self.name}] Canceling VLA Action goal for safe stop...")
            cancel_future = self._goal_handle.cancel_goal_async()
            
            # 서버가 Cancel 요청을 처리할 시간을 잠깐 벌어줌
            t0 = time.time()
            while not cancel_future.done() and time.time() - t0 < 1.0:
                time.sleep(0.1)

        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        
        if self.executor_thread is not None:
            self.executor_thread.join(timeout=1.0)

        self._is_connected = False
        print(f"[{self.name}] Disconnected.")

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features = {"ee_pose": (16,)} 
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "arm_action": (self.config.arm_action_dim,),
            "gripper_action": (1,)
        }
    
    @property
    def is_calibrated(self) -> bool:
        """ROS 2 및 C++ 서버에서 캘리브레이션/초기화가 이미 완료되었다고 가정합니다."""
        return True

    def calibrate(self) -> None:
        """파이썬 레벨의 캘리브레이션은 불필요하므로 패스합니다."""
        pass

    def configure(self) -> None:
        """파이썬 레벨의 추가 하드웨어 설정은 불필요하므로 패스합니다."""
        pass