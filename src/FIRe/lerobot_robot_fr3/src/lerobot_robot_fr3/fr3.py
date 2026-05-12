import threading
import time
from typing import Any, Dict, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Header, Float64MultiArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger
from rclpy.parameter import Parameter

try:
    from cho_interfaces.action import VisionLanguageAction
    from cho_interfaces.msg import ActionChunk
except ImportError:
    raise ImportError(
        "cho_interfaces package is not found. Please make sure it is built and sourced properly. "
        "Please source your ROS 2 workspace."
    )

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.processor import RobotAction, RobotObservation
from .config_fr3 import FR3RobotConfig
from .utils import RobotStateManager
from .utils.math_utils import _wxyz_to_xyzw
from .tasks import Task


class FR3Robot(Robot):
    config_class = FR3RobotConfig
    name = "fr3_vla"

    def __init__(self, config: FR3RobotConfig, task: str):
        super().__init__(config)
        self.config = config
        if task == "factory-peg_insert":
            from .tasks.factory.factory import Factory
            self.task = Factory(name="peg_insert")
        elif task == "factory-gear_mesh":
            from .tasks.factory.factory import Factory
            self.task = Factory(name="gear_mesh")
        elif task == "factory-nut_thread":
            from .tasks.factory.factory import Factory
            self.task = Factory(name="nut_thread")
        elif task == "reach":
            from .tasks.reach.reach import Reach
            self.task = Reach(name="reach")
        else:
            raise ValueError(f"Unknown task name: {task}")

        self.node = None
        self.executor_thread = None
        self._is_connected = False
        
        # Action Client 관련 변수
        self._action_client = None
        self._goal_handle = None
        self._goal_accepted = False
        
        # update robot state & can get robot state from this
        self.robot_state_manager = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        if not rclpy.ok():
            rclpy.init()
        
        self.node = rclpy.create_node(
            'lerobot_fr3_vla_client',
            parameter_overrides=[
                Parameter(
                    'use_sim_time',
                    Parameter.Type.BOOL,
                    self.config.use_sim_time,
                )
            ],
        )
        self.node.get_logger().info(
            f"Clock mode: {'sim_time' if self.config.use_sim_time else 'real_time'}"
        )

        self.action_cb_group = ReentrantCallbackGroup()

        self.robot_state_manager = RobotStateManager(self.node)
        self.task.allocate_robot(self.robot_state_manager)

        self.pub_action_chunk = self.node.create_publisher(
            ActionChunk, self.config.action_topic, 10
        )

        # 2. Action Client 생성
        self._action_client = ActionClient(
            self.node, VisionLanguageAction, self.config.vla_action_server,
            callback_group=self.action_cb_group
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

        print(f"[{self.name}] Waiting for first EE pose observation...")
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            # 초기 Pose 객체는 0으로 채워져 있으므로, 0이 아닌 값이 들어왔다면 TF가 업데이트된 것
            if np.any(self.robot_state_manager.ee_pose != 0.0):
                break
            time.sleep(0.1)
        else:
            print("Warning: Did not receive EE pose (TF transform) within timeout.")

        self.task.reset()

        self._is_connected = True
        print(f"[{self.name}] VLA Client Connected and Ready!")

    def _send_vla_goal(self):
        """Action Server에 제어 권한(Goal)을 요청합니다."""
        goal_msg = VisionLanguageAction.Goal()
        goal_msg.model_name = self.config.model_name
        goal_msg.inference_frequency = self.config.inference_frequency

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_callback)

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
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(self.node)
        try:
            executor.spin()
        finally:
            executor.shutdown()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")
        
        obs_dict = self.task.get_observation()

        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        if not self._goal_accepted:
            self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
            return action
        
        arm_action = self.task.get_arm_action(action)
        gripper_action = self.task.get_gripper_action(action)

        processed_arm_action, processed_gripper_action = self.task.process_action(arm_action, gripper_action)

        self.apply_action(processed_arm_action, processed_gripper_action)

        processed_action = {
            "processed_arm_action": processed_arm_action,
            "processed_gripper_action": processed_gripper_action,
        }

        return processed_action
    
    def send_processed_action(self, action:RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        if not self._goal_accepted:
            self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
            return action
        
        processed_arm_action = self.task.get_arm_action(action)
        processed_gripper_action = self.task.get_gripper_action(action)

        self.apply_action(processed_arm_action, processed_gripper_action)

        processed_action = {
            "processed_arm_action": processed_arm_action,
            "processed_gripper_action": processed_gripper_action,
        }

        return processed_action
        


    def apply_action(
        self, processed_arm_action: np.ndarray, processed_gripper_action: np.ndarray = None
    ) -> None:
        msg = self.wrap_action_to_msg(processed_arm_action, processed_gripper_action)
        # send action chunk message to VLA Action Server
        self.pub_action_chunk.publish(msg)

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        print(f"[{self.name}] Sending Task Success signal to VLA Action Server...")
        trigger_client = self.node.create_client(Trigger, '/vla/trigger_success')
        
        # 서비스가 준비되었는지 짧게 대기
        if trigger_client.wait_for_service(timeout_sec=2.0):
            req = Trigger.Request()
            future = trigger_client.call_async(req)
            
            # 비동기 완료 대기 (최대 1초)
            t0 = time.time()
            while not future.done() and time.time() - t0 < 1.0:
                time.sleep(0.1)
                
            if future.done():
                try:
                    response = future.result()
                    print(f"[{self.name}] Trigger Response: {response.success} - {response.message}")
                except Exception as e:
                    print(f"[{self.name}] Failed to get response from Trigger service: {e}")
        else:
            print(f"[{self.name}] Warning: /vla/trigger_success service is not available.")

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
    def observation_features(self) -> dict[str, tuple]:
        return self.task.observation_features

    @property
    def action_features(self) -> dict[str, tuple]:
        return self.task.action_features
    
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def wrap_action_to_msg(self, arm_action: np.ndarray, gripper_action: np.ndarray = None) -> ActionChunk:

        arm_action[3:7] = _wxyz_to_xyzw(arm_action[3:7])
        gripper_action = gripper_action if gripper_action is not None else np.array([])
        
        msg = ActionChunk()
        msg.header = Header(stamp=self.node.get_clock().now().to_msg(), frame_id="base_link")
        msg.action_space = self.config.action_space
        msg.relative = self.config.is_relative
        msg.rotation_type = self.config.rotation_type
        msg.chunk_size = len(arm_action) // self.config.arm_action_dim

        msg.arm_actions = arm_action.tolist()
        msg.gripper_actions = gripper_action.tolist()

        return msg
