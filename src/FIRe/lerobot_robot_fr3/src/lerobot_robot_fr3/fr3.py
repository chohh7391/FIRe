import threading
import time
from typing import Any, Dict
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
        "cho_interfaces 패키지를 찾을 수 없습니다. "
        "ROS 2 워크스페이스를 source 했는지 확인하세요."
    )

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from .config_fr3 import FR3RobotConfig
from .utils import RobotStateManager
from lerobot_ft_sensor.configuration_ft_sensor import FTSensorConfig
from lerobot_ft_sensor.ft_sensor import FTSensor
from .utils.math_utils import (
    quat_from_angle_axis,
    quat_mul,
    get_euler_xyz,
    quat_from_euler_xyz,
)



class FR3Robot(Robot):
    config_class = FR3RobotConfig
    name = "fr3_vla"

    def __init__(self, config: FR3RobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.ft_sensor = FTSensor(self.config.ft_sensor)
        
        self.node = None
        self.executor_thread = None
        self._is_connected = False
        
        # Action Client 관련 변수
        self._action_client = None
        self._goal_handle = None
        self._goal_accepted = False
        
        # update robot state & can get robot state from this
        self.robot_state_manager = None
        self.camera_state_manager = None

        self.prev_actions = None

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

        # connect to ft sensor
        self.ft_sensor.connect()

        # connect to cameras
        for cam in self.cameras.values():
            cam.connect()

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

    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")
        
        contact_lower, contact_upper = [5.0, 10.0]
        contact_rand = np.random.rand()
        contact_penalty_thresholds = np.array([
            contact_lower + contact_rand * (contact_upper - contact_lower)
        ])


        height, base_height = 0.025, 0.0  # hole
        # height, base_height = 0.02, 0.005  # gear
        # height, base_height = 0.025, 0.01  # bolt
        
        fixed_tip_pos_local = np.zeros(3, dtype=np.float32)
        fixed_tip_pos_local[2] += (height + base_height)
        # if task == "gear_mesh":
        #     fixed_tip_pos_local[0] = medium_gear_base_offset[0]

        fixed_pos = np.array([0.6, 0.0, 0.05])
        fixed_quat = np.array([1.0, 0.0, 0.0, 0.0])

        fixed_tip_pos = fixed_pos + fixed_tip_pos_local
        self.fixed_pos_obs_frame = fixed_tip_pos

        # TODO: change ee_pos, ee_quat to fingertip frame and so on
        obs_dict = {}
        # obs_dict["fingertip_pos"] = self.robot_state_manager.ee_pos
        obs_dict["fingertip_pos_rel_fixed"] = self.robot_state_manager.ee_pos - self.fixed_pos_obs_frame
        obs_dict["fingertip_quat"] = self.robot_state_manager.ee_quat
        obs_dict["ee_linvel"] = self.robot_state_manager.ee_linvel
        obs_dict["ee_angvel"] = self.robot_state_manager.ee_angvel
        obs_dict["force_threshold"] = contact_penalty_thresholds
        obs_dict["ft_force"] = self.ft_sensor.async_read()["force"]
        if self.prev_actions is not None:
            obs_dict["prev_actions"] = self.prev_actions
        else:
            obs_dict["prev_actions"] = np.zeros((7,), dtype=np.float32)

        return obs_dict
    
    def get_vla_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")
        
        obs_dict = {}
        for cam_key, cam in self.cameras.items():
            obs_dict[f"video.{cam_key}_view"] = cam.async_read(timeout_ms=1000)

        obs_dict["state.eef_position"] = self.robot_state_manager.ee_pos
        obs_dict["state.eef_quaternion"] = self.robot_state_manager.ee_quat
        obs_dict["state.gripper_qpos"] = np.array([
            self.robot_state_manager.joint_states["position"][7], self.robot_state_manager.joint_states["position"][7]
        ])
        
        return obs_dict

    def send_action(self, action: Dict[str, Any], is_raw_action: bool = True) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        if not self._goal_accepted:
            self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
            return action
        
        if is_raw_action:
            processed_arm_action = self._process_action(action["arm_actions"])
            arm_action_np = np.array(processed_arm_action).reshape(-1)
        else:
            arm_action_np = np.array(action["arm_actions"]).reshape(-1)
        
        gripper_action_np = np.array(action.get("gripper_actions", [])).reshape(-1)
        chunk_size = len(arm_action_np) // self.config.arm_action_dim

        msg = ActionChunk()
        # 헤더 시간 및 프레임 명시 (Tester 참고)
        msg.header = Header(stamp=self.node.get_clock().now().to_msg(), frame_id="base_link")
        msg.action_space = self.config.action_space
        msg.relative = self.config.is_relative
        msg.rotation_type = self.config.rotation_type
        msg.chunk_size = chunk_size
        
        msg.arm_actions = arm_action_np.tolist()
        msg.gripper_actions = gripper_action_np.tolist()

        self.pub_action_chunk.publish(msg)

        self.prev_actions = np.concatenate([action["arm_actions"], action["success_prediction"]])
        return action
    
    # def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
    #     if not self.is_connected:
    #         raise ConnectionError(f"{self.name} is not connected.")

    #     if not self._goal_accepted:
    #         self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
    #         return action

    #     arm_action_np = np.array(action["arm_actions"]).reshape(-1)
    #     gripper_action_np = np.array(action.get("gripper_actions", [])).reshape(-1)
    #     chunk_size = len(arm_action_np) // self.config.arm_action_dim

    #     msg = ActionChunk()
    #     msg.header = Header(stamp=self.node.get_clock().now().to_msg(), frame_id="base_link")
    #     msg.action_space = self.config.action_space
    #     msg.relative = False
    #     msg.rotation_type = self.config.rotation_type
    #     msg.chunk_size = chunk_size
        
    #     msg.arm_actions = arm_action_np.tolist()
    #     msg.gripper_actions = gripper_action_np.tolist()

    #     self.pub_action_chunk.publish(msg)

    #     # self.prev_actions = action["arm_actions"] + success_prediction
        
    #     return action

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        for cam in self.cameras.values():
            cam.disconnect()

        self.ft_sensor.disconnect()

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
    def _cameras_ft(self) -> Dict[str, tuple]:
        return {
            f"video.{cam}_view": (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, tuple]:
        
        features = {
            # "fingertip_pos": (3,),
            "fingertip_pos_rel_fixed": (3,),
            "fingertip_quat": (4,),
            "ee_linvel": (3,),
            "ee_angvel": (3,),
            "force_threshold": (1,),
            "ft_force": (3,),
            # "prev_actions": (6,),
            "prev_actions": (7,)
        }

        return features
    
    @property
    def vla_observation_features(self) -> dict[str, tuple]:
        features = {
            **self._cameras_ft,
            "state.eef_position": (3,),
            "state.eef_quaternion": (4,),
            "state.gripper_qpos": (2,),
        }
        return features

    @property
    def action_features(self) -> dict[str, tuple]:
        return {
            "arm_actions": (self.config.arm_action_dim,),
            "gripper_actions": (1,)
        }
    
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _process_action(self, raw_action):
        pos_action = raw_action[0:3] * np.array([0.02, 0.02, 0.02])
        rot_action = raw_action[3:6]

        # if task == "nut_thread":
        #     rot_action[2] = -(rot_action[2] + 1.0) * 0.5  # [-1, 0]
        rot_action = rot_action * np.array([0.097, 0.097, 0.097])

        ctrl_target_pos = self.robot_state_manager.ee_pos + pos_action
        delta_pos = ctrl_target_pos - self.fixed_pos_obs_frame
        pos_error_clipped = np.clip(
            delta_pos, -0.05, 0.05
        )
        ctrl_target_pos = self.fixed_pos_obs_frame + pos_error_clipped

        angle = np.linalg.norm(rot_action)
        axis = rot_action / angle if angle > 1e-6 else np.array([0.0, 0.0, 1.0])

        rot_action_quat = quat_from_angle_axis(angle, axis)
        ctrl_target_quat = quat_mul(rot_action_quat, self.robot_state_manager.ee_quat)

        target_euler_xyz = get_euler_xyz(ctrl_target_quat)
        target_euler_xyz[0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[1] = 0.0

        ctrl_target_quat = quat_from_euler_xyz(
            roll=target_euler_xyz[0], pitch=target_euler_xyz[1], yaw=target_euler_xyz[2]
        )

        return np.concatenate([ctrl_target_pos, ctrl_target_quat])

