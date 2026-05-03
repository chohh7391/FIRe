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
from .utils import RobotStateManager, CameraSensorManager, FTSensorManager
from lerobot_ft_sensor.ft_sensor import FTSensor
from .utils.math_utils import (
    quat_from_angle_axis,
    quat_mul,
    get_euler_xyz,
    quat_from_euler_xyz,
    _wxyz_to_xyzw,
)


class FR3Robot(Robot):
    config_class = FR3RobotConfig
    name = "fr3_vla"

    def __init__(self, config: FR3RobotConfig):
        super().__init__(config)
        self.config = config

        self.node = None
        self.executor_thread = None
        self._is_connected = False

        # Action Client 관련 변수
        self._action_client = None
        self._goal_handle = None
        self._goal_accepted = False

        # update robot state & can get robot state from this
        self.robot_state_manager = None

        # ----- Action / EMA -----
        # factory: 6-dim action (no success-prediction).
        # We keep arm_actions as 6-dim. The 7th slot in self.actions is unused
        # but kept for compatibility with the existing message format.
        self.actions = np.zeros((6,), dtype=np.float32)
        self.prev_actions = None
        # factory: cfg.ctrl.ema_factor is a single value (no randomization).
        # Set to whatever value was used in your factory training cfg.
        self.ema_factor = 1.0  # TODO: replace with the value from cfg.ctrl.ema_factor

        # ----- factory action scaling -----
        # factory uses pos_threshold / rot_threshold to scale [-1, 1] actions
        # directly into delta-pos / axis-angle deltas.
        # Replace with values from cfg.ctrl.pos_action_threshold / rot_action_threshold.
        self.pos_threshold = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        self.rot_threshold = np.array([0.097, 0.097, 0.097], dtype=np.float32)

        # 5 cm clip from fixed_pos_action_frame (cfg.ctrl.pos_action_bounds in factory).
        self.pos_action_bounds = np.array([0.05, 0.05, 0.05], dtype=np.float32)

        # Whether to apply unidirectional rotation on yaw (factory's nut_thread setting).
        # Set to False for peg_insert / gear_mesh.
        self.unidirectional_rot = False

        # ----- fixed asset frame (used only for the 5 cm bounds clip and obs) -----
        height, base_height = 0.025, 0.0  # hole
        # height, base_height = 0.02, 0.005  # gear
        # height, base_height = 0.025, 0.01  # bolt

        fixed_tip_pos_local = np.zeros(3, dtype=np.float32)
        fixed_tip_pos_local[2] += (height + base_height)

        fixed_pos = np.array([0.6, 0.0, 0.05])
        fixed_quat = np.array([1.0, 0.0, 0.0, 0.0])

        fixed_tip_pos = fixed_pos + fixed_tip_pos_local
        self.fixed_pos_obs_frame = fixed_tip_pos

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
        # self.camera_sensor_manager = CameraSensorManager(self.node, self.config.cameras, fps=30.0)

        self.pub_action_chunk = self.node.create_publisher(
            ActionChunk, self.config.action_topic, 10
        )

        self._action_client = ActionClient(
            self.node, VisionLanguageAction, self.config.vla_action_server,
            callback_group=self.action_cb_group
        )

        self.executor_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.executor_thread.start()

        print(f"[{self.name}] Waiting for VLA Action Server: {self.config.vla_action_server}...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            raise ConnectionError("VLA Action Server is not available.")

        self._send_vla_goal()

        print(f"[{self.name}] Waiting for first EE pose observation...")
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if np.any(self.robot_state_manager.ee_pose != 0.0):
                break
            time.sleep(0.1)
        else:
            print("Warning: Did not receive EE pose (TF transform) within timeout.")

        # # connect to cameras
        # self.camera_sensor_manager.connect()

        self.reset()

        self._is_connected = True
        print(f"[{self.name}] VLA Client Connected and Ready!")

    def _send_vla_goal(self):
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
        """Factory observation: NO force_threshold / ft_force, NO prev_actions[3:5]=0 trick."""
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        prev_actions = self.actions.copy()  # factory: no zeroing of [3:5]

        obs_dict = {}
        # obs_dict["fingertip_pos"] = self.robot_state_manager.ee_pos
        obs_dict["fingertip_pos_rel_fixed"] = self.robot_state_manager.ee_pos - self.fixed_pos_obs_frame
        obs_dict["fingertip_quat"] = self.robot_state_manager.ee_quat
        obs_dict["ee_linvel"] = self.robot_state_manager.ee_linvel
        obs_dict["ee_angvel"] = self.robot_state_manager.ee_angvel
        obs_dict["prev_actions"] = prev_actions

        return obs_dict

    def get_vla_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        obs_dict = {}
        # for cam_key, cam_data in self.camera_sensor_manager.data.items():
        #     obs_dict[f"video.{cam_key}_view"] = cam_data

        obs_dict["state.eef_position"] = self.robot_state_manager.ee_pos
        obs_dict["state.eef_quaternion"] = self.robot_state_manager.ee_quat
        obs_dict["state.gripper_qpos"] = np.array([
            self.robot_state_manager.joint_states["position"][7],
            self.robot_state_manager.joint_states["position"][7]
        ])

        return obs_dict

    def send_action(self, action: Dict[str, Any], is_raw_action: bool = True) -> Dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self.name} is not connected.")

        if not self._goal_accepted:
            self.node.get_logger().warn("Cannot send action: Goal not accepted yet.")
            return action

        if is_raw_action:
            # factory: 6-dim arm action only (no success_prediction).
            self.actions[0:6] = (
                self.ema_factor * action["arm_actions"].copy()
                + (1 - self.ema_factor) * self.actions[0:6]
            )

            processed_arm_action = self._process_action()
            # convert quaternion data from wxyz to xyzw
            processed_arm_action[3:7] = _wxyz_to_xyzw(processed_arm_action[3:7])
            arm_action_np = np.array(processed_arm_action).reshape(-1)
        else:
            processed_arm_action = action["arm_actions"]
            processed_arm_action[3:7] = _wxyz_to_xyzw(processed_arm_action[3:7])
            arm_action_np = np.array(processed_arm_action).reshape(-1)

        # for saving
        self.processed_arm_action = processed_arm_action

        gripper_action_np = np.array(action.get("gripper_actions", [])).reshape(-1)
        chunk_size = len(arm_action_np) // self.config.arm_action_dim

        msg = ActionChunk()
        msg.header = Header(stamp=self.node.get_clock().now().to_msg(), frame_id="base_link")
        msg.action_space = self.config.action_space
        msg.relative = self.config.is_relative
        msg.rotation_type = self.config.rotation_type
        msg.chunk_size = chunk_size

        msg.arm_actions = arm_action_np.tolist()
        msg.gripper_actions = gripper_action_np.tolist()

        self.pub_action_chunk.publish(msg)

        self.prev_actions = self.actions.copy()
        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        # self.camera_sensor_manager.disconnect()

        print(f"[{self.name}] Sending Task Success signal to VLA Action Server...")
        trigger_client = self.node.create_client(Trigger, '/vla/trigger_success')

        if trigger_client.wait_for_service(timeout_sec=2.0):
            req = Trigger.Request()
            future = trigger_client.call_async(req)

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

        if self._goal_handle is not None and self._goal_accepted:
            print(f"[{self.name}] Canceling VLA Action goal for safe stop...")
            cancel_future = self._goal_handle.cancel_goal_async()

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
        """Factory obs: no force_threshold, no ft_force, prev_actions is 6-dim."""
        features = {
            # "fingertip_pos": (3,),
            "fingertip_pos_rel_fixed": (3,),
            "fingertip_quat": (4,),
            "ee_linvel": (3,),
            "ee_angvel": (3,),
            "prev_actions": (6,),
        }
        return features

    @property
    def vla_observation_features(self) -> dict[str, tuple]:
        features = {
            # **self._cameras_ft,
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

    # ------------------------------------------------------------------
    # FACTORY action processing
    # ------------------------------------------------------------------
    def _process_action(self):
        """factory_env._apply_action ported to numpy / single env.

        - pos action: EE-relative delta scaled by pos_threshold,
          then 5 cm clipped relative to fixed_pos_action_frame.
        - rot action: axis-angle delta scaled by rot_threshold,
          composed with current EE quat, roll/pitch then forced upright.
        """
        # self.actions[:] = 0.0
        # self.actions[2] = -1.0
        
        # Step (0): Scale actions to allowed range.
        pos_action = self.actions[0:3] * self.pos_threshold

        rot_action = self.actions[3:6].copy()
        if self.unidirectional_rot:
            rot_action[2] = -(rot_action[2] + 1.0) * 0.5  # [-1, 0]
        rot_action = rot_action * self.rot_threshold

        # Step (1): Position target = EE + delta, clipped within bounds of fixed frame.
        ctrl_target_pos = self.robot_state_manager.ee_pos + pos_action

        fixed_pos_action_frame = self.fixed_pos_obs_frame
        delta_pos = ctrl_target_pos - fixed_pos_action_frame
        pos_error_clipped = np.clip(
            delta_pos, -self.pos_action_bounds[0], self.pos_action_bounds[0]
        )
        ctrl_target_pos = fixed_pos_action_frame + pos_error_clipped

        # Step (2): Orientation target = axis-angle delta * current EE quat.
        angle = np.linalg.norm(rot_action)
        if angle > 1e-6:
            axis = rot_action / angle
            rot_actions_quat = quat_from_angle_axis(angle, axis)
        else:
            rot_actions_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        ctrl_target_quat = quat_mul(rot_actions_quat, self.robot_state_manager.ee_quat)

        # Step (3): Force upright (factory restricts roll = π, pitch = 0).
        target_roll, target_pitch, target_yaw = get_euler_xyz(ctrl_target_quat)
        target_roll = 3.14159
        target_pitch = 0.0
        ctrl_target_quat = quat_from_euler_xyz(
            roll=target_roll, pitch=target_pitch, yaw=target_yaw
        )

        return np.concatenate([ctrl_target_pos, ctrl_target_quat])

    def reset(self) -> None:
        """factory: initial actions are all zero (no-movement init)."""
        self.actions = np.zeros_like(self.actions)
        self.prev_actions = np.zeros_like(self.actions)