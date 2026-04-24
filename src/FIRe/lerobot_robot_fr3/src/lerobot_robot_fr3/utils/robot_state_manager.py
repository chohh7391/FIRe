import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped
from typing import Dict
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup


class RobotStateManager:
    def __init__(self, node: Node) -> None:
        self.node = node

        self._joint_states = JointState()
        self._ee_pose = Pose()
        self._ee_vel = Twist()

        self.cb_group = ReentrantCallbackGroup()

        self._joint_states_sub = self.node.create_subscription(
            JointState, "/joint_states", self._update_joint_states, 10,
            callback_group=self.cb_group
        )
        self._ee_pose_sub = self.node.create_subscription(
            PoseStamped, "/ee_state/pose", self._update_ee_pose, 10,
            callback_group=self.cb_group
        )
        self._ee_twist_sub = self.node.create_subscription(
            TwistStamped, "/ee_state/twist", self._update_ee_vel, 10,
            callback_group=self.cb_group
        )

    @property
    def joint_states(self) -> Dict[str, np.ndarray]:
        joint_positions = np.array(self._joint_states.position, dtype=np.float32)
        joint_velocities = np.array(self._joint_states.velocity, dtype=np.float32)
        joint_efforts = np.array(self._joint_states.effort, dtype=np.float32)
        return {
            "position": joint_positions,
            "velocity": joint_velocities,
            "effort":   joint_efforts
        }

    @property
    def ee_pose(self) -> np.ndarray:
        """return np.array([x, y, z, qw, qx, qy, qz])"""
        pos = self._ee_pose.position
        quat = self._ee_pose.orientation
        return np.array([
            pos.x, pos.y, pos.z,
            quat.w, quat.x, quat.y, quat.z
        ], dtype=np.float32)

    @property
    def ee_vel(self) -> np.ndarray:
        """return np.array([lx, ly, lz, ax, ay, az])"""
        lv = self._ee_vel.linear
        av = self._ee_vel.angular
        return np.array([
            lv.x, lv.y, lv.z,
            av.x, av.y, av.z
        ], dtype=np.float32)

    @property
    def ee_pos(self) -> np.ndarray:
        """return np.array([x, y, z])"""
        return self.ee_pose[0:3]

    @property
    def ee_quat(self) -> np.ndarray:
        """return np.array([qw, qx, qy, qz])"""
        return self.ee_pose[3:7]

    @property
    def ee_linvel(self) -> np.ndarray:
        """return np.array([lx, ly, lz])"""
        return self.ee_vel[0:3]

    @property
    def ee_angvel(self) -> np.ndarray:
        """return np.array([ax, ay, az])"""
        return self.ee_vel[3:6]

    def _update_joint_states(self, msg: JointState):
        self._joint_states = msg

    def _update_ee_pose(self, msg: PoseStamped):
        self._ee_pose = msg.pose

    def _update_ee_vel(self, msg: TwistStamped):
        self._ee_vel = msg.twist