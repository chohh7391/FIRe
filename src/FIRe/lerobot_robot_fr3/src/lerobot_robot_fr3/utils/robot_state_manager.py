import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from typing import Dict
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class RobotStateManager:
    def __init__(self, node: Node) -> None:
        self.node = node
        
        self._joint_states = JointState()
        self._ee_pose = Pose()
        self._ee_vel = Twist()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        self.cb_group = ReentrantCallbackGroup()

        self._joint_states_sub = self.node.create_subscription(
            JointState, "/joint_states", self._update_joint_states, 10,
            callback_group=self.cb_group
        )

        self._timer = self.node.create_timer(
            0.01, self._update_ee_pose,
            callback_group=self.cb_group
        )
        

    @property
    def joint_states(self) -> Dict[str, np.ndarray]:
        joint_positions = np.array([self._joint_states.position], dtype=np.float32).squeeze()
        joint_velocities = np.array([self._joint_states.velocity], dtype=np.float32).squeeze()
        joint_efforts = np.array([self._joint_states.effort], dtype=np.float32).squeeze()

        return {
            "position": joint_positions,
            "velocity": joint_velocities,
            "effort": joint_efforts
        }
    
    @property
    def ee_pose(self) -> np.ndarray:
        """
        return np.array([x, y, z, qw, qx, qy, qz]])
        """
        pos = self._ee_pose.position
        quat = self._ee_pose.orientation
        return np.array([
            pos.x, pos.y, pos.z,
            quat.w, quat.x, quat.y, quat.z
        ], dtype=np.float32)
    
    @property
    def ee_vel(self) -> np.ndarray:
        """
        return np.array([lx, ly, lz, ax, ay, az])
        """
        lv = self._ee_vel.linear
        av = self._ee_vel.angular
        return np.array([
            lv.x, lv.y, lv.z,
            av.x, av.y, av.z
        ], dtype=np.float32)
    
    @property
    def ee_pos(self) -> np.ndarray:
        return self.ee_pose[0:3]
    
    @property
    def ee_quat(self) -> np.ndarray:
        return self.ee_pose[3:7]
    
    @property
    def ee_linvel(self) -> np.ndarray:
        return self.ee_vel[0:3]
    
    @property
    def ee_angvel(self) -> np.ndarray:
        return self.ee_vel[3:6]
    
    def _update_joint_states(self, msg: JointState):
        self._joint_states = msg

    def _update_ee_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self._tf_buffer.lookup_transform(
                "fr3_link0", "fr3_hand", now
            )

            self._ee_pose.position.x = trans.transform.translation.x
            self._ee_pose.position.y = trans.transform.translation.y
            self._ee_pose.position.z = trans.transform.translation.z
            self._ee_pose.orientation.w = trans.transform.rotation.w
            self._ee_pose.orientation.x = trans.transform.rotation.x
            self._ee_pose.orientation.y = trans.transform.rotation.y
            self._ee_pose.orientation.z = trans.transform.rotation.z

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.node.get_logger().warn(f'Could not transform fr3_link0 to fr3_hand: {str(e)}')
