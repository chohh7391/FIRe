import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped
from typing import Dict
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.callback_groups import ReentrantCallbackGroup

try:
    from franka_msgs.msg import FrankaRobotState
    FRANKA_MSGS_AVAILABLE = True
except ImportError:
    FRANKA_MSGS_AVAILABLE = False


FRANKA_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
EE_POSE_TOPIC      = "/ee_state/pose"
EE_TWIST_TOPIC     = "/ee_state/twist"
TOPIC_DETECT_SEC   = 2.0   # 토픽 감지 대기 시간


class RobotStateManager:
    def __init__(self, node: Node) -> None:
        self.node = node

        self._joint_states = JointState()
        self._ee_pose      = Pose()
        self._ee_vel       = Twist()

        self.cb_group = ReentrantCallbackGroup()

        # joint states는 항상 구독
        self._joint_states_sub = self.node.create_subscription(
            JointState, "/joint_states", self._update_joint_states, 10,
            callback_group=self.cb_group
        )

        # EE state 소스 자동 감지
        self._ee_source = self._detect_ee_source()
        self._setup_ee_subscriptions()

        self.node.get_logger().info(f"[RobotStateManager] EE source: {self._ee_source}")

    # ── 소스 감지 ──────────────────────────────────────────────────────────────

    def _detect_ee_source(self) -> str:
        """
        TOPIC_DETECT_SEC 동안 대기하며 토픽 존재 여부 확인.
        우선순위: ee_state (custom) > franka_robot_state_broadcaster
        """
        deadline = self.node.get_clock().now().nanoseconds + int(TOPIC_DETECT_SEC * 1e9)

        while self.node.get_clock().now().nanoseconds < deadline:
            topic_names = [name for name, _ in self.node.get_topic_names_and_types()]

            if EE_POSE_TOPIC in topic_names and EE_TWIST_TOPIC in topic_names:
                return "ee_state"

            if FRANKA_STATE_TOPIC in topic_names:
                if not FRANKA_MSGS_AVAILABLE:
                    self.node.get_logger().warn(
                        "[RobotStateManager] franka_msgs not installed. "
                        "Install franka_ros2 to use FrankaRobotState."
                    )
                else:
                    return "franka_broadcaster"

            rclpy.spin_once(self.node, timeout_sec=0.1)

        # fallback: ee_state로 구독 시도 (토픽이 늦게 뜰 수 있음)
        self.node.get_logger().warn(
            "[RobotStateManager] No EE topic detected within timeout. "
            f"Defaulting to '{EE_POSE_TOPIC}'."
        )
        return "ee_state"

    def _setup_ee_subscriptions(self):
        if self._ee_source == "ee_state":
            self._ee_pose_sub = self.node.create_subscription(
                PoseStamped, EE_POSE_TOPIC, self._update_ee_pose, 10,
                callback_group=self.cb_group
            )
            self._ee_twist_sub = self.node.create_subscription(
                TwistStamped, EE_TWIST_TOPIC, self._update_ee_vel, 10,
                callback_group=self.cb_group
            )

        elif self._ee_source == "franka_broadcaster":
            self._franka_state_sub = self.node.create_subscription(
                FrankaRobotState, FRANKA_STATE_TOPIC, self._update_from_franka_state, 10,
                callback_group=self.cb_group
            )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _update_joint_states(self, msg: JointState):
        self._joint_states = msg

    def _update_ee_pose(self, msg: PoseStamped):
        self._ee_pose = msg.pose

    def _update_ee_vel(self, msg: TwistStamped):
        self._ee_vel = msg.twist

    def _update_from_franka_state(self, msg: "FrankaRobotState"):
        """
        FrankaRobotState.o_t_ee: float64[16], column-major 4x4 homogeneous transform
          [ R00 R10 R20 0 ]
          [ R01 R11 R21 0 ]  → column-major이므로 reshape 후 transpose
          [ R02 R12 R22 0 ]
          [ tx  ty  tz  1 ]
        """
        T = np.array(msg.o_t_ee, dtype=np.float64).reshape(4, 4, order="F")
        pos = T[:3, 3]
        quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw

        self._ee_pose.position.x    = float(pos[0])
        self._ee_pose.position.y    = float(pos[1])
        self._ee_pose.position.z    = float(pos[2])
        self._ee_pose.orientation.x = float(quat_xyzw[0])
        self._ee_pose.orientation.y = float(quat_xyzw[1])
        self._ee_pose.orientation.z = float(quat_xyzw[2])
        self._ee_pose.orientation.w = float(quat_xyzw[3])

        # o_dp_ee_c: [vx, vy, vz, wx, wy, wz]
        vel = msg.o_dp_ee_c
        self._ee_vel.linear.x  = float(vel[0])
        self._ee_vel.linear.y  = float(vel[1])
        self._ee_vel.linear.z  = float(vel[2])
        self._ee_vel.angular.x = float(vel[3])
        self._ee_vel.angular.y = float(vel[4])
        self._ee_vel.angular.z = float(vel[5])

    # ── Properties (기존과 동일) ───────────────────────────────────────────────

    @property
    def joint_states(self) -> Dict[str, np.ndarray]:
        return {
            "position": np.array(self._joint_states.position, dtype=np.float32),
            "velocity": np.array(self._joint_states.velocity, dtype=np.float32),
            "effort":   np.array(self._joint_states.effort,   dtype=np.float32),
        }

    @property
    def ee_pose(self) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz]"""
        pos  = self._ee_pose.position
        quat = self._ee_pose.orientation
        return np.array([pos.x, pos.y, pos.z,
                         quat.w, quat.x, quat.y, quat.z], dtype=np.float32)

    @property
    def ee_vel(self) -> np.ndarray:
        """[lx, ly, lz, ax, ay, az]"""
        lv = self._ee_vel.linear
        av = self._ee_vel.angular
        return np.array([lv.x, lv.y, lv.z,
                         av.x, av.y, av.z], dtype=np.float32)

    @property
    def ee_pos(self)    -> np.ndarray: return self.ee_pose[0:3]

    @property
    def ee_quat(self)   -> np.ndarray: return self.ee_pose[3:7]

    @property
    def ee_linvel(self) -> np.ndarray: return self.ee_vel[0:3]

    @property
    def ee_angvel(self) -> np.ndarray: return self.ee_vel[3:6]