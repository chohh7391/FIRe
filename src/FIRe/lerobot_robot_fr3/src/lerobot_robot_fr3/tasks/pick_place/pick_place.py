import numpy as np
from typing import Dict, Optional, Tuple

from lerobot_robot_fr3.tasks.base_task import Task
from .pick_place_cfg import PickPlaceTaskCfg


class PickPlace(Task):
    """Real-robot deployment of the vla_lab pick_place policy.

    Unlike Factory/Forge (task-space fingertip control), this task uses
    joint-position control: the policy emits 7 joint deltas + 1 binary gripper,
    and the joint targets are `default_joint_pos + scale * action`.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.create_config()
        self.create_buffer()

    # ── config / buffers ──────────────────────────────────────────────────────
    def create_config(self) -> None:
        if self.name == "pick_place":
            self.env_cfg = PickPlaceTaskCfg()
        else:
            raise ValueError(f"Unknown task name: {self.name}")

        self.ctrl_cfg = self.env_cfg.ctrl
        self.task_cfg = self.env_cfg.task

        self.action_scale = self.ctrl_cfg.action_scale

        self.default_arm = np.array(self.task_cfg.default_arm_joint_pos, dtype=np.float32)
        self.default_finger = np.array(self.task_cfg.default_finger_joint_pos, dtype=np.float32)
        self.default_joint_pos = np.concatenate([self.default_arm, self.default_finger])

        # Fixed observation constants (no perception; robot root ≈ world).
        self.object_position = np.array(self.task_cfg.object_position, dtype=np.float32)
        self.green_cube_position = np.array(self.task_cfg.green_cube_position, dtype=np.float32)
        self.target_object_position = np.concatenate([
            np.array(self.task_cfg.target_object_position, dtype=np.float32),
            np.array(self.task_cfg.target_object_quat, dtype=np.float32),
        ]).astype(np.float32)

    def create_buffer(self) -> None:
        # Raw 8-dim policy action (7 joint + 1 gripper); also serves as `last_action` obs.
        self.action = np.zeros(self.env_cfg.action_space, dtype=np.float32)

    def reset(self) -> None:
        self.action[:] = 0.0

    # ── observation ───────────────────────────────────────────────────────────
    def _joint_pos_rel(self) -> np.ndarray:
        arm = np.asarray(self.robot.joint_states["position"], dtype=np.float32)[:7]
        finger = np.asarray(self.robot.gripper_qpos, dtype=np.float32)[:2]
        current = np.concatenate([arm, finger])
        return (current - self.default_joint_pos).astype(np.float32)

    def _joint_vel_rel(self) -> np.ndarray:
        arm = np.asarray(self.robot.joint_states["velocity"], dtype=np.float32)[:7]
        finger = np.asarray(self.robot.gripper_qvel, dtype=np.float32)[:2]
        # default joint velocity is zero, so joint_vel_rel == joint_vel.
        return np.concatenate([arm, finger]).astype(np.float32)

    def get_observation(self) -> Dict[str, np.ndarray]:
        return {
            "joint_pos": self._joint_pos_rel(),
            "joint_vel": self._joint_vel_rel(),
            "object_position": self.object_position,
            "target_object_position": self.target_object_position,
            "actions": self.action.copy(),  # last applied raw policy action
            "green_cube_position": self.green_cube_position,
        }

    def get_vla_observation(self) -> Dict[str, np.ndarray]:
        if self.camera_sensor is not None:
            camera_data = self.camera_sensor.data
        else:
            camera_data = {
                "left": np.zeros((256, 256, 3), dtype=np.uint8),
                "right": np.zeros((256, 256, 3), dtype=np.uint8),
                "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            }
        return {
            "video.left_view": np.expand_dims(camera_data["left"].astype(np.uint8), axis=(0, 1)),
            "video.right_view": np.expand_dims(camera_data["right"].astype(np.uint8), axis=(0, 1)),
            "video.wrist_view": np.expand_dims(camera_data["wrist"].astype(np.uint8), axis=(0, 1)),
            "state.eef_position": np.expand_dims(self.robot.ee_pos.astype(np.float64), axis=(0, 1)),
            "state.eef_quaternion": np.expand_dims(self.robot.ee_quat.astype(np.float64), axis=(0, 1)),
            "state.gripper_qpos": np.expand_dims(self.robot.gripper_qpos.astype(np.float64), axis=(0, 1)),
        }

    # ── action ────────────────────────────────────────────────────────────────
    def get_arm_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(action["arm_actions"], dtype=np.float32).reshape(-1)

    def get_gripper_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        if "gripper_actions" in action:
            return np.asarray(action["gripper_actions"], dtype=np.float32).reshape(-1)
        return np.array([-1.0], dtype=np.float32)

    def process_action(
        self,
        arm_action: np.ndarray,
        gripper_action: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        arm_action = np.asarray(arm_action, dtype=np.float32).reshape(-1)[:7]
        gripper_raw = (
            float(np.asarray(gripper_action, dtype=np.float32).reshape(-1)[0])
            if gripper_action is not None and np.size(gripper_action) > 0
            else -1.0
        )

        # Store the raw policy action for the next step's `last_action` observation.
        self.action = np.concatenate([
            arm_action,
            np.array([gripper_raw], dtype=np.float32),
        ]).astype(np.float32)

        # JointPositionAction: absolute joint targets = default + scale * action.
        joint_target = self.default_arm + self.action_scale * arm_action

        # BinaryJointPositionAction: action > 0 -> open, else close.
        gripper_cmd = (
            self.task_cfg.gripper_open if gripper_raw > 0.0 else self.task_cfg.gripper_close
        )
        processed_gripper = np.array([gripper_cmd], dtype=np.float32)

        return joint_target.astype(np.float32), processed_gripper

    # ── logging ───────────────────────────────────────────────────────────────
    def get_log(self) -> Dict[str, np.ndarray]:
        return {
            "ee_pos": self.robot.ee_pos,
            "ee_quat": self.robot.ee_quat,
            "joint_pos": np.asarray(self.robot.joint_states["position"], dtype=np.float32)[:7],
        }

    # ── feature specs ─────────────────────────────────────────────────────────
    @property
    def observation_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "joint_pos": (9,),
            "joint_vel": (9,),
            "object_position": (3,),
            "target_object_position": (7,),
            "actions": (8,),
            "green_cube_position": (3,),
        }

    @property
    def action_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "arm_actions": (7,),
            "gripper_actions": (1,),
        }

    @property
    def log_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "ee_pos": (3,),
            "ee_quat": (4,),
            "joint_pos": (7,),
        }

    # ── control metadata (joint-space) ────────────────────────────────────────
    @property
    def control_action_space(self) -> str:
        return "joint"

    @property
    def control_arm_action_dim(self) -> int:
        return 7
