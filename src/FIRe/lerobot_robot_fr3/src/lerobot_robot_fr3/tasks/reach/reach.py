from typing import Dict, Any, Tuple
from lerobot_robot_fr3.tasks.base_task import Task
from .reach_cfg import ReachEnvCfg
import numpy as np

class Reach(Task):
    def __init__(self, name: str):
        super().__init__(name)

        self.create_config()
        self.create_buffer()

    def create_config(self):
        self.env_cfg = ReachEnvCfg()

    def create_buffer(self):
        self.action = np.zeros(self.env_cfg.action_space, dtype=np.float32)

        self.default_joint_pos = np.array([
            0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04
        ])

    def reset(self):
        pass

    def get_observation(self) -> Dict[str, np.ndarray]:
        pose_command = np.array(
            [0.5, 0.0, 0.325, 0.0, 1.0, 0.0, 0.0], dtype=np.float32
        )
        arm_joint_pos = self.robot.joint_states["position"]
        gripper_joint_pos = self.robot.gripper_joint_states["position"]

        arm_joint_vel = self.robot.joint_states["velocity"]
        gripper_joint_vel = self.robot.gripper_joint_states["velocity"]

        joint_pos = np.concatenate([
            arm_joint_pos,
            gripper_joint_pos,
        ])
        joint_vel = np.concatenate([
            arm_joint_vel,
            gripper_joint_vel,
        ])

        obs_dict = {
            "joint_pos": joint_pos - self.default_joint_pos,
            "joint_vel": joint_vel,
            "pose_command": pose_command,
            "actions": self.action.copy(),
        }
        print(f"obs_dict: {obs_dict}")
        return obs_dict
    
    def get_arm_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return action["arm_actions"]
    
    def get_gripper_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return action["gripper_actions"]
    
    @property
    def observation_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "joint_pos": (9,),
            "joint_vel": (9,),
            "pose_command": (7,),
            "actions": (self.env_cfg.action_space,),
        }
    
    @property
    def action_features(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "arm_actions": (self.env_cfg.action_space,),
        }
    
    def process_action(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.action = arm_action.copy()
        print(f"action: {self.action}")
        arm_action[0:3] = 0.0
        arm_action[0] = 0.5
        arm_action[2] = 0.4
        arm_action[3:7] = [0.0, 1.0, 0.0, 0.0]
        return arm_action, gripper_action

