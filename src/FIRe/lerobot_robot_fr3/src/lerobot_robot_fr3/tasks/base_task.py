from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
from lerobot_robot_fr3.utils import RobotStateManager


class Task(ABC):
    def __init__(self, name: str):
        self.name = name
        self.robot: RobotStateManager = None
    
    @abstractmethod
    def create_config(self):
        raise NotImplementedError
    
    @abstractmethod
    def create_buffer(self):
        raise NotImplementedError
    
    def allocate_robot(self, robot: RobotStateManager):
        self.robot = robot
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> dict[str, np.ndarray]:
        raise NotImplementedError
    
    @abstractmethod
    def get_arm_action(self, action: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_gripper_action(self, action: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_features(self) -> dict[str, tuple]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_features(self) -> dict[str, tuple]:
        raise NotImplementedError
    
    @abstractmethod
    def process_action(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
