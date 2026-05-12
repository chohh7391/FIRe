from typing import Tuple, Dict
import numpy as np
from abc import ABC, abstractmethod
from lerobot_robot_fr3.utils import RobotStateManager


class Task(ABC):
    def __init__(self, name: str):
        self.name = name
        self._robot: RobotStateManager = None

    @property
    def robot(self):
        return self._robot
    
    def allocate_robot(self, robot: RobotStateManager):
        self._robot = robot
    
    @abstractmethod
    def create_config(self):
        raise NotImplementedError
    
    @abstractmethod
    def create_buffer(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError
    
    @abstractmethod
    def get_arm_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_gripper_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_log(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError
    
    @abstractmethod
    def process_action(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_features(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_features(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_features(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError
