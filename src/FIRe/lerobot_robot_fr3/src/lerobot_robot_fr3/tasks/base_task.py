from typing import Tuple, Dict
import numpy as np
from abc import ABC, abstractmethod
from lerobot_robot_fr3.utils import RobotStateManager, CameraSensorManager, FTSensorManager


class Task(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self._robot: RobotStateManager = None
        self._camera_sensor: CameraSensorManager = None
        self._ft_sensor: FTSensorManager = None
    
    @property
    def robot(self) -> RobotStateManager:
        return self._robot
    
    @property
    def camera_sensor(self) -> CameraSensorManager:
        return self._camera_sensor
    
    @property
    def ft_sensor(self) -> FTSensorManager:
        return self._ft_sensor
    
    def allocate_managers(
        self, 
        robot: RobotStateManager,
        camera_sensor: CameraSensorManager = None,
        ft_sensor: FTSensorManager = None,
    ) -> None:
        self._robot = robot
        self._camera_sensor = camera_sensor
        self._ft_sensor = ft_sensor
    
    @abstractmethod
    def create_config(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def create_buffer(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
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
